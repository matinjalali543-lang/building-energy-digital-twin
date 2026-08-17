"""Shared fixtures.

Everything is built from short synthetic driver windows rather than the
committed dataset, so the suite runs in seconds and CI never needs the network
or the 130 MB OPSD download. The handful of tests that assert something about
the real committed data are marked and skip cleanly if it is absent.

Module-scoped fixtures are used where an object is only ever read. Fitting a
thermal model per test dominated the runtime of this suite before that change.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Pin the backend for the test process only. The library deliberately does not,
# because doing so makes every notebook figure render as text -- see
# bet.visualization.
matplotlib.use("Agg")

from bet.data import DEFAULT_DRIVERS  # noqa: E402
from bet.plant.building import (  # noqa: E402
    BatterySpec,
    BuildingSpec,
    ComfortBand,
    HvacSpec,
    OccupancySchedule,
)


def make_drivers(hours: int = 24 * 21, start: str = "2019-11-01", seed: int = 7):
    """A synthetic driver window: weather, PV and prices.

    Deliberately includes negative prices, because they are what makes the HVAC
    mode binary do anything at all, and a fixture without them would let the
    relevant test pass vacuously.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    day = np.arange(hours) / 24.0

    outdoor = 6.0 + 5.0 * np.sin(2 * np.pi * (hour - 15) / 24) + 3.0 * np.sin(
        2 * np.pi * day / 9
    )
    daylight = np.clip(np.cos((hour - 12) * np.pi / 9), 0, None) ** 2
    irradiance = 620.0 * daylight * (0.45 + 0.55 * rng.random(hours))
    pv = 250.0 * irradiance / 1000.0 * 0.86

    price = (
        42.0
        + 26.0 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 9.0, hours)
    )
    # Force a handful of deeply negative hours into the midday trough.
    midday = (hour >= 11) & (hour <= 14) & (np.arange(hours) // 24 % 5 == 0)
    price[midday] = -80.0

    return pd.DataFrame(
        {
            "outdoor_temp_c": outdoor,
            "irradiance_w_m2": irradiance,
            "sun_elevation_deg": np.clip(daylight * 40, 0, None),
            "wind_speed_m_s": np.full(hours, 3.5),
            "pv_kw": pv,
            "price_eur_mwh": price,
        },
        index=pd.DatetimeIndex(index, name="utc_timestamp"),
    )


@pytest.fixture(scope="module")
def drivers():
    return make_drivers()


@pytest.fixture(scope="module")
def short_drivers():
    return make_drivers(hours=24 * 4, start="2019-11-05", seed=11)


@pytest.fixture
def spec() -> BuildingSpec:
    """The reference building, with load variability off.

    Deterministic on purpose: physics tests assert exact recursions, and a
    stochastic gain term would make them flaky for reasons unrelated to what
    they check. Tests that need the variability enable it explicitly.
    """
    return BuildingSpec(load_variability=0.0)


@pytest.fixture
def noisy_spec() -> BuildingSpec:
    return BuildingSpec()


@pytest.fixture
def small_spec() -> BuildingSpec:
    """A lighter, faster building for tests that only need dynamics."""
    return BuildingSpec(
        name="Test Office",
        floor_area_m2=2000.0,
        ua_w_per_k=1500.0,
        capacitance_j_per_m2_k=100e3,
        solar_aperture_m2=50.0,
        load_variability=0.0,
        schedule=OccupancySchedule(),
        comfort=ComfortBand(),
        hvac=HvacSpec(
            heating_capacity_kw_thermal=120.0, cooling_capacity_kw_thermal=110.0
        ),
        battery=BatterySpec(energy_capacity_kwh=80.0, power_kw=40.0),
        pv_capacity_kwp=60.0,
        grid_import_limit_kw=250.0,
        grid_export_limit_kw=80.0,
    )


@pytest.fixture(scope="module")
def identified_model(drivers):
    """A thermal model fitted once from telemetry, shared across tests."""
    from bet.ingestion.providers import record_telemetry
    from bet.plant.controllers import ThermostatController
    from bet.plant.simulator import Plant
    from bet.twin.residuals import identify_thermal_model

    spec = BuildingSpec(load_variability=0.0)
    history = record_telemetry(Plant(spec, drivers), ThermostatController(spec))
    usable = history.dropna(
        subset=["indoor_temp_c", "outdoor_temp_c", "irradiance_w_m2"]
    )
    return identify_thermal_model(usable)


@pytest.fixture
def true_model(spec):
    """The plant's actual parameters, expressed as a ThermalModel."""
    from bet.twin.residuals import ThermalModel

    return ThermalModel(
        ua_kw_per_k=spec.ua_w_per_k / 1e3,
        capacitance_kwh_per_k=spec.capacitance_j_per_k / 3.6e6,
        solar_coefficient_kw_per_w_m2=spec.solar_aperture_m2 * 0.6 / 1e3,
        occupancy_gain_kw=spec.floor_area_m2 * spec.people_gain_w_m2 / 1e3,
        baseline_gain_kw=(
            spec.floor_area_m2
            * (spec.lighting_gain_w_m2 + spec.plug_gain_w_m2)
            * spec.base_load_fraction
            / 1e3
        ),
    )


@pytest.fixture
def real_drivers():
    """The committed dataset, or skip."""
    if not DEFAULT_DRIVERS.exists():
        pytest.skip(f"{DEFAULT_DRIVERS} not present")
    from bet.data import load_drivers

    return load_drivers()
