"""Plant physics, twin state, and the wall between them.

The rule these follow, learned across five sibling repos: assert on physics,
conservation and totals — never on an individual control decision. Where a
specific action matters, the fixture makes it uniquely optimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet.ingestion.providers import CsvReplayProvider, PlantStreamProvider, record_telemetry
from bet.plant.building import (
    BatterySpec,
    BuildingSpec,
    ComfortBand,
    HvacSpec,
    OccupancySchedule,
)
from bet.plant.controllers import (
    NullController,
    ThermostatController,
    ThermostatWithSelfConsumption,
    run_controller,
)
from bet.plant.simulator import (
    SECONDS_PER_HOUR,
    ControlAction,
    Plant,
    internal_gain_kw,
    occupancy_fraction,
)
from bet.twin.state import Measurement, Quality, TwinStateStore

TOL = 1e-9


# --- the plant's physics -------------------------------------------------


def test_electrical_energy_balance_closes(spec, drivers):
    plant = Plant(spec, drivers)
    run_controller(ThermostatController(spec), plant)
    assert np.abs(plant.energy_balance_residual()).max() < 1e-9


def test_thermal_recursion_reproduces_exactly(spec, drivers):
    plant = Plant(spec, drivers)
    result = run_controller(ThermostatController(spec), plant)
    frame = result.frame

    previous = np.concatenate([[21.0], frame["indoor_temp_c"].to_numpy()[:-1]])
    solar = frame["irradiance_w_m2"].to_numpy() * spec.solar_aperture_m2 * 0.6 / 1e3
    gains = internal_gain_kw(occupancy_fraction(frame.index, spec), spec)
    expected = previous + (
        spec.ua_w_per_k * (frame["outdoor_temp_c"].to_numpy() - previous)
        + (gains + solar + frame["hvac_thermal_kw"].to_numpy()) * 1e3
    ) * SECONDS_PER_HOUR / spec.capacitance_j_per_k

    assert np.abs(expected - frame["indoor_temp_c"].to_numpy()).max() < 1e-9


def test_free_running_temperature_decays_at_the_stated_time_constant(small_spec):
    """A step in outdoor temperature must decay at C/UA.

    Run with no gains, no sun and no HVAC, so the only dynamics are the
    exponential the time constant describes.
    """
    hours = 400
    index = pd.date_range("2019-06-01", periods=hours, freq="h", tz="UTC")
    drivers = pd.DataFrame(
        {
            "outdoor_temp_c": np.zeros(hours),
            "irradiance_w_m2": np.zeros(hours),
            "pv_kw": np.zeros(hours),
            "price_eur_mwh": np.full(hours, 50.0),
        },
        index=index,
    )
    dark = BuildingSpec(
        floor_area_m2=small_spec.floor_area_m2,
        ua_w_per_k=small_spec.ua_w_per_k,
        capacitance_j_per_m2_k=small_spec.capacitance_j_per_m2_k,
        solar_aperture_m2=0.0,
        people_gain_w_m2=0.0,
        lighting_gain_w_m2=0.0,
        plug_gain_w_m2=0.0,
        load_variability=0.0,
        battery=None,
    )
    plant = Plant(dark, drivers, initial_indoor_temp_c=20.0)
    temps = run_controller(NullController(), plant).frame["indoor_temp_c"].to_numpy()

    tau = dark.time_constant_hours
    # After one time constant the gap to ambient should have fallen to 1/e.
    remaining = temps[int(round(tau)) - 1] / 20.0
    assert remaining == pytest.approx(np.exp(-1), abs=0.03)


def test_battery_respects_its_energy_limits(spec, short_drivers):
    plant = Plant(spec, short_drivers)
    floor = spec.battery.soc_min_frac * spec.battery.energy_capacity_kwh
    ceiling = spec.battery.soc_max_frac * spec.battery.energy_capacity_kwh
    while not plant.finished:
        # Ask for far more than the battery can take, every single step.
        plant.step(ControlAction(hvac_thermal_kw=0.0, battery_kw=10_000.0))
    soc = plant.history_frame()["battery_soc_kwh"].to_numpy()
    assert soc.min() >= floor - 1e-9
    assert soc.max() <= ceiling + 1e-9


def test_actions_are_clipped_to_equipment_ratings(spec, short_drivers):
    plant = Plant(spec, short_drivers)
    while not plant.finished:
        plant.step(ControlAction(hvac_thermal_kw=1e6, battery_kw=1e6))
    frame = plant.history_frame()
    assert frame["hvac_thermal_kw"].max() <= spec.hvac.heating_capacity_kw_thermal + 1e-9
    assert frame["battery_kw"].max() <= spec.battery.power_kw + 1e-9


def test_thermostat_holds_the_band_far_better_than_no_control(spec, drivers):
    controlled = run_controller(ThermostatController(spec), Plant(spec, drivers))
    free = run_controller(NullController(), Plant(spec, drivers))
    assert controlled.comfort_violations() < free.comfort_violations()


def test_greedy_battery_never_costs_more_than_the_plain_thermostat(spec, drivers):
    plain = run_controller(ThermostatController(spec), Plant(spec, drivers))
    greedy = run_controller(ThermostatWithSelfConsumption(spec), Plant(spec, drivers))
    assert greedy.total_cost_eur <= plain.total_cost_eur + 1e-6


def test_plant_refuses_to_run_past_its_drivers(spec, short_drivers):
    plant = Plant(spec, short_drivers)
    while not plant.finished:
        plant.step(ControlAction())
    with pytest.raises(RuntimeError, match="already run all"):
        plant.step(ControlAction())


def test_plant_rejects_missing_columns(spec, short_drivers):
    with pytest.raises(ValueError, match="missing column"):
        Plant(spec, short_drivers.drop(columns=["pv_kw"]))


# --- building spec validation --------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"floor_area_m2": 0}, "floor_area_m2 must be > 0"),
        ({"ua_w_per_k": -1}, "ua_w_per_k must be > 0"),
        ({"base_load_fraction": 1.5}, "base_load_fraction must be in"),
        ({"load_variability": 1.5}, "load_variability must be in"),
    ],
)
def test_building_spec_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BuildingSpec(**kwargs)


def test_building_with_no_thermal_inertia_is_refused():
    """A sub-hour time constant makes an hourly model meaningless."""
    with pytest.raises(ValueError, match="no usable thermal inertia"):
        BuildingSpec(ua_w_per_k=5e6, capacitance_j_per_m2_k=1e3)


def test_comfort_band_setback_must_be_looser():
    with pytest.raises(ValueError, match="meant to be looser"):
        ComfortBand(occupied_min_c=21.0, setback_min_c=22.0)


def test_implausible_cop_is_refused():
    with pytest.raises(ValueError, match="not a heat pump"):
        HvacSpec(cop_heating=9.0)


def test_battery_initial_soc_must_lie_in_band():
    with pytest.raises(ValueError, match="must lie within"):
        BatterySpec(initial_soc_frac=0.99)


def test_schedule_end_must_follow_start():
    with pytest.raises(ValueError, match="must be after"):
        OccupancySchedule(start_hour=19, end_hour=7)


def test_derived_storage_quantities(spec):
    assert spec.thermal_storage_kwh() == pytest.approx(
        spec.capacitance_j_per_k * spec.comfort.occupied_width_k / 3.6e6
    )
    assert spec.thermal_storage_kwh_electric() == pytest.approx(
        spec.thermal_storage_kwh() / spec.hvac.cop_heating
    )


# --- the twin's state ----------------------------------------------------


def test_out_of_order_measurements_are_rejected():
    store = TwinStateStore()
    now = pd.Timestamp("2019-11-01 10:00", tz="UTC")
    assert store.apply(Measurement("sensor_zone_temp", now, 22.0, "degC"))
    stale = Measurement("sensor_zone_temp", now - pd.Timedelta(hours=1), 99.0, "degC")
    assert not store.apply(stale)
    assert store.state.value("indoor_temp_c") == 22.0
    assert store.rejected_out_of_order == 1


def test_quality_ages_from_good_through_stale_to_missing():
    store = TwinStateStore(default_staleness_minutes=120)
    now = pd.Timestamp("2019-11-01 10:00", tz="UTC")
    store.apply(Measurement("sensor_zone_temp", now, 22.0, "degC"))

    store.advance_to(now)
    assert store.state.quality("indoor_temp_c") is Quality.GOOD
    store.advance_to(now + pd.Timedelta(hours=1))
    assert store.state.quality("indoor_temp_c") is Quality.STALE
    store.advance_to(now + pd.Timedelta(hours=5))
    assert store.state.quality("indoor_temp_c") is Quality.MISSING
    assert "indoor_temp_c" not in store.state.usable_fields()


def test_implausible_readings_are_flagged_not_dropped():
    store = TwinStateStore()
    now = pd.Timestamp("2019-11-01 10:00", tz="UTC")
    store.apply(Measurement("sensor_zone_temp", now, 300.0, "degC"))
    assert store.state.quality("indoor_temp_c") is Quality.SUSPECT
    # Still recorded: a sensor reading 300 C is itself information.
    assert store.state.value("indoor_temp_c") == 300.0


def test_ordinary_sensor_noise_near_zero_is_not_suspect():
    """The regression that mattered: a hard zero floor on power flagged 229 of
    1,641 clean readings, all of them a +/-2 kW meter reading slightly negative
    at night."""
    store = TwinStateStore()
    now = pd.Timestamp("2019-11-01 02:00", tz="UTC")
    store.apply(Measurement("sensor_pv_power", now, -1.4, "kW"))
    store.apply(Measurement("sensor_hvac_power", now, -0.9, "kW"))
    assert store.flagged_suspect == 0


def test_unknown_sensors_are_counted_not_raised():
    store = TwinStateStore()
    now = pd.Timestamp("2019-11-01 10:00", tz="UTC")
    assert not store.apply(Measurement("sensor_unknown", now, 1.0, "kW"))
    assert store.rejected_unknown_sensor == 1


def test_measurement_without_a_unit_is_refused():
    with pytest.raises(ValueError, match="not a measurement"):
        Measurement("sensor_zone_temp", pd.Timestamp("2019-11-01", tz="UTC"), 1.0, "")


# --- ingestion -----------------------------------------------------------


def test_telemetry_is_degraded_not_exact(spec, short_drivers):
    """The twin must not receive the plant's internal state."""
    plant = Plant(spec, short_drivers)
    observed = record_telemetry(
        plant, ThermostatController(spec), PlantStreamProvider(dropout_rate=0.0)
    )
    truth = plant.history_frame()
    difference = (observed["indoor_temp_c"] - truth["indoor_temp_c"]).abs()
    assert difference.max() > 0.0, "telemetry is identical to ground truth"
    assert difference.mean() < 1.0, "telemetry noise is implausibly large"


def test_dropouts_produce_gaps_not_zeros(spec, short_drivers):
    observed = record_telemetry(
        Plant(spec, short_drivers),
        ThermostatController(spec),
        PlantStreamProvider(dropout_rate=0.5),
    )
    assert observed["indoor_temp_c"].isna().sum() > 0


def test_csv_replay_requires_time_order(spec, short_drivers):
    plant = Plant(spec, short_drivers)
    frame = run_controller(ThermostatController(spec), plant).frame
    with pytest.raises(ValueError, match="time order"):
        CsvReplayProvider(frame.iloc[::-1])
