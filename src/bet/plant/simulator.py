"""The "physical" building — the plant the twin is a twin *of*.

This module is the ground truth, and it is deliberately walled off from
`bet.twin`. The plant knows its own indoor temperature exactly; the twin only
ever sees what `Plant.telemetry()` chooses to emit, complete with sensor noise,
dropouts and quality flags. Nothing in `bet.twin` imports from here, and a test
enforces that.

That wall is the whole reason the project can claim anything. A "digital twin"
that reads the simulator's internals is measuring itself.

## The model

A single-zone RC network, integrated with explicit Euler at the driver
resolution:

    C dT/dt = UA (T_out - T) + Q_internal + Q_solar + Q_hvac

with a separate, purely bookkeeping electrical balance:

    grid_import - grid_export
        = base_load + hvac_electric + battery_charge - battery_discharge - pv

Explicit Euler is stable here by a wide margin — the default building's time
constant is 50 hours against a one-hour step — and `tests/test_thermal_model.py`
pins both the decay rate and the closure of the energy balance.

## The building is sealed, and that has a visible consequence

There is no free cooling here: no window opening, no night purge, no economizer
cycle. `ua_w_per_k` is a single fixed conductance covering fabric and a constant
ventilation rate, so the only way to reject heat is the mechanical chiller.

The consequence shows up immediately if you run `NullController`: free-running
indoor temperature reaches **58 °C** in summer. That is not a broken model, it is
the model being consistent. At peak the building takes about 218 kW of internal
gain plus roughly 97 kW of solar, and

    315 kW / 5 kW/K = 63 K above outdoor air

is simply where a sealed box with that conductance settles. Real offices never
get there because they ventilate, and this one cannot.

So `NullController` is a **diagnostic for unforced dynamics, not a credible
operating mode**, and the free-running row is never quoted as a baseline. Every
reported result uses mechanical cooling, where the chiller holds the band and
the number never arises. Adding a free-cooling term is listed as future work: it
is easy in the simulator and awkward in the optimizer, because ventilation heat
transfer is a product of a decision and a temperature difference and therefore
not linear.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from bet.plant.building import SECONDS_PER_HOUR, BuildingSpec

LOCAL_TZ = "Europe/Berlin"


@dataclass(frozen=True)
class ControlAction:
    """What a controller asks the plant to do in one interval.

    HVAC thermal power is **signed**: positive heats, negative cools. One signed
    number rather than two non-negative ones, because a single reversible unit
    physically cannot do both at once — the sign convention makes that
    impossible to express rather than merely discouraged.

    The optimizer works with separate non-negative variables and a mode binary,
    because an MILP cannot take a sign convention as a constraint; it collapses
    them to this signed form on the way out.
    """

    hvac_thermal_kw: float = 0.0
    # Positive charges the battery, negative discharges it.
    battery_kw: float = 0.0

    def clipped(self, spec: BuildingSpec) -> ControlAction:
        """Clip to what the equipment can actually deliver."""
        hvac = float(
            np.clip(
                self.hvac_thermal_kw,
                -spec.hvac.cooling_capacity_kw_thermal,
                spec.hvac.heating_capacity_kw_thermal,
            )
        )
        battery = self.battery_kw
        if spec.battery is None:
            battery = 0.0
        else:
            battery = float(np.clip(battery, -spec.battery.power_kw, spec.battery.power_kw))
        return replace(self, hvac_thermal_kw=hvac, battery_kw=battery)


@dataclass(frozen=True)
class PlantState:
    """True state. The twin never sees this object."""

    timestamp: pd.Timestamp
    indoor_temp_c: float
    battery_soc_kwh: float
    step_index: int = 0


@dataclass(frozen=True)
class PlantReading:
    """One interval of plant output, before sensors get hold of it.

    Every quantity that a meter or sensor could observe, plus the true state.
    `Plant.telemetry()` degrades this into what the twin is allowed to see.
    """

    timestamp: pd.Timestamp
    indoor_temp_c: float
    outdoor_temp_c: float
    irradiance_w_m2: float
    occupancy_fraction: float
    base_load_kw: float
    hvac_electric_kw: float
    hvac_thermal_kw: float
    pv_kw: float
    battery_kw: float
    battery_soc_kwh: float
    grid_import_kw: float
    grid_export_kw: float
    price_eur_mwh: float

    @property
    def site_load_kw(self) -> float:
        """Everything the building consumes, before PV and battery."""
        return self.base_load_kw + self.hvac_electric_kw

    @property
    def net_cost_eur(self) -> float:
        """Cost of this interval at the wholesale price (one-hour steps)."""
        return (self.grid_import_kw - self.grid_export_kw) * self.price_eur_mwh / 1e3


def occupancy_fraction(index: pd.DatetimeIndex, spec: BuildingSpec) -> np.ndarray:
    """Fraction of design occupancy, by local time."""
    local = index.tz_convert(LOCAL_TZ) if index.tz is not None else index
    hour = local.hour.to_numpy()
    weekday = local.dayofweek.to_numpy()

    in_hours = (hour >= spec.schedule.start_hour) & (hour < spec.schedule.end_hour)
    if spec.schedule.weekdays_only:
        in_hours &= weekday < 5
    return np.where(in_hours, 1.0, spec.schedule.setback_fraction)


def internal_gain_kw(occupancy: np.ndarray, spec: BuildingSpec) -> np.ndarray:
    """Heat from people, lighting and equipment, kW.

    People scale with occupancy directly. Lighting and equipment fall to
    `base_load_fraction` rather than to zero, because servers, lifts and standby
    loads do not go home.
    """
    occupied = occupancy > spec.schedule.setback_fraction
    lit_fraction = np.where(occupied, 1.0, spec.base_load_fraction)
    watts = spec.floor_area_m2 * (
        spec.people_gain_w_m2 * occupancy
        + (spec.lighting_gain_w_m2 + spec.plug_gain_w_m2) * lit_fraction
    )
    return watts / 1e3


def electrical_base_load_kw(occupancy: np.ndarray, spec: BuildingSpec) -> np.ndarray:
    """Non-HVAC electricity: lighting and plug load only.

    People generate heat but do not draw power, so they appear in
    `internal_gain_kw` and not here. Conflating the two is a common way to
    overstate a building's electrical load by 20%.
    """
    occupied = occupancy > spec.schedule.setback_fraction
    lit_fraction = np.where(occupied, 1.0, spec.base_load_fraction)
    watts = (
        spec.floor_area_m2
        * (spec.lighting_gain_w_m2 + spec.plug_gain_w_m2)
        * lit_fraction
    )
    return watts / 1e3


def comfort_bounds(
    index: pd.DatetimeIndex, spec: BuildingSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper indoor temperature limits for each interval."""
    occupancy = occupancy_fraction(index, spec)
    occupied = occupancy > spec.schedule.setback_fraction
    lower = np.where(occupied, spec.comfort.occupied_min_c, spec.comfort.setback_min_c)
    upper = np.where(occupied, spec.comfort.occupied_max_c, spec.comfort.setback_max_c)
    return lower, upper


class Plant:
    """A simulated building advancing one interval at a time.

    Drivers (weather, PV, price) are supplied up front; the controller supplies
    an action each step. Faults, if any, are applied by wrapping the plant with
    `bet.plant.faults.FaultyPlant`.
    """

    def __init__(
        self,
        spec: BuildingSpec,
        drivers: pd.DataFrame,
        initial_indoor_temp_c: float = 21.0,
        dt_hours: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        required = {"outdoor_temp_c", "irradiance_w_m2", "pv_kw", "price_eur_mwh"}
        missing = required - set(drivers.columns)
        if missing:
            raise ValueError(f"drivers frame is missing column(s): {sorted(missing)}")
        if drivers.empty:
            raise ValueError("drivers frame is empty")
        if not drivers.index.is_monotonic_increasing:
            raise ValueError("drivers must be sorted by timestamp")

        band = (spec.comfort.setback_min_c, spec.comfort.setback_max_c)
        if not band[0] - 10 <= initial_indoor_temp_c <= band[1] + 10:
            raise ValueError(
                f"initial_indoor_temp_c ({initial_indoor_temp_c}) is implausibly far "
                f"outside the comfort band {band}"
            )

        self.spec = spec
        self.drivers = drivers
        self.dt_hours = dt_hours
        self.rng = rng if rng is not None else np.random.default_rng(20260817)

        self._occupancy = occupancy_fraction(drivers.index, spec)
        self._gain_kw = internal_gain_kw(self._occupancy, spec)
        self._base_kw = electrical_base_load_kw(self._occupancy, spec)
        # PV scales linearly with array size; the driver file is for the spec's
        # nominal rating, so a scenario that changes it rescales here.
        self._pv_scale = spec.pv_capacity_kwp / 250.0

        soc0 = (
            spec.battery.initial_soc_frac * spec.battery.energy_capacity_kwh
            if spec.battery
            else 0.0
        )
        self.state = PlantState(
            timestamp=drivers.index[0],
            indoor_temp_c=initial_indoor_temp_c,
            battery_soc_kwh=soc0,
            step_index=0,
        )
        self.history: list[PlantReading] = []

    # --- introspection the controller is allowed to use --------------------

    @property
    def n_steps(self) -> int:
        return len(self.drivers)

    @property
    def finished(self) -> bool:
        return self.state.step_index >= self.n_steps

    def occupancy_at(self, step: int) -> float:
        return float(self._occupancy[step])

    def comfort_at(self, step: int) -> tuple[float, float]:
        lower, upper = comfort_bounds(self.drivers.index[[step]], self.spec)
        return float(lower[0]), float(upper[0])

    # --- the step function -------------------------------------------------

    def step(self, action: ControlAction) -> PlantReading:
        """Advance one interval under `action` and return what happened."""
        if self.finished:
            raise RuntimeError(
                f"the plant has already run all {self.n_steps} intervals; "
                "construct a new Plant to run again"
            )

        index = self.state.step_index
        row = self.drivers.iloc[index]
        action = action.clipped(self.spec)

        solar_kw = row["irradiance_w_m2"] * self.spec.solar_aperture_m2 * 0.6 / 1e3
        gain_kw = float(self._gain_kw[index])
        base_kw = float(self._base_kw[index])

        # --- thermal update
        temp = self.state.indoor_temp_c
        net_w = (
            self.spec.ua_w_per_k * (row["outdoor_temp_c"] - temp)
            + (gain_kw + solar_kw + action.hvac_thermal_kw) * 1e3
        )
        new_temp = temp + net_w * self.dt_hours * SECONDS_PER_HOUR / self.spec.capacitance_j_per_k

        # --- HVAC electricity, from the signed thermal demand
        if action.hvac_thermal_kw >= 0:
            hvac_electric = action.hvac_thermal_kw / self.spec.hvac.cop_heating
        else:
            hvac_electric = -action.hvac_thermal_kw / self.spec.hvac.cop_cooling

        # --- battery, clipped to what the state of charge allows
        battery_kw, new_soc = self._apply_battery(action.battery_kw)

        pv_kw = float(row["pv_kw"]) * self._pv_scale
        net_kw = base_kw + hvac_electric + battery_kw - pv_kw
        grid_import = max(net_kw, 0.0)
        surplus = max(-net_kw, 0.0)
        grid_export = min(surplus, self.spec.grid_export_limit_kw)

        reading = PlantReading(
            timestamp=self.state.timestamp,
            indoor_temp_c=new_temp,
            outdoor_temp_c=float(row["outdoor_temp_c"]),
            irradiance_w_m2=float(row["irradiance_w_m2"]),
            occupancy_fraction=float(self._occupancy[index]),
            base_load_kw=base_kw,
            hvac_electric_kw=hvac_electric,
            hvac_thermal_kw=action.hvac_thermal_kw,
            pv_kw=pv_kw,
            battery_kw=battery_kw,
            battery_soc_kwh=new_soc,
            grid_import_kw=grid_import,
            grid_export_kw=grid_export,
            price_eur_mwh=float(row["price_eur_mwh"]),
        )
        self.history.append(reading)

        next_index = index + 1
        self.state = PlantState(
            timestamp=(
                self.drivers.index[next_index]
                if next_index < self.n_steps
                else self.state.timestamp + pd.Timedelta(hours=self.dt_hours)
            ),
            indoor_temp_c=new_temp,
            battery_soc_kwh=new_soc,
            step_index=next_index,
        )
        return reading

    def _apply_battery(self, requested_kw: float) -> tuple[float, float]:
        """Clip a battery request to the energy limits and apply it."""
        battery = self.spec.battery
        if battery is None or requested_kw == 0.0:
            return 0.0, self.state.battery_soc_kwh

        soc = self.state.battery_soc_kwh
        floor = battery.soc_min_frac * battery.energy_capacity_kwh
        ceiling = battery.soc_max_frac * battery.energy_capacity_kwh

        if requested_kw > 0:  # charging
            headroom_kw = (ceiling - soc) / (battery.charge_efficiency * self.dt_hours)
            power = min(requested_kw, max(headroom_kw, 0.0))
            soc += battery.charge_efficiency * power * self.dt_hours
        else:  # discharging
            available_kw = (soc - floor) * battery.discharge_efficiency / self.dt_hours
            power = -min(-requested_kw, max(available_kw, 0.0))
            soc += power * self.dt_hours / battery.discharge_efficiency

        return power, float(np.clip(soc, floor, ceiling))

    # --- output -------------------------------------------------------------

    def history_frame(self) -> pd.DataFrame:
        """Everything that has happened so far, as a frame."""
        if not self.history:
            return pd.DataFrame()
        frame = pd.DataFrame([vars(r) for r in self.history]).set_index("timestamp")
        frame["site_load_kw"] = frame["base_load_kw"] + frame["hvac_electric_kw"]
        frame["cost_eur"] = (
            (frame["grid_import_kw"] - frame["grid_export_kw"]) * frame["price_eur_mwh"] / 1e3
        )
        return frame

    def energy_balance_residual(self) -> np.ndarray:
        """Electrical balance residual per interval. Should be ~0 everywhere.

        Exposed rather than hidden inside a test so that a scenario or a fault
        that quietly breaks conservation shows up in the results, not only in CI.
        """
        frame = self.history_frame()
        if frame.empty:
            return np.zeros(0)
        return (
            frame["grid_import_kw"]
            - frame["grid_export_kw"]
            - frame["site_load_kw"]
            - frame["battery_kw"]
            + frame["pv_kw"]
        ).to_numpy()


@dataclass
class SimulationResult:
    """A completed run: what the plant did, and what it cost."""

    frame: pd.DataFrame
    spec: BuildingSpec
    label: str = "run"
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost_eur(self) -> float:
        return float(self.frame["cost_eur"].sum())

    @property
    def hvac_energy_kwh(self) -> float:
        return float(self.frame["hvac_electric_kw"].sum())

    @property
    def site_energy_kwh(self) -> float:
        return float(self.frame["site_load_kw"].sum())

    @property
    def grid_energy_kwh(self) -> float:
        return float(self.frame["grid_import_kw"].sum())

    @property
    def peak_import_kw(self) -> float:
        return float(self.frame["grid_import_kw"].max())

    def comfort_violations(self, tolerance_k: float = 0.5) -> int:
        """Occupied intervals outside the comfort band by more than `tolerance_k`.

        A tolerance is applied because a controller that lands exactly on the
        bound will cross it by numerical noise, and counting that as discomfort
        would make every result look worse than it is.
        """
        lower, upper = comfort_bounds(self.frame.index, self.spec)
        occupied = self.frame["occupancy_fraction"].to_numpy() > (
            self.spec.schedule.setback_fraction
        )
        temp = self.frame["indoor_temp_c"].to_numpy()
        breached = (temp < lower - tolerance_k) | (temp > upper + tolerance_k)
        return int((breached & occupied).sum())

    def summary(self) -> str:
        return (
            f"{self.label}: {self.total_cost_eur:,.0f} EUR, "
            f"HVAC {self.hvac_energy_kwh / 1e3:,.1f} MWh, "
            f"site {self.site_energy_kwh / 1e3:,.1f} MWh, "
            f"peak {self.peak_import_kw:,.0f} kW, "
            f"{self.comfort_violations()} comfort violations"
        )
