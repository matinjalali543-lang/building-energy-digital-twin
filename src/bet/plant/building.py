"""The reference building: a validated specification of the simulated plant.

Every number here is an **assumption about a hypothetical building**, not a
measurement of a real one. They are chosen to land inside published ranges for
European offices rather than to flatter any result, and
`bet.plant.calibration` checks that they do.

The physics is a single-zone RC model — one thermal capacitance behind one
conductance. That is the standard reduced-order representation used for building
control, and it is deliberately not a whole-building simulation: it has few
enough parameters to state honestly, and it is linear, which is what lets the
controller be an MILP rather than a black box.

Its main limitation is stated rather than hidden: one zone cannot represent a
building where the south face bakes while the north face freezes. See
`docs/formulation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class OccupancySchedule:
    """When the building is in use, in **local** time.

    Local, not UTC: people follow the local clock, and a schedule expressed in
    UTC drifts by an hour across the daylight-saving boundary.
    """

    start_hour: int = 7
    end_hour: int = 19
    weekdays_only: bool = True
    # Fraction of design occupancy outside working hours: cleaners, servers,
    # security. Not zero, because a building that empties completely is a
    # simpler thermal problem than a real one.
    setback_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.start_hour < 24:
            raise ValueError(f"start_hour must be in [0, 24), got {self.start_hour}")
        if not 0 < self.end_hour <= 24:
            raise ValueError(f"end_hour must be in (0, 24], got {self.end_hour}")
        if self.end_hour <= self.start_hour:
            raise ValueError(
                f"end_hour ({self.end_hour}) must be after start_hour ({self.start_hour})"
            )
        if not 0 <= self.setback_fraction <= 1:
            raise ValueError(
                f"setback_fraction must be in [0, 1], got {self.setback_fraction}"
            )

    @property
    def occupied_hours_per_week(self) -> int:
        days = 5 if self.weekdays_only else 7
        return (self.end_hour - self.start_hour) * days


@dataclass(frozen=True)
class ComfortBand:
    """Indoor temperature limits, tighter when people are present.

    The setback band is what the optimizer trades against: a wide unoccupied
    band is thermal storage, and a narrow one is not.
    """

    occupied_min_c: float = 21.0
    occupied_max_c: float = 24.0
    setback_min_c: float = 16.0
    setback_max_c: float = 28.0

    def __post_init__(self) -> None:
        if self.occupied_max_c <= self.occupied_min_c:
            raise ValueError(
                f"occupied_max_c ({self.occupied_max_c}) must exceed "
                f"occupied_min_c ({self.occupied_min_c})"
            )
        if self.setback_max_c <= self.setback_min_c:
            raise ValueError(
                f"setback_max_c ({self.setback_max_c}) must exceed "
                f"setback_min_c ({self.setback_min_c})"
            )
        # A setback band narrower than the occupied band would be perverse: it
        # would mean holding tighter control when nobody is there.
        if self.setback_min_c > self.occupied_min_c:
            raise ValueError(
                f"setback_min_c ({self.setback_min_c}) must not be above "
                f"occupied_min_c ({self.occupied_min_c}) -- the setback band is "
                "meant to be looser, not tighter"
            )
        if self.setback_max_c < self.occupied_max_c:
            raise ValueError(
                f"setback_max_c ({self.setback_max_c}) must not be below "
                f"occupied_max_c ({self.occupied_max_c})"
            )

    @property
    def occupied_width_k(self) -> float:
        return self.occupied_max_c - self.occupied_min_c


@dataclass(frozen=True)
class HvacSpec:
    """A reversible heat pump serving the whole zone.

    Coefficients of performance are **constant**, which is the model's most
    consequential simplification: a real heat pump loses capacity and efficiency
    exactly when it is needed most, on the coldest days. Stated in
    `docs/formulation.md` rather than buried.
    """

    heating_capacity_kw_thermal: float = 450.0
    cooling_capacity_kw_thermal: float = 400.0
    cop_heating: float = 3.0
    cop_cooling: float = 3.5

    def __post_init__(self) -> None:
        for attr in (
            "heating_capacity_kw_thermal",
            "cooling_capacity_kw_thermal",
            "cop_heating",
            "cop_cooling",
        ):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be > 0, got {getattr(self, attr)}")
        if self.cop_heating > 6 or self.cop_cooling > 8:
            raise ValueError(
                "a constant COP above 6 (heating) or 8 (cooling) is not a heat "
                "pump, it is a rounding error waiting to flatter the results"
            )

    @property
    def max_electrical_kw(self) -> float:
        """Largest electrical draw either mode can produce."""
        return max(
            self.heating_capacity_kw_thermal / self.cop_heating,
            self.cooling_capacity_kw_thermal / self.cop_cooling,
        )


@dataclass(frozen=True)
class BatterySpec:
    """The optional BESS. Secondary here -- the fabric is the primary store."""

    energy_capacity_kwh: float = 250.0
    power_kw: float = 125.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    initial_soc_frac: float = 0.50

    def __post_init__(self) -> None:
        if self.energy_capacity_kwh <= 0:
            raise ValueError(f"energy_capacity_kwh must be > 0, got {self.energy_capacity_kwh}")
        if self.power_kw <= 0:
            raise ValueError(f"power_kw must be > 0, got {self.power_kw}")
        for attr in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < getattr(self, attr) <= 1:
                raise ValueError(f"{attr} must be in (0, 1], got {getattr(self, attr)}")
        if self.soc_max_frac <= self.soc_min_frac:
            raise ValueError("soc_max_frac must exceed soc_min_frac")
        if not self.soc_min_frac <= self.initial_soc_frac <= self.soc_max_frac:
            raise ValueError(
                f"initial_soc_frac ({self.initial_soc_frac}) must lie within "
                f"[{self.soc_min_frac}, {self.soc_max_frac}]"
            )

    @property
    def usable_kwh(self) -> float:
        return (self.soc_max_frac - self.soc_min_frac) * self.energy_capacity_kwh


@dataclass(frozen=True)
class BuildingSpec:
    """A single-zone office building.

    Default parameters describe a 7,500 m² medium-weight European office. See
    `bet.plant.calibration` for the benchmark comparison that justifies them.
    """

    name: str = "Reference Office"
    floor_area_m2: float = 7500.0

    # Envelope conductance plus ventilation, lumped. W per kelvin of
    # inside-outside difference.
    ua_w_per_k: float = 5000.0
    # Effective thermal capacitance, J/K. Expressed per square metre of floor
    # because that is how construction weight is usually quoted:
    # ~80 kJ/m2K lightweight, ~120 medium, ~200+ heavyweight.
    capacitance_j_per_m2_k: float = 120e3

    # Glazing area times solar heat gain coefficient times shading factor.
    solar_aperture_m2: float = 180.0

    # Internal gains at full occupancy, W/m2.
    people_gain_w_m2: float = 8.0
    lighting_gain_w_m2: float = 9.0
    plug_gain_w_m2: float = 12.0
    # Fraction of lighting and plug load that persists overnight.
    base_load_fraction: float = 0.15
    # Day-to-day variability in lighting and plug load, as a standard deviation
    # of a multiplicative factor. Not decoration: with it set to zero the base
    # load is an exact function of the calendar, and a forecaster with
    # hour-of-day features predicts it with R^2 = 1.0000 and MAE = 0.0000.
    # That is an artefact of a noiseless simulator, not a fact about buildings,
    # and reporting it would look indistinguishable from a leak. Real plug and
    # lighting loads move with headcount, weather and what people are doing.
    load_variability: float = 0.12

    pv_capacity_kwp: float = 250.0
    grid_import_limit_kw: float = 600.0
    grid_export_limit_kw: float = 250.0

    schedule: OccupancySchedule = field(default_factory=OccupancySchedule)
    comfort: ComfortBand = field(default_factory=ComfortBand)
    hvac: HvacSpec = field(default_factory=HvacSpec)
    battery: BatterySpec | None = field(default_factory=BatterySpec)

    def __post_init__(self) -> None:
        for attr in (
            "floor_area_m2",
            "ua_w_per_k",
            "capacitance_j_per_m2_k",
            "pv_capacity_kwp",
            "grid_import_limit_kw",
        ):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be > 0, got {getattr(self, attr)}")
        for attr in (
            "solar_aperture_m2",
            "people_gain_w_m2",
            "lighting_gain_w_m2",
            "plug_gain_w_m2",
            "grid_export_limit_kw",
        ):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be >= 0, got {getattr(self, attr)}")
        if not 0 <= self.base_load_fraction <= 1:
            raise ValueError(
                f"base_load_fraction must be in [0, 1], got {self.base_load_fraction}"
            )
        if not 0 <= self.load_variability < 1:
            raise ValueError(
                f"load_variability must be in [0, 1), got {self.load_variability}"
            )
        # A time constant under an hour would make hourly simulation meaningless:
        # the building would reach equilibrium inside a single step, and the
        # thermal-storage argument the whole project rests on would be false.
        if self.time_constant_hours < 1.0:
            raise ValueError(
                f"time constant is {self.time_constant_hours:.2f} h, under one "
                "simulation step. With C/UA this small the building has no usable "
                "thermal inertia and an hourly model cannot represent it."
            )

    # --- derived quantities ------------------------------------------------

    @property
    def capacitance_j_per_k(self) -> float:
        return self.capacitance_j_per_m2_k * self.floor_area_m2

    @property
    def time_constant_hours(self) -> float:
        """RC time constant. How long free-running temperature takes to decay 63%."""
        return self.capacitance_j_per_k / self.ua_w_per_k / SECONDS_PER_HOUR

    @property
    def peak_internal_gain_kw(self) -> float:
        total = self.people_gain_w_m2 + self.lighting_gain_w_m2 + self.plug_gain_w_m2
        return total * self.floor_area_m2 / 1e3

    @property
    def base_electrical_kw(self) -> float:
        """Lighting and plug load outside occupied hours."""
        lit = self.lighting_gain_w_m2 + self.plug_gain_w_m2
        return lit * self.floor_area_m2 * self.base_load_fraction / 1e3

    @property
    def occupied_electrical_kw(self) -> float:
        """Lighting and plug load during occupied hours (people are not a load)."""
        lit = self.lighting_gain_w_m2 + self.plug_gain_w_m2
        return lit * self.floor_area_m2 / 1e3

    def thermal_storage_kwh(self, band_k: float | None = None) -> float:
        """Heat the fabric can absorb across a temperature band, kWh-thermal.

        This is the number that makes the building a battery. Across the
        occupied comfort band the default building holds roughly as much usable
        energy as a large commercial BESS — and unlike the BESS it is already
        paid for.
        """
        band = self.comfort.occupied_width_k if band_k is None else band_k
        return self.capacitance_j_per_k * band / 3.6e6

    def thermal_storage_kwh_electric(self, band_k: float | None = None) -> float:
        """The same store expressed as electricity, via the heating COP."""
        return self.thermal_storage_kwh(band_k) / self.hvac.cop_heating

    def summary(self) -> str:
        return (
            f"{self.name}: {self.floor_area_m2:,.0f} m2, "
            f"UA {self.ua_w_per_k / 1e3:.1f} kW/K, "
            f"tau {self.time_constant_hours:.0f} h, "
            f"fabric stores {self.thermal_storage_kwh():.0f} kWh-th "
            f"({self.thermal_storage_kwh_electric():.0f} kWh-e) across "
            f"{self.comfort.occupied_width_k:.0f} K"
        )


REFERENCE_BUILDING = BuildingSpec()
