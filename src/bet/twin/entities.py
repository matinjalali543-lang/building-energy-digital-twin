"""The twin's data model — the brief's §6 entities.

This is the twin's *description* of the physical system: what exists, what it is
called, what units it reports in, and what state each thing carries. It is
deliberately separate from `bet.plant`, which is the physical system itself.

Nothing here imports from `bet.plant`, and `tests/test_twin_isolation.py`
enforces that. The twin must be constructible from a site survey and a list of
sensor tags — the kind of information you would actually have about a real
building — without access to whatever is generating the measurements.

Every entity carries an `entity_id`, human-readable `metadata`, and where it
holds a measured quantity, an explicit `unit`. Units are strings rather than a
unit library on purpose: the aim is that a reading can never be stored without
someone having written down what it is in, not to do dimensional algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AssetKind(StrEnum):
    """What sort of thing an asset is. Drives which residual checks apply."""

    HVAC = "hvac"
    PV = "pv"
    BATTERY = "battery"
    METER = "meter"
    SENSOR = "sensor"
    ZONE = "zone"


@dataclass(frozen=True)
class Entity:
    """Base for everything in the model."""

    entity_id: str
    name: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")


@dataclass(frozen=True)
class Sensor(Entity):
    """A measuring instrument.

    `nominal_accuracy` and `max_staleness_minutes` are not decoration: the state
    estimator uses staleness to decide when a held value stops being usable, and
    the anomaly detector uses accuracy to set a floor below which a residual is
    just instrument noise.
    """

    unit: str = ""
    quantity: str = ""  # e.g. "temperature", "power"
    asset_id: str | None = None
    nominal_accuracy: float = 0.0
    max_staleness_minutes: int = 120

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.unit:
            raise ValueError(
                f"sensor {self.entity_id!r} has no unit. A reading whose unit "
                "nobody wrote down is not a measurement."
            )
        if self.max_staleness_minutes <= 0:
            raise ValueError(
                f"sensor {self.entity_id!r}: max_staleness_minutes must be > 0"
            )
        if self.nominal_accuracy < 0:
            raise ValueError(
                f"sensor {self.entity_id!r}: nominal_accuracy must be >= 0"
            )


@dataclass(frozen=True)
class EnergyMeter(Entity):
    """A revenue or sub-meter. Reports power; energy is its integral."""

    unit: str = "kW"
    measures: str = "grid_import"
    sensor_id: str | None = None


@dataclass(frozen=True)
class WeatherStation(Entity):
    """Where outdoor conditions come from.

    `is_onsite` matters for fault attribution. A rooftop sensor failing is a
    site problem; a forecast feed going stale is a supplier problem, and the two
    look identical in the data unless the model records which one it is.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    is_onsite: bool = True
    provides: tuple[str, ...] = ("outdoor_temp_c", "irradiance_w_m2")


@dataclass(frozen=True)
class Asset(Entity):
    """A piece of equipment with a rating and a performance expectation."""

    kind: AssetKind = AssetKind.HVAC
    rated_capacity: float = 0.0
    capacity_unit: str = "kW"
    zone_id: str | None = None
    commissioned: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rated_capacity < 0:
            raise ValueError(f"asset {self.entity_id!r}: rated_capacity must be >= 0")


@dataclass(frozen=True)
class PVSystem(Asset):
    """A photovoltaic array.

    Expected output is a function of measured irradiance, which is what makes
    soiling detectable at all — raw generation falls every autumn for entirely
    innocent reasons.
    """

    kind: AssetKind = AssetKind.PV
    peak_power_kwp: float = 0.0
    tilt_degrees: float = 30.0
    azimuth_degrees: float = 0.0

    def expected_output_kw(
        self, irradiance_w_m2: float, reference_irradiance: float = 1000.0
    ) -> float:
        """First-order expectation: output proportional to plane-of-array irradiance.

        Crude — it ignores the temperature coefficient and inverter clipping —
        and deliberately so. A performance expectation that is simple and
        slightly wrong in a *known* direction is more useful for fault detection
        than a complex one whose errors nobody can characterise.
        """
        return self.peak_power_kwp * irradiance_w_m2 / reference_irradiance


@dataclass(frozen=True)
class BatteryAsset(Asset):
    """Electrical storage, as the twin understands it."""

    kind: AssetKind = AssetKind.BATTERY
    energy_capacity_kwh: float = 0.0
    power_kw: float = 0.0
    round_trip_efficiency: float = 0.9


@dataclass(frozen=True)
class Zone(Entity):
    """A thermal zone. This building has one; the model allows more."""

    floor_area_m2: float = 0.0
    setpoint_min_c: float = 21.0
    setpoint_max_c: float = 24.0
    sensor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.setpoint_max_c <= self.setpoint_min_c:
            raise ValueError(
                f"zone {self.entity_id!r}: setpoint_max_c must exceed setpoint_min_c"
            )


@dataclass(frozen=True)
class Tariff(Entity):
    """How energy is priced.

    Import and export are separate because they are genuinely different prices,
    and `import_markup_eur_mwh` exists because the wholesale price is not what a
    building pays — network charges and levies ride on every kWh.
    """

    import_markup_eur_mwh: float = 60.0
    export_ratio: float = 0.7
    demand_charge_eur_kw_month: float = 0.0

    def import_price(self, wholesale_eur_mwh: float) -> float:
        return wholesale_eur_mwh + self.import_markup_eur_mwh

    def export_price(self, wholesale_eur_mwh: float) -> float:
        return wholesale_eur_mwh * self.export_ratio


@dataclass(frozen=True)
class Building(Entity):
    """The root of the model: zones, assets, sensors, meters, weather, tariff."""

    floor_area_m2: float = 0.0
    zones: tuple[Zone, ...] = ()
    assets: tuple[Asset, ...] = ()
    sensors: tuple[Sensor, ...] = ()
    meters: tuple[EnergyMeter, ...] = ()
    weather: WeatherStation | None = None
    tariff: Tariff | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        ids = [e.entity_id for e in (*self.zones, *self.assets, *self.sensors, *self.meters)]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate entity_id(s) in building {self.entity_id!r}: {duplicates}. "
                "Ids address measurements, so a collision silently merges two things."
            )
        known_zones = {zone.entity_id for zone in self.zones}
        orphans = [
            asset.entity_id
            for asset in self.assets
            if asset.zone_id is not None and asset.zone_id not in known_zones
        ]
        if orphans:
            raise ValueError(
                f"asset(s) {orphans} reference a zone that does not exist in this building"
            )

    def asset(self, entity_id: str) -> Asset:
        for asset in self.assets:
            if asset.entity_id == entity_id:
                return asset
        raise KeyError(f"no asset {entity_id!r}; have {[a.entity_id for a in self.assets]}")

    def sensor(self, entity_id: str) -> Sensor:
        for sensor in self.sensors:
            if sensor.entity_id == entity_id:
                return sensor
        raise KeyError(f"no sensor {entity_id!r}; have {[s.entity_id for s in self.sensors]}")

    def assets_of_kind(self, kind: AssetKind) -> tuple[Asset, ...]:
        return tuple(asset for asset in self.assets if asset.kind == kind)

    def describe(self) -> str:
        return (
            f"{self.name or self.entity_id}: {self.floor_area_m2:,.0f} m2, "
            f"{len(self.zones)} zone(s), {len(self.assets)} asset(s), "
            f"{len(self.sensors)} sensor(s), {len(self.meters)} meter(s)"
        )


def reference_building_model() -> Building:
    """The twin's description of the reference office.

    Written as a site survey would be — ratings, tags, units — and deliberately
    *not* derived from `BuildingSpec`. If it were generated from the plant's
    parameters the twin would know the building's true UA and capacitance, which
    no real twin does, and the calibration work in `bet.twin.calibration` would
    be answering a question it had already been given.
    """
    zone = Zone(
        entity_id="zone_main",
        name="Open plan office",
        floor_area_m2=7500.0,
        setpoint_min_c=21.0,
        setpoint_max_c=24.0,
        sensor_ids=("sensor_zone_temp",),
    )
    return Building(
        entity_id="building_ref",
        name="Reference Office",
        metadata={
            "use": "office",
            "location": "Berlin, DE",
            "note": "simulated building -- see docs/decisions.md section B",
        },
        floor_area_m2=7500.0,
        zones=(zone,),
        assets=(
            Asset(
                entity_id="hvac_main",
                name="Reversible heat pump",
                kind=AssetKind.HVAC,
                rated_capacity=450.0,
                capacity_unit="kW thermal",
                zone_id="zone_main",
            ),
            PVSystem(
                entity_id="pv_roof",
                name="Rooftop array",
                rated_capacity=250.0,
                capacity_unit="kWp",
                peak_power_kwp=250.0,
                tilt_degrees=30.0,
                azimuth_degrees=0.0,
            ),
            BatteryAsset(
                entity_id="battery_main",
                name="Site battery",
                rated_capacity=125.0,
                capacity_unit="kW",
                energy_capacity_kwh=250.0,
                power_kw=125.0,
                round_trip_efficiency=0.9025,
            ),
        ),
        sensors=(
            Sensor(
                entity_id="sensor_zone_temp",
                name="Zone temperature",
                unit="degC",
                quantity="temperature",
                asset_id="zone_main",
                nominal_accuracy=0.3,
                max_staleness_minutes=120,
            ),
            Sensor(
                entity_id="sensor_outdoor_temp",
                name="Outdoor air temperature",
                unit="degC",
                quantity="temperature",
                nominal_accuracy=0.5,
                max_staleness_minutes=180,
            ),
            Sensor(
                entity_id="sensor_irradiance",
                name="Plane-of-array irradiance",
                unit="W/m2",
                quantity="irradiance",
                nominal_accuracy=15.0,
                max_staleness_minutes=180,
            ),
            Sensor(
                entity_id="sensor_pv_power",
                name="PV inverter output",
                unit="kW",
                quantity="power",
                asset_id="pv_roof",
                nominal_accuracy=2.0,
                max_staleness_minutes=60,
            ),
            Sensor(
                entity_id="sensor_battery_soc",
                name="Battery state of charge",
                unit="kWh",
                quantity="energy",
                asset_id="battery_main",
                nominal_accuracy=2.5,
                max_staleness_minutes=60,
            ),
        ),
        meters=(
            EnergyMeter(
                entity_id="meter_grid",
                name="Main incomer",
                unit="kW",
                measures="grid_import",
                sensor_id="sensor_grid_power",
            ),
            EnergyMeter(
                entity_id="meter_hvac",
                name="HVAC sub-meter",
                unit="kW",
                measures="hvac_electric",
                sensor_id="sensor_hvac_power",
            ),
        ),
        weather=WeatherStation(
            entity_id="weather_site",
            name="Rooftop weather station",
            latitude=52.52,
            longitude=13.405,
            is_onsite=True,
        ),
        tariff=Tariff(
            entity_id="tariff_default",
            name="Wholesale pass-through plus network charges",
            import_markup_eur_mwh=60.0,
            export_ratio=0.7,
        ),
    )
