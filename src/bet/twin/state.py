"""The twin's state: what it currently believes, and how confident it is.

This is the part that makes the project a twin rather than a model. A model
maps inputs to outputs. A twin **holds a state** that is updated by measurement,
survives gaps, and knows which parts of itself are trustworthy.

Three behaviours matter more than the rest, and all three are what a real
telemetry feed forces on you:

**Out-of-order arrivals are rejected, not applied.** A late packet carrying an
older timestamp must not overwrite a newer value. Applying it silently rewinds
the twin, and every downstream residual is then computed against a state that
never existed.

**Missing is not the same as unchanged.** A sensor that stops reporting leaves
its last value in place, but flagged `STALE` and ageing. Past
`max_staleness_minutes` it becomes `MISSING`. Silently forward-filling a dead
sensor for a week is how a twin ends up confidently wrong.

**Quality travels with the value.** Every field carries its own flag, so a
controller can refuse to act on a stale zone temperature while still trusting a
fresh meter reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import pandas as pd


class Quality(StrEnum):
    """How much to trust a field.

    Ordered from best to worst, which `worst_of` relies on.
    """

    GOOD = "good"
    STALE = "stale"  # last known value, still inside its staleness window
    SUSPECT = "suspect"  # present but failing a plausibility check
    MISSING = "missing"  # no usable value

    @property
    def usable(self) -> bool:
        """Whether a controller should act on this value at all."""
        return self in (Quality.GOOD, Quality.STALE)


_QUALITY_ORDER = [Quality.GOOD, Quality.STALE, Quality.SUSPECT, Quality.MISSING]


def worst_of(*qualities: Quality) -> Quality:
    """The worst quality among several — used to summarise a whole state."""
    if not qualities:
        return Quality.MISSING
    return max(qualities, key=_QUALITY_ORDER.index)


@dataclass(frozen=True)
class Measurement:
    """One reading, as it arrives from the field.

    Carries its own unit and quality because the brief's §7 requires it, and
    because a pipeline that strips either one cannot be debugged later.
    """

    sensor_id: str
    timestamp: pd.Timestamp
    value: float
    unit: str
    quality: Quality = Quality.GOOD

    def __post_init__(self) -> None:
        if not self.unit:
            raise ValueError(
                f"measurement from {self.sensor_id!r} has no unit; "
                "a bare number is not a measurement"
            )


@dataclass(frozen=True)
class FieldState:
    """One state variable: its value, when it was last really measured, its quality."""

    value: float | None = None
    measured_at: pd.Timestamp | None = None
    quality: Quality = Quality.MISSING

    def age_minutes(self, now: pd.Timestamp) -> float | None:
        if self.measured_at is None:
            return None
        return (now - self.measured_at).total_seconds() / 60.0


# The state variables the twin tracks, and the sensor each is fed by.
TRACKED_FIELDS: dict[str, str] = {
    "indoor_temp_c": "sensor_zone_temp",
    "outdoor_temp_c": "sensor_outdoor_temp",
    "irradiance_w_m2": "sensor_irradiance",
    "pv_kw": "sensor_pv_power",
    "battery_soc_kwh": "sensor_battery_soc",
    "grid_import_kw": "sensor_grid_power",
    "hvac_electric_kw": "sensor_hvac_power",
}

# Plausibility bounds. A reading outside these is flagged SUSPECT rather than
# dropped: the twin should say "I do not believe this" rather than pretend it
# never arrived, because a sensor reading -999 is itself information.
#
# Note the lower bounds on the non-negative quantities are slightly BELOW zero,
# and that is deliberate. An earlier version used a hard 0.0 floor and flagged
# **229 of 1,641 measurements** as SUSPECT over ten days of clean operation --
# every one of them a power sensor reading a small negative value at night,
# which is what a +/-2 kW instrument does when the true value is zero. A
# plausibility check that fires on ordinary instrument noise is not a check, it
# is a false-positive generator feeding straight into the anomaly detector.
#
# The allowances below are roughly three times each sensor's stated accuracy.
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "indoor_temp_c": (-10.0, 60.0),
    "outdoor_temp_c": (-40.0, 55.0),
    "irradiance_w_m2": (-50.0, 1400.0),
    "pv_kw": (-6.0, 1000.0),
    "battery_soc_kwh": (-8.0, 10000.0),
    "grid_import_kw": (-10.0, 5000.0),
    "hvac_electric_kw": (-6.0, 2000.0),
}


@dataclass(frozen=True)
class TwinState:
    """The twin's belief about the building at one instant."""

    timestamp: pd.Timestamp | None = None
    fields: dict[str, FieldState] = field(default_factory=dict)

    def get(self, name: str) -> FieldState:
        return self.fields.get(name, FieldState())

    def value(self, name: str, default: float | None = None) -> float | None:
        state = self.get(name)
        return state.value if state.value is not None else default

    def quality(self, name: str) -> Quality:
        return self.get(name).quality

    @property
    def overall_quality(self) -> Quality:
        if not self.fields:
            return Quality.MISSING
        return worst_of(*(f.quality for f in self.fields.values()))

    def usable_fields(self) -> dict[str, float]:
        """Only what a controller should be willing to act on."""
        return {
            name: state.value
            for name, state in self.fields.items()
            if state.value is not None and state.quality.usable
        }

    def to_series(self) -> pd.Series:
        data = {name: state.value for name, state in self.fields.items()}
        data["timestamp"] = self.timestamp
        data["overall_quality"] = self.overall_quality.value
        return pd.Series(data)

    def summary(self) -> str:
        good = sum(1 for f in self.fields.values() if f.quality is Quality.GOOD)
        return (
            f"{self.timestamp}: {good}/{len(self.fields)} fields GOOD, "
            f"overall {self.overall_quality.value}"
        )


class TwinStateStore:
    """Holds the current state and applies measurements to it.

    Deliberately not a database. The brief's §3 says to avoid infrastructure
    complexity in version 1, and every behaviour worth testing — ordering,
    staleness, quality — is a property of the update logic, not of where the
    bytes are kept.
    """

    def __init__(
        self,
        tracked: dict[str, str] | None = None,
        staleness_limits: dict[str, int] | None = None,
        default_staleness_minutes: int = 120,
    ) -> None:
        self.tracked = dict(tracked or TRACKED_FIELDS)
        self._by_sensor = {sensor: name for name, sensor in self.tracked.items()}
        self.staleness_limits = dict(staleness_limits or {})
        self.default_staleness_minutes = default_staleness_minutes

        self.state = TwinState(fields={name: FieldState() for name in self.tracked})
        self.rejected_out_of_order = 0
        self.rejected_unknown_sensor = 0
        self.flagged_suspect = 0

    # --- updating ----------------------------------------------------------

    def apply(self, measurement: Measurement) -> bool:
        """Apply one measurement. Returns False if it was rejected.

        Rejection is counted rather than raised: a telemetry feed with a few
        late or unknown packets is normal, and a store that crashes on the first
        one is not usable. The counters make the rate visible.
        """
        name = self._by_sensor.get(measurement.sensor_id)
        if name is None:
            self.rejected_unknown_sensor += 1
            return False

        current = self.state.get(name)
        if current.measured_at is not None and measurement.timestamp < current.measured_at:
            # A late packet from the past. Applying it would rewind the twin.
            self.rejected_out_of_order += 1
            return False

        quality = measurement.quality
        low, high = PLAUSIBLE_RANGE.get(name, (float("-inf"), float("inf")))
        if quality is Quality.GOOD and not (low <= measurement.value <= high):
            quality = Quality.SUSPECT
            self.flagged_suspect += 1

        fields = dict(self.state.fields)
        fields[name] = FieldState(
            value=measurement.value,
            measured_at=measurement.timestamp,
            quality=quality,
        )
        timestamp = self.state.timestamp
        if timestamp is None or measurement.timestamp > timestamp:
            timestamp = measurement.timestamp
        self.state = TwinState(timestamp=timestamp, fields=fields)
        return True

    def apply_many(self, measurements: list[Measurement]) -> int:
        return sum(1 for m in measurements if self.apply(m))

    def advance_to(self, now: pd.Timestamp) -> TwinState:
        """Move the clock forward and age every field.

        This is what turns "no new data" into an explicit statement about
        confidence. Without it, a twin whose feed died last Tuesday looks
        exactly like one receiving live data.
        """
        fields = {}
        for name, state in self.state.fields.items():
            if state.value is None:
                fields[name] = state
                continue
            age = state.age_minutes(now) or 0.0
            limit = self.staleness_limits.get(name, self.default_staleness_minutes)
            if state.quality is Quality.SUSPECT:
                quality = Quality.SUSPECT
            elif age > limit:
                quality = Quality.MISSING
            elif age > 0:
                quality = Quality.STALE
            else:
                quality = Quality.GOOD
            fields[name] = replace(state, quality=quality)

        self.state = TwinState(timestamp=now, fields=fields)
        return self.state

    # --- reporting ---------------------------------------------------------

    def health(self) -> dict[str, int]:
        counts = dict.fromkeys((q.value for q in Quality), 0)
        for state in self.state.fields.values():
            counts[state.quality.value] += 1
        counts["rejected_out_of_order"] = self.rejected_out_of_order
        counts["rejected_unknown_sensor"] = self.rejected_unknown_sensor
        counts["flagged_suspect"] = self.flagged_suspect
        return counts

    def health_frame(self) -> pd.DataFrame:
        now = self.state.timestamp
        rows = []
        for name, state in self.state.fields.items():
            rows.append(
                {
                    "field": name,
                    "sensor": self.tracked[name],
                    "value": state.value,
                    "quality": state.quality.value,
                    "age_minutes": (
                        round(state.age_minutes(now), 1)
                        if now is not None and state.measured_at is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows).set_index("field")
