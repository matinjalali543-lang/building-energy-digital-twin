"""Injectable, labelled faults — the ground truth anomaly detection is scored on.

Because the plant is simulated, a fault can be introduced at a known time with a
known magnitude. That turns anomaly detection from a gallery of flagged points
into something measurable: precision, recall, detection latency, and the
false-alarm rate on clean data.

## Two kinds of fault, and the difference is the point

**Physical faults** change what the building actually does. A decayed heat pump
burns more electricity for the same heat; a soiled array generates less. They
cost money whether or not anyone notices.

**Sensor faults** change only what is *measured*. A stuck thermometer or a
drifting meter leaves the building's behaviour untouched and corrupts the twin's
picture of it.

The distinction matters because the two demand opposite responses. A physical
fault should trigger maintenance on the asset. A sensor fault should trigger
maintenance on the *instrument* — and acting on it as though it were physical
means "fixing" a healthy chiller while the twin keeps lying. A detector that
reports "consumption is abnormal" without separating them is only half a
diagnosis, so `Fault.kind` is carried through into every anomaly record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from bet.plant.building import BuildingSpec
from bet.plant.simulator import ControlAction, Plant, PlantReading


@dataclass(frozen=True)
class Fault:
    """A labelled fault over a known window.

    `severity_fraction` is interpreted per subclass but always means "how much",
    on a 0-1 scale, so that detection difficulty can be swept coherently.
    """

    name: str
    asset: str
    kind: str  # "physical" or "sensor"
    start: pd.Timestamp
    end: pd.Timestamp
    severity_fraction: float = 0.2
    ramp: bool = False  # gradual onset rather than a step
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("physical", "sensor"):
            raise ValueError(f"kind must be 'physical' or 'sensor', got {self.kind!r}")
        if not 0 < self.severity_fraction <= 1:
            raise ValueError(
                f"severity_fraction must be in (0, 1], got {self.severity_fraction}"
            )
        if pd.Timestamp(self.end) <= pd.Timestamp(self.start):
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")

    def active(self, timestamp: pd.Timestamp) -> bool:
        return pd.Timestamp(self.start) <= timestamp < pd.Timestamp(self.end)

    def intensity(self, timestamp: pd.Timestamp) -> float:
        """How far the fault has developed at `timestamp`, on 0-1.

        A ramped fault is much harder to catch than a step of the same final
        magnitude, because early on it hides inside normal residual scatter.
        Reporting detection latency against a ramp is the honest test.
        """
        if not self.active(timestamp):
            return 0.0
        if not self.ramp:
            return self.severity_fraction
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        progress = (timestamp - start) / (end - start)
        return self.severity_fraction * float(np.clip(progress, 0.0, 1.0))

    def label_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean ground truth over an index."""
        return np.array([self.active(t) for t in index], dtype=bool)


def hvac_efficiency_decay(
    start: str, end: str, severity: float = 0.25, ramp: bool = True
) -> Fault:
    """Heat pump loses `severity` of its COP — more electricity for the same heat."""
    return Fault(
        name="hvac_efficiency_decay",
        asset="hvac",
        kind="physical",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        severity_fraction=severity,
        ramp=ramp,
        description=(
            "Refrigerant loss or fouled coils: the unit still meets the setpoint, "
            "so comfort looks fine and only the electricity bill moves."
        ),
    )


def pv_soiling(start: str, end: str, severity: float = 0.30, ramp: bool = True) -> Fault:
    """Array output falls by `severity` — dust, snow, or a failed string."""
    return Fault(
        name="pv_soiling",
        asset="pv",
        kind="physical",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        severity_fraction=severity,
        ramp=ramp,
        description=(
            "Soiling, shading or a dead string. Only detectable against expected "
            "output for the measured irradiance -- raw generation falls in autumn "
            "anyway."
        ),
    )


def stuck_temperature_sensor(start: str, end: str) -> Fault:
    """The indoor thermometer freezes at its last value.

    Severity is 1.0 because a stuck sensor is not partially stuck.
    """
    return Fault(
        name="stuck_temperature_sensor",
        asset="zone_temperature_sensor",
        kind="sensor",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        severity_fraction=1.0,
        description=(
            "The reading stops changing. The building is fine; the twin is blind, "
            "and a controller trusting it will drive the zone out of the band."
        ),
    )


def meter_drift(start: str, end: str, severity: float = 0.15, ramp: bool = True) -> Fault:
    """The electricity meter reads high by up to `severity`."""
    return Fault(
        name="meter_drift",
        asset="main_meter",
        kind="sensor",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        severity_fraction=severity,
        ramp=ramp,
        description=(
            "Calibration drift. Consumption appears to rise with nothing physical "
            "behind it -- the classic false positive for a consumption-only detector."
        ),
    )


def hvac_capacity_loss(start: str, end: str, severity: float = 0.5) -> Fault:
    """A stage or compressor fails: `severity` of capacity is unavailable."""
    return Fault(
        name="hvac_capacity_loss",
        asset="hvac",
        kind="physical",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        severity_fraction=severity,
        description=(
            "A failed stage. Unlike efficiency decay this one shows up as comfort "
            "loss on demanding days and as nothing at all on mild ones."
        ),
    )


@dataclass
class FaultSchedule:
    """A set of faults and the ground-truth labels they imply."""

    faults: list[Fault] = field(default_factory=list)

    def active_at(self, timestamp: pd.Timestamp) -> list[Fault]:
        return [fault for fault in self.faults if fault.active(timestamp)]

    def labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """One boolean column per fault, plus `any_fault`."""
        frame = pd.DataFrame(
            {fault.name: fault.label_index(index) for fault in self.faults},
            index=index,
        )
        frame["any_fault"] = (
            frame.any(axis=1) if len(self.faults) else np.zeros(len(index), dtype=bool)
        )
        return frame

    def labels_for_asset(self, index: pd.DatetimeIndex, asset: str) -> np.ndarray:
        relevant = [f for f in self.faults if f.asset == asset]
        if not relevant:
            return np.zeros(len(index), dtype=bool)
        return np.logical_or.reduce([f.label_index(index) for f in relevant])

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fault": f.name,
                    "asset": f.asset,
                    "kind": f.kind,
                    "start": f.start,
                    "end": f.end,
                    "hours": int(
                        (pd.Timestamp(f.end) - pd.Timestamp(f.start)).total_seconds() // 3600
                    ),
                    "severity": f.severity_fraction,
                    "onset": "ramp" if f.ramp else "step",
                }
                for f in self.faults
            ]
        )


def default_schedule(year: int = 2019) -> FaultSchedule:
    """Five faults spread across the test window, deliberately varied.

    Two physical and two sensor faults plus a capacity failure, mixing step and
    ramp onsets and mixing "costs money quietly" with "breaks comfort loudly".
    A detector that catches only the loud ones is not much use, and this schedule
    makes that visible.

    Windows total roughly **29% of the test period**, and that share is chosen
    deliberately. An earlier draft faulted 68% of hours, which quietly broke the
    evaluation: a detector that flagged every hour would have scored 68%
    precision and looked respectable. Keeping faults a minority of the record is
    what makes precision mean something.
    """
    return FaultSchedule(
        [
            pv_soiling(f"{year}-10-21", f"{year}-10-28", severity=0.30, ramp=True),
            hvac_efficiency_decay(f"{year}-11-08", f"{year}-11-15", severity=0.25, ramp=True),
            stuck_temperature_sensor(f"{year}-11-28", f"{year}-12-01"),
            meter_drift(f"{year}-12-06", f"{year}-12-10", severity=0.15, ramp=True),
            hvac_capacity_loss(f"{year}-12-19", f"{year}-12-21", severity=0.5),
        ]
    )


class FaultyPlant(Plant):
    """A plant with faults applied.

    Physical faults are applied inside `step`, so they change what the building
    really does. Sensor faults are applied in `observe`, so the plant's true
    state is untouched and only the twin's view is corrupted — which is exactly
    the difference the detector has to work out.
    """

    def __init__(
        self,
        spec: BuildingSpec,
        drivers: pd.DataFrame,
        schedule: FaultSchedule | None = None,
        **kwargs,
    ) -> None:
        super().__init__(spec, drivers, **kwargs)
        self.schedule = schedule or FaultSchedule()
        self._last_observed_temp: float | None = None
        self._held_measurements: dict[str, float] = {}

    def step(self, action: ControlAction) -> PlantReading:
        timestamp = self.state.timestamp
        physical = [
            f for f in self.schedule.active_at(timestamp) if f.kind == "physical"
        ]

        original_spec = self.spec
        original_scale = self._pv_scale
        try:
            for fault in physical:
                level = fault.intensity(timestamp)
                if fault.name == "hvac_efficiency_decay":
                    self.spec = replace(
                        self.spec,
                        hvac=replace(
                            self.spec.hvac,
                            cop_heating=self.spec.hvac.cop_heating * (1 - level),
                            cop_cooling=self.spec.hvac.cop_cooling * (1 - level),
                        ),
                    )
                elif fault.name == "hvac_capacity_loss":
                    self.spec = replace(
                        self.spec,
                        hvac=replace(
                            self.spec.hvac,
                            heating_capacity_kw_thermal=(
                                self.spec.hvac.heating_capacity_kw_thermal * (1 - level)
                            ),
                            cooling_capacity_kw_thermal=(
                                self.spec.hvac.cooling_capacity_kw_thermal * (1 - level)
                            ),
                        ),
                    )
                elif fault.name == "pv_soiling":
                    self._pv_scale *= 1 - level
            return super().step(action)
        finally:
            self.spec = original_spec
            self._pv_scale = original_scale

    def corrupt_measurements(self, batch: list) -> list:
        """Apply sensor faults to the measurements a provider has already made.

        Order matters and this is the correct one. A stuck sensor freezes the
        value it last *reported* -- noise and all -- so it must be applied after
        the instrument noise, not before. Applying it first produced a stuck
        value with fresh noise added each hour, which is statistically identical
        to a healthy sensor and which no flatline detector could ever catch.
        """
        from dataclasses import replace as _replace

        if not batch:
            return batch
        timestamp = batch[0].timestamp
        sensor_faults = [
            f for f in self.schedule.active_at(timestamp) if f.kind == "sensor"
        ]
        if not sensor_faults:
            self._held_measurements = {m.sensor_id: m.value for m in batch}
            return batch

        out = []
        for measurement in batch:
            value = measurement.value
            for fault in sensor_faults:
                level = fault.intensity(timestamp)
                if (
                    fault.name == "stuck_temperature_sensor"
                    and measurement.sensor_id == "sensor_zone_temp"
                ):
                    value = self._held_measurements.get("sensor_zone_temp", value)
                elif fault.name == "meter_drift" and measurement.sensor_id in (
                    "sensor_grid_power",
                    "sensor_hvac_power",
                ):
                    value = value * (1 + level)
            out.append(_replace(measurement, value=value))
        return out

    def observe(self, reading: PlantReading) -> PlantReading:
        """Degrade a true reading into what the instruments report."""
        sensor_faults = [
            f for f in self.schedule.active_at(reading.timestamp) if f.kind == "sensor"
        ]
        observed = reading
        for fault in sensor_faults:
            level = fault.intensity(reading.timestamp)
            if fault.name == "stuck_temperature_sensor":
                held = (
                    self._last_observed_temp
                    if self._last_observed_temp is not None
                    else reading.indoor_temp_c
                )
                observed = replace(observed, indoor_temp_c=held)
            elif fault.name == "meter_drift":
                factor = 1 + level
                observed = replace(
                    observed,
                    grid_import_kw=observed.grid_import_kw * factor,
                    base_load_kw=observed.base_load_kw * factor,
                )
        if not any(f.name == "stuck_temperature_sensor" for f in sensor_faults):
            self._last_observed_temp = observed.indoor_temp_c
        return observed
