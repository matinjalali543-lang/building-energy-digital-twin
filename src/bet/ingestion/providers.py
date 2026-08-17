"""Telemetry sources — the brief's §7 ingestion layer.

`DataProvider` is the seam where a real building would be plugged in. Two
implementations ship: one replaying a CSV, one generating a stream from the
plant as it runs. An MQTT or HTTP provider would be a third, and the point of
having two already is that the seam is demonstrated rather than asserted — a
protocol with a single implementation is a guess about what abstraction is
needed.

Every emitted point carries timestamp, value, unit and quality, per §7.

## Sensors are imperfect on purpose

`PlantStreamProvider` adds Gaussian noise sized to each sensor's stated accuracy
and drops packets at a configurable rate. Without that, the twin would receive
the plant's exact internal state and every residual would be zero, which would
make the anomaly detection phase meaningless — you cannot detect a deviation
from a model that is exact by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from bet.twin.entities import Building
from bet.twin.state import Measurement, Quality

# Which plant quantity feeds which sensor. The twin's TRACKED_FIELDS maps
# sensors to state fields; this maps sensors back to what the plant produces.
SENSOR_SOURCES: dict[str, str] = {
    "sensor_zone_temp": "indoor_temp_c",
    "sensor_outdoor_temp": "outdoor_temp_c",
    "sensor_irradiance": "irradiance_w_m2",
    "sensor_pv_power": "pv_kw",
    "sensor_battery_soc": "battery_soc_kwh",
    "sensor_grid_power": "grid_import_kw",
    "sensor_hvac_power": "hvac_electric_kw",
}

SENSOR_UNITS: dict[str, str] = {
    "sensor_zone_temp": "degC",
    "sensor_outdoor_temp": "degC",
    "sensor_irradiance": "W/m2",
    "sensor_pv_power": "kW",
    "sensor_battery_soc": "kWh",
    "sensor_grid_power": "kW",
    "sensor_hvac_power": "kW",
}

# Instrument accuracies, used as the noise scale. Kept here rather than read
# from the Building so a provider can be used without a full model.
DEFAULT_ACCURACY: dict[str, float] = {
    "sensor_zone_temp": 0.3,
    "sensor_outdoor_temp": 0.5,
    "sensor_irradiance": 15.0,
    "sensor_pv_power": 2.0,
    "sensor_battery_soc": 2.5,
    "sensor_grid_power": 3.0,
    "sensor_hvac_power": 2.0,
}


class DataProvider(Protocol):
    """Anything that can emit measurements in time order."""

    name: str

    def measurements(self) -> Iterator[list[Measurement]]:
        """Yield one batch of measurements per interval."""
        ...


@dataclass
class CsvReplayProvider:
    """Replay a recorded telemetry file.

    The provider a real deployment would start with: historical data on disk,
    read back in order. Columns are matched to sensors by `SENSOR_SOURCES`.
    """

    frame: pd.DataFrame
    name: str = "csv-replay"
    sensors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.frame.empty:
            raise ValueError("cannot replay an empty frame")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError(
                "telemetry must be replayed in time order; the frame's index is "
                "not sorted, and a store that receives it will reject most rows "
                "as out of order"
            )
        available = [
            sensor
            for sensor, column in SENSOR_SOURCES.items()
            if column in self.frame.columns
        ]
        if self.sensors is None:
            self.sensors = tuple(available)
        else:
            unknown = set(self.sensors) - set(available)
            if unknown:
                raise ValueError(
                    f"requested sensor(s) {sorted(unknown)} have no matching column "
                    f"in the frame; available: {available}"
                )

    def measurements(self) -> Iterator[list[Measurement]]:
        for timestamp, row in self.frame.iterrows():
            batch = []
            for sensor in self.sensors:
                value = row[SENSOR_SOURCES[sensor]]
                if pd.isna(value):
                    continue  # a gap is an absence, not a zero
                batch.append(
                    Measurement(
                        sensor_id=sensor,
                        timestamp=timestamp,
                        value=float(value),
                        unit=SENSOR_UNITS[sensor],
                        quality=Quality.GOOD,
                    )
                )
            yield batch


@dataclass
class PlantStreamProvider:
    """Emit measurements from a running plant, with noise and dropouts.

    Not a replay: this reads the plant as it steps, which is what the MPC loop
    needs. It is also the only place sensor imperfection is introduced, so the
    twin's view is degraded in exactly one auditable spot.
    """

    name: str = "plant-stream"
    noise_scale: float = 1.0
    dropout_rate: float = 0.01
    rng: np.random.Generator | None = None
    accuracy: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {self.noise_scale}")
        self.rng = self.rng or np.random.default_rng(20260817)
        self.accuracy = dict(self.accuracy or DEFAULT_ACCURACY)

    def observe(self, reading, sensors: tuple[str, ...] | None = None) -> list[Measurement]:
        """Turn one plant reading into a batch of noisy, occasionally missing points."""
        wanted = sensors or tuple(SENSOR_SOURCES)
        batch = []
        for sensor in wanted:
            attribute = SENSOR_SOURCES[sensor]
            if not hasattr(reading, attribute):
                continue
            if self.rng.random() < self.dropout_rate:
                continue  # packet lost; the store will age the field to STALE
            sigma = self.accuracy.get(sensor, 0.0) * self.noise_scale
            value = float(getattr(reading, attribute))
            if sigma > 0:
                value += float(self.rng.normal(0.0, sigma))
            batch.append(
                Measurement(
                    sensor_id=sensor,
                    timestamp=reading.timestamp,
                    value=value,
                    unit=SENSOR_UNITS[sensor],
                    quality=Quality.GOOD,
                )
            )
        return batch


def sensors_for(building: Building) -> tuple[str, ...]:
    """Sensor ids the building declares that this package knows how to feed."""
    declared = {sensor.entity_id for sensor in building.sensors}
    declared |= {meter.sensor_id for meter in building.meters if meter.sensor_id}
    return tuple(sensor for sensor in SENSOR_SOURCES if sensor in declared)


def record_telemetry(
    plant,
    controller,
    provider: PlantStreamProvider | None = None,
    sensors: tuple[str, ...] | None = None,
    store=None,
) -> pd.DataFrame:
    """Run a plant and return **what the twin observed**, not what happened.

    The distinction is the whole point of the plant/twin wall. Reading
    `plant.history_frame()` gives the building's exact internal state; a real
    twin never has that. This returns the noisy, occasionally-missing view the
    instruments produced, which is what residuals and anomaly detection must be
    computed from.

    Control actions are included because the twin genuinely knows them — it
    issued them — and the thermal model cannot be identified without them.
    """
    from bet.twin.state import TwinStateStore

    provider = provider or PlantStreamProvider()
    store = store or TwinStateStore()
    sensors = sensors or tuple(SENSOR_SOURCES)

    rows = []
    while not plant.finished:
        reading = plant.step(controller.act(plant))
        observed = plant.observe(reading) if hasattr(plant, "observe") else reading
        batch = provider.observe(observed, sensors)
        if hasattr(plant, "corrupt_measurements"):
            batch = plant.corrupt_measurements(batch)
        store.apply_many(batch)
        state = store.advance_to(observed.timestamp)

        row = {"timestamp": observed.timestamp}
        for field_name, field_state in state.fields.items():
            row[field_name] = field_state.value if field_state.quality.usable else np.nan
        # Commanded, not measured: the controller issued it, so the twin has it
        # exactly. Occupancy likewise comes from the schedule, not a sensor.
        row["hvac_thermal_kw"] = reading.hvac_thermal_kw
        row["occupancy_fraction"] = reading.occupancy_fraction
        row["price_eur_mwh"] = reading.price_eur_mwh
        rows.append(row)

    return pd.DataFrame(rows).set_index("timestamp")
