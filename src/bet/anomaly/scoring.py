"""Scoring detection against known faults.

Because the plant is simulated, every fault has a known start, end and asset. So
detection can be *measured* rather than demonstrated:

* **precision** — of the hours flagged, how many were really faulted;
* **recall** — of the faulted hours, how many were flagged;
* **detection latency** — hours between a fault starting and the first flag;
* **false-alarm rate on clean data** — flags per week when nothing is wrong.

The last one matters most in practice and is the one usually left out. A
detector with excellent recall that also cries wolf twice a day gets switched
off within a month, and no confusion matrix computed only over the fault window
will ever reveal that.

Attribution is scored too, not just detection. A flag on the PV asset during an
HVAC fault is not a hit, because it would send someone to the wrong roof.

## One confusion the numbers deliberately expose

`meter_drift` is injected on `main_meter` and no residual series is mapped to
it, so flags raised during that window count as **false positives on the HVAC
asset**. That is not a scoring bug, it is the finding: a drifting meter and a
decayed heat pump produce the *same* signature in the HVAC residual -- metered
electricity higher than the delivered heat implies -- and nothing in that one
signal can separate them.

Telling them apart needs a second, independent measurement, such as reconciling
the HVAC sub-meter against the main incomer. That is exactly why `Fault.kind`
distinguishes physical from sensor faults, and exactly why a detector that
reports only "consumption is abnormal" leaves the operator to guess whether to
overhaul a chiller or recalibrate an instrument.

    python -m bet.anomaly.scoring
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bet.anomaly.detectors import AnomalyDetector, anomalies_to_frame, default_detectors
from bet.data import load_drivers, split
from bet.ingestion.providers import record_telemetry
from bet.plant.building import REFERENCE_BUILDING, BuildingSpec
from bet.plant.controllers import ThermostatController
from bet.plant.faults import FaultSchedule, FaultyPlant, default_schedule
from bet.plant.simulator import Plant
from bet.twin.residuals import all_residuals, identify_thermal_model

# Which residual series speaks for which asset. A detector firing on the wrong
# series is not a hit even if the timing is right.
SERIES_TO_ASSET = {
    "indoor_temp": "zone_temperature_sensor",
    "pv_output": "pv",
    "hvac_electric": "hvac",
}


def confusion(flagged: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 for boolean hour-level arrays."""
    true_positive = int(np.sum(flagged & truth))
    false_positive = int(np.sum(flagged & ~truth))
    false_negative = int(np.sum(~flagged & truth))

    precision = true_positive / (true_positive + false_positive) if flagged.any() else float("nan")
    recall = true_positive / (true_positive + false_negative) if truth.any() else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def detection_latency_hours(
    flagged: pd.Series, schedule: FaultSchedule, asset: str
) -> dict[str, float]:
    """Hours from each fault's start to the first flag on the right asset.

    NaN means the fault was never detected, which is deliberately distinct from
    a large latency — averaging a miss in as "very late" would flatter a
    detector that simply never noticed.
    """
    out = {}
    for fault in schedule.faults:
        if fault.asset != asset:
            continue
        window = flagged.loc[
            (flagged.index >= fault.start) & (flagged.index < fault.end)
        ]
        hits = window[window]
        out[fault.name] = (
            float((hits.index[0] - pd.Timestamp(fault.start)).total_seconds() / 3600)
            if len(hits)
            else float("nan")
        )
    return out


def false_alarm_rate(flagged: pd.Series, truth: np.ndarray) -> float:
    """Flags per week over hours where nothing was actually wrong."""
    clean = ~truth
    clean_hours = int(clean.sum())
    if clean_hours == 0:
        return float("nan")
    return float(np.sum(flagged.to_numpy() & clean) / clean_hours * 168)


def score_detector(
    detector: AnomalyDetector,
    residual_series,
    clean_residuals: pd.Series,
    schedule: FaultSchedule,
) -> dict:
    """Calibrate on clean residuals, detect on faulted ones, score."""
    detector.fit(clean_residuals)
    anomalies = detector.detect(residual_series)

    index = residual_series.residual.index
    flagged = pd.Series(False, index=index)
    for anomaly in anomalies:
        if anomaly.timestamp in flagged.index:
            flagged.loc[anomaly.timestamp] = True

    asset = SERIES_TO_ASSET.get(residual_series.name, residual_series.asset)
    truth = schedule.labels_for_asset(index, asset)

    result = confusion(flagged.to_numpy(), truth)
    result.update(
        {
            "detector": detector.name,
            "series": residual_series.name,
            "asset": asset,
            "flags": int(flagged.sum()),
            "false_alarms_per_week": false_alarm_rate(flagged, truth),
            "faulted_hours": int(truth.sum()),
        }
    )
    latencies = detection_latency_hours(flagged, schedule, asset)
    result["latency_hours"] = (
        float(np.nanmean(list(latencies.values()))) if latencies else float("nan")
    )
    result["latencies"] = latencies
    return result


def run_evaluation(
    spec: BuildingSpec = REFERENCE_BUILDING,
    schedule: FaultSchedule | None = None,
    threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full evaluation: calibrate on a clean window, score on a faulted one."""
    schedule = schedule or default_schedule()
    drivers = load_drivers()

    # Clean history for calibration -- the twin's own model and the detectors'
    # thresholds both come from a period with no faults in it.
    clean_history = record_telemetry(
        Plant(spec, split(drivers, "train")), ThermostatController(spec)
    )
    usable = clean_history.dropna(
        subset=["indoor_temp_c", "outdoor_temp_c", "irradiance_w_m2"]
    )
    model = identify_thermal_model(usable)
    clean_residuals = all_residuals(
        usable, model, spec.pv_capacity_kwp, spec.hvac.cop_heating, spec.hvac.cop_cooling
    )

    # Faulted history for scoring.
    faulted_history = record_telemetry(
        FaultyPlant(spec, split(drivers, "test"), schedule), ThermostatController(spec)
    )
    faulted_residuals = all_residuals(
        faulted_history,
        model,
        spec.pv_capacity_kwp,
        spec.hvac.cop_heating,
        spec.hvac.cop_cooling,
    )

    # A flatline check only means something on a signal that should always be
    # moving. Applied to PV it fires every night -- 575 flags and 51 false
    # alarms a week when it was run across all three series -- because a dark
    # array is legitimately flat. Restricting it to zone temperature is not
    # tuning to the answer; it is the check simply not being defined elsewhere.
    detector_series = {
        "flatline": ("indoor_temp",),
    }

    rows, all_anomalies = [], []
    for detector in default_detectors(threshold):
        allowed = detector_series.get(detector.name)
        for name, series in faulted_residuals.items():
            if allowed is not None and name not in allowed:
                continue
            fresh = type(detector)(threshold=detector.threshold, name=detector.name)
            if hasattr(detector, "window_hours"):
                fresh.window_hours = detector.window_hours
            if hasattr(detector, "drift"):
                fresh.drift = detector.drift
            result = score_detector(
                fresh, series, clean_residuals[name].residual, schedule
            )
            rows.append(result)
            all_anomalies.extend(fresh.detect(series))

    table = pd.DataFrame(rows)
    table = table[
        [
            "detector", "series", "asset", "precision", "recall", "f1",
            "latency_hours", "false_alarms_per_week", "flags", "faulted_hours",
            "true_positive", "false_positive", "false_negative",
        ]
    ].set_index(["detector", "series"])
    return table, anomalies_to_frame(all_anomalies)


def main() -> int:
    table, anomalies = run_evaluation()
    out = Path(__file__).resolve().parents[3] / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "anomaly_scores.csv")
    anomalies.to_csv(out / "anomalies.csv")

    columns = ["fault", "asset", "kind", "hours", "onset"]
    print(default_schedule().summary()[columns].to_string(index=False))
    print()
    print(table.round(3).to_string())
    print(f"\n{len(anomalies)} anomaly records written to {out / 'anomalies.csv'}")
    if len(anomalies):
        print("\nexample record:")
        first = anomalies.iloc[0]
        print(f"  {anomalies.index[0]}  {first['asset']}  severity={first['severity']}")
        print(f"  {first['explanation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
