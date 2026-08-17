"""Forecast accuracy, and selecting a model per series.

The brief (§10) asks for MAE, RMSE, sMAPE where useful, error by hour and error
during peak periods. "Where useful" is doing work: sMAPE is reported for base
load, which stays well away from zero, and suppressed for irradiance, which is
exactly zero every night and would make any percentage metric meaningless.

    python -m bet.forecasting.evaluate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bet.data import load_drivers, split
from bet.forecasting.features import FORECAST_TARGETS, FeatureSpec, issue_times
from bet.forecasting.models import candidates

LOCAL_TZ = "Europe/Berlin"
PEAK_HOURS = (17, 18, 19, 20)

# Series that are zero for long stretches, where percentage errors are
# undefined and where an all-hours average mostly measures predicting darkness.
ZERO_HEAVY = ("irradiance_w_m2", "pv_kw")


def _clean(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    return actual[keep], predicted[keep]


def mae(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def bias(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(p - a)) if a.size else float("nan")


def smape(actual, predicted, *, floor: float = 1e-9) -> float:
    a, p = _clean(actual, predicted)
    if not a.size:
        return float("nan")
    denominator = (np.abs(a) + np.abs(p)) / 2
    if np.any(denominator < floor):
        return float("nan")
    return float(np.mean(np.abs(a - p) / denominator) * 100)


def r2(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    if a.size < 2:
        return float("nan")
    total = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - p) ** 2) / total) if total > 0 else float("nan")


def metrics(actual, predicted, *, target: str | None = None) -> dict[str, float]:
    result = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "bias": bias(actual, predicted),
        "R2": r2(actual, predicted),
        "sMAPE_%": smape(actual, predicted),
        "n": int(np.isfinite(np.asarray(actual, dtype=float)).sum()),
    }
    if target in ZERO_HEAVY:
        # Not a computation failure: a statement that the question is ill-posed
        # for a series that is exactly zero half the time.
        result["sMAPE_%"] = float("nan")
    return result


def backtest(forecaster, frame: pd.DataFrame, issues, horizon: int = 24) -> pd.DataFrame:
    """Run one forecaster over many issue times. Tidy rows out."""
    records, skipped = [], 0
    for issue in issues:
        try:
            predicted = forecaster.predict(frame, issue, horizon)
        except ValueError:
            skipped += 1
            continue
        actual = frame[forecaster.target].reindex(predicted.index)
        records.append(
            pd.DataFrame(
                {
                    "issue_time": issue,
                    "target_time": predicted.index,
                    "horizon_step": np.arange(horizon),
                    "actual": actual.to_numpy(),
                    "predicted": predicted.to_numpy(),
                }
            )
        )
    if not records:
        raise ValueError(
            f"{forecaster.name} produced no forecasts across {len(issues)} issue "
            "times -- every one lacked usable history"
        )
    out = pd.concat(records, ignore_index=True)
    out.attrs.update(
        {"model": forecaster.name, "target": forecaster.target, "skipped": skipped}
    )
    return out


def error_by_hour(results: pd.DataFrame) -> pd.DataFrame:
    working = results.dropna(subset=["actual", "predicted"]).copy()
    stamps = pd.DatetimeIndex(working["target_time"])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    working["hour"] = stamps.tz_convert(LOCAL_TZ).hour
    grouped = working.groupby("hour")
    return pd.DataFrame(
        {
            "MAE": grouped.apply(
                lambda g: mae(g["actual"], g["predicted"]), include_groups=False
            ),
            "bias": grouped.apply(
                lambda g: bias(g["actual"], g["predicted"]), include_groups=False
            ),
            "n": grouped.size(),
        }
    )


def error_in_peak_hours(results: pd.DataFrame, peak_hours=PEAK_HOURS) -> dict[str, float]:
    working = results.dropna(subset=["actual", "predicted"]).copy()
    stamps = pd.DatetimeIndex(working["target_time"])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    working["hour"] = stamps.tz_convert(LOCAL_TZ).hour
    peak = working[working["hour"].isin(peak_hours)]
    if peak.empty:
        return {"MAE": float("nan"), "n": 0}
    return {"MAE": mae(peak["actual"], peak["predicted"]), "n": int(len(peak))}


def error_by_horizon_step(results: pd.DataFrame) -> pd.Series:
    working = results.dropna(subset=["actual", "predicted"])
    return working.groupby("horizon_step").apply(
        lambda g: mae(g["actual"], g["predicted"]), include_groups=False
    )


def skill_score(results: pd.DataFrame, baseline: pd.DataFrame) -> float:
    """Fractional MAE improvement over a baseline. Negative means worse."""
    base = mae(baseline["actual"], baseline["predicted"])
    if not np.isfinite(base) or base == 0:
        return float("nan")
    return float(1 - mae(results["actual"], results["predicted"]) / base)


def compare(
    target: str,
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_issues,
    test_issues,
    horizon: int = 24,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Fit every candidate on train, score on test, tabulate."""
    spec = spec or FeatureSpec()
    rows, raw = [], {}
    for forecaster in candidates(target, spec):
        forecaster.fit(train_frame, train_issues, horizon)
        results = backtest(forecaster, frame, test_issues, horizon)
        raw[forecaster.name] = results
        summary = metrics(results["actual"], results["predicted"], target=target)
        summary["model"] = forecaster.name
        summary["MAE_peak"] = error_in_peak_hours(results)["MAE"]
        rows.append(summary)

    table = pd.DataFrame(rows).set_index("model")
    table["skill_vs_persistence"] = [
        skill_score(raw[name], raw["persistence"]) for name in table.index
    ]
    table["selected"] = table.index == table["MAE"].idxmin()
    return table[
        ["MAE", "RMSE", "MAE_peak", "bias", "R2", "sMAPE_%", "skill_vs_persistence",
         "n", "selected"]
    ]


def observed_history(frame: pd.DataFrame | None = None, spec=None) -> pd.DataFrame:
    """Drivers plus what the building's meters actually recorded.

    The twin forecasts **observed** history, not the schedule it was generated
    from. That distinction is not pedantic: recomputing base load from the
    calendar hands the forecaster the exact generating process, and a
    gradient-boosted model with hour-of-day features then predicts it with
    R^2 = 1.0000 and MAE = 0.0000. An earlier version did precisely that, and
    the result looked indistinguishable from a leak.

    Running the plant and reading its meter is what a real twin has access to,
    and it carries the day-to-day load variability that makes the forecasting
    problem real.
    """
    from bet.plant.building import REFERENCE_BUILDING
    from bet.plant.controllers import ThermostatController, run_controller
    from bet.plant.simulator import Plant

    spec = spec or REFERENCE_BUILDING
    frame = load_drivers() if frame is None else frame
    history = run_controller(ThermostatController(spec), Plant(spec, frame)).frame

    out = frame.copy()
    out["base_load_kw"] = history["base_load_kw"]
    out["occupancy_fraction"] = history["occupancy_fraction"]
    out["indoor_temp_c"] = history["indoor_temp_c"]
    return out


def run_selection(horizon: int = 24, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Compare every candidate on every forecast target."""
    spec = spec or FeatureSpec()
    frame = observed_history()
    train = split(frame, "train")
    validation = split(frame, "validation")

    train_issues = issue_times(train, horizon, spec, step_hours=24)
    validation_issues = [
        issue
        for issue in issue_times(frame, horizon, spec, step_hours=24)
        if validation.index[0] <= issue <= validation.index[-1] - pd.Timedelta(hours=horizon)
    ]

    blocks = []
    for target in FORECAST_TARGETS:
        table = compare(
            target, frame, train, train_issues, validation_issues, horizon, spec
        )
        table["target"] = target
        blocks.append(table.reset_index())
    return pd.concat(blocks, ignore_index=True).set_index(["target", "model"])


def main() -> int:
    results = run_selection()
    out = Path(__file__).resolve().parents[3] / "results"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "forecast_selection.csv")

    for target in FORECAST_TARGETS:
        block = results.loc[target]
        print(f"\n=== {target} ===")
        print(
            block[["MAE", "RMSE", "R2", "skill_vs_persistence"]].round(4).to_string()
        )
        print(f"  lowest MAE: {block['MAE'].idxmin()}")
    print(f"\nwrote {out / 'forecast_selection.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
