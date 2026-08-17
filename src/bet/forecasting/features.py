"""Feature construction, with leakage made structurally impossible.

Same discipline as the sibling repo, re-implemented rather than imported: these
repositories share no code, and a leakage barrier copied without being
re-derived is a barrier nobody has checked.

Two design choices carry it:

**Slice before you look.** The first statement of `build_features` discards
every row at or after `issue_time`. Everything afterwards operates on `past`, so
a feature *cannot* reference the future — the rows are not in scope. That is
stronger than computing features over the whole frame and masking them, which is
correct only for as long as everyone remembers to mask.

**One function, both paths.** `build_features` serves training and inference
alike, so the two cannot drift.

## What actually needs forecasting here, and what does not

Worth stating because it differs from a pure price-arbitrage project, and
getting it wrong would mean forecasting something that is simply *published*:

**Wholesale price — known, not forecast.** The day-ahead auction clears at
midday for all 24 hours of the next day, so a controller planning tomorrow has
the real prices rather than an estimate.

**Occupancy — known.** It comes from the building's operating schedule.

**Outdoor temperature and irradiance — forecast.** A real site takes a met
service feed; here they are predicted from history.

**Building base load — forecast.** Driven by occupancy and season.

**Indoor temperature — neither.** It is *simulated forward* by the twin's
thermal model, and the gap between that prediction and the measurement is the
calibration and anomaly signal (see `bet.twin.residuals`).

So the price forecasting that dominated the sibling repo has no counterpart here.
Treating a published price as an unknown would have manufactured an error that
does not exist in operation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"

# Quantities a real deployment would genuinely have to predict.
FORECAST_TARGETS = ("outdoor_temp_c", "irradiance_w_m2", "base_load_kw")

# Quantities available without prediction, and why.
KNOWN_AHEAD = {
    "price_eur_mwh": "published by the day-ahead auction before the day begins",
    "occupancy_fraction": "comes from the building's operating schedule",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Which features to build."""

    lags_hours: tuple[int, ...] = (24, 48, 168)
    rolling_windows_hours: tuple[int, ...] = (24, 168)
    include_calendar: bool = True

    def __post_init__(self) -> None:
        if not self.lags_hours:
            raise ValueError("lags_hours must not be empty")
        if any(lag <= 0 for lag in self.lags_hours):
            raise ValueError(f"lags_hours must all be positive, got {self.lags_hours}")

    @property
    def max_lookback_hours(self) -> int:
        return max((*self.lags_hours, *self.rolling_windows_hours))

    def check_horizon(self, horizon: int) -> None:
        """Refuse a horizon a lag would reach into."""
        offenders = [lag for lag in self.lags_hours if lag < horizon]
        if offenders:
            raise ValueError(
                f"lag(s) {offenders} are shorter than the {horizon}-step horizon. "
                f"A lag of L at horizon step k reads target - L, which lies inside "
                f"the forecast window whenever k >= L. Use lags of at least "
                f"{horizon} hours, or shorten the horizon."
            )


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features of the target timestamps.

    Not leakage: the calendar is knowable arbitrarily far ahead. Hour and month
    are sine/cosine encoded so hour 23 sits next to hour 0 rather than 23 units
    away — a tree can learn the wraparound, a linear model cannot.
    """
    local = index.tz_convert(LOCAL_TZ)
    hour = local.hour.to_numpy()
    month = local.month.to_numpy()
    dayofweek = local.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "dayofweek": dayofweek.astype(float),
            "is_weekend": (dayofweek >= 5).astype(float),
            "is_working_hours": (
                (hour >= 7) & (hour < 19) & (dayofweek < 5)
            ).astype(float),
        },
        index=index,
    )


def build_features(
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Features for `horizon` periods starting at `issue_time`.

    Uses **only** rows strictly before `issue_time`. Rows whose lags are
    unavailable come back as NaN rather than imputed — a gap in the history is
    something the caller should know about, not something to paper over.
    """
    spec = spec or FeatureSpec()
    spec.check_horizon(horizon)

    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        issue_time = issue_time.tz_localize("UTC")

    # The leakage barrier. Everything below sees only `past`.
    past = frame.loc[frame.index < issue_time]
    if past.empty:
        raise ValueError(
            f"no history before issue_time {issue_time}; the frame starts at "
            f"{frame.index[0]}"
        )
    if target not in past.columns:
        raise KeyError(f"column {target!r} is not in the frame")

    step = pd.Timedelta(hours=1)
    targets = pd.DatetimeIndex(
        [issue_time + step * k for k in range(horizon)], name=frame.index.name
    )
    history = past[target]

    columns: dict[str, np.ndarray] = {}
    for lag in spec.lags_hours:
        columns[f"lag_{lag}h"] = history.reindex(targets - step * lag).to_numpy()

    for window in spec.rolling_windows_hours:
        recent = history.loc[history.index >= issue_time - step * window]
        for name, value in (
            ("mean", recent.mean()),
            ("std", recent.std()),
            ("min", recent.min()),
            ("max", recent.max()),
        ):
            columns[f"last_{window}h_{name}"] = np.full(
                horizon, float(value) if pd.notna(value) else np.nan
            )

    columns["last_observed"] = np.full(horizon, float(history.iloc[-1]))

    features = pd.DataFrame(columns, index=targets)
    features["horizon_step"] = np.arange(horizon, dtype=float)
    if spec.include_calendar:
        features = features.join(_calendar(targets))
    return features


def build_training_set(
    frame: pd.DataFrame,
    issue_times: Sequence[pd.Timestamp],
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack `build_features` over many issue times.

    Labels are read from `frame` at the target timestamps. That is supervision,
    not leakage — but it is the only place the future is touched, and it is
    confined to this function.
    """
    spec = spec or FeatureSpec()
    blocks, labels = [], []
    for issue_time in issue_times:
        features = build_features(frame, issue_time, horizon, target, spec)
        blocks.append(features)
        labels.append(frame[target].reindex(features.index))

    if not blocks:
        raise ValueError("issue_times is empty, so there is nothing to train on")

    X = pd.concat(blocks)
    y = pd.concat(labels)
    keep = X.notna().all(axis=1) & y.notna()
    return X.loc[keep], y.loc[keep]


def issue_times(
    frame: pd.DataFrame,
    horizon: int = 24,
    spec: FeatureSpec | None = None,
    step_hours: int = 24,
) -> pd.DatetimeIndex:
    """Issue times spaced `step_hours` apart with enough history behind them.

    A rolling controller re-issues hourly, so its forecasters are trained on
    `step_hours=3` or similar. Models trained only at midnight learn
    `horizon_step` as a proxy for hour-of-day, since at that issue time the two
    coincide, and serving them at 13:00 breaks the association silently.
    """
    spec = spec or FeatureSpec()
    step = pd.Timedelta(hours=1)
    earliest = frame.index[0] + step * spec.max_lookback_hours
    latest = frame.index[-1] - step * (horizon - 1)
    if earliest > latest:
        return pd.DatetimeIndex([])
    return pd.date_range(earliest, latest, freq=f"{step_hours}h")


def assert_no_future_reference(
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Prove features ignore everything at or after `issue_time`.

    Builds features once on the real frame, then again on a frame whose future
    has been overwritten with noise, and requires the two to be identical.
    Exposed here rather than living only in the test suite so a newly added
    feature can be checked ad hoc.
    """
    rng = rng or np.random.default_rng(0)
    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        issue_time = issue_time.tz_localize("UTC")

    honest = build_features(frame, issue_time, horizon, target, spec)

    corrupted = frame.copy()
    future = corrupted.index >= issue_time
    if not future.any():
        raise ValueError(
            f"nothing at or after {issue_time} to corrupt -- this check would "
            "pass vacuously"
        )
    for column in corrupted.columns:
        if pd.api.types.is_float_dtype(corrupted[column]):
            corrupted.loc[future, column] = rng.normal(1e6, 1e5, int(future.sum()))
        elif pd.api.types.is_bool_dtype(corrupted[column]):
            corrupted.loc[future, column] = ~corrupted.loc[future, column]

    tampered = build_features(corrupted, issue_time, horizon, target, spec)
    if not honest.equals(tampered):
        differing = [
            column for column in honest.columns if not honest[column].equals(tampered[column])
        ]
        raise AssertionError(
            f"features changed when the future was corrupted, so they leak. "
            f"Offending column(s): {differing}"
        )
