"""Forecasters for the quantities a real deployment must predict.

Baselines first, per the brief's §10. A gradient-boosted model that cannot beat
"yesterday, same hour" has not earned its place, and on at least one of these
series that is a live possibility rather than a rhetorical one.

All forecasters take the same two arguments at predict time — the frame and an
issue time — and get their features from `features.build_features`, so none can
see past the leakage barrier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bet.forecasting.features import FeatureSpec, build_features, build_training_set

# Quantities that cannot go negative. Clipping is not cosmetic: a negative
# irradiance or base load handed to the thermal model produces a schedule that
# cannot be executed.
NON_NEGATIVE = ("irradiance_w_m2", "base_load_kw", "pv_kw")


@dataclass
class Forecaster(ABC):
    """Common interface. `name` is what appears in results tables."""

    target: str
    spec: FeatureSpec = field(default_factory=FeatureSpec)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, frame: pd.DataFrame, issue_times, horizon: int) -> Forecaster: ...

    @abstractmethod
    def _raw_predict(
        self, frame: pd.DataFrame, issue_time: pd.Timestamp, horizon: int
    ) -> np.ndarray: ...

    def predict(
        self, frame: pd.DataFrame, issue_time: pd.Timestamp, horizon: int
    ) -> pd.Series:
        values = np.asarray(self._raw_predict(frame, issue_time, horizon), dtype=float)
        if self.target in NON_NEGATIVE:
            values = np.clip(values, 0.0, None)
        index = pd.DatetimeIndex(
            [pd.Timestamp(issue_time) + pd.Timedelta(hours=k) for k in range(horizon)]
        )
        return pd.Series(values, index=index, name=f"{self.target}_forecast")

    def _features(self, frame, issue_time, horizon) -> pd.DataFrame:
        return build_features(frame, issue_time, horizon, self.target, self.spec)


@dataclass
class PersistenceForecaster(Forecaster):
    """Yesterday, same hour. The benchmark everything else has to beat.

    For a series with strong daily seasonality this is a hard baseline, not a
    straw man: it carries the whole shape of the day at zero parameters.
    """

    @property
    def name(self) -> str:
        return "persistence"

    def fit(self, frame, issue_times, horizon) -> PersistenceForecaster:
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        features = self._features(frame, issue_time, horizon)
        prediction = features["lag_24h"].to_numpy(dtype=float)
        fallback = float(features["last_observed"].iloc[0])
        return np.where(np.isnan(prediction), fallback, prediction)


@dataclass
class SeasonalMeanForecaster(Forecaster):
    """The brief's "statistical baseline": the recent mean for this hour of day.

    Averages the same clock hour over the trailing week, which smooths the
    single-day noise that daily persistence inherits wholesale.
    """

    lookback_days: int = 7

    @property
    def name(self) -> str:
        return "seasonal mean"

    def fit(self, frame, issue_times, horizon) -> SeasonalMeanForecaster:
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        issue_time = pd.Timestamp(issue_time)
        if issue_time.tz is None:
            issue_time = issue_time.tz_localize("UTC")
        past = frame.loc[frame.index < issue_time, self.target]
        window = past.loc[past.index >= issue_time - pd.Timedelta(days=self.lookback_days)]
        if window.empty:
            window = past

        by_hour = window.groupby(window.index.hour).mean()
        overall = float(window.mean())
        targets = [issue_time + pd.Timedelta(hours=k) for k in range(horizon)]
        return np.array(
            [float(by_hour.get(t.hour, overall)) for t in targets], dtype=float
        )


@dataclass
class _SklearnForecaster(Forecaster):
    """Shared plumbing for the supervised models."""

    model: object = None
    _columns: list[str] = field(default_factory=list, repr=False)

    def fit(self, frame, issue_times, horizon) -> _SklearnForecaster:
        X, y = build_training_set(frame, issue_times, horizon, self.target, self.spec)
        if X.empty:
            raise ValueError(
                f"no usable training rows for {self.target}; every candidate row "
                "had a missing feature or label"
            )
        self._columns = list(X.columns)
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        if not self._columns:
            raise RuntimeError(f"{self.name} has not been fitted")
        features = self._features(frame, issue_time, horizon)[self._columns]
        if features.isna().to_numpy().any():
            missing = features.columns[features.isna().any()].tolist()
            raise ValueError(
                f"cannot forecast {self.target} at {issue_time}: feature(s) "
                f"{missing} are unavailable, so the history before this issue "
                "time has a gap. Choose an issue time with complete history."
            )
        return self.model.predict(features.to_numpy())


@dataclass
class RidgeForecaster(_SklearnForecaster):
    """Linear regression with L2 regularisation on standardised features."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )

    @property
    def name(self) -> str:
        return "ridge"


@dataclass
class GradientBoostingForecaster(_SklearnForecaster):
    """Histogram gradient boosting — LightGBM-class, no extra dependency."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int | None = 6
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    @property
    def name(self) -> str:
        return "gradient boosting"


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": PersistenceForecaster,
    "seasonal_mean": SeasonalMeanForecaster,
    "ridge": RidgeForecaster,
    "gradient_boosting": GradientBoostingForecaster,
}


def build_forecaster(kind: str, target: str, spec: FeatureSpec | None = None, **kwargs):
    if kind not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown forecaster {kind!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[kind](target=target, spec=spec or FeatureSpec(), **kwargs)


def candidates(target: str, spec: FeatureSpec | None = None) -> list[Forecaster]:
    """The comparison set: two baselines, then two learned models."""
    spec = spec or FeatureSpec()
    return [
        PersistenceForecaster(target=target, spec=spec),
        SeasonalMeanForecaster(target=target, spec=spec),
        RidgeForecaster(target=target, spec=spec),
        GradientBoostingForecaster(target=target, spec=spec),
    ]
