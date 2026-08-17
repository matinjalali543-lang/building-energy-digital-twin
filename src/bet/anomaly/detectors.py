"""Anomaly detection on residuals, with the record the brief's §12 asks for.

Every detector consumes a `ResidualSeries` and emits `Anomaly` records carrying
timestamp, asset, severity, expected, observed, deviation and an explanation in
words. The explanation is not decoration: an alert that says "residual 3.4 sigma"
tells an operator nothing they can act on, and one that says "PV is producing
31% less than this irradiance implies" tells them to go and look at the roof.

## Thresholds are calibrated on clean data, and that is not optional

Each detector is fitted on a fault-free window to learn what normal residual
scatter looks like, then applied to the test window. Setting a threshold by
looking at the data you are about to score on is how detectors come to have
suspiciously good numbers.

## Three detectors, because they fail differently

`ResidualThreshold` is memoryless: it fires when a single residual is large. It
catches step faults immediately and can miss a slow ramp entirely, because no
individual hour is unusual.

`RollingZScore` compares against the recent past, so it adapts to drift — which
also means a slow enough fault becomes the new normal and stops being reported.

`Cusum` accumulates small consistent deviations, which is exactly the signature
of a ramp. It is slower on steps and much better on gradual decay. Three of the
five injected faults ramp, so this is the one that ought to earn its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from bet.twin.residuals import ResidualSeries


@dataclass(frozen=True)
class Anomaly:
    """One detection. The brief's §12 fields, plus what raised it."""

    timestamp: pd.Timestamp
    asset: str
    quantity: str
    severity: str  # "low" | "medium" | "high"
    expected: float
    observed: float
    deviation: float
    unit: str
    detector: str
    score: float
    explanation: str

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "quantity": self.quantity,
            "severity": self.severity,
            "expected": round(self.expected, 3),
            "observed": round(self.observed, 3),
            "deviation": round(self.deviation, 3),
            "unit": self.unit,
            "detector": self.detector,
            "score": round(self.score, 3),
            "explanation": self.explanation,
        }


def _severity(score: float, threshold: float) -> str:
    ratio = abs(score) / threshold if threshold > 0 else 0.0
    if ratio >= 2.5:
        return "high"
    if ratio >= 1.5:
        return "medium"
    return "low"


def _explain(series: ResidualSeries, expected: float, observed: float) -> str:
    """A sentence an operator could act on."""
    deviation = observed - expected
    direction = "above" if deviation > 0 else "below"
    if abs(expected) > 1e-6:
        relative = f" ({abs(deviation) / abs(expected):.0%})"
    else:
        relative = ""
    subject = {
        "pv_output": "PV output is",
        "hvac_electric": "HVAC electricity is",
        "indoor_temp": "Indoor temperature is",
    }.get(series.name, f"{series.name} is")
    return (
        f"{subject} {abs(deviation):.2f} {series.unit} {direction} the "
        f"{expected:.2f} {series.unit} expected{relative}."
    )


class AnomalyDetector(Protocol):
    """Fit on clean residuals, then detect on new ones."""

    name: str

    def fit(self, residuals: pd.Series) -> AnomalyDetector: ...

    def score(self, residuals: pd.Series) -> pd.Series: ...

    def detect(self, series: ResidualSeries) -> list[Anomaly]: ...


@dataclass
class _BaseDetector:
    threshold: float = 3.0
    name: str = "base"
    _centre: float = field(default=0.0, repr=False)
    _scale: float = field(default=1.0, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(self, residuals: pd.Series):
        clean = pd.Series(residuals).dropna()
        if len(clean) < 24:
            raise ValueError(
                f"{self.name}: only {len(clean)} clean residuals supplied. A "
                "threshold calibrated on less than a day of data is a guess."
            )
        self._centre = float(clean.median())
        # Median absolute deviation, scaled to be comparable with a standard
        # deviation for normal data. Robust on purpose: a couple of outliers in
        # the calibration window would inflate a plain std and blind the detector.
        mad = float((clean - self._centre).abs().median())
        self._scale = max(mad * 1.4826, 1e-9)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted. Calibrate it on a fault-free "
                "window before detecting, or its threshold means nothing."
            )

    def detect(self, series: ResidualSeries) -> list[Anomaly]:
        self._require_fitted()
        scores = self.score(series.residual)
        flagged = scores.abs() >= self.threshold

        anomalies = []
        for timestamp in scores.index[flagged.fillna(False)]:
            expected = float(series.predicted.get(timestamp, np.nan))
            observed = float(series.measured.get(timestamp, np.nan))
            if not (np.isfinite(expected) and np.isfinite(observed)):
                continue
            score = float(scores[timestamp])
            anomalies.append(
                Anomaly(
                    timestamp=timestamp,
                    asset=series.asset,
                    quantity=series.name,
                    severity=_severity(score, self.threshold),
                    expected=expected,
                    observed=observed,
                    deviation=observed - expected,
                    unit=series.unit,
                    detector=self.name,
                    score=score,
                    explanation=_explain(series, expected, observed),
                )
            )
        return anomalies


@dataclass
class ResidualThreshold(_BaseDetector):
    """Fires when a single residual is large relative to calibrated scatter.

    Memoryless, so it catches a step the hour it happens and can miss a slow
    ramp completely — no individual hour of a gradual decay is unusual.
    """

    name: str = "residual threshold"

    def score(self, residuals: pd.Series) -> pd.Series:
        self._require_fitted()
        return (pd.Series(residuals) - self._centre) / self._scale


@dataclass
class RollingZScore(_BaseDetector):
    """Compares each residual against the recent past.

    Adapts to slow drift, which cuts both ways: a fault developing slowly enough
    becomes the new normal and stops being reported. That is a real failure mode
    and the scoring below exposes it rather than hiding it.
    """

    window_hours: int = 168
    name: str = "rolling z-score"

    def score(self, residuals: pd.Series) -> pd.Series:
        self._require_fitted()
        series = pd.Series(residuals)
        rolling = series.rolling(self.window_hours, min_periods=24)
        centre = rolling.median()
        scale = rolling.std().clip(lower=self._scale * 0.25)
        return (series - centre) / scale


@dataclass
class Cusum(_BaseDetector):
    """Accumulates consistent small deviations — built for ramps.

    Tracks a one-sided sum in each direction, reset to zero whenever the
    evidence turns. `drift` is the per-step allowance in calibrated sigma below
    which a deviation is treated as noise; without it the sum wanders off on
    ordinary scatter.
    """

    drift: float = 0.5
    name: str = "cusum"

    def score(self, residuals: pd.Series) -> pd.Series:
        self._require_fitted()
        normalised = ((pd.Series(residuals) - self._centre) / self._scale).fillna(0.0)
        values = normalised.to_numpy()

        high = np.zeros(len(values))
        low = np.zeros(len(values))
        for i in range(1, len(values)):
            high[i] = max(0.0, high[i - 1] + values[i] - self.drift)
            low[i] = min(0.0, low[i - 1] + values[i] + self.drift)
        combined = np.where(high >= -low, high, low)
        return pd.Series(combined, index=normalised.index)


def default_detectors(threshold: float = 4.0) -> list[AnomalyDetector]:
    """One of each, at thresholds that make them comparable.

    CUSUM's statistic accumulates, so a threshold that is sensible for a single
    residual would fire constantly. It gets a larger one; the scoring reports
    both false-alarm rates so the choice can be judged rather than trusted.
    """
    return [
        ResidualThreshold(threshold=threshold, name="residual threshold"),
        RollingZScore(threshold=threshold, name="rolling z-score"),
        Cusum(threshold=threshold * 3, name="cusum"),
        Flatline(threshold=1.0, name="flatline"),
    ]


def anomalies_to_frame(anomalies: list[Anomaly]) -> pd.DataFrame:
    """Tidy frame of anomaly records, sorted in time."""
    if not anomalies:
        return pd.DataFrame(
            columns=[
                "timestamp", "asset", "quantity", "severity", "expected",
                "observed", "deviation", "unit", "detector", "score", "explanation",
            ]
        ).set_index("timestamp")
    frame = pd.DataFrame([a.as_dict() for a in anomalies])
    return frame.sort_values("timestamp").set_index("timestamp")


@dataclass
class Flatline(_BaseDetector):
    """Flags a reading that has stopped moving.

    Added after measurement, not by anticipation. The three residual-based
    detectors above all scored **zero recall on the stuck temperature sensor**,
    and the reason is structural rather than a tuning failure: the twin predicts
    the next temperature from the *previous measured* one, so when the sensor
    freezes both sides of the residual freeze together. The residual is then only
    as large as the hourly change the model expected -- a few tenths of a degree,
    well inside normal scatter.

    A stuck sensor's real signature is not a large residual but an absence of
    variation, which no residual detector can see. This one watches the measured
    series directly and fires when its rolling spread collapses below what
    instrument noise alone would produce.

    The general lesson, worth more than this detector: a fault class can be
    undetectable by an entire family of methods, and only measuring recall per
    fault reveals it.
    """

    window_hours: int = 6
    name: str = "flatline"

    def score(self, residuals: pd.Series) -> pd.Series:
        self._require_fitted()
        series = pd.Series(residuals)
        spread = series.rolling(self.window_hours, min_periods=self.window_hours).std()
        # Score rises as the spread falls below the calibrated noise floor.
        return (self._scale / spread.clip(lower=1e-9)).fillna(0.0)

    def detect(self, series: ResidualSeries) -> list[Anomaly]:
        # Runs on the measured signal, not the residual: the whole point is that
        # the residual carries no information when the sensor is frozen.
        self._require_fitted()
        measured = series.measured
        spread = measured.rolling(
            self.window_hours, min_periods=self.window_hours
        ).std()
        flagged = spread < self._scale

        anomalies = []
        for timestamp in spread.index[flagged.fillna(False)]:
            observed = float(measured.get(timestamp, np.nan))
            if not np.isfinite(observed):
                continue
            local_spread = float(spread[timestamp])
            anomalies.append(
                Anomaly(
                    timestamp=timestamp,
                    asset=series.asset,
                    quantity=series.name,
                    severity="high" if local_spread < self._scale * 0.25 else "medium",
                    expected=float(self._scale),
                    observed=local_spread,
                    deviation=local_spread - float(self._scale),
                    unit=series.unit,
                    detector=self.name,
                    score=float(self._scale / max(local_spread, 1e-9)),
                    explanation=(
                        f"{series.name} has not varied for {self.window_hours} h "
                        f"(spread {local_spread:.3f} {series.unit} against a "
                        f"{self._scale:.3f} {series.unit} noise floor) -- the "
                        "sensor appears frozen rather than the building being steady."
                    ),
                )
            )
        return anomalies

    def fit(self, residuals: pd.Series):
        """Calibrate the noise floor from the measured signal's own variation."""
        clean = pd.Series(residuals).dropna()
        if len(clean) < 24:
            raise ValueError(f"{self.name}: only {len(clean)} clean values supplied")
        rolling = clean.rolling(self.window_hours, min_periods=self.window_hours).std()
        floor = float(rolling.dropna().quantile(0.02))
        self._centre = 0.0
        self._scale = max(floor, 1e-6)
        self._fitted = True
        return self
