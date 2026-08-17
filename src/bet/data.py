"""Loading the committed driver series.

The only module that knows where the data file lives. Everything downstream
takes frames or dataclasses, so "where did this number come from" stays
answerable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DRIVERS = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "drivers_hourly.csv"
)

REQUIRED_COLUMNS = ("outdoor_temp_c", "irradiance_w_m2", "pv_kw", "price_eur_mwh")

# Chronological, never random. The test window is autumn/winter on purpose: it
# carries both heating and shoulder-season cooling, so the HVAC mode binary is
# exercised in both directions. A summer window would leave heating untested.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2019-01-01", "2019-09-01"),
    "validation": ("2019-09-01", "2019-10-15"),
    "test": ("2019-10-15", "2020-01-01"),
}


def load_drivers(path: str | Path | None = None) -> pd.DataFrame:
    """Read the committed exogenous drivers as a UTC-indexed frame."""
    path = Path(path) if path is not None else DEFAULT_DRIVERS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It should be committed; if you are working from a "
            "partial checkout, regenerate it with `python data/download_weather.py "
            "&& python data/download_prices.py && python data/prepare_dataset.py`."
        )
    frame = pd.read_csv(path, index_col="utc_timestamp", parse_dates=["utc_timestamp"])
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return frame.sort_index()


def split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Slice one of the named chronological splits."""
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(SPLITS)}")
    start, end = SPLITS[name]
    window = frame.loc[(frame.index >= start) & (frame.index < end)]
    if window.empty:
        raise ValueError(
            f"split {name!r} ({start} to {end}) selected no rows from a frame "
            f"covering {frame.index[0]} to {frame.index[-1]}"
        )
    return window


def whole_days(frame: pd.DataFrame, periods_per_day: int = 24) -> pd.DataFrame:
    """Trim to complete days.

    Control is solved a day at a time; a partial day at either end would be
    solved over a short horizon and its cost would not be comparable.
    """
    counts = frame.groupby(frame.index.floor("D")).transform("size")
    return frame.loc[counts.iloc[:, 0] == periods_per_day]


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """Summary of the drivers, used by the EDA notebook and the README."""
    rows = []
    for column in REQUIRED_COLUMNS:
        series = frame[column].dropna()
        rows.append(
            {
                "series": column,
                "n": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "median": series.median(),
                "max": series.max(),
            }
        )
    summary = pd.DataFrame(rows).set_index("series")
    summary.loc["price_eur_mwh", "negative_hours"] = int(
        (frame["price_eur_mwh"] < 0).sum()
    )
    return summary
