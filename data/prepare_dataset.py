"""Join the pinned weather and price downloads into the committed driver series.

    python data/download_weather.py
    python data/download_prices.py
    python data/prepare_dataset.py

These are the **exogenous drivers**: everything the building experiences but
does not control. The building's own behaviour — occupancy, HVAC, lighting,
plug loads, indoor temperature — is produced by `bet.plant.simulator` and is
not in this file. Keeping the two apart is what lets the twin be tested against
a plant whose true state it cannot read.

One join detail that is easy to get wrong: PVGIS stamps each hour at **11
minutes past** (the satellite observation time), so the timestamps must be
floored to the hour before joining to the price series, or every row misses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

YEAR = 2019
PRICE_COLUMN = "DE_LU_price_day_ahead"

DEFAULT_WEATHER = Path(__file__).parent / "raw" / f"pvgis_{YEAR}_berlin.json"
DEFAULT_PRICES = Path(__file__).parent / "raw" / "opsd_time_series_60min.csv"
DEFAULT_OUT = Path(__file__).parent / "processed" / "drivers_hourly.csv"


def load_weather(path: Path) -> pd.DataFrame:
    """PVGIS hourly series -> a UTC-indexed frame."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_weather.py` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload["outputs"]["hourly"]

    # "20190101:0011" -> 2019-01-01 00:00 UTC. Dropping the minutes is the
    # floor described in the module docstring; PVGIS means "the 00:00 hour".
    stamps = pd.to_datetime(
        [row["time"][:8] + row["time"][9:11] for row in hourly],
        format="%Y%m%d%H",
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "outdoor_temp_c": [row["T2m"] for row in hourly],
            "irradiance_w_m2": [row["G(i)"] for row in hourly],
            "sun_elevation_deg": [row["H_sun"] for row in hourly],
            "wind_speed_m_s": [row["WS10m"] for row in hourly],
            "pv_kw": [row["P"] / 1e3 for row in hourly],
        },
        index=pd.DatetimeIndex(stamps, name="utc_timestamp"),
    ).sort_index()

    meta = payload["inputs"]
    frame.attrs["radiation_db"] = meta["meteo_data"]["radiation_db"]
    frame.attrs["meteo_db"] = meta["meteo_data"]["meteo_db"]
    frame.attrs["peak_power_kwp"] = meta["pv_module"]["peak_power"]
    return frame


def load_prices(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_prices.py` first."
        )
    frame = pd.read_csv(
        path,
        usecols=["utc_timestamp", PRICE_COLUMN],
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[PRICE_COLUMN].rename("price_eur_mwh").sort_index()


def prepare(
    weather_path: Path = DEFAULT_WEATHER,
    price_path: Path = DEFAULT_PRICES,
    out_path: Path = DEFAULT_OUT,
) -> pd.DataFrame:
    weather = load_weather(weather_path)
    prices = load_prices(price_path)
    print(f"weather {len(weather)} hours, prices {len(prices)} hours")

    joined = weather.join(prices, how="left")
    missing = int(joined["price_eur_mwh"].isna().sum())
    if missing:
        print(f"  {missing} hour(s) have weather but no published price")

    # Drop rather than interpolate. Two missing prices in a year is not worth
    # inventing, and a fabricated price is a fabricated arbitrage opportunity.
    joined = joined.dropna(subset=["price_eur_mwh"])

    out = joined.round(4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, lineterminator="\n")

    price = out["price_eur_mwh"]
    print(f"\nwrote {out_path} ({len(out)} rows, {out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  radiation {weather.attrs['radiation_db']}, meteo {weather.attrs['meteo_db']}")
    print(f"  outdoor temp  {out.outdoor_temp_c.min():6.1f} .. {out.outdoor_temp_c.max():5.1f} C")
    print(f"  PV ({weather.attrs['peak_power_kwp']:.0f} kWp) peak {out.pv_kw.max():.0f} kW, "
          f"{out.pv_kw.sum() / 1e3:.0f} MWh/yr, capacity factor "
          f"{out.pv_kw.mean() / weather.attrs['peak_power_kwp']:.3f}")
    print(f"  price {price.min():.1f} .. {price.max():.1f} EUR/MWh, "
          f"mean {price.mean():.1f}, negative hours {int((price < 0).sum())}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    prepare(args.weather, args.prices, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
