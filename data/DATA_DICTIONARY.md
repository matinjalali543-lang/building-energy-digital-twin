# Data dictionary

## What is real and what is simulated

This distinction runs through the whole project, so it comes first.

`data/processed/drivers_hourly.csv` holds the **exogenous drivers** — everything
the building experiences but does not control. All of it is real measured data.

Everything *inside* the building — occupancy, lighting, plug loads, HVAC
operation and indoor temperature — is produced by `bet.plant.simulator` and is
**simulated**. It is not in this file, and it is never presented as measurement.

The split is deliberate and load-bearing: it is what lets the twin be tested
against a plant whose true state it is not allowed to read.

## `drivers_hourly.csv`

8,758 hourly rows, 2019-01-01 00:00 UTC to 2019-12-31 23:00 UTC. Committed, so a
clone runs with no network access.

| Column | Unit | Description |
|---|---|---|
| `utc_timestamp` | — | Index. UTC, hourly, gapless, monotonic. |
| `outdoor_temp_c` | °C | Dry-bulb air temperature at 2 m (ERA5 reanalysis). |
| `irradiance_w_m2` | W/m² | Global irradiance **in the plane of the PV array** (30° tilt, due south). |
| `sun_elevation_deg` | ° | Solar elevation. Zero or negative at night. |
| `wind_speed_m_s` | m/s | Wind speed at 10 m. |
| `pv_kw` | kW | AC output of a 250 kWp rooftop array, computed by PVGIS from the irradiance and temperature above. |
| `price_eur_mwh` | EUR/MWh | Day-ahead wholesale price, DE/LU bidding zone. |

### Units and conventions

- Power in **kW**, energy in **kWh**, temperature in **°C**, price in
  **EUR/MWh**, all at a one-hour step — so a kW value and its kWh contribution
  are numerically equal.
- Timestamps are **timezone-aware UTC** throughout. Local time is derived where
  it matters (occupancy schedules, solar position), never assumed.
- `irradiance_w_m2` is plane-of-array, not horizontal. Using it as horizontal
  irradiance would overstate winter gains.

## Provenance

### Weather and PV — PVGIS

European Commission Joint Research Centre, PVGIS `seriescalc` v5.2. Real
measured data, not a typical year:

- **Irradiance**: PVGIS-SARAH2, derived from Meteosat satellite imagery.
- **Temperature and wind**: ERA5 reanalysis.
- **PV output**: computed by PVGIS from the above for the specified array.

Pinned request: latitude 52.52, longitude 13.405 (Berlin), year 2019, 250 kWp,
30° tilt, 0° azimuth (south), 14% system loss. The response was verified
**byte-identical across repeated calls**, so `download_weather.py` pins it by
sha256 and fails loudly if PVGIS ever changes it.

A note on the join: PVGIS stamps each hour at **11 minutes past** — the
satellite observation time. Timestamps are floored to the hour before joining to
prices. Skipping that step silently drops every row.

### Prices — Open Power System Data

Snapshot `2020-10-06`, column `DE_LU_price_day_ahead`. Upstream source is the
ENTSO-E Transparency Platform. See `ATTRIBUTION.md`.

Two hours in 2019 have weather but no published price. They are **dropped, not
interpolated** — an invented price is an invented arbitrage opportunity, and two
hours in a year is not worth fabricating.

## Outliers

**None are removed.** The 211 negative-price hours are real market events, and
they are directly load-bearing here: they are what makes an unconstrained HVAC
model heat and cool simultaneously to get paid for consuming. Removing them
would delete the phenomenon that forced the mode binary into the optimizer.

Price range is −90.0 to 121.5 EUR/MWh; outdoor temperature −7.2 to 37.3 °C.

## Splits

Chronological, never random — a random split over a time series leaks the future
into the past.

| Split | Range | Hours | Purpose |
|---|---|---:|---|
| train | 2019-01-01 → 2019-09-01 | 5,832 | Fit forecasters |
| validation | 2019-09-01 → 2019-10-15 | 1,056 | Model selection |
| test | 2019-10-15 → 2019-12-31 | 1,870 | **Headline numbers** |

The test window is deliberately autumn/winter: it carries both heating and
shoulder-season cooling, so the HVAC mode binary is exercised in both
directions. A summer test window would have left the heating path untested.

## Why the building is simulated

No public dataset carries metered electrical load, indoor temperature,
occupancy and HVAC state for a single building together. Indoor temperature is
the one that matters most — without it there is no thermal state to estimate,
and the comfort-constrained control that distinguishes a digital twin from a
dashboard cannot be built at all.

So the building is simulated, from a documented physical model
(`docs/formulation.md`) with parameters calibrated against published office
benchmarks rather than chosen to flatter the result. The weather it responds to
and the prices it pays are real.
