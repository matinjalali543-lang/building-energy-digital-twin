# Attribution

Two external data sources. Neither is redistributed whole; the committed file is
a derived, joined subset.

## Weather and PV — PVGIS

**European Commission, Joint Research Centre — Photovoltaic Geographical
Information System (PVGIS)**, API v5.2, `seriescalc` endpoint.

- Endpoint: https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
- Tool and documentation: https://re.jrc.ec.europa.eu/pvg_tools/en/
- Pinned request: lat 52.52, lon 13.405, year 2019, 250 kWp, 30° tilt, 0°
  azimuth, 14% loss
- Response sha256:
  `454d1773904e70eaacf3deb42c2befed5eb65f1c758a3a091a8259d2ef5b8ec1`

Underlying databases, as reported by the API itself in the response metadata:

| Quantity | Database |
|---|---|
| Solar irradiance | **PVGIS-SARAH2** (Meteosat-derived) |
| Air temperature, wind | **ERA5** reanalysis (ECMWF / Copernicus) |

The JRC states that PVGIS data may be used freely, including commercially,
without fee, with the source acknowledged. This repository acknowledges it here
and in `README.md`. ERA5 is distributed under the Copernicus licence; SARAH2 is
produced by EUMETSAT CM SAF. If you intend to redistribute the raw series rather
than a derived subset, read those terms directly rather than relying on this
summary.

## Prices — Open Power System Data

**Open Power System Data**, time series data package, snapshot **`2020-10-06`**.

- Landing page: https://data.open-power-system-data.org/time_series/2020-10-06
- File: `time_series_60min_singleindex.csv`, sha256
  `6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8`
- Column used: `DE_LU_price_day_ahead`
- Package author: Jonathan Muehlenpfordt, Neon Neue Energieökonomik

**Upstream source: the ENTSO-E Transparency Platform.** OPSD republishes it; it
did not originate the measurements.

The `datapackage.json` for this snapshot carries **no machine-readable licence
field** — that was checked rather than assumed. So no licence is asserted here.
If you intend to redistribute this data or use it commercially, read the terms
at the OPSD landing page and at ENTSO-E first.

## What was changed

`data/processed/drivers_hourly.csv` is a derived subset, not a copy of either
source:

1. PVGIS timestamps floored from `:11` past to the hour;
2. seven of PVGIS's fields kept, the rest dropped;
3. one price column selected from OPSD's 300;
4. the two joined on UTC timestamp, and the two hours lacking a published price
   dropped rather than interpolated.

Reproducible with:

```bash
python data/download_weather.py && python data/download_prices.py && python data/prepare_dataset.py
```

## Licence

The **code** in this repository is MIT (see `LICENSE`). That covers the code
only, not the data, and not the simulated building's parameters — which are the
author's own assumptions, documented in `docs/formulation.md`.
