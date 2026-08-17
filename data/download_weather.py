"""Download the pinned PVGIS hourly weather and PV series.

PVGIS is the European Commission's PV performance tool. `seriescalc` returns
**real measured hourly weather** for a chosen year and location — ERA5
reanalysis temperature and wind, SARAH2 satellite irradiance — plus the PV
output that array would have produced. No API key, no registration.

Everything about the request is pinned: location, year, array geometry and
rating. The response was verified byte-identical across repeated calls, so the
checksum below is a real reproducibility guarantee rather than a gesture.

    python data/download_weather.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Berlin-Mitte. Any location works; this one is pinned so the committed dataset
# is reproducible, and it pairs with the DE/LU price zone.
LATITUDE = 52.52
LONGITUDE = 13.405
YEAR = 2019

# Rooftop array on the reference building: 250 kWp, 30 degree tilt, due south,
# 14% system loss (PVGIS default). Sized in configs/base.yaml to match.
PEAK_POWER_KWP = 250
TILT_DEGREES = 30
AZIMUTH_DEGREES = 0  # PVGIS convention: 0 = south
SYSTEM_LOSS_PCT = 14

URL = (
    "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    f"?lat={LATITUDE}&lon={LONGITUDE}"
    f"&startyear={YEAR}&endyear={YEAR}"
    "&outputformat=json&browser=0"
    f"&pvcalculation=1&peakpower={PEAK_POWER_KWP}&loss={SYSTEM_LOSS_PCT}"
    f"&angle={TILT_DEGREES}&aspect={AZIMUTH_DEGREES}"
)

# Verified stable: two calls two seconds apart returned identical bytes.
EXPECTED_SHA256 = "454d1773904e70eaacf3deb42c2befed5eb65f1c758a3a091a8259d2ef5b8ec1"

DEFAULT_DEST = Path(__file__).parent / "raw" / f"pvgis_{YEAR}_berlin.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(dest: Path = DEFAULT_DEST, *, force: bool = False) -> Path:
    """Fetch the pinned series to `dest`, verifying its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if sha256_of(dest) == EXPECTED_SHA256:
            print(f"already downloaded and verified: {dest}")
            return dest
        print(f"{dest} exists but does not match the pinned checksum -- refetching")

    print(f"requesting {YEAR} hourly weather and PV for {LATITUDE}, {LONGITUDE}")
    with urllib.request.urlopen(URL) as response:
        payload = response.read()

    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"checksum mismatch for the PVGIS response\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  got      {actual}\n"
            "PVGIS has changed its output for this pinned request. The committed "
            "dataset was derived from the expected response, so regenerating from "
            "this one would produce different data. Check whether the radiation "
            "database version has moved before updating the checksum."
        )

    dest.write_bytes(payload)
    print(f"verified and saved to {dest} ({len(payload) / 1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    download(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
