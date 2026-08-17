"""Download the pinned Open Power System Data snapshot for wholesale prices.

Only the DE/LU day-ahead price column is used here. The file is 130 MB because
OPSD ships 300 columns covering most of Europe; the prepare step slices one.

The snapshot is pinned by date rather than tracking "latest". OPSD stopped
publishing after 2020-10-06, so the pin is simply the last release.

    python data/download_prices.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

SNAPSHOT = "2020-10-06"
URL = (
    "https://data.open-power-system-data.org/time_series/"
    f"{SNAPSHOT}/time_series_60min_singleindex.csv"
)

EXPECTED_SHA256 = "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"
EXPECTED_BYTES = 130_339_665

DEFAULT_DEST = Path(__file__).parent / "raw" / "opsd_time_series_60min.csv"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(dest: Path = DEFAULT_DEST, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if sha256_of(dest) == EXPECTED_SHA256:
            print(f"already downloaded and verified: {dest}")
            return dest
        print(f"{dest} exists but the checksum does not match -- re-downloading")

    print(f"downloading {EXPECTED_BYTES / 1e6:.0f} MB from {URL}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(URL) as response, tmp.open("wb") as out:
        downloaded = 0
        while block := response.read(1 << 20):
            out.write(block)
            downloaded += len(block)
            print(f"\r  {downloaded / 1e6:6.1f} MB", end="", flush=True)
    print()

    actual = sha256_of(tmp)
    if actual != EXPECTED_SHA256:
        tmp.unlink()
        raise RuntimeError(
            f"checksum mismatch for {URL}\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  got      {actual}\n"
            "The upstream snapshot has changed, so the committed dataset can no "
            "longer be reproduced from it."
        )
    tmp.replace(dest)
    print(f"verified and saved to {dest}")
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
