"""Check the reference building against published benchmarks.

A simulated building can be made to produce any number you like. This module
exists so that it cannot be made to produce a *flattering* one unnoticed: it
runs the plant under the incumbent thermostat and compares the result with
ranges that real European offices occupy.

The ranges below are broad on purpose. They are order-of-magnitude sanity
bounds — "is this a plausible office at all" — not a claim to have replicated
any specific building. A parameter set that fails them is wrong; one that passes
them is merely not obviously wrong.

    python -m bet.plant.calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bet.data import load_drivers
from bet.plant.building import REFERENCE_BUILDING, BuildingSpec
from bet.plant.controllers import ThermostatController, run_controller
from bet.plant.simulator import Plant


@dataclass(frozen=True)
class Benchmark:
    """One quantity, its plausible range, and where the range comes from."""

    name: str
    unit: str
    low: float
    high: float
    source: str

    def check(self, value: float) -> bool:
        return self.low <= value <= self.high


# Ranges for European air-conditioned offices. Deliberately wide: the point is
# to catch a building that is physically absurd, not to certify one.
BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        "electricity intensity", "kWh/m2/yr", 70, 250,
        "CIBSE TM46 / typical EU office electricity benchmarks",
    ),
    Benchmark(
        "peak electrical demand", "W/m2", 20, 80,
        "typical office peak demand density",
    ),
    Benchmark(
        "HVAC share of electricity", "%", 15, 55,
        "HVAC is usually the largest single end use in an air-conditioned office",
    ),
    Benchmark(
        "thermal time constant", "h", 10, 150,
        "lightweight to heavyweight construction",
    ),
    Benchmark(
        "PV yield", "kWh/kWp/yr", 800, 1300,
        "central and northern Europe, fixed tilt",
    ),
    Benchmark(
        "PV share of consumption", "%", 5, 60,
        "rooftop array on an office of this size",
    ),
    Benchmark(
        "load factor", "-", 0.25, 0.75,
        "mean over peak demand for a scheduled commercial building",
    ),
)


def measure(
    spec: BuildingSpec = REFERENCE_BUILDING, drivers: pd.DataFrame | None = None
) -> dict[str, float]:
    """Run a year under the thermostat and return the benchmark quantities."""
    drivers = load_drivers() if drivers is None else drivers
    plant = Plant(spec, drivers)
    result = run_controller(ThermostatController(spec), plant)
    frame = result.frame

    hours = len(frame)
    site_kwh = result.site_energy_kwh
    pv_kwh = float(frame["pv_kw"].sum())
    peak_kw = float(frame["site_load_kw"].max())

    return {
        "electricity intensity": site_kwh / spec.floor_area_m2 * (8760 / hours),
        "peak electrical demand": peak_kw * 1e3 / spec.floor_area_m2,
        "HVAC share of electricity": 100 * result.hvac_energy_kwh / site_kwh,
        "thermal time constant": spec.time_constant_hours,
        "PV yield": pv_kwh / spec.pv_capacity_kwp * (8760 / hours),
        "PV share of consumption": 100 * pv_kwh / site_kwh,
        "load factor": float(frame["site_load_kw"].mean()) / peak_kw,
    }


def report(
    spec: BuildingSpec = REFERENCE_BUILDING, drivers: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Benchmark table: measured value, plausible range, pass or fail."""
    measured = measure(spec, drivers)
    rows = []
    for benchmark in BENCHMARKS:
        value = measured[benchmark.name]
        rows.append(
            {
                "quantity": benchmark.name,
                "unit": benchmark.unit,
                "measured": round(value, 2),
                "low": benchmark.low,
                "high": benchmark.high,
                "within": benchmark.check(value),
                "source": benchmark.source,
            }
        )
    return pd.DataFrame(rows).set_index("quantity")


def main() -> int:
    spec = REFERENCE_BUILDING
    print(spec.summary())
    print(
        f"  peak internal gain {spec.peak_internal_gain_kw:.0f} kW, "
        f"occupied lighting+plug {spec.occupied_electrical_kw:.0f} kW, "
        f"overnight base {spec.base_electrical_kw:.0f} kW"
    )
    if spec.battery:
        print(
            f"  battery {spec.battery.energy_capacity_kwh:.0f} kWh "
            f"({spec.battery.usable_kwh:.0f} usable) vs fabric "
            f"{spec.thermal_storage_kwh_electric():.0f} kWh-e across the comfort band"
        )

    table = report(spec)
    print()
    print(table.drop(columns="source").to_string())

    failed = table[~table["within"]]
    if len(failed):
        print("\nOUTSIDE the plausible range:")
        for name, row in failed.iterrows():
            print(f"  {name}: {row['measured']} {row['unit']} "
                  f"(expected {row['low']}-{row['high']}) -- {row['source']}")
    else:
        print("\nall quantities inside published ranges")

    out = Path(__file__).resolve().parents[3] / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calibration.csv")
    print(f"wrote {out / 'calibration.csv'}")
    return 0 if not len(failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
