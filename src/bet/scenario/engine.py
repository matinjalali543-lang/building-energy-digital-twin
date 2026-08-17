"""What-if scenarios — the brief's §13.

The brief's one hard requirement is that *scenario input must not modify the
original data*, and it is worth taking literally. A scenario engine that mutates
its inputs produces results which depend on the order scenarios were run in,
which is the kind of bug that survives review because every individual run looks
right.

Two defences, and the second is what actually makes it safe:

* `BuildingSpec` and every part of it is a frozen dataclass, so `replace()`
  returns a new object rather than editing one.
* Driver frames are copied before any column is touched.

`tests/test_scenario_isolation.py` asserts the original frame is byte-identical
after a full sweep, and that the baseline result is reproducible afterwards.

    python -m bet.scenario.engine
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from bet.data import load_drivers, split
from bet.ingestion.providers import record_telemetry
from bet.optimization.mpc import MpcResult, run_mpc
from bet.plant.building import REFERENCE_BUILDING, BuildingSpec
from bet.plant.controllers import ThermostatController, run_controller
from bet.plant.simulator import Plant
from bet.twin.residuals import ThermalModel, identify_thermal_model


@dataclass(frozen=True)
class Scenario:
    """One what-if: how to change the building, the world, or both.

    Both hooks return **new** objects. Neither is permitted to mutate what it
    is given, and the isolation test is what enforces that rather than trust.
    """

    name: str
    question: str
    change_building: Callable[[BuildingSpec], BuildingSpec] | None = None
    change_drivers: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def apply_building(self, spec: BuildingSpec) -> BuildingSpec:
        return self.change_building(spec) if self.change_building else spec

    def apply_drivers(self, drivers: pd.DataFrame) -> pd.DataFrame:
        # Copy first, unconditionally. A scenario that only reads still gets a
        # copy, so no caller can ever hand a mutable view onward by accident.
        working = drivers.copy(deep=True)
        return self.change_drivers(working) if self.change_drivers else working


def _scale_occupancy(spec: BuildingSpec, factor: float) -> BuildingSpec:
    """More people means more heat and more plug load, not just more heads."""
    return replace(
        spec,
        people_gain_w_m2=spec.people_gain_w_m2 * factor,
        plug_gain_w_m2=spec.plug_gain_w_m2 * factor,
    )


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("baseline", "What does the reference building cost to run?"),
    Scenario(
        "occupancy +20%",
        "What if the building fills up?",
        change_building=lambda s: _scale_occupancy(s, 1.2),
    ),
    Scenario(
        "PV doubled",
        "Is a bigger array worth it?",
        change_building=lambda s: replace(s, pv_capacity_kwp=s.pv_capacity_kwp * 2),
    ),
    Scenario(
        "battery doubled",
        "Does more storage help, given the fabric already stores 250 kWh-e?",
        change_building=lambda s: replace(
            s,
            battery=replace(
                s.battery,
                energy_capacity_kwh=s.battery.energy_capacity_kwh * 2,
                power_kw=s.battery.power_kw * 2,
            ),
        ),
    ),
    Scenario(
        "no battery",
        "How much of the saving is the battery, and how much the walls?",
        change_building=lambda s: replace(s, battery=None),
    ),
    Scenario(
        "HVAC efficiency -25%",
        "What does a degraded heat pump cost over a season?",
        change_building=lambda s: replace(
            s,
            hvac=replace(
                s.hvac,
                cop_heating=s.hvac.cop_heating * 0.75,
                cop_cooling=s.hvac.cop_cooling * 0.75,
            ),
        ),
    ),
    Scenario(
        "HVAC capacity -50%",
        "Can the building still be held comfortable on one stage?",
        change_building=lambda s: replace(
            s,
            hvac=replace(
                s.hvac,
                heating_capacity_kw_thermal=s.hvac.heating_capacity_kw_thermal * 0.5,
                cooling_capacity_kw_thermal=s.hvac.cooling_capacity_kw_thermal * 0.5,
            ),
        ),
    ),
    Scenario(
        "prices +50%",
        "How does the case change if energy gets dearer?",
        change_drivers=lambda d: d.assign(price_eur_mwh=d["price_eur_mwh"] * 1.5),
    ),
    Scenario(
        "comfort band +/-1 K",
        "How much is a wider comfort band worth?",
        change_building=lambda s: replace(
            s,
            comfort=replace(
                s.comfort,
                occupied_min_c=s.comfort.occupied_min_c - 1.0,
                occupied_max_c=s.comfort.occupied_max_c + 1.0,
            ),
        ),
    ),
)


@dataclass
class ScenarioResult:
    """One scenario, run under both the incumbent and the MPC controller."""

    scenario: Scenario
    thermostat: object
    mpc: MpcResult
    spec: BuildingSpec
    metadata: dict = field(default_factory=dict)

    def row(self) -> dict:
        saving = self.thermostat.total_cost_eur - self.mpc.total_cost_eur
        return {
            "scenario": self.scenario.name,
            "thermostat_eur": self.thermostat.total_cost_eur,
            "mpc_eur": self.mpc.total_cost_eur,
            "saving_eur": saving,
            "saving_pct": 100 * saving / self.thermostat.total_cost_eur,
            "mpc_hvac_mwh": self.mpc.hvac_energy_kwh / 1e3,
            "mpc_peak_kw": self.mpc.peak_import_kw,
            "mpc_violations": self.mpc.comfort_violations(),
            "thermostat_violations": self.thermostat.comfort_violations(),
        }


class ScenarioEngine:
    """Runs scenarios against a fixed baseline, without disturbing it."""

    def __init__(
        self,
        spec: BuildingSpec = REFERENCE_BUILDING,
        drivers: pd.DataFrame | None = None,
        model: ThermalModel | None = None,
        objective: str = "cost",
        lookahead: int = 24,
    ) -> None:
        self.spec = spec
        self.drivers = drivers if drivers is not None else split(load_drivers(), "test")
        self.objective = objective
        self.lookahead = lookahead
        self.model = model if model is not None else self._identify()

    def _identify(self) -> ThermalModel:
        """Fit the twin's model once, on clean training telemetry.

        Fitted on the baseline building and then held fixed across every
        scenario. That is the realistic case: a twin calibrated last month does
        not get recalibrated because someone asks a what-if question, and a
        scenario that silently re-identifies would report the benefit of a
        better model as though it were the benefit of the change.
        """
        train = split(load_drivers(), "train")
        history = record_telemetry(
            Plant(self.spec, train), ThermostatController(self.spec)
        )
        usable = history.dropna(
            subset=["indoor_temp_c", "outdoor_temp_c", "irradiance_w_m2"]
        )
        return identify_thermal_model(usable)

    def run(self, scenario: Scenario) -> ScenarioResult:
        spec = scenario.apply_building(self.spec)
        drivers = scenario.apply_drivers(self.drivers)

        thermostat = run_controller(
            ThermostatController(spec), Plant(spec, drivers), f"{scenario.name} (thermostat)"
        )
        mpc = run_mpc(
            Plant(spec, drivers),
            spec,
            self.model,
            drivers,
            self.lookahead,
            self.objective,
            f"{scenario.name} (mpc)",
        )
        return ScenarioResult(scenario, thermostat, mpc, spec)

    def run_all(self, scenarios=DEFAULT_SCENARIOS) -> pd.DataFrame:
        return pd.DataFrame([self.run(s).row() for s in scenarios]).set_index("scenario")


def main() -> int:
    engine = ScenarioEngine()
    print(f"twin model held fixed across scenarios: {engine.model.summary()}\n")

    before = engine.drivers.copy(deep=True)
    table = engine.run_all()

    # The brief's §13 requirement, checked here as well as in the test suite.
    unchanged = before.equals(engine.drivers)
    print(table.round(2).to_string())
    print(f"\noriginal driver frame unchanged after the sweep: {unchanged}")
    if not unchanged:
        raise RuntimeError(
            "a scenario mutated the shared driver frame -- results now depend on "
            "the order scenarios were run in"
        )

    out = Path(__file__).resolve().parents[3] / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scenarios.csv")
    print(f"wrote {out / 'scenarios.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
