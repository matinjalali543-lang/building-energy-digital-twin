"""Solving the control MILP and extracting a plan.

HiGHS handles it. The mode binaries make it a MILP, but only one per interval —
24 for a day-ahead horizon — so it solves in tens of milliseconds and an hourly
MPC loop over a full year is minutes rather than hours.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pyomo.environ import ConcreteModel, SolverFactory, TransformationFactory, value
from pyomo.opt import TerminationCondition

from bet.plant.simulator import ControlAction

DEFAULT_SOLVER = "appsi_highs"


@dataclass
class ControlPlan:
    """A solved plan over the horizon, plus what it is expected to cost."""

    hvac_thermal_kw: np.ndarray
    battery_kw: np.ndarray
    indoor_temp_c: np.ndarray
    grid_import_kw: np.ndarray
    grid_export_kw: np.ndarray
    peak_import_kw: float

    total_cost_eur: float
    energy_cost_eur: float
    consumption_cost_eur: float
    peak_cost_eur: float
    comfort_penalty_eur: float
    comfort_slack_k: float

    objective_terms: tuple[str, ...]
    solve_time_s: float
    termination: str
    metadata: dict = field(default_factory=dict)

    def action_at(self, step: int = 0) -> ControlAction:
        """The action for one interval, in the form the plant accepts.

        The MILP works with two non-negative thermal variables and a binary;
        the plant takes one signed number. Collapsing them here is the only
        place the two conventions meet.
        """
        return ControlAction(
            hvac_thermal_kw=float(self.hvac_thermal_kw[step]),
            battery_kw=float(self.battery_kw[step]),
        )

    def cost_breakdown(self) -> dict[str, float]:
        available = {
            "energy": self.energy_cost_eur,
            "consumption": self.consumption_cost_eur,
            "peak": self.peak_cost_eur,
        }
        breakdown = {term: available[term] for term in self.objective_terms}
        if self.comfort_penalty_eur > 1e-9:
            breakdown["comfort_penalty"] = self.comfort_penalty_eur
        return breakdown

    @property
    def hvac_energy_kwh(self) -> float:
        return float(np.abs(self.hvac_thermal_kw).sum())

    def summary(self) -> str:
        return (
            f"{self.total_cost_eur:,.2f} EUR over {len(self.hvac_thermal_kw)} steps, "
            f"peak {self.peak_import_kw:.0f} kW, "
            f"comfort slack {self.comfort_slack_k:.2f} K, "
            f"{self.termination}, {self.solve_time_s * 1e3:.0f} ms"
        )


def solve_control(
    model: ConcreteModel,
    solver_name: str = DEFAULT_SOLVER,
    time_limit: float | None = None,
    relax_binaries: bool = False,
) -> ControlPlan:
    """Solve and extract.

    `relax_binaries` drops the integrality requirement, which is not an
    optimisation trick — it is how `tests/test_control_model.py` demonstrates
    that the mode binary is doing real work. Relaxed, the model heats and cools
    in the same interval during negative-price hours.
    """
    if relax_binaries:
        TransformationFactory("core.relax_integer_vars").apply_to(model)

    solver = SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["time_limit"] = time_limit

    started = time.perf_counter()
    results = solver.solve(model, load_solutions=False)
    elapsed = time.perf_counter() - started

    condition = results.solver.termination_condition
    if condition != TerminationCondition.optimal:
        raise RuntimeError(_explain(condition, model, solver_name))
    model.solutions.load_from(results)

    periods = sorted(model.T)
    heat = np.array([value(model.q_heat[t]) for t in periods])
    cool = np.array([value(model.q_cool[t]) for t in periods])
    spec = model._spec

    if spec.battery is not None:
        battery = np.array(
            [value(model.charge[t]) - value(model.discharge[t]) for t in periods]
        )
    else:
        battery = np.zeros(len(periods))

    slack = float(
        sum(value(model.slack_low[t]) + value(model.slack_high[t]) for t in periods)
    )

    return ControlPlan(
        hvac_thermal_kw=heat - cool,
        battery_kw=battery,
        indoor_temp_c=np.array([value(model.indoor_temp[t]) for t in periods]),
        grid_import_kw=np.array([value(model.grid_import[t]) for t in periods]),
        grid_export_kw=np.array([value(model.grid_export[t]) for t in periods]),
        peak_import_kw=float(value(model.peak)),
        total_cost_eur=float(value(model.objective)),
        energy_cost_eur=float(value(model.energy_cost)),
        consumption_cost_eur=float(value(model.consumption_cost)),
        peak_cost_eur=float(value(model.peak_cost)),
        comfort_penalty_eur=float(value(model.comfort_penalty)),
        comfort_slack_k=slack,
        objective_terms=tuple(model._terms),
        solve_time_s=elapsed,
        termination=str(condition),
        metadata={
            "relaxed": relax_binaries,
            "thermal_model": model._thermal_model.summary(),
        },
    )


def simultaneous_heat_and_cool(plan_model: ConcreteModel, tolerance: float = 1e-3) -> int:
    """Count intervals where the solution both heats and cools.

    Zero for the MILP; non-zero for its relaxation during negative-price hours.
    That contrast is the justification for the binary and is asserted in the
    test suite rather than left as a claim in a docstring.
    """
    return sum(
        1
        for t in plan_model.T
        if value(plan_model.q_heat[t]) > tolerance
        and value(plan_model.q_cool[t]) > tolerance
    )


def _explain(condition, model: ConcreteModel, solver_name: str) -> str:
    if condition == TerminationCondition.infeasible:
        spec = model._spec
        return (
            "the control MILP is infeasible. With comfort slack enabled this "
            "should be nearly impossible, so the cause is usually a hard limit "
            "rather than the temperature band:\n"
            f"  - site load exceeds the grid import limit ({spec.grid_import_limit_kw} kW);\n"
            f"  - surplus PV exceeds the export limit ({spec.grid_export_limit_kw} kW) "
            "and the battery cannot absorb it;\n"
            "  - the terminal state-of-charge constraint cannot be met within the "
            "battery's power rating over this horizon.\n"
            "If comfort slack was disabled, a horizon starting outside the band "
            "is also infeasible by construction."
        )
    if condition == TerminationCondition.maxTimeLimit:
        return (
            "the solver hit its time limit. For a horizon of this size that "
            "usually means the lookahead is far longer than intended."
        )
    return f"solver {solver_name!r} terminated with {condition}, not an optimal solution"
