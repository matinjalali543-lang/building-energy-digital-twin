"""The MPC loop — the brief's §15, and the point where the twin closes.

    estimate state -> forecast horizon -> solve -> apply FIRST action only
      -> plant advances -> measure -> update twin -> repeat

The controller never touches the plant's true state. It reads the twin, whose
state comes from noisy telemetry, and it plans against the twin's *identified*
thermal model, whose parameters are estimated and biased. Both of those are the
point: this measures what a twin built from real information is worth, not what
a controller with the physics handed to it could do.

## Three arms, and what each isolates

| Arm | Model | Forecasts | Isolates |
|---|---|---|---|
| `identified` | fitted from telemetry | predicted | what you could actually deploy |
| `true_model` | the plant's real parameters | predicted | the cost of **model error** |
| `perfect` | real parameters | actuals | the cost of **forecast error** |

Running all three separates two things usually reported as one number. A twin
that saves less than expected has either the wrong physics or the wrong
weather, and the fix is different in each case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bet.optimization.control import ControlHorizon, build_control_model
from bet.optimization.solve import ControlPlan, solve_control
from bet.plant.building import BuildingSpec
from bet.plant.simulator import Plant, comfort_bounds
from bet.twin.residuals import ThermalModel


@dataclass
class MpcResult:
    """What the controller actually did, hour by hour."""

    frame: pd.DataFrame
    spec: BuildingSpec
    label: str
    n_solves: int
    total_solve_time_s: float
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost_eur(self) -> float:
        return float(self.frame["cost_eur"].sum())

    @property
    def hvac_energy_kwh(self) -> float:
        return float(self.frame["hvac_electric_kw"].sum())

    @property
    def site_energy_kwh(self) -> float:
        return float(self.frame["site_load_kw"].sum())

    @property
    def peak_import_kw(self) -> float:
        return float(self.frame["grid_import_kw"].max())

    def comfort_violations(self, tolerance_k: float = 0.5) -> int:
        lower, upper = comfort_bounds(self.frame.index, self.spec)
        occupied = self.frame["occupancy_fraction"].to_numpy() > (
            self.spec.schedule.setback_fraction
        )
        temp = self.frame["indoor_temp_c"].to_numpy()
        breached = (temp < lower - tolerance_k) | (temp > upper + tolerance_k)
        return int((breached & occupied).sum())

    def summary(self) -> str:
        return (
            f"{self.label}: {self.total_cost_eur:,.0f} EUR, "
            f"HVAC {self.hvac_energy_kwh / 1e3:,.1f} MWh, "
            f"site {self.site_energy_kwh / 1e3:,.1f} MWh, "
            f"peak {self.peak_import_kw:,.0f} kW, "
            f"{self.comfort_violations()} violations, "
            f"{self.n_solves} solves in {self.total_solve_time_s:.0f}s"
        )


def _horizon_from_frame(
    window: pd.DataFrame, spec: BuildingSpec, base_load_kw: np.ndarray
) -> ControlHorizon:
    lower, upper = comfort_bounds(window.index, spec)
    from bet.plant.simulator import occupancy_fraction

    return ControlHorizon(
        timestamps=window.index,
        outdoor_temp_c=window["outdoor_temp_c"].to_numpy(dtype=float),
        irradiance_w_m2=window["irradiance_w_m2"].to_numpy(dtype=float),
        occupancy_fraction=occupancy_fraction(window.index, spec),
        base_load_kw=base_load_kw,
        pv_kw=window["pv_kw"].to_numpy(dtype=float) * (spec.pv_capacity_kwp / 250.0),
        price_eur_mwh=window["price_eur_mwh"].to_numpy(dtype=float),
        comfort_min_c=lower,
        comfort_max_c=upper,
    )


def run_mpc(
    plant: Plant,
    spec: BuildingSpec,
    model: ThermalModel,
    drivers: pd.DataFrame,
    lookahead: int = 24,
    objective: str = "cost",
    label: str = "mpc",
    forecasters: dict | None = None,
    history: pd.DataFrame | None = None,
    solver_name: str = "appsi_highs",
    resolve_every: int = 1,
) -> MpcResult:
    """Drive `plant` with receding-horizon control.

    `forecasters` maps target name to a fitted forecaster. Omit it to run with
    perfect knowledge of the drivers — the `perfect` arm.

    Only the first `resolve_every` actions of each plan are executed; the rest
    exist to stop the controller behaving myopically at the horizon boundary.
    """
    from bet.plant.simulator import electrical_base_load_kw, occupancy_fraction

    n_solves, solve_time = 0, 0.0
    plan: ControlPlan | None = None
    plan_age = 0

    while not plant.finished:
        step = plant.state.step_index
        if plan is None or plan_age >= resolve_every:
            end = min(step + lookahead, len(drivers))
            window = drivers.iloc[step:end]
            if len(window) < 2:
                break

            if forecasters is not None and history is not None:
                window = _apply_forecasts(window, forecasters, history, plant, step)

            base_load = electrical_base_load_kw(
                occupancy_fraction(window.index, spec), spec
            )
            horizon = _horizon_from_frame(window, spec, base_load)
            control_model = build_control_model(
                horizon,
                spec,
                model,
                initial_temp_c=plant.state.indoor_temp_c,
                initial_soc_kwh=plant.state.battery_soc_kwh,
                objective=objective,
            )
            plan = solve_control(control_model, solver_name)
            n_solves += 1
            solve_time += plan.solve_time_s
            plan_age = 0

        plant.step(plan.action_at(plan_age))
        plan_age += 1

    return MpcResult(
        frame=plant.history_frame(),
        spec=spec,
        label=label,
        n_solves=n_solves,
        total_solve_time_s=solve_time,
        metadata={
            "objective": objective,
            "lookahead": lookahead,
            "thermal_model": model.summary(),
            "forecast_driven": forecasters is not None,
        },
    )


def _apply_forecasts(window, forecasters, history, plant, step):
    """Replace known drivers with what the forecasters predict.

    Prices are left alone: the day-ahead auction publishes them, so a controller
    planning the next 24 hours genuinely has them. Substituting a forecast there
    would invent an error that does not exist in operation.
    """
    issue_time = window.index[0]
    forecast = window.copy()
    for target, forecaster in forecasters.items():
        if target not in forecast.columns:
            continue
        try:
            predicted = forecaster.predict(history, issue_time, len(window))
        except (ValueError, RuntimeError):
            continue  # no usable history; fall back to the known driver
        forecast[target] = predicted.to_numpy()[: len(window)]
    return forecast
