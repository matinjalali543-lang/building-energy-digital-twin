"""The control model: a MILP over HVAC, battery and grid, inside a comfort band.

Built from the **twin's identified thermal model**, not the plant's true
parameters. That is deliberate and is what makes the eventual comparison
meaningful: a controller given the real physics is measuring an upper bound
nobody can buy.

## Why this one needs a binary, when the sibling battery repo did not

`forecast-driven-bess-dispatch` showed that forbidding simultaneous charge and
discharge was unnecessary — round-trip losses already made it wasteful, the
optimum never did it, and the constraint never bound. The equivalent constraint
here **does** bind, and the reason is negative prices.

The size of the effect depends entirely on the tariff, and the honest numbers
are smaller than the first measurement suggested.

Priced at **raw wholesale**, the LP relaxation heats and cools in the same hour
201 times a year, 200 of them in negative-price hours: paid to consume, it runs
heater against chiller to burn electricity.

Priced at the **reference tariff** — wholesale plus a 60 EUR/MWh markup for
network charges and levies — only 15 hours of 2019 have a negative delivered
price. Solving all 364 whole days, the relaxation misbehaves in **3 hours** and
the binary is worth **0.41 EUR across the year**.

The binary is kept because it is physically correct — one reversible unit
cannot do both at once — and because at 24 binaries per horizon it costs
nothing. But it is close to non-binding at the reference tariff, and the earlier
claim that it is load-bearing holds only for a pure pass-through tariff or a
market with deeper negative prices. `tests/test_control_model.py` pins the
behaviour at zero markup, where the effect is unambiguous.

## Objectives

Four variants, because minimising cost alone has a consequence worth seeing.

`cost` alone saves 27.1% of HVAC spend against a thermostat while consuming
**33.2% more energy** — pre-heating and pre-cooling widen temperature
excursions, and wider excursions lose more heat through the envelope. Reporting
only the saving would mislead precisely the person most likely to read it, so
`cost_energy` prices kWh alongside euros and makes the trade explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    minimize,
)

from bet.plant.building import BuildingSpec
from bet.twin.residuals import ThermalModel

# Cost terms each objective variant includes.
OBJECTIVES: dict[str, tuple[str, ...]] = {
    "cost": ("energy",),
    "cost_energy": ("energy", "consumption"),
    "cost_peak": ("energy", "peak"),
    "cost_energy_peak": ("energy", "consumption", "peak"),
}

# Price applied to each kWh consumed in the `consumption` term, EUR/kWh. Not a
# market price -- a policy weight expressing how much a kWh saved is worth
# beyond its cost, standing in for carbon or for a corporate energy target.
DEFAULT_ENERGY_WEIGHT_EUR_KWH = 0.05

# Charged per kW of peak grid import over the horizon.
DEFAULT_PEAK_WEIGHT_EUR_KW = 1.0


@dataclass(frozen=True)
class ControlHorizon:
    """Everything the controller is allowed to know about the horizon ahead.

    Prices are here as *known* values, not forecasts: the day-ahead auction
    clears before the day begins. Weather and base load are forecasts and are
    labelled as such by where they came from, not by a flag on this object.
    """

    timestamps: pd.DatetimeIndex
    outdoor_temp_c: np.ndarray
    irradiance_w_m2: np.ndarray
    occupancy_fraction: np.ndarray
    base_load_kw: np.ndarray
    pv_kw: np.ndarray
    price_eur_mwh: np.ndarray
    comfort_min_c: np.ndarray
    comfort_max_c: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        for name in (
            "outdoor_temp_c", "irradiance_w_m2", "occupancy_fraction",
            "base_load_kw", "pv_kw", "price_eur_mwh", "comfort_min_c", "comfort_max_c",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.size != n:
                raise ValueError(
                    f"{name} has {array.size} values but the horizon has {n} steps"
                )
            if not np.isfinite(array).all():
                bad = int(np.argmax(~np.isfinite(array)))
                raise ValueError(
                    f"{name} is non-finite at step {bad}. A horizon with a hole in "
                    "it cannot be optimised against -- fix the forecast rather "
                    "than letting the solver meet a NaN."
                )
            object.__setattr__(self, name, array)
        if (self.comfort_max_c <= self.comfort_min_c).any():
            bad = int(np.argmax(self.comfort_max_c <= self.comfort_min_c))
            raise ValueError(
                f"comfort band is empty at step {bad}: "
                f"[{self.comfort_min_c[bad]}, {self.comfort_max_c[bad]}]"
            )

    def __len__(self) -> int:
        return len(self.timestamps)


def build_control_model(
    horizon: ControlHorizon,
    spec: BuildingSpec,
    model: ThermalModel,
    initial_temp_c: float,
    initial_soc_kwh: float | None = None,
    objective: str = "cost",
    import_markup_eur_mwh: float = 60.0,
    export_ratio: float = 0.7,
    energy_weight_eur_kwh: float = DEFAULT_ENERGY_WEIGHT_EUR_KWH,
    peak_weight_eur_kw: float = DEFAULT_PEAK_WEIGHT_EUR_KW,
    comfort_slack: bool = True,
) -> ConcreteModel:
    """Build the control MILP for one horizon.

    `comfort_slack` allows the band to be violated at a steep penalty. Without
    it a horizon that starts outside the band — which happens whenever the
    controller takes over from a thermostat, or after a fault — is simply
    infeasible, and an infeasible controller stops rather than doing its best.
    """
    if objective not in OBJECTIVES:
        raise KeyError(
            f"unknown objective {objective!r}; expected one of {sorted(OBJECTIVES)}"
        )
    terms = OBJECTIVES[objective]
    n = len(horizon)
    dt = model.dt_hours

    m = ConcreteModel(name=f"building_control[{objective}]")
    m.T = RangeSet(0, n - 1)

    buy = horizon.price_eur_mwh + import_markup_eur_mwh
    sell = horizon.price_eur_mwh * export_ratio
    m.buy = Param(m.T, initialize=dict(enumerate(buy)))
    m.sell = Param(m.T, initialize=dict(enumerate(sell)))

    hvac = spec.hvac
    # Thermal power, split by mode so the objective can price each at its own COP.
    m.q_heat = Var(m.T, bounds=(0, hvac.heating_capacity_kw_thermal))
    m.q_cool = Var(m.T, bounds=(0, hvac.cooling_capacity_kw_thermal))
    # 1 = heating allowed this step, 0 = cooling allowed. The load-bearing binary.
    m.mode = Var(m.T, within=Binary)

    m.indoor_temp = Var(m.T, bounds=(0.0, 45.0))
    m.slack_low = Var(m.T, within=NonNegativeReals)
    m.slack_high = Var(m.T, within=NonNegativeReals)

    m.grid_import = Var(m.T, bounds=(0, spec.grid_import_limit_kw))
    m.grid_export = Var(m.T, bounds=(0, spec.grid_export_limit_kw))
    m.peak = Var(within=NonNegativeReals)

    battery = spec.battery
    if battery is not None:
        m.charge = Var(m.T, bounds=(0, battery.power_kw))
        m.discharge = Var(m.T, bounds=(0, battery.power_kw))
        m.soc = Var(
            m.T,
            bounds=(
                battery.soc_min_frac * battery.energy_capacity_kwh,
                battery.soc_max_frac * battery.energy_capacity_kwh,
            ),
        )
        start_soc = (
            initial_soc_kwh
            if initial_soc_kwh is not None
            else battery.initial_soc_frac * battery.energy_capacity_kwh
        )

        def soc_balance(m, t):
            previous = start_soc if t == m.T.first() else m.soc[t - 1]
            return m.soc[t] == (
                previous
                + battery.charge_efficiency * m.charge[t] * dt
                - m.discharge[t] * dt / battery.discharge_efficiency
            )

        m.soc_balance = Constraint(m.T, rule=soc_balance)
        # Return the battery to where it started, so a finite horizon cannot
        # flatter itself by ending empty.
        m.terminal_soc = Constraint(expr=m.soc[m.T.last()] >= start_soc)

    # --- the mode binary -------------------------------------------------
    def heating_gate(m, t):
        return m.q_heat[t] <= hvac.heating_capacity_kw_thermal * m.mode[t]

    def cooling_gate(m, t):
        return m.q_cool[t] <= hvac.cooling_capacity_kw_thermal * (1 - m.mode[t])

    m.heating_gate = Constraint(m.T, rule=heating_gate)
    m.cooling_gate = Constraint(m.T, rule=cooling_gate)

    # --- thermal dynamics, from the twin's identified model ---------------
    def thermal_dynamics(m, t):
        previous = initial_temp_c if t == m.T.first() else m.indoor_temp[t - 1]
        net_kw = (
            model.ua_kw_per_k * (horizon.outdoor_temp_c[t] - previous)
            + model.solar_coefficient_kw_per_w_m2 * horizon.irradiance_w_m2[t]
            + model.occupancy_gain_kw * horizon.occupancy_fraction[t]
            + model.baseline_gain_kw
            + m.q_heat[t]
            - m.q_cool[t]
        )
        return m.indoor_temp[t] == previous + net_kw * dt / model.capacitance_kwh_per_k

    m.thermal_dynamics = Constraint(m.T, rule=thermal_dynamics)

    def comfort_lower(m, t):
        return m.indoor_temp[t] >= horizon.comfort_min_c[t] - m.slack_low[t]

    def comfort_upper(m, t):
        return m.indoor_temp[t] <= horizon.comfort_max_c[t] + m.slack_high[t]

    m.comfort_lower = Constraint(m.T, rule=comfort_lower)
    m.comfort_upper = Constraint(m.T, rule=comfort_upper)
    if not comfort_slack:
        m.no_slack_low = Constraint(m.T, rule=lambda m, t: m.slack_low[t] == 0)
        m.no_slack_high = Constraint(m.T, rule=lambda m, t: m.slack_high[t] == 0)

    # --- electrical balance ----------------------------------------------
    def power_balance(m, t):
        hvac_electric = m.q_heat[t] / hvac.cop_heating + m.q_cool[t] / hvac.cop_cooling
        stored = (m.charge[t] - m.discharge[t]) if battery is not None else 0
        return (
            m.grid_import[t] - m.grid_export[t]
            == horizon.base_load_kw[t] + hvac_electric + stored - horizon.pv_kw[t]
        )

    m.power_balance = Constraint(m.T, rule=power_balance)
    m.peak_definition = Constraint(m.T, rule=lambda m, t: m.peak >= m.grid_import[t])

    # --- costs, as named expressions --------------------------------------
    m.energy_cost = Expression(
        expr=sum(
            (m.buy[t] * m.grid_import[t] - m.sell[t] * m.grid_export[t]) * dt / 1e3
            for t in m.T
        )
    )
    m.consumption_cost = Expression(
        expr=energy_weight_eur_kwh
        * sum(
            (
                horizon.base_load_kw[t]
                + m.q_heat[t] / hvac.cop_heating
                + m.q_cool[t] / hvac.cop_cooling
            )
            * dt
            for t in m.T
        )
    )
    m.peak_cost = Expression(expr=peak_weight_eur_kw * m.peak)
    # Steep enough that comfort is never traded for money, but finite so the
    # model degrades gracefully instead of reporting "infeasible".
    m.comfort_penalty = Expression(
        expr=500.0 * sum(m.slack_low[t] + m.slack_high[t] for t in m.T)
    )

    available = {
        "energy": m.energy_cost,
        "consumption": m.consumption_cost,
        "peak": m.peak_cost,
    }
    m.objective = Objective(
        expr=sum(available[term] for term in terms) + m.comfort_penalty,
        sense=minimize,
    )

    m._terms = terms
    m._spec = spec
    m._horizon = horizon
    m._thermal_model = model
    return m
