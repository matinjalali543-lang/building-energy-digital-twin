"""Baseline controllers — what the building does today.

These belong with the plant rather than with the optimizer, because they
represent the incumbent control system: the thermostat that is already in the
building and that any proposed controller has to beat. The MPC controller lives
in `bet.optimization.mpc` and is the thing under test.

The comparison that matters throughout this project is **MPC against the
thermostat**, not MPC against nothing. Beating "no control at all" would be
trivial and would tell a buyer nothing.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bet.plant.building import SECONDS_PER_HOUR, BuildingSpec
from bet.plant.simulator import ControlAction, Plant, SimulationResult


class Controller(Protocol):
    """Anything that can decide an action given the plant's current step."""

    name: str

    def act(self, plant: Plant) -> ControlAction: ...


class ThermostatController:
    """A reactive deadband thermostat. No forecast, no prices, no lookahead.

    It predicts where temperature would drift with no HVAC, and applies exactly
    enough heating or cooling to land on the nearest comfort bound. That is
    slightly better than a real bang-bang thermostat, which would overshoot —
    so the baseline is, if anything, generous to the incumbent.

    It never looks at the price, which is the entire point of the comparison.
    """

    name = "thermostat"

    def __init__(self, spec: BuildingSpec) -> None:
        self.spec = spec

    def act(self, plant: Plant) -> ControlAction:
        step = plant.state.step_index
        row = plant.drivers.iloc[step]
        lower, upper = plant.comfort_at(step)

        temp = plant.state.indoor_temp_c
        solar_kw = row["irradiance_w_m2"] * self.spec.solar_aperture_m2 * 0.6 / 1e3
        gain_kw = float(plant._gain_kw[step])  # noqa: SLF001 - same package

        # Where temperature lands with no HVAC this interval.
        drift_w = self.spec.ua_w_per_k * (row["outdoor_temp_c"] - temp) + (
            gain_kw + solar_kw
        ) * 1e3
        dt_seconds = plant.dt_hours * SECONDS_PER_HOUR
        drift_c = temp + drift_w * dt_seconds / self.spec.capacitance_j_per_k

        thermal_kw = 0.0
        if drift_c < lower:
            thermal_kw = (lower - drift_c) * self.spec.capacitance_j_per_k / dt_seconds / 1e3
        elif drift_c > upper:
            thermal_kw = (upper - drift_c) * self.spec.capacitance_j_per_k / dt_seconds / 1e3

        return ControlAction(hvac_thermal_kw=thermal_kw, battery_kw=0.0)


class ThermostatWithSelfConsumption(ThermostatController):
    """The thermostat plus a greedy battery: store surplus PV, cover shortfall.

    The controller that ships in the inverter. It never consults the price
    either, so it isolates what the *battery hardware* is worth from what
    *optimizing it* is worth.
    """

    name = "thermostat + greedy battery"

    def act(self, plant: Plant) -> ControlAction:
        action = super().act(plant)
        if self.spec.battery is None:
            return action

        step = plant.state.step_index
        row = plant.drivers.iloc[step]
        pv_kw = float(row["pv_kw"]) * plant._pv_scale  # noqa: SLF001 - same package
        hvac_electric = (
            action.hvac_thermal_kw / self.spec.hvac.cop_heating
            if action.hvac_thermal_kw >= 0
            else -action.hvac_thermal_kw / self.spec.hvac.cop_cooling
        )
        demand_kw = float(plant._base_kw[step]) + hvac_electric  # noqa: SLF001
        surplus = pv_kw - demand_kw

        battery_kw = float(
            np.clip(surplus, -self.spec.battery.power_kw, self.spec.battery.power_kw)
        )
        return ControlAction(hvac_thermal_kw=action.hvac_thermal_kw, battery_kw=battery_kw)


class NullController:
    """No HVAC at all. Free-running temperature.

    Not a serious baseline — it violates comfort constantly. It exists so the
    thermal model can be inspected without control masking it, and so tests can
    assert on the building's unforced dynamics.
    """

    name = "free-running"

    def act(self, plant: Plant) -> ControlAction:
        return ControlAction()


def run_controller(
    controller: Controller, plant: Plant, label: str | None = None
) -> SimulationResult:
    """Drive a plant to completion with a controller."""
    while not plant.finished:
        plant.step(controller.act(plant))
    return SimulationResult(
        frame=plant.history_frame(),
        spec=plant.spec,
        label=label or controller.name,
        metadata={"controller": controller.name},
    )
