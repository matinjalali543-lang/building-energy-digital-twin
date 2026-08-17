"""Control MILP, forecast leakage, anomaly detection and scenario isolation.

Three of these check that a safeguard can actually fail. A test that cannot
fail produces confidence rather than information, and every one of the
safeguards here would look fine while being broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet.anomaly.detectors import Cusum, Flatline, ResidualThreshold, anomalies_to_frame
from bet.forecasting import features as F
from bet.optimization.control import OBJECTIVES, build_control_model
from bet.optimization.mpc import _horizon_from_frame, run_mpc
from bet.optimization.solve import simultaneous_heat_and_cool, solve_control
from bet.plant.controllers import ThermostatController, run_controller
from bet.plant.faults import FaultSchedule, FaultyPlant, default_schedule, pv_soiling
from bet.plant.simulator import Plant, electrical_base_load_kw, occupancy_fraction
from bet.scenario.engine import DEFAULT_SCENARIOS, Scenario, ScenarioEngine
from bet.twin.residuals import ResidualSeries, identify_thermal_model, pv_residuals

TOL = 1e-6


def _horizon(drivers, spec, hours=24, start=0):
    """Default to day 0, which the fixture guarantees contains negative prices.

    An earlier default of 48 landed on a day without any, which would have made
    the mode-binary test pass without ever exercising the condition it exists
    to check.
    """
    window = drivers.iloc[start : start + hours]
    base = electrical_base_load_kw(occupancy_fraction(window.index, spec), spec)
    return _horizon_from_frame(window, spec, base)


# --- the control MILP ----------------------------------------------------


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_every_objective_solves_and_respects_physics(
    drivers, spec, identified_model, objective
):
    model = build_control_model(
        _horizon(drivers, spec), spec, identified_model, 21.5, objective=objective
    )
    plan = solve_control(model)
    assert plan.termination == "optimal"
    assert plan.hvac_thermal_kw.max() <= spec.hvac.heating_capacity_kw_thermal + TOL
    assert -plan.hvac_thermal_kw.min() <= spec.hvac.cooling_capacity_kw_thermal + TOL
    assert plan.grid_import_kw.max() <= spec.grid_import_limit_kw + TOL


def test_the_mode_binary_is_doing_real_work(drivers, spec, identified_model):
    """The justification for making this a MILP, checked rather than asserted.

    Priced at raw wholesale — no import markup — the relaxation heats and cools
    in the same interval during negative-price hours, being paid to burn
    electricity. The binary removes it. At the reference tariff's 60 EUR/MWh
    markup the effect nearly vanishes, which is why this test pins zero markup.
    """
    horizon = _horizon(drivers, spec)
    assert (horizon.price_eur_mwh < 0).any(), "fixture must contain negative prices"

    milp = build_control_model(
        horizon, spec, identified_model, 21.5, objective="cost", import_markup_eur_mwh=0.0
    )
    solve_control(milp)

    relaxed = build_control_model(
        horizon, spec, identified_model, 21.5, objective="cost", import_markup_eur_mwh=0.0
    )
    solve_control(relaxed, relax_binaries=True)

    assert simultaneous_heat_and_cool(milp) == 0
    assert simultaneous_heat_and_cool(relaxed) > 0, (
        "the relaxation did not misbehave, so this test proves nothing about "
        "the binary"
    )


def test_comfort_band_is_respected_when_it_is_reachable(
    drivers, spec, identified_model
):
    horizon = _horizon(drivers, spec)
    plan = solve_control(
        build_control_model(horizon, spec, identified_model, 22.0, objective="cost")
    )
    assert plan.comfort_slack_k < 0.5
    assert (plan.indoor_temp_c >= horizon.comfort_min_c - 0.5).all()
    assert (plan.indoor_temp_c <= horizon.comfort_max_c + 0.5).all()


def test_slack_keeps_a_bad_start_feasible(drivers, spec, identified_model):
    """A horizon beginning far outside the band must still return a plan."""
    horizon = _horizon(drivers, spec)
    plan = solve_control(
        build_control_model(horizon, spec, identified_model, 5.0, objective="cost")
    )
    assert plan.termination == "optimal"
    assert plan.comfort_slack_k > 0


def test_adding_the_peak_term_lowers_the_peak(drivers, spec, identified_model):
    horizon = _horizon(drivers, spec)
    cost = solve_control(
        build_control_model(horizon, spec, identified_model, 22.0, objective="cost")
    )
    peaked = solve_control(
        build_control_model(horizon, spec, identified_model, 22.0, objective="cost_peak")
    )
    assert peaked.peak_import_kw <= cost.peak_import_kw + TOL


def test_cost_breakdown_lists_only_optimised_terms(drivers, spec, identified_model):
    for objective, terms in OBJECTIVES.items():
        plan = solve_control(
            build_control_model(
                _horizon(drivers, spec), spec, identified_model, 22.0, objective=objective
            )
        )
        assert tuple(t for t in plan.cost_breakdown() if t != "comfort_penalty") == terms


def test_unknown_objective_is_refused(drivers, spec, identified_model):
    with pytest.raises(KeyError, match="unknown objective"):
        build_control_model(
            _horizon(drivers, spec), spec, identified_model, 22.0, objective="vibes"
        )


def test_horizon_rejects_a_non_finite_forecast(drivers, spec):
    horizon = _horizon(drivers, spec)
    broken = horizon.outdoor_temp_c.copy()
    broken[3] = np.nan
    from bet.optimization.control import ControlHorizon

    with pytest.raises(ValueError, match="non-finite at step 3"):
        ControlHorizon(
            timestamps=horizon.timestamps,
            outdoor_temp_c=broken,
            irradiance_w_m2=horizon.irradiance_w_m2,
            occupancy_fraction=horizon.occupancy_fraction,
            base_load_kw=horizon.base_load_kw,
            pv_kw=horizon.pv_kw,
            price_eur_mwh=horizon.price_eur_mwh,
            comfort_min_c=horizon.comfort_min_c,
            comfort_max_c=horizon.comfort_max_c,
        )


def test_mpc_beats_the_thermostat_on_cost(drivers, spec, identified_model):
    window = drivers.iloc[: 24 * 7]
    thermostat = run_controller(ThermostatController(spec), Plant(spec, window))
    mpc = run_mpc(Plant(spec, window), spec, identified_model, window, 24, "cost")
    assert mpc.total_cost_eur < thermostat.total_cost_eur


def test_mpc_applies_only_the_committed_head(drivers, spec, identified_model):
    window = drivers.iloc[: 24 * 3]
    one = run_mpc(Plant(spec, window), spec, identified_model, window, 24, "cost")
    six = run_mpc(
        Plant(spec, window), spec, identified_model, window, 24, "cost", resolve_every=6
    )
    assert six.n_solves < one.n_solves
    assert len(six.frame) == len(one.frame)


# --- forecast leakage ----------------------------------------------------


@pytest.mark.parametrize("target", ["outdoor_temp_c", "irradiance_w_m2"])
def test_features_ignore_the_future(drivers, target):
    issue = drivers.index[24 * 8]
    F.assert_no_future_reference(drivers, issue, 24, target)


def test_the_leakage_check_can_actually_fail(drivers, monkeypatch):
    """A leakage test that cannot fail is worse than none: it produces
    confidence instead of information."""
    original = F.build_features

    def leaking(frame, issue, horizon, target, spec=None):
        out = original(frame, issue, horizon, target, spec)
        out["leaked"] = frame[target].reindex(out.index).to_numpy()
        return out

    monkeypatch.setattr(F, "build_features", leaking)
    with pytest.raises(AssertionError, match="leaked"):
        F.assert_no_future_reference(drivers, drivers.index[24 * 8], 24, "outdoor_temp_c")


def test_a_lag_shorter_than_the_horizon_is_refused(drivers):
    with pytest.raises(ValueError, match="shorter than the 48-step horizon"):
        F.build_features(drivers, drivers.index[24 * 8], 48, "outdoor_temp_c")


def test_corrupting_a_vacuous_window_is_refused(drivers):
    beyond = drivers.index[-1] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="vacuously"):
        F.assert_no_future_reference(drivers, beyond, 24, "outdoor_temp_c")


# --- anomaly detection ---------------------------------------------------


def _pv_series(spec, drivers, schedule):
    from bet.ingestion.providers import record_telemetry

    history = record_telemetry(
        FaultyPlant(spec, drivers, schedule), ThermostatController(spec)
    )
    return pv_residuals(history.dropna(subset=["pv_kw", "irradiance_w_m2"]), spec.pv_capacity_kwp)


def test_detectors_must_be_calibrated_before_use(spec, drivers):
    series = _pv_series(spec, drivers, FaultSchedule())
    with pytest.raises(RuntimeError, match="has not been fitted"):
        ResidualThreshold().detect(series)


def test_a_large_pv_fault_is_detected(spec, drivers):
    """A 60% loss of output must be caught. If it is not, nothing subtler will be."""
    start = drivers.index[24 * 10]
    end = drivers.index[24 * 14]
    schedule = FaultSchedule(
        [pv_soiling(str(start.date()), str(end.date()), severity=0.6, ramp=False)]
    )
    clean = _pv_series(spec, drivers, FaultSchedule())
    faulted = _pv_series(spec, drivers, schedule)

    detector = Cusum(threshold=12.0).fit(clean.residual)
    anomalies = detector.detect(faulted)
    assert anomalies, "a 60% PV loss produced no detections at all"

    truth = schedule.labels_for_asset(faulted.residual.index, "pv")
    flagged = pd.Series(False, index=faulted.residual.index)
    for anomaly in anomalies:
        if anomaly.timestamp in flagged.index:
            flagged.loc[anomaly.timestamp] = True
    assert (flagged.to_numpy() & truth).sum() > 0


def test_anomaly_records_carry_an_actionable_explanation(spec, drivers):
    start, end = drivers.index[24 * 10], drivers.index[24 * 14]
    schedule = FaultSchedule(
        [pv_soiling(str(start.date()), str(end.date()), severity=0.6, ramp=False)]
    )
    clean = _pv_series(spec, drivers, FaultSchedule())
    detector = Cusum(threshold=12.0).fit(clean.residual)
    anomalies = detector.detect(_pv_series(spec, drivers, schedule))

    record = anomalies[0]
    assert record.asset and record.severity in {"low", "medium", "high"}
    assert record.unit
    assert "expected" in record.explanation or "below" in record.explanation
    frame = anomalies_to_frame(anomalies)
    assert {"severity", "expected", "observed", "deviation", "explanation"} <= set(
        frame.columns
    )


def test_flatline_catches_what_residual_detectors_cannot():
    """The structural finding: a frozen sensor produces no unusual residual,
    because the twin predicts from the previous measured value and both sides
    freeze together."""
    index = pd.date_range("2019-11-01", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    measured = pd.Series(21.0 + rng.normal(0, 0.3, 200), index=index)
    measured.iloc[100:160] = measured.iloc[99]  # frozen
    series = ResidualSeries(
        name="indoor_temp",
        asset="zone_temperature_sensor",
        measured=measured,
        predicted=measured.shift(1).bfill(),
        unit="degC",
    )
    detector = Flatline(threshold=1.0).fit(measured.iloc[:90])
    stamps = [a.timestamp for a in detector.detect(series)]
    assert stamps, "flatline detector found nothing in a frozen window"
    assert any(index[100] <= t <= index[160] for t in stamps)


# --- scenario isolation --------------------------------------------------


def test_scenarios_do_not_mutate_the_originals(spec, drivers, identified_model):
    """The brief's §13 requirement, taken literally.

    A scenario engine that mutates its inputs gives results that depend on the
    order scenarios ran in — a bug that survives review because each individual
    run looks correct.
    """
    window = drivers.iloc[: 24 * 3]
    before = window.copy(deep=True)
    engine = ScenarioEngine(spec, window, identified_model)

    for scenario in DEFAULT_SCENARIOS[:4]:
        engine.run(scenario)

    pd.testing.assert_frame_equal(before, window)
    assert engine.spec == spec


def test_a_scenario_changing_drivers_leaves_the_original_alone(
    spec, drivers, identified_model
):
    window = drivers.iloc[: 24 * 3]
    before = window["price_eur_mwh"].copy()
    engine = ScenarioEngine(spec, window, identified_model)
    engine.run(
        Scenario(
            "prices x3",
            "does tripling prices leak?",
            change_drivers=lambda d: d.assign(price_eur_mwh=d["price_eur_mwh"] * 3),
        )
    )
    pd.testing.assert_series_equal(before, window["price_eur_mwh"])


# --- the plant / twin wall -----------------------------------------------


def test_the_twin_package_does_not_import_the_plant():
    """The wall that makes every claim in this repo meaningful.

    A twin that reads the simulator's internals is measuring itself.
    """
    import ast
    import pathlib

    twin_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "bet" / "twin"
    offenders = []
    for path in twin_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "bet.plant"
            ):
                offenders.append(f"{path.name}: from {node.module}")
            elif isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}: import {a.name}"
                    for a in node.names
                    if a.name.startswith("bet.plant")
                ]
    assert not offenders, f"bet.twin must not import bet.plant: {offenders}"


def test_identification_needs_enough_data(spec, drivers):
    from bet.ingestion.providers import record_telemetry

    history = record_telemetry(Plant(spec, drivers), ThermostatController(spec))
    with pytest.raises(ValueError, match="under two days"):
        identify_thermal_model(history.iloc[:20])


def test_fault_schedule_labels_are_a_minority_of_the_record():
    """Fault coverage above ~50% makes precision meaningless: a detector that
    flags everything would score well."""
    index = pd.date_range("2019-10-15", "2019-12-31 23:00", freq="h", tz="UTC")
    labels = default_schedule().labels(index)
    share = labels["any_fault"].mean()
    assert 0.05 < share < 0.5, f"fault coverage is {share:.0%} of the record"
