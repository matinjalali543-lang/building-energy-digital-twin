"""Digital twin dashboard — the brief's §16 pages.

A **visualization layer, not the optimization engine**. Every number comes from
the same functions the command line calls, so the app cannot drift from the
reported results; if it disagrees with `results/`, one of them is broken.

Plotly is used here because interactive time-series exploration genuinely helps;
matplotlib produces the committed figures, which must render headless in CI.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bet.data import load_drivers, split  # noqa: E402
from bet.ingestion.providers import record_telemetry  # noqa: E402
from bet.optimization.control import OBJECTIVES  # noqa: E402
from bet.optimization.mpc import run_mpc  # noqa: E402
from bet.plant.building import REFERENCE_BUILDING  # noqa: E402
from bet.plant.calibration import report as calibration_report  # noqa: E402
from bet.plant.controllers import ThermostatController, run_controller  # noqa: E402
from bet.plant.faults import FaultyPlant, default_schedule  # noqa: E402
from bet.plant.simulator import Plant, comfort_bounds  # noqa: E402
from bet.twin.entities import reference_building_model  # noqa: E402
from bet.twin.residuals import all_residuals, identify_thermal_model  # noqa: E402

st.set_page_config(page_title="Building Energy Digital Twin", layout="wide")

RESULTS = Path(__file__).resolve().parents[1] / "results"


@st.cache_data(show_spinner=False)
def cached_drivers() -> pd.DataFrame:
    return load_drivers()


@st.cache_resource(show_spinner="Identifying the thermal model from telemetry…")
def cached_model():
    spec = REFERENCE_BUILDING
    train = split(cached_drivers(), "train")
    history = record_telemetry(Plant(spec, train), ThermostatController(spec))
    usable = history.dropna(subset=["indoor_temp_c", "outdoor_temp_c", "irradiance_w_m2"])
    return identify_thermal_model(usable)


@st.cache_data(show_spinner="Running the controllers…")
def cached_runs(_spec, objective: str, days: int):
    window = split(cached_drivers(), "test").iloc[: days * 24]
    thermostat = run_controller(ThermostatController(_spec), Plant(_spec, window))
    mpc = run_mpc(Plant(_spec, window), _spec, cached_model(), window, 24, objective, "MPC")
    return thermostat, mpc, window


def _load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


st.title("Building Energy Digital Twin")
st.caption(
    "A **simulated** 7,500 m² office driven by real Berlin 2019 weather and real "
    "DE/LU day-ahead prices. The physical building is `plant/simulator.py` — "
    "nothing here is a measurement of a real building."
)

with st.sidebar:
    st.header("Building")
    battery_kwh = st.slider("Battery capacity (kWh)", 0, 1000, 250, 50)
    pv_kwp = st.slider("PV array (kWp)", 0, 750, 250, 25)
    band = st.slider("Comfort band widening (± K)", 0.0, 3.0, 0.0, 0.5)
    cop = st.slider("HVAC efficiency multiplier", 0.5, 1.2, 1.0, 0.05)

    st.header("Control")
    objective = st.selectbox("Objective", list(OBJECTIVES), index=0)
    days = st.slider("Days to simulate", 3, 30, 7)

    st.caption(
        "Every figure is recomputed from these settings using the same "
        "functions the command line calls."
    )

base = REFERENCE_BUILDING
spec = replace(
    base,
    pv_capacity_kwp=max(pv_kwp, 1e-6),
    battery=replace(base.battery, energy_capacity_kwh=battery_kwh, power_kw=battery_kwh / 2)
    if battery_kwh > 0
    else None,
    comfort=replace(
        base.comfort,
        occupied_min_c=base.comfort.occupied_min_c - band,
        occupied_max_c=base.comfort.occupied_max_c + band,
    ),
    hvac=replace(
        base.hvac,
        cop_heating=base.hvac.cop_heating * cop,
        cop_cooling=base.hvac.cop_cooling * cop,
    ),
)

overview, building_tab, assets, forecast_tab, anomalies_tab, scenarios_tab, about = st.tabs(
    ["Overview", "Building", "Assets", "Forecast", "Anomalies", "Scenario Lab", "About"]
)

# ------------------------------------------------------------- overview
with overview:
    thermostat, mpc, window = cached_runs(spec, objective, days)
    saving = thermostat.total_cost_eur - mpc.total_cost_eur

    columns = st.columns(5)
    columns[0].metric("Thermostat", f"{thermostat.total_cost_eur:,.0f} EUR")
    columns[1].metric(
        "MPC", f"{mpc.total_cost_eur:,.0f} EUR",
        delta=f"{-saving:,.0f}", delta_color="inverse",
    )
    columns[2].metric("Saving", f"{100 * saving / thermostat.total_cost_eur:.1f}%")
    columns[3].metric("Peak import", f"{mpc.peak_import_kw:,.0f} kW")
    columns[4].metric(
        "Comfort violations", mpc.comfort_violations(),
        delta=int(mpc.comfort_violations() - thermostat.comfort_violations()),
        delta_color="inverse",
    )

    energy_change = (
        100 * (mpc.hvac_energy_kwh - thermostat.hvac_energy_kwh) / thermostat.hvac_energy_kwh
    )
    if energy_change > 1:
        st.warning(
            f"The MPC saves {100 * saving / thermostat.total_cost_eur:.1f}% of cost "
            f"while consuming **{energy_change:+.1f}% more HVAC energy**. "
            "Cost-optimal is not energy-optimal — try the `cost_energy` objective."
        )

    frame = mpc.frame
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Grid, PV and battery", "Indoor temperature", "Price"),
    )
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["grid_import_kw"], name="Grid import",
                   line={"color": "#8a8f94"}), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["pv_kw"], name="PV", fill="tozeroy",
                   line={"color": "#e8a33d"}), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["hvac_electric_kw"], name="HVAC",
                   line={"color": "#a83232"}), row=1, col=1)

    lower, upper = comfort_bounds(frame.index, spec)
    figure.add_trace(
        go.Scatter(x=frame.index, y=upper, name="Comfort band",
                   line={"width": 0}, showlegend=False), row=2, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=lower, name="Comfort band", fill="tonexty",
                   fillcolor="rgba(61,122,90,0.15)", line={"width": 0}), row=2, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["indoor_temp_c"], name="Indoor",
                   line={"color": "#1f4e79", "width": 2}), row=2, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["price_eur_mwh"], name="Wholesale price",
                   line={"color": "#6b4c8a"}), row=3, col=1)

    figure.update_yaxes(title_text="kW", row=1, col=1)
    figure.update_yaxes(title_text="°C", row=2, col=1)
    figure.update_yaxes(title_text="EUR/MWh", row=3, col=1)
    figure.update_layout(height=760, hovermode="x unified", margin={"t": 40})
    st.plotly_chart(figure, use_container_width=True)

# ------------------------------------------------------------- building
with building_tab:
    st.subheader("Is this a plausible office?")
    st.write(
        "Parameters are checked against published benchmarks rather than chosen "
        "to flatter a result. CI fails if any drifts outside its range."
    )
    st.dataframe(
        calibration_report(spec).drop(columns="source").round(2), width="stretch"
    )
    st.subheader("Twin's thermal model, identified from telemetry")
    model = cached_model()
    left, right = st.columns(2)
    left.metric("Identified UA", f"{model.ua_kw_per_k:.2f} kW/K", "true 5.00")
    right.metric(
        "Identified capacitance", f"{model.capacitance_kwh_per_k:.0f} kWh/K", "true 250"
    )
    st.info(
        "The twin never receives the plant's parameters. The +43% capacitance "
        "bias is what closed-loop identification does: the training data comes "
        "from a building under thermostat control, so HVAC power is itself a "
        "response to the error the regression is explaining."
    )

# --------------------------------------------------------------- assets
with assets:
    st.subheader("Site model")
    model_entities = reference_building_model()
    st.write(model_entities.describe())
    st.dataframe(
        pd.DataFrame(
            [
                {"id": a.entity_id, "name": a.name, "kind": str(a.kind),
                 "rating": a.rated_capacity, "unit": a.capacity_unit}
                for a in model_entities.assets
            ]
        ),
        width="stretch",
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"id": s.entity_id, "name": s.name, "unit": s.unit,
                 "accuracy": s.nominal_accuracy,
                 "max staleness (min)": s.max_staleness_minutes}
                for s in model_entities.sensors
            ]
        ),
        width="stretch",
    )

# ------------------------------------------------------------- forecast
with forecast_tab:
    st.subheader("What actually needs forecasting")
    st.markdown(
        "- **Wholesale price — known, not forecast.** The day-ahead auction "
        "clears at midday for all 24 hours of the next day.\n"
        "- **Occupancy — known.** It comes from the operating schedule.\n"
        "- **Weather and base load — forecast.**\n"
        "- **Indoor temperature — neither.** It is simulated forward by the "
        "twin's thermal model; the gap against measurement is the anomaly signal."
    )
    selection = _load_csv("forecast_selection.csv")
    if selection is not None:
        for target in selection["target"].unique():
            st.write(f"**{target}**")
            block = selection[selection["target"] == target].set_index("model")
            columns = [c for c in ("MAE", "RMSE", "R2", "skill_vs_persistence") if c in block]
            st.dataframe(block[columns].round(4), width="stretch")
    else:
        st.info("Run `python -m bet.forecasting.evaluate` to generate this table.")

# ------------------------------------------------------------ anomalies
with anomalies_tab:
    st.subheader("Detection, scored against known faults")
    st.dataframe(default_schedule().summary(), width="stretch")
    scores = _load_csv("anomaly_scores.csv")
    if scores is not None:
        st.dataframe(scores.round(3), width="stretch")
        st.info(
            "All three residual-based detectors score zero recall on the stuck "
            "temperature sensor, and not through bad tuning: the twin predicts "
            "from the previous measured value, so when the sensor freezes both "
            "sides of the residual freeze together. The flatline detector, "
            "watching for an absence of variation, catches it at 0.94 recall."
        )
    else:
        st.info("Run `python -m bet.anomaly.scoring` to generate this table.")

    if st.button("Show live residuals with faults injected"):
        window = split(cached_drivers(), "test").iloc[: days * 24]
        history = record_telemetry(
            FaultyPlant(spec, window, default_schedule()), ThermostatController(spec)
        )
        residuals = all_residuals(history.dropna(), cached_model(), spec.pv_capacity_kwp)
        figure = go.Figure()
        for name, series in residuals.items():
            figure.add_trace(
                go.Scatter(x=series.residual.index, y=series.residual, name=name)
            )
        figure.update_layout(height=420, yaxis_title="residual", hovermode="x unified")
        st.plotly_chart(figure, use_container_width=True)

# --------------------------------------------------------- scenario lab
with scenarios_tab:
    st.subheader("What-if")
    table = _load_csv("scenarios.csv")
    if table is not None:
        st.dataframe(table.set_index("scenario").round(2), width="stretch")
        st.success(
            "The result that contradicted the plan: removing the battery "
            "collapses the saving from 428 EUR (7.30%) to 57 EUR (0.98%). "
            "Roughly 87% of the value is the battery and 13% the thermal mass, "
            "despite the two storing the same nameplate energy — because "
            "shifting heat in the fabric raises envelope losses, and a battery "
            "loses only its round trip."
        )
    else:
        st.info("Run `python -m bet.scenario.engine` to generate this table.")
    st.caption(
        "Scenarios never mutate the originals; a test asserts the driver frame "
        "is byte-identical after a full sweep."
    )

# ------------------------------------------------------------------ about
with about:
    st.markdown(
        """
### Twin, shadow, or model?

The loop closes — the controller's actions are applied to the plant, which
advances and emits the next telemetry, which updates the twin. That is a twin's
architecture.

What it is **not** is a twin of anything real. The physical building is
`plant/simulator.py`.

**Nothing in `bet.twin` imports `bet.plant`**, and an AST test enforces it. The
twin sees only noisy, occasionally-missing sensor readings. A twin that reads
the simulator's internals is measuring itself.

### Limitations that would change a real decision

- Single thermal zone — one capacitance cannot represent a south face baking
  while a north face freezes.
- No free cooling, night purge or economizer.
- Constant COP, where a real heat pump loses efficiency when it is needed most.
- A daily demand-charge proxy, where a real one is monthly.
- Point forecasts only, so the controller cannot be made risk-aware.

Full detail in `docs/decisions.md`.
"""
    )
