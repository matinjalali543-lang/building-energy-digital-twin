# Project 02 — AI Energy Digital Twin

## 1. Project Overview

Build a portfolio-grade Digital Twin for a small commercial building or micro-energy system.

The Digital Twin should represent:
- Building electricity demand
- HVAC demand
- Lighting
- Plug loads
- PV generation
- Battery storage
- Electricity price
- Weather conditions

The system should use historical/streaming-like data to:
1. Estimate the current state of the building.
2. Forecast future behavior.
3. Detect deviations between expected and observed behavior.
4. Optimize energy operation.
5. Visualize the digital twin state.

The project should demonstrate:
- Energy modeling
- Machine learning
- Optimization
- Time-series analytics
- Software architecture
- Dashboard development

## 2. Core Concept

Do not make this merely a dashboard.

The Digital Twin must have a computational representation of the physical system:

```text
Physical Building
      ↕
Data / Telemetry
      ↕
Digital Twin State
      ↓
Forecasting
      ↓
Optimization
      ↓
Recommended Control
      ↓
Scenario Simulation
```

## 3. Technology Stack

Primary:
- Python 3.11+
- pandas
- NumPy
- scikit-learn
- XGBoost
- Pyomo
- HiGHS
- Plotly
- Streamlit
- pytest

Optional:
- FastAPI
- SQLite
- MQTT simulator
- Docker
- TimescaleDB or PostgreSQL

Avoid infrastructure complexity in version 1.

## 4. Repository Structure

```text
ai-energy-digital-twin/
├── README.md
├── pyproject.toml
├── configs/
├── data/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── twin/
│   ├── forecasting/
│   ├── optimization/
│   ├── anomaly/
│   └── evaluation/
├── tests/
├── app/
├── reports/
└── results/
```

## 5. Phase 0 — Define the Reference Building

Create a realistic but explicitly simulated commercial building.

Suggested assumptions:
- Office building
- 5,000–10,000 m²
- Weekday/weekend schedule
- HVAC system
- Lighting
- Plug loads
- Rooftop PV
- Optional BESS

Define:
- Occupancy schedule
- Operating hours
- Thermal zones
- Equipment categories
- Peak load
- PV capacity
- Battery capacity

Document all assumptions.

## 6. Phase 1 — Digital Twin Data Model

Define entities such as:

```text
Building
Zone
Asset
Sensor
EnergyMeter
WeatherStation
Battery
PVSystem
Tariff
```

Each entity should have:
- ID
- Metadata
- Unit
- Timestamp where relevant
- State variables
- Quality flags

Use a consistent schema.

## 7. Phase 2 — Data Ingestion

Implement a simulated telemetry layer.

Inputs:
- Building load
- Indoor temperature
- Outdoor temperature
- Irradiance
- PV output
- Occupancy
- HVAC status
- Battery SOC

Support:
- CSV replay
- Synthetic real-time stream

Every data point must have:
- Timestamp
- Value
- Unit
- Quality/status

## 8. Phase 3 — State Estimation

The Digital Twin should maintain the latest known state.

Example:

```text
TwinState
├── timestamp
├── building_load
├── indoor_temperature
├── outdoor_temperature
├── occupancy
├── pv_output
├── battery_soc
└── asset_states
```

Implement:
- Latest-state update
- Missing sensor handling
- Out-of-order timestamp handling
- Data-quality flags

## 9. Phase 4 — Baseline Building Model

Create a transparent baseline model.

At minimum model electricity demand as a function of:
- Hour
- Day of week
- Temperature
- Occupancy
- Lagged demand

Possible approaches:
- Linear regression
- Gradient boosting
- XGBoost

The baseline must be interpretable.

## 10. Phase 5 — Forecasting

Forecast:
- Building load
- PV generation
- Optional indoor temperature

Use:
- Persistence baseline
- Statistical baseline
- ML model

Use chronological train/validation/test splits.

Report:
- MAE
- RMSE
- sMAPE where useful
- Error by hour
- Error during peak periods

## 11. Phase 6 — Digital Twin Calibration

The twin should compare:
- Predicted state
- Measured state

Calculate residuals:

```text
residual = measured - predicted
```

Use residuals for:
- Model calibration
- Anomaly detection
- Asset performance monitoring

Do not call a simple ML prediction model a complete Digital Twin. The project must maintain a stateful computational representation and support scenario simulation.

## 12. Phase 7 — Anomaly Detection

Detect:
- Abnormal energy consumption
- Unexpected HVAC behavior
- PV underperformance
- Battery anomalies

Start with:
- Residual threshold
- Rolling z-score

Then optionally:
- Isolation Forest
- Autoencoder

Each anomaly must contain:
- Timestamp
- Asset
- Severity
- Expected value
- Observed value
- Deviation
- Explanation

## 13. Phase 8 — Scenario Engine

Create a simulation layer that can answer:

- What if occupancy increases 20%?
- What if PV capacity increases?
- What if electricity prices increase?
- What if HVAC efficiency decreases?
- What if battery capacity doubles?
- What if an HVAC unit fails?

Scenario input must not modify the original data.

## 14. Phase 9 — Optimization Engine

Use Pyomo to optimize operation.

Decision variables may include:
- HVAC operating level
- Battery charging
- Battery discharging
- Grid import
- Grid export
- Flexible load shifting

Objectives:
- Minimize energy cost
- Minimize peak demand
- Maintain comfort

Comfort constraints:
- Indoor temperature lower bound
- Indoor temperature upper bound

Use a weighted objective or lexicographic approach.

## 15. Phase 10 — MPC-Style Control

Implement a simplified Model Predictive Control loop:

```text
Current Twin State
       ↓
Forecast horizon
       ↓
Optimization
       ↓
Implement first action
       ↓
New measurement
       ↓
Update Twin
       ↓
Repeat
```

This is much more impressive than a static optimization notebook.

## 16. Phase 11 — Digital Twin Dashboard

Build Streamlit dashboard pages:

### Overview
- Current energy consumption
- Current PV
- Battery SOC
- Grid import
- Cost
- Alerts

### Building
- Load profile
- Occupancy
- Temperature

### Assets
- HVAC
- PV
- Battery

### Forecast
- Actual vs predicted

### Optimization
- Recommended actions
- Cost before/after
- Peak before/after

### Anomalies
- Active alerts
- Historical anomalies

### Scenario Lab
Allow users to modify:
- Occupancy
- PV size
- Battery size
- Tariff
- HVAC efficiency

## 17. Phase 12 — Architecture

Use a modular architecture:

```text
Data Layer
    ↓
Twin State Layer
    ↓
Models
 ┌──┼───────────┐
 ↓  ↓           ↓
Forecast  Anomaly  Scenario
 └──┼───────────┘
    ↓
Optimization
    ↓
Dashboard/API
```

## 18. Testing

Test:
- Timestamp handling
- State updates
- Data validation
- Forecasting
- Residual calculations
- Anomaly detection
- Optimization constraints
- Scenario isolation

Create an integration test for:

```text
data → twin → forecast → optimization → result
```

## 19. Performance and Engineering

Avoid unnecessary microservices.

Use clean interfaces:
- `DataProvider`
- `TwinStateStore`
- `Forecaster`
- `AnomalyDetector`
- `ScenarioEngine`
- `Optimizer`

Use dependency injection where helpful.

## 20. README

Explain:
- What a Digital Twin is in this project
- Physical-to-digital mapping
- Architecture
- Data model
- Forecasting
- Anomaly detection
- Optimization
- MPC loop
- Dashboard
- Results
- Limitations

Explicitly distinguish:
- Digital Twin
- Digital Model
- Dashboard

## 21. Stretch Goals

After baseline:
- Real MQTT broker
- FastAPI service
- PostgreSQL/TimescaleDB
- Docker
- Reinforcement learning controller
- Physics-informed ML
- Fault diagnosis
- Multi-building digital twin
- Cloud deployment

## 22. Definition of Done

The project is complete when:
- A simulated building can be represented digitally.
- Telemetry updates the twin state.
- Forecasts are generated.
- Deviations are detected.
- Scenarios can be simulated.
- Optimization produces control recommendations.
- The MPC loop works in simulation.
- Dashboard communicates the complete workflow.
- Tests pass.
- README is professional and reproducible.

---

**Decisions taken after this brief, including every place it was overridden, are
recorded in [docs/decisions.md](docs/decisions.md).**
