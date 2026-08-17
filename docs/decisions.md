# Decisions made after the brief

`PROJECT_BRIEF.md` holds the brief as written, unchanged. This file records every
decision taken after it — including the places where the brief was overridden,
and why. Nothing here was decided silently.

## A. Repository name (supersedes §4)

The brief names the repo `ai-energy-digital-twin`. It is called
**`building-energy-digital-twin`** instead: "AI energy" describes half the repos
in this series and says nothing about what is being modelled, whereas "building"
separates it at a glance from the battery and grid repos beside it.

The Python package is `bet`.

## B. What kind of twin this actually is (§20)

The brief asks for the distinction to be drawn explicitly. Drawn honestly:

| | Data flow | Here |
|---|---|---|
| Digital **Model** | Manual, nothing automatic | no |
| Digital **Shadow** | Automatic physical → digital | more than this |
| Digital **Twin** | Automatic in both directions | **structurally, yes** |

The loop genuinely closes: the controller's actions are applied to the plant,
which advances its state and emits the next telemetry, which updates the twin.
That is the architecture of a twin.

What it is **not** is a twin of anything real. The "physical" building is
`src/bet/plant/simulator.py`. Every claim in this repository is a claim about a
simulation, and the README says so in its first screen rather than implying a
building exists somewhere.

## C. Data: real drivers, simulated interior (§6, §7)

The brief permits a fully simulated building. Two of the three signal families
are real anyway, because inventing them would make the results circular:

| Signal | Source | Real? |
|---|---|---|
| Outdoor temperature, irradiance | PVGIS (ERA5 / SARAH2), Berlin 2019 | **real** |
| PV generation | PVGIS `seriescalc`, 250 kWp array | **real** physics on real weather |
| Wholesale price | OPSD DE/LU day-ahead 2019 | **real** |
| Occupancy, lighting, plug, HVAC, indoor temperature | `plant/simulator.py` | **simulated** |

The interior is simulated because no public dataset carries metered load,
indoor temperature, occupancy and HVAC state for one building together — and
without indoor temperature there is no thermal state to twin, which would leave
the comfort-constrained control that distinguishes this project impossible to
build.

## D. The HVAC model needs a binary — but far less than first measured

This decision changed once it was tested against the tariff the model actually
uses, and the correction is recorded rather than quietly folded in.

**What the planning probe found.** Priced at raw wholesale, the LP relaxation
heats and cools in the same hour **201 times a year, 200 of them in
negative-price hours**. Paid to consume, it runs heater against chiller to burn
electricity. That looked like a sharp contrast with the sibling battery repo,
where the equivalent constraint was shown to be unnecessary.

**What the built model shows.** The probe used raw wholesale as the delivered
price. The reference tariff adds a 60 EUR/MWh markup for network charges and
levies, and that changes the picture substantially:

| Import markup | Hours with negative delivered price | Relaxation violations |
|---:|---:|---:|
| 0 EUR/MWh | 211 | 201 |
| 30 EUR/MWh | 45 | — |
| **60 EUR/MWh (reference)** | **15** | **3** |

Solving all 364 whole days of 2019 at the reference tariff, the relaxation
misbehaves in **3 hours** and the binary is worth **0.41 EUR across the year**.

So the binary is retained — it is physically correct, a single reversible unit
cannot heat and cool at once, and it costs nothing at this size — but the claim
that it is load-bearing holds only for a pure wholesale pass-through tariff, or
for a market with deeper and more frequent negative prices than Germany saw in
2019. The contrast with the sibling repo is real and much smaller than first
stated.

The test suite pins the behaviour at zero markup, where the effect is large
enough to be unambiguous.

## E. Cost-optimal is not energy-optimal (§14)

Also measured before building: against a reactive thermostat, the
cost-minimising MILP saves **27.1% of HVAC cost while consuming 33.2% more
energy**. Pre-heating and pre-cooling into cheap hours widens temperature
excursions, and wider excursions lose more heat through the envelope.

Reporting only the cost saving would be misleading to exactly the person most
likely to read it. The optimizer therefore ships with an explicit
`cost_energy` objective variant that prices kWh alongside euros, and both
numbers are reported side by side everywhere.

## F. The battery is secondary, and is benchmarked against the walls (§5)

§5 lists the BESS as optional. It is included, but the building's thermal mass
is the primary storage asset and the two are compared directly: across its 3 K
comfort band the fabric stores about **750 kWh-thermal, roughly 250 kWh-electric
at COP 3** — comparable to a large commercial battery, and already paid for.

Making the battery co-equal would have restated the sibling repo; omitting it
would have left the comparison unmade.

## G. Anomalies are scored, not just surfaced (§12)

Because the plant is simulated, faults can be injected at known times with known
magnitudes: HVAC efficiency decay, PV soiling, a stuck temperature sensor, a
failed HVAC stage, a drifting meter. Detection is therefore reported as
**precision, recall, detection latency and false-alarm rate on clean data**,
rather than as a gallery of flagged points that cannot be judged.

## H. Plotly and matplotlib, each where it belongs (§3)

The brief lists both. Plotly drives the Streamlit dashboard, where interactive
time-series exploration genuinely helps. matplotlib produces the committed
README figures, which must render headless in CI as PNGs. Neither library sets
the global backend at import — a lesson from the previous repo, where doing so
made every notebook figure render as the text `<Figure size ...>`.

## I. Infrastructure deferred (§3, §21)

FastAPI, MQTT, SQLite, TimescaleDB and Docker are all listed as optional, and
§3 says outright to avoid infrastructure complexity in version 1. None are
built. The `DataProvider` protocol is the seam they would plug into, and it has
two implementations (CSV replay and a synthetic stream) so that the seam is
demonstrated rather than merely asserted.

## J. Build sequencing

Built phase by phase with a commit each, verified locally, then pushed as a
complete history — matching the five sibling repos.
