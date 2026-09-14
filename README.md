# Multi-Agent Energy Market Simulation

> Competitive equilibrium for coupled electricity, hydrogen, certificate, and end-product markets — decentralised **ADMM** with a **social-planner** benchmark.

**Kian Jafarinejad** · PhD Researcher, TU Delft · [K.Jafarinejad@tudelft.nl](mailto:K.Jafarinejad@tudelft.nl)

---

## About

Julia model of a multi-agent energy system where independent firms trade across **five coupled spot markets**, with optional **endogenous investment** (VRES, electrolyzer, green offtaker) and **CVaR risk aversion**. A centralised social planner solves the same technology stack as a welfare benchmark.

Investment is committed **once**, before operations, against **15 equiprobable scenarios**: 5 diverse weather years (which drive VRES availability *and*, through a degree-day model, electricity demand) crossed with 3 natural-gas price levels (**1× / 2× / 3×** TTF). Every risk-aware agent optimises over the full set. See [DOCUMENTATION.md §9.8](DOCUMENTATION.md#98-gas-prices-and-the-15-scenario-grid).

| | |
|---|---|
| **Theory & maths** | [DOCUMENTATION.md](DOCUMENTATION.md) — equilibrium (§4), formulation (§5), ADMM (§6), calibration (§9) |
| **Scenarios** | [DOCUMENTATION.md §9.7–§9.8](DOCUMENTATION.md#97-weather-scenarios-representative-days-and-availability-factors) — weather years, demand coupling, gas price grid |
| **Bilateral contracts** | [DOCUMENTATION.md §2](DOCUMENTATION.md#contract-pools-me-pap--me-top--me-sop) — shared capacity $C$, settlement $K$, PaP/ToP/SoP, risk at $\gamma<1$ |
| **Configuration** | [Data/data.yaml](Data/data.yaml) |
| **Outputs** | [DOCUMENTATION.md §12](DOCUMENTATION.md#12-output-files) |

---

## Entry points

| Script | What it computes |
|--------|------------------|
| [`social_planner.jl`](social_planner.jl) | Centralised welfare optimum — **complete risk trading** |
| [`market_exposure.jl`](market_exposure.jl) | Decentralised equilibrium via ADMM — **incomplete risk trading** |
| [`green_h2_social_planner.jl`](green_h2_social_planner.jl) | Partial planner — **electrolyzer + green offtaker** merged (one coalition CVaR) |
| [`green_social_planner.jl`](green_social_planner.jl) | Partial planner — **VRES + electrolyzer + green offtaker** merged (one coalition CVaR) |
| [`me_pap.jl`](me_pap.jl) | ME + bilateral **PPA** (PaP) + **HPA** (pay-as-produced) |
| [`me_top.jl`](me_top.jl) | ME + PPA (PaP) + **HPA** (take-or-pay) |
| [`me_sop.jl`](me_sop.jl) | ME + PPA (PaP) + **HPA** (send-or-pay) |

Contract economics (shared capacity, strikes, volume modes): [DOCUMENTATION.md §2](DOCUMENTATION.md#contract-pools-me-pap--me-top--me-sop).

**Recommended order:** `social_planner.jl` → `market_exposure.jl`. At `gamma = 1`, ADMM warm-starts from planner prices, quantities, and capacities (ME = SP in a few iterations). At `gamma < 1`, **only prices** are loaded — primal/cap seeds from the matching-β planner would bias ME toward extra green. See [DOCUMENTATION.md §6.6](DOCUMENTATION.md#66-warm-start-from-social-planner).

At `gamma = 1`, decentralised and planner solutions should agree; at `gamma < 1`, they represent different risk institutions — details in [§4.8](DOCUMENTATION.md#48-literature-labels-and-price-interpretation-daertrycke-et-al) and [§4.10.6](DOCUMENTATION.md#4106-what-changes-in-strategy-when-gamma--1-or-beta-rises) (SP invests more green; ME invests less wind).

---

## Requirements

| Component | Purpose |
|-----------|---------|
| [Julia](https://julialang.org/downloads/) **1.9+** | Runtime (tested on 1.12) |
| [Gurobi](https://www.gurobi.com/) **10+** | ADMM agent subproblems |
| [Ipopt](https://coin-or.github.io/Ipopt/) via `Ipopt.jl` | Social planner QCP |

Academic Gurobi licenses: [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Installation

```bash
git clone <repository-url>
cd Now
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Verify Gurobi:

```bash
julia --project=. -e "using Gurobi; Gurobi.Env(); println(\"Gurobi OK\")"
```

Ensure input data exists under `Data/` and `Input/` (see [DOCUMENTATION.md §9.7](DOCUMENTATION.md#97-weather-scenarios-representative-days-and-availability-factors)). The five weather years are built from ERA5 reanalysis under one common NL calibration, with electricity demand coupled to temperature via a heating/cooling degree-day model, then reduced to 8 representative days with **RepresentativePeriodsFinder.jl** (hierarchical clustering). To regenerate all weather inputs:

```bash
julia Input/rep_periods/setup_env.jl                                                # one-time: instantiate the RPF sub-environment
julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl  # ERA5 → clustering → timeseries_<label>.csv + output_<label>/
```

---

## Usage

```bash
# 1. Benchmark (run first)
julia --project=. social_planner.jl

# 2. Decentralised equilibrium
julia --project=. market_exposure.jl

# 3. Optional — bilateral contract pools
julia --project=. me_pap.jl    # or me_top.jl / me_sop.jl
```

Results are written to `social_planner_results/`, `market_exposure_results/`, and `me_pap_results/` (or `me_top_results/`, `me_sop_results/`).

**Figures** (after planner + market-exposure runs):

```bash
pip install pandas numpy matplotlib
python visualization/visualize_results.py
```

---

## Project structure

```
Now/
├── social_planner.jl              # Centralised benchmark
├── market_exposure.jl             # ADMM — five spot markets
├── me_pap.jl / me_top.jl / me_sop.jl  # ADMM + PPA/HPA (see DOCUMENTATION.md §2)
├── green_h2_social_planner.jl     # Partial planner (H₂ chain merged)
├── green_social_planner.jl        # Partial planner (full green chain merged)
├── Data/data.yaml                 # All configuration
├── Input/                         # Timeseries & representative days
├── Source/                        # Agents, markets, ADMM, planner
├── DOCUMENTATION.md               # Full technical reference
└── visualization/                 # Python comparison plots
```

---

## Configuration (essentials)

Edit **`Data/data.yaml`**:

```yaml
General:
  nTimesteps: 24
  nReprDays: 8
  base_year: 2025

# The uncertainty set: 5 weather years × 3 gas price levels = 15 equiprobable
# scenarios, all of which every risk-aware agent optimises over (see §9.8).
Scenarios:
  weather_years: [1, 2, 3, 4, 5]
  gas_price_multipliers: [1.00, 2.00, 3.00]

# Single source of truth for fuel- and carbon-linked costs. Conventional plant
# SRMCs and grey ammonia marginal cost are both DERIVED from these, so one gas
# price moves power and ammonia together.
Fuel:
  GasPrice:  34.40       # €/MWh_th — TTF 2024 annual average
  CO2Price:  64.79       # €/tCO₂ — EU-ETS 2024 annual average

ADMM:
  max_iter: 5000        # iteration budget (RA ME typically ~3k–4k iters)
  epsilon: 0.2          # ME flow-market Boyd ε_abs (MW/slot; L2 bar ≈ 10.73 MW; see §6.5)
  gamma: 0.5            # 1 = risk-neutral; 0.5 = risk-averse (see §4.10)
  beta: 0.4             # CVaR tail level; sweep 0.2, 0.4, 0.6, 0.8 at gamma = 0.5
```

**Risk-averse sweep.** At `gamma = 0.5`, use `beta = 1 - k/15` so the tail spans `k` whole scenarios. The standard sweep `0.2, 0.4, 0.6, 0.8` lands on 12, 9, 6, and 3 scenarios. Do **not** set `beta = 0.95` (narrower than one of 15 scenarios). Compare **within** each entry point: more RA ⇒ SP adds green (gas hedge); ME cuts wind (weather CVaR). Headline metrics: `E[SW]` and resource cost in `Cost_Metrics.csv` / `run_summary.txt`.

Add agents by adding blocks under `Power:`, `Hydrogen:`, etc. — no code changes required for supported types. Full parameter reference: [DOCUMENTATION.md §9](DOCUMENTATION.md#9-configuration-reference-datayaml).

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Gurobi license error | Set `GUROBI_HOME`; run `grbgetkey` |
| ADMM does not converge | Run SP first; at `gamma < 1` expect thousands of iterations from a cold `g_bar`. Raise `epsilon` only with the §6.5 justification, or raise `max_iter`. `epsilon_cap` does **not** stop ME. See [§6.5](DOCUMENTATION.md#65-convergence-tolerances-boyd-style) |
| Save crash after ADMM converged | Start a **fresh** `julia --project=.` (not a VS Code REPL). Julia 1.12 world-age can keep a stale tee. CSVs are already written if the crash is in `tee_run_log.jl`. |
| SP non-optimal | Check `data.yaml` for infeasible capacities or demands |

More detail: [DOCUMENTATION.md](DOCUMENTATION.md) and `ADMM_Convergence.csv` / `ADMM_Diagnostics.csv`.

---

## References

Key sources are listed in [DOCUMENTATION.md §14](DOCUMENTATION.md#14-references), including Boyd et al. (2011) for ADMM, d'Aertrycke et al. (2018) for risk-trading institutions, and Hoschle et al. (2018) for risk-averse equilibrium.

---

## License

Developed for academic research at TU Delft. Contact the author for licensing and collaboration.
