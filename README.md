# Multi-Agent Energy Market Simulation

> Competitive equilibrium for coupled electricity, hydrogen, certificate, and end-product markets — decentralised **ADMM** with a **social-planner** benchmark.

**Kian Jafarinejad** · PhD Researcher, TU Delft · [K.Jafarinejad@tudelft.nl](mailto:K.Jafarinejad@tudelft.nl)

---

## About

Julia model of a multi-agent energy system where independent firms trade across **five coupled spot markets**, with optional **endogenous investment** (VRES, electrolyzer, green offtaker) and **CVaR risk aversion**. A centralised social planner solves the same technology stack as a welfare benchmark.

| | |
|---|---|
| **Theory & maths** | [DOCUMENTATION.md](DOCUMENTATION.md) — equilibrium (§4), formulation (§5), ADMM (§6), calibration (§9) |
| **Bilateral contracts** | [DOCUMENTATION.md §2](DOCUMENTATION.md#contract-pools-me-pap--me-top--me-sop) — shared capacity $C$, settlement $K$, PaP/ToP/SoP, risk at $\gamma<1$ |
| **Configuration** | [Data/data.yaml](Data/data.yaml) |
| **Outputs** | [DOCUMENTATION.md §12](DOCUMENTATION.md#12-output-files) |

---

## Entry points

| Script | What it computes |
|--------|------------------|
| [`social_planner.jl`](social_planner.jl) | Centralised welfare optimum — **complete risk trading** |
| [`market_exposure.jl`](market_exposure.jl) | Decentralised equilibrium via ADMM — **incomplete risk trading** |
| [`me_pap.jl`](me_pap.jl) | ME + bilateral **PPA** (PaP) + **HPA** (pay-as-produced) |
| [`me_top.jl`](me_top.jl) | ME + PPA (PaP) + **HPA** (take-or-pay) |
| [`me_sop.jl`](me_sop.jl) | ME + PPA (PaP) + **HPA** (send-or-pay) |

Contract economics (shared capacity, strikes, volume modes): [DOCUMENTATION.md §2](DOCUMENTATION.md#contract-pools-me-pap--me-top--me-sop).
| [`green_h2_social_planner.jl`](green_h2_social_planner.jl) | Partial planner — **electrolyzer + green offtaker** merged (one coalition CVaR) |
| [`green_social_planner.jl`](green_social_planner.jl) | Partial planner — **VRES + electrolyzer + green offtaker** merged (one coalition CVaR) |

**Recommended order:** `social_planner.jl` → `market_exposure.jl` (ADMM warm-starts from planner prices, quantities, and capacities). See [DOCUMENTATION.md §6.6](DOCUMENTATION.md#66-warm-start-from-social-planner).

At `gamma = 1`, decentralised and planner solutions should agree; at `gamma < 1`, they represent different risk institutions — details in [§4.8](DOCUMENTATION.md#48-literature-labels-and-price-interpretation-daertrycke-et-al).

---

## Requirements

| Component | Purpose |
|-----------|---------|
| [Julia](https://julialang.org/downloads/) **1.9+** | Runtime (tested on 1.10) |
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

Ensure input data exists under `Data/` and `Input/` (see [DOCUMENTATION.md §9.7](DOCUMENTATION.md#97-weather-and-representative-day-inputs)). Representative days are selected with **RepresentativePeriodsFinder.jl** (hierarchical clustering) from ERA5 weather. To regenerate all scenario inputs:

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
  base_year: 2021

ADMM:
  nScenarioYears: 10    # weather scenarios for market_exposure.jl and me_*.jl
  max_iter: 2000        # iteration budget
  epsilon: 0.1          # convergence tolerance (Boyd-scaled; see §6.5)
  gamma: 1.0            # 1 = risk-neutral; 0.5 = risk-averse (see §4.10)
  beta: 0.95            # CVaR tail level; higher = more risk-averse when gamma<1
```

Add agents by adding blocks under `Power:`, `Hydrogen:`, etc. — no code changes required for supported types. Full parameter reference: [DOCUMENTATION.md §9](DOCUMENTATION.md#9-configuration-reference-datayaml).

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Gurobi license error | Set `GUROBI_HOME`; run `grbgetkey` |
| ADMM does not converge | Run SP first; lower `epsilon` or raise `max_iter`; see [§6.5](DOCUMENTATION.md#65-convergence-tolerances-boyd-style) |
| SP non-optimal | Check `data.yaml` for infeasible capacities or demands |

More detail: [DOCUMENTATION.md](DOCUMENTATION.md) and `ADMM_Convergence.csv` / `ADMM_Diagnostics.csv`.

---

## References

Key sources are listed in [DOCUMENTATION.md §14](DOCUMENTATION.md#14-references), including Boyd et al. (2011) for ADMM, d'Aertrycke et al. (2018) for risk-trading institutions, and Hoschle et al. (2018) for risk-averse equilibrium.

---

## License

Developed for academic research at TU Delft. Contact the author for licensing and collaboration.
