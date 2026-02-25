# Multi-Agent Energy Market Simulation

**Author:** Kian Jafarinejad — PhD Researcher at TU Delft ([K.Jafarinejad@tudelft.nl](mailto:K.Jafarinejad@tudelft.nl))

*A multi-market, multi-agent equilibrium model with endogenous investment, risk aversion, and rigorous convergence diagnostics.*

A multi-agent equilibrium model for coupled **electricity**, **hydrogen**, **green-certificate**, and **end-product** markets, solved via **ADMM** (Alternating Direction Method of Multipliers) with a centralised **social planner** benchmark.

At a high level:

- **Agents** (generators, electrolyzer, offtakers, GC demand) each solve their own optimisation problem.
- **Markets** (power, hydrogen, certificates, end product) are cleared by **prices** updated iteratively by ADMM.
- A **social planner** model with a single welfare-maximising objective provides a rigorous benchmark for the decentralised ADMM solution.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the Model](#running-the-model)
- [Output Files](#output-files)
- [Visualisation & Analysis](#visualisation--analysis)
- [How It Works](#how-it-works)
- [Extending the Model](#extending-the-model)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

This project simulates a multi-agent energy system where independent agents (generators, consumers, hydrogen producers, offtakers) trade across five interconnected markets. The agents are coordinated through ADMM, a distributed optimisation algorithm that iteratively adjusts prices until all markets clear simultaneously.

A centralized **social planner** benchmark maximises total welfare in a single optimization, providing theoretical equilibrium prices and quantities against which the distributed ADMM solution can be compared.

If you are mainly interested in the **mathematical formulation** and algorithmic details, see `DOCUMENTATION.md` (technical documentation). This `README` focuses on **installation**, **configuration**, and **how to run and interpret** the model.

Both the ADMM and social planner use the **same problem definition** from the `Source/` folder. Changing a constraint or objective in one place automatically propagates to both.

---

## Features

- **Five coupled markets**: Electricity, Electricity Guarantees of Origin (GC), Hydrogen, Hydrogen GC, End Product
- **Seven agent types**: VRES generator, conventional generator, elastic consumer, electrolyzer, green offtaker, grey offtaker, EP importer
- **Endogenous capacity investment**: VRES, electrolyzer, and green offtaker decide yearly capacity and investment (MW), with fixed annualised CAPEX proportional to installed capacity.
- **Optional risk aversion (CVaR)**: Those three "green" agents can include a CVaR risk term in their objectives via per-agent `gamma` (risk weight) and `beta` (confidence level); default `gamma = 0.0` keeps them risk-neutral.
- **Distributed ADMM**: Per-agent QP subproblems coordinated by iterative price updates
- **Three-stage adaptive penalty (Boyd rule)**: Market-specific ρ adaptation with (1) normal rp/rd balancing, (2) a gentle push far from tolerance, and (3) a fixed-ρ stability zone near convergence.
- **Centralised benchmark**: Social planner with dual-variable price recovery and the same physical/investment structure as ADMM
- **Green certificate mandate**: 42% H₂ GC requirement for offtakers (configurable)
- **Representative days**: Temporal reduction from 8760 hours to a small set of representative days with weights
- **Quadratic demand**: Elastic electricity and GC demand with inverse-demand parameterisation
- **Comprehensive diagnostics**: Convergence plots, market histories, GC compliance, per-agent quantities, and capacity-investment summaries

---

## Prerequisites

### Julia

- **Julia 1.9+** (tested with 1.10). Download from [julialang.org](https://julialang.org/downloads/).
- Add Julia to your system PATH so that `julia` is available from the command line.

### Gurobi Optimizer

- **Gurobi 10.0+** with a valid license. Academic licenses are free: [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).
- After installing Gurobi, ensure the `GUROBI_HOME` environment variable is set and `grbgetkey` has been used to activate the license.
- The Julia `Gurobi.jl` package will automatically link to the installed Gurobi library.

### Operating System

- Tested on **Windows 10/11**. Should work on macOS and Linux with appropriate path adjustments.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Now
```

Or download and extract the project archive into a folder (e.g. `Now/`).

### Step 2: Install Julia Dependencies

Open a terminal in the project root (`Now/`) and run:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This reads `Project.toml` and `Manifest.toml` to install the exact package versions:

| Package | Purpose |
|---|---|
| `JuMP` | Algebraic modelling for optimization |
| `Gurobi` | QP solver (requires Gurobi installation + license) |
| `DataFrames` | Tabular data manipulation |
| `CSV` | Read/write CSV files |
| `YAML` | Parse `data.yaml` configuration |
| `MathOptInterface` | Solver status constants (e.g. `MOI.OPTIMAL`) |
| `ProgressBars` | ADMM iteration progress bar |
| `TimerOutputs` | Profiling ADMM loop sections |
| `ArgParse` | Command-line argument parsing (future use) |
| `DataStructures` | Additional data structures (available if needed) |

### Step 3: Verify Gurobi

```bash
julia --project=. -e "using Gurobi; env = Gurobi.Env(); println(\"Gurobi OK\")"
```

You should see `Gurobi OK` (plus license information). If this fails, check your Gurobi installation and license.

### Step 4: Verify Input Data

Ensure the following files exist:

```
Data/data.yaml
Input/timeseries_2021.csv
Input/output_2021/decision_variables_short.csv
Input/output_2021/ordering_variable.csv
```

The timeseries CSV must have columns: `Time`, `SOLAR`, `LOAD_E`, `LOAD_H`, `LOAD_EP`, `WIND_ONSHORE` (8760 rows for a full year at hourly resolution).

The representative-days CSV must have columns: `periods`, `weights`, `selected_periods`.

---

## Quick Start

The following minimal sequence should work on a properly configured machine:

1. **Instantiate Julia environment**

   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

2. **Check that Gurobi is available to Julia**

   ```bash
   julia --project=. -e "using Gurobi; Gurobi.Env(); println(\"Gurobi OK\")"
   ```

3. **Run the distributed ADMM simulation**

   ```bash
   julia --project=. market_exposure.jl
   ```

4. **Run the social planner benchmark**

   ```bash
   julia --project=. social_planner.jl
   ```

5. **Optionally generate figures** (after both runs have completed)

   ```bash
   python visualization/visualize_results.py
   ```

For a deeper understanding of what these scripts do internally, see the **How It Works** section below and the full `DOCUMENTATION.md`.

## Project Structure

```
Now/
├── market_exposure.jl          # ADMM distributed simulation (entry point)
├── social_planner.jl           # Centralized benchmark (entry point)
├── Project.toml                # Julia dependencies
├── Manifest.toml               # Dependency lock file
├── DOCUMENTATION.md            # Full technical documentation
├── README.md                   # This file
│
├── Data/
│   └── data.yaml               # All configuration: agents, markets, ADMM
│
├── Input/
│   ├── timeseries_<year>.csv   # Full-year hourly profiles (normalized 0–1)
│   └── output_<year>/
│       ├── decision_variables_short.csv   # Representative days
│       └── ordering_variable.csv          # Ordering matrix
│
├── Source/                     # All problem definition and algorithm logic
│   ├── define_*.jl             # Parameter and market setup
│   ├── build_*.jl              # JuMP model construction (ADMM + planner)
│   ├── solve_*.jl              # Per-iteration objective reset + optimize
│   ├── ADMM.jl                 # Main ADMM coordination loop
│   ├── ADMM_subroutine.jl      # Per-agent ADMM step
│   ├── update_rho.jl           # Adaptive penalty update
│   ├── save_results.jl         # Market-exposure CSV output
│   └── save_social_planner_results.jl  # Planner CSV output
│
├── market_exposure_results/    # Output from market_exposure.jl
└── social_planner_results/     # Output from social_planner.jl
```

---

## Configuration

All configuration is in **`Data/data.yaml`**. The file is fully annotated with inline comments.

### Key Settings

```yaml
General:
  nTimesteps: 24      # Hours per representative day
  nReprDays: 3        # Number of representative days
  nYears: 1           # Number of scenario years
  base_year: 2021     # Calendar year for timeseries

ADMM:
  rho_initial: 1.0    # Starting penalty weight
  max_iter: 10000     # Max ADMM iterations
  epsilon: 0.1        # Convergence tolerance (L2 norm)
```

### Adding a New Agent

1. Add the agent block under the appropriate section in `data.yaml` (e.g. `Power:`, `Hydrogen:`, `Hydrogen_Offtaker:`).
2. Set the `Type` field to one of the supported types (`VRES`, `Conventional`, `Consumer`, `GreenProducer`, `GreenOfftaker`, `GreyOfftaker`, `EPImporter`, `GC_Demand`).
3. Provide all required parameters for that type.
4. Run — no code changes needed. The `define_common_parameters!` function automatically classifies the agent and assigns it to the correct markets.

### Changing Tolerances

The convergence tolerance `epsilon` in the ADMM block controls the base L2 residual threshold for **all five markets** (see `define_results.jl`). Residual norms are L2 over all time slots (hours × representative days × years), so "per-slot" imbalances are smaller than the raw norms might suggest.

---

## Running the Model

### Market Exposure (ADMM)

```bash
julia --project=. market_exposure.jl
```

This will:
1. Load configuration and timeseries.
2. Build per-agent JuMP models.
3. Run the ADMM coordination loop (progress bar shown).
4. Print convergence status, iteration count, and final residuals.
5. Write results to `market_exposure_results/`.

**Typical runtime:** 1–30 minutes depending on `max_iter`, `epsilon`, and system hardware.

### Social Planner (Benchmark)

```bash
julia --project=. social_planner.jl
```

This will:
1. Load the same configuration and timeseries.
2. Build a single centralised QP model.
3. Solve it (usually < 1 second).
4. Write results to `social_planner_results/`.

### Comparing Results

After running both scripts, compare `market_exposure_results/Agent_Quantities_Final.csv` against `social_planner_results/Agent_Summary.csv` to verify that the ADMM solution converges toward the social planner benchmark.

---

## Output Files

### Market Exposure (`market_exposure_results/`)

| File | Description |
|---|---|
| `ADMM_Convergence.csv` | Primal and dual residuals per iteration for all 5 markets |
| `ADMM_Diagnostics.csv` | ρ, mean price, mean imbalance per iteration for all markets |
| `Electricity_Market_History.csv` | Per-iteration history for the electricity market |
| `Hydrogen_Market_History.csv` | Per-iteration history for the hydrogen market |
| `Electricity_GC_Market_History.csv` | Per-iteration history for the elec-GC market |
| `H2_GC_Market_History.csv` | Per-iteration history for the H₂-GC market |
| `End_Product_Market_History.csv` | Per-iteration history for the EP market |
| `Agent_Summary.csv` | Agent group membership and per-agent objective value at the final iteration |
| `Agent_Quantities_Final.csv` | Final-iteration net quantities per agent per market (sums over all (h,d,y)) |
| `Offtaker_GC_Diagnostics.csv` | Green-certificate compliance per offtaker |
| `H2_Producer_Diagnostics.csv` | GC-to-production ratio for electrolyzers |
| `Capacity_Investments.csv` | For VRES, electrolyzer, and green offtaker: per-year installed capacity and investment (MW) at the final ADMM iteration |
| `TimerOutput.yaml` | Profiling data (time per ADMM section) |

### Social Planner (`social_planner_results/`)

| File | Description |
|---|---|
| `Market_Prices.csv` | Equilibrium prices (duals of balance constraints) |
| `Agent_Summary.csv` | Per-agent total quantity and ADMM-style objective value (cost − revenue) evaluated at planner prices |
| `Capacity_Investments_Planner.csv` | For VRES, electrolyzer, and green offtaker: per-year installed capacity and investment (MW) in the planner solution |

---

## Visualisation & Analysis

The folder `visualization/` contains a Python script for **side-by-side comparison** of the ADMM market-exposure results and the social planner benchmark:

- **File**: `visualization/visualize_results.py`
- **Inputs** (expected to exist after running both Julia scripts):
  - `social_planner_results/Market_Prices.csv`
  - `social_planner_results/Agent_Summary.csv`
  - `market_exposure_results/Market_Prices.csv`
  - `market_exposure_results/Agent_Quantities_Final.csv`
  - Other diagnostic CSVs in `market_exposure_results/`
- **Outputs**:
  - Publication-ready figures in `visualization/figures/` (`.png` and `.pdf`), including:
    - Price differences (social planner vs ADMM) per market
    - Quantity comparison per agent
    - Price evolution over time
    - ADMM convergence plots (primal/dual residuals)
    - Market history (price and imbalance)
    - Price heatmaps (hour-of-day vs representative day)

### Python environment

The script uses standard scientific Python packages:

- `pandas`
- `numpy`
- `matplotlib`

You can install them, for example, via:

```bash
pip install pandas numpy matplotlib
```

Then run:

```bash
python visualization/visualize_results.py
```

For notebook-style use, you can also open the script in an IDE and execute the `# %%` cells interactively.

---

## How It Works

### ADMM (Alternating Direction Method of Multipliers)

Each iteration:

1. **Agent solves**: Each agent independently minimises its cost minus revenue plus an ADMM quadratic penalty that pushes its net position toward a market consensus target.
2. **Imbalance**: Sum of all agents' net positions in each market. Should be zero at equilibrium.
3. **Price update**: prices are updated in the **direction of imbalance**. In compact form for a given market \(k\),

   \[
   \lambda_{k}^{(t+1)} = \lambda_{k}^{(t)} - \eta_k^{(t)} \,\rho_k^{(t)} \, r_k^{(t)},
   \]

   where \(r_k^{(t)}\) is the current imbalance (sum of net positions), \(\rho_k^{(t)}\) is the penalty weight, and \(\eta_k^{(t)} \in (0,1]\) is a residual-aware step size. **Excess supply** (\(r_k>0\)) reduces prices; **excess demand** (\(r_k<0\)) increases prices.
4. **Penalty adaptation**: ρ increases if the market is far from clearing (primal residual too large), decreases if agents are oscillating (dual residual too large).
5. **Convergence**: Stops when all markets have both primal and dual residuals below tolerance.

### Social Planner

Solves a single QP maximising total welfare = Σ(consumer utility − producer costs), subject to:
- All individual agent constraints (capacity, conversion, GC mandates).
- Market-clearing constraints (supply = demand in each market).

Equilibrium prices are the **dual variables** (shadow prices) of the balance constraints.

### Market Coupling

```
Electricity Market  ←──  VRES, Conventional, Consumer, Electrolyzer
        │
        ├──→  Elec GC Market  ←──  VRES, Electrolyzer, GC Demand
        │
        └──→  Electrolyzer  ──→  H₂ Market  ←──  Green Offtaker
                    │
                    └──→  H₂ GC Market  ←──  Green Offtaker, Grey Offtaker
                                │
                                └──→  EP Market  ←──  Green, Grey, Importer  ──→  D_EP
```

---

## Extending the Model

### Adding a New Agent Type

1. Create `Source/define_<type>_parameters.jl` to populate `mod.ext[:parameters]`.
2. Create `Source/build_<type>_agent.jl` with:
   - `build_<type>_agent!()` for ADMM (with λ/ρ/ḡ penalty terms).
   - `add_<type>_agent_to_planner!()` for social planner (same constraints, welfare expression, no ADMM terms).
3. Create `Source/solve_<type>_agent.jl` to rebuild the objective each ADMM iteration.
4. Add the agent section in `data.yaml`.
5. Update `market_exposure.jl` and `social_planner.jl` to include the new files and iterate over the new agent group.
6. Update `define_common_parameters.jl` to map the new type to the appropriate markets.

### Adding a New Market

1. Create `Source/define_<market>_market_parameters.jl`.
2. Add `λ_<market>`, `g_bar_<market>`, `ρ_<market>` to the ADMM placeholder arrays in `define_common_parameters.jl`.
3. Add the new market to `define_results.jl` (results["λ"], ADMM ρ, Imbalances, Residuals, Tolerance).
4. Add the imbalance computation, residual computation, price update, and convergence check for the new market in `ADMM.jl`.
5. Add the market-clearing constraint in `build_social_planner.jl`.

### Changing Objectives or Constraints

Simply edit the relevant `build_*_agent.jl` file. Both the ADMM version (`build_*_agent!`) and the planner version (`add_*_to_planner!`) live in the same file, so you can update them together and ensure consistency.

---

## Troubleshooting

### "No Gurobi license found"

- Ensure `GUROBI_HOME` is set correctly.
- Run `grbgetkey <your-key>` to activate the license.
- Academic licenses require renewal annually.

### ADMM Does Not Converge

- **Increase `max_iter`** in `data.yaml`.
- **Increase `epsilon`** (looser tolerance) to check if the algorithm is close.
- **Check `ADMM_Diagnostics.csv`**: if prices oscillate, the ρ adaptation may need tuning. Reduce `ρ_max` or use gentler factors for the oscillating market in `update_rho.jl`.
- **Check residuals**: if primal residuals plateau above tolerance, the L2 norm may be hitting an "error floor" due to the representative-day discretisation. The solution may still be economically meaningful (compare against the social planner).

### Social Planner Warns "Non-Optimal Status"

- Check if the model is infeasible (conflicting constraints or insufficient capacity).
- Check `data.yaml` for misconfigured parameters (e.g. Total_Demand too high for available capacity).

### CSV Parsing Warnings

The "parsed expected N columns" warning from CSV.jl is harmless — it indicates the ordering_variable CSV has trailing data that is automatically handled.

---

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Foundations and Trends in Machine Learning, 3(1), 1–122.
- Dunning, I., Huchette, J., & Lubin, M. (2017). *JuMP: A Modeling Language for Mathematical Optimization*. SIAM Review, 59(2), 295–320.

---

## License

This project is developed for academic research at TU Delft. Contact the author for licensing and collaboration inquiries.
