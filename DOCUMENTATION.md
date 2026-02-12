# Multi-Agent Energy Market Simulation — Technical Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Markets](#2-markets)
3. [Agents](#3-agents)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [ADMM Algorithm](#5-admm-algorithm)
6. [Social Planner Benchmark](#6-social-planner-benchmark)
7. [Data and Indexing](#7-data-and-indexing)
8. [Configuration Reference (data.yaml)](#8-configuration-reference-datayaml)
9. [Project Structure](#9-project-structure)
10. [File Reference](#10-file-reference)
11. [Output Files](#11-output-files)
12. [Code Conventions](#12-code-conventions)

---

## 1. Overview

This project implements a **multi-agent equilibrium model** for coupled electricity, hydrogen, green-certificate, and end-product markets, coordinated via **ADMM** (Alternating Direction Method of Multipliers). Each agent has its own JuMP optimization model; market-clearing is achieved by iteratively updating prices and penalty terms so that supply and demand balance in each market.

The project includes two entry points:

- **`market_exposure.jl`** — Distributed ADMM simulation where agents optimise independently and are coordinated through iterative price signals.
- **`social_planner.jl`** — Centralised welfare-maximisation benchmark where all agents are optimised jointly in a single model. Equilibrium prices emerge as dual variables of market-clearing constraints.

Both scripts share the **same** problem definition (objectives, constraints, variables) from `Source/`. If you change a constraint or objective in a `build_*` file, the change automatically propagates to both the market exposure and the social planner.

---

## 2. Markets

The model contains five interconnected markets:

| Market | Key | Description | Unit | Supply Side | Demand Side |
|---|---|---|---|---|---|
| **Electricity** | `elec` | Physical power exchange | MWh | VRES generator, conventional generator | Consumer, electrolyzer |
| **Electricity GC** | `elec_GC` | Guarantees of Origin (1:1 with renewable MWh) | MWh_GC | VRES generator | Electrolyzer, GC demand agent |
| **Hydrogen** | `H2` | Physical hydrogen exchange | MWh_H2 | Electrolyzer | Green offtaker |
| **Hydrogen GC** | `H2_GC` | H₂ green certificates (from certified electricity) | MWh_H2 | Electrolyzer | Green offtaker, grey offtaker |
| **End Product** | `EP` | Ammonia / downstream product | MWh_EP | Green offtaker, grey offtaker, EP importer | Fixed demand `D_EP` |

### Market coupling

The markets are coupled through the **electrolyzer**, which sits at the nexus:

- It **buys** electricity (elec market) and electricity GCs (elec_GC market).
- It **sells** hydrogen (H2 market) and hydrogen GCs (H2_GC market).
- The conversion constraint `h2_out = η × e_in` links the electricity and hydrogen markets.
- The annual green-backing constraint links the elec_GC and H2_GC markets.

The **end-product market** is coupled to H2 and H2_GC through the offtakers, who convert hydrogen into the end product and must comply with the GC mandate.

---

## 3. Agents

### 3.1 Power-Sector Agents

| Agent | Type | Description |
|---|---|---|
| `Gen_VRES_01` | `VRES` | Variable renewable (e.g. solar). Zero marginal cost. Produces both electricity and elec GCs (1:1). Constrained by hourly availability factor × capacity. |
| `Gen_Conv_01` | `Conventional` | Dispatchable thermal plant. Constant availability (AF = 1). Marginal cost sets price floor. No GC production. |
| `Cons_Elec_01` | `Consumer` | Elastic electricity demand. Quadratic utility `U(d) = A_E·d − ½B_E·d²` gives inverse demand `p(d) = A_E − B_E·d`. Bounded by `PeakLoad × load_profile`. |

### 3.2 Hydrogen-Sector Agent

| Agent | Type | Description |
|---|---|---|
| `Prod_H2_Green` | `GreenProducer` | PEM electrolyzer. Converts electricity to H₂ with efficiency `η = 1/SpecificConsumption`. Buys elec + elec GCs; sells H₂ + H₂ GCs. Annual green-backing constraint ensures GCs purchased ≥ `(1/η) × GCs issued`. |

### 3.3 Offtaker Agents

| Agent | Type | Description |
|---|---|---|
| `Offtaker_Green` | `GreenOfftaker` | Buys green H₂ and converts it 1:1 (via `Alpha`) to end product. Must buy H₂ GCs for ≥ 42% of EP output (annual mandate `gamma_GC = 0.42`). Tight stoichiometric link: `ep = (1/α) × h2_in`. |
| `Offtaker_Grey` | `GreyOfftaker` | Produces EP from conventional (grey) feedstock at `MarginalCost`. Must buy H₂ GCs for ≥ `gamma_GC × gamma_NH3 × ep` (only the H₂-feedstock fraction). |
| `Offtaker_Import` | `EPImporter` | Imports EP from outside the system at `ImportCost`. No H₂ or GC involvement. Acts as a price cap on the EP market. |

### 3.4 Electricity GC Demand Agent

| Agent | Type | Description |
|---|---|---|
| `Demand_GC_Elec_01` | `GC_Demand` | Elastic demand for electricity GCs. Quadratic utility `U(d) = A_GC·d − ½B_GC·d²`. Bounded by `PeakLoad × load_profile`. |

### 3.5 EP Demand Agent (Placeholder)

Currently empty (`EP_Demand: {}`). EP demand is inelastic and fully defined by `EP_market.Total_Demand × normalized_profile`. The block is a placeholder for future elastic EP demand agents.

---

## 4. Mathematical Formulation

### 4.1 Agent Objectives (ADMM)

Each agent minimises its **augmented Lagrangian**:

```
min  Σ_{h,d,y} W[d,y] × ( cost_i − revenue_i )
   + Σ_k  (ρ_k / 2) × Σ_{h,d,y} W[d,y] × ( g_i^k − ḡ_i^k )²
```

where:
- `cost_i − revenue_i` is the agent's private cost minus revenue across all markets.
- `g_i^k` is the agent's net position in market `k` (positive = supply, negative = demand).
- `ḡ_i^k` is the consensus target for agent `i` in market `k`.
- `ρ_k` is the penalty weight for market `k`.
- `W[d,y]` scales representative days to a full year.

#### Specific objective terms by agent type

**VRES generator:**
```
min Σ W × ( MC×g − λ_elec×g − λ_GC×g )  +  (ρ_elec/2)×Σ W×(g − ḡ_elec)²  +  (ρ_GC/2)×Σ W×(g − ḡ_GC)²
```

**Conventional generator:**
```
min Σ W × ( MC×g − λ_elec×g )  +  (ρ_elec/2)×Σ W×(g − ḡ_elec)²
```

**Consumer:**
```
min Σ W × ( λ_elec×d − U(d) )  +  (ρ_elec/2)×Σ W×(−d − ḡ_elec)²
where U(d) = A_E×d − (B_E/2)×d²
```

**Electrolyzer:**
```
min Σ W × ( λ_elec×e_in + λ_GC×gc_e + op_cost×h2 − λ_H2×h2 − λ_H2GC×gc_h2 )
  + (ρ_elec/2)×Σ W×(−e_in − ḡ_elec)²
  + (ρ_GC/2)×Σ W×(−gc_e − ḡ_GC)²
  + (ρ_H2/2)×Σ W×(h2 − ḡ_H2)²
  + (ρ_H2GC/2)×Σ W×(gc_h2 − ḡ_H2GC)²
```

### 4.2 Key Constraints

| Constraint | Equation | Scope | Rationale |
|---|---|---|---|
| VRES capacity | `g ≤ AF × Capacity` | Per (h,d,y) | Generation limited by resource availability |
| Conventional capacity | `g ≤ Capacity` | Per (h,d,y) | Generation limited by installed capacity |
| Consumer load | `d ≤ PeakLoad × load_profile` | Per (h,d,y) | Maximum consumption bound |
| H₂ conversion | `h2_out = η × e_in` | Per (h,d,y) | Stoichiometric mass/energy balance |
| H₂ GC physical limit | `gc_h2 ≤ h2_out` | Per (h,d,y) | Cannot certify more than produced |
| Green-backing (annual) | `Σ W×gc_elec ≥ (1/η)×Σ W×gc_h2` | Per year | Temporal flexibility in GC procurement |
| Green offtaker stoichiometry | `ep = (1/α) × h2_in` | Per (h,d,y) | No H₂ waste; tight conversion |
| GC mandate (green/grey) | `Σ W×gc_h2 ≥ γ_GC × Σ W×ep` | Per year | 42% renewable mandate |
| Grey GC mandate | `Σ W×gc_h2 ≥ γ_GC × γ_NH3 × Σ W×ep` | Per year | Only H₂-feedstock fraction |

### 4.3 Social Planner Objective

The social planner maximises total welfare:

```
max Σ_i  welfare_i
```

where `welfare_i` is:
- **Consumers**: `U(d) = A×d − (B/2)×d²` (utility)
- **Generators**: `−MC × g` (negative production cost)
- **Electrolyzer**: `−op_cost × h2_out` (negative operational cost)
- **Offtakers**: `−processing_cost × ep` (negative processing/import cost)

Revenue/expenditure terms cancel out in the aggregate (they are transfers between agents). Market-clearing constraints enforce supply = demand.

---

## 5. ADMM Algorithm

### 5.1 Iteration Structure

Each ADMM iteration `k` proceeds as follows:

1. **For each agent** (via `ADMM_subroutine!`):
   a. Update consensus target: `ḡ_i = q_i^{k-1} − (1/(n+1)) × imbalance^{k-1}`
   b. Update prices `λ`, penalty `ρ` from the global ADMM state.
   c. Rebuild objective with updated parameters.
   d. Solve the agent's QP.
   e. Record the solution quantities.

2. **Compute market imbalances**: For each market, sum all agents' net positions. For EP, subtract fixed demand `D_EP`.

3. **Compute residuals**:
   - **Primal residual** = `‖imbalance‖₂` (L2 norm; measures market-clearing violation).
   - **Dual residual** = `‖ρ × Δ(consensus deviation)‖₂` (measures position stability).

4. **Update prices**: `λ^{k+1} = λ^k − ρ × imbalance^k` (gradient ascent on the dual).

5. **Update ρ** (Boyd rule): For each market independently:
   - If `primal > 2 × dual` → increase `ρ` (under-penalised).
   - If `dual > 2 × primal` → decrease `ρ` (over-penalised).
   - Otherwise → keep `ρ` unchanged.

6. **Convergence check**: All five markets must have both primal and dual residuals below their tolerance.

### 5.2 Consensus Formula (Sharing ADMM)

The consensus target for agent `i` in a market with `n` participants:

```
ḡ_i^k = q_i^{k-1} − (1/(n+1)) × Σ_j q_j^{k-1}
```

The `(n+1)` denominator comes from the sharing ADMM formulation, which introduces one "market copy" alongside the `n` agent copies. This distributes the imbalance correction equally.

### 5.3 Adaptive Penalty (ρ)

| Market | Increase factor | Decrease factor | ρ_max | Rationale |
|---|---|---|---|---|
| `elec`, `elec_GC` | 1.10 | 1/1.10 | 100,000 | Loosely coupled; aggressive adaptation is safe |
| `H2`, `H2_GC`, `EP` | 1.01 | 1/1.01 | 1.0 | Tightly coupled hydrogen chain; gentle updates prevent oscillation |

### 5.4 Convergence Tolerances

| Market | Tolerance | Rationale |
|---|---|---|
| `elec`, `elec_GC` | `epsilon` (base) | Electricity markets are large and liquid |
| `H2`, `H2_GC`, `EP` | `10 × epsilon` | Stiffer numerical behaviour; looser tolerance prevents chasing extremely small residuals |

### 5.5 Sign Convention

| Role | Net position sign | Example |
|---|---|---|
| Supplier / seller | **Positive** | VRES generation `+g`, H₂ sales `+h2_out` |
| Buyer / consumer | **Negative** | Electricity demand `−d`, H₂ purchase `−h2_in` |

Market imbalance = Σ (net positions). Positive imbalance = excess supply → price decreases. Negative imbalance = excess demand → price increases.

---

## 6. Social Planner Benchmark

The social planner (`social_planner.jl`) solves a single centralised QP that maximises total welfare subject to all individual agent constraints plus market-clearing balance constraints. It serves as the theoretical first-best benchmark.

### 6.1 Market-Clearing Constraints

| Constraint | Equation |
|---|---|
| Electricity balance | `Σ generation − Σ demand − Σ electrolyzer_elec_buy = 0` (per h,d,y) |
| Elec GC balance | `Σ VRES_generation − Σ electrolyzer_GC_buy − Σ GC_demand = 0` (per h,d,y) |
| H₂ balance | `Σ H₂_production − Σ H₂_consumption − Σ offtaker_H₂_buy = 0` (per h,d,y) |
| H₂ GC balance | `Σ W×H₂_GC_supply − Σ W×H₂_GC_demand = 0` (per year, annual) |
| EP balance | `Σ offtaker_EP_supply − D_EP − Σ EP_demand = 0` (per h,d,y) |

### 6.2 Price Recovery

Equilibrium prices are extracted as **dual variables** (shadow prices) of the market-clearing constraints. By LP/QP duality, the dual of a balance constraint equals the equilibrium price at that timestep.

### 6.3 Code Architecture

All problem definition lives in `Source/build_*.jl` files. Each file contains:

- `build_*_agent!()` — Builds the ADMM version (with `λ`, `ρ`, `ḡ` penalty terms).
- `add_*_agent_to_planner!()` — Adds the same variables/constraints to the planner model **without** ADMM terms. Returns the agent's welfare contribution.

`build_social_planner.jl` orchestrates the calls to all `add_*_to_planner!` functions, adds market-clearing constraints, and sets the Max(total welfare) objective.

---

## 7. Data and Indexing

### 7.1 Temporal Dimensions

| Dimension | Set | Size | Description |
|---|---|---|---|
| Hours | `JH = 1:nTimesteps` | 24 | Hours within each representative day |
| Representative days | `JD = 1:nReprDays` | 3 | Representative days (clustered from 365) |
| Years | `JY = 1:nYears` | 1 | Scenario years |

### 7.2 Representative-Day Weights

`W[jd, jy]` = number of real calendar days that representative day `jd` stands for in year `jy`. Used to scale per-representative-day objective values to a full-year total.

### 7.3 Years Mapping

`years = Dict(1 => 2021, 2 => 2022, ...)` maps scenario index to calendar year. This bridges the model's integer indices (`JY`) with the timeseries/representative-day CSVs that are keyed by calendar year.

### 7.4 3D Arrays

All prices, quantities, and imbalances are stored as 3D arrays `[jh, jd, jy]`. Scalar diagnostics (mean price, mean imbalance) are computed per iteration for CSV output.

---

## 8. Configuration Reference (data.yaml)

### 8.1 General

| Parameter | Value | Description |
|---|---|---|
| `nTimesteps` | 24 | Hours per representative day (hourly resolution) |
| `nReprDays` | 3 | Representative days (trade-off: speed vs. accuracy) |
| `nYears` | 1 | Single-year snapshot |
| `base_year` | 2021 | Calendar year for timeseries data |

### 8.2 ADMM

| Parameter | Value | Description |
|---|---|---|
| `rho_initial` | 1.0 | Default penalty weight (neutral starting point) |
| `max_iter` | 10,000 | Maximum ADMM iterations |
| `epsilon` | 0.1 | Convergence tolerance (L2 residual norm) |

### 8.3 Market Parameters

| Market | `initial_price` | `rho_initial` | Notes |
|---|---|---|---|
| `elec_market` | 50.0 €/MWh | 1.0 | Typical EU wholesale price |
| `elec_GC_market` | 5.0 €/MWh_GC | 0.3 | Modest GC premium |
| `H2_market` | 0.0 €/MWh_H2 | 0.5 | Internal transfer good; ADMM discovers price |
| `H2_GC_market` | 50.0 €/MWh_GC | 0.3 | Renewable premium |
| `EP_market` | 700.0 €/t_EP | 3.0 | Ammonia market level; also has `Demand_Column`, `Total_Demand` |

### 8.4 Agent Parameters

See `Data/data.yaml` for the full annotated configuration. Key parameters:

- **VRES**: `Capacity`, `Profile_Column`, `MarginalCost`
- **Conventional**: `Capacity`, `MarginalCost`
- **Consumer**: `PeakLoad`, `Load_Column`, `A_E`, `B_E` (quadratic utility)
- **Electrolyzer**: `Capacity_Electrolyzer`, `Capacity_H2_Output`, `SpecificConsumption`, `OperationalCost`
- **Green offtaker**: `Capacity_H2_In`, `Capacity_EP_Out`, `Alpha`, `ProcessingCost`
- **Grey offtaker**: `Capacity`, `MarginalCost`, `gamma_NH3`
- **EP importer**: `Capacity`, `ImportCost`
- **GC demand**: `PeakLoad`, `Load_Column`, `A_GC`, `B_GC`

---

## 9. Project Structure

```
Now/
├── market_exposure.jl          # Entry point: distributed ADMM simulation
├── social_planner.jl           # Entry point: centralized benchmark
├── Project.toml                # Julia project dependencies
├── Manifest.toml               # Julia dependency lock file
├── DOCUMENTATION.md            # This file
├── README.md                   # Quick-start guide (installation, running)
│
├── Data/
│   └── data.yaml               # All configuration: agents, markets, ADMM settings
│
├── Input/
│   ├── timeseries_2021.csv     # Full-year hourly profiles (SOLAR, LOAD_E, LOAD_H, LOAD_EP, WIND_ONSHORE)
│   ├── timeseries_2022.csv     # (one per year; columns are normalized 0–1 profiles)
│   ├── ...
│   ├── output_2021/
│   │   ├── decision_variables_short.csv   # Representative days: periods, weights, selected_periods
│   │   └── ordering_variable.csv          # Ordering matrix (for upstream representative-day selection)
│   ├── output_2022/
│   │   └── ...
│   └── ...
│
├── Source/
│   ├── define_common_parameters.jl       # Sets, weights, market flags, ADMM placeholders
│   ├── define_power_parameters.jl        # VRES / Conventional / Consumer parameters
│   ├── define_H2_parameters.jl           # Electrolyzer parameters
│   ├── define_offtaker_parameters.jl     # Offtaker parameters (green, grey, importer)
│   ├── define_elec_GC_demand_parameters.jl  # GC demand parameters
│   ├── define_EP_demand_parameters.jl    # Placeholder for elastic EP demand
│   │
│   ├── define_electricity_market_parameters.jl    # Electricity market setup
│   ├── define_H2_market_parameters.jl             # H₂ market setup
│   ├── define_electricity_GC_market_parameters.jl # Elec GC market setup
│   ├── define_H2_GC_market_parameters.jl          # H₂ GC market setup
│   ├── define_EP_market_parameters.jl             # EP market setup + D_EP demand profile
│   │
│   ├── build_power_agent.jl          # JuMP model: power agents (ADMM + planner)
│   ├── build_H2_agent.jl             # JuMP model: electrolyzer (ADMM + planner)
│   ├── build_offtaker_agent.jl       # JuMP model: offtakers (ADMM + planner)
│   ├── build_elec_GC_demand_agent.jl # JuMP model: GC demand (ADMM + planner)
│   ├── build_EP_demand_agent.jl      # JuMP model: EP demand placeholder (ADMM + planner)
│   ├── build_social_planner.jl       # Orchestrate planner: call add_*_to_planner!, add balance constraints, set objective
│   │
│   ├── solve_power_agent.jl          # Re-set objective & optimize (power)
│   ├── solve_H2_agent.jl             # Re-set objective & optimize (electrolyzer)
│   ├── solve_offtaker_agent.jl       # Re-set objective & optimize (offtakers)
│   ├── solve_elec_GC_demand_agent.jl # Re-set objective & optimize (GC demand)
│   ├── solve_EP_demand_agent.jl      # Placeholder (EP demand)
│   │
│   ├── define_results.jl             # Initialize result & ADMM state dictionaries
│   ├── ADMM.jl                       # Main ADMM coordination loop
│   ├── ADMM_subroutine.jl            # Per-agent step: update params, solve, record
│   ├── update_rho.jl                 # Adaptive penalty update (Boyd rule)
│   ├── save_results.jl               # Write market-exposure CSV outputs
│   └── save_social_planner_results.jl # Write social-planner CSV outputs
│
├── market_exposure_results/          # Output from market_exposure.jl
│   ├── ADMM_Convergence.csv          # Primal & dual residuals per iteration
│   ├── ADMM_Diagnostics.csv          # ρ, mean price, mean imbalance per iteration
│   ├── Electricity_Market_History.csv
│   ├── Hydrogen_Market_History.csv
│   ├── Electricity_GC_Market_History.csv
│   ├── H2_GC_Market_History.csv
│   ├── End_Product_Market_History.csv
│   ├── Agent_Summary.csv             # Agent group membership
│   ├── Agent_Quantities_Final.csv    # Final-iteration net quantities per agent
│   ├── Offtaker_GC_Diagnostics.csv   # GC compliance per offtaker
│   ├── H2_Producer_Diagnostics.csv   # H₂ GC-to-production ratio
│   └── TimerOutput.yaml              # Profiling data
│
└── social_planner_results/           # Output from social_planner.jl
    ├── Market_Prices.csv             # Equilibrium prices (duals of balance constraints)
    └── Agent_Summary.csv             # Per-agent quantity & welfare contribution
```

---

## 10. File Reference

### 10.1 Runner Scripts

| File | Purpose |
|---|---|
| `market_exposure.jl` | Entry point for distributed ADMM. Sections 1–13: env, packages, dirs, source loading, data loading, results folder, agent init, market params, agent params, build models, run ADMM, save results. |
| `social_planner.jl` | Entry point for centralised benchmark. Sections 1–12: same structure as market_exposure but builds a single planner model instead of per-agent models + ADMM loop. |

### 10.2 Parameter Definition Files

| File | Role |
|---|---|
| `define_common_parameters.jl` | Creates `mod.ext` dictionaries (sets, parameters, timeseries, variables, constraints, expressions). Fills JH/JD/JY, W, P, γ, β. Determines market participation from agent type. Pre-allocates ADMM placeholder arrays. |
| `define_power_parameters.jl` | VRES: capacity, AF profile. Conventional: capacity, constant AF=1. Consumer: PeakLoad, LOAD_E profile, A_E, B_E. |
| `define_H2_parameters.jl` | Electrolyzer: Capacity_Electrolyzer, Capacity_H2_Output, SpecificConsumption, OperationalCost, η_elec_H2. |
| `define_offtaker_parameters.jl` | Copies all keys from agent block; sets gamma_GC = 0.42 (regulatory mandate). |
| `define_elec_GC_demand_parameters.jl` | PeakLoad, Load_Column, A_GC, B_GC, LOAD_GC timeseries. |
| `define_EP_demand_parameters.jl` | Placeholder; copies EP_Demand block if present. |
| `define_*_market_parameters.jl` | Each market: name, initial_price, rho_initial, prices list. EP market also builds 3D demand array D_EP. |
| `define_results.jl` | Initialises results["λ"], per-agent quantity buffers, ADMM ρ lists, Imbalances, PriceHistory, ImbalanceMean, Residuals, Tolerance. |

### 10.3 Model Building Files

| File | ADMM Function | Planner Function |
|---|---|---|
| `build_power_agent.jl` | `build_power_agent!()` — variables, constraints, objective with λ/ρ/ḡ | `add_power_agent_to_planner!()` — same constraints, welfare expression (no ADMM terms) |
| `build_H2_agent.jl` | `build_H2_agent!()` — electrolyzer with 4-market ADMM terms | `add_H2_agent_to_planner!()` — same constraints, welfare = −op_cost |
| `build_offtaker_agent.jl` | `build_offtaker_agent!()` — green/grey/importer with ADMM terms | `add_offtaker_agent_to_planner!()` — same constraints, welfare = −processing_cost |
| `build_elec_GC_demand_agent.jl` | `build_elec_GC_demand_agent!()` — GC demand with ADMM | `add_elec_GC_demand_agent_to_planner!()` — utility expression |
| `build_EP_demand_agent.jl` | `build_EP_demand_agent!()` — placeholder | `add_EP_demand_agent_to_planner!()` — placeholder |
| `build_social_planner.jl` | — | `build_social_planner!()` — orchestrates all add_*_to_planner!, adds balance constraints, sets Max(welfare) |

### 10.4 Solve Files

| File | Role |
|---|---|
| `solve_power_agent.jl` | Rebuilds objective with current λ, ḡ, ρ for VRES/Conventional/Consumer; calls `optimize!`. |
| `solve_H2_agent.jl` | Rebuilds 5-term economic objective + 4 ADMM penalties; calls `optimize!`. |
| `solve_offtaker_agent.jl` | Rebuilds objective for green/grey/importer; calls `optimize!`. |
| `solve_elec_GC_demand_agent.jl` | Rebuilds utility − expenditure + ADMM penalty; calls `optimize!`. |
| `solve_EP_demand_agent.jl` | Placeholder; just calls `optimize!`. |

### 10.5 ADMM Files

| File | Role |
|---|---|
| `ADMM.jl` | Main loop: iterate agents → imbalances → primal residuals → dual residuals → price update → ρ update → convergence check. Progress bar. Summary printout. |
| `ADMM_subroutine.jl` | Per-agent step: update g_bar/λ/ρ on model → dispatch to solve_* → extract & record quantities. H₂-GC prices averaged to annual scalars. |
| `update_rho.jl` | Boyd rule: per-market adaptive ρ update with market-specific factors and caps. |

### 10.6 Save Files

| File | Role |
|---|---|
| `save_results.jl` | Writes: ADMM_Convergence.csv, ADMM_Diagnostics.csv, per-market history CSVs, Agent_Summary.csv, Agent_Quantities_Final.csv, Offtaker_GC_Diagnostics.csv, H2_Producer_Diagnostics.csv. |
| `save_social_planner_results.jl` | Writes: Market_Prices.csv (duals), Agent_Summary.csv (quantities + welfare). |

---

## 11. Output Files

### 11.1 Market Exposure Results

| File | Contents |
|---|---|
| `ADMM_Convergence.csv` | Columns: `iter`, `{market}_primal`, `{market}_dual` for each of the 5 markets. One row per ADMM iteration. Used for convergence plots. |
| `ADMM_Diagnostics.csv` | Columns: `iter`, `{market}_rho`, `{market}_price_mean`, `{market}_imb_mean` for each market. Used for price/imbalance evolution analysis. |
| `{Market}_Market_History.csv` | Per-market CSV with: `iter`, `rho`, `price_mean`, `imb_mean`, `primal_res`, `dual_res`. |
| `Agent_Summary.csv` | Columns: `AgentID`, `Group`. Group membership table. |
| `Agent_Quantities_Final.csv` | Columns: `AgentID`, `Group`, `elec_net_sum`, `H2_net_sum`, `elec_GC_net_sum`, `H2_GC_net_sum`, `EP_net_sum`. Sum of final-iteration 3D quantities. |
| `Offtaker_GC_Diagnostics.csv` | Columns: `AgentID`, `Type`, `EP_total`, `H2_in_total`, `H2_GC_total`, `GC_share`, `GC_mandate`, `GC_slack`. |
| `H2_Producer_Diagnostics.csv` | Columns: `AgentID`, `H2_total`, `H2_GC_total`, `GC_per_H2`. |
| `TimerOutput.yaml` | Profiling: time spent in imbalances, residuals, price updates, solve, etc. |

### 11.2 Social Planner Results

| File | Contents |
|---|---|
| `Market_Prices.csv` | Columns: `Time`, `Elec_Price`, `H2_Price`. One row per (jy, jd, jh) timestep. Prices = duals of balance constraints. |
| `Agent_Summary.csv` | Columns: `Agent`, `Type`, `Total_Quantity`, `Welfare_Contribution`. |

---

## 12. Code Conventions

### 12.1 JuMP Model Storage

Each agent's JuMP model uses `mod.ext` dictionaries:
- `mod.ext[:sets]` — Index ranges (JH, JD, JY).
- `mod.ext[:parameters]` — Scalars and arrays (capacities, costs, ADMM λ/ḡ/ρ).
- `mod.ext[:timeseries]` — 3D hourly profiles (AF, LOAD_E, etc.).
- `mod.ext[:variables]` — JuMP decision variables.
- `mod.ext[:constraints]` — JuMP constraints.
- `mod.ext[:expressions]` — JuMP expressions (net positions, objective terms).

### 12.2 Anonymous Variables (Planner)

In the social planner, all variables use anonymous JuMP syntax with `base_name` to avoid naming conflicts when multiple agents share the same planner model:
```julia
q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")
```

### 12.3 Commenting Standard

Every `.jl` file follows this standard:
- **File header**: Purpose, arguments, side effects, context.
- **Section dividers**: `# ── Section Name ──` or `# ---` blocks.
- **Per-line/block comments**: Every non-trivial line explains WHAT it does and WHY.
- **Mathematical formulas**: Objectives and constraints are documented with their full mathematical form in comments above the code.

### 12.4 Data Flow

```
data.yaml  ──→  define_common_parameters!  ──→  mod.ext[:parameters]
               define_*_parameters!              mod.ext[:timeseries]
                                                 mod.ext[:sets]
                        │
                        ▼
               build_*_agent!  ──→  mod.ext[:variables]
                                     mod.ext[:constraints]
                                     mod.ext[:expressions]
                        │
                        ▼
               ┌─────────────────────────────────┐
               │  ADMM loop (market_exposure.jl)  │
               │  or                              │
               │  build_social_planner!           │
               │  (social_planner.jl)             │
               └─────────────────────────────────┘
                        │
                        ▼
               save_results / save_social_planner_results!
                        │
                        ▼
               CSV files in *_results/
```
