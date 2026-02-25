# Multi-Agent Energy Market Simulation — Technical Documentation

## Table of Contents

0. [Notation and Units](#0-notation-and-units)
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

## 0. Notation and Units

This section introduces the symbols used throughout the documentation. All sums in the optimisation problems follow this notation.

### 0.1 Indices and Sets

- \( i \in \mathcal{I} \): agents (VRES, conventional generator, consumer, electrolyzer, green offtaker, grey offtaker, importer, GC demand).
- \( k \in \mathcal{K} \): markets
  - \(k = \text{elec}, \text{elec\_GC}, \text{H2}, \text{H2\_GC}, \text{EP}\).
- \( h \in \mathcal{H} = \{1,\dots,n_{\text{Timesteps}}\} \): hours within a representative day.
- \( d \in \mathcal{D} = \{1,\dots,n_{\text{ReprDays}}\} \): representative days.
- \( y \in \mathcal{Y} = \{1,\dots,n_{\text{Years}}\} \): scenario years.

In the code, these appear as `JH`, `JD`, `JY`.

### 0.2 Time Weights and Probabilities

- \( W_{d,y} \): number of real calendar days represented by representative day \(d\) in year \(y\).
- \( P_y \): probability (or relative weight) of scenario year \(y\) in the CVaR constructions. These are usually normalised so that \(\sum_y P_y = 1\).

### 0.3 Prices, Quantities, and Net Positions

- \( \lambda_k(h,d,y) \): price in market \(k\) at time \((h,d,y)\).
- \( q_i^k(h,d,y) \): **physical quantity** traded by agent \(i\) in market \(k\) at time \((h,d,y)\), sign-free.
- \( g_i^k(h,d,y) \): **net position** of agent \(i\) in market \(k\) at time \((h,d,y)\), following the sign convention:
  - \(g_i^k > 0\): agent \(i\) is a **supplier** in market \(k\).
  - \(g_i^k < 0\): agent \(i\) is a **buyer** in market \(k\).
- Market **imbalance** in market \(k\) at time \((h,d,y)\):

  \[
  r_k(h,d,y) = \sum_{i \in \mathcal{I}_k} g_i^k(h,d,y) - D_k(h,d,y),
  \]

  where \(D_k\) is exogenous demand (for EP only; 0 otherwise).

The **aggregate imbalance norm** used by ADMM is:

\[
\|r_k\|_2 = \left( \sum_{h,d,y} r_k(h,d,y)^2 \right)^{1/2}.
\]

### 0.4 Units

- Electricity: MWh.
- Electricity GC: MWh\(_\text{GC}\) (1 certificate per renewable MWh).
- Hydrogen: MWh\(_\text{H2}\) (or equivalent energy-based unit).
- Hydrogen GC: MWh\(_\text{GC,H2}\).
- End product (EP): MWh\(_\text{EP}\) or t\(_\text{EP}\) (consistent within the model, governed by `Alpha`).

All monetary values are in **EUR** (e.g. €/MWh, €/t, €/MW-year).

### 0.5 Risk Parameters and CVaR

- \( \gamma_i \in [0,1] \): risk weight for agent \(i\).
  - \( \gamma_i = 1 \): risk-neutral (expected loss only).
  - \( 0 < \gamma_i < 1 \): risk-averse (mix of expectation and CVaR).
- \( \beta \in (0,1) \): CVaR confidence level (e.g. 0.95).
- \( \alpha_i \): Value-at-Risk (VaR) proxy for agent \(i\).
- \( u_i(y) \): shortfall above VaR for agent \(i\) in year \(y\).
- \( \mathrm{CVaR}_i \): Conditional Value-at-Risk for agent \(i\).

The **agent-level CVaR** of loss is, in continuous notation:

\[
\mathrm{CVaR}_i = \min_{\alpha_i} \left[ \alpha_i + \frac{1}{1-\beta} \mathbb{E}[(\ell_i - \alpha_i)_+] \right],
\]

where \(\ell_i\) is the random loss and \((x)_+ = \max\{x,0\}\). The code implements the usual linearised form with \(\alpha_i, u_i(y)\) and empirical probabilities \(P_y\).

The **social planner** uses a single \(\gamma\) and a single social CVaR on aggregate welfare (see §6.4).

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
| `Gen_VRES_01` | `VRES` | Variable renewable (e.g. solar). Zero marginal cost. Produces both electricity and elec GCs (1:1). Constrained by hourly availability factor × **endogenous capacity**. Decides yearly installed capacity and investment (MW), incurring fixed annualised CAPEX `FixedCost_per_MW × capacity`. |
| `Gen_Conv_01` | `Conventional` | Dispatchable thermal plant. Constant availability (AF = 1). Marginal cost sets price floor. No GC production. |
| `Cons_Elec_01` | `Consumer` | Elastic electricity demand. Quadratic utility `U(d) = A_E·d − ½B_E·d²` gives inverse demand `p(d) = A_E − B_E·d`. Bounded by `PeakLoad × load_profile`. |

### 3.2 Hydrogen-Sector Agent

| Agent | Type | Description |
|---|---|---|
| `Prod_H2_Green` | `GreenProducer` | PEM electrolyzer with **endogenous H₂ output capacity**. Converts electricity to H₂ with efficiency `η = 1/SpecificConsumption`. Buys elec + elec GCs; sells H₂ + H₂ GCs. Annual green-backing constraint ensures GCs purchased ≥ `(1/η) × GCs issued`. Decides yearly H₂ capacity and investment (MW_H₂), incurring fixed annualised CAPEX `FixedCost_per_MW_Electrolyzer × capacity`. |

### 3.3 Offtaker Agents

| Agent | Type | Description |
|---|---|---|
| `Offtaker_Green` | `GreenOfftaker` | Buys green H₂ and converts it 1:1 (via `Alpha`) to end product. Must buy H₂ GCs for ≥ 42% of EP output (annual mandate `gamma_GC = 0.42`). Tight stoichiometric link: `ep = (1/α) × h2_in`. Has **endogenous EP output capacity** `cap_EP_y[jy]` with investment `inv_EP[jy]` and fixed annualised CAPEX `FixedCost_per_MW_EP_Out × cap_EP_y`. |
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

Each agent minimises its **augmented Lagrangian** (possibly risk-averse for some agents):

```
min  γ_i × [ Σ_{h,d,y} W[d,y] × ( cost_i − revenue_i ) + FixedCAPEX_i ]
   + (1 − γ_i) × CVaR_i(loss_i)
   + Σ_k  (ρ_k / 2) × Σ_{h,d,y} W[d,y] × ( g_i^k − ḡ_i^k )²
```

where:
- `cost_i − revenue_i` is the agent's private cost minus revenue across all markets.
- `g_i^k` is the agent's net position in market `k` (positive = supply, negative = demand).
- `ḡ_i^k` is the consensus target for agent `i` in market `k`.
- `ρ_k` is the penalty weight for market `k`.
- `W[d,y]` scales representative days to a full year.
- `γ_i` is a **per-agent risk weight** (`γ=1` → risk-neutral, `γ<1` → risk-averse). Non-trivial CVaR is used only for VRES, electrolyzer, and green offtaker.
- `CVaR_i(loss_i)` is an agent-specific Conditional Value-at-Risk term constructed with auxiliary variables `(α_i, u_i[jy])` over yearly scenarios `jy ∈ JY`, at confidence level `β`.

More explicitly:

- The **deterministic, expected-loss term**

  \[
  \sum_{h,d,y} W_{d,y}\,\bigl(\mathrm{cost}_i(h,d,y) - \mathrm{rev}_i(h,d,y)\bigr)
  \]

  contains fuel/operational costs, certificate purchases, and investment annuities on the **cost** side, and all market revenues (price × net position) on the **revenue** side.

- The **risk term** \( \mathrm{CVaR}_i(\mathrm{loss}_i) \) captures the tail of the loss distribution over years \(y\). It is only active when \( \gamma_i < 1 \); for \( \gamma_i=1 \) the CVaR part drops out and the agent becomes risk-neutral.

- The **quadratic ADMM penalties**

  \[
  \sum_k \frac{\rho_k}{2}\sum_{h,d,y} W_{d,y}\,\bigl(g_i^k(h,d,y)-\bar g_i^k(h,d,y)\bigr)^2
  \]

  ensure that, in equilibrium, each agent’s net position \(g_i^k\) coincides with a consensus allocation \(\bar g_i^k\) that satisfies market-clearing. Economically, this can be read as a **soft enforcement of market balance**: deviating from the consensus quantity becomes increasingly expensive as \(\rho_k\) grows.

The ADMM part is purely **algorithmic**: it does not change the underlying economic problem. At convergence, all net positions are equal to their consensus copies and all quadratic penalties are zero, so the solution coincides with that of the risk-adjusted competitive equilibrium defined by the first two terms.

#### CVaR formulation (per agent)

For each risk-averse agent (VRES, electrolyzer, green offtaker), CVaR is linearised via:
- `α_i` — VaR proxy (free variable, `≥ 0`)
- `u_i[jy]` — shortfall per scenario year (`≥ 0`)
- `cvar_i` — CVaR value (`≥ 0`)

Constraints:
```
u_i[jy] ≥ loss_i[jy] − α_i                          ∀ jy ∈ JY
cvar_i  ≥ α_i + (1/(1−β)) × Σ_y P[jy] × u_i[jy]
```

**Dynamic constraint updates**: In ADMM, the loss expressions `loss_i[jy]` depend on current market prices `λ` (which change every iteration). Because JuMP expressions bake in coefficient values at creation time, the CVaR shortfall and linking constraints must be **deleted and re-added** in every ADMM iteration with the freshly recomputed loss expressions. This happens in the `solve_*_agent!` functions.

#### Specific objective terms by agent type

**VRES generator (with endogenous capacity and CVaR):**
```
min  γ × [ Σ_y loss_VRES[y] + F_cap × Σ_y cap_VRES[y] ]
   + (1−γ) × CVaR_VRES
   + (ρ_elec/2)×Σ W×(g − ḡ_elec)²
   + (ρ_GC/2)×Σ W×(g − ḡ_GC)²

where loss_VRES[y] = Σ_{h,d} W × ( MC×g − λ_elec×g − λ_GC×g )
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

**Electrolyzer (with endogenous H₂ capacity and CVaR):**
```
min  γ × [ Σ_y loss_H2[y] + F_cap × Σ_y cap_H2[y] ]
   + (1−γ) × CVaR_H2
   + (ρ_elec/2)×Σ W×(−e_in − ḡ_elec)²
   + (ρ_GC/2)×Σ W×(−gc_e − ḡ_GC)²
   + (ρ_H2/2)×Σ W×(h2 − ḡ_H2)²
   + (ρ_H2GC/2)×Σ W×(gc_h2 − ḡ_H2GC)²

where loss_H2[y] = Σ_{h,d} W × ( λ_elec×e_in + λ_GC×gc_e + op×h2 − λ_H2×h2 − λ_H2GC×gc_h2 )
```

**Green offtaker (with endogenous EP capacity and CVaR):**
```
min  γ × [ Σ_y loss_G[y] + F_cap × Σ_y cap_EP[y] ]
   + (1−γ) × CVaR_G
   + (ρ_H2/2)×Σ W×(−h2_in − ḡ_H2)²
   + (ρ_H2GC/2)×Σ W×(−gc_h2 − ḡ_H2GC)²
   + (ρ_EP/2)×Σ W×(ep − ḡ_EP)²

where loss_G[y] = Σ_{h,d} W × ( λ_H2×h2_in + λ_H2GC×gc_h2 + proc×ep − λ_EP×ep )
```

These templates are implemented in the `build_*_agent.jl` files as follows:

- All **price-dependent terms** (e.g. \( \lambda_\text{elec}\,g \), \( \lambda_\text{H2}\,h2\_in \)) are expressed via JuMP `@expression` blocks whose coefficients are updated each ADMM iteration.
- The **capacity-investment linkage** is enforced via yearly variables (e.g. `cap_VRES[y]`, `inv_VRES[y]`) and simple linear relationships: investment in year \(y\) expands the capacity available in all hours of that year.
- For risk-averse agents, the **loss-per-year** expressions `loss_VRES[y]`, `loss_H2[y]`, `loss_G[y]` are recomputed in every ADMM iteration with the *current* prices, so that the CVaR always measures risk with respect to the most recent price trajectory.

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

The social planner maximises **risk-adjusted social welfare** with a **single** social CVaR:

```
max  γ × Σ_y sw_aux[y]  −  (1−γ) × CVaR_social
```

where `sw_aux[y]` is an epigraph proxy for aggregate social welfare per year (see §6.4 for why), and `CVaR_social` penalises tail risk across scenario years. When `γ=1` (risk-neutral), the CVaR term vanishes and the planner reduces to standard welfare maximisation.

#### Per-agent welfare contributions

Each `add_*_to_planner!` function returns a `Dict{Int, Any}` of per-year welfare expressions (no per-agent CVaR — CVaR is applied once to the aggregate). Revenue/expenditure terms cancel out in the aggregate (they are transfers between agents). The per-agent welfare terms are:

- **Consumers**: `U(d) = A×d − (B/2)×d²` (quadratic utility)
- **Generators**: `−MC × g` (negative production cost) minus fixed CAPEX on endogenous capacity for VRES
- **Electrolyzer**: `−op_cost × h2_out` minus fixed CAPEX on endogenous H₂ capacity
- **Green offtaker**: `−processing_cost × ep` minus fixed CAPEX on endogenous EP capacity
- **Other offtakers/importer**: `−processing_cost × ep` or `−import_cost × ep`

#### Social welfare aggregation

```
social_welfare[y] = Σ_i  welfare_per_year_i[y]      (includes quadratic consumer utility)
```

Market-clearing constraints enforce supply = demand. The single social CVaR applies to the full aggregate welfare (including consumer utility), ensuring the risk-averse planner accounts for all welfare components when assessing tail risk.

### 4.4 Risk Aversion and Risk-Neutral Consistency

This section summarises the **risk-aversion architecture** and explains precisely when the **ADMM equilibrium** coincides with the **social planner** solution.

#### 4.4.1 Agent-level vs system-level risk

- In the **ADMM (market exposure) case**:
  - A subset of agents (VRES, electrolyzer, green offtaker) can be risk-averse with their own parameters \((\gamma_i,\beta_i)\).
  - Each such agent minimises a **private risk-adjusted loss**:

    \[
    \gamma_i\,\mathbb{E}[\ell_i] + (1-\gamma_i)\,\mathrm{CVaR}_i(\ell_i),
    \]

    subject to its own technological constraints and the ADMM penalties.
  - Risk is therefore **heterogeneous and decentralised**: different agents may have different attitudes to risk; financial transfers between agents do not directly enter the risk measure.

- In the **social planner case**:
  - There is a **single** system-wide risk parameter \(\gamma\) and confidence level \(\beta\).
  - The planner maximises a **single risk-adjusted social welfare**:

    \[
    \gamma\,\mathbb{E}\bigl[SW\bigr] - (1-\gamma)\,\mathrm{CVaR}_\text{social}(-SW),
    \]

    where \(SW\) is aggregate welfare (including consumer utility and production/investment costs).
  - Risk is therefore **centralised**: society as a whole is risk-averse with respect to aggregate welfare, rather than each agent separately.

These two formulations represent different normative assumptions about **who bears risk** and **how it is shared**. The ADMM run with per-agent CVaR corresponds to a market in which agents individually care about their own tail losses; the social planner corresponds to a benevolent regulator who cares about systemic tail outcomes.

#### 4.4.2 Risk-neutral benchmark and equivalence

When both formulations are made **risk-neutral**, they collapse to the same underlying convex optimisation problem:

- In ADMM:
  - Set \( \gamma_i = 1 \) for all agents that can be risk-averse (VRES, electrolyzer, green offtaker).
  - This eliminates all per-agent CVaR terms from their objectives.

- In the social planner:
  - Set the planner-wide risk weight \( \gamma = 1 \).
  - This eliminates \(\mathrm{CVaR}_\text{social}\) from the planner’s objective, so the model becomes a quadratic (but not quadratically constrained) welfare maximisation with standard consumer surplus and producer surplus terms.

Under these settings:

1. **Agent technology and preferences** are identical in both formulations:
   - The same constraints on capacities, conversion efficiencies, and mandates apply.
   - The same quadratic utility and cost functions are used.
2. **Market-clearing conditions** are enforced:
   - In ADMM, via the augmented Lagrangian and convergence of primal/dual residuals.
   - In the planner, via explicit equality constraints.
3. **Welfare decomposition** coincides with the sum of individual profit/utility functions.

As a result, in the **limit of exact ADMM convergence** (all markets have residuals within tolerance, and ρ updates have stabilised), the ADMM allocation coincides with the planner’s allocation, and the recovered equilibrium prices coincide with the planner’s dual variables up to numerical tolerance. This is the formal sense in which the **risk-neutral social planner and the risk-neutral ADMM equilibrium should produce the same result**.

In practice, small discrepancies can arise from:

- Finite ADMM stopping tolerance (non-zero residuals),
- Different initialisations of prices and ρ,
- Numerical tolerances in the solver (Gurobi) and the QP/QCP/LP transformation.

These differences are typically negligible for economic interpretation and are visible in the diagnostic plots and CSVs.

---

## 5. ADMM Algorithm

### 5.1 Iteration Structure

Each ADMM iteration `k` proceeds as follows:

1. **For each agent** (via `ADMM_subroutine!`):
   a. Update consensus target: `ḡ_i = q_i^{k-1} − (1/(n+1)) × imbalance^{k-1}`
   b. Update prices `λ`, penalty `ρ` from the global ADMM state.
   c. Rebuild objective with updated parameters.
   d. For CVaR agents (VRES, electrolyzer, green offtaker): recompute loss expressions with current `λ`, then delete and re-add CVaR shortfall and linking constraints with the fresh losses.
   e. Solve the agent's QP.
   f. Record the solution quantities.

2. **Compute market imbalances**: For each market, sum all agents' net positions. For EP, subtract fixed demand `D_EP`.

3. **Compute residuals**:
   - **Primal residual** = `‖imbalance‖₂` (L2 norm; measures market-clearing violation).
   - **Dual residual** = `‖ρ × Δ(consensus deviation)‖₂` (measures position stability).

   More precisely, for each market \(k\):

   - Let \( r_k^t(h,d,y) \) be the **market imbalance** at iteration \(t\).
   - Let \( \Delta z_k^t(h,d,y) \) be the **change in consensus deviation** (difference between successive consensus copies) at iteration \(t\).

   Then:

   \[
   \|r_k^t\|_2 = \Bigl(\sum_{h,d,y} r_k^t(h,d,y)^2\Bigr)^{1/2},\qquad
   \|s_k^t\|_2 = \rho_k^t\,\Bigl(\sum_{h,d,y} (\Delta z_k^t(h,d,y))^2\Bigr)^{1/2}.
   \]

   The primal residual \(\|r_k^t\|_2\) measures **how far the market is from clearing**, while the dual residual \(\|s_k^t\|_2\) measures **how stable the agents’ net positions are** from one iteration to the next.

4. **Update prices**: `λ^{k+1} = λ^k − η_k × ρ_k × imbalance^k` (gradient ascent on the dual with **residual-aware step size** `η_k ∈ (0,1]` per market). Far from convergence, `η_k = 1` and we recover the standard update. Near convergence (when both primal and dual residuals are within a modest multiple of tolerance), `η_k` is reduced (e.g. 0.3) to damp oscillations in tightly coupled markets while keeping `ρ_k` fixed.

5. **Update ρ** (history-aware multi-regime rule): For each market independently:
   - **Regime 1 (rp vs rd imbalanced)**: If `primal > balance_threshold × dual` → increase `ρ` (under-penalised). If `dual > balance_threshold × primal` → decrease `ρ` (over-penalised). Increases are applied only when they have not worsened the recent residual history.
   - **Regime 2 (far from tolerance, rp ≈ rd)**: If both residuals are much larger than the market tolerance but comparable, apply a **gentle multiplicative increase** to ρ (again, only if the recent residual history has not deteriorated) to avoid stalling with large residuals.
   - **Regime 3 (near-convergence stability with hysteresis)**: If both residuals are within a modest multiple of tolerance and close to the **best residuals seen so far** for that market, **freeze ρ permanently**. From that point on, ADMM behaves like fixed-ρ ADMM in that market, preventing later ρ updates from kicking the algorithm out of a good basin and eliminating small oscillations around the optimum.

6. **Convergence check**: All five markets must have both primal and dual residuals below their tolerance.

### 5.2 Consensus Formula (Sharing ADMM)

The consensus target for agent `i` in a market with `n` participants:

```
ḡ_i^k = q_i^{k-1} − (1/(n+1)) × Σ_j q_j^{k-1}
```

The `(n+1)` denominator comes from the sharing ADMM formulation, which introduces one "market copy" alongside the `n` agent copies. This distributes the imbalance correction equally.

### 5.3 Adaptive Penalty (ρ)

The adaptive penalty mechanism is implemented in `update_rho.jl` and is designed as a **state-of-the-art, history-aware extension** of the classic Boyd rule. It combines:

1. **Per-market tuning** (different growth caps and factors per market),
2. **Three behavioural regimes** (balance, gentle push, stability),
3. **Hysteresis and residual-history safeguards** that prevent the algorithm from leaving a good basin once it has found one and avoid harmful ρ increases.

Per-market parameters:

| Market | Increase factor | Decrease factor | ρ_max | Notes |
|---|---|---|---|---|
| `elec`, `elec_GC` | 1.05 | 1/1.05 | 5,000 | Dominant, tightly coupled electricity/GC markets; moderate adaptation with a low cap avoids ill-conditioning while still correcting imbalances quickly. |
| `H2` | 1.01 | 1/1.01 | 100 | Strongly coupled to electricity and EP; very gentle updates minimise oscillation when H₂ capacity/investment kinks are active. |
| `H2_GC` | 1.05 | 1/1.05 | 100 | Hourly GC market but thin volumes; moderate adaptation with a conservative cap. |
| `EP` | 1.01 | 1/1.01 | 100 | Stiff EP market; slow adaptation avoids limit cycles when EP capacities/investments bind. |

In addition to these static parameters, the algorithm maintains:

- **Best residuals per market**: `best_primal[key]`, `best_dual[key]` track the smallest primal and dual residuals seen so far. They serve as a *hysteresis anchor* for deciding when a market has truly entered a near-solution region.
- **A short residual history per market**: `R_hist[key]` stores a short window of `R = rp + rd`. Before increasing ρ, the rule checks whether residuals have improved (or at least not deteriorated) over this window; if they have worsened, the increase is skipped.
- **Per-market freeze flags**: once a market hits residuals that are both within a modest multiple of tolerance and close to its best-ever residuals, its `ρ` is frozen permanently. Subsequent iterations keep ρ fixed for that market.

This combination yields a robust behaviour:

- **Far from tolerance**: ρ can adapt aggressively enough to overcome stalling and large imbalances.
- **Near the solution**: ρ becomes effectively fixed and the algorithm behaves like a stable fixed-ρ ADMM, eliminating the classic oscillatory patterns observed with naive adaptive schemes.

In pseudo-code, the **per-market update** at iteration \(t\) can be summarised as:

```text
for each market k:
    compute rp = ||r_k^t||_2, rd = ||s_k^t||_2
    update best_primal[k], best_dual[k], R_hist[k]

    if rho_frozen[k]:
        continue   # Regime 3: fixed-ρ near convergence

    if rp <= c_freeze * ε_pri_k  and  rd <= c_freeze * ε_dual_k
       and rp, rd close to best_primal[k], best_dual[k]:
        rho_frozen[k] = true     # enter permanent fixed-ρ regime
        continue

    if rp > balance_threshold * rd or rd > balance_threshold * rp:
        # Regime 1: classic Boyd-like rule with safeguards
        if rp > balance_threshold * rd and residual_history_improved(k):
            ρ_k^{t+1} = min(ρ_max[k], ρ_inc[k] * ρ_k^t)
        elseif rd > balance_threshold * rp and residual_history_improved(k):
            ρ_k^{t+1} = ρ_dec[k] * ρ_k^t
        else
            ρ_k^{t+1} = ρ_k^t
    elseif rp, rd >> ε_pri_k, ε_dual_k and residual_history_improved(k):
        # Regime 2: gentle push when both residuals are large but balanced
        ρ_k^{t+1} = min(ρ_max[k], ρ_soft_inc[k] * ρ_k^t)
    else:
        ρ_k^{t+1} = ρ_k^t
```

where:

- `c_freeze` controls how close to tolerance we require residuals to be before freezing ρ,
- `ρ_inc[k]`, `ρ_dec[k]`, and `ρ_soft_inc[k]` are the multiplicative factors from the table above,
- `residual_history_improved(k)` checks whether the short history window in `R_hist[k]` has improved (or at least not deteriorated), guarding against **myopic** ρ increases that would worsen convergence.

### 5.4 Convergence Tolerances (Boyd-style)

Instead of a single scalar tolerance, the implementation follows the **absolute + relative** stopping criteria proposed by Boyd et al. (2011) for ADMM. For each market `k` we define:

- Absolute tolerance `ε_abs` (MW-scale), taken from `ADMM.epsilon_abs` in `data.yaml` if present, otherwise from `ADMM.epsilon`.
- Relative tolerance `ε_rel` (dimensionless), taken from `ADMM.epsilon_rel` in `data.yaml` if present, otherwise `0.0`.

Let:

- `n = nTimesteps × nReprDays × nYears` be the number of time slots in the horizon.
- `Scale_primal[k]` and `Scale_dual[k]` be fixed reference magnitudes for the primal and dual residuals of market `k`, captured from the first non-zero residual observed for that market (stored in `ADMM["ResidualScale"]`).

Then the per-market primal and dual tolerances are:

```
ε_pri_k  = ε_abs * sqrt(n) + ε_rel * Scale_primal[k]
ε_dual_k = ε_abs * sqrt(n) + ε_rel * Scale_dual[k]
```

The stopping rule is:

- **Primal**: for every market `k`, the L2 norm of the imbalance vector must satisfy `‖r_k‖₂ ≤ ε_pri_k`.
- **Dual**: for every market `k`, the L2 norm of the change in consensus deviation must satisfy `‖s_k‖₂ ≤ ε_dual_k`.

All five markets must simultaneously satisfy both conditions for convergence to be declared.

This has three advantages over a single scalar `epsilon`:

1. **Scale awareness**: Markets with large typical flows (e.g. electricity) naturally get larger absolute tolerances than thin markets (e.g. GC), while still using a common `(ε_abs, ε_rel)` pair.
2. **Robustness to refinement**: If the temporal resolution or the number of representative days changes (n increases), the `sqrt(n)` factor keeps the per-slot accuracy comparable.
3. **Numerical realism**: Once residuals are small relative to the problem’s own scale (`Scale_*[k]`), the criteria do not force the algorithm to chase tiny numerical oscillations; they recognise that the solution is “good enough” in the sense of Boyd et al.

### 5.5 Sign Convention

| Role | Net position sign | Example |
|---|---|---|
| Supplier / seller | **Positive** | VRES generation `+g`, H₂ sales `+h2_out` |
| Buyer / consumer | **Negative** | Electricity demand `−d`, H₂ purchase `−h2_in` |

Market imbalance = Σ (net positions). Positive imbalance = excess supply → price decreases. Negative imbalance = excess demand → price increases.

---

## 6. Social Planner Benchmark

The social planner (`social_planner.jl`) solves a single centralised convex QCP (quadratically constrained program) that maximises risk-adjusted social welfare subject to all individual agent constraints plus market-clearing balance constraints. It serves as the theoretical first-best benchmark. When `γ=1` (risk-neutral), the CVaR term vanishes and the planner reduces to a standard QP, matching the ADMM risk-neutral equilibrium.

### 6.1 Market-Clearing Constraints

| Constraint | Equation |
|---|---|
| Electricity balance | `Σ generation − Σ demand − Σ electrolyzer_elec_buy = 0` (per h,d,y) |
| Elec GC balance | `Σ VRES_generation − Σ electrolyzer_GC_buy − Σ GC_demand = 0` (per h,d,y) |
| H₂ balance | `Σ H₂_production − Σ H₂_consumption − Σ offtaker_H₂_buy = 0` (per h,d,y) |
| H₂ GC balance | `Σ H₂_GC_supply − Σ H₂_GC_demand = 0` (per h,d,y — hourly, same as other markets) |
| EP balance | `Σ offtaker_EP_supply − D_EP − Σ EP_demand = 0` (per h,d,y) |

### 6.2 Price Recovery (Two-Step QCP Dual Recovery)

Equilibrium prices are the **dual variables** (shadow prices) of the market-clearing constraints. However, the epigraph formulation (§6.4) makes the model a QCP, and **Gurobi does not provide dual variables for QCP models**.

The solution is a **two-step dual recovery** procedure:

1. **Step 1 — QCP solve**: Solve the full QCP to obtain optimal primal values (quantities, capacities, CVaR variables). Accept both `OPTIMAL` and `LOCALLY_SOLVED` (for convex QCPs, local = global optimum).

2. **Step 2 — Convert QCP → LP**: Fix the demand variables (`d_E`, `d_GC_E`, `d_EP`) at their QCP-optimal values. Delete the quadratic epigraph constraints and re-add them as linear constraints (with demand welfare evaluated as a constant). This converts the model to a pure LP.

3. **Step 3 — LP solve**: Re-solve the LP. Gurobi provides full dual variables for LPs solved to `OPTIMAL`.

4. **Step 4 — Save results**: Extract duals (prices) and variable values.

5. **Step 5 — Cleanup**: Unfix demand variables, delete linear epigraph constraints, restore original quadratic epigraph constraints (model back to QCP form for potential re-use).

This approach is **exact**: the LP has the same feasible allocation as the QCP (demand fixed at optimal values), so the duals represent the correct marginal prices at the risk-averse optimal allocation.

**Why fixing demand variables works**: The only quadratic constraints are the epigraph constraints `sw_aux[y] ≤ social_welfare[y]`, where `social_welfare[y]` contains `−B/2 × d²` terms from elastic demand agents. Fixing `d` makes `d²` a constant, so the constraints become linear. The replacement constraints use numerically evaluated demand welfare (constant) plus the still-variable supply-side welfare (linear), producing purely linear constraints.

### 6.3 Code Architecture

All problem definition lives in `Source/build_*.jl` files. Each file contains:

- `build_*_agent!()` — Builds the ADMM version (with `λ`, `ρ`, `ḡ` penalty terms and per-agent CVaR for risk-averse agents).
- `add_*_agent_to_planner!()` — Adds the same variables/constraints to the planner model **without** ADMM terms and **without** per-agent CVaR. Returns a `Dict{Int, Any}` of per-year welfare expressions.

`build_social_planner.jl` orchestrates the calls to all `add_*_to_planner!` functions, adds market-clearing constraints, aggregates per-year welfare into `social_welfare`, adds the epigraph formulation and single social CVaR, and sets the risk-adjusted objective.

### 6.4 Epigraph Formulation for Social CVaR

The social planner applies **one single CVaR** to the aggregate social welfare (not per-agent CVaR). This ensures risk aversion considers all welfare components (consumer utility, production costs, investment costs) holistically.

**Problem**: `social_welfare[y]` includes quadratic consumer utility terms (`A·d − B/2·d²`). Putting `−social_welfare[y]` inside the CVaR shortfall constraint would create a quadratic constraint (QC), turning the model into a QCP. Gurobi cannot provide duals for QCPs.

**Solution — epigraph reformulation**: Introduce auxiliary variables `sw_aux[y]` with epigraph constraints:

```
sw_aux[y] ≤ social_welfare[y]     (quadratic constraint, standard convex form)
```

The CVaR constraints then reference `sw_aux` instead of the quadratic `social_welfare`, making them purely linear:

```
u_social[y]  ≥ −sw_aux[y] − α_social                          ∀ y ∈ JY
cvar_social  ≥ α_social + (1/(1−β)) × Σ_y P[y] × u_social[y]
```

The objective is also linear:

```
max  γ × Σ_y sw_aux[y]  −  (1−γ) × cvar_social
```

Since the objective maximises `sw_aux`, the epigraph constraint binds at optimality (`sw_aux[y] = social_welfare[y]`), making the formulation mathematically equivalent to applying CVaR directly to `social_welfare`.

The epigraph constraints are the **only** quadratic constraints in the model. They are in Gurobi's standard convex QC form (PSD Q-matrix on the `≤` side). All other constraints (CVaR, market-clearing, capacity bounds) are purely linear. The dual recovery procedure (§6.2) handles the QCP→LP conversion for price extraction.

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
│   ├── build_social_planner.jl       # Orchestrate planner: call add_*_to_planner!, add balance constraints, epigraph + social CVaR, set risk-adjusted objective
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
│   ├── update_rho.jl                 # Adaptive penalty update (Boyd rule with 3 regimes: balance, gentle push, fixed-ρ near convergence)
│   ├── save_results.jl               # Write market-exposure CSV outputs (including capacity & investment summaries)
│   └── save_social_planner_results.jl # Write social-planner CSV outputs (including capacity & investment summaries)
│
├── market_exposure_results/          # Output from market_exposure.jl
│   ├── ADMM_Convergence.csv          # Primal & dual residuals per iteration
│   ├── ADMM_Diagnostics.csv          # ρ, mean price, mean imbalance per iteration
│   ├── Electricity_Market_History.csv
│   ├── Hydrogen_Market_History.csv
│   ├── Electricity_GC_Market_History.csv
│   ├── H2_GC_Market_History.csv
│   ├── End_Product_Market_History.csv
│   ├── Agent_Summary.csv             # Agent group membership and ADMM objective value
│   ├── Agent_Quantities_Final.csv    # Final-iteration net quantities per agent
│   ├── Offtaker_GC_Diagnostics.csv   # GC compliance per offtaker
│   ├── H2_Producer_Diagnostics.csv   # H₂ GC-to-production ratio
│   ├── Capacity_Investments.csv      # VRES/electrolyzer/green offtaker yearly capacity & investment (ADMM)
│   └── TimerOutput.yaml              # Profiling data
│
└── social_planner_results/           # Output from social_planner.jl
    ├── Market_Prices.csv             # Equilibrium prices (duals of balance constraints)
    ├── Agent_Summary.csv             # Per-agent quantity & ADMM-style objective value
    └── Capacity_Investments_Planner.csv  # VRES/electrolyzer/green offtaker yearly capacity & investment (planner)
```

---

## 10. File Reference

### 10.1 Runner Scripts

| File | Purpose |
|---|---|
| `market_exposure.jl` | Entry point for distributed ADMM. Sections 1–13: env, packages, dirs, source loading, data loading, results folder, agent init, market params, agent params, build models, run ADMM, save results. |
| `social_planner.jl` | Entry point for centralised benchmark. Sections 1–12: same structure as market_exposure but builds a single planner model instead of per-agent models + ADMM loop. Section 11 implements the two-step QCP dual recovery (QCP solve → fix demand vars + replace QC → LP solve → extract duals). |

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
| `build_power_agent.jl` | `build_power_agent!()` — power agents (VRES with capacity & CVaR, conventional, consumer) | `add_power_agent_to_planner!()` — same constraints, returns `Dict{Int, Any}` of per-year welfare (no per-agent CVaR) |
| `build_H2_agent.jl` | `build_H2_agent!()` — electrolyzer with 4-market ADMM terms, endogenous capacity & CVaR | `add_H2_agent_to_planner!()` — same constraints, returns per-year welfare = −op_cost − fixed CAPEX (no per-agent CVaR) |
| `build_offtaker_agent.jl` | `build_offtaker_agent!()` — green/grey/importer (green with EP capacity & CVaR) | `add_offtaker_agent_to_planner!()` — same constraints, returns per-year welfare = −processing/import cost − fixed CAPEX (no per-agent CVaR) |
| `build_elec_GC_demand_agent.jl` | `build_elec_GC_demand_agent!()` — GC demand with ADMM | `add_elec_GC_demand_agent_to_planner!()` — returns per-year utility expression |
| `build_EP_demand_agent.jl` | `build_EP_demand_agent!()` — placeholder | `add_EP_demand_agent_to_planner!()` — returns per-year utility expression |
| `build_social_planner.jl` | — | `build_social_planner!()` — orchestrates all add_*_to_planner!, adds balance constraints, aggregates welfare, adds epigraph + single social CVaR, sets risk-adjusted objective |

### 10.4 Solve Files

| File | Role |
|---|---|
| `solve_power_agent.jl` | Rebuilds objective with current λ, ḡ, ρ. For VRES: recomputes loss expressions with current λ, deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_H2_agent.jl` | Rebuilds objective with current λ, ḡ, ρ. Recomputes loss expressions with current λ (4-market), deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_offtaker_agent.jl` | Rebuilds objective for green/grey/importer. For GreenOfftaker: recomputes loss expressions with current λ, deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
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
| `save_social_planner_results.jl` | Called after two-step dual recovery (LP re-solve). Writes: Market_Prices.csv (duals of balance constraints), Agent_Summary.csv (quantities + welfare), Capacity_Investments_Planner.csv. |

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
| `Market_Prices.csv` | Columns: `Time`, `Elec_Price`, `H2_Price`, `Elec_GC_Price`, `H2_GC_Price`, `EP_Price`. One row per (jy, jd, jh) timestep. Prices = duals of balance constraints, obtained via the two-step QCP dual recovery (§6.2). Raw duals are divided by representative-day weights `W[jd,jy]` to recover the true per-MWh price. |
| `Agent_Summary.csv` | Columns: `Agent`, `Type`, `Total_Quantity`, `Welfare_Contribution`. |
| `Capacity_Investments_Planner.csv` | Per-agent yearly capacity and investment for VRES, electrolyzer, and green offtaker. |

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
