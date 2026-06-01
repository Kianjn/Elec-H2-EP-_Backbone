# Multi-Agent Energy Market Simulation — Technical Documentation

## Table of Contents

0. [Notation and Units](#0-notation-and-units)
1. [Overview](#1-overview)
2. [Markets](#2-markets) — incl. [Contract pools (PPA + HPA)](#contract-pools-market_exposure_contractsjl-only) and [How contract capacity is determined](#how-contract-capacity-is-determined)
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

- $i \in \mathcal{I}$: agents (VRES, conventional generator, consumer, electrolyzer, green offtaker, grey offtaker, importer, GC demand).
- $k \in \mathcal{K}$: markets
  - $k = \text{elec}, \text{elec\_GC}, \text{H2}, \text{H2\_GC}, \text{EP}$.
- $h \in \mathcal{H} = \{1,\dots,n_{\text{Timesteps}}\}$: hours within a representative day.
- $d \in \mathcal{D} = \{1,\dots,n_{\text{ReprDays}}\}$: representative days.
- $y \in \mathcal{Y} = \{1,\dots,n_{\text{Years}}\}$: scenario years.

In the code, these appear as `JH`, `JD`, `JY`.

### 0.2 Time Weights and Probabilities

- $W_{d,y}$: number of real calendar days represented by representative day $d$ in year $y$.
- $P_y$: probability (or relative weight) of scenario year $y$ in the CVaR constructions. These are usually normalised so that $\sum_y P_y = 1$.

### 0.3 Prices, Quantities, and Net Positions

- $\lambda_k(h,d,y)$: price in market $k$ at time $(h,d,y)$.
- $q_i^k(h,d,y)$: **physical quantity** traded by agent $i$ in market $k$ at time $(h,d,y)$, sign-free.
- $g_i^k(h,d,y)$: **net position** of agent $i$ in market $k$ at time $(h,d,y)$, following the sign convention:
  - $g_i^k > 0$: agent $i$ is a **supplier** in market $k$.
  - $g_i^k < 0$: agent $i$ is a **buyer** in market $k$.
- Market **imbalance** in market $k$ at time $(h,d,y)$:

  ```math
  r_k(h,d,y) = \sum_{i \in \mathcal{I}_k} g_i^k(h,d,y) - D_k(h,d,y),
  ```

  where $D_k$ is exogenous demand (for EP only; 0 otherwise).

The **aggregate imbalance norm** used by ADMM is:

```math
\|r_k\|_2 = \left( \sum_{h,d,y} r_k(h,d,y)^2 \right)^{1/2}.
```

### 0.4 Units

- Electricity: MWh.
- Electricity GC: MWh$_\text{GC}$ (1 certificate per renewable MWh).
- Hydrogen: MWh$_\text{H2}$ (or equivalent energy-based unit).
- Hydrogen GC: MWh$_\text{GC,H2}$.
- End product (EP): MWh$_\text{EP}$ or t$_\text{EP}$ (consistent within the model, governed by `Alpha`).

All monetary values are in **EUR** (e.g. €/MWh, €/t, €/MW-year).

### 0.5 Risk Parameters and CVaR

- $\gamma_i \in [0,1]$: risk weight for agent $i$.
  - $\gamma_i = 1$: risk-neutral (expected loss only).
  - $0 < \gamma_i < 1$: risk-averse (mix of expectation and CVaR).
- $\beta \in (0,1)$: CVaR confidence level (e.g. 0.95).
- $\alpha_i$: Value-at-Risk (VaR) proxy for agent $i$.
- $u_i(y)$: shortfall above VaR for agent $i$ in year $y$.
- $\mathrm{CVaR}_i$: Conditional Value-at-Risk for agent $i$.

The **agent-level CVaR** of loss is, in continuous notation:

```math
\mathrm{CVaR}_i = \min_{\alpha_i} \left[ \alpha_i + \frac{1}{1-\beta} \mathbb{E}[(\ell_i - \alpha_i)_+] \right],
```

where $\ell_i$ is the random loss and $(x)_+ = \max\{x,0\}$. The code implements the usual linearised form with $\alpha_i, u_i(y)$ and empirical probabilities $P_y$.

The **social planner** uses a single $\gamma$ and a single social CVaR on aggregate welfare (see §6.4).

---

## 1. Overview

This project implements a **multi-agent equilibrium model** for coupled electricity, hydrogen, green-certificate, and end-product markets, coordinated via **ADMM** (Alternating Direction Method of Multipliers). Each agent has its own JuMP optimization model; market-clearing is achieved by iteratively updating prices and penalty terms so that supply and demand balance in each market.

The project includes three entry points:

- **`market_exposure.jl`** — Distributed ADMM simulation where agents optimise independently and are coordinated through iterative price signals. Five markets: electricity, elec GC, H₂, H₂ GC, end product.
- **`market_exposure_contracts.jl`** — Same as market_exposure but with two bilateral contract pools:
  - **PPA** between VRES and GreenProducer (electricity + elec_GC bundled),
  - **HPA** between GreenProducer and GreenOfftaker (hydrogen + H2_GC equivalent bundled).
  Both are pay-as-produced and clear via ADMM alongside the five standard markets.
- **`social_planner.jl`** — Centralised welfare-maximisation benchmark where all agents are optimised jointly in a single model. Equilibrium prices emerge as dual variables of market-clearing constraints.

The base ADMM and social planner share the **same** problem definition from `Source/`. The contracts case uses contract-specific build/solve/ADMM/save modules that extend the base logic. The social planner is unchanged and does not include the contract pool.

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

### Contract pools (market_exposure_contracts.jl only)

In `market_exposure_contracts.jl`, two additional bilateral pools are cleared:

| Market | Key | Description | Unit | Supply Side | Demand Side |
|---|---|---|---|---|---|
| **PPA** | `ppa` | Bilateral VRES–GreenProducer electricity flow (bundled with elec_GC) | MWh | VRES (`g_ppa`) | GreenProducer (`g_ppa_from`) |
| **HPA** | `hpa` | Bilateral GreenProducer–GreenOfftaker hydrogen flow (bundled with H2_GC equivalent) | MWh_H2 | GreenProducer (`h2_hpa`) | GreenOfftaker (`h2_hpa_from`) |

- **Contract capacity** (`ppa_cap`, `hpa_cap`): upper bound on contract flow at each hour; scalar consensus per bilateral sub-market; **no separate capacity price**.
- **Contract energy** (`g_ppa`, `h2_hpa`): delivered energy at each timestep, cleared at `λ_ppa` / `λ_hpa`.
- **Pay-as-produced**: if output is zero at a timestep, contract delivery is zero and payment is zero.

Each pool uses the same ADMM structure as other markets:
- energy imbalance per sub-market must go to zero,
- scalar capacity consensus (`+cap` supplier and `-cap` buyer in net-position convention) must converge.

#### How contract capacity is determined

There is **no separate capacity price** (no λ for capacity consensus). Contracted capacities (`ppa_cap`, `hpa_cap`) are determined by two mechanisms working together:

**1. Economic optimisation (each party chooses independently)**

Each party has a contract capacity variable (`ppa_cap` or `hpa_cap`) as a **decision variable**. The choice is driven by economic incentives:

- **PPA**:
  - VRES revenue: `λ_ppa × g_ppa`, with `g_ppa ≤ ppa_cap`.
  - GreenProducer cost: `λ_ppa × g_ppa_from`, with `g_ppa_from ≤ ppa_cap`.
  - VRES total generation is split: `g_EOM + g_ppa ≤ AF × cap_VRES`.

- **HPA**:
  - GreenProducer revenue: `λ_hpa × h2_hpa`, with `h2_hpa ≤ hpa_cap` and `h2_hpa ≤ h2_out`.
  - GreenOfftaker cost: `λ_hpa × h2_hpa_from`, with `h2_hpa_from ≤ hpa_cap`.
  - Contracted H2 is removed from producer pool positions (`H2` and associated `H2_GC` pool stream).

**2. ADMM consensus (both must agree)**

The two parties would generally choose different capacities if unconstrained. ADMM enforces agreement via a quadratic penalty:

- **Supplier side** minimises: `(ρ_cap/2) × (cap − ḡ_cap)²`
- **Buyer side** minimises: `(ρ_cap/2) × (−cap − ḡ_cap)²`

Here `ḡ_cap` is the consensus target `z_cap` from the per-agent capacity ADMM equality split (see §5.4). As iterations proceed, the capacity residual `‖x_cap − z_cap‖` shrinks and `λ_cap` accumulates the missing first-order force that drives `x_cap → z_cap` exactly in the limit.

**3. Equilibrium outcome**

At equilibrium:

1. Both parties choose the same contract capacity (consensus satisfied).
2. Contract energy matches between supplier and buyer (cleared by λ).
3. Equilibrium contract capacity is where both parties’ preferred values coincide at equilibrium λ.

The capacity commitment is implicitly priced through energy prices (`λ_ppa`, `λ_hpa`): higher capacity allows more delivered contract energy when production is available, so the pay-as-produced structure bundles capacity and energy.

### Market coupling

The markets are coupled through the **electrolyzer**, which sits at the nexus:

- It **buys** electricity (elec market) and electricity GCs (elec_GC market).
- It **sells** hydrogen (H2 market) and hydrogen GCs (H2_GC market).
- The conversion constraint `h2_out = η × e_in` links the electricity and hydrogen markets.
- The annual green-backing constraint links the elec_GC and H2_GC markets.

The **end-product market** is coupled to H2 and H2_GC through the offtakers, who convert hydrogen into the end product and must comply with the GC mandate.

In the **contracts case**:
- PPA couples VRES and GreenProducer: VRES splits generation `g_EOM + g_ppa`, and GreenProducer uses `e_in_pool + g_ppa_from`.
- HPA couples GreenProducer and GreenOfftaker: GreenProducer splits hydrogen pool sale vs contract (`h2_out - h2_hpa` to pool, `h2_hpa` to contract).

---

## 3. Agents

### 3.1 Power-Sector Agents

| Agent | Type | Description |
|---|---|---|
| `Gen_VRES_01` | `VRES` | Variable renewable (e.g. solar). Zero marginal cost. Produces both electricity and elec GCs (1:1). Constrained by hourly availability factor × **endogenous capacity**. Decides yearly installed capacity and investment (MW), incurring fixed annualised CAPEX `FixedCost_per_MW × capacity`. In `market_exposure_contracts.jl`: splits generation into `g_EOM` (pool) and `g_ppa` (PPA); `g_ppa ≤ ppa_cap` at every hour; revenue includes `λ_ppa × g_ppa`. |
| `Gen_Conv_01` | `Conventional` | Dispatchable thermal fleet proxy. Constant availability (AF = 1). Uses a 3-stage increasing marginal-cost curve (coal-like, biomass-like, NG-like blocks) with configurable stage shares, stage-start marginal costs, and a final high-load marginal cost. No GC production. |
| `Cons_Elec_01` | `Consumer` | Elastic electricity demand. Quadratic utility `U(d) = A_E·d − ½B_E·d²` gives inverse demand `p(d) = A_E − B_E·d`. Bounded by `PeakLoad × load_profile`. |

### 3.2 Hydrogen-Sector Agent

| Agent | Type | Description |
|---|---|---|
| `Prod_H2_Green` | `GreenProducer` | PEM electrolyzer with **endogenous H₂ output capacity**. Converts electricity to H₂ with efficiency `η = 1/SpecificConsumption`. Buys elec + elec GCs; sells H₂ + H₂ GCs. Annual green-backing constraint ensures GCs purchased ≥ `(1/η) × GCs issued`. Decides yearly H₂ capacity and investment (MW_H₂), incurring fixed annualised CAPEX `FixedCost_per_MW_Electrolyzer × capacity`. In `market_exposure_contracts.jl`: receives `g_ppa_from` from VRES and buys `e_in_pool`; total input = `e_in_pool + g_ppa_from`. It also sells `h2_hpa` to GreenOfftaker under HPA (`h2_hpa ≤ hpa_cap`, pay-as-produced at `λ_hpa`), while pool sales are only from non-contracted output. |

### 3.3 Offtaker Agents

| Agent | Type | Description |
|---|---|---|
| `Offtaker_Green` | `GreenOfftaker` | Buys green H₂ and converts it 1:1 (via `Alpha`) to end product. Must buy H₂ GCs for ≥ 42% of EP output (annual mandate `gamma_GC = 0.42`). Tight stoichiometric link: `ep = (1/α) × h2_in`. Has **endogenous EP output capacity** `cap_EP_y[jy]` with investment `inv_EP[jy]` and fixed annualised CAPEX `FixedCost_per_MW_EP_Out × cap_EP_y`. In contracts case, buys `h2_hpa_from` under HPA (pay-as-produced at `λ_hpa`) in addition to pool H₂ purchases. |
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

  ```math
  \sum_{h,d,y} W_{d,y}\,\bigl(\mathrm{cost}_i(h,d,y) - \mathrm{rev}_i(h,d,y)\bigr)
  ```

  contains fuel/operational costs, certificate purchases, and investment annuities on the **cost** side, and all market revenues (price × net position) on the **revenue** side.

- The **risk term** $\mathrm{CVaR}_i(\mathrm{loss}_i)$ captures the tail of the loss distribution over years $y$. It is only active when $\gamma_i < 1$; for $\gamma_i=1$ the CVaR part drops out and the agent becomes risk-neutral.

- The **quadratic ADMM penalties**

  ```math
  \sum_k \frac{\rho_k}{2}\sum_{h,d,y} W_{d,y}\,\bigl(g_i^k(h,d,y)-\bar g_i^k(h,d,y)\bigr)^2
  ```

  ensure that, in equilibrium, each agent’s net position $g_i^k$ coincides with a consensus allocation $\bar g_i^k$ that satisfies market-clearing. Economically, this can be read as a **soft enforcement of market balance**: deviating from the consensus quantity becomes increasingly expensive as $\rho_k$ grows.

The ADMM part is purely **algorithmic**: it does not change the underlying economic problem. At convergence, all net positions are equal to their consensus copies and all quadratic penalties are zero, so the solution coincides with that of the risk-adjusted competitive equilibrium defined by the first two terms.

#### CVaR formulation (per agent)

For each risk-averse agent (VRES, electrolyzer, green offtaker), CVaR is linearised via:

**Important**: The loss that enters CVaR must be the **full** per-year loss, including the fixed capacity cost (F_cap × cap). If only the operational loss is used, then when γ < 1 the fixed cost appears only in the γ-weighted term, so the effective weight on F_cap becomes γ instead of 1. With nYears = 1 (no scenarios), changing γ would then change the objective, breaking the equivalence between social planner and market exposure. The correct formulation uses `loss_total[y] = loss_operational[y] + F_cap × cap[y]` in the CVaR shortfall constraints. With one scenario, CVaR = loss_total, so the objective reduces to loss_total regardless of γ.
- `α_i` — VaR proxy (free variable, `≥ 0`)
- `u_i[jy]` — shortfall per scenario year (`≥ 0`)
- `cvar_i` — CVaR value (`≥ 0`)

Constraints:
```
u_i[jy] ≥ loss_i[jy] − α_i                          ∀ jy ∈ JY
cvar_i  ≥ α_i + (1/(1−β)) × Σ_y P[jy] × u_i[jy]
```

**Dynamic constraint updates**: In ADMM, the loss expressions `loss_i[jy]` depend on iteration-specific market prices `λ` (which change every iteration). Because JuMP expressions bake in coefficient values at creation time, the CVaR shortfall and linking constraints must be **deleted and re-added** in every ADMM iteration with the freshly recomputed loss expressions. This happens in the `solve_*_agent!` functions.

#### Specific objective terms by agent type

**VRES generator (with endogenous capacity and CVaR):**
```
min  γ × [ Σ_y loss_VRES[y] + F_cap × Σ_y cap_VRES[y] ]
   + (1−γ) × CVaR_VRES
   + (ρ_elec/2)×Σ W×(g − ḡ_elec)²
   + (ρ_GC/2)×Σ W×(g − ḡ_GC)²

where loss_VRES[y] = Σ_{h,d} W × ( MC×g − λ_elec×g − λ_GC×g )
```

**VRES in contracts case** (`build_power_agent_contracts.jl`): Splits generation into `g_EOM` (pool) and `g_ppa` (PPA). Loss includes `−λ_ppa×g_ppa`; penalties add `(ρ_ppa/2)×Σ W×(g_ppa − ḡ_ppa)²` and `(ρ_ppa_cap/2)×(ppa_cap − ḡ_ppa_cap)²`. Constraint: `g_ppa ≤ ppa_cap` at every hour.

**Conventional generator (3-stage increasing cost):**
```
min Σ W × ( Σ_s (base_s×g_s + 0.5×slope_s×g_s²) − λ_elec×g )  +  (ρ_elec/2)×Σ W×(g − ḡ_elec)²

subject to: g = Σ_s g_s,   0 ≤ g_s ≤ cap_s
```
where stage capacities/costs are built from `Capacity`, `StageCapacityShares`, `StageBaseCosts`, and `FinalMarginalCost` in `data.yaml`. Shares are normalized internally, and slopes are derived to ensure continuity across stage boundaries (`end MC_1 = base_2`, `end MC_2 = base_3`, `end MC_3 = FinalMarginalCost`). Changing shares changes the conventional fleet's aggregate average variable cost.

**Why these parameter values are used (default case):**
- `StageBaseCosts = [35, 55, 85]` €/MWh represents a stylized thermal merit order (lower-cost baseload-like block, medium-cost block, higher-cost flexible block).
- `FinalMarginalCost = 140` €/MWh captures the high-load tail where less efficient/flexible thermal generation sets marginal cost.
- Equal default shares (1/3 each) are neutral starting assumptions; users can rebalance shares to encode system-specific thermal composition.

**Why linear-within-stage is realistic enough:**
- Real unit commitment/economic dispatch stacks are piecewise and heterogeneous; aggregated systems are commonly approximated by piecewise-linear or piecewise-quadratic supply curves.
- Over a limited dispatch band per stage, a linear marginal-cost profile is a practical local approximation to heat-rate/fuel/emissions variability.
- Enforcing continuity across stage boundaries avoids artificial price jumps while preserving increasing scarcity cost as dispatch moves to higher-cost blocks.

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

**GreenProducer in contracts case** (`build_H2_agent_contracts.jl`): Uses `e_in_pool` (pool) and `g_ppa_from` (PPA). Loss includes `+λ_ppa×g_ppa_from`; conversion `h2_out = η×(e_in_pool + g_ppa_from)`; penalties add PPA terms `(ρ_ppa/2)×Σ W×(−g_ppa_from − ḡ_ppa)²` and `(ρ_ppa_cap/2)×(−ppa_cap − ḡ_ppa_cap)²`. It also sells `h2_hpa` under HPA with terms `−λ_hpa×h2_hpa`, `(ρ_hpa/2)×Σ W×(h2_hpa − ḡ_hpa)²`, and `(ρ_hpa_cap/2)×(hpa_cap − ḡ_hpa_cap)²`.

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

- All **price-dependent terms** (e.g. $\lambda_\text{elec}\,g$, $\lambda_\text{H2}\,h2\_in$) are expressed via JuMP `@expression` blocks whose coefficients are updated each ADMM iteration.
- The **capacity-investment linkage** is enforced via yearly variables (e.g. `cap_VRES[y]`, `inv_VRES[y]`) and simple linear relationships: investment in year $y$ expands the capacity available in all hours of that year.
- For risk-averse agents, the **loss-per-year** expressions `loss_VRES[y]`, `loss_H2[y]`, `loss_G[y]` are recomputed in every ADMM iteration with the iteration-specific prices, so that CVaR always measures risk against the latest price trajectory.

### 4.2 Key Constraints

| Constraint | Equation | Scope | Rationale |
|---|---|---|---|
| VRES capacity | `g ≤ AF × Capacity` | Per (h,d,y) | Generation limited by resource availability |
| Conventional staging | `g = Σ_s g_s`, `0 ≤ g_s ≤ cap_s` | Per (h,d,y) | Piecewise thermal stack with increasing stage costs and fixed total capacity |
| Consumer load | `d ≤ PeakLoad × load_profile` | Per (h,d,y) | Maximum consumption bound |
| H₂ conversion | `h2_out = η × e_in` | Per (h,d,y) | Stoichiometric mass/energy balance |
| H₂ GC physical limit | `gc_h2 ≤ h2_out` | Per (h,d,y) | Cannot certify more than produced |
| Green-backing (annual) | `Σ W×gc_elec ≥ (1/η)×Σ W×gc_h2` | Per year | Temporal flexibility in GC procurement |
| Green offtaker stoichiometry | `ep = (1/α) × h2_in` | Per (h,d,y) | No H₂ waste; tight conversion |
| GC mandate (green/grey) | `Σ W×gc_h2 ≥ γ_GC × Σ W×ep` | Per year | 42% renewable mandate |
| Grey GC mandate | `Σ W×gc_h2 ≥ γ_GC × γ_NH3 × Σ W×ep` | Per year | Only H₂-feedstock fraction |

### 4.3 Social Planner Objective

The social planner maximises **risk-adjusted social welfare** with a **single** social CVaR:

```math
\max \; \gamma \sum_y \text{sw\_aux}[y] \;-\; (1-\gamma)\,\text{CVaR}_\text{social}
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

```math
\text{social\_welfare}[y] = \sum_i \text{welfare\_per\_year\_i}[y]
```

Market-clearing constraints enforce supply = demand. The single social CVaR applies to the full aggregate welfare (including consumer utility), ensuring the risk-averse planner accounts for all welfare components when assessing tail risk.

### 4.4 Risk Aversion and Risk-Neutral Consistency

This section summarises the **risk-aversion architecture** and explains precisely when the **ADMM equilibrium** coincides with the **social planner** solution.

#### 4.4.1 Agent-level vs system-level risk

- In the **ADMM (market exposure) case**:
  - A subset of agents (VRES, electrolyzer, green offtaker) can be risk-averse with their own parameters $(\gamma_i,\beta_i)$.
  - Each such agent minimises a **private risk-adjusted loss**:

    ```math
    \gamma_i\,\mathbb{E}[\ell_i] + (1-\gamma_i)\,\mathrm{CVaR}_i(\ell_i),
    ```

    subject to its own technological constraints and the ADMM penalties.
  - Risk is therefore **heterogeneous and decentralised**: different agents may have different attitudes to risk; financial transfers between agents do not directly enter the risk measure.

- In the **social planner case**:
  - There is a **single** system-wide risk parameter $\gamma$ and confidence level $\beta$.
  - The planner maximises a **single risk-adjusted social welfare**:

    ```math
    \gamma\,\mathbb{E}\bigl[SW\bigr] - (1-\gamma)\,\mathrm{CVaR}_\text{social}(-SW),
    ```

    where $SW$ is aggregate welfare (including consumer utility and production/investment costs).
  - Risk is therefore **centralised**: society as a whole is risk-averse with respect to aggregate welfare, rather than each agent separately.

These two formulations represent different normative assumptions about **who bears risk** and **how it is shared**. The ADMM run with per-agent CVaR corresponds to a market in which agents individually care about their own tail losses; the social planner corresponds to a benevolent regulator who cares about systemic tail outcomes.

#### 4.4.2 Risk-neutral benchmark and equivalence

When both formulations are made **risk-neutral**, they collapse to the same underlying convex optimisation problem:

- In ADMM:
  - Set $\gamma_i = 1$ for all agents that can be risk-averse (VRES, electrolyzer, green offtaker).
  - This eliminates all per-agent CVaR terms from their objectives.

- In the social planner:
  - Set the planner-wide risk weight $\gamma = 1$.
  - This eliminates $\mathrm{CVaR}_\text{social}$ from the planner’s objective, so the model becomes a quadratic (but not quadratically constrained) welfare maximisation with standard consumer surplus and producer surplus terms.

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
- Numerical tolerances in the SP solver and ADMM solver (IPOPT for SP, Gurobi for ADMM).

These differences are typically negligible for economic interpretation and are visible in the diagnostic plots and CSVs.

---

## 5. ADMM Algorithm

### 5.1 Iteration Structure

Each ADMM iteration `k` proceeds as follows:

1. **For each agent** (via `ADMM_subroutine!`):
   a. Update consensus target: `ḡ_i = q_i^{k-1} − (1/(n+1)) × imbalance^{k-1}`
   b. Update prices `λ`, penalty `ρ` from the global ADMM state.
   c. Rebuild objective with updated parameters.
   d. For CVaR agents (VRES, electrolyzer, green offtaker): recompute loss expressions with iteration-specific `λ`, then delete and re-add CVaR shortfall and linking constraints with the fresh losses.
   e. Solve the agent's QP.
   f. Record the solution quantities.

2. **Compute market imbalances**: For each market, sum all agents' net positions. For EP, subtract fixed demand `D_EP`.

3. **Compute residuals**:
   - **Primal residual** = `‖imbalance‖₂` (L2 norm; measures market-clearing violation).
   - **Dual residual** = `‖ρ × Δ(consensus deviation)‖₂` (measures position stability).

   More precisely, for each market $k$:

   - Let $r_k^t(h,d,y)$ be the **market imbalance** at iteration $t$.
   - Let $\Delta z_k^t(h,d,y)$ be the **change in consensus deviation** (difference between successive consensus copies) at iteration $t$.

   Then:

   ```math
   \|r_k^t\|_2 = \Bigl(\sum_{h,d,y} r_k^t(h,d,y)^2\Bigr)^{1/2},\qquad
   \|s_k^t\|_2 = \rho_k^t\,\Bigl(\sum_{h,d,y} (\Delta z_k^t(h,d,y))^2\Bigr)^{1/2}.
   ```

   The primal residual $\|r_k^t\|_2$ measures **how far the market is from clearing**, while the dual residual $\|s_k^t\|_2$ measures **how stable the agents’ net positions are** from one iteration to the next.

4. **Update prices**: `λ^{k+1} = λ^k − η_k × ρ_k × imbalance^k` (dual ascent with **scale-aware residual damping** `η_k ∈ (0,1]` per market).  
   The base damping is computed from each market's residual level at that iteration relative to its Boyd-style tolerance scale (`ε_pri_k`, `ε_dual_k`), so step behavior remains robust when horizon size changes (e.g., 1-year vs 10-scenario runs).

5. **Update ρ** (residual balancing): For each market independently:
   - if `primal > μ × dual` -> increase `ρ`,
   - if `dual > μ × primal` -> decrease `ρ`,
   - else keep `ρ` unchanged.
   Market-specific multiplicative rates and bounds are applied.

6. **Capacity ρ update**: The same residual-balancing rule is applied per capacity agent `m` with local residuals `r_m`, `s_m`.

7. **Convergence check**: All five markets must have both primal and dual residuals below their tolerance.

### 5.2 Consensus Formula (Sharing ADMM)

The consensus target for agent `i` in a market with `n` participants:

```
ḡ_i^k = q_i^{k-1} − (1/(n+1)) × Σ_j q_j^{k-1}
```

The `(n+1)` denominator comes from the sharing ADMM formulation, which introduces one "market copy" alongside the `n` agent copies. This distributes the imbalance correction equally.

### 5.3 Adaptive Penalty (ρ)

`update_rho.jl` and `update_rho_contracts.jl` implement a minimal residual-balancing controller.

For each market (and each capacity agent, see §5.4), after residuals are computed:

- if `r_p > μ r_d`, increase `ρ`;
- if `r_d > μ r_p`, decrease `ρ`;
- otherwise keep `ρ` unchanged.

The threshold is `μ = 1.2` and can be passed through ADMM state as `rho_balance_threshold`.

This rule is used because it is:

1. **ADMM-consistent**: direct Boyd-style residual balancing.
2. **Deterministic**: no auxiliary controller states are required to interpret behavior.
3. **Stable under coupling**: every market follows the same single control law.

`ρ` still has market-specific multiplicative rates and caps to reflect numerical stiffness by market type.

#### 5.3.1 Per-market parameters

| Market | Increase factor | Decrease factor | ρ_max | Reasoning |
|---|---|---|---|---|
| `elec`, `elec_GC` | 1.05 | 1/1.05 | 5,000 | Large-volume core markets; can tolerate moderately faster adaptation. |
| `H2`, `EP` | 1.01 | 1/1.01 | 100 | More kink-sensitive due to coupling and capacity effects; slower updates reduce oscillation risk. |
| `H2_GC` | 1.05 | 1/1.05 | 100 | Thin but hourly market; moderate adaptation with conservative cap. |
| `ppa`, `ppa_cap` | 1.05 | 1/1.05 | 500 | Thin bilateral pool; conservative but responsive. |
| `hpa`, `hpa_cap` | 1.05 | 1/1.05 | 500 | Same logic as PPA for hydrogen contracts. |

Capacity consensus is a per-agent equality split (§5.4), so each capacity-owning agent has its own `ρ_cap[m]` update with the same residual-balancing rule.

#### 5.3.2 Pseudo-code

```text
for each flow market k:
    rp = primal_residual(k)
    rd = dual_residual(k)
    if rp > μ*rd:
        ρ_k <- min(ρ_max_k, τ_k * ρ_k)
    elseif rd > μ*rp:
        ρ_k <- max(ρ_min_k, ρ_k / τ_k)
    else
        ρ_k <- ρ_k

for each capacity-owning agent m:
    rp_m = ||x_cap_m - z_cap_m||
    rd_m = ||ρ_m * (z_cap_m^k - z_cap_m^{k-1})||
    if rp_m > μ*rd_m:
        ρ_m <- min(ρ_max_cap, τ_cap * ρ_m)
    elseif rd_m > μ*rp_m:
        ρ_m <- max(ρ_min_cap, ρ_m / τ_cap)
    else
        ρ_m <- ρ_m
```

### 5.4 Capacity ADMM (Equality Split per Agent)

Capacity consensus is treated as a **textbook ADMM equality split** at the agent level, not as a soft penalty against a derived target. This subsection gives the formal model, residual definitions, and the rationale for each design choice.

#### 5.4.1 Formal model

For every capacity-owning agent `m ∈ {VRES, GreenProducer, GreenOfftaker}` and year `y` we introduce:

- **Primal** `x_{m,y}` — the agent's own capacity variable (`cap_VRES`, `cap_H2_y`, or `cap_EP_y` depending on agent type);
- **Auxiliary** `z_{m,y}` — a capacity target derived from the agent's latest realized flow profile (fallback: ADMM flow target if no history yet). For VRES, `z = max_{h,d} g_elec/AF`; for electrolyzers, `z = max_{h,d} h2_out`; for green offtakers, `z = max_{h,d} ep`. In the contracts case, VRES and electrolyzer targets use pool + contract flows.
- **Dual** `λ_{m,y}` — Lagrange multiplier for the equality `x_{m,y} = z_{m,y}`;
- **Per-agent penalty** `ρ_m` — scalar weight, one per agent.

The agent solves, each ADMM iteration:

```
x_m^k = argmin_{x ≥ 0}  f_m(x, ...)
                       + Σ_y [ λ_{m,y}^{k-1} · (x_y - z_{m,y}^k)
                              + (ρ_m^{k-1}/2) · (x_y - z_{m,y}^k)² ]
```

where `f_m` is the agent's own economic loss (operational + CAPEX − revenue, plus CVaR / risk and other ADMM market penalties). After all agents solve we perform the **dual ascent**:

```
λ_{m,y}^k = λ_{m,y}^{k-1} + ρ_m^{k-1} · (x_{m,y}^k - z_{m,y}^k)
```

This is the standard equality-split ADMM update (Boyd et al. 2011, §3.1).

#### 5.4.2 Residuals

Per-agent residuals follow the Boyd definition for the `x = z` split:

```
Primal:  r_m^k = || x_m^k - z_m^k ||_2     (over years)
Dual:    s_m^k = || ρ_m^{k-1} · (z_m^k - z_m^{k-1}) ||_2     (Δz, not Δx)
```

The dual residual uses the change in the **auxiliary** `z` (not the change in the primal `x`). This is the ADMM-correct definition: if `x` has frozen but `z` is still drifting, a Δx-based residual would falsely declare convergence; Δz captures the true ADMM dual progress (Boyd et al. 2011, Eq. 3.12).

For diagnostics and the one-line summary, the aggregate residuals reported in `ADMM_Convergence.csv` as `cap_primal` / `cap_dual` are the L2 norms over agents:

```
r_cap^k = sqrt(Σ_m r_m^k²),    s_cap^k = sqrt(Σ_m s_m^k²)
```

#### 5.4.3 Stopping rule

Convergence is checked **per agent**, not on the aggregate. For each capacity-owning agent the Boyd absolute + relative test:

```
ε_pri_m  = ε_abs · sqrt(n_yr) + ε_rel · ResidualScale_Primal_m
ε_dual_m = ε_abs · sqrt(n_yr) + ε_rel · ResidualScale_Dual_m
```

with `ResidualScale_*_m` initialised from the first non-zero observation per agent. Capacity is converged iff `r_m ≤ ε_pri_m` and `s_m ≤ ε_dual_m` for **every** `m`. *Why per-agent and not aggregate*: averaging residuals across agents can hide a single laggard whose split is still far from feasibility; an aggregate test would declare convergence even when one agent type (e.g. a strongly binding electrolyzer) has not satisfied the equality. The per-agent test is direction-correct: capacity is "done" when every agent's split is satisfied.

The optional knob `cap_tol_relax` (default 100 in the contracts case) multiplies the right-hand side of the per-agent test; see §5.7.

#### 5.4.4 Per-agent ρ controller

Each agent now follows the same minimal residual-balancing rule as §5.3.4, but applied per agent `m`:

- if `r_m > μ s_m` -> increase `ρ_m`,
- if `s_m > μ r_m` -> decrease `ρ_m`,
- else keep `ρ_m`.

Default per-agent parameters: increase factor 1.05, decrease factor 1/1.05, `ρ_max = 30`, `ρ_min = 0.10`. Configurable via `data.yaml`:

```yaml
ADMM:
  rho_cap_initial: 0.1
  rho_cap_inc_factor: 1.05
  rho_cap_max: 30.0
```

Why this simplification makes sense for capacity: capacity markets are strongly coupled to other markets, so a single residual-ratio rule keeps each capacity split controller interpretable and consistent across agents.

#### 5.4.5 Why the equality split

The capacity block uses the augmented-Lagrangian equality split:

```
L_cap = λ_cap · (x - z) + (ρ_cap/2) · (x - z)^2
```

This structure is used because:

1. the linear dual term provides first-order correction toward `x = z`;
2. the quadratic term provides curvature and numerical regularization;
3. both terms are in currency units and vanish at consensus.

With dual ascent `λ <- λ + ρ (x-z)`, the split follows textbook ADMM dynamics for equality constraints.

#### 5.4.6 Why per-agent ρ (and not a single global ρ_cap)

A single global `ρ_cap` forces a compromise across very heterogeneous agent types:

- **VRES**: capacity grows by tens of MW per step; CAPEX is moderate; the binding constraint is `cap ≥ peak(g)/AF`.
- **Electrolyzer**: capacity is tightly coupled to four markets simultaneously (elec, elec_GC, H2, H2_GC); CAPEX is high; the binding constraint is `cap ≥ peak(h2_out)`.
- **Green offtaker**: capacity is decoupled from elec but tied to EP flow; CAPEX is small relative to operational margin.

When the controller picks a single `ρ_cap` that suits one of these, the others sit either in the dead band (no progress) or over the kink (limit cycles). Per-agent `ρ_m` removes this compromise; each agent's controller specialises to its own residual scale.

#### 5.4.7 Units check (and why penalties don't bias the social-planner equivalence)

The economic loss `f_m` is in € (currency); `λ · (x - z)` has units `[€/MW] · [MW] = €`; `(ρ/2) · (x - z)²` has units `[€/MW²] · [MW²] = €`. All three terms add cleanly. Because the ADMM penalty and dual terms vanish exactly at consensus (`x = z`), they have no effect on the centralised social-planner optimum: the planner does not solve a per-agent subproblem, hence has no `λ_cap` / `ρ_cap` / `z_cap` parameters (`add_*_to_planner!` functions never touch these). At γ = 1 (risk-neutral) the ADMM equilibrium therefore converges to the same primal/dual solution as the planner by the first welfare theorem — see §5.5 for the SP warm-start that we rely on for fast convergence.

#### 5.4.8 Iteration order in the main loop

Each ADMM iteration `k` for the capacity block runs:

1. **Derive `z^k`**: `ADMM_subroutine` computes `z_m^k` for every cap agent from realized flow histories (fallback to ADMM targets when history is not yet available) and pushes it to history (`ADMM["Capacity"]["z"][m]`).
   - `z` uses optional under-relaxation
     `z^k <- α·z_raw^k + (1-α)·z^{k-1}` with `α = cap_z_relax` (default 1.0 = off),
     then re-projects (i) nondecreasing-by-year capacity and (ii) the agent's model-feasible minimum installed-capacity floor.
2. **Set parameters on agent model**: `:z_cap = z_m^k`, `:λ_cap = λ_m^{k-1}` (read from history), `:ρ_cap = ρ_m^{k-1}` (read from history).
3. **Agent solves**: produces `x_m^k`.
4. **Dual ascent**: `λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k)`, pushed to history.
5. **Residuals**: `r_m^k`, `s_m^k` computed and pushed.
6. **Controller**: `update_rho!` updates `ρ_m^k` per agent using residual balancing.
7. **Convergence**: per-agent test (§5.4.3).

This ordering is identical for `market_exposure` and `market_exposure_contracts`; only the `z` derivation differs (the contracts case adds the PPA / HPA flow contributions when computing the peak of `g_bar + g_bar_ppa`, etc.).

**Why this choice (`z` under-relaxation):**

In tightly-coupled runs, raw `z` targets can jump sharply when flow consensus oscillates across markets. Because the capacity dual residual uses `Δz`, these jumps can produce very large `s_m` and trigger controller overreaction even when `x` is moving in the right direction. Under-relaxation damps target motion, reducing artificial dual spikes and improving monotonic progress toward the split fixed point.

`z` projection enforces feasibility against the model structure: year-to-year monotonic capacity and minimum installed-capacity floor implied by nonnegative investment variables.

### 5.5 Convergence Tolerances (Boyd-style)

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

#### Relative tolerance ε_rel

The optional `ε_rel` term adds a scale-relative component. When `ε_rel > 0`, markets with larger typical residual magnitudes get proportionally larger tolerances. Set `epsilon_rel: 0.01` in `data.yaml` to enable a 1% relative tolerance. The default is `0.0`.

#### Choosing ε and recommended values

The choice of `ε_abs` (or `epsilon` in `data.yaml`) trades off convergence speed vs. price/quantity accuracy:

- smaller `ε_abs` -> tighter residuals, closer dual prices to benchmark;
- larger `ε_abs` -> earlier stopping with looser dual accuracy.

Because the stopping test scales with `sqrt(n_slots)`, effective absolute tolerance grows with horizon size. For multi-scenario runs (e.g. `24*8*10` slots), use smaller `epsilon` when close price agreement with the social planner is required.

#### Two epsilon values: `epsilon` vs `epsilon_contracts`

The **contracts case** (`market_exposure_contracts`) has more coupled markets (standard flows + contract energy + contract capacity + capacity consensus) and stronger interdependence (VRES splits pool vs contract; electrolyzer does the same). As a result, convergence is slower and residuals tend to be larger than in `market_exposure`. To avoid running to `max_iter` without declaring convergence when results are already good enough, the contracts case uses a separate tolerance:

- **`epsilon`** — Used by `market_exposure`.
- **`epsilon_contracts`** — Used by `market_exposure_contracts` when set in `data.yaml`. If not set, the contracts case falls back to `epsilon`.

Both cases use the same convergence logic; only the tolerance value differs. The capacity consensus in the contracts case additionally uses `cap_tol_relax` (see §5.7).

### 5.6 Warm-start from Social Planner

Warm-starting ADMM from the social planner solution is **critical** for fast convergence. Without it, agents start with zero consensus targets and zero capacity seeds, which biases them toward suboptimal quantities.

Three warm-start components: (1) **Price (λ)**: Load hourly prices from `Market_Prices.csv`. (2) **Primal (quantities)**: Load from `SP_Primal_Quantities.csv` so iteration 1 has ḡ = SP; without this, ḡ = 0 biases agents toward zero. (3) **Capacity**: Load from `SP_Capacities.csv`, set `set_start_value` on capacity variables, and preload the capacity ADMM auxiliary state `z_cap` so iteration 1 starts with `x_cap` and `z_cap` aligned around the SP solution. Run `social_planner.jl` first, then `market_exposure.jl`. A single message is printed: `ADMM warm-start: λ from SP prices, primal quantities from SP, capacity seeds for N agents`.

### 5.7 Contract Pools ADMM (market_exposure_contracts.jl)

In the contracts case, the ADMM loop (`ADMM_contracts.jl`) extends the standard loop with:

1. **PPA energy imbalance** per VRES sub-market: supplier (`g_ppa`) vs buyer (`g_ppa_from`).
2. **HPA energy imbalance** per GreenProducer sub-market: supplier (`h2_hpa`) vs buyer (`h2_hpa_from`).
3. **Capacity consensus** for both pools: scalar imbalance between supplier `+cap` and buyer `-cap`.
4. **Price updates**: `λ_ppa` and `λ_hpa` update like other 3D prices; capacity consensus has no separate price.
5. **ρ adaptation** (`update_rho_contracts.jl`): `ppa/ppa_cap` and `hpa/hpa_cap` follow the same residual-balancing logic (inc 1.05, dec 1/1.05, ρ_max 500).

**Relaxed tolerances for the contracts case.** Because the contracts case has more coupled markets and stronger interdependence (VRES splits pool vs contract; capacity consensus depends on both `g_bar_elec` and `g_bar_ppa`), two additional parameters relax convergence criteria:

- **`epsilon_contracts`** — Contracts-base tolerance for all flow markets. See §5.5 *Two epsilon values*.
- **`cap_tol_relax`** — Multiplier for the capacity consensus tolerance. Effective cap tolerance = standard (ε_pri, ε_dual) × `cap_tol_relax`. Default 100. This allows convergence when flow markets have cleared even if capacity consensus lags, since capacity is tightly coupled to flows that are still settling.

For details on how both pools choose contract capacities under pay-as-produced logic, see §2 *Contract pools* → *How contract capacity is determined*.

### 5.8 Sign Convention

| Role | Net position sign | Example |
|---|---|---|
| Supplier / seller | **Positive** | VRES generation `+g`, H₂ sales `+h2_out` |
| Buyer / consumer | **Negative** | Electricity demand `−d`, H₂ purchase `−h2_in` |

Market imbalance = Σ (net positions). Positive imbalance = excess supply → price decreases. Negative imbalance = excess demand → price increases.

### 5.9 Practical Convergence Behavior and Monotonicity

With coupled multi-market ADMM (especially with endogenous investments and contract couplings), strict **per-iteration monotonic decrease** of every residual is generally not guaranteed. What the controller is designed to guarantee in practice is stronger **best-so-far progress** and anti-stall recovery:

- Residual merit is tracked in normalized form (relative to market-specific Boyd tolerances).
- If short-term residual motion worsens, per-market dual step scales are reduced automatically.
- If a long stall/worsening phase is detected, the algorithm restarts from the best checkpoint found so far and continues with smaller steps.

This design avoids the common "improve-then-wander" ADMM behavior in hard regimes while preserving convergence speed in easy regimes. In empirical runs, this yields:

- fast coarse convergence in early iterations,
- fewer late oscillation plateaus,
- improved ability to reach tighter `epsilon` values without inflating tolerance.

---

## 6. Social Planner Benchmark

The social planner (`social_planner.jl`) solves a single centralised convex QCP (quadratically constrained program) that maximises risk-adjusted social welfare subject to all individual agent constraints plus market-clearing balance constraints. It serves as the theoretical first-best benchmark. When `γ=1` (risk-neutral), the CVaR term drops from the objective economically; the same unified epigraph/QCP structure is kept across runs.

### 6.1 Market-Clearing Constraints

| Constraint | Equation |
|---|---|
| Electricity balance | `Σ generation − Σ demand − Σ electrolyzer_elec_buy = 0` (per h,d,y) |
| Elec GC balance | `Σ VRES_generation − Σ electrolyzer_GC_buy − Σ GC_demand = 0` (per h,d,y) |
| H₂ balance | `Σ H₂_production − Σ H₂_consumption − Σ offtaker_H₂_buy = 0` (per h,d,y) |
| H₂ GC balance | `Σ H₂_GC_supply − Σ H₂_GC_demand = 0` (per h,d,y — hourly, same as other markets) |
| EP balance | `Σ offtaker_EP_supply − D_EP − Σ EP_demand = 0` (per h,d,y) |

### 6.2 Price Recovery (Direct QCP Duals)

Equilibrium prices are the **dual variables** (shadow prices) of the market-clearing constraints.
The planner is solved directly as a convex QCP with IPOPT (SP-only solver).

Workflow:

1. **QCP solve**: Solve the full social-planner QCP. Accept `OPTIMAL` and `LOCALLY_SOLVED` (convex QCP).
2. **Dual availability check**: Require `has_duals(planner) == true`. If duals are unavailable, the run fails and the solver/settings must be changed.
3. **Price extraction**: Read duals of the five balance constraints and divide by representative-day weight `W[jd,jy]` to recover per-MWh prices written to `Market_Prices.csv`.

Primal quantities and capacities are read from the same solved QCP model; no reformulation or proxy stage is used in the benchmark pipeline.

Why IPOPT (and not Gurobi) for SP duals:
- In this project’s large/scaled SP QCP instances, Gurobi can return primal-optimal status (`LOCALLY_SOLVED`) while still failing to expose usable QCP duals after tightened barrier settings.
- The social planner benchmark requires reliable dual multipliers for market-price comparison; IPOPT delivers these multipliers directly for the solved QCP in this workflow.
- ADMM remains on Gurobi for subproblem performance and existing calibration; only SP solver is switched.

#### ADMM note on capacity tolerance scaling

In `market_exposure` ADMM, flow-market tolerances use Boyd-style horizon scaling (`ε_abs * sqrt(n_slots) + ε_rel * scale`), where `n_slots = nHours × nReprDays × nYears`.  
Capacity consensus is not a full flow tensor; it is low-dimensional (yearly scalar/vector). Therefore, capacity convergence is checked on a scalar basis (`sqrt(1)`), avoiding premature convergence declarations from over-loose `sqrt(n_slots)` scaling on the capacity channel.

### 6.3 Code Architecture

All problem definition lives in `Source/build_*.jl` files. Each file contains:

- `build_*_agent!()` — Builds the ADMM version (with `λ`, `ρ`, `ḡ` penalty terms and per-agent CVaR for risk-averse agents).
- `add_*_agent_to_planner!()` — Adds the same variables/constraints to the planner model **without** ADMM terms and **without** per-agent CVaR. Returns a `Dict{Int, Any}` of per-year welfare expressions.

`build_social_planner.jl` orchestrates the calls to all `add_*_to_planner!` functions, adds market-clearing constraints, aggregates per-year welfare into `social_welfare`, adds the epigraph formulation and single social CVaR, and sets the risk-adjusted objective.

### 6.4 Epigraph Formulation for Social CVaR

The social planner applies **one single CVaR** to the aggregate social welfare (not per-agent CVaR). This ensures risk aversion considers all welfare components (consumer utility, production costs, investment costs) holistically.

**Problem**: `social_welfare[y]` includes quadratic terms from both elastic demand utility (`A·d − B/2·d²`) and conventional stage costs (`base_s·q_s + 0.5·slope_s·q_s²`). Putting `−social_welfare[y]` inside the CVaR shortfall constraint creates a nonlinear coupling in the shortfall constraints and complicates robust dual extraction/interpretation for market prices.

**Solution — epigraph reformulation**: Introduce auxiliary variables `sw_aux[y]` with epigraph constraints:

```
sw_aux[y] ≤ social_welfare[y]     (quadratic constraint, standard convex form)
```

The CVaR constraints then reference `sw_aux` instead of the quadratic `social_welfare`, making them purely linear:

```
u_social[y]  ≥ −sw_aux[y] − α_social                          ∀ y ∈ JY
cvar_social  ≥ α_social + (1/(1−β)) × Σ_y P[y] × u_social[y]
```

**Important**: `α_social` and `cvar_social` must be **free** (no lower bound). When social welfare is positive, the social loss = −sw_aux is negative. The optimal VaR α for CVaR of a negative loss is negative. With α ≥ 0, cvar_social would be forced ≥ 0, so the objective would become γ·sw_aux instead of sw_aux when γ < 1 — breaking SP/ME equivalence for nYears=1. With α free, CVaR = loss when nYears=1, so the objective reduces to sw_aux regardless of γ.

The objective is also linear:

```
max  γ × Σ_y sw_aux[y]  −  (1−γ) × cvar_social
```

Since the objective maximises `sw_aux`, the epigraph constraint binds at optimality (`sw_aux[y] = social_welfare[y]`), making the formulation mathematically equivalent to applying CVaR directly to `social_welfare`.

The epigraph constraints are the **only** quadratic constraints in the model (convex QC form). All other constraints (CVaR, market-clearing, capacity bounds) are purely linear. Prices are extracted directly from QCP duals as described in §6.2.

### 6.5 Investment Decisions: Social Planner vs. Market Exposure

Both the social planner and market exposure include **endogenous investment** in VRES capacity (`cap_VRES`), electrolyzer H₂ capacity (`cap_H2_y`), and green offtaker EP capacity (`cap_EP_y`). The formulations are structurally identical:

- **Social planner**: Each agent's capacity variables are added to the centralised planner model. The planner optimises all quantities and capacities jointly in a single optimisation. Market-clearing constraints enforce supply = demand. The optimal capacities emerge from the welfare-maximising solution.

- **Market exposure (ADMM)**: Each agent has its own capacity variables in its decentralised model. Agents optimise independently, but they must agree on a **consensus capacity** via an ADMM penalty: each agent minimises `(ρ_cap/2) × (cap − cap_bar)²`, where `cap_bar` is the capacity implied by the flow consensus (e.g. for VRES: `cap_bar[y] = max over (h,d) of g_bar[h,d,y] / AF[h,d,y]`). At convergence, all agents choose the same capacity and `cap_bar` matches the agreed-upon level.

**Why warm-start matters for investment**: Without capacity warm-start from the SP, the first ADMM iteration has `cap_bar` derived from zero flows (ḡ = 0), so `cap_bar = 0`. Agents are then penalised toward zero capacity, which is far from the equilibrium. With SP capacity seeds (`set_start_value`) and primal warm-start (ḡ = SP), `cap_bar` is consistent with SP flows and the capacity penalty pulls agents toward the SP investment levels from the first iteration. This dramatically speeds convergence of the investment consensus.

---

## 7. Data and Indexing

### 7.1 Temporal Dimensions

| Dimension | Set | Size | Description |
|---|---|---|---|
| Hours | `JH = 1:nTimesteps` | 24 | Hours within each representative day |
| Representative days | `JD = 1:nReprDays` | 8 | Representative days (configured in `data.yaml`) |
| Years | `JY = 1:nYears` | typically 10 scenarios | Active horizon uses `ADMM.nScenarioYears` when set (for both SP and ADMM); otherwise falls back to `General.nYears` |

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
| `nReprDays` | 8 | Representative days (trade-off: speed vs. accuracy) |
| `nYears` | 1 | Base-year horizon used by `social_planner.jl` |
| `base_year` | 2021 | Calendar year for timeseries data |

### 8.2 ADMM

| Parameter | Value | Description |
|---|---|---|
| `rho_initial` | 1.0 | Default penalty weight (neutral starting point) |
| `nScenarioYears` | 10 | Scenario years used by `market_exposure*.jl` (e.g., 2021..2030) |
| `max_iter` | 200 | Maximum ADMM iterations |
| `epsilon` | 0.2 | Convergence tolerance for `market_exposure`; see §5.5 for accuracy/speed trade-off. |
| `epsilon_contracts` | 1.0 | [market_exposure_contracts only] Contracts tolerance; if unset, falls back to `epsilon`. |
| `cap_tol_relax` | 100 | [market_exposure_contracts only] Multiplier for capacity consensus tolerance. See §5.7. |
| `rho_cap_initial` | 0.1 | Initial per-agent capacity penalty for the equality split (§5.4). |
| `rho_cap_inc_factor` | 1.05 | Per-agent capacity controller increase factor; decrease factor is the reciprocal. See §5.4.4. |
| `rho_cap_max` | 30 | Per-agent capacity penalty upper bound. See §5.4.4 for justification. |
| `cap_z_relax` | 1.0 | Under-relaxation factor for capacity target update `z^k <- α z_raw^k + (1-α) z^{k-1}`. `1.0` disables damping (default). Use `0.2–0.8` only if target oscillations cause large `Δz` dual spikes. See §5.4.8. |

### 8.3 Market Parameters

| Market | `initial_price` | `rho_initial` | Notes |
|---|---|---|---|
| `elec_market` | 80.0 €/MWh | 1.0 | Electricity price seed for ADMM warm start |
| `elec_GC_market` | 10.0 €/MWh_GC | 0.3 | Electricity certificate market seed |
| `H2_market` | 50.0 €/MWh_H2 | 0.5 | Hydrogen market seed |
| `H2_GC_market` | 30.0 €/MWh_GC | 1.0 | Hydrogen certificate market seed |
| `EP_market` | 150.0 €/MWh_EP | 3.0 | End-product market seed; also has `Demand_Column`, `Total_Demand` |

### 8.4 Contracts (market_exposure_contracts.jl only)

| Block | Parameter | Value | Description |
|---|---|---|---|
| `PPAs` | `initial_price` | 60.0 €/MWh | Seed for `λ_ppa` (pay-as-produced electricity+elec_GC) |
| `PPAs` | `rho_initial` | 0.5 | ADMM penalty seed for PPA pool |
| `HPAs` | `initial_price` | 60.0 €/MWh_H2 | Seed for `λ_hpa` (pay-as-produced hydrogen+H2_GC equivalent) |
| `HPAs` | `rho_initial` | 0.5 | ADMM penalty seed for HPA pool |

PPA and HPA both clear energy at their λ prices and enforce scalar capacity consensus (`ppa_cap`, `hpa_cap`) with no separate capacity price.

### 8.5 Agent Parameters

See `Data/data.yaml` for the full annotated configuration. Key parameters:

- **VRES**: `Capacity`, `Profile_Column`, `MarginalCost`
- **Conventional**: `Capacity`, `StageCapacityShares`, `StageBaseCosts`, `FinalMarginalCost` (`MarginalCost` kept only as legacy fallback)
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
├── market_exposure.jl          # Entry point: distributed ADMM simulation (5 markets)
├── market_exposure_contracts.jl # Entry point: ADMM with bilateral PPA + HPA contracts
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
│   ├── timeseries_2021.csv     # Representative-day hourly profiles (SOLAR, LOAD_E, LOAD_H, LOAD_EP, WIND)
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
│   ├── update_rho.jl                 # Adaptive penalty update (Boyd rule with 3 regimes)
│   ├── save_results.jl               # Write market-exposure CSV outputs
│   ├── ADMM_contracts.jl             # ADMM loop with PPA + HPA pools
│   ├── ADMM_subroutine_contracts.jl  # Per-agent step with PPA/HPA g_bar/λ/ρ
│   ├── update_rho_contracts.jl      # Adaptive penalty update including ppa/hpa and cap consensuses
│   ├── build_power_agent_contracts.jl # VRES with g_EOM, g_ppa, ppa_cap
│   ├── build_H2_agent_contracts.jl  # GreenProducer with PPA buy-side + HPA sell-side
│   ├── build_offtaker_agent_contracts.jl # GreenOfftaker with HPA buy-side
│   ├── define_contract_parameters.jl # Contract market flags and parameters
│   ├── define_contract_market_parameters.jl
│   ├── define_results_contracts.jl   # Results and ADMM state for PPA + HPA pools
│   ├── save_results_contracts.jl     # PPAs.csv, HPAs.csv, Green_Agents_Detail.csv, ADMM outputs
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
│   ├── Agent_Summary.csv             # Agent group membership and ADMM objective value
│   ├── Agent_Quantities_Final.csv    # Final-iteration net quantities per agent
│   ├── Offtaker_GC_Diagnostics.csv   # GC compliance per offtaker
│   ├── H2_Producer_Diagnostics.csv   # H₂ GC-to-production ratio
│   ├── Capacity_Investments.csv      # VRES/electrolyzer/green offtaker yearly capacity & investment (ADMM)
│   └── TimerOutput.yaml              # Profiling data
│
├── market_exposure_contracts_results/ # Output from market_exposure_contracts.jl
│   ├── ADMM_Convergence.csv          # Same as market_exposure + PPA/HPA + cap-consensus columns
│   ├── ADMM_Diagnostics.csv          # Same + PPA/HPA + cap-consensus diagnostics
│   ├── Electricity_Market_History.csv
│   ├── Hydrogen_Market_History.csv
│   ├── Electricity_GC_Market_History.csv
│   ├── H2_GC_Market_History.csv
│   ├── End_Product_Market_History.csv
│   ├── Agent_Summary.csv             # Same structure as market_exposure (no contract columns)
│   ├── Market_Prices.csv             # Same + per-submarket PPA/HPA price columns
│   ├── PPAs.csv                      # Per-VRES PPA summary
│   ├── HPAs.csv                      # Per-GreenProducer HPA summary
│   └── Green_Agents_Detail.csv       # Detailed PPA breakdown for VRES and GreenProducer
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
| `market_exposure_contracts.jl` | Entry point for ADMM with bilateral PPA + HPA contracts. Same structure as market_exposure but uses contract-specific modules: define_contract_parameters, define_contract_market_parameters, define_results_contracts, build_power_agent_contracts, build_H2_agent_contracts, build_offtaker_agent_contracts, ADMM_contracts, save_results_contracts. Outputs to `market_exposure_contracts_results/`. |
| `social_planner.jl` | Entry point for centralised benchmark. Sections 1–12: same structure as market_exposure but builds a single planner model instead of per-agent models + ADMM loop. Section 11 solves the planner as a convex QCP with IPOPT and requires direct dual availability. |

### 10.2 Parameter Definition Files

| File | Role |
|---|---|
| `define_common_parameters.jl` | Creates `mod.ext` dictionaries (sets, parameters, timeseries, variables, constraints, expressions). Fills JH/JD/JY, W, P, γ, β. Determines market participation from agent type. Pre-allocates ADMM placeholder arrays. |
| `define_power_parameters.jl` | VRES: capacity, AF profile. Conventional: capacity, AF=1, 3-stage cost curve (`ConvStageCap`, `ConvStageBaseCost`, `ConvStageSlope`) built from share + absolute stage-cost inputs (coal/biomass/NG style), with no global average-cost rescaling. Consumer: PeakLoad, LOAD_E profile, A_E, B_E. |
| `define_H2_parameters.jl` | Electrolyzer: Capacity_Electrolyzer, Capacity_H2_Output, SpecificConsumption, OperationalCost, η_elec_H2. |
| `define_offtaker_parameters.jl` | Copies all keys from agent block; sets gamma_GC = 0.42 (regulatory mandate). |
| `define_elec_GC_demand_parameters.jl` | PeakLoad, Load_Column, A_GC, B_GC, LOAD_GC timeseries. |
| `define_EP_demand_parameters.jl` | Placeholder; copies EP_Demand block if present. |
| `define_*_market_parameters.jl` | Each market: name, initial_price, rho_initial, prices list. EP market also builds 3D demand array D_EP. |
| `define_results.jl` | Initialises results["λ"], per-agent quantity buffers, ADMM ρ lists, Imbalances, PriceHistory, ImbalanceMean, Residuals, Tolerance. |

### 10.3 Model Building Files

| File | ADMM Function | Planner Function |
|---|---|---|
| `build_power_agent.jl` | `build_power_agent!()` — power agents (VRES with capacity & CVaR, conventional with stage dispatch variables and convex stage costs, consumer) | `add_power_agent_to_planner!()` — same constraints and returns `Dict{Int, Any}` of per-year welfare (no per-agent CVaR) |
| `build_H2_agent.jl` | `build_H2_agent!()` — electrolyzer with 4-market ADMM terms, endogenous capacity & CVaR | `add_H2_agent_to_planner!()` — same constraints, returns per-year welfare = −op_cost − fixed CAPEX (no per-agent CVaR) |
| `build_offtaker_agent.jl` | `build_offtaker_agent!()` — green/grey/importer (green with EP capacity & CVaR) | `add_offtaker_agent_to_planner!()` — same constraints, returns per-year welfare = −processing/import cost − fixed CAPEX (no per-agent CVaR) |
| `build_elec_GC_demand_agent.jl` | `build_elec_GC_demand_agent!()` — GC demand with ADMM | `add_elec_GC_demand_agent_to_planner!()` — returns per-year utility expression |
| `build_EP_demand_agent.jl` | `build_EP_demand_agent!()` — placeholder | `add_EP_demand_agent_to_planner!()` — returns per-year utility expression |
| `build_power_agent_contracts.jl` | `build_power_agent_contracts!()` — VRES with g_EOM, g_ppa, ppa_cap; conventional/consumer delegate to build_power_agent! | — (contracts case only; planner unchanged) |
| `build_H2_agent_contracts.jl` | `build_H2_agent_contracts!()` — GreenProducer with PPA buy-side and HPA sell-side | — (contracts case only) |
| `build_offtaker_agent_contracts.jl` | `build_offtaker_agent_contracts!()` — GreenOfftaker with HPA buy-side (others delegate to base build_offtaker_agent!) | — (contracts case only) |
| `build_social_planner.jl` | — | `build_social_planner!()` — orchestrates all add_*_to_planner!, adds balance constraints, aggregates welfare, adds epigraph + single social CVaR, sets risk-adjusted objective |

### 10.4 Solve Files

| File | Role |
|---|---|
| `solve_power_agent.jl` | Rebuilds objective with iteration-specific λ, ḡ, ρ. For VRES: recomputes loss expressions with iteration-specific λ, deletes and re-adds CVaR shortfall/linking constraints. For conventional: applies the 3-stage convex variable cost (with legacy linear-MC fallback only if stage inputs are absent). Calls `optimize!`. |
| `solve_H2_agent.jl` | Rebuilds objective with iteration-specific λ, ḡ, ρ. Recomputes loss expressions with iteration-specific λ (4-market), deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_offtaker_agent.jl` | Rebuilds objective for green/grey/importer. For GreenOfftaker: recomputes loss expressions with iteration-specific λ, deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_elec_GC_demand_agent.jl` | Rebuilds utility − expenditure + ADMM penalty; calls `optimize!`. |
| `solve_EP_demand_agent.jl` | Placeholder; just calls `optimize!`. |

### 10.5 ADMM Files

| File | Role |
|---|---|
| `ADMM.jl` | Main loop: iterate agents → imbalances → primal/dual residuals → scale-aware price update → adaptive ρ update → convergence check. Progress bar + summary printout. |
| `ADMM_subroutine.jl` | Per-agent step: update g_bar/λ/ρ on model → dispatch to solve_* → extract & record quantities. H₂-GC remains hourly (full 3D), consistent with other markets. |
| `ADMM_contracts.jl` | Same ADMM flow as `ADMM.jl` plus PPA + HPA energy/capacity consensuses and λ_ppa/λ_hpa updates. |
| `ADMM_subroutine_contracts.jl` | Per-agent step with PPA/HPA g_bar, λ, ρ updates and extraction. Dispatches to contracts solvers for power/H2/offtaker contract agents. |
| `update_rho.jl` | Minimal residual-balancing ρ update with market-specific rates/caps; includes per-agent capacity ρ update for the `x_cap = z_cap` split. |
| `update_rho_contracts.jl` | Residual-balancing ρ update for standard markets, per-agent capacity split, and ppa/hpa energy/capacity consensuses. |

### 10.6 Save Files

| File | Role |
|---|---|
| `save_results.jl` | Writes: ADMM_Convergence.csv, ADMM_Diagnostics.csv, per-market history CSVs, Agent_Summary.csv, Agent_Quantities_Final.csv, Offtaker_GC_Diagnostics.csv, H2_Producer_Diagnostics.csv. |
| `save_results_contracts.jl` | Writes the same major ADMM outputs as save_results (with PPA/HPA columns) plus: PPAs.csv, HPAs.csv, Green_Agents_Detail.csv. Agent_Summary matches market_exposure structure (no explicit contract columns). |
| `save_social_planner_results.jl` | Called after direct QCP solve with duals available. Writes: Market_Prices.csv (duals of balance constraints), Agent_Summary.csv (quantities + welfare), Capacity_Investments_Planner.csv. |

---

## 11. Output Files

### 11.1 Market Exposure Results

| File | Contents |
|---|---|
| `ADMM_Convergence.csv` | Columns: `iter`, `{market}_primal`, `{market}_dual` for each of the 5 markets, plus `cap_primal` / `cap_dual` (aggregate L2 over agents) and **per-agent** `cap_primal_<m>` / `cap_dual_<m>` columns from the equality-split capacity ADMM (§5.4). One row per ADMM iteration. Used for convergence plots. |
| `ADMM_Diagnostics.csv` | Columns: `iter`, `{market}_rho`, `{market}_price_mean`, `{market}_imb_mean` for each flow market, plus per-agent `cap_rho_<m>` columns (one per cap-owning agent). The legacy single `cap_rho` column has been removed because capacity uses a per-agent ρ controller. |
| `Capacity_Consensus.csv` | Per-iteration, per-agent, per-year snapshot of the capacity equality split. Columns: `iter`, `AgentID`, `jy`, `x_cap`, `z_cap`, `lambda_cap`, `rho_cap`, `primal_local`, `dual_local`. Use this to identify the agent / year that gates capacity convergence; analogous to `{Market}_Market_History.csv` but at the (iter, agent, year) granularity that the per-agent split naturally produces. See §5.4 for the formal model. |
| `{Market}_Market_History.csv` | Per-market CSV with: `iter`, `rho`, `price_mean`, `imb_mean`, `primal_res`, `dual_res`. |
| `Agent_Summary.csv` | Columns: `AgentID`, `Group`. Group membership table. |
| `Agent_Quantities_Final.csv` | Columns: `AgentID`, `Group`, `elec_net_sum`, `H2_net_sum`, `elec_GC_net_sum`, `H2_GC_net_sum`, `EP_net_sum`. Sum of final-iteration 3D quantities. |
| `Offtaker_GC_Diagnostics.csv` | Columns: `AgentID`, `Type`, `EP_total`, `H2_in_total`, `H2_GC_total`, `GC_share`, `GC_mandate`, `GC_slack`. |
| `H2_Producer_Diagnostics.csv` | Columns: `AgentID`, `H2_total`, `H2_GC_total`, `GC_per_H2`. |
| `TimerOutput.yaml` | Profiling: time spent in imbalances, residuals, capacity dual update, price updates, solve, etc. |

### 11.2 Market Exposure with Contracts Results (`market_exposure_contracts_results/`)

`market_exposure_contracts.jl` produces the same major ADMM outputs as market_exposure (ADMM_Convergence, ADMM_Diagnostics, `Capacity_Consensus.csv`, 5× Market_History, Agent_Summary, Market_Prices), with additional PPA/HPA and corresponding cap-consensus columns in convergence and diagnostics. The per-agent capacity columns and `Capacity_Consensus.csv` are mirrored from the base case (§5.4). It adds focal contract outputs:

| File | Contents |
|---|---|
| `PPAs.csv` | Per-VRES summary: `capacity_contracted_MW`, `energy_transferred_MWh`, `ppa_price_EUR_per_MWh`. |
| `HPAs.csv` | Per-GreenProducer summary: `capacity_contracted_MW`, `energy_transferred_MWh`, `hpa_price_EUR_per_MWh`. |
| `Green_Agents_Detail.csv` | Per-agent PPA breakdown (VRES and GreenProducer): total capacity, contracted vs pool energy, and prices. |

### 11.3 Social Planner Results

| File | Contents |
|---|---|
| `Market_Prices.csv` | Columns: `Time`, `Elec_Price`, `H2_Price`, `Elec_GC_Price`, `H2_GC_Price`, `EP_Price`. One row per (jy, jd, jh) timestep. Prices = direct QCP duals of balance constraints (§6.2). Raw duals are divided by representative-day weights `W[jd,jy]` to recover the true per-MWh price. |
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

---

## 13. References

1. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.  
   Core references used here: epigraph reformulation, convex duality, and KKT optimality conditions.

2. R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk*, 2(3), 2000.  
   Foundational CVaR optimization reformulation used for the planner risk term.  
   Open copy: [University of Washington PDF](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)

3. IPOPT Documentation (latest release), nonlinear optimization and multiplier reporting.  
   Main docs: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)  
   (Used to justify direct QCP dual extraction for the social-planner benchmark in this project.)

4. A. Eichfelder, A. Schöbel, L. Schmitz, “A tutorial on properties of the epigraph reformulation,” *Optimization Online*, 2024.  
   Additional modern reference for epigraph reformulation properties and KKT interpretation.  
   PDF: [Optimization Online](https://optimization-online.org/wp-content/uploads/2024/10/Epigraph_reformulation-1.pdf)
