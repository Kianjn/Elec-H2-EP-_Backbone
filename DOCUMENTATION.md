# Multi-Agent Energy Market Simulation — Technical Documentation

## Table of Contents

0. [Notation and Units](#0-notation-and-units)
1. [Overview](#1-overview)
2. [Markets](#2-markets) — incl. [Contract pools](#contract-pools-me-top--me-sop-me-pap-legacy-alias), [Price: λ vs K](#price-clearing-λ-versus-settlement-k), [Settlement equations](#ppa--always-pay-as-produced), [Shared capacity $C$](#how-contract-capacity-is-determined), [Risk allocation](#risk-allocation--who-bears-what)
3. [Agents](#3-agents)
4. [Equilibrium Theory](#4-equilibrium-theory-mcp-structure-competition-and-objectives) — MCP, competition, objectives, risk institutions; [CVaR, γ, β](#410-risk-aversion-cvar-γ-and-β)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [ADMM Algorithm](#6-admm-algorithm) — [why ADMM](#60-why-admm-alternatives-and-literature), [Boyd mapping](#610-mapping-to-boyd-et-al-2011), [ρ controller](#63-adaptive-penalty-ρ), [warm-start](#66-warm-start-from-social-planner)
7. [Social Planner Benchmark](#7-social-planner-benchmark) — incl. [IPOPT tolerance (`ipopt_tol`)](#ipopt-settings-and-convergence-tolerance-ipopt_tol)
8. [Data and Indexing](#8-data-and-indexing) — incl. [scenario mapping](#83-scenario-labels-jy-mapping)
9. [Configuration Reference (data.yaml)](#9-configuration-reference-datayaml) — incl. [NL Calibration and Data Sources](#96-nl-calibration-and-data-sources), [Weather Scenarios](#97-weather-scenarios-representative-days-and-availability-factors), [Gas Prices and the 15-Scenario Grid](#98-gas-prices-and-the-15-scenario-grid)
10. [Project Structure](#10-project-structure)
11. [File Reference](#11-file-reference)
12. [Output Files](#12-output-files)
13. [Code Conventions](#13-code-conventions)
14. [References](#14-references)

> **Math on GitHub:** Inline: `$\cdots$`; display: `$$\cdots$$`. Inside one `$$` block: **no blank lines**; **do not wrap a single equation row across two source lines** (use `\begin{aligned}...\end{aligned}`). Always **brace subscripts** (`_{i}` not `_i`); never split math (`CVaR$_\beta$` breaks — use `$\mathrm{CVaR}_{\beta}$`). Prefer `\left\lbrace` / `\right\rbrace` over `\{` / `\}`; use `_{+}` not `_+`; use `^{\ast}` not `^*` in inline math. **Code identifiers** (`elec_GC`, `sw_aux`) stay in backticks, not in `$...$`.

---

## 0. Notation and Units

This section introduces the symbols used throughout the documentation. All sums in the optimisation problems follow this notation.

### 0.1 Indices and Sets

- $i \in \mathcal{I}$: agents (VRES, conventional generator, consumer, electrolyzer, green offtaker, grey offtaker, importer, GC demand).
- $k \in \mathcal{K}$: market index. In code, $k$ is one of **`elec`**, **`elec_GC`**, **`H2`**, **`H2_GC`**, **`EP`**.
- $h \in \mathcal{H} = \lbrace 1,\dots,n_{\mathrm{timesteps}} \rbrace$: hours within a representative day.
- $d \in \mathcal{D} = \lbrace 1,\dots,n_{\mathrm{reprDays}} \rbrace$: representative days.
- $y \in \mathcal{Y} = \lbrace 1,\dots,n_{\mathrm{years}} \rbrace$: scenario years.

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

$$
r_k(h,d,y) = \sum_{i \in \mathcal{I}_k} g_i^k(h,d,y) - D_k(h,d,y),
$$

where $D_k$ is exogenous demand (for EP only; 0 otherwise).

The **aggregate imbalance norm** used by ADMM is:

$$
\|r_k\|_2 = \left( \sum_{h,d,y} r_k(h,d,y)^2 \right)^{1/2}.
$$

### 0.4 Units

- Electricity: MWh.
- Electricity GC: $\text{MWh}_{\text{GC}}$ (1 certificate per renewable MWh).
- Hydrogen: $\text{MWh}_{\text{H2}}$ (or equivalent energy-based unit).
- Hydrogen GC: $\text{MWh}_{\text{GC,H2}}$.
- End product (EP): $\text{MWh}_{\text{EP}}$ or $\text{t}_{\text{EP}}$ (consistent within the model, governed by `Alpha`).

All monetary values are in **EUR** (e.g. €/MWh, €/t, €/MW-year).

### 0.5 Risk Parameters and CVaR

- $\gamma$ (or $\gamma_i$): **weight on expected loss vs CVaR** in the objective (§4.10). $\gamma=1$ ⇒ risk-neutral (CVaR inactive); $\gamma<1$ ⇒ risk-averse (typically $\gamma=0.5$ in sensitivity runs).
- $\beta$: **CVaR confidence level** (Rockafellar–Uryasev; implemented in code). $\mathrm{CVaR}_{\beta}$ averages loss in the worst $(1-\beta)$ share of scenarios; **higher $\beta$ ⇒ smaller tail ⇒ more risk-averse** (e.g. $\beta=0.8$ ⇒ worst 20%; $\beta=0.2$ ⇒ worst 80%). At fixed $\gamma=0.5$, sweep $\beta$ over $0.2$, $0.4$, $0.6$, and $0.8$ with **increasing** $\beta$ for stronger tail focus (§4.10.4). Hoschle et al. use $\beta$ with the **opposite** direction — see **§4.10.4** comparison paragraph.
- $\alpha_i$, $u_i(y)$, $\mathrm{CVaR}_{i}$: Rockafellar–Uryasev auxiliaries for agent $i$ (VaR proxy, shortfall, conditional tail loss).

Full definitions, Hoschle-style calibration workflow, and equilibrium effects: **§4.10**. Social planner CVaR structure: **§7.4**. Reporting: **§7.6**.

### 0.6 Contracts (ME+C only)

- $q_{h,d,y}$: bilateral **hedge** MW (PPA or HPA); not a physical reroute of pool sales.
- $C$: **shared** contract capacity (MW), one scalar per link, fixed in both subproblems.
- $K$: settlement **strike** (€/MWh), one scalar every hour (W-mean of bundled spot).
- $\lambda^{\mathrm{contract}}$: ADMM dual matching $q^{\mathrm{sell}}=q^{\mathrm{buy}}$; **not** $K$.
- $q^{\mathrm{pay}}$: HPA CfD notional ($q$ under SoP; $C$ under 100% hourly ToP).
- $s=\max(0,C-q)$: SoP seller shortfall (LP: $s\ge C-q$, $s\le C$).

Equations and justifications: **§2 Contract pools**.

---

## 1. Overview

This project implements a **multi-agent equilibrium model** for coupled electricity, hydrogen, green-certificate, and end-product markets, coordinated via **ADMM** (Alternating Direction Method of Multipliers). Each agent has its own JuMP optimization model; market-clearing is achieved by iteratively updating prices and penalty terms so that supply and demand balance in each market.

The project includes **seven entry points** in five economic categories. Each has a **code name** (script / folder) and an **economic label** aligned with d’Aertrycke et al. (2018), *Risk trading in capacity equilibrium models* (see §4.8 and §14):

| Script | Code name | Economic case (competitive spot, capacity investment) |
|---|---|---|
| **`social_planner.jl`** | Social planner (SP) | **Complete risk trading** — centralised risk-averse welfare maximisation with a single social CVaR on aggregate welfare. |
| **`market_exposure.jl`** | Market exposure (ME) | **Incomplete risk trading** — decentralised equilibrium via ADMM; risk-averse agents hedge **private** tail losses with per-agent CVaR, without an explicit risk market. |
| **`me_pap.jl`** / **`me_top.jl`** / **`me_sop.jl`** | Market exposure with bilateral contracts | Same **incomplete risk trading** as ME, plus **PPA** (always PaP CfD) and **HPA** (ToP or SoP; `me_pap` aliases SoP). See §2 *Contract pools*. |
| **`green_h2_social_planner.jl`** | Green H₂ partial planner (GH2-SP) | **Partial complete risk trading** — electrolyzer + green offtaker merged; one coalition CVaR; ADMM + external spot markets unchanged. |
| **`green_social_planner.jl`** | Green partial planner (G-SP) | **Partial complete risk trading** — solar + wind + electrolyzer + green offtaker merged; one coalition CVaR; ADMM + external spot markets unchanged. |

**When $\gamma = 1$ (risk-neutral):** SP is the stochastic welfare-maximisation benchmark; ME (and ME+C) should converge to the same quantities and spot prices as SP in the limit of exact ADMM convergence (first welfare theorem).

**When $0 < \gamma < 1$ (risk-averse):** SP is the **centralised risk-pooling / complete-risk-trading** benchmark; ME and ME+C are **private-hedging / incomplete-risk-trading** equilibria. Quantities and prices need not (and generally should not) match SP — that divergence is expected theory, not a formulation error. SP balance duals remain valid **risk-adjusted social shadow prices** for each commodity (§4.8, §7.2); ADMM $\lambda$ are valid equilibrium prices for the decentralised case.

Entry-point details:

- **`market_exposure.jl`** — Distributed ADMM; five markets: electricity, elec GC, H₂, H₂ GC, end product.
- **`me_top.jl`**, **`me_sop.jl`** (legacy **`me_pap.jl`** = SoP alias) — Same risk architecture as ME, with bilateral **PPA** (always PaP CfD) and **HPA** (ToP or SoP). See §2 *Contract pools*.
- **`social_planner.jl`** — Single centralised model; commodity prices are dual variables of market-clearing constraints, scaled to €/MWh per §7.2 (`W` and effective scenario weight $\mu_y$).
- **`green_h2_social_planner.jl`** — ADMM with **Prod_H2_Green + Offtaker_Green** as one agent (`GreenH2_Coalition`); internal H₂/H₂-GC flows are not traded on spot markets; **one coalition CVaR** (complete risk sharing inside the coalition). See §4.11.
- **`green_social_planner.jl`** — ADMM with **Gen_VRES_Solar + Gen_VRES_Wind + Prod_H2_Green + Offtaker_Green** merged (`Green_Coalition`); same institution as GH2-SP extended upstream to VRES. See §4.11.

The base ADMM and social planner share the **same** problem definition from `Source/`. The contracts case and the two partial planner cases use **separate entry points and extension modules**; `market_exposure.jl` and `social_planner.jl` are unchanged.

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

### Contract pools (me_top / me_sop; me_pap legacy alias)

Entry points **`me_top.jl`** and **`me_sop.jl`** (and legacy **`me_pap.jl`**) keep the five spot markets of `market_exposure.jl` and add two **bilateral hedge pools**. They share `ADMM_contracts.jl`, `contract_capacity.jl`, `contract_strike.jl`, and `contract_settlement.jl`. The only economic switch is HPA **volume mode**.

| Script | Label | HPA volume in code | Results folder |
|--------|-------|--------------------|----------------|
| `me_top.jl` | **ME ToP** | Take-or-pay | `me_top_results/` |
| `me_sop.jl` | **ME SoP** | Send-or-pay | `me_sop_results/` |
| `me_pap.jl` | **ME PaP (legacy alias)** | Mapped to SoP (`HPA_VOLUME_MODE = "sop"`) | `me_pap_results/` |

**PPA is always pay-as-produced** and identical in all three scripts. There is **no** distinct HPA pay-as-produced entry point: a pure HPA-PaP would be the energy CfD on $q$ with no ToP notional and no SoP penalty; `me_pap.jl` does **not** implement that (it aliases SoP so old run scripts keep working). Compare **ToP vs SoP** with `me_top.jl` vs `me_sop.jl`. The PaP algebra is still the nested energy leg of both HPA modes, and is the **entire** PPA.

| Market | Key | What is traded | Unit | Seller | Buyer |
|---|---|---|---|---|---|
| **PPA** | `ppa` | Hedge quantity (financial; pool power unchanged) | MWh | VRES `g_ppa` | GreenProducer `g_ppa_from` |
| **HPA** | `hpa` | Hedge quantity (financial; pool H₂ unchanged) | MWh_H2 | GreenProducer `h2_hpa` | GreenOfftaker `h2_hpa_from` |

Three objects, never interchangeable:

| Object | Role | In cash / CVaR? |
|--------|------|-----------------|
| $q_{h,d,y}$ | Hedge MW at each slot, matched by ADMM | Enters settlement |
| $C$ | Scalar MW ceiling, **one shared value** per link | Enters ToP/SoP volume terms; PPA/PaP only as $q \le C$ |
| $K$ | Scalar strike €/MWh, same every hour | Enters the CfD (and SoP penalty) |
| $\lambda^{\mathrm{contract}}$ | ADMM dual on $q^{\mathrm{sell}}-q^{\mathrm{buy}}$ | **Algorithmic only** — not the strike |

Spot sales remain at pool $\lambda$. The hedge does **not** carve MW out of $g^{\mathrm{EOM}}$ or $h^{\mathrm{out}}$. ADMM details: §6.7.

#### Overview: real contracts vs this implementation

A real PPA/HPA is one package signed before operations: strike $K$, capacity $C$, and a volume rule (who pays when $q \neq C$). At **ADMM convergence** the model has the same three terms: one shared $C$ per link, one scalar $K$ every hour, and ToP or SoP on the HPA (PPA always PaP).

There is **no** explicit individual-rationality constraint (“sign only if both beat autarky”). Uptake is **emergent**: hedge cash flows that improve private CVaR ($\gamma < 1$) can keep $C,q>0$; at $\gamma=1$ they typically collapse. That is the incomplete-risk-trading analogue of “no trade when agents are risk-neutral.”

Hard physical rerouting (must-deliver slices taken out of the pool) is **not** modelled. That would change operational feasibility even at $\gamma=1$ and confound the hedge-motive test.

#### Why financial CfDs (not $K\cdot q$ on top of spot)

Two representations appear in the literature:

1. **Financial/hedging contracts** (typical in ADMM market-clearing). Reciprocity is $q^{\mathrm{sell}}=q^{\mathrm{buy}}$; cash is a settlement on that $q$. Risk-neutral agents then have little reason to keep $C>0$.
2. **Physical-delivery contracts** (planning / games). The contract constrains dispatch, so $C>0$ can appear without risk aversion.

This thesis uses (1), so $\gamma=1$ is a **sanity benchmark** (expect $C\approx 0$) and $\gamma<1$ isolates **tail-risk transfer**. References: Tushar et al., IEEE TPWRS 2020; Wang et al., PESGM 2019; Huang et al., PESGM 2021; Yadeta et al., arXiv 2025; Neumann et al., Applied Energy 2025 (ToP in hydrogen *planning*, not ADMM).

**Why the floating leg is mandatory.** Pool sales already pay $\lambda^{\mathrm{spot}}\cdot q^{\mathrm{pool}}$. A hedge that *also* paid $K\cdot q$ with no float would be a free add-on to full spot revenue. ADMM then drives $K\to 0$ (or copies $\lambda^{\mathrm{contract}}\to 0$ into $K$), and risk-averse agents have nothing to hedge. The CfD $(K-\lambda^{\mathrm{spot}}_{\mathrm{bundle}})\cdot q$ offsets pool-price risk on the hedged slice. If $q$ equals physical pool volume on that commodity, net cash on the slice **locks at $K$**:

$$
\lambda^{\mathrm{spot}}\,q + (K-\lambda^{\mathrm{spot}})\,q = Kq.
$$

Copying $\lambda^{\mathrm{contract}}$ into $K$ was tried and rejected: that dual only matches quantities and is not an economic strike (it can sit near 0, or go negative, once $q$ matches).

Temporary $q^{\mathrm{sell}}\neq q^{\mathrm{buy}}$ during ADMM is a trial point. At convergence the energy residual must clear. Low $q$ at a positive $C$ is “abuse” only if it contradicts the **chosen** volume rule (PaP pays on $q$; ToP is *meant* to pay on $C$).

#### Price: clearing λ versus settlement K

| Object | Role | In private cash / CVaR? | When it moves |
|--------|------|-------------------------|---------------|
| $\lambda^{\mathrm{contract}}$ | ADMM dual on hedge-quantity imbalance (3D, like spot $\lambda$) | **No** — linear ADMM term + $(\rho/2)(q-\bar q)^2$ only | Dual ascent every iteration |
| $K$ | Settlement strike (€/MWh), broadcast to all hours | **Yes** — CfD (and SoP penalty) | Provisional each iter; snapshotted at convergence |
| $C$ | Shared MW ceiling per link | ToP/SoP volume terms; PPA only via $q\le C$ | Bargaining update after subproblems; snapshotted at convergence |

**Default (`price_benchmark: negotiated`).** Let $W_{d,y}$ be representative-day weights and $n_{\mathrm{ts}}$ hours per day. The strike is the W-mean of bundled physical spot:

$$
K = \frac{\sum_{y,d,h} W_{d,y}\,\lambda^{\mathrm{bundle}}_{h,d,y}}{\sum_{y,d} W_{d,y}\,n_{\mathrm{ts}}}.
$$

PPA bundle: $\lambda^{\mathrm{elec}}+\lambda^{\mathrm{elecGC}}$. HPA bundle: $\lambda^{\mathrm{H2}}+\lambda^{\mathrm{H2GC}}$. That is the **fair** CfD index — the expected bundled pool price the hedge is written against. Yaml alternatives (`electricity`, `ammonia`, `NG`) replace the index with an exogenous field $B$; HPA `price_structure: cfd` floats vs $B$ instead of vs pool. **Do not** stack yaml `cfd` on top of the pool CfD. PPAs always use the bundled-spot strike.

#### Risk neutrality versus risk aversion ($\gamma$)

| Setting | Typical outcome | Interpretation |
|---------|-----------------|----------------|
| $\gamma=1$ | $C\approx 0$, tiny $q$ | No tail-risk motive ⇒ no hedge premium. |
| $\gamma<1$ | $C>0$ possible | Private CVaR can value locking $K$ on a slice; ToP vs SoP changes **whose** tail moves. |

Compare `PPAs.csv` / `HPAs.csv` across $\gamma$ and `me_top` vs `me_sop`. Compare to `social_planner.jl` for **institutional** risk gaps (§4.8), not only price levels.

#### Two time axes

| Axis | Behaviour |
|---|---|
| **ADMM iterations** | $\lambda^{\mathrm{contract}}$, provisional $K$, and $C$ update every iteration. |
| **Hours $(h,d,y)$** | At convergence, $K$ and $C$ are **scalars** (same €/MWh and MW every hour). Flows $q$ vary; $K$ and $C$ do not. |

Each iteration: $\lambda^{\mathrm{contract}}$ dual-ascent on energy imbalance; $K$ refreshed as the W-mean above; $C$ **fixed** in both JuMP models (`apply_shared_contract_caps!`), then updated once from both cap shadows (`update_shared_contract_capacity!`). At termination, `finalize_contract_terms!` snapshots $K$ and $C$ into `ADMM["ContractStrikes"]`. There is **no mid-run freeze**.

#### PPA — always pay-as-produced

**Seller** VRES $i$: physical pool $g^{\mathrm{EOM}}_{i,h,d,y}$; hedge $g^{\mathrm{PPA}}_{i,h,d,y}$. **Buyer** GreenProducer: hedge $g^{\mathrm{PPA}}_{i\leftarrow,h,d,y}$. Constraints (same $C_i$ on both sides):

$$
\begin{aligned}
0 &\le g^{\mathrm{PPA}}_{i,h,d,y} \le C_i, \\
g^{\mathrm{PPA}}_{i,h,d,y} &\le \mathrm{AF}_{i,h,d,y}\, x_i, \\
0 &\le g^{\mathrm{PPA}}_{i\leftarrow,h,d,y} \le C_i, \\
g^{\mathrm{PPA}}_{i\leftarrow,h,d,y} &\le e^{\mathrm{pool}}_{h,d,y}.
\end{aligned}
$$

Availability $g\le \mathrm{AF}\,x$ is why solar hedges go to zero at night: PaP pays only on $q$, so unused $C$ is not billed. The electrolyzer bound $g^{\mathrm{PPA}}_{i\leftarrow}\le e^{\mathrm{pool}}$ keeps the hedge from exceeding physical intake; it does **not** reroute pool MW.

Let $\lambda^{\mathrm{el}}_{\mathrm{bundle}} = \lambda^{\mathrm{elec}}+\lambda^{\mathrm{elecGC}}$. Settlement (seller revenue = buyer cost):

$$
\pi^{\mathrm{PPA}}_{i,h,d,y} = \bigl(K^{\mathrm{PPA}}_i - \lambda^{\mathrm{el}}_{\mathrm{bundle},h,d,y}\bigr)\, g^{\mathrm{PPA}}_{i,h,d,y}.
$$

$K^{\mathrm{PPA}}_i$ is the W-mean of $\lambda^{\mathrm{el}}_{\mathrm{bundle}}$ (one scalar). Bundling electricity + GC is required because the physical green electron is a **package** in this model (1:1 GO). Hedging only $\lambda^{\mathrm{elec}}$ would leave GC price risk unhedged.

ADMM matches $g^{\mathrm{PPA}}_i = g^{\mathrm{PPA}}_{i\leftarrow}$ via $\lambda^{\mathrm{PPA}}_i$ (linear term **outside** CVaR) and $(\rho/2)(q-\bar q)^2$. That dual is **not** $K$.

#### HPA — volume modes (equations)

Hedge $q_{h,d,y}$ (MW_H2), shared ceiling $C$, strike $K$. Default floating index $\lambda^{\mathrm{H2}}_{\mathrm{bundle}} = \lambda^{\mathrm{H2}}+\lambda^{\mathrm{H2GC}}$. Yaml `cfd` replaces $\lambda^{\mathrm{H2}}_{\mathrm{bundle}}$ with benchmark $B_{h,d,y}$. Physical $h^{\mathrm{out}}$ stays in the H₂ / H₂-GC pools.

**Energy CfD (nested in every HPA mode),** with notional $q^{\mathrm{pay}}$ defined below:

$$
\pi^{\mathrm{CfD}}_{h,d,y} = \bigl(K - \lambda^{\mathrm{H2}}_{\mathrm{bundle},h,d,y}\bigr)\, q^{\mathrm{pay}}_{h,d,y}.
$$

**Pay-as-produced (taxonomy; PPA uses this; not a distinct HPA entry point).** $q^{\mathrm{pay}}=q$. Cash moves only with delivery. $C$ is a rate limit, not a bill.

**Take-or-pay (`me_top.jl`) — 100% hourly ToP.** Auxiliary $s_{h,d,y}$ is forced to $C-q$ (lower *and* upper bound in JuMP, given $q\le C$):

$$
s_{h,d,y} = C - q_{h,d,y}, \qquad q^{\mathrm{pay}}_{h,d,y} = q_{h,d,y} + s_{h,d,y} = C.
$$

Buyer pays and seller receives $\pi^{\mathrm{CfD}}$ on **$C$ every hour**, even if $q\ll C$. This is the polar opposite of PaP: utilization risk sits on the offtaker. Commercial contracts often use a take-or-pay *rate* $\theta\in(0,1)$ on an annual quantity; $q^{\mathrm{pay}}=\max(q,\theta C)$ is a one-line extension. We implement $\theta=1$ hourly so ToP vs SoP is a clean qualitative contrast, not a calibrated $\theta$ sweep.

**Send-or-pay (`me_sop.jl`, and `me_pap.jl`).** Buyer: $q^{\mathrm{pay}}=q$ (PaP energy CfD). Seller pays an extra strike-scaled shortfall:

$$
s_{h,d,y} = \max\bigl(0,\, C - q_{h,d,y}\bigr), \qquad
\pi^{\mathrm{seller}} = \pi^{\mathrm{CfD}}(q) - K\, s_{h,d,y}.
$$

LP form (min-cost, $K$ typically $>0$ so the solver takes the smallest feasible $s$):

$$
s \ge C-q, \qquad 0 \le s \le C.
$$

The upper bound $s\le C$ is required: if $K$ goes slightly negative, an unbounded $s$ makes Gurobi return **0 solutions**. An auxiliary “obligation” $o\le\min(C,h^{\mathrm{out}})$ is **not** used: in a min-$s$ problem the solver sets $o=0$ (toothless SoP).

**Why ToP and SoP are not symmetric in cash.** ToP scales the **CfD notional** to $C$ (price hedge on a reserved MW). SoP leaves the CfD on delivered $q$ and adds a **penalty** $Ks$ on the seller. They are not $K\cdot C$ vs $-K\cdot C$ with opposite signs on the same notional.

#### Agent objective (contracts) — cash vs ADMM

For a risk-aware contract agent the **private loss** that enters CVaR contains the CfD (and SoP $Ks$), **not** $\lambda^{\mathrm{contract}}\cdot q$. The ADMM matching terms sit **outside** CVaR, same as spot $\rho$ penalties:

$$
\begin{aligned}
\min \quad
&\gamma\Bigl(F^{\mathrm{cap}}x + \sum_y P_y \ell_y\Bigr) + (1-\gamma)\,\mathrm{CVaR}(\ell) \\
&+ \sum \tfrac{\rho}{2} W (g-\bar g)^2
- \sum W\,\lambda^{\mathrm{PPA}} g^{\mathrm{PPA}}
+ \cdots
\end{aligned}
$$

(seller sign on $\lambda^{\mathrm{PPA}}g^{\mathrm{PPA}}$; buyer opposite). $\ell_y$ includes $\pm\pi^{\mathrm{CfD}}$ and, for SoP sellers, $+Ks$. At consensus the $\rho$ terms vanish; $\lambda^{\mathrm{contract}}$ is an algorithm dual, not a tariff.

Shared $C$ is **JuMP-fixed** in the subproblem. Agents do not pick a private $C_s$/$C_b$. The cap dual of $q\le C$ (and the reduced cost of the fixed cap) is read **after** `optimize!` to vote on expanding $C$.

#### Risk allocation — who bears what?

All contract cases share the same **price hedge** at convergence: strike $K$ is a scalar applied every hour, and the CfD $(K-\lambda^{\mathrm{spot}}_{\mathrm{bundle}})\cdot q$ offsets pool-price swings on the hedged slice. **Contract capacity** $C$ is a negotiated MW ceiling on the bilateral pipe in every entry point; there is **no separate €/MW capacity tariff** cleared by ADMM. What differs by volume mode is **who bears utilization, production, and volume risk** when delivery $q$ is below $C$.

**Shared risks (all `me_pap` / `me_top` / `me_sop`):**

| Risk | Who is mainly exposed | Mechanism |
|------|------------------------|-----------|
| **Spot price (pool markets)** | Agents still on the pool | Any flow sold/bought on `elec`, `H2`, `EP`, GC pools remains at clearing $\lambda$; contracts only cover the bilateral slice. |
| **Strike vs spot** | Both contract parties | At convergence, settlement uses scalar $K$ not $\lambda^{\mathrm{contract}}$; if spot diverges from $K$, the party that would have preferred the spot price bears the opportunity cost. |
| **Plant investment** | VRES, electrolyzer, green offtaker | Endogenous `cap_VRES`, `cap_H2_y`, `cap_EP_y` and fixed CAPEX are separate from HPA/PPA **energy** settlement. |
| **Private tail risk ($\gamma < 1$)** | Each agent separately | ME keeps **incomplete risk trading** (private CVaR); contract cash flows enter each agent’s loss, so volume modes change **which agent’s** CVaR moves. CfD (`HPAs.price_structure: cfd`) further splits fixed vs benchmark-index legs for HPA when $\gamma < 1$. |

**PPA (always PaP — all three entry points):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **VRES (seller)** | Receives $(K^{\mathrm{PPA}} - \lambda^{\mathrm{elec}} - \lambda^{\mathrm{elecGC}}) \cdot g^{\mathrm{PPA}}$ | **Production risk:** hedge volume scales with renewable output; zero output ⇒ zero PPA cash. $C^{\mathrm{PPA}}$ caps how much can be hedged, but **unused headroom is not paid**. |
| **Green producer (buyer)** | Pays $(K^{\mathrm{PPA}} - \lambda^{\mathrm{elec}} - \lambda^{\mathrm{elecGC}}) \cdot g^{\mathrm{PPA}}$ | **Input cost risk:** hedge only covers MWh contracted; if VRES delivers little, PPA cash is small but the plant still buys pool electricity. No minimum offtake on the PPA leg. |

PPA shifts **renewable volume risk** to the VRES and **procurement / conversion risk** to the electrolyzer, while **locking the green-electricity price** at $K^{\mathrm{PPA}}$ on the hedged slice (full lock when $q$ equals physical pool volume).

**HPA — pay-as-produced (taxonomy; not a current HPA entry point):**

$C$ is a **rate limit**, not a bill. Cash follows $q$ only. PPA is this mode. `me_pap.jl` does **not** run HPA-PaP (it aliases SoP).

**HPA — take-or-pay (`me_top.jl`):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **Green offtaker (buyer)** | Pays $(K - \lambda^{\mathrm{H2}}_{\mathrm{bundle}})\cdot C$ every hour ($q^{\mathrm{pay}}=C$) | **Minimum offtake:** CfD notional is contracted MW even when $q\ll C$. |
| **Green producer (seller)** | Receives the same | **Revenue floor** on the hedge notional; $q\le C$ still caps the contract leg. |

ToP shifts utilization risk to the **buyer** (reserved-capacity economics). Code uses 100% hourly ToP ($q^{\mathrm{pay}}=C$), not $\max(q,C)$ with slack, because the auxiliary is equality-constrained to $C-q$.

**HPA — send-or-pay (`me_sop.jl` and `me_pap.jl`):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **Green offtaker (buyer)** | Pays $(K - \lambda^{\mathrm{H2}}_{\mathrm{bundle}}) \cdot q$ | Same delivery-linked CfD as PaP; **no** minimum offtake premium. |
| **Green producer (seller)** | Receives $(K - \lambda^{\mathrm{H2}}_{\mathrm{bundle}}) \cdot q - K \cdot s$, with shortfall $s = \max(0, C - q)$ | **Delivery risk:** penalised $K \cdot s$ when contract quantity is below contracted capacity $C$. |

SoP **shifts delivery risk to the producer**: the buyer is not forced to pay for undelivered molecules, but the seller pays a **strike-scaled penalty** when it could have delivered more under the contract.

**Summary — volume risk on the HPA leg:**

```text
                    Volume / utilization risk borne mainly by
                 Buyer (offtaker)          Seller (producer)
              ┌─────────────────────┬─────────────────────┐
   HPA PaP    │  sourcing if q low  │  revenue if q low   │  (taxonomy; PPA only)
   me_top     │  CfD notional = C   │  floor notional C   │
   me_sop     │  CfD on q only      │  + K·s if q < C     │  (me_pap aliases this)
              └─────────────────────┴─────────────────────┘
```

**Relation to $\gamma < 1$:** At $\gamma = 1$ the table is an average-cost story and $C\to 0$ is expected. At $\gamma < 1$, private CVaR can support $C>0$. Compare `me_top` vs `me_sop` (not `me_pap` as a third HPA mode). The planner remains **complete** risk pooling (§4.8).

#### HPA price structure (`data.yaml` → `HPAs`)

**Fixed price (default):** $\pi^{\mathrm{CfD}}=(K-\lambda^{\mathrm{H2}}_{\mathrm{bundle}})q^{\mathrm{pay}}$ with $q^{\mathrm{pay}}=q$ (SoP buyer) or $q^{\mathrm{pay}}=C$ (ToP). $K$ is the W-mean of bundled H₂+H₂-GC spot when `price_benchmark: negotiated`.

**Yaml `cfd`:** float vs benchmark field $B$ instead of vs pool: $(K - B_{h,d,y}) \cdot q^{\mathrm{pay}}$. Set `HPAs.price_structure: cfd`. Do **not** stack this on top of the pool CfD.

| `price_benchmark` | Field $B$ at convergence | Source in model |
|-------------------|----------------------|-----------------|
| `negotiated` | bundled $\lambda^{\mathrm{H2}}+\lambda^{\mathrm{H2GC}}$ | Physical H₂ + H₂-GC spot (default CfD index) |
| `electricity` | $\lambda^{\mathrm{elec}}$ | Electricity spot |
| `ammonia` | $\lambda^{\mathrm{EP}}$ | End-product / ammonia pool |
| `NG` | Grey-chain proxy | Grey ammonia MC ÷ `Alpha` → €/MWh_H2, derived from the `Fuel` block at the **mean** gas multiplier (§9.8) |

At convergence: $K^{\mathrm{HPA}} = \text{W-weighted mean of the strike index}$ over the horizon (scalar, uniform over hours).

#### How contract capacity is determined

$C$ is a **single shared scalar** per link (`ADMM["SharedCap"]`), JuMP-**fixed** on both parties (`apply_shared_contract_caps!`), then updated **outside** the subproblems. `me_top`, `me_sop`, and `me_pap` share this update; only HPA settlement differs.

##### Why not two-party ADMM on $C$

Letting each side pick $C_s$ and $C_b$ and matching them like energy **failed**. Under ToP the buyer’s cash is increasing in $C$ (pays on $C$), so the buyer’s local problem drives $C_b\to 0$ while the seller drives $C_s$ up. The “consensus” $C$ then oscillates or collapses. Energy $q$ is a flow with opposite signs in a market-clearing constraint; $C$ is a **common parameter** of both settlements. A parameter that one party wants large and the other small is not a Boyd equality split.

##### Why not infer $C$ from $q$

Observed $q$ is **censored** by $q\le C$. If the pipe is binding, $q^{\mathrm{peak}}\approx C$ whether the desired pipe is $C$ or $10C$. Energy imbalance is also uninformative: when both are rationed at the same $C$, $q^{\mathrm{sell}}-q^{\mathrm{buy}}=0$ even though both want more MW. The correct signal is the **shadow** of $q\le C$ (W-weighted positive dual $\mu$) and/or the reduced cost of the fixed cap (min problem: $\mathrm{RC}<0$ ⇒ raising $C$ helps).

Party $i$ **wants more** if $\max(\mu_i, -\mathrm{RC}_i) > \texttt{dual\_tol}$. Expand only if **both** available duals vote yes (unanimous consent). If duals are missing, a conservative fallback requires both sides to be using the full slice ($q_s\approx q_d\approx C$).

##### Expand, hold, idle snap

$q$ cannot jump in the same iteration $C$ jumps (agents solve with $C$ fixed). So:

- **Expand** only after `up_confirm_iters` consecutive unanimous votes (default 3) — ignore one noisy dual.
- **Hold** when they stop asking for more. Do **not** set $C^{\mathrm{target}}=0$ on a “no” vote: that bang-bangs (expand 8.75 MW, then shrink to 0, then bind again) and never settles.
- **Idle snap to 0** only if mean mutual utilisation $\mathrm{mean}(\min(q_s,q_d))/C \le 2\%$ for 3 iterations. That is the risk-neutral unused-pipe outcome, not a default shrink.

Do not trim unused headroom toward $q^{\mathrm{peak}}$ every non-expand iteration: after an expand, $q$ needs several ADMM steps to fill the new pipe; trimming immediately undoes the expand.

##### Step size (why not 8.75 MW)

A damped additive crawl

$$
C \leftarrow (1-\tau)C + \tau(C+\eta) = C + \tau\eta
$$

with $\tau=0.35$, $\eta=25$ MW produced **8.75 MW** steps. That is not an economic quantity. The physical bottleneck is the **smaller plant**, not the 55 GW VRES fleet:

$$
C^{\mathrm{PPA}}_{\mathrm{phys}} = \min\bigl(x^{\mathrm{VRES}},\, x^{\mathrm{H2}}/\eta\bigr), \qquad
C^{\mathrm{HPA}}_{\mathrm{phys}} = \min\bigl(x^{\mathrm{H2}},\, x^{\mathrm{EP}}/\alpha\bigr).
$$

Those are $\sim 1$ GW in the NL calibration. An 8.75 MW crawl cannot reach that scale before shadows drop, so equilibrium $C$ was a token of tens of MW **because of the step**, not because of preferences.

Sizing $\Delta C$ on dual *magnitude* ($\Delta C \propto \min(\mu_s,\mu_b)$) would mix €/MW-year into MW and needs an arbitrary scale. Duals already **vote**. Geometry of the known feasible set sets the **step**:

$$
\Delta C = \min\Bigl( C_{\mathrm{phys}}-C,\; \max(\eta_{\min},\, \phi\,(C_{\mathrm{phys}}-C)),\; \eta_{\max} \Bigr),
$$

with shipped $\eta_{\min}=25$, $\phi=0.2$, $\eta_{\max}=500$ (`expand_step`, `expand_frac`, `expand_max`). Applied **undamped** so yaml knobs mean what they say. Interpretation: each confirmed expand takes 20% of remaining headroom (at least 25 MW, at most 500 MW) toward the physical cap — a damped Newton step on a scalar with a known upper bound. $\eta_{\max}$ stops one iterate from swallowing the whole electrolyzer before counterparties re-solve.

##### Residuals and stopping

| Residual | Definition | Why |
|----------|------------|-----|
| Energy primal/dual | Boyd on $q^{\mathrm{sell}}-q^{\mathrm{buy}}$ | Same as spot markets |
| Cap primal | $|C-q^{\mathrm{peak}}|$ | Diagnostic of unused headroom; **not** a two-party $C_s-C_b$ |
| Cap dual | $|C^k-C^{k-1}|$ | $C$ must stop moving |

`shared_contract_capacity_settled` is **fail-closed**: $|C^k-C^{k-1}|\le \texttt{settle\_tol}$ for `settle_iters` consecutive iterations, not currently expanding, no pending expand vote. Holding a used $C$ is settled. Flow residuals alone must not stop the loop while $C$ is still crawling (that was a fake 7-iter “converged”).

`finalize_contract_terms!` writes $C$ to `PPAs.csv` / `HPAs.csv`.

#### One ADMM iteration (contracts case)

```text
For iteration k = 1, 2, …
  1. Refresh ḡ, spot λ, provisional K; fix shared C; solve all subproblems
  2. Record hedge q (not private C_s / C_b)
  3. Spot + contract energy imbalances; Boyd residuals
  4. update_shared_contract_capacity! (shadow vote → expand / hold / idle snap)
  5. Dual-ascent λ_contract; adapt ρ (spot then contracts)
  6. Stop only if spot + q + physical cap + C-settled all pass
After loop: finalize_contract_terms! → scalar K, C
```

#### Contract implementation files

| File | Role |
|------|------|
| `Source/contract_strike.jl` | $K$ as W-mean of bundled spot (not $\lambda^{\mathrm{contract}}$); `finalize_contract_terms!` |
| `Source/contract_capacity.jl` | Shared $C$: init, JuMP fix, bargaining update, settle test |
| `Source/contract_settlement.jl` | PPA CfD; HPA ToP / SoP LP forms |
| `Source/ADMM_contracts.jl` | Loop; energy residuals; $C$ update; stop test |
| `Source/ADMM_subroutine_contracts.jl` | Per-agent $\bar q$, $\lambda^{\mathrm{contract}}$, $K$; `apply_shared_contract_caps!` |

At convergence `ADMM["ContractStrikes"]` holds scalar $K$ and $C$ for reporting.

#### Recommended workflow

1. Run `social_planner.jl` (equilibrium benchmark).
2. Run `market_exposure.jl` (uncontracted ME).
3. Run `me_top.jl` and/or `me_sop.jl` with the same `data.yaml` (SP warm-start). `me_pap.jl` is a SoP alias.

**Suggested comparisons:**

| Question | What to run / read |
|----------|-------------------|
| Pool vs contracts at $\gamma=1$ | `market_exposure.jl` vs `me_sop.jl`; expect $C \approx 0$ |
| ToP vs SoP volume risk | Same $\gamma$; `me_top.jl` vs `me_sop.jl`; `HPAs.csv` + §2 *Risk allocation* |
| Hedging with contracts | $\gamma=0.5$, sweep `beta`; check $C>0$ and contract energy |
| Institution vs SP | `social_planner.jl` vs ME contract case; `Risk_Metrics.csv` |

### Market coupling

The markets are coupled through the **electrolyzer**, which sits at the nexus:

- It **buys** electricity (elec market) and electricity GCs (`elec_GC` market).
- It **sells** hydrogen (H2 market) and hydrogen GCs (`H2_GC` market).
- The conversion constraint `h2_out = η × e_in` links the electricity and hydrogen markets.
- The annual green-backing constraint links the elec_GC and `H2_GC` markets.

The **end-product market** is coupled to H2 and `H2_GC` through the offtakers, who convert hydrogen into the end product and must comply with the GC mandate.

In the **contracts case**:
- PPA couples VRES and GreenProducer: VRES sells `g_EOM` to the pool and holds hedge `g_ppa`; GreenProducer buys pool power `e_in_pool` and hedge `g_ppa_from`.
- HPA couples GreenProducer and GreenOfftaker: GreenProducer sells physical `h2_out` to the pool and holds hedge `h2_hpa`.

---

## 3. Agents

### 3.1 Power-Sector Agents

| Agent | Type | Description |
|---|---|---|
| `Gen_VRES_01` | `VRES` | Variable renewable (e.g. solar). Zero marginal cost. Produces both electricity and elec GCs (1:1). Constrained by hourly availability factor × **endogenous capacity**. Makes **one** installed-capacity and investment decision (`cap_VRES`, `inv_VRES`), incurring fixed annualised CAPEX `FixedCost_per_MW × cap_VRES` (same capacity in all weather scenarios). In contract entry points: physical pool dispatch `g_EOM ≤ AF × cap`; hedge `g_ppa ≤ ppa_cap` with shared scalar $C^{\mathrm{PPA}}$; CfD settlement $(K^{\mathrm{PPA}} - \lambda^{\mathrm{elec}} - \lambda^{\mathrm{elecGC}}) \times g^{\mathrm{PPA}}$. |
| `Gen_Conv_CCGT` | `Conventional` | CCGT block (14,040 MW). Constant availability (AF = 1). Flat SRMC from `Fuel` block; gas-linked across scenarios. No GC production. |
| `Gen_Conv_Coal` | `Conventional` | Hard-coal block (1,800 MW). Flat SRMC; unaffected by gas-price scenarios. |
| `Gen_Conv_Biomass` | `Conventional` | Biomass block (2,160 MW). Flat SRMC; ETS zero-rated fuel. Merit order among the three emerges from market clearing rather than an enforced stage stack. |
| `Cons_Elec_01` | `Consumer` | Elastic electricity demand. Quadratic utility `U(d) = A_E·d − ½B_E·d²` gives inverse demand `p(d) = A_E − B_E·d`. Bounded by `PeakLoad × load_profile`. |

### 3.2 Hydrogen-Sector Agent

| Agent | Type | Description |
|---|---|---|
| `Prod_H2_Green` | `GreenProducer` | PEM electrolyzer. **IEA nameplate is electrical input** (`Capacity_Electrolyzer`, MW_e). Implied H₂ output = MW_e / `SpecificConsumption`. Endogenous JuMP variable `cap_H2_y` is H₂ output; annualised CAPEX is `FixedCost_per_MW_Electrolyzer × SpecificConsumption × cap_H2_y` so the IEA €/MW_e figure is applied to electrical MW. Buys elec + elec GCs; sells H₂ + H₂ GCs. Annual green-backing: GCs purchased ≥ `(1/η) ×` GCs issued. In contract entry points: receives `g_ppa_from` (PPA) and sells `h2_hpa` under HPA. |

### 3.3 Offtaker Agents

| Agent | Type | Description |
|---|---|---|
| `Offtaker_Green` | `GreenOfftaker` | Haber–Bosch plant. **Nameplate is ammonia product output** (`Capacity_EP_Out`, MW_EP / t NH₃-yr), not H₂ feed. H₂ intake = `ep / Alpha`. Tight link `ep = α × h2_in`. Endogenous `cap_EP_y` with CAPEX `FixedCost_per_MW_EP_Out × cap_EP_y` (synthesis + ASU, excluding the electrolyzer). GC mandate 42% of H₂ intake. |
| `Offtaker_Grey` | `GreyOfftaker` | Produces EP from conventional (grey, SMR) feedstock at a **scenario-dependent** marginal cost `MC[jy]` derived from the gas and CO₂ prices (§9.8). Does **not** buy physical H₂; its H₂ feedstock is inferred as `ep / gamma_NH3` (`gamma_NH3` = MWh_EP per MWh_H₂, default 0.75). Must buy H₂ GCs for ≥ `gamma_GC × (1/gamma_NH3) × ep` — i.e. ≥ 42% of the **implied H₂ feedstock**. |
| `Offtaker_Import` | `EPImporter` | Imports EP from outside the system at `ImportCost`. No H₂ or GC involvement. Acts as a price cap on the EP market. |

### 3.4 Electricity GC Demand Agent

| Agent | Type | Description |
|---|---|---|
| `Demand_GC_Elec_01` | `GC_Demand` | Elastic demand for electricity GCs. Quadratic utility `U(d) = A_GC·d − ½B_GC·d²`. Bounded by `PeakLoad × load_profile`. |

### 3.5 EP Demand Agent (Placeholder)

Currently empty (`EP_Demand: {}`). EP demand is inelastic and fully defined by `EP_market.Total_Demand × normalized_profile`. The block is a placeholder for future elastic EP demand agents.

---

## 4. Equilibrium Theory: MCP Structure, Competition, and Objectives

This section states the **economic problem** the model solves, its classification as a **mixed complementarity problem (MCP)**, and how that relates to the **competitive** (Walrasian) equilibrium solved by ADMM and the **social planner** benchmark. Agent-level formulas and constraint tables are in §5; the ADMM loop is in §6; the planner solve is in §7.

### 4.1 What kind of equilibrium is this?

At the economic level (ignoring ADMM penalties), the decentralised model (`market_exposure.jl`, `me_pap.jl`, `me_top.jl`, or `me_sop.jl`) seeks a **simultaneous market equilibrium** in which:

1. **Each agent** optimises its own objective (profit, utility, or risk-adjusted loss) subject to its **technological and regulatory constraints**, treating market prices as **given**.
2. **All spot markets** clear: total supply equals total demand in electricity, electricity GC, hydrogen, H₂ GC, and end product at every timestep $(h,d,y)$.
3. **Capacity** (VRES, electrolyzer, green offtaker) is chosen **once** before weather uncertainty resolves (non-anticipative scalar per agent), consistent with a **long-run competitive capacity** problem with stochastic operations.
4. **Prices** $\lambda_k(h,d,y)$ are the multipliers that enforce balance — commodity **shadow prices** at the optimum.

This is a **price-taking competitive equilibrium** — a **multi-commodity, temporal partial equilibrium** (indexed by hour, representative day, and scenario year), **not** a **spatial** (nodal) network model: there are no buses, pipelines, or transport arcs between locations. It is **not** a Nash–Cournot game: no agent optimises against a conjectured reaction of rivals' quantities on price. See §4.3 for the Gabriel-style taxonomy (perfect-competition MCP, not EPEC/Cournot).

The **social planner** (`social_planner.jl`) solves a **single** optimisation — welfare maximisation with optional social CVaR — whose KKT conditions (at $\gamma=1$, convexity) characterise the **same** competitive allocation (first welfare theorem). It is the centralised mathematical dual of the decentralised equilibrium, not a separate game. **ADMM** (`market_exposure*.jl`) is the decentralised **solver** for that equilibrium (§6); it does not change the economic definition.

### 4.2 MCP, KKT, and variational inequality form

A **mixed complementarity problem (MCP)** collects:

- **Primal variables** $x$ (quantities, capacities, CVaR auxiliaries),
- **Dual variables** $\lambda$ (market prices),
- **Equations** $F(x,\lambda)=0$ (stationarity / first-order conditions),
- **Complementarities** $0 \le a \perp b \ge 0$ linking primal feasibility and dual sign.

For this model, at ADMM convergence (penalties inactive), the economic equilibrium is equivalently described as:

**Agent problems.** Each agent $i$ solves a convex program (QP or QCP with CVaR linearisation):

$$
\begin{aligned}
& \min_{x_i \in \mathcal{X}_i} \; \pi_i(x_i, \lambda) \\
& \quad\text{or}\quad \min_{x_i \in \mathcal{X}_i} \; \gamma_i \mathbb{E}[\ell_i(x_i,\lambda)] + (1-\gamma_i)\,\mathrm{CVaR}_{i}(\ell_i)
\end{aligned}
$$

where $\mathcal{X}_i$ encodes technology (capacity, conversion, mandates) and $\pi_i$ is **private expenditure minus revenue** (equivalently **negative profit**). First-order conditions:

$$
\nabla_{x_i} \pi_i(x_i, \lambda) + \sum_k \lambda_k \, \nabla_{x_i} g_i^k(x_i) \in \mathcal{N}_{\mathcal{X}_i}(x_i)
$$

where $g_i^k$ is agent $i$'s net position in market $k$ (supply positive, demand negative).

**Market clearing.** For each market $k$ and timestep:

$$
\sum_{i \in \mathcal{I}_k} g_i^k(x_i) = D_k
$$

(EP includes exogenous $D_{\mathrm{EP}}$; other markets have $D_k=0$ unless an elastic demand agent is present.)

**Prices.** $\lambda_k$ are the **free multipliers** on these **equality** balance constraints (signed prices allowed). There is no $0 \le \text{imbalance} \perp \lambda \ge 0$ pair on pool markets because balance is modelled as **equality**, not inequality.

Together, stationarity + balance is a **nonlinear complementarity system** — an MCP in the broad sense, or a **variational inequality (VI)** on the product of feasible sets $\times\,\mathbb{R}^{|\lambda|}$. Under convexity and Slater conditions, it coincides with the **KKT system** of the coupled competitive problem.

**What ADMM is.** ADMM is a **distributed algorithm** that alternates agent subproblem solves with price updates until the MCP/KKT conditions are satisfied. The quadratic $\rho$-penalties are **not** part of the economic equilibrium; they vanish at the solution. The equilibrium object is the **risk-adjusted competitive MCP** defined by agent objectives + clearing + technology; ADMM is one numerical method to find it.

### 4.3 Why this is not Nash — and what MCP class it is (Gabriel taxonomy)

#### Not Nash–Cournot

In a **Nash–Cournot** (or **Nash–Bertrand**) model, agent $i$'s problem would include **strategic feedback**: a change in $q_i$ shifts the clearing price $\lambda(q)$, so the first-order condition contains terms like $\partial \lambda / \partial q_i$. Equilibrium requires **mutual best responses** — each agent's optimum given rivals' quantities.

In this codebase, $\lambda_k(h,d,y)$ enters each agent objective as a **fixed parameter** updated only by the **outer** ADMM / clearing loop. Inside `build_*_agent!`, no agent sees rivals' decision variables and no derivative $\partial \lambda / \partial q_i$ appears. Agent $i$ solves "my cost minus revenue at these prices," not "my profit given that the market will re-clear after I move."

That is the definition of **perfect competition / price taking**. The number of agents (one VRES block vs many) is a **modelling aggregation choice**, not the Cournot "few firms" definition — what matters is whether **market power** is in the mathematics. Here it is not.

| | **This model** | **Nash–Cournot** | **Monopoly** |
|---|---|---|---|
| **Agents** | Many price-taking types | Few strategic suppliers | One decision maker |
| **$\lambda$ in agent FOC** | Exogenous parameter | Endogenous via $\partial \lambda/\partial q_i$ | From inverse demand |
| **Equilibrium** | Competitive MCP / KKT | Nash in quantities | Single optimizer |
| **Gabriel label** | Perfect-competition MCP | Cournot oligopoly MCP | Not applicable (single firm) |

#### Not EPEC / MPEC / hierarchical

Gabriel, Conejo, Fuller, Hobbs & Ruiz, *Complementarity Modeling in Energy Markets*, distinguish:

- **MCP** — coupled stationarity + complementarity (e.g. market clearing); several agents linked by shared $\lambda$.
- **MPEC** — one **optimizer** whose constraints include another problem's equilibrium (KKT or MCP).
- **EPEC** — **multiple** leaders, each solving an MPEC; followers form a **Nash** or competitive equilibrium among themselves.

This project is a **plain competitive MCP** (or KKT system): symmetric agents, same level, no leader–follower stack. It is **not** EPEC (no upper-level planner/regulator with equilibrium constraints). It is **not** MPEC unless you artificially wrap the planner around the market (the social planner is a separate **benchmark** solve, not an EPEC upper level).

#### Not spatial price equilibrium (SPE)

In Gabriel (and Takayama & Judge), **spatial price equilibrium** means **nodes + arcs**: goods move between locations; nodal prices differ by transport cost; Kirchhoff-type or mass-balance on a **network**.

This model has **no geography**. Indices $(h,d,y)$ are **time and scenario**, not space. The correct name is **multi-market temporal competitive equilibrium** or **intertemporal partial equilibrium** across coupled commodities (elec, GC, H₂, H₂-GC, EP). Five markets clear at each $(h,d,y)$; the electrolyzer **couples** markets technologically, not via a transport network.

#### What to call it (Gabriel-aligned)

| Name | Fits this model? |
|---|---|
| **Perfect-competition / Walrasian MCP** | **Yes** — price takers, clearing as equations, KKT coupling |
| **Multi-commodity temporal equilibrium** | **Yes** — no spatial network |
| **Partial equilibrium** | **Yes** — not full economy-wide GE with income feedback |
| **Spatial price equilibrium (SPE)** | **No** — requires nodes and transport |
| **Cournot / Nash oligopoly MCP** | **No** |
| **EPEC / MPEC** | **No** — no hierarchical structure in ME |
| **Monopoly** | **No** |

At $\gamma=1$, the social planner is the **dual** of this competitive MCP (welfare maximisation). At $\gamma<1$, SP and ME are different **risk institutions** (§4.8), still not a game.

**Capacity investment** is part of the same competitive long-run problem: fixed annualised CAPEX $F_{\mathrm{cap}} \cdot \mathrm{cap}$ enters the cost side; at equilibrium, marginal value of capacity equals annuity (up to risk premia when $\gamma<1$).

**Mandates** (42% H₂ GC) are **regulatory constraints** in $\mathcal{X}_i$, not strategic rules. They couple markets but do not turn the model into a game.

### 4.4 Why objectives are defined this way

Economic primitives are chosen so that **competitive equilibrium = welfare optimum** when $\gamma=1$ and markets are complete in the planner sense.

**Suppliers (VRES, conventional, electrolyzer, offtakers).** Minimise **cost minus revenue** (equivalently maximise profit):

$$
\min \sum_{h,d,y} W_{d,y}\bigl(\mathrm{VC}(q) + \mathrm{FC}\cdot\mathrm{cap} - \lambda^\top g\bigr)
$$

- **VRES:** $\mathrm{VC} \approx 0$; revenue from elec + elec_GC; CAPEX on `cap_VRES`.
- **Conventional:** flat variable cost `MC[y] × g` per plant (or legacy convex **staged** stack when `StageTechnologies` is set) — merit order via market price when using flat plants.
- **Electrolyzer:** buys elec + elec_GC; sells H₂ + H₂_GC; pays OPEX; CAPEX on electrical MW (IEA kWe, converted onto H₂-side capacity); **green-backing** links GC purchases to H₂ GC issuance.
- **Green/grey offtakers:** processing/import cost; sell EP; buy H₂ and/or H₂ GC per route; **GC mandate** on implied or actual H₂ use.

**Consumers (electricity, GC).** Minimise **expenditure minus utility**:

$$
\begin{aligned}
& \min \sum_{h,d,y} W_{d,y}\bigl(\lambda \cdot d - U(d)\bigr) \\
& \quad\Leftrightarrow\quad \max \sum W_{d,y}\, U(d) - \lambda\cdot d
\end{aligned}
$$

Quadratic $U(d) = A d - \frac{B}{2}d^2$ gives **linear inverse demand** $p(d)=A-Bd$ — a standard **partial-equilibrium** demand specification. Maximising utility minus expenditure at prices $\lambda$ is the **Marshallian** form of the same competitive consumer problem.

**Sign convention.** Net position $g_i^k>0$ = supply; $g_i^k<0$ = demand. Market imbalance $r_k = \sum_i g_i^k - D_k$ must be zero at equilibrium.

**Transfers.** Revenue for one agent is expenditure for another; they **cancel in social welfare** (§5.3). The planner tracks **real** costs and utilities, not financial transfers.

### 4.5 Risk-adjusted competitive equilibrium ($\gamma < 1$)

With **private CVaR** (ME / ME+C), risk-averse agents minimise $\gamma_{i}\,\mathbb{E}[\ell_{i}] + (1-\gamma_{i})\,\mathrm{CVaR}_{i}(\ell_{i})$ with **full** loss $\ell_{i}$ (operational + $F_{\mathrm{cap}}\cdot\mathrm{cap}$) — still **price-taking**, but **incomplete risk trading** (d'Aertrycke et al.). The **social planner** pools tail risk via **one social CVaR** on aggregate welfare (**complete risk trading**). Definitions, $\gamma$ vs $\beta$, and effects on investment: **§4.10**. Labels: **§4.8**. Planner maths: **§7.4**.

### 4.6 Social planner as the welfare dual

The planner maximises (§7):

$$
\begin{aligned}
\max \;& \gamma \sum_y P_y\,\mathrm{sw}^{\mathrm{aux}}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}} \\
\text{s.t.}\;& \text{all agent constraints + market clearing}
\end{aligned}
$$

Per-agent welfare contributions are **utility minus real cost** (no $\lambda$ terms). At $\gamma=1$, if the coupled problem is convex, any competitive equilibrium $(x^{\ast},\lambda^{\ast})$ solves the planner and vice versa (**first welfare theorem**). That is why SP and ME should agree at $\gamma=1$ when ADMM converges.

The planner is **not** an MCP solved directly as complementarity; it is a **mathematical program**. Its KKT multipliers on balance constraints are the **competitive prices** (§7.2).

### 4.7 Contracts case (ME+C): still competitive MCP

`me_pap.jl`, `me_top.jl`, or `me_sop.jl` adds **PPA** and **HPA** bilateral pools (§2):

- Same **price-taking** structure on pool markets and on contract clearing prices $\lambda_{\mathrm{ppa}}$, $\lambda_{\mathrm{hpa}}$.
- **Settlement** at scalar strike $K$ (PPA hedge; HPA volume mode set by entry point). $K$ and bilateral capacity choices evolve each ADMM iteration and are snapshotted at convergence.
- **Contract capacity** $C$ per link is a bilateral consensus variable (each side enters a local cap decision; ADMM drives agreement).

Economically: contracts add bilateral **cash-flow/risk-sharing channels** and additional clearing conditions for contract energy/capacity, while spot markets keep physical dispatch balance. At $\gamma < 1$, contracts can appear because **private CVaR** values fixed-price bilateral cash flows; at $\gamma = 1$, unused contracts ($C \approx 0$) are equilibrium-consistent.

### 4.8 Literature labels and price interpretation (d'Aertrycke et al.)

Mapping to d'Aertrycke, Ehrenmann, Ralph & Smeers (2018), *Risk trading in capacity equilibrium models* (see §14):

| Entry point | $\gamma = 1$ | $0 < \gamma < 1$ |
|---|---|---|
| **`social_planner.jl`** | Risk-neutral **competitive capacity equilibrium** (stochastic welfare max; duals = expected marginal social value) | **Competitive capacity equilibrium with complete risk trading** (social CVaR on aggregate welfare; duals = risk-adjusted social shadow prices) |
| **`market_exposure.jl`** | Risk-neutral decentralised competitive equilibrium (ADMM); should match SP | **Competitive capacity equilibrium with incomplete risk trading** (private per-agent CVaR; ADMM $\lambda$ = equilibrium commodity prices for that institution) |
| **`me_pap.jl`, `me_top.jl`, or `me_sop.jl`** | Same as ME, plus PPA/HPA pools | Same incomplete-risk-trading label as ME, with bilateral contract pools |

**Complete risk trading (SP, $\gamma < 1$):** one system-wide CVaR on aggregate welfare — centralised tail-risk pooling.

**Incomplete risk trading (ME / ME+C, $\gamma < 1$):** private CVaR per agent; no modelled risk market. ME+C adds bilateral **hedging/settlement** contracts, not complete financial risk trading.

**Prices when $\gamma < 1$:**

- **SP:** balance duals divided by $W_{d,y}\cdot\mu_y$ (§7.2) are **risk-adjusted social shadow prices** — valid KKT multipliers of the planner QCP, not required to equal ME prices.
- **ME:** ADMM $\lambda$ at convergence are **competitive equilibrium prices** under incomplete risk trading — each agent's FOCs include private CVaR.

SP–ME price gaps at $\gamma<1$ measure **different risk institutions**, not solver error.

### 4.9 Summary diagram

```text
                    ┌─────────────────────────────────────┐
                    │   Competitive equilibrium (MCP/KKT)  │
                    │   • Price-taking agents              │
                    │   • Market clearing (5 pool markets) │
                    │   • Technology + mandates            │
                    │   • Optional private CVaR (ME)       │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     social_planner.jl    market_exposure.jl    me_pap.jl / me_top.jl / me_sop.jl
     (welfare max +        (ADMM solves MCP)     (+ PPA/HPA pools)
      social CVaR)
              │                    │
              └──────── γ=1 ───────┘  same primal/dual (welfare theorem)
                        γ<1          different risk institution
```

### 4.10 Risk aversion: CVaR, γ, and β

This subsection is the **main reference** for risk in the model: CVaR definitions, the **Hoschle et al.** two-parameter calibration ($\gamma$ then $\beta$), and equilibrium effects on investment and prices. Objective structure follows Höschle, Le Cadre, Smeers, Papavasiliou & Belmans (2018) and d’Aertrycke et al. (2018); see §14 refs. 5 and 7.

#### 4.10.1 Random loss and why we care about tails

Weather scenarios $y \in JY$ (e.g. ten scenarios labelled $1,\dots,10$ with distinct VRES profiles) create **uncertainty** in revenues and costs. For each risk-averse agent $i$, define **per-scenario loss** $\ell_{i,y}$ (€):

$$
\ell_{i,y} = \underbrace{\sum_{h,d} W_{d,y}\bigl(\mathrm{cost}_{i}(h,d,y) - \mathrm{rev}_{i}(h,d,y)\bigr)}_{\text{operational loss in year }y} + \underbrace{F_i^{\mathrm{cap}}\cdot \mathrm{cap}_i}_{\text{annualised CAPEX}}
$$

**Operational** loss depends on scenario (wind, sun, prices). **Capacity** `cap_i` is one scalar chosen **before** knowing which scenario occurs — so a large investment hurts in **every** scenario if revenues disappoint. CVaR must use this **full** $\ell_{i,y}$ (code: `loss_total[y]`); see §5.1.

A **risk-neutral** agent ($\gamma=1$) cares only about $\mathbb{E}[\ell_i] = \sum_y P_y \ell_{i,y}$. A **risk-averse** agent also dislikes **bad tail outcomes** — years where loss is much worse than average (e.g. low VRES output, high fuel prices, weak margins).

#### 4.10.2 VaR and CVaR — definitions

Fix confidence level $\beta \in (0,1)$.

- **Value-at-Risk (VaR)** at level $\beta$: a threshold $\alpha$ such that loss exceeds $\alpha$ with probability at most $1-\beta$ (in the discrete scenario case, a quantile of the loss distribution).

- **Conditional Value-at-Risk (CVaR)** at level $\beta$: the **expected loss in the worst $(1-\beta)$ fraction** of scenarios (tail average). Also called **Expected Shortfall**. Example: $\beta=0.8$ ⇒ average loss in the worst **20%** of weather years (extreme tail); $\beta=0.2$ ⇒ average in the worst **80%** (broad tail, typically a **lower** CVaR number because milder bad years are included).

Rockafellar & Uryasev (2000) show CVaR is **coherent** (subadditive, monotone) and can be optimised by convex programming:

$$
\mathrm{CVaR}_{\beta}(\ell) = \min_{\alpha} \left\lbrace \alpha + \frac{1}{1-\beta}\,\mathbb{E}\bigl[(\ell - \alpha)_{+}\bigr] \right\rbrace, \quad (x)_{+} = \max(x,0)
$$

**In code** (agents and planner), this becomes linear constraints with auxiliaries $\alpha$, $u_y$, $\mathrm{CVaR}$:

$$
u_y \ge \ell_y - \alpha \quad \forall y, \qquad \mathrm{CVaR} \ge \alpha + \frac{1}{1-\beta}\sum_y P_y\, u_y
$$

Minimising CVaR in the objective (or penalising it) pushes the solution toward **lower tail losses**, not only lower average loss.

#### 4.10.3 The $\gamma$ objective — risk-neutral vs risk-averse

**Market exposure (private CVaR)** — VRES, electrolyzer, green offtaker minimise:

$$
\min \;\; \gamma_i \,\mathbb{E}[\ell_i] + (1-\gamma_i)\,\mathrm{CVaR}_{i}(\ell_i) + \text{(ADMM penalties)}
$$

equivalently written in code as $\gamma \cdot (F_{\mathrm{cap}}\cdot\mathrm{cap} + \sum_y P_y \ell^{\mathrm{op}}_y) + (1-\gamma)\cdot\mathrm{CVaR}$ with $\ell^{\mathrm{op}}$ the operational part.

| $\gamma$ | Name | Objective emphasis |
|---|---|---|
| **$1$** | **Risk-neutral** | $(1-\gamma)\,\mathrm{CVaR} = 0$ — only $\mathbb{E}[\ell]$ matters. SP and ME should agree (§5.4.2). |
| **$0.5$** | **Risk-averse (Hoschle base)** | Equal weight on **mean loss** and **CVaR**; then sweep $\beta$ for intensity (§4.10.4). |
| **$\to 0$** | **Very tail-focused** | Almost only CVaR matters — extremely conservative toward worst scenarios. |

**Social planner** maximises welfare with symmetric structure on **social loss** $L_y = -\mathrm{SW}_y$:

$$
\max \;\; \gamma \sum_y P_y\,\mathrm{sw}^{\mathrm{aux}}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}_{\beta}(-\mathrm{SW})
$$

$\gamma=1$: maximise expected social welfare only. $\gamma<1$: trade some expected welfare for **lower tail risk** on aggregate outcomes (complete risk trading).

**Why $\gamma=1$ is risk-neutral:** the CVaR term is multiplied by $(1-\gamma)$. At $\gamma=1$ it **drops out of the objective** entirely; optimality is driven only by expected profit/welfare.

#### 4.10.4 Hoschle-style calibration: $\gamma$ then $\beta$

This project follows the **two-step risk calibration** used in Hoschle et al. (2018) and related equilibrium literature:

| Step | Parameter | Setting | Role |
|---|---|---|---|
| **1 — Benchmark** | $\gamma$ | **$1$** | Risk-neutral: only $\mathbb{E}[\ell]$ / expected welfare. SP and ME should agree (§5.4.2). |
| **2 — Turn on CVaR** | $\gamma$ | **$0.5$** | Equal weight on **mean** and **CVaR** in the objective (Hoschle case-study default for risk-averse runs). |
| **3 — Risk-aversion intensity** | $\beta$ | **$0.2,\,0.4,\,0.6,\,0.8$** | At fixed $\gamma=0.5$, sweep $\beta$ to vary how aggressively agents penalise bad scenarios. With 15 scenarios these correspond to tails of exactly 12, 9, 6, and 3 scenarios. |

**Higher $\beta$ ⇒ more risk-averse** (Rockafellar–Uryasev confidence level in code). Mechanism:

- $\mathrm{CVaR}_{\beta}$ averages loss over the worst $(1-\beta)$ share of scenarios.
- $\beta=0.2$ ⇒ worst **80%** (12 of 15) — broad tail average, mildest CVaR penalty in the standard sweep.
- $\beta=0.4$ ⇒ worst **60%** (9 of 15).
- $\beta=0.6$ ⇒ worst **40%** (6 of 15).
- $\beta=0.8$ ⇒ worst **20%** (3 of 15) — narrow, extreme tail, strongest CVaR penalty in the standard sweep.

**Lower bound on $\beta$'s resolution.** The tail must contain at least one scenario: $(1-\beta) \ge 1/n_{\mathrm{scen}}$. With 15 scenarios that caps $\beta$ at **0.933**; anything finer (including the `data.yaml` default of `0.95`) collapses CVaR onto the single worst scenario and makes $\beta$ inert. `print_risk_metrics_summary!` detects this and prints an explicit warning with the correct $\beta$ values.

**$\gamma$ and $\beta$ are complementary, not interchangeable:**

- **$\gamma$** switches risk aversion **on/off** and sets the **split** between $\mathbb{E}[\cdot]$ and $\mathrm{CVaR}_{\beta}(\cdot)$ in the objective.
- **$\beta$** defines **which part of the loss distribution** enters CVaR once $\gamma<1$. At fixed $\gamma=0.5$, varying $\beta$ traces **increasing risk aversion** as $\beta$ **increases** (from $0.2$ to $0.8$).

**Two $\beta$ conventions (Hoschle vs this codebase).** Hoschle et al. (2018) and this project both use a $\gamma$–CVaR objective of the form $\gamma\,\mathbb{E}[\cdot] + (1-\gamma)\,\mathrm{CVaR}$, but they assign **risk neutrality** and **tail depth** differently. Hoschle fixes $\gamma=0.5$ for risk-averse case studies and sweeps **their** $\beta$ from $1$ (risk-neutral on their axis) down to $0.1$ (very risk-averse); **lower Hoschle $\beta$ = stronger aversion**. The CVaR constraints in code follow **Rockafellar–Uryasev**: $\beta$ is a **confidence level**, $\mathrm{CVaR}_{\beta}$ averages the worst $(1-\beta)$ share of scenarios, and **higher $\beta$ = narrower, more extreme tail = stronger aversion** at fixed $\gamma<1$. This codebase sets **$\gamma=1$** for the risk-neutral benchmark (the $(1-\gamma)\,\mathrm{CVaR}$ term vanishes) rather than Hoschle’s $\beta=1$, then uses **$\gamma=0.5$** with Rockafellar $\beta\in\lbrace 0.2,0.4,0.6,0.8\rbrace$ for the sensitivity sweep. The **economic direction** is the same (more conservative outcomes as aversion rises), but the **parameter labels are not interchangeable**: Hoschle $\beta=0.2$ (high aversion) is not the same run as Rockafellar $\beta=0.2$ (broad 80% tail, mildest in our sweep).

In `data.yaml`, both SP and ME read **`ADMM.gamma`** and **`ADMM.beta`**. Defaults: `gamma: 1.0`, `beta: 0.95`. The shipped $\beta$ is a **placeholder that is inactive at $\gamma=1$** and must be lowered to at most $0.933$ before any risk-averse run — see the resolution bound above.

**Per-agent `gamma`/`beta` in agent blocks do not currently override.** Entry points build each agent's data dict as `merge(General, agent_block, ADMM)`, so the ADMM block wins whenever both define the same key. The duplicate keys on VRES / electrolyzer / green offtaker are therefore inert while `ADMM.gamma` / `ADMM.beta` are set — change risk aversion in the `ADMM` block (or remove those keys from it) if you want per-agent values to take effect.

**Multi-scenario requirement:** with one scenario, $\mathrm{CVaR}=\ell$ always — changing $\gamma$ has **no effect** on the optimum if `loss_total` is specified correctly (§5.1). Risk aversion is meaningful once the `Scenarios` grid has more than one entry; the default grid has **15** (5 weather years × 3 gas levels, §9.8). Note also that $\beta$ must be coarse enough for the grid: with 15 equiprobable scenarios, `beta = 0.95` asks for a tail narrower than a single scenario, so use $\beta = 1 - k/15$ for a tail of $k$ scenarios (e.g. `0.8` for the worst 3).

#### 4.10.5 Complete vs incomplete risk trading (reminder)

| | **SP ($\gamma<1$)** | **ME ($\gamma<1$)** |
|---|---|---|
| **Who is risk-averse?** | Society once (social CVaR on $\mathrm{SW}_y$) | VRES, electrolyzer, green offtaker **separately** (private CVaR) |
| **Risk market?** | Complete pooling (centralised) | No explicit risk trading |
| **Compare quantities to SP?** | SP is benchmark | Generally **no** (§4.8) |
| **Compare `Risk_Metrics.csv`?** | Yes — ex-post social CVaR gap (§7.6) | Yes |

#### 4.10.6 What changes in strategy when $\gamma < 1$ or $\beta$ rises?

Risk aversion reshapes **capacity**, **dispatch**, and **prices** because bad scenarios get **more weight** in the optimiser (directly via private CVaR in ME, via social CVaR in SP). **Within** $\gamma=0.5$ runs, **increasing $\beta$** strengthens this effect: CVaR focuses on a **narrower, more extreme** tail, so capacity and dispatch shift further toward hedging catastrophic years. (Hoschle et al. Fig. 6 shows monotonic capacity shifts as **their** $\beta$ decreases from $1$ toward $0$ at fixed $\gamma=0.5$ — analogous direction to **increasing** Rockafellar $\beta$ here.)

**VRES (solar/wind):**
- **Low-renewable scenarios** ($y$ with weak SOLAR/WIND) imply low output per MW installed → high $\ell_{i,y}$ for fixed `cap_VRES`.
- Higher effective penalty on tail → often **lower optimal `cap_VRES`** or less aggressive investment than at $\gamma=1$, and dispatch that hedges tail revenue risk.

**Electrolyzer:**
- Bad VRES years → high electricity prices and/or low own output → electrolyzer margins worsen in tail.
- Risk-averse electrolyzer → often **smaller `cap_H2`**, less exposure to “green H₂ bet” across weather draws; may shift operational patterns that reduce tail procurement cost.

**Green offtaker:**
- Tail scenarios with expensive H₂/GC or weak EP margins → **lower `cap_EP`** or tighter coupling of H₂ intake to mandate at minimum cost in bad years.

**System-level (SP vs ME at $\gamma<1$):**
- **Private** CVaRs do not internalise **social** tail risk — ME can have **worse ex-post social CVaR** than SP (§7.6) even when each agent “hedges” privately.
- **Prices** $\lambda$ shift: agents' FOCs include CVaR gradients, so equilibrium prices differ from $\gamma=1$ and from SP prices.
- **Consumers/conventional/grey** agents have no CVaR in the model — risk enters only through **market prices** and quantities set by risk-averse partners.

**Why investment moves more than hourly dispatch:** `cap` is committed for **all** scenarios; CVaR on `loss_total` penalises “large cap + bad weather” combinations. Dispatch can still vary by $y$ within each scenario; capacity is the main **irreversible** risk lever.

#### 4.10.7 Practical parameter workflow (Hoschle-style)

1. **Risk-neutral benchmark:** `gamma = 1` (any `beta`; inactive). Run SP then ME — verify convergence and quantity/price match (§5.4.2). Risk parameters only matter when $\gamma<1$.
2. **Risk-averse base case:** `gamma = 0.5` in the `ADMM` block (applies to SP and ME simultaneously), with `beta = 0.8` so the tail spans the worst 3 of the 15 scenarios.
3. **Risk-aversion sweep:** at fixed `gamma = 0.5`, run separate cases with `beta = 0.2, 0.4, 0.6, 0.8` — **higher `beta` = more risk-averse** (narrower tail). With 15 scenarios these land exactly on whole scenario counts (worst 12, 9, 6, and 3 respectively, since $\beta = 1 - k/15$), so every point in the sweep is well resolved. Compare capacities, prices, and `Risk_Metrics.csv` across the sweep.
4. **Institution comparison:** for each $(\gamma,\beta)$ pair, compare SP (complete risk trading) vs ME (incomplete); do **not** expect equal quantities/prices at $\gamma<1$ (§4.8, §7.6).

#### 4.10.8 Auxiliary variables (reading outputs)

| Symbol (code) | Meaning |
|---|---|
| `alpha_*` / `alpha_social` | Optimised VaR threshold $\alpha$ for CVaR formula |
| `u_*[y]` / `u_social[y]` | Shortfall $(\ell_y - \alpha)_{+}$ in scenario $y$ |
| `CVaR_*` / `CVaR_social` | Tail-average loss at level $\beta$ |
| `sw_aux[y]` | Planner epigraph proxy for social welfare in $y$ (§7.4) |

### 4.11 Partial social planners (green coalitions via ADMM)

Between **full decentralisation** (ME: private CVaR per firm) and the **full social planner** (one social CVaR on aggregate welfare), the model supports two **partial** benchmarks solved with the **same ADMM price loop** as ME:

| Entry point | Coalition agent | Merged members | Internal risk institution |
|---|---|---|---|
| `green_h2_social_planner.jl` | `GreenH2_Coalition` | Electrolyzer + green offtaker | **One coalition CVaR** on combined loss (complete sharing **within** the H₂→EP chain) |
| `green_social_planner.jl` | `Green_Coalition` | Solar + wind + electrolyzer + green offtaker | Same, extended to the full green supply chain |

**What stays decentralised:** consumer, conventional generator, grey/import offtakers, GC demand — same spot-market clearing and ADMM $\lambda$ updates as ME.

**What changes inside the coalition:**

- **Physical internal links** replace spot H₂ (and internal H₂-GC) trades: `h2` flows electrolyzer → offtaker; `ep = α·h2`.
- **Green coalition** additionally internalises VRES output against electrolyzer purchases: net elec position = $\sum g_{\mathrm{VRES}} - e_{\mathrm{in}}$; net elec-GC = $\sum g_{\mathrm{VRES}} - q_{\mathrm{elec\_gc}}$.
- **One CVaR** (`CVaR_coalition`) on the coalition’s full per-scenario loss (operational + all fixed capacity costs), with $(1-\gamma)$ weight — analogous to social CVaR but **only for the merged firms**.

**What does not change:** five external spot markets still clear via ADMM; prices are equilibrium $\lambda$ for the partial-planner institution, not SP duals.

**Implementation:** ME-style entry scripts (`green_h2_social_planner.jl`, `green_social_planner.jl`) with merged agent modules (`define_merged_agent.jl`, `build_merged_agent.jl`, `solve_merged_agent.jl`, `merged_agent_setup.jl`); one branch in `ADMM_subroutine.jl` for `agents[:merged]`. `PartialPlanners` block in `data.yaml` (§9.2.2). Base `market_exposure.jl` / `social_planner.jl` untouched.

**Outputs:** `green_h2_social_planner_results/` or `green_social_planner_results/` — same CSV layout as ME (`save_results.jl`).

**Comparison ladder (same $\gamma$, $\beta$):**

1. **ME** — incomplete private CVaR (worst tail allocation among decentralised cases).
2. **GH2-SP / G-SP** — complete sharing **inside** the green coalition only.
3. **SP** — complete sharing on **aggregate social welfare** (best tail management centrally).

Do **not** expect GH2-SP and G-SP to match each other or SP at $\gamma<1$; compare each to SP on ex-post social tail metrics (§7.6).

---

## 5. Mathematical Formulation

### 5.1 Agent Objectives (ADMM)

Each agent minimises its **augmented Lagrangian** (possibly risk-averse for some agents). For a capacity-owning, risk-aware agent (VRES, electrolyzer, green offtaker) the form realised in code is:

$$
\begin{aligned}
\min \quad & \gamma_i \Bigl(F_i^{\mathrm{cap}}\cdot\mathrm{cap}_i + \sum_y P_y\,\ell^{\mathrm{op}}_{i,y}\Bigr)
           + (1-\gamma_i)\,\mathrm{CVaR}_{i}(\ell_i) \\
& \quad + \sum_k \frac{\rho_k}{2}\sum_{h,d,y} W_{d,y}\bigl(g_i^k(h,d,y)-\bar{g}_i^k(h,d,y)\bigr)^2 \\
& \quad + \lambda_i^{\mathrm{cap}}\,(\mathrm{cap}_i - z_i^{\mathrm{cap}})
           + \frac{\rho_i^{\mathrm{cap}}}{2}\,(\mathrm{cap}_i - z_i^{\mathrm{cap}})^2
\end{aligned}
$$

where the operational loss in scenario $y$ is the representative-day-weighted private cost–revenue over that year's hours,

$$
\ell^{\mathrm{op}}_{i,y} \;=\; \sum_{h,d} W_{d,y}\bigl(\mathrm{cost}_i(h,d,y) - \mathrm{rev}_i(h,d,y)\bigr),
$$

and the total loss that enters CVaR is $\ell_{i,y} = F_i^{\mathrm{cap}}\cdot\mathrm{cap}_i + \ell^{\mathrm{op}}_{i,y}$ (CAPEX is non-anticipative, so it appears in every scenario). Symbols map to code names as follows:

- `cost_i − revenue_i` is the agent's private cost minus revenue across all markets (fuel/operational costs and certificate purchases on the cost side; price × net position on the revenue side).
- `P_y` is the scenario probability (`P[jy] = 1/nYears`; uniform on the default 15-scenario grid).
- `g_i^k` is the agent's net position in market `k` (positive = supply, negative = demand).
- `ḡ_i^k` is the consensus target for agent `i` in market `k`.
- `ρ_k` is the penalty weight for market `k`.
- `W[d,y]` scales representative days to a full year.
- `γ_i` is a **per-agent risk weight** (`γ=1` → risk-neutral, `γ<1` → risk-averse). Non-trivial CVaR is used only for VRES, electrolyzer, and green offtaker.
- $\mathrm{CVaR}_{i}(\ell_i)$ is an agent-specific Conditional Value-at-Risk on yearly loss scenarios, with auxiliary variables $\alpha_i$, $u_i(y)$ over $y \in JY$, at confidence level $\beta$.
- $(\lambda_i^{\mathrm{cap}},\,z_i^{\mathrm{cap}},\,\rho_i^{\mathrm{cap}})$ are the **per-agent capacity equality-split** terms (§6.4); agents without endogenous capacity omit them. Agents without CVaR (conventional, consumer, grey offtaker, importer, GC demand) also drop the $\gamma$/CVaR wrapping and minimise plain expected cost–revenue + ADMM penalties.

Three points where this form is easy to get wrong:

1. **CAPEX sits inside the $\gamma$-weighted term**, not as a separate addend outside it. At $\gamma=1$ the agent still pays the full annuity; at $\gamma<1$ CAPEX is also inside every scenario's loss that feeds CVaR.
2. **Scenario probabilities $P_y$ weight the expected operational loss.** Writing a bare $\sum_{h,d,y} W_{d,y}(\cdots)$ without $P_y$ would overweight years when $nYears>1$.
3. **Capacity ADMM terms are part of the agent objective** every iteration, in addition to the market-quantity penalties. They vanish at consensus exactly as the market ones do.

The ADMM penalties (market and capacity) are **algorithmic** only (§4.2, §6); at convergence they vanish and the solution is the **risk-adjusted competitive MCP** of §4.

#### CVaR formulation (per agent)

For each risk-averse agent (VRES, electrolyzer, green offtaker), CVaR is linearised via three auxiliaries:

- `α_i` — VaR proxy (free variable)
- `u_i[jy]` — shortfall per scenario year (`≥ 0`)
- `cvar_i` — CVaR value (`≥ 0`)

with the Rockafellar–Uryasev constraints

$$
\begin{aligned}
u_{i,y} &\ge \ell_{i,y} - \alpha_i \quad \forall y \in \mathcal{Y} \\
\mathrm{CVaR}_{i} &\ge \alpha_i + \frac{1}{1-\beta}\sum_{y \in \mathcal{Y}} P_y\, u_{i,y}
\end{aligned}
$$

**Important.** The loss that enters CVaR must be the **full** per-scenario loss, including the fixed capacity cost (`F_cap × cap`). Capacity `cap` is a **scalar** (non-anticipative: the same installed MW in every weather scenario). If only the operational loss is used, then when $\gamma < 1$ the fixed cost appears only in the $\gamma$-weighted term, so the effective weight on `F_cap` becomes $\gamma$ instead of $1$. With one scenario, changing $\gamma$ would then change the objective, breaking the equivalence between social planner and market exposure. The correct formulation therefore uses $\ell_{i,y} = \ell^{\mathrm{op}}_{i,y} + F_i^{\mathrm{cap}}\cdot\mathrm{cap}_i$ in the shortfall constraints (same `cap` in every scenario). With one scenario, $\mathrm{CVaR}_{i} = \ell_i$, so the objective reduces to total loss regardless of $\gamma$.

**Dynamic constraint updates.** In ADMM, the loss expressions `loss_i[jy]` depend on iteration-specific market prices `λ` (which change every iteration). Because JuMP expressions bake in coefficient values at creation time, the CVaR shortfall and linking constraints must be **deleted and re-added** in every ADMM iteration with the freshly recomputed loss expressions. This happens in the `solve_*_agent!` functions.

#### Specific objective terms by agent type

The blocks below are **pseudocode** using JuMP / `data.yaml` names (backticks). They are not math-delimited; underscores are safe inside code fences.

**VRES generator (with endogenous capacity and CVaR):**

```text
min  γ × [ F_cap × cap_VRES + Σ_y P[y] × loss_VRES[y] ]
   + (1−γ) × CVaR_VRES
   + (ρ_elec/2)×Σ W×(g − ḡ_elec)²
   + (ρ_GC/2)×Σ W×(g − ḡ_GC)²

where loss_VRES[y] = Σ_{h,d} W × ( MC×g − λ_elec×g − λ_GC×g )
      g[h,d,y] ≤ AF[h,d,y] × cap_VRES     (same cap_VRES in all scenarios y)
```

**VRES in contracts case** (`build_power_agent_contracts.jl` / `solve_power_agent_contracts.jl`): Physical pool `g_EOM`; hedge `g_ppa`. CVaR loss includes the CfD `−(K − λ_elec − λ_elec_GC)×g_ppa` (not `λ_ppa×g_ppa`). The ADMM dual `−Σ W λ_ppa g_ppa` and `(ρ_ppa/2)(g_ppa − ḡ_ppa)²` sit **outside** CVaR. `ppa_cap` is JuMP-fixed to shared $C$. Bounds: `g_ppa ≤ C` and `g_ppa ≤ AF × cap_VRES`.

**Conventional generator (flat single plant, default NL calibration):**

```text
min Σ W × ( MC_y × g − λ_elec×g )  +  (ρ_elec/2)×Σ W×(g − ḡ_elec)²

subject to: 0 ≤ g ≤ Capacity
```

Each conventional agent names one `Technology` (fuel, efficiency, VOM). Its short-run marginal cost `MC_y` is **derived per scenario** from the `Fuel` block (§9.8). Three agents (`Gen_Conv_CCGT`, `Gen_Conv_Coal`, `Gen_Conv_Biomass`) replace the former single 3-stage stack; merit order (CCGT → coal → biomass at base gas, reordering under gas shocks) emerges from competitive clearing rather than an enforced monotonic stage curve.

**Legacy conventional (3-stage increasing cost):** still supported when `StageTechnologies` is set instead of `Technology`:

```text
min Σ W × ( Σ_s (base_s,y×g_s + 0.5×slope_s,y×g_s²) − λ_elec×g )  +  (ρ_elec/2)×Σ W×(g − ḡ_elec)²

subject to: g = Σ_s g_s,   0 ≤ g_s ≤ cap_s
```

Stage capacities come from `Capacity` and `StageCapacityShares` (normalised internally). Stage costs are **derived per scenario** from the `Fuel` block and the technology listed for each stage — `base_{s,y}` and `slope_{s,y}` therefore carry a scenario index $y$, because the gas price differs across the grid (§9.8). Slopes are set so marginal cost is continuous across stage boundaries (`end MC_1 = base_2`, `end MC_2 = base_3`, `end MC_3 = OCGT SRMC`). Changing shares reshapes the fleet's aggregate average variable cost.

**Why linear-within-stage is realistic enough:**
- Real unit commitment/economic dispatch stacks are piecewise and heterogeneous; aggregated systems are commonly approximated by piecewise-linear or piecewise-quadratic supply curves.
- Over a limited dispatch band per stage, a linear marginal-cost profile is a practical local approximation to heat-rate/fuel/emissions variability.
- Enforcing continuity across stage boundaries avoids artificial price jumps while preserving increasing scarcity cost as dispatch moves to higher-cost blocks.

**Consumer:**

```text
min Σ W × ( λ_elec×d − U(d) )  +  (ρ_elec/2)×Σ W×(−d − ḡ_elec)²
where U(d) = A_E×d − (B_E/2)×d²
```

**Electrolyzer (with endogenous H₂ capacity and CVaR):**

```text
min  γ × [ F_cap × cap_H2_y + Σ_y P[y] × loss_H2[y] ]
   + (1−γ) × CVaR_H2
   + (ρ_elec/2)×Σ W×(−e_in − ḡ_elec)²
   + (ρ_GC/2)×Σ W×(−gc_e − ḡ_GC)²
   + (ρ_H2/2)×Σ W×(h2 − ḡ_H2)²
   + (ρ_H2GC/2)×Σ W×(gc_h2 − ḡ_H2GC)²

where loss_H2[y] = Σ_{h,d} W × ( λ_elec×e_in + λ_GC×gc_e + op×h2 − λ_H2×h2 − λ_H2GC×gc_h2 )
      dispatch and conversion indexed by scenario y; cap_H2_y is scalar (non-anticipative)
```

**GreenProducer in contracts case** (`build_H2_agent_contracts.jl`): Physical conversion `h2_out = η×e_in_pool`. Hedge `g_ppa_from` / `h2_hpa` enter CVaR via PPA buyer CfD and HPA seller terms (ToP notional $C$ or SoP $Ks$). ADMM `λ_ppa`/`λ_hpa` linear terms and $\rho$ quadratics are outside CVaR. Caps are shared-$C$ fixes, not two-party consensus.

**GreenOfftaker in contracts case** (`build_offtaker_agent_contracts.jl`): Hedge `h2_hpa_from`. CVaR includes the HPA buyer CfD (`q` under SoP, notional $C$ under ToP). Physical EP conversion stays on pool hydrogen.

**Green offtaker (with endogenous EP capacity and CVaR):**

```text
min  γ × [ F_cap × cap_EP_y + Σ_y P[y] × loss_G[y] ]
   + (1−γ) × CVaR_G
   + (ρ_H2/2)×Σ W×(−h2_in − ḡ_H2)²
   + (ρ_H2GC/2)×Σ W×(−gc_h2 − ḡ_H2GC)²
   + (ρ_EP/2)×Σ W×(ep − ḡ_EP)²

where loss_G[y] = Σ_{h,d} W × ( λ_H2×h2_in + λ_H2GC×gc_h2 + proc×ep − λ_EP×ep )
      cap_EP_y is scalar (non-anticipative); ep and purchases vary by scenario y
```

These templates are implemented in the `build_*_agent.jl` files as follows:

- All **price-dependent terms** (e.g. $\lambda_{\mathrm{elec}}\,g$ with code variable `g`, $\lambda_{\mathrm{H2}}\,q_{\mathrm{in}}$ with `h2_in`) are expressed via JuMP `@expression` blocks whose coefficients are updated each ADMM iteration.
- The **capacity-investment linkage** uses **scalar** variables (e.g. `cap_VRES`, `inv_VRES`): one installed-capacity decision before operations, with `cap = cap_initial + inv`. The same `cap` binds generation in **every** weather scenario `y ∈ JY` via `g[h,d,y] ≤ AF[h,d,y] × cap`. Index `JY` indexes **parallel weather scenarios**, not sequential calendar investment years.
- For risk-averse agents, the **loss-per-scenario** expressions `loss_VRES[y]`, `loss_H2[y]`, `loss_G[y]` are recomputed in every ADMM iteration with the iteration-specific prices, so that CVaR always measures risk against the latest price trajectory. Fixed CAPEX enters once as `F_cap × cap` in both the expected and CVaR terms.

### 5.2 Key Constraints

| Constraint | Equation | Scope | Rationale |
|---|---|---|---|
| VRES capacity | `g ≤ AF × cap_VRES` | Per (h,d,y) | Generation limited by resource availability × **endogenous** installed capacity |
| Conventional staging (legacy) | `g = Σ_s g_s`, `0 ≤ g_s ≤ cap_s` | Per (h,d,y) | Piecewise thermal stack with increasing stage costs |
| Conventional flat plant | `0 ≤ g ≤ Capacity` | Per (h,d,y) | Single-technology dispatch; SRMC constant within scenario |
| Consumer load | `d ≤ PeakLoad × load_profile` | Per (h,d,y) | Maximum consumption bound |
| H₂ conversion | `h2_out = η × e_in` | Per (h,d,y) | Stoichiometric mass/energy balance |
| H₂ GC physical limit | `gc_h2 ≤ h2_out` | Per (h,d,y) | Cannot certify more than produced |
| Green-backing (annual) | `Σ W×gc_elec ≥ (1/η)×Σ W×gc_h2` | Per year | Temporal flexibility in GC procurement |
| Green offtaker stoichiometry | `ep = α × h2_in` | Per (h,d,y) | No H₂ waste; α = MWh_EP per MWh_H₂ (default 0.75) |
| Green GC mandate | `Σ W×gc_h2 ≥ γ_GC × Σ W×h2_in` | Per year | ≥ 42% of **H₂ intake** must be green-certified |
| Grey GC mandate | `Σ W×gc_h2 ≥ γ_GC × (1/γ_NH3) × Σ W×ep` | Per year | ≥ 42% of **implied H₂ feedstock** (`ep/γ_NH3`) must be green-certified |

### 5.3 Social Planner Objective

The social planner maximises **risk-adjusted social welfare** with a **single** social CVaR:


$$
\max \; \gamma \sum_y P_y\,\mathrm{sw}^{\mathrm{aux}}_y \;-\; (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}
$$


where `sw_aux[y]` is an epigraph proxy for aggregate social welfare per weather scenario (see §7.4 for why), and `CVaR_social` penalises tail risk across scenarios. When `γ=1` (risk-neutral), the CVaR term vanishes and the planner reduces to expected welfare maximisation.

#### Per-agent welfare contributions

Each `add_*_to_planner!` function returns a `Dict{Int, Any}` of per-year welfare expressions (no per-agent CVaR — CVaR is applied once to the aggregate). Revenue/expenditure terms cancel out in the aggregate (they are transfers between agents). The per-agent welfare terms are:

- **Consumers**: `U(d) = A×d − (B/2)×d²` (quadratic utility)
- **Generators**: `−MC × g` (negative production cost) minus fixed CAPEX on endogenous capacity for VRES
- **Electrolyzer**: `−op_cost × h2_out` minus fixed CAPEX on endogenous H₂ capacity
- **Green offtaker**: `−processing_cost × ep` minus fixed CAPEX on endogenous EP capacity
- **Other offtakers/importer**: `−processing_cost × ep` or `−import_cost × ep`

#### Social welfare aggregation


$$
\mathrm{socialWelfare}_y = \sum_i \mathrm{welfare}_{i,y}
$$


Market-clearing constraints enforce supply = demand. The single social CVaR applies to the full aggregate welfare (including consumer utility), ensuring the risk-averse planner accounts for all welfare components when assessing tail risk.

### 5.4 Risk Aversion and Risk-Neutral Consistency

This section gives the **technical** risk-aversion formulation and SP–ME equivalence conditions. Economic interpretation and literature labels are in **§4.5–§4.8**.

#### 5.4.1 Agent-level vs system-level risk

- In the **ADMM (market exposure) case**:
  - A subset of agents (VRES, electrolyzer, green offtaker) can be risk-averse with their own parameters $(\gamma_i,\beta_i)$.
  - Each such agent minimises a **private risk-adjusted loss**:

$$
\gamma_i\,\mathbb{E}[\ell_i] + (1-\gamma_i)\,\mathrm{CVaR}_{i}(\ell_i),
$$

subject to its own technological constraints and the ADMM penalties.
  - Risk is therefore **heterogeneous and decentralised**: different agents may have different attitudes to risk; financial transfers between agents do not directly enter the risk measure.

- In the **social planner case**:
  - There is a **single** system-wide risk parameter $\gamma$ and confidence level $\beta$.
  - The planner maximises a **single risk-adjusted social welfare**:

$$
\gamma\,\mathbb{E}\bigl[\mathrm{SW}\bigr] - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}(-\mathrm{SW}),
$$

where $\mathrm{SW}$ is aggregate welfare (including consumer utility and production/investment costs).
  - Risk is therefore **centralised**: society as a whole is risk-averse with respect to aggregate welfare, rather than each agent separately.

These two formulations represent different normative assumptions about **who bears risk** and **how it is shared**. The ADMM run with per-agent CVaR corresponds to a market in which agents individually care about their own tail losses; the social planner corresponds to a benevolent regulator who cares about systemic tail outcomes.

#### 5.4.2 Risk-neutral benchmark and equivalence

When both formulations are made **risk-neutral**, they collapse to the same underlying convex optimisation problem:

- In ADMM:
  - Set $\gamma_i = 1$ for all agents that can be risk-averse (VRES, electrolyzer, green offtaker).
  - This eliminates all per-agent CVaR terms from their objectives.

- In the social planner:
  - Set the planner-wide risk weight $\gamma = 1$.
  - This eliminates $\mathrm{CVaR}^{\mathrm{social}}$ from the planner’s objective, so the model becomes a quadratic (but not quadratically constrained) welfare maximisation with standard consumer surplus and producer surplus terms.

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

#### 5.4.3 Risk institutions and prices

Economic case names, literature mapping (d'Aertrycke et al.), and price interpretation when $\gamma < 1$ are in **§4.8**. Technical CVaR consistency conditions remain in §5.4.1–§5.4.2.

---

## 6. ADMM Algorithm

ADMM is the **numerical engine** for `market_exposure.jl` and `me_pap.jl`, `me_top.jl`, or `me_sop.jl`. It does **not** define the economic equilibrium (that is the competitive MCP in §4); it **finds** decentralised prices and quantities that satisfy that equilibrium’s KKT conditions. This section is the main reference for **why** ADMM is used, **how** it maps to Boyd et al. (2011), and **what** each residual, penalty, and warm-start component does in code.

**Implementation map:** `Source/ADMM.jl` (main loop), `ADMM_subroutine.jl` (per-agent step), `update_rho.jl` / `update_rho_contracts.jl` (ρ adaptation), `define_results.jl` (warm-start from SP), `ADMM_contracts.jl` (contracts extension).

### 6.0 Why ADMM, alternatives, and literature

#### 6.0.1 What we are solving numerically

At $\gamma=1$, the decentralised competitive equilibrium is the solution of a **large coupled convex program**: each agent minimises private cost minus revenue subject to technology constraints; **market-clearing** links agents through shared prices $\lambda_k$ (dual variables of balance constraints). With CVaR ($\gamma<1$), agent subproblems stay convex (linear CVaR reformulation); the coupled system remains a convex equilibrium problem, but the **risk institution** differs from the social planner (§4.8).

Key structural features that drive the solver choice:

| Feature | Implication |
|---|---|
| **Seven agent types**, many instances | Natural **agent decomposition** — each subproblem is a medium QP/QCP |
| **Five coupled spot markets** (+ contracts) | Multiple consensus constraints; prices must clear jointly |
| **3D quantities** $(h,d,y)$ | Large tensors; residuals are **L2 norms** over all slots |
| **Endogenous capacity** | Extra equality-split ADMM block per capacity owner (§6.4) |
| **CVaR agents** | Loss expressions depend on current $\lambda$ → constraints rebuilt each iteration |

#### 6.0.2 Alternative approaches

| Approach | Role in literature | Why not primary here |
|---|---|---|
| **Monolithic welfare QP/QCP** (social planner) | Centralised benchmark; KKT = competitive prices | Solves the **planner** problem, not the **decentralised** institution; one huge model; no per-agent privacy/modularity |
| **MCP / PATH** (mixed complementarity) | Standard for spatial/nodal equilibrium (Gabriel et al.) | Hoschle et al. (2018) report PATH **failing** on larger **risk-averse** equilibrium instances; hard to maintain when agents/CVaR blocks change each iteration |
| **Full KKT NLP** (single IPOPT on coupled system) | Theoretically equivalent | Same scale and modularity issues; fragile dual extraction on large QCPs (why SP uses IPOPT alone, §7.2) |
| **Nash / Cournot solvers** | Strategic games | Wrong economics — this model is **price-taking** MCP, not oligopoly (§4.3) |
| **Auction / tâtonnement** | Pedagogical price adjustment | No convergence guarantee for coupled multi-market capacity models |

The **social planner** (`social_planner.jl`) remains indispensable as a **benchmark** (welfare theorem at $\gamma=1$, complete risk trading at $\gamma<1$), but it answers a different **institutional** question than ADMM.

#### 6.0.3 Why ADMM was chosen

1. **Decomposition by agent.** Each iteration solves independent convex subproblems (Gurobi). Adding or modifying an agent type touches one `build_*_agent.jl` / `solve_*_agent.jl` pair, not a monolithic KKT system.

2. **Proven in energy equilibrium.** Hoschle et al. (2018) use ADMM for **risk-averse capacity market equilibrium** with CVaR; d’Aertrycke et al. (2018) frame incomplete vs complete risk trading in the same equilibrium class. This project extends that pattern to **multi-commodity hydrogen/ammonia markets** with investment.

3. **Prices are dual variables.** Commodity prices $\lambda_k$ are updated by **dual ascent** on market imbalances — the standard ADMM interpretation of Lagrange multipliers for balance constraints (§6.10).

4. **Diagnostics.** Primal/dual residuals and `ADMM_Convergence.csv` give **market-by-market** convergence evidence — essential when five markets and capacity splits can converge at different rates.

5. **Warm-start from SP.** The centralised solution provides $\lambda$, quantities, and capacities for iteration 0 (§6.6), cutting iterations from thousands to hundreds in typical NL runs.

#### 6.0.4 ADMM is the solver, not the economics

ADMM iterations, consensus targets $\bar{g}$, and penalties $\rho$ are **algorithmic artefacts**. At convergence (residuals below tolerance), they vanish from the economics: agents trade at $\lambda^{\ast}$, markets clear, and ADMM penalties are zero. The equilibrium **definition** is in §4–§5; ADMM is one reliable way to **compute** it. The social planner is the independent check that, at $\gamma=1$, decentralised and centralised optima coincide.

### 6.1 Iteration Structure

`ADMM.jl` (and `ADMM_contracts.jl` for the contract cases) repeats the steps below until every market **and** every capacity split is within tolerance (§6.5), or `max_iter` is reached. The step numbers are the reference used throughout §6; the deeper details of each step live in the subsections it points to, so nothing is re-derived twice.

1. **Agent solves (x-update).** For each agent, `ADMM_subroutine!` refreshes the ADMM parameters on the agent's JuMP model — consensus target $\bar g$ (§6.2), price $\lambda$, penalty $\rho$, and the capacity-split terms (§6.4) — then rebuilds the objective and solves the convex subproblem with Gurobi. CVaR agents (VRES, electrolyzer, green offtaker) first rebuild their loss expressions and CVaR constraints against the current $\lambda$. The solved net positions are recorded.

2. **Market imbalances.** For each market, sum all participants' net positions; for EP subtract fixed demand $D_{\mathrm{EP}}$. With supply positive and demand negative (§6.8), a positive imbalance means excess supply and a negative one excess demand.

3. **Residuals.** For each market compute the **primal residual** — the L2 norm of the imbalance, i.e. distance from clearing — and the **dual residual** — the L2 norm of $\rho\times$(change in each agent's consensus deviation), i.e. how much positions are still moving. Each capacity-owning agent uses the analogous per-agent residuals (§6.4.2). Formal definitions and the stopping test are in §6.5. On iteration 1 the dual residual is $\infty$ (no previous iterate), so at least two iterations always run.

4. **Price update (dual ascent).** Each spot-market price moves against its imbalance,
$$
\lambda_k \leftarrow \lambda_k - \eta_k\,\rho_k\,\mathrm{imbalance}_k,
$$
so excess supply lowers the price and excess demand raises it. The factor $\eta_k \in [0.25, 1]$ is **scale-aware damping**: it is $1$ while $\max(r^{\mathrm{pri}}, r^{\mathrm{dual}}) \ge 1.5\times\max(\varepsilon_{\mathrm{pri}},\varepsilon_{\mathrm{dual}})$ (comfortably above the market's Boyd tolerance, §6.5) and shrinks smoothly toward $0.25$ as that ratio falls below 1.5, which stops thin markets from oscillating near the stopping region and keeps step sizes horizon-robust. It is further multiplied by a per-market factor `η_scale` adapted online (§6.9). The H₂-GC price is then projected onto $\lambda_{H2\_GC}\ge 0$ (§6.9).

   Capacity multipliers are updated **earlier in the same iteration** — after primal residuals are computed and before dual residuals — via the analogous ascent $\lambda^{\mathrm{cap}}_m \leftarrow \lambda^{\mathrm{cap}}_m + \rho_m\,(x_m - z_m)$ (§6.4). They are **not** updated in the same code block as the spot-market $\lambda$.

5. **Adaptive penalty.** `update_rho!` adjusts every market $\rho_k$ and every per-agent capacity $\rho_m$ by residual balancing (§6.3).

6. **Convergence test.** Convergence is declared once **all five** spot markets **and every** capacity-owning agent satisfy both their primal and dual Boyd tolerances (§6.5). Capacity is tested per agent, never on the aggregate.

### 6.2 Consensus Formula (Sharing ADMM)

The consensus target for agent $i$ in a market with $n$ participants is built from the previous iterate's quantities and the market imbalance:

$$
\bar{g}_i^k = q_i^{k-1} - \frac{1}{n+1}\,\mathrm{imbalance}^{k-1}
$$

where $\mathrm{imbalance} = \sum_j q_j$ for markets without fixed demand, and $\mathrm{imbalance} = \sum_j q_j - D_{\mathrm{EP}}$ for the EP market (§6.1 step 2). Writing $\bar g_i = q_i - \tfrac{1}{n+1}\sum_j q_j$ is therefore correct for electricity / H₂ / GC markets, but **not** for EP — there the fixed demand must sit inside the imbalance.

(In code: `ḡ_i^k`, `q_i`, etc.) The `(n+1)` denominator comes from the sharing ADMM formulation, which introduces one "market copy" alongside the `n` agent copies. This distributes the imbalance correction equally.

#### 6.2.1 Algorithm variables — what λ, ρ, and ḡ are

| Symbol (code) | Name | Economic / algorithmic role | Updated when |
|---|---|---|---|
| `λ_k` (`results["λ"]`) | **Market price** (dual) | Shadow price of market $k$ balance; €/MWh at convergence | After each iteration: dual ascent on imbalance (§6.1 step 4) |
| `ρ_k` (`ADMM_state["ρ"]`) | **Penalty weight** | Augmented-Lagrangian curvature on $\|g_i^k - \bar{g}_i^k\|^2$; **not** an economic parameter | After residuals: `update_rho!` (§6.3) |
| `ḡ_i^k` (`g_bar_*`) | **Consensus target** | Per-agent quantity target in sharing ADMM; pulls $g_i$ toward market-clearing | Before each agent solve: `ADMM_subroutine!` |
| `imbalance_k` | **Primal residual (unnormalised)** | $\sum_i g_i^k$ (+ fixed demand for EP); should be **0** at equilibrium | After all agents solve |
| `r_k`, `s_k` | **Primal / dual residuals** | Scalar L2 norms for stopping test (§6.5) | After imbalance and dual-residual pass |

**Intuition:** Agents optimise against **fixed** $\lambda^{k}$ and $\bar{g}_{i}^{k}$ (price-taking). The coordinator raises $\lambda$ when there is excess demand (negative imbalance) and lowers it when there is excess supply. $\rho$ controls how hard agents are pulled toward $\bar{g}$; if primal residuals stall while dual residuals are tiny, $\rho$ is **increased** so consensus is enforced faster.

### 6.3 Adaptive Penalty (ρ)

`update_rho.jl` and `update_rho_contracts.jl` implement **Boyd-style residual balancing** (Boyd et al. 2011, §3.4.1): adapt $\rho$ so primal and dual residuals decrease at similar rates.

For each market (and each capacity agent, see §6.4), **after** residuals are computed:

- if `r_p > μ r_d`, **increase** `ρ` (consensus too loose — penalise deviations harder);
- if `r_d > μ r_p`, **decrease** `ρ` (dual changing too fast — soften penalty);
- otherwise **keep** `ρ` unchanged (balanced progress).

The hysteresis threshold is **`μ = 1.2`** (`rho_balance_threshold` in `ADMM_state`). Without $\mu>1$, $\rho$ would chatter every iteration.

**Why this controller is “smart” for this model:**

1. **Per-market independence.** Electricity, H₂, GC, and EP can have very different residual scales. Each market’s $\rho_k$ adapts to its own $(r_p, r_d)$ — a single global penalty would over-penalise thin markets or under-penalise elec.

2. **Asymmetric step sizes (τ).** Core markets (`elec`, `elec_GC`) use $\tau=1.05$; tightly coupled `H2` and `EP` use $\tau=1.01$ to avoid limit cycles when electrolyzer capacity couples four prices at once.

3. **ρ bounds.** `ρ_max` prevents exploding penalties (ill-conditioned agent QPs); `ρ_min` prevents $\rho\to 0$ while primal residual is still large.

4. **Same law for capacity.** Per-agent `ρ_cap[m]` uses the identical ratio test (§6.4.4) so VRES, electrolyzer, and green offtaker capacity splits do not share one compromised penalty.

`ρ` initial values come from `data.yaml` (`rho_initial` per market, `rho_cap_initial` for capacity). They are **numerical knobs**, not calibrated economic parameters. The λ-step damping, the H₂-GC price projection, and the other numerical stabilisers in `ADMM.jl` (which sit *outside* `update_rho!`) are collected in §6.9.

#### 6.3.1 Per-market parameters

| Market | Increase factor | Decrease factor | ρ_max | Reasoning |
|---|---|---|---|---|
| `elec`, `elec_GC` | 1.05 | 1/1.05 | 5,000 | Large-volume core markets; can tolerate moderately faster adaptation. |
| `H2`, `EP` | 1.01 | 1/1.01 | 100 | More kink-sensitive due to coupling and capacity effects; slower updates reduce oscillation risk. |
| `H2_GC` | 1.05 | 1/1.05 | 100 | Thin but hourly market; moderate adaptation with conservative cap. |
| `ppa`, `ppa_cap` | 1.05 | 1/1.05 | 500 | Thin bilateral pool; conservative but responsive. |
| `hpa`, `hpa_cap` | 1.05 | 1/1.05 | 500 | Same logic as PPA for hydrogen contracts. |

Capacity consensus is a per-agent equality split (§6.4), so each capacity-owning agent has its own `ρ_cap[m]` update with the same residual-balancing rule.

#### 6.3.2 Pseudo-code

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

### 6.4 Capacity ADMM (Equality Split per Agent)

Capacity consensus is treated as a **textbook ADMM equality split** at the agent level, not as a soft penalty against a derived target. This subsection gives the formal model, residual definitions, and the rationale for each design choice.

#### 6.4.1 Formal model

For every capacity-owning agent `m ∈ {VRES, GreenProducer, GreenOfftaker}` we introduce **one scalar** equality split (not one per scenario year — capacity itself is non-anticipative):

- **Primal** `x_m` — the agent's own capacity variable (`cap_VRES`, `cap_H2_y`, or `cap_EP_y`);
- **Auxiliary** `z_m` — a scalar capacity target equal to the **peak** of the agent's latest realised flow profile across the whole $(h,d,y)$ horizon (fallback: ADMM flow target if no history yet). For VRES, $z = \max_{h,d,y} g_{\mathrm{elec}}/\mathrm{AF}$; for electrolyzers, $z = \max_{h,d,y} h2_{\mathrm{out}}$; for green offtakers, $z = \max_{h,d,y} ep$. In the contracts case, VRES and electrolyzer targets use pool + contract flows. An optional EMA with weight `cap_z_relax` (code default 1.0 = no smoothing) blends $z$ against its previous value.
- **Dual** `λ_m` — Lagrange multiplier for the equality `x_m = z_m`;
- **Per-agent penalty** `ρ_m` — scalar weight, one per agent.

The agent solves, each ADMM iteration:

```
x_m^k = argmin_{x ≥ 0}  f_m(x, ...)
                       + λ_m^{k-1} · (x - z_m^k)
                       + (ρ_m^{k-1}/2) · (x - z_m^k)²
```

where `f_m` is the agent's own economic loss (operational + CAPEX − revenue, plus CVaR / risk and other ADMM market penalties). After all agents solve we perform the **dual ascent** (this happens after primal residuals are computed and before dual residuals — see §6.1):

```
λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k)
```

This is the standard equality-split ADMM update (Boyd et al. 2011, §3.1). The Capacity_Consensus.csv file still writes one row per `(iter, agent, jy)` for joinability against scenario-indexed results; the underlying decision is the same scalar for every `jy`.

#### 6.4.2 Residuals

Per-agent residuals follow the Boyd definition for the `x = z` split:

```
Primal:  r_m^k = | x_m^k - z_m^k |                 (scalar absolute residual)
Dual:    s_m^k = | ρ_m^{k-1} · (z_m^k - z_m^{k-1}) |   (Δz, not Δx)
```

The dual residual uses the change in the **auxiliary** `z` (not the change in the primal `x`). This is the ADMM-correct definition: if `x` has frozen but `z` is still drifting, a Δx-based residual would falsely declare convergence; Δz captures the true ADMM dual progress (Boyd et al. 2011, Eq. 3.12).

For diagnostics and the one-line summary, the aggregate residuals reported in `ADMM_Convergence.csv` as `cap_primal` / `cap_dual` are the L2 norms over agents:

```
r_cap^k = sqrt(Σ_m r_m^k²),    s_cap^k = sqrt(Σ_m s_m^k²)
```

#### 6.4.3 Stopping rule

Convergence is checked **per agent**, not on the aggregate. For each capacity-owning agent the Boyd test uses a **dedicated scalar MW tolerance** `ε_cap` (`ADMM.epsilon_cap`), not the flow-market `ε_abs`:

```
ε_pri_m  = ε_cap + ε_rel · ResidualScale_Primal_m
ε_dual_m = ε_cap + ε_rel · ResidualScale_Dual_m
```

Capacity is one decision, so there is **no** `sqrt(n_slots)` factor (unlike flow markets, §6.5). Reusing the per-slot flow `ε_abs` on a scalar split would make capacity ~50× tighter in MW than the market L2 test — that is a dimensional mismatch, not “the same accuracy.” In the contracts case the right-hand side is further multiplied by `cap_tol_relax` (§6.7). Capacity is converged iff `r_m ≤ ε_pri_m` and `s_m ≤ ε_dual_m` for **every** `m`. *Why per-agent and not aggregate*: averaging residuals across agents can hide a single laggard whose split is still far from feasibility; an aggregate test would declare convergence even when one agent type (e.g. a strongly binding electrolyzer) has not satisfied the equality. The per-agent test is direction-correct: capacity is "done" when every agent's split is satisfied.

The optional knob `cap_tol_relax` multiplies the right-hand side of the per-agent **physical investment capacity** test in the **contracts** case only (shipped `data.yaml` value: **10**; code fallback if the key is absent: **100**). Plain ME leaves it at 1. See §6.7. Bilateral contract capacity $C$ (PPA/HPA cap consensus) uses `ε_cap` without this extra multiplier.

#### 6.4.4 Per-agent ρ controller

Each capacity-owning agent's penalty $\rho_m$ follows the **same** residual-balancing rule as the flow markets (§6.3), applied to that agent's own $(r_m, s_m)$: increase when the primal residual dominates, decrease when the dual dominates, otherwise hold. Only the bounds differ — increase factor 1.05, decrease factor 1/1.05, $\rho_{\max}=30$, $\rho_{\min}=0.10$, one controller **per agent**. The three knobs below are **code defaults** (they are not present in the shipped `data.yaml`; override them there only if you need to):

```yaml
ADMM:
  rho_cap_initial: 0.1      # present in data.yaml
  rho_cap_inc_factor: 1.05  # code default
  rho_cap_max: 30.0         # code default
```

Using a separate controller per agent (rather than one global $\rho_{\mathrm{cap}}$) is essential because the capacity owners have very different residual scales; the justification is §6.4.6.

#### 6.4.5 Why the equality split

The capacity block uses the augmented-Lagrangian equality split:

```
L_cap = λ_cap · (x - z) + (ρ_cap/2) · (x - z)^2
```

This structure is used because:

1. the linear dual term provides first-order correction toward `x = z`;
2. the quadratic term provides curvature and numerical regularization;
3. both terms are in currency units and vanish at consensus.

With dual ascent `λ <- λ + ρ (x-z)`, the split follows textbook ADMM dynamics for equality constraints.

#### 6.4.6 Why per-agent ρ (and not a single global ρ_cap)

A single global `ρ_cap` forces a compromise across very heterogeneous agent types:

- **VRES**: capacity grows by tens of MW per step; CAPEX is moderate; the binding constraint is `cap ≥ peak(g)/AF`.
- **Electrolyzer**: capacity is tightly coupled to four markets simultaneously (elec, elec_GC, H2, H2_GC); CAPEX is high; the binding constraint is `cap ≥ peak(h2_out)`.
- **Green offtaker**: capacity is decoupled from elec but tied to EP flow; CAPEX is small relative to operational margin.

When the controller picks a single `ρ_cap` that suits one of these, the others sit either in the dead band (no progress) or over the kink (limit cycles). Per-agent `ρ_m` removes this compromise; each agent's controller specialises to its own residual scale.

#### 6.4.7 Units check (and why penalties don't bias the social-planner equivalence)

The economic loss `f_m` is in € (currency); `λ · (x - z)` has units `[€/MW] · [MW] = €`; `(ρ/2) · (x - z)²` has units `[€/MW²] · [MW²] = €`. All three terms add cleanly. Because the ADMM penalty and dual terms vanish exactly at consensus (`x = z`), they have no effect on the centralised social-planner optimum: the planner does not solve a per-agent subproblem, hence has no `λ_cap` / `ρ_cap` / `z_cap` parameters (`add_*_to_planner!` functions never touch these). At γ = 1 (risk-neutral) the ADMM equilibrium therefore converges to the same primal/dual solution as the planner by the first welfare theorem — see §6.6 for the SP warm-start that we rely on for fast convergence.

#### 6.4.8 Iteration order in the main loop

Capacity is woven into the full §6.1 iteration; it is **not** a self-contained 7-step block that ends with its own convergence test. The capacity-relevant order inside iteration `k` is:

1. **Derive `z^k` and set parameters** (inside `ADMM_subroutine!`, before the agent solve): compute `z_m^k` from realized flow histories (fallback to ADMM targets when history is empty), optionally under-relax
   `z^k ← α·z_raw^k + (1-α)·z^{k-1}` with `α = cap_z_relax` (code default 1.0 = off), re-project the model-feasible capacity floor, then write `:z_cap = z_m^k`, `:λ_cap = λ_m^{k-1}`, `:ρ_cap = ρ_m^{k-1}` onto the agent model.
2. **Agent solves**: produces `x_m^k`.
3. **Market imbalances**, then **primal residuals** (flow markets and per-agent capacity `r_m^k = |x_m^k − z_m^k|`).
4. **Capacity dual ascent**: `λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k)`, pushed to history — this happens **before** dual residuals and **before** the spot-market λ update.
5. **Dual residuals** (flow markets and per-agent capacity `s_m^k = |ρ_m^{k-1} · (z_m^k − z_m^{k-1})|`).
6. **Spot-market λ update**, then **`update_rho!`** (which also updates every per-agent `ρ_m`).
7. **Single convergence test** at the end of the iteration: all five spot markets **and** every capacity agent (§6.5). Capacity does not declare convergence on its own.

This ordering is identical for `market_exposure` and the ME contract cases (`me_pap`, `me_top`, `me_sop`); only the `z` derivation differs (the contracts case adds the PPA / HPA flow contributions when computing the peak of `g_bar + g_bar_ppa`, etc.).

**Why this choice (`z` under-relaxation):**

In tightly-coupled runs, raw `z` targets can jump sharply when flow consensus oscillates across markets. Because the capacity dual residual uses `Δz`, these jumps can produce very large `s_m` and trigger controller overreaction even when `x` is moving in the right direction. Under-relaxation damps target motion, reducing artificial dual spikes and improving monotonic progress toward the split fixed point.

`z` projection enforces feasibility against the model structure: minimum installed-capacity floor implied by nonnegative investment variables (scalar `cap` per agent).

### 6.5 Convergence Tolerances (Boyd-style)

Instead of a single scalar tolerance, the implementation follows the **absolute + relative** stopping criteria proposed by Boyd et al. (2011) for ADMM. Three knobs are stored in `data.yaml` → `ADMM`:

- Flow-market absolute tolerance `ε_abs` (MW **per slot**), from `epsilon_abs` if present, otherwise `epsilon` (ME) or `epsilon_contracts` (ME+C).
- Capacity absolute tolerance `ε_cap` (MW, **scalar**), from `epsilon_cap` (default `5.0`).
- Relative tolerance `ε_rel` (dimensionless), from `epsilon_rel` if present, otherwise `0.0`.

Let:

- `n = nTimesteps × nReprDays × nYears` be the number of time slots in the horizon.
- `Scale_primal[k]` and `Scale_dual[k]` be fixed reference magnitudes for the primal and dual residuals of market `k`, captured from the first non-zero residual observed for that market (stored in `ADMM["ResidualScale"]`).

Then the per-market primal and dual tolerances are:

```
ε_pri_k  = ε_abs * sqrt(n) + ε_rel * Scale_primal[k]
ε_dual_k = ε_abs * sqrt(n) + ε_rel * Scale_dual[k]
```

The stopping rule is:

- **Primal (flow)**: for every spot market `k`, the L2 norm of the imbalance vector must satisfy `‖r_k‖₂ ≤ ε_pri_k`.
- **Dual (flow)**: for every spot market `k`, the L2 norm of the change in consensus deviation must satisfy `‖s_k‖₂ ≤ ε_dual_k`.
- **Capacity**: for every capacity-owning agent `m`, both the primal and dual scalar residuals must satisfy the per-agent Boyd test of §6.4.3 (`ε_cap`, not `ε_abs · √n`).

**All five spot markets and every capacity-owning agent** must simultaneously satisfy their conditions for convergence to be declared (`within_tol(...) && within_tol_cap()` in `ADMM.jl`). Declaring "five markets done" while capacity is still drifting is not convergence.

This has three advantages over a single scalar `epsilon`:

1. **Scale awareness**: Markets with large typical flows (e.g. electricity) naturally get larger absolute L2 tolerances than thin markets (e.g. GC), while still using a common per-slot `ε_abs`.
2. **Robustness to refinement**: If the temporal resolution or the number of representative days changes (n increases), the `sqrt(n)` factor keeps the **per-slot** (RMS) accuracy comparable: `‖r‖₂ ≤ ε_abs √n` is equivalent to RMS imbalance ≤ `ε_abs`.
3. **Numerical realism**: Once residuals are small relative to the problem’s own scale, the criteria do not force the algorithm to chase tiny numerical oscillations; they recognise that the solution is “good enough” in the sense of Boyd et al.

#### Relative tolerance ε_rel

The optional `ε_rel` term adds a scale-relative component. When `ε_rel > 0`, markets with larger typical residual magnitudes get proportionally larger tolerances. Set `epsilon_rel: 0.01` in `data.yaml` to enable a 1% relative tolerance. The default is `0.0` (absolute tests only), so the numbers below are exact.

#### Chosen values (and why)

Default horizon: $n = 24 \times 8 \times 15 = 2{,}880$ slots, $\sqrt{n} = \sqrt{2880} \approx 53.67$. Shipped values:

| Knob | Value | What it controls | Effective test |
|------|-------|------------------|----------------|
| `epsilon` | **0.2** MW/slot | ME flow markets | L2 ≤ $0.2 \times 53.67 = \mathbf{10.73}$ MW |
| `epsilon_contracts` | **2.0** MW/slot | ME+C flow markets (PPA/HPA energy uses the same) | L2 ≤ $2.0 \times 53.67 = \mathbf{107.3}$ MW |
| `epsilon_cap` | **5.0** MW | Scalar $x=z$ split (physical cap; also PPA/HPA $C$) | $|x-z| \le \mathbf{5.0}$ MW (ME and contract $C$); $\times$ `cap_tol_relax` for ME+C physical cap |
| `cap_tol_relax` | **10** | Extra ME+C multiplier on physical investment only | ME+C physical cap ≤ $\mathbf{50}$ MW |

**1. Flow-market `epsilon = 0.2` (ME).**
Boyd's `sqrt(n)` rule means the per-slot accuracy is exactly `ε_abs`: RMS imbalance ≤ 0.2 MW. On the Dutch system this is negligible:

- vs peak load ~18,000 MW: $0.2 / 18000 = 0.0011\%$.
- Money-metric upper bound (treat 0.2 MW as a persistent hourly imbalance at a typical electricity price ~32 €/MWh): $0.2 \times 32 \times 8760 \approx 56$ k€/year, against $\mathbb{E}[\mathrm{SW}] \approx 49$ bn € — about **0.00011%** of welfare. The actual money-metric is smaller because the residual is an L2 budget across 2,880 slots, not a bias in every hour.
- The previous value `0.1` already cleared all five markets in the $\gamma=0.5$, $\beta=0.8$ ME run (L2 residuals ~1.6–3.1 vs a 5.37 MW bar). Doubling to 0.2 keeps the same economic interpretation and gives modest headroom without letting prices wander: a 0.2 MW RMS gap cannot move a 30 €/MWh electricity price in any economically reportable way.

**2. Flow-market `epsilon_contracts = 2.0` (ME+C).**
ME+C adds PPA/HPA energy tensors plus two scalar $C$ splits on top of the five spot markets, so residual floors sit higher than in plain ME. **2.0 MW RMS is Boyd's $\varepsilon_{\mathrm{rel}}\sim 10^{-4}$ applied to NL peak load** ($10^{-4}\times 18{,}000 = 1.8$ MW). That is **10×** the ME bar (`2.0 / 0.2`), still **7× tighter** than Boyd's looser $10^{-3}$ relative (18 MW RMS). Money-metric: $2.0 \times 32 \times 8760 \approx 560$ k€/year (**0.0011%** of 49 bn € welfare). The L2 bar of 107 MW looks large only if one forgets it is an Euclidean budget over 2,880 slots; the economically meaningful figure is the 2 MW RMS.

**3. Capacity `epsilon_cap = 5.0` MW.**
Capacity is a **scalar** (one MW number per agent), so Boyd's `sqrt(n)` factor does **not** apply. Reusing a 0.1–1 MW cap test against a flow L2 bar of tens of MW is a dimensional mismatch, not “the same accuracy.” Relative to SP capacities (wind 76.4 GW, solar 55.0 GW, electrolyzer 1.43 GW, green offtaker 1.07 GW):

- 5 MW is $5/76358 = 0.0065\%$ of wind and $0.009\%$ of solar — **tighter than Boyd $10^{-4}$** on VRES.
- 5 MW is $0.35\%$ of the electrolyzer and $0.47\%$ of the offtaker — between Boyd $10^{-3}$ and a few-tenths of a percent, and **one typical PEM stack** (5–20 MW). Planning models do not commission 1 MW of electrolysis.
- Money-metric at shipped annuities: 5 MW of wind × 18,500 €/MW-year = **92.5 k€/year**; 5 MW of electrolyzer × 262,000 €/MW_e-year ≈ **1.3 M€/year** — both **≪ 0.003%** of 49 bn € welfare.
- **Observed ADMM floor.** In the $\gamma=0.5$, $\beta=0.8$ ME run, all five spot markets were inside the old 5.37 MW bar from iteration 418 onward. Wind $|x-z|$ never fell to 0.1 MW: it reached a best of **0.17 MW at iter 476**, then oscillated in **0.23–0.67 MW** (`ρ_cap` frozen near 0.9). That band is a penalty-augmented limit cycle. A 1 MW bar sits above the ME floor but was **too tight for bilateral contract $C$** (the same scalar-split trap on a hedge that inherits PPA/HPA energy chatter). 5 MW sits above that floor with margin; chasing 1 MW of $C$ is asking the solver to kill a sub-stack consensus gap.

**4. Contracts physical-cap extra factor `cap_tol_relax = 10`.**
In ME+C, physical `z_cap` uses **pool flow only** (same as ME): hedges do not occupy a second slice of plant capacity. Effective bar: $10 \times 5.0 = \mathbf{50}$ MW = $0.065\%$ of the 76 GW wind fleet (Boyd $10^{-3}$ is 0.1%) and $3.5\%$ of the 1.43 GW electrolyzer. Bilateral hedge capacity $C$ does **not** get this extra factor; it uses `ε_cap = 5.0` MW like a scalar contract position.

**What we are not doing.** These values do not change the equilibrium (penalties vanish at $x=z$ and market imbalance 0). They only stop the loop from burning hundreds of iterations on sub-stack capacity chatter after prices and dispatch have already cleared. For a tighter numerical study, lower `epsilon` / `epsilon_cap`; do not raise `max_iter` in the hope that a 1 MW contract-$C$ split will eventually die.

#### Two epsilon values: `epsilon` vs `epsilon_contracts`

The **ME contract cases** (`me_pap`, `me_top`, `me_sop`) have more coupled markets (standard flows + contract energy + contract capacity + capacity consensus) and stronger interdependence (VRES splits pool vs contract; electrolyzer does the same). As a result, convergence is slower and residuals tend to be larger than in `market_exposure`. To avoid running to `max_iter` without declaring convergence when results are already good enough, the contracts case uses a separate tolerance:

- **`epsilon`** — Used by `market_exposure` (shipped **0.2** MW/slot).
- **`epsilon_contracts`** — Used by `me_pap.jl` / `me_top.jl` / `me_sop.jl` when set in `data.yaml` (shipped **2.0** MW/slot, 10× ME). If not set, the contracts case falls back to `epsilon`.
- **`epsilon_cap`** — Shared scalar MW bar for capacity splits in both ME and ME+C (shipped **5.0** MW).

Both cases use the same convergence logic; only the flow-market `ε_abs` and the contracts-only `cap_tol_relax` multiplier differ. **`cap_tol_relax`** relaxes **physical investment capacity** consensus only (§6.4, §6.7); bilateral contract $C$ uses `ε_cap` without that multiplier.

### 6.6 Warm-start from Social Planner

Warm-starting ADMM from the social planner solution is **critical** for fast, reliable convergence. Implementation: `define_results.jl` (and `define_results_contracts.jl`) reads `social_planner_results/` when present. Run **`social_planner.jl` first**, then **`market_exposure.jl`**.

#### 6.6.1 Three warm-start components

| # | Component | Source file | What it fixes |
|---|---|---|---|
| **1** | **Prices $\lambda$** | `social_planner_results/Market_Prices.csv` | Without SP prices, ADMM uses scalar `initial_price` from `data.yaml` — far from peak/off-peak structure → large early imbalances and slow λ search |
| **2** | **Primal quantities** | `social_planner_results/SP_Primal_Quantities.csv` | Pre-populates `results["g"]`, `h2`, etc. so iteration 1 has **previous quantities = SP** → consensus targets $\bar{g} \approx$ SP → **near-zero imbalance on iteration 1** when λ also matches SP |
| **3** | **Capacity** | `social_planner_results/SP_Capacities.csv` | `set_start_value` on `cap_*` variables; seeds capacity ADMM auxiliaries `z_cap` so $x_{\mathrm{cap}} \approx z_{\mathrm{cap}} \approx$ SP from the start |

#### 6.6.2 Why each component matters

**Prices only (no primal warm-start):** Agents still solve with $\bar{g} = 0$ on iteration 1 (empty quantity history → zero consensus target). They are penalised toward zero net positions → wrong dispatch, huge imbalances, many iterations to recover.

**Primal without capacity:** Flow consensus may clear while `z_cap` is derived from zero flows → capacity penalty pulls investment toward **zero MW** — exactly wrong for a model where VRES/H₂/EP capacity is endogenous.

**Full warm-start:** First ADMM iteration approximates **“each agent re-optimises at SP prices with SP quantities as targets”** — close to the KKT point. Typical effect: convergence in **hundreds** of iterations instead of thousands, and tighter final residuals at a given `epsilon`.

#### 6.6.3 Operational notes

- Primal warm-start requires **matching horizon**: `nTimesteps × nReprDays × nYears` rows in CSV must equal ADMM shape. With the default 15-scenario grid that is `24 × 8 × 15 = 2880` rows, so the SP results must come from a run over the **same** `Scenarios` block (otherwise primal warm-start is skipped with a warning). One-year SP **price** files are automatically replicated across years; one-year **primal** files are **not** — they are rejected. Primal warm-start is also gated on a successful price warm-start (`sp_loaded == true` in `define_results.jl`): a primal CSV alone is not enough.
- SP prices are loaded **as-is** (no H2_GC clamp at load); negative H2_GC at SP optimum would be inconsistent with equilibrium — the **floor** is applied after each λ update in `ADMM.jl`.
- Console message when all three load: `ADMM warm-start: λ from SP prices, primal quantities from SP, capacity seeds for N agents`.
- If `social_planner_results/` is missing, ADMM still runs from `initial_price` scalars — valid, but use for debugging only when benchmarking against SP.

### 6.7 Contract Pools ADMM (me_pap / me_top / me_sop)

The contracts loop (`ADMM_contracts.jl`) **extends** the full standard ME iteration of §6.1 — it does not replace it. Shared-cap fixing happens **inside** each `ADMM_subroutine_contracts!` call. Spot-market λ updates precede contract λ updates; `update_rho_contracts!` is last. **All three entry points** use the same contract ADMM; only HPA settlement (ToP vs SoP) differs. `me_pap.jl` aliases SoP.

**Per iteration (contract additions on top of §6.1):**

| Step | What happens |
|------|----------------|
| (base) | Full §6.1 iteration for the five spot markets and physical capacity |
| +1 | Inside each agent subroutine: `apply_shared_contract_caps!` — fix `ppa_cap` / `hpa_cap` to `SharedCap` |
| +2 | Contract **energy** imbalances: `g_ppa` vs `g_ppa_from`; `h2_hpa` vs `h2_hpa_from` |
| +3 | `update_shared_contract_capacity!` — bargaining update for $C$; `record_shared_cap_residuals!` |
| +4 | $\lambda^{\mathrm{contract}}$ dual ascent (same logic as spot markets, with η damping) |
| +5 | `update_rho_contracts!` — adapt $\rho$ on energy and cap-alignment residuals (after the spot-market ρ update) |

**Markets tracked for convergence** (all must clear):

- The five **spot** markets and every **physical capacity** agent (§6.5).
- **PPA/HPA energy** (`ppa`, `hpa`): 3D imbalances; $\lambda^{\mathrm{ppa}}_i$, $\lambda^{\mathrm{hpa}}_j$ prices; Boyd residuals like spot markets.
- **Shared cap** (`ppa_cap_*`, `hpa_cap_*` in `ADMM_Convergence.csv`): scalar $|C - q^{\mathrm{peak}}|$ and $|C^k - C^{k-1}|$ — **not** supplier-cap minus buyer-cap.
- **Shared $C$ settled** (`shared_contract_capacity_settled`): ADMM does **not** stop while any link is expanding, pending an expand (`CapSignal` up), or moving more than `contract_cap.settle_tol` MW per iteration. Holding a positive $C$ after both sides stop asking for more is allowed. Idle snap to zero still has to finish and then stay flat. Flow residuals alone can pass while $C$ is still crawling.

**Tolerances** (contracts case is more coupled; see §6.5):

- **`epsilon_contracts`** — Base per-slot tolerance for flow markets (shipped 2.0 vs ME `epsilon` 0.2; 10× ratio).
- **`epsilon_cap`** — Scalar MW bar for physical and bilateral contract capacity (shipped 5.0 MW).
- **`cap_tol_relax`** — Extra multiplier for **physical** investment capacity only (shipped `data.yaml` value: 10 → 50 MW).
- **`contract_cap.settle_tol`** — Max allowed $|C^k - C^{k-1}|$ (MW) when declaring convergence (default 0.5).

Full economic interpretation of $C$, $K$, volume modes, and the $C$ step-size rule: §2 *Contract pools*. Configuration: `ADMM.contract_cap`, `PPAs`, `HPAs` in `data.yaml`.

### 6.8 Sign Convention

| Role | Net position sign | Example |
|---|---|---|
| Supplier / seller | **Positive** | VRES generation `+g`, H₂ sales `+h2_out` |
| Buyer / consumer | **Negative** | Electricity demand `−d`, H₂ purchase `−h2_in` |

Market imbalance = Σ (net positions). Positive imbalance = excess supply → price decreases. Negative imbalance = excess demand → price increases.

### 6.9 Numerical stabilisers and practical convergence

Coupled multi-market ADMM with endogenous investment (and, in the contract cases, bilateral pools) does **not** decrease every residual monotonically each iteration. Beyond the core updates of §6.1–§6.5, `ADMM.jl` applies a few lightweight stabilisers. All of them vanish at convergence and **none** changes the equilibrium being computed (§6.0.4):

| Stabiliser | What it does | Detail |
|---|---|---|
| **Scale-aware η damping** | Shrinks the λ step from $1$ toward $0.25$ as a market nears its Boyd tolerance, damping end-game oscillation. | §6.1 step 4 |
| **Per-market `η_scale`** | An extra multiplier on each market's λ step, adapted online: reduced (×0.85, floor 0.15) when that market's normalized merit worsens step-on-step, eased back up (×1.03, cap 1.0) when it improves. | `ADMM.jl` |
| **H₂-GC price projection** | Clamps $\lambda_{H2\_GC}\ge 0$ after each update (projected ADMM). A producer never issues certificates at a negative price, so supply is $0$ there; without the clamp, negative prices attract unbounded demand and create a persistent limit cycle. | §6.1 step 4 |
| **Adaptive ρ** | Boyd residual balancing, per market and per capacity agent. | §6.3 |
| **Best-iterate tracking** | Records the lowest-**merit** iterate seen so far and reports it if `max_iter` is reached without convergence. | `ADMM.jl` |

Here **merit** is the maximum over markets of (residual ÷ its Boyd tolerance), so a single number is comparable across markets of very different magnitude.

**Experimental steering (present in code, disabled by default).** The loop also contains a checkpoint rollback, per-market "basin guards", and a rescue mode that would blend $\lambda$ and $\rho$ back toward best-so-far values during long stalls. These are gated **off** (`enable_recovery_steering = false`, and the basin-guard activation threshold set beyond `max_iter`) because they tended to over-steer tightly coupled runs. The shipped algorithm relies only on the plain ADMM updates plus the stabilisers in the table above; treat the gated branches as inactive when reading `ADMM.jl`.

In practice this gives fast coarse convergence in the first iterations and few late plateaus. The main levers for tighter final residuals are a smaller `epsilon` (§6.5) and the social-planner warm-start (§6.6), **not** the disabled steering.

### 6.10 Mapping to Boyd et al. (2011)

Boyd et al., *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers* (2011), formulate consensus problems as:

$$
\min \sum_i f_i(x_i) \quad \text{s.t.}\quad x_i = z,\;\; \sum_i x_i = 0
$$

This project uses the **sharing ADMM** variant for market $k$: each agent $i$ has net position $g_i^k$; market clearing is $\sum_i g_i^k = 0$ (plus fixed demand for EP). The implementation introduces consensus copies via targets $\bar{g}_i^k$ and penalty $\frac{\rho_k}{2}\|g_i^k - \bar{g}_i^k\|^2$ in each agent objective (§5.1), with coordinator updates:

| Boyd / standard ADMM step | This codebase | Reference |
|---|---|---|
| $x$-update (local minimise augmented Lagrangian) | `ADMM_subroutine!` → `solve_*_agent!` (Gurobi) | §6.1 step 1 |
| $z$-update (consensus average) | $\bar{g}_i^k = q_i^{k-1} - \frac{1}{n+1}\mathrm{imbalance}^{k-1}$ | §6.2 |
| Dual ascent $u \leftarrow u + \rho(x-z)$ | $\lambda^{k+1} = \lambda^k - \eta\rho\,\mathrm{imbalance}^k$ (sign: imbalance = supply−demand) | §6.1 step 4, §6.8 |
| Primal residual $\|Ax+Bz-c\|_2$ | $\|\mathrm{imbalance}^k\|_2$ per market | §6.1 step 3 |
| Dual residual $\|\rho A^\top(z^k-z^{k-1})\|_2$ | $\|\rho\,\Delta(\text{consensus deviation})\|_2$; capacity uses $\Delta z$ | §6.1 step 3, §6.4.2 |
| Stopping: $\|r\|_2 \le \epsilon_{\mathrm{pri}}$, $\|s\|_2 \le \epsilon_{\mathrm{dual}}$ | Boyd abs + rel tolerances with $\sqrt{n_{\mathrm{slots}}}$ scaling | §6.5 |
| Adaptive $\rho$ (residual balancing) | `update_rho!` with $\mu=1.2$ | §6.3 |

**Extensions beyond vanilla Boyd:**

- **Multiple coupled consensus constraints** (five markets + capacity splits + optional contract pools) — one ADMM loop, market-specific $\rho_k$.
- **Projected dual step** for H2_GC ($\lambda\ge 0$).
- **η damping** and the H₂-GC projection — engineering stabilisers for tightly coupled energy markets (§6.9); not in the original Boyd proof but standard in applied ADMM.
- **CVaR rebuild** each iteration — agent $f_i$ changes with $\lambda$; subproblem remains convex but is re-solved from scratch.

At convergence, Boyd’s conditions imply $\mathrm{imbalance}\approx 0$ and stable consensus → **competitive equilibrium** (§4.1).

---

## 7. Social Planner Benchmark

The social planner (`social_planner.jl`) implements the **complete risk trading** case (§4.8): a single centralised convex QCP that maximises risk-adjusted social welfare with one **social CVaR** on aggregate welfare, subject to all individual agent constraints plus market-clearing balance constraints. It is the first-best benchmark under **centralised risk pooling**, not the decentralised incomplete-risk-trading equilibrium solved by ADMM.

When `γ=1` (risk-neutral), the CVaR term drops from the objective economically and the planner reduces to standard stochastic welfare maximisation; the same unified epigraph/QCP structure is kept across runs. When `γ<1`, the case is still an optimisation problem (d’Aertrycke [b]), but ME/ME+C need not match its primal or dual solution.

### 7.1 Market-Clearing Constraints

| Constraint | Equation |
|---|---|
| Electricity balance | `Σ generation − Σ demand − Σ electrolyzer_elec_buy = 0` (per h,d,y) |
| Elec GC balance | `Σ VRES_generation − Σ electrolyzer_GC_buy − Σ GC_demand = 0` (per h,d,y) |
| H₂ balance | `Σ H₂_production − Σ H₂_consumption − Σ offtaker_H₂_buy = 0` (per h,d,y) |
| H₂ GC balance | `Σ H₂_GC_supply − Σ H₂_GC_demand = 0` (per h,d,y — hourly, same as other markets) |
| EP balance | `Σ offtaker_EP_supply − D_EP − Σ EP_demand = 0` (per h,d,y) |

### 7.2 Price Recovery (Direct QCP Duals)

Equilibrium prices are the **dual variables** (shadow prices) of the market-clearing constraints. This interpretation holds for **both** $\gamma = 1$ and $\gamma < 1$:

- **Risk-neutral ($\gamma = 1$):** each dual is the expected **marginal social value** of relaxing that market’s balance by one unit (classical welfare economics).
- **Risk-averse ($\gamma < 1$):** each dual is the **risk-adjusted marginal social value** — the marginal impact on the planner objective $\gamma \sum_y \mathrm{sw}^{\mathrm{aux}}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}$ from relaxing that balance by one unit. These are the correct **commodity shadow prices** for the complete-risk-trading benchmark; they are **not** required to equal ADMM prices when agents use private CVaR (§4.8).

The social CVaR enters through `sw_aux` and linear shortfall constraints (§7.4); the balance constraints themselves are unchanged. The epigraph binds at optimality (`sw_aux[y] = social_welfare[y]` in code; $\mathrm{sw}^{\mathrm{aux}}_y = \mathrm{socialWelfare}_y$ mathematically), so risk adjustment affects **levels** of prices and quantities, not the fact that balance duals are well-defined shadow prices.

The planner is solved directly as a convex QCP with IPOPT (SP-only solver).

Workflow:

1. **QCP solve**: Solve the full social-planner QCP. Accept `OPTIMAL` and `LOCALLY_SOLVED` (convex QCP).
2. **Dual availability check**: Require `has_duals(planner) == true`. If duals are unavailable, the run fails and the solver/settings must be changed.
3. **Price extraction**: Read duals of the five balance constraints and convert them to per-MWh prices for `Market_Prices.csv` using the scaling in **Dual scaling** below.

Primal quantities and capacities are read from the same solved QCP model; no reformulation or proxy stage is used in the benchmark pipeline.

#### Dual scaling (balance duals → €/MWh)

The objective sums welfare with two scalings per scenario year $y$:

- Representative-day weights $W_{d,y}$ on each hourly term.
- Scenario weights in the risk-adjusted objective: $\gamma P_y$ on `sw_aux[y]` in the expected term, plus additional weight on tail scenarios through the social CVaR.

Raw balance-constraint duals therefore carry both $W$ and an effective scenario multiplier. For each market $k$ and timestep $(h,d,y)$:

$$
\text{price}_k(h,d,y) = \frac{\text{dual}(\text{balance}_k)_{h,d,y}}{W_{d,y} \cdot \mu_y}
$$

where $\mu_y$ is the **effective scenario weight** for year $y$:

- **Risk-neutral ($\gamma = 1$):** $\mu_y = P_y$.
- **Risk-averse ($\gamma < 1$):** $\mu_y = \gamma P_y + \xi_y$, where $\xi_y$ is the CVaR tail contribution (dual of the shortfall constraint $u_y \ge -\mathrm{sw}^{\mathrm{aux}}_y - \alpha_{\mathrm{social}}$).

In code (`save_social_planner_results.jl`), $\mu_y$ is read as the dual of the epigraph constraint `sw_aux[y] ≤ social_welfare[y]`. At optimality this dual equals the full marginal weight on scenario $y$ in the planner objective ($\gamma P_y + \xi_y$). If that dual is numerically zero, the implementation falls back to $\gamma P_y$.

The epigraph supplies $\mu_y$ for this normalization as well as the linear CVaR structure (§7.4); it is part of the single QCP solve, not a separate reformulation step.

Why IPOPT (and not Gurobi) for SP duals:
- In this project’s large/scaled SP QCP instances, Gurobi can return primal-optimal status (`LOCALLY_SOLVED`) while still failing to expose usable QCP duals after tightened barrier settings.
- The social planner benchmark requires reliable dual multipliers for market-price comparison; IPOPT delivers these multipliers directly for the solved QCP in this workflow.
- ADMM subproblems use Gurobi; `social_planner.jl` uses IPOPT by default (`SocialPlanner.solver` in `data.yaml`).

#### IPOPT settings and convergence tolerance (`ipopt_tol`)

Solver options live in the `SocialPlanner` block of `data.yaml` and are applied in `social_planner.jl`. The default **`ipopt_tol: 1.0e-6`** is a deliberate choice for this model’s economic scales — not a loose “good enough for debugging” setting.

**What `tol` means.** IPOPT’s `tol` is a **KKT residual tolerance**: it bounds combined primal/dual infeasibility and complementarity at the reported solution. It is **not** a direct “price must be within X €/MWh” knob. Still, for a well-scaled convex QCP, a satisfied KKT tolerance of order $10^{-6}$ implies that reported shadow prices and welfare are accurate **far below** any resolution used in reporting or ADMM warm-start.

**Why `1e-6` is sufficient here.** Typical magnitudes from NL-calibrated risk-neutral and risk-averse social-planner runs ($\gamma=0.5$, $\beta\in\lbrace 0.2,\ldots,0.8\rbrace$) and the implied absolute error if one (conservatively) treats `tol` as a relative scale on each quantity:

| Quantity | Typical magnitude (current 15-scenario calibration) | Order-of-magnitude error at `tol = 1e-6` |
|---|---|---|
| Electricity price | ~106 €/MWh | ~$1.1\times10^{-4}$ €/MWh (~0.01 cent/MWh) |
| H₂ price | ~71 €/MWh | ~$7\times10^{-5}$ €/MWh |
| EP price | ~175 €/MWh | ~$1.7\times10^{-4}$ €/MWh |
| H₂ GC price | ~113 €/MWh_GC | ~$1.1\times10^{-4}$ €/MWh_GC |
| Expected social welfare | ~43 bn € | ~43 k€ |
| VRES installed capacity | ~40 GW (solar + wind) | ~40 kW |

These residuals are negligible for:

- **Price CSVs and plots** (€/MWh, two significant figures in practice).
- **ADMM warm-start** from `Market_Prices.csv` and `SP_Capacities.csv` (§6.6).
- **Welfare and CVaR comparisons** between SP and ME at the bn-€ scale.

**Why not `1e-8` by default?** Tighter KKT tolerance mainly buys marginally cleaner dual multipliers. On the **stiff social CVaR QCP** (epigraph + tail constraints, especially at **low $\beta$** with few scenarios), `1e-8` often makes IPOPT declare `LOCALLY_INFEASIBLE` even when the primal allocation is already economically meaningful. That failure mode triggered unnecessary auxiliary solves and retries without improving reported prices in any material way.

**Default workflow at $\gamma<1$.** IPOPT’s restoration phase often reports `LOCALLY_INFEASIBLE` from a cold start even though the RA QCP has the **same constraints** as the RN problem (only the objective changes). The shipped path is:

1. CVaR seeds in `seed_social_cvar_starts!` (epigraph + Rockafellar–Uryasev feasible; the old $\mathtt{sw\_aux}=\alpha=-10^{15}$, $u=0$, $\mathrm{CVaR}=0$ start was itself infeasible).
2. **`risk_warmstart: true`**: solve $\gamma=1$ first, copy dispatch/capacity, reseat $\alpha,u,\mathrm{CVaR}$ at the target $\beta$.
3. Adaptive barrier (`ipopt_mu_strategy: adaptive`).
4. If the RA pass still fails, retry at `ipopt_retry_tol = 1e-5` from the failed iterate.

| Parameter | Default | Role |
|---|---|---|
| `ipopt_tol` | `1.0e-6` | Primary KKT tolerance |
| `ipopt_max_iter` | `5000` | Iteration cap |
| `ipopt_print_level` | `0` | `0` = silent; `3`–`5` = IPOPT log |
| `risk_warmstart` | `true` | If $\gamma<1$: extra $\gamma=1$ solve, copy primals, reseat CVaR |
| `risk_warmstart_beta` | `ADMM.beta` | $\beta$ for that auxiliary solve (idle at $\gamma=1$) |
| `ipopt_mu_strategy` | `adaptive` | IPOPT barrier update |
| `ipopt_retry_tol` | `1e-5` | Used only if the first RA solve fails (code default) |
| `ipopt_retry_max_iter` | `8000` | Iteration cap on retry (code default) |

Set `risk_warmstart: false` only to skip the extra RN solve (e.g. $\gamma=1$, or after you have confirmed a single RA pass succeeds). See also **§9.2.1**.

#### ADMM note on capacity tolerance scaling

In `market_exposure` ADMM, flow-market tolerances use Boyd-style horizon scaling (`ε_abs * sqrt(n_slots) + ε_rel * scale`), where `n_slots = nHours × nReprDays × nYears`.  
Capacity consensus is not a full flow tensor; it is a scalar MW split. It is therefore tested against a dedicated `ε_cap` (shipped 5.0 MW) with **no** `sqrt(n_slots)` factor — see §6.5. Applying the flow `ε_abs` unscaled to capacity would make the cap test ~50× tighter in MW than the market L2 test.

### 7.3 Code Architecture

All problem definition lives in `Source/build_*.jl` files. Each file contains:

- `build_*_agent!()` — Builds the ADMM version (with `λ`, `ρ`, `ḡ` penalty terms and per-agent CVaR for risk-averse agents).
- `add_*_agent_to_planner!()` — Adds the same variables/constraints to the planner model **without** ADMM terms and **without** per-agent CVaR. Returns a `Dict{Int, Any}` of per-year welfare expressions.

`build_social_planner.jl` orchestrates the calls to all `add_*_to_planner!` functions, adds market-clearing constraints, aggregates per-year welfare into `social_welfare`, adds the epigraph formulation and single social CVaR, and sets the risk-adjusted objective.

### 7.4 Epigraph Formulation for Social CVaR

The social planner applies **one single CVaR** to the aggregate social welfare (not per-agent CVaR). This ensures risk aversion considers all welfare components (consumer utility, production costs, investment costs) holistically.

**Problem**: `social_welfare[y]` includes quadratic terms from both elastic demand utility (`A·d − B/2·d²`) and conventional stage costs (`base_s·q_s + 0.5·slope_s·q_s²`). Putting `−social_welfare[y]` directly inside the CVaR shortfall constraints would place those quadratics in the CVaR block.

**Solution — epigraph reformulation**: Introduce auxiliary variables `sw_aux[y]` (math: $\mathrm{sw}^{\mathrm{aux}}_y$) with epigraph constraints:

$$
\begin{aligned}
& \mathrm{sw}^{\mathrm{aux}}_y \le \mathrm{socialWelfare}_y \quad \forall y \in \mathcal{Y} \\
& \quad\text{(quadratic in $\mathrm{socialWelfare}_y$; convex QC)}
\end{aligned}
$$

The CVaR constraints then reference `sw_aux` instead of the quadratic `social_welfare`, making them purely linear:

$$
\begin{aligned}
u_y &\ge -\mathrm{sw}^{\mathrm{aux}}_y - \alpha_{\mathrm{social}} \quad \forall y \in \mathcal{Y} \\
\mathrm{CVaR}^{\mathrm{social}} &\ge \alpha_{\mathrm{social}} + \frac{1}{1-\beta}\sum_{y \in \mathcal{Y}} P_y\, u_y
\end{aligned}
$$

**Important**: `α_social` and `cvar_social` must be **free** (no lower bound). When social welfare is positive, the social loss $-\mathrm{sw}^{\mathrm{aux}}_y$ is negative. The optimal VaR $\alpha$ for CVaR of a negative loss is negative. With $\alpha \ge 0$, $\mathrm{CVaR}^{\mathrm{social}}$ would be forced $\ge 0$, so the objective would become $\gamma \sum_y \mathrm{sw}^{\mathrm{aux}}_y$ instead of $\sum_y \mathrm{sw}^{\mathrm{aux}}_y$ when $\gamma < 1$ — breaking SP/ME equivalence for `nYears = 1`. With $\alpha$ free, $\mathrm{CVaR}^{\mathrm{social}}$ equals social loss when there is only one scenario, so the objective reduces to $\sum_y \mathrm{sw}^{\mathrm{aux}}_y$ regardless of $\gamma$.

The objective is also linear:

$$
\max \;\gamma \sum_{y \in \mathcal{Y}} \mathrm{sw}^{\mathrm{aux}}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}
$$

Since the objective maximises `sw_aux`, the epigraph constraint binds at optimality (`sw_aux[y] = social_welfare[y]` in code; $\mathrm{sw}^{\mathrm{aux}}_y = \mathrm{socialWelfare}_y$ in math), making the formulation mathematically equivalent to applying CVaR directly to `social_welfare`.

#### Epigraph and solver choice

The planner remains a **convex QCP**: welfare quadratics live in the epigraph constraints `sw_aux[y] ≤ social_welfare[y]`, not in an LP reformulation. IPOPT solves this QCP in one step and returns KKT multipliers. The epigraph is not required for IPOPT to handle quadratics (they could instead sit inside CVaR shortfall constraints); it is used because (1) it yields a standard linear Rockafellar–Uryasev CVaR block on `sw_aux`, and (2) epigraph duals supply the effective scenario weight $\mu_y$ used when converting balance duals to €/MWh (§7.2).

The epigraph constraints are the **only** quadratic constraints in the model (convex QC form). All other constraints (CVaR, market-clearing, capacity bounds) are purely linear. Commodity prices are recovered from balance-constraint duals with $W$ and $\mu_y$ scaling as described in §7.2.

### 7.5 Investment Decisions: Stochastic Single Year (SP and ME)

Both the social planner and market exposure include **endogenous investment** in VRES capacity (`cap_VRES`), electrolyzer H₂ capacity (`cap_H2_y`), and green offtaker EP capacity (`cap_EP_y`). The index `JY` lists **weather scenarios** (parallel uncertainty), not sequential calendar years. Each investing agent chooses **one** capacity level and **one** investment increment; dispatch (`g`, `e_in`, `ep`, …) is **scenario-indexed** only.

The formulations are structurally identical in SP and ME:

- **Non-anticipativity**: `cap[jy] = cap` for all scenarios (implemented as a scalar JuMP variable).
- **Fixed CAPEX**: `F_cap × cap` appears **once** in the objective (not summed over scenarios).
- **Operational economics**: scenario losses `loss[y]` are weighted by `P[y]` in the expected (`γ`) term; CVaR uses `loss_total[y] = loss_operational[y] + F_cap × cap`.
- **Availability**: `g[h,d,y] ≤ AF[h,d,y] × cap` — different weather affects dispatch and revenues, not the installed MW decision itself.

- **Social planner**: Each agent's scalar capacity is added to the centralised planner model. The planner optimises one investment per agent jointly with scenario-indexed dispatch. Expected welfare uses `Σ_y P[y] × sw_aux[y]`.

- **Market exposure (ADMM)**: Each agent holds scalar `cap` in its decentralised model and reaches a **consensus capacity** through the per-agent equality split $x_m = z_m$ of §6.4 — i.e. a dual term $\lambda^{\mathrm{cap}}_m(x-z)$ **plus** the quadratic penalty $(\rho^{\mathrm{cap}}_m/2)(x-z)^2$. The target `z_cap` is the capacity implied by flow consensus over **all** scenarios (e.g. for VRES, `z_cap = max over (h,d,y) of g_bar[h,d,y] / AF[h,d,y]`). At convergence every agent's `cap` equals its `z_cap`. Full model, residuals, and controller: §6.4.

**Why warm-start matters for investment**: Without capacity warm-start from the SP, the first ADMM iteration has `z_cap` derived from zero flows (ḡ = 0), so `z_cap = 0`. Agents are then penalised toward zero capacity, which is far from the equilibrium. With SP capacity seeds (`set_start_value`) and primal warm-start (ḡ = SP), `z_cap` is consistent with SP flows and the capacity penalty pulls agents toward the SP investment levels from the first iteration. This dramatically speeds convergence of the investment consensus.

### 7.6 Risk metrics post-processing (CVaR reporting)

After each run, the project writes **`Risk_Metrics.csv`**, **`Social_Welfare_Per_Year.csv`**, **`Welfare_By_Group_Per_Year.csv`**, and **`Welfare_By_Agent_Per_Year.csv`**, and prints a **risk metrics** block to the console. Implementation: `Source/compute_social_risk_metrics.jl`, called from `save_social_planner_results.jl`, `save_results.jl`, and `save_results_contracts.jl`. The planner **objective is unchanged**: demand utility stays inside social welfare and social CVaR. The extra files are a **reporting split** so RA / contract effects are visible on the non-demand remainder.

#### What is reported

| Quantity | Social planner (SP) | ADMM (ME / ME+C) |
|---|---|---|
| **Expected social welfare** $\mathbb{E}[\mathrm{SW}]=\sum_y P_y\,\mathrm{SW}_y$ | From solved `sw_aux` (binds to aggregate welfare) | **Ex-post**: recomputed from converged quantities using the same planner welfare accounting (no $\lambda$ transfers) |
| **Social CVaR** | Value of `CVaR_social` from the solved planner | **Ex-post social CVaR**: same Rockafellar formula as SP applied to $L_y=-\mathrm{SW}_y$ from the ADMM allocation |
| **$\alpha$ (VaR proxy)** | `alpha_social` from the planner | From the ex-post CVaR calculation |
| **Sum of private agent CVaRs** | n/a | $\mathrm{CVaR}_{\mathrm{VRES}}+\mathrm{CVaR}_{\mathrm{H2}}+\mathrm{CVaR}_{\mathrm{Green}}$ at ADMM convergence (internal to agent problems) |
| **Gap vs SP** | 0 | `social_CVaR_gap_vs_SP` = ex-post ADMM social CVaR minus SP social CVaR (requires `social_planner_results/Risk_Metrics.csv` from a prior SP run) |
| **Demand vs rest** | Same split | `expected_welfare_demand` = electricity + GC (+ H₂/EP if present) **utility**; `expected_welfare_ex_demand` = all other planner welfare (generation, H₂, ammonia, import **costs**). Transfers still cancel. `share_demand_of_E_SW` can exceed 100% because the rest is typically negative. |

**Demand vs rest (diagnostics only).** Net social welfare is dominated by consumer utility \(U(d)\). Risk aversion and contracts mainly move **real-resource costs** (VRES annuity, thermal fuel, electrolyzer, Haber–Bosch, imports). Compare cases on `expected_welfare_ex_demand` and `Welfare_By_Group_Per_Year.csv`, not only on \(\mathbb{E}[\mathrm{SW}]\). The ex-post CVaR of the rest (`welfare_ex_demand_CVaR`) is **not** in any agent’s objective.

**Important distinctions:**

1. **Expected social welfare** is the **probability-weighted mean** $\sum_y P_y \mathrm{SW}_y$, not the welfare in a single “most likely” year. With uniform $P_y=1/n_Y$, it is the arithmetic average across scenario years.

2. **Social CVaR** in the code is **CVaR of social loss** $L_y = -\mathrm{SW}_y$ (tail of bad aggregate outcomes). The planner **minimizes** $(1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}$ in the objective (equivalently penalizes bad tails). **Lower social CVaR is better** (less tail risk).

3. **SP social CVaR** is the **complete risk trading** optimum: one coherent tail-risk measure on **aggregate** welfare.

4. **ADMM ex-post social CVaR** answers: “If we take the decentralized equilibrium allocation and aggregate welfare by year, how bad is the tail?” It uses the **same** $\beta$ and $P_y$ as SP but reflects **incomplete risk trading** (private CVaRs in agents, no social CVaR in the ADMM iteration).

5. **Sum of private CVaRs** is **not** additive social risk; it is reported for diagnostics only. Do not expect it to equal social CVaR.

#### Theory-based comparison when $\gamma < 1$

Under the usual ordering (same technology, same $\beta$, SP = centralized tail-risk management):

- **Ex-post social CVaR (ADMM)** should be **$\geq$ SP social CVaR** (ADMM weakly worse tail on aggregate loss): the planner pools risk efficiently; decentralized private hedging generally cannot improve the **system** tail metric.

- **Quantities and prices** need not match SP (§4.8); only the **risk metric** is constructed to be comparable ex post.

When $\gamma = 1$, social CVaR is inactive in objectives; reported values should still compute but are not central to the optimum.

#### Output files

| File | Contents |
|---|---|
| `Risk_Metrics.csv` | One row per metric (`expected_social_welfare`, `social_CVaR`, `alpha_social`, `sum_private_CVaR`, `social_CVaR_gap_vs_SP`, `expected_welfare_demand`, `expected_welfare_ex_demand`, `share_demand_of_E_SW`, `welfare_ex_demand_CVaR`, plus `expected_welfare_group_*`) |
| `Social_Welfare_Per_Year.csv` | `scenario_year`, `probability`, `social_welfare`, `social_loss`, `welfare_demand`, `welfare_ex_demand` |
| `Welfare_By_Group_Per_Year.csv` | Per scenario year, planner welfare by group (`elec_demand`, `gc_demand`, `vres`, `conventional`, `h2_producer`, offtakers, …) |
| `Welfare_By_Agent_Per_Year.csv` | Same split at agent id |
| `Private_CVaR_By_Agent.csv` | ADMM only: per risk-averse agent `CVaR_private` and `alpha_private` |

#### Console log

Logs report **positive welfare** (bn EUR): expected welfare, **demand utility vs ex-demand rest**, **tail welfare** (= average welfare in the worst $(1-\beta)$ share of scenarios; higher is better), min scenario welfare, spread, and the spread/tail of the ex-demand remainder. Internally the solver uses loss $L_y=-\mathrm{SW}_y$; `Risk_Metrics.csv` stores both `social_CVaR` (the model's epigraph variable) and `social_CVaR_recomputed` (ex-post from the realised per-scenario welfares). ADMM runs add **SP tail welfare (benchmark)** and **tail welfare gap vs SP** (negative ⇒ ME worse on aggregate tail).

**Tail lines are always the ex-post value.** At $\gamma = 1$ the CVaR term carries zero objective weight, so the solver's `cvar_social` variable is left arbitrarily loose and is not a meaningful number; reporting it would produce nonsense such as a tail welfare below the minimum scenario welfare, and an ADMM-vs-SP gap of several billion euros where the two are in fact identical. Both the console lines and the SP benchmark comparison therefore use the recomputed value.

Example block:

Actual output of `social_planner.jl` on the 15-scenario grid at $\gamma=0.5$, $\beta=0.8$:

```
------------------------------------------------------------------------
  Social planner risk metrics
------------------------------------------------------------------------
  Case:                    social_planner
  gamma:                   0.5000
  beta:                    0.8000  (tail = worst 20% of scenario years)
  E[social welfare]:                42.542 bn EUR
  Tail welfare (CVaR):              41.741 bn EUR  (higher = safer tail)
  Min scenario welfare:             41.677 bn EUR
  Welfare spread (max−min):          1.957 bn EUR
  Risk-adjusted objective:            42.142 bn EUR
```

Two checks worth running on any such output:

**The objective identity.** `Risk-adjusted objective` must equal $\gamma\,\mathbb{E}[\mathrm{SW}] + (1-\gamma)\,\mathrm{CVaR\text{-}welfare}$. Here $0.5 \times 42.542 + 0.5 \times 41.741 = 42.142$ — exact, confirming the reported pieces are mutually consistent.

**Risk aversion moves the right way.** Comparing against the risk-neutral run of the same grid ($\gamma=1$):

| Metric | $\gamma=1$ | $\gamma=0.5$, $\beta=0.8$ | Direction |
|---|---|---|---|
| E[social welfare] | 42.544 | 42.542 | slightly **lower** — expected welfare is sacrificed |
| Tail welfare (CVaR) | 41.655 | 41.741 | **higher** — the tail is made safer |
| Welfare spread | 1.997 | 1.957 | **narrower** — outcomes less dispersed |

That is the textbook signature of CVaR aversion: pay a little expected welfare to buy a better worst case. The effect is small here because the scenarios are not yet very dispersed in welfare terms (the spread is ~4.7% of the mean); it grows as $\beta$ rises or $\gamma$ falls.

When $\beta$ is too fine for the scenario count, the reporter appends an explicit warning and the $\beta$ values that give a tail of whole scenarios:

```
  (NOTE: beta = 0.95 implies a 5.0% tail, narrower than one of the 15
   equiprobable scenarios (6.7%), so CVaR collapses to the worst scenario.
   For a tail spanning k scenarios use beta = 1 - k/15, e.g. 0.800 for k=3.)
```

---

## 8. Data and Indexing

### 8.1 Temporal Dimensions

| Dimension | Set | Size | Description |
|---|---|---|---|
| Hours | `JH = 1:nTimesteps` | 24 | Hours within each representative day |
| Representative days | `JD = 1:nReprDays` | 8 | Representative days (configured in `data.yaml`) |
| Scenarios | `JY = 1:nYears` | 15 | Parallel scenarios = weather years × gas price levels (built from the `Scenarios` block, for both SP and ME); **not** sequential investment years |

### 8.2 Representative-Day Weights

`W[jd, jy]` = number of real calendar days that representative day `jd` stands for in scenario `jy`. Used to scale per-representative-day objective values to a full-year total. Weights are read from `Input/output_<label>/decision_variables_short.csv` and always sum to **365**. They are **fractional day-equivalents**, not integer day counts, because they are rebalanced so the weighted annual means reproduce the full-year means exactly — see §9.7.

### 8.3 Scenario Labels (JY mapping)

`build_scenario_grid` (in `Source/define_scenarios.jl`) expands the `Scenarios` block into two lookups keyed by the flat scenario index `jy`:

- `years[jy]` — the **weather file label** used to load `timeseries_<label>.csv` and `output_<label>/…`. Labels are **not calendar dates** (label 1 is built from 2015 ERA5 weather; see the table in §9.7). Because three gas scenarios share each weather label, each file is read **once per label** and reused.
- `gas_multiplier[jy]` — the scaling applied to `Fuel.GasPrice` in that scenario.

With the default grid, `years = Dict(1=>1, 2=>1, 3=>1, 4=>2, …, 15=>5)` and `gas_multiplier = [1.0, 1.1, 1.2, 1.0, …]`: gas varies fastest, so `jy = 1..3` are weather year 1 at the three gas levels. **Investment is decided once** before operations; scenarios differ only in availability factors, demand, fuel-linked costs, dispatch, and the scenario-weighted expected profit / CVaR. See §7.5, §9.7, and §9.8.

### 8.4 3D Arrays

All prices, quantities, and imbalances are stored as 3D arrays `[jh, jd, jy]`. Scalar diagnostics (mean price, mean imbalance) are computed per iteration for CSV output.

---

## 9. Configuration Reference (data.yaml)

### 9.1 General

| Parameter | Value | Description |
|---|---|---|
| `nTimesteps` | 24 | Hours per representative day (hourly resolution) |
| `nReprDays` | 8 | Representative days (trade-off: speed vs. accuracy) |
| `nYears` | 1 | Fallback scenario count, used only if the `Scenarios` block is absent |
| `base_year` | 2025 | `[NL]` Calibration-year **label** for installed capacities (CBS 2025 nader voorlopig), fuel prices (2024 annual averages), and costs; **not** a scenario file label and **not read by any code** — metadata for the documentation only; see §9.6–§9.8 |

### 9.1.1 Scenarios

| Parameter | Value | Description |
|---|---|---|
| `weather_years` | `[1, 2, 3, 4, 5]` | Weather file labels; each reads `Input/timeseries_<label>.csv` and `Input/output_<label>/` (§9.7) |
| `gas_price_multipliers` | `[1.00, 1.10, 1.20]` | Scaling applied to `Fuel.GasPrice`; coal, biomass, and CO₂ are unaffected (§9.8) |

The scenario count is **derived**: `nYears = 5 × 3 = 15`, all equiprobable.

### 9.1.2 Fuel

| Parameter | Value | Description |
|---|---|---|
| `GasPrice` | 34.40 €/MWh_th | `[NL]` TTF 2024 annual average `[TTF-2024]` |
| `CoalPrice` | 12.54 €/MWh_th | API2 2024 average (≈102 €/t ÷ 8.14 MWh_th/t) `[API2-2024]` |
| `BiomassPrice` | 33.00 €/MWh_th | NW-Europe industrial wood pellet 2024 `[PELLET-2024]` |
| `CO2Price` | 64.79 €/tCO₂ | `[NL]` EU-ETS EUA 2024 annual average `[ETS-2024]` |
| `GasEmissionFactor` | 0.2016 tCO₂/MWh_th | Natural gas, 56.1 kg/GJ `[IPCC-EF]` |
| `CoalEmissionFactor` | 0.3406 tCO₂/MWh_th | Hard coal, 94.6 kg/GJ `[IPCC-EF]` |
| `BiomassEmissionFactor` | 0.0 | ETS zero-rated |

This block is the **single source of truth** for every fuel- and carbon-linked cost: the conventional generator's stage costs and the grey offtaker's marginal cost are both derived from it, so one gas price moves power and ammonia together (§9.8).

### 9.2 ADMM

| Parameter | Value | Description |
|---|---|---|
| `rho_initial` | 1.0 | Default penalty weight (neutral starting point) |
| `nScenarioYears` | 15 | Fallback scenario count, used only if the `Scenarios` block is absent (§9.1.1) |
| `max_iter` | 1000 | Maximum ADMM iterations |
| `epsilon` | 0.2 | ME flow-market Boyd `ε_abs` (MW per slot); L2 bar = 0.2 × √2880 ≈ 10.73 MW. See §6.5. |
| `epsilon_contracts` | 2.0 | [me_pap / me_top / me_sop] Flow-market `ε_abs`; 10× ME (Boyd $10^{-4}$ of NL peak). L2 bar ≈ 107 MW (§6.5). |
| `epsilon_cap` | 5.0 | Scalar MW bar for physical (and bilateral contract) capacity splits. Not scaled by √n_slots (§6.4.3, §6.5). |
| `cap_tol_relax` | 10 | [me_pap / me_top / me_sop only] Extra multiplier on `ε_cap` for **physical investment** capacity (§6.4): 10 × 5 = 50 MW. Does **not** apply to bilateral contract $C$. |
| `contract_cap.initial` | 0 | Seed $C$ (MW). `0` = no-contract warm start; raise if $q \le C$ is presolved away (§2). |
| `contract_cap.relaxation` | 0.35 | Unused on expand/hold (kept for compatibility). Idle unused $C$ still snaps to 0. |
| `contract_cap.expand_step` | 25 | Minimum MW added to $C$ when **both** sides’ cap shadows want more $C$ (§2). Applied **undamped**. |
| `contract_cap.expand_frac` | 0.2 | Fraction of remaining headroom to the physical bottleneck added on each expand. |
| `contract_cap.expand_max` | 500 | Cap on one expand (MW). |
| `contract_cap.up_confirm_iters` | 3 | Consecutive unanimous “want more” votes before an expand (**code default**; not in shipped yaml). |
| `contract_cap.settle_tol` | 0.5 | Max $|C^k - C^{k-1}|$ (MW) before ADMM may stop (§6.7). |
| `contract_cap.settle_iters` | 3 | Consecutive iterations $C$ must stay within `settle_tol` (and not be expanding). Hold at a used $C$ is settled. |
| `contract_cap.bind_tol` | 1e-6 | Numerical tolerance: treat $q^{\mathrm{peak}} \approx C$ as binding. |
| `contract_cap.dual_tol` | 1e-4 | Min W-weighted shadow of $q \le C$ (€/MW-year) to count as “wants more $C$”. |
| `rho_cap_initial` | 0.1 | Initial per-agent capacity penalty for the equality split (§6.4). Present in shipped `data.yaml`. |
| `rho_cap_inc_factor` | 1.05 | Per-agent capacity controller increase factor; decrease factor is the reciprocal. **Code default** (not in shipped `data.yaml`). See §6.4.4. |
| `rho_cap_max` | 30 | Per-agent capacity penalty upper bound. **Code default** (not in shipped `data.yaml`). See §6.4.4. |
| `cap_z_relax` | 1.0 | Under-relaxation factor for capacity target update `z^k ← α z_raw^k + (1-α) z^{k-1}`. `1.0` disables damping. **Code default** (not in shipped `data.yaml`). Use `0.2–0.8` only if target oscillations cause large `Δz` dual spikes. See §6.4.8. |
| `gamma` | 1.0 | Risk weight on expected loss vs CVaR ($\gamma=1$ risk-neutral; $\gamma=0.5$ risk-averse base case). Shared by SP and ME. See §4.10. |
| `beta` | 0.95 | CVaR confidence level (Rockafellar–Uryasev); **higher $\beta$ = more risk-averse** at fixed $\gamma<1$. Sensitivity sweep: $0.2,0.4,0.6,0.8$ at $\gamma=0.5$ (§4.10.4). Inactive when $\gamma=1$. |

### 9.2.1 SocialPlanner (`SocialPlanner` block)

Used **only** by `social_planner.jl`. ADMM subproblems continue to use Gurobi.

| Parameter | Default | Description |
|---|---|---|
| `solver` | `ipopt` | SP solver (`ipopt` recommended; `gurobi` optional) |
| `ipopt_tol` | `1.0e-6` | IPOPT KKT tolerance. Justification and magnitude table: **§7.2** |
| `ipopt_max_iter` | `5000` | IPOPT iteration cap |
| `ipopt_print_level` | `0` | `0` = silent; `3`–`5` = iteration log |
| `risk_warmstart` | `true` | When $\gamma<1$: $\gamma=1$ auxiliary solve, copy dispatch/capacity, reseat CVaR. Prevents IPOPT restoration failure. |
| `risk_warmstart_beta` | `ADMM.beta` | $\beta$ for auxiliary warm-start only (does not change the RN primal) |
| `ipopt_mu_strategy` | `adaptive` | IPOPT barrier strategy |
| `ipopt_retry_tol` | `1e-5` | Retry tolerance if first solve fails (**code default** — commented out in shipped `data.yaml`) |
| `ipopt_retry_max_iter` | `8000` | Retry iteration cap (**code default** — commented out in shipped `data.yaml`) |

### 9.2.2 PartialPlanners (`PartialPlanners` block)

Used by `green_h2_social_planner.jl` and `green_social_planner.jl` only. See **§4.11**.

| Key | Members merged | Output folder |
|---|---|---|
| `GreenH2` | `Prod_H2_Green`, `Offtaker_Green` | `green_h2_social_planner_results/` |
| `Green` | `Gen_VRES_Solar`, `Gen_VRES_Wind`, `Prod_H2_Green`, `Offtaker_Green` | `green_social_planner_results/` |

Each block specifies `coalition_id` (JuMP agent ID) and `members` (YAML cross-refs to existing agent blocks).

### 9.3 Market Parameters

`initial_price` values are ADMM **warm-start seeds** (the social planner solves directly and recovers prices as duals). They are set near the expected NL-calibrated equilibrium so ADMM starts close to the solution.

| Market | `initial_price` | `rho_initial` | Notes |
|---|---|---|---|
| `elec_market` | 77.0 €/MWh | 1.0 | Seed near NL 2024 wholesale with realistic VRES/gas costs |
| `elec_GC_market` | 3.0 €/MWh_GC | 0.3 | Seed near realistic GoO clearing (abundant VRES ⇒ low premium) |
| `H2_market` | 155.0 €/MWh_H2 | 0.5 | Seed ≈ 5 €/kg green H₂ at realistic power prices |
| `H2_GC_market` | 25.0 €/MWh_GC | 1.0 | Seed near green-H₂ certificate value under the 42% mandate |
| `EP_market` | 118.0 €/MWh_EP | 3.0 | Seed just above the grey ammonia MC (~103 €/MWh_EP at base gas; §9.8); `Total_Demand` ≈ 1970 → ~3 Mt/y via `Demand_Column: LOAD_EP`. (The ~1000 €/t NH₃ ≈ 194 €/MWh_EP figure in §9.6 is a market-level reference, not this seed.) |

The empty top-level `EP_Demand: {}` block is a reserved placeholder for a future elastic EP-demand agent; with an empty dict, no such agent is created and EP demand stays inelastic via `EP_market.Total_Demand`.

See §9.6 for the full list of NL-calibrated inputs and their sources.

### 9.4 PPAs and HPAs (me_pap / me_top / me_sop)

```yaml
PPAs:
  initial_price: 60.0     # λ_ppa seed at ADMM start
  rho_initial: 0.5

HPAs:
  initial_price: 80.0
  rho_initial: 0.5
  price_structure: fixed    # fixed | cfd
  price_benchmark: negotiated  # negotiated | electricity | ammonia | NG
```

| Block | Parameter | Description |
|---|---|---|
| `ADMM.contract_cap` | `initial`, `expand_step`, `expand_frac`, `expand_max`, `bind_tol`, `dual_tol` | Shared bilateral capacity $C$ bargaining (§2, `contract_capacity.jl`) |
| `PPAs` | `initial_price`, `rho_initial` | Seed for `λ_ppa` and ADMM energy penalty on PPA flow |
| `HPAs` | `initial_price`, `rho_initial` | Seed for `λ_hpa` and ADMM energy penalty on HPA flow |
| `HPAs` | `price_structure` | `fixed` or `cfd` (§2 *HPA price structure*) |
| `HPAs` | `price_benchmark` | Benchmark $B$ for strike at convergence and CfD reference leg |

Per-agent overrides under `PPAs.Gen_VRES_Solar`, `HPAs.Prod_H2_Green`, etc.

PPA and HPA use **one shared scalar capacity** $C$ (MW) per bilateral link — stored in `ADMM["SharedCap"]`, updated by `contract_capacity.jl`, **not** chosen independently by buyer and seller. There is no separate MW clearing price for $C$. Settlement strike $K$ is a scalar €/MWh (uniform over hours) equal to the W-mean of bundled physical spot (PPA: elec+GC; HPA: H₂+H₂-GC when `negotiated`). $\lambda^{\mathrm{contract}}$ clears **hedge quantity** $q$ only; it is not the strike.

### 9.5 Agent Parameters

See `Data/data.yaml` for the full annotated configuration. Key parameters:

- **VRES**: `Capacity`, `Profile_Column`, `MarginalCost`
- **Conventional**: `Capacity`, `Technology` (name/fuel/efficiency/vom) for flat plants; or legacy `StageCapacityShares`, `StageTechnologies`, `PeakTechnology` (`StageBaseCosts` / `FinalMarginalCost` / scalar `MarginalCost` still accepted)
- **Consumer**: `PeakLoad`, `Load_Column`, `A_E`, `B_E` (quadratic utility)
- **Electrolyzer**: `Capacity_Electrolyzer`, `Capacity_H2_Output`, `SpecificConsumption`, `OperationalCost`
- **Green offtaker**: `Capacity_H2_In`, `Capacity_EP_Out`, `Alpha`, `ProcessingCost`
- **Grey offtaker**: `Capacity`, `GasIntensity`, `CO2Intensity`, `VariableOM`, `gamma_NH3` (legacy scalar `MarginalCost` still accepted as a fallback)
- **EP importer**: `Capacity`, `ImportCost`
- **GC demand**: `PeakLoad`, `Load_Column`, `A_GC`, `B_GC`

### 9.6 NL Calibration and Data Sources

This model is calibrated to the **Netherlands, base year 2025** (installed capacities from CBS 2025 nader voorlopig; fuel and wholesale prices from **2024** annual averages — the latest complete market year). Every input that represents Dutch reality is tagged `[NL]` in `Data/data.yaml` with a short source key; the keys are resolved here and in §14. Endogenous variables (installed capacities after investment, all market prices) are **outputs**, not inputs — the values below are the *inputs/seeds* the optimiser starts from or is bounded by.

**Unit convention.** Ammonia (the end product, EP) is accounted on an energy basis using its lower heating value **LHV = 18.6 MJ/kg ⇒ 5.167 MWh_EP per tonne NH₃** `[LHV]`. All €/t ↔ €/MWh_EP and Mt ↔ TWh conversions use this factor (e.g. 1000 €/t ÷ 5.167 ≈ 194 €/MWh_EP; 3 Mt/yr × 5.167 ≈ 15.5 TWh/yr).

#### NL-calibrated inputs (sourced)

| Parameter (agent) | Value | Basis / derivation | Source |
|---|---|---|---|
| `base_year` | 2025 | Calendar year of the CBS capacity and cost calibration; fuel/market anchor year 2024; weather files use labels 1..5 | `[CBS-RE]`, `[CBS-EP]` |
| Solar `Capacity` | 25,881 MW | CBS installed solar PV at end-2025 nader voorlopig (25,881 MWp) | `[CBS-RE]` |
| Wind `Capacity` | 11,782 MW | CBS installed wind (on+offshore) at end-2025 nader voorlopig | `[CBS-RE]` |
| Solar `FixedCost_per_MW` | 95,000 €/MW-yr | ≈0.79 M€/MW CAPEX+FOM, 25 yr, 8% WACC ⇒ LCOE ≈50 €/MWh | `[IRENA-2024]` |
| Wind `FixedCost_per_MW` | 185,000 €/MW-yr | ≈2.2 M€/MW (offshore-leaning) CAPEX+FOM ⇒ LCOE ≈62 €/MWh | `[IRENA-2024]` |
| Conventional capacity | 14,040 / 1,800 / 2,160 MW | CCGT / coal / biomass — 78/10/12% of 18 GW fossil proxy; matches 2024 generation shares | `[CBS-EP]` |
| Conventional flat SRMC at base gas | 83.3 / 85.4 / 89.8 €/MWh_e | CCGT / coal / biomass from `Fuel` block; merit order via market | `[TTF-2024]`, `[ETS-2024]`, `[IPCC-EF]` |
| `Fuel.GasPrice` | 34.40 €/MWh_th | TTF 2024 annual average; anchor for gas-linked power and ammonia costs (§9.8) | `[TTF-2024]` |
| `Fuel.CO2Price` | 64.79 €/tCO₂ | EU-ETS EUA 2024 annual average | `[ETS-2024]` |
| Consumer `PeakLoad` | 19,500 MW | NL system peak 19.48 GW (ENTSO-E 2024); annual load ≈109 TWh (2024 net consumption) | `[CBS-EP]`, `[ENTSOE-PEAK-2024]` |
| Electrolyser `SpecificConsumption` | 1.5 MWh_e/MWh_H₂ | PEM efficiency ≈67% LHV | `[IEA-H2]` |
| Electrolyser `Capacity_Electrolyzer` (seed) | 800 MW_e | IEA-style electrical nameplate; implied H₂ = 800/1.5 = 533.3 MW_H2 ≈ 20% of NL NH₃ H₂ feed | `[PBL-2019]`, `[IEA-H2]` |
| Electrolyser `FixedCost_per_MW_Electrolyzer` | 262,000 €/MW_e-yr | IEA 2160 USD/kWe × 0.92 €/USD = 1.987 M€/MW_e; CRF 8%/20 yr + 3% FOM | `[IEA-H2-2024]` |
| Green offtaker `Capacity_EP_Out` (seed) | 400 MW_EP | Product nameplate ≈20% of NL ammonia; H₂ feed = 400/0.75 = 533.3 MW_H2 | `[PBL-2019]` |
| Green offtaker `FixedCost_per_MW_EP_Out` | 158,000 €/MW_EP-yr | IEA 770 USD/(t NH₃/y) synthesis+ASU × 0.92 × 8760/5.167 = 1.201 M€/MW_EP; CRF+3% FOM | `[IEA-NH3-2024]` |
| EP demand `Total_Demand` | 1,970 | ≈3 Mt NH₃/yr (Yara Sluiskil ~1.8 Mt + OCI Geleen ~1.2 Mt) ⇒ ~15.5 TWh/yr | `[PBL-2019]`, `[LHV]` |
| EP `initial_price` | 100 €/MWh_EP | ADMM starting guess only (not a calibration target): derived grey MC 85.7 plus the cost of the GC mandate | `[H2EU-2023]`, `[LHV]` |
| Green offtaker `Alpha`, Grey `gamma_NH3` | 0.75 | H₂↔EP conversion ≈70–80% Haber–Bosch LHV efficiency | `[H2EU-2023]` |
| Grey offtaker `Capacity` | 1,570 MW_EP | ≈80% of NL ammonia nameplate (domestic conventional + ex-import share) | `[PBL-2019]` |
| Grey offtaker `GasIntensity` | 1.720 MWh_th/MWh_EP | SMR route, 32 GJ_LHV per t NH₃ ÷ 5.167 MWh/t | `[FE-BAT]`, `[DECHEMA-2030]` |
| Grey offtaker `CO2Intensity` | 0.348 tCO₂/MWh_EP | 1.8 tCO₂/t NH₃ (process + fuel) ÷ 5.167 MWh/t, charged in full | `[FE-BAT]` |
| ⇒ derived grey MC at base gas | 85.7 €/MWh_EP (≈443 €/t) | 59.2 gas + 22.5 CO₂ + 4.0 O&M; consistent with 2024 NW Europe ammonia $528–581/t | `[H2EU-2023]`, `[TTF-2024]`, `[ETS-2024]`, `[OCI-NH3-2024]` |
| GC mandate `gamma_GC` (in code) | 0.42 | EU RED III RFNBO-in-industry target (≥42% renewable H₂ by 2030) | `[RED-III]` |
| GC demand `PeakLoad` | 17,500 MW_GC | Most NL consumption seeks green certification (~system load) | `[CBS-EP]` |
| GC demand `A_GC` | 10 €/MWh_GC | Max WTP ≈ recent EU GoO price peak (clears lower) | `[GoO]` |

#### Engineering estimates and modeling choices (not direct NL measurements)

These are physically reasonable but not pulled from a single NL statistic; flagged in `data.yaml` so they are not mistaken for empirical data:

- **Green offtaker `ProcessingCost` = 12 €/MWh_EP (≈62 €/t)** — variable O&M for Haber–Bosch + air separation + compression (not the plant CAPEX).
- **Electrolyser `OperationalCost` = 3 €/MWh_H₂** — variable O&M estimate (water, stack); fixed O&M is inside the IEA 3%-of-CAPEX annuity.
- **Consumer `A_E` = 500 €/MWh, `B_E` = 0.0025; GC demand `B_GC` = 0.0005** — quadratic-utility shape parameters; chosen to keep electricity demand near-inelastic at the ~20 GW scale and the GC market well-scaled (not NL price observations).
- **Importer `ImportCost` = 250 €/MWh_EP** — inactive (`Capacity = 0`); retained for re-enabling imports.
- **Risk parameters `gamma` = 1.0 (risk-neutral default; set `0.5` for risk-averse runs), `beta` = 0.95; all `rho_initial`, tolerances, `max_iter`** — algorithmic/risk-preference settings, not NL data.

### 9.7 Weather Scenarios, Representative Days, and Availability Factors

This section documents how the hourly input files are built, how they enter the model, and what each weather label represents. Weather is only **one** of the two scenario dimensions; the gas-price dimension and the resulting 15-scenario grid are described in §9.8.

#### Pipeline at a glance

Everything below is executed by one driver, `Input/rep_periods/generate_representative_days.jl`. **RepresentativePeriodsFinder.jl (RPF) performs the representative-day selection** — it is not bypassed or replaced at any point.

| # | Step | Who does it | Output |
|---|---|---|---|
| 1 | Choose which 5 of 10 candidate ERA5 years to use | `_select_diverse.jl` (run once, offline) | the fixed label → source-year mapping |
| 2 | Fetch hourly GHI, 100 m wind, 2 m temperature | Open-Meteo / ERA5 | `weather_cache/` |
| 3 | Convert to SOLAR / WIND capacity factors; apply the **common** NL calibration | driver | 8760 h CF series |
| 4 | Build temperature-coupled `LOAD_E` (degree-day model) | driver | 8760 h load series |
| 5 | **Select 8 representative days and their weights** | **RPF (hierarchical clustering)** | `results/<label>/` |
| 6 | Rebalance the **weights only** so annual means are exact | driver (Dykstra projection) | corrected `weights` column |
| 7 | Export to model format | driver | `timeseries_<label>.csv`, `output_<label>/` |

Step 6 is the only post-processing applied to RPF's output, and it touches **nothing except the weight vector**: the eight selected calendar days, their 24-hour profiles, the chronological ordering, and the 365-day total are all exactly as RPF produced them. The justification, with measured numbers, is under *Why the weights are rebalanced* below.

#### Weather labels vs source years

The model uses **5 weather labels** `1`–`5` (`Scenarios.weather_years`). These are **indices for distinct weather scenarios**, not calendar years. Every label enters the **same single operating year**: the optimiser chooses one investment level, then evaluates expected profit and CVaR across all scenarios. Labels are **not** years in which an agent reinvests each period.

The five labels were **selected**, not assumed. `Input/rep_periods/_select_diverse.jl` builds a **31-dimensional feature vector** per candidate year — 12 monthly mean solar CFs, 12 monthly mean wind CFs, and 7 annual aggregates (mean solar CF, mean wind CF, their standard deviations, dunkelflaute frequency, mean heating degree-hours, mean cooling degree-hours). Features are **z-scored** across the ten candidates so that no feature dominates through its units, then the script does an **exhaustive** search over all $\binom{10}{5} = 252$ subsets for the one maximising the **minimum pairwise Euclidean distance** (a max-min / maximin-dispersion criterion). Maximising the *minimum* distance rather than the average is what prevents the answer from being four similar years plus one outlier: every pair in the chosen set must be far apart.

Crucially, the search runs **after** the common NL calibration (below), so it compares physically comparable years rather than rewarding a calibration artefact — an earlier version scored the raw ERA5 years while only the reference year had been scaled to NL targets, which made that year an automatic outlier and biased the whole selection.

All ten candidates, with the chosen five marked (rerun the script to reproduce; selected subset scores a max-min distance of 7.405):

| Candidate # | Source year | Solar CF | Wind CF | Dunkelflaute | Mean HDD | Mean CDD | Selected |
|---|---|---|---|---|---|---|---|
| 1 | 2015 | 0.1800 | 0.2800 | 3.01% | 5.559 | 0.0968 | **yes** — wind maximum |
| 2 | 2010 | 0.1764 | 0.1922 | 8.22% | 7.451 | 0.0956 | **yes** — wind minimum, coldest, worst dunkelflaute |
| 3 | 2012 | 0.1745 | 0.2291 | 3.84% | 6.181 | 0.0790 | no — dominated by 2016 |
| 4 | 2013 | 0.1729 | 0.2368 | 4.93% | 6.667 | 0.1018 | no — close to 2016 / 2017 |
| 5 | 2014 | 0.1764 | 0.2447 | 5.48% | 4.898 | 0.0712 | no — close to 2017 |
| 6 | 2016 | 0.1793 | 0.2208 | 4.66% | 5.825 | 0.1064 | **yes** — mid-range anchor |
| 7 | 2017 | 0.1726 | 0.2457 | 4.66% | 5.510 | 0.0819 | **yes** — solar minimum, high wind |
| 8 | 2018 | 0.1925 | 0.2388 | 4.38% | 5.570 | 0.2195 | **yes** — solar maximum, hottest by 2× |
| 9 | 2019 | 0.1851 | 0.2566 | 2.74% | 5.392 | 0.1643 | no — between 2015 and 2018 |
| 10 | 2011 | 0.1753 | 0.2571 | 3.56% | 5.465 | 0.0512 | no — close to 2017 |

The candidate numbering above is internal to the selection script. The five winners (2015, 2010, 2016, 2017, 2018) are renumbered **`1`–`5`** for use in the model, giving the mapping in the next table. The set captures the extreme of every feature: highest and lowest wind (2015, 2010), highest and lowest solar (2018, 2017), coldest and hottest (2010, 2018), and best and worst dunkelflaute (2015, 2010). The rejected years are genuinely interior — 2019 and 2011, for instance, sit between years already chosen rather than extending the set.

| Label | Source year (ERA5) | Solar CF | Wind CF | Dunkelflaute | Mean HDD | Mean CDD | Role in the uncertainty set |
|---|---|---|---|---|---|---|---|
| 1 | 2015 | 0.180 | **0.280** | **3.0%** | 5.56 | 0.10 | Benign high-wind reference: best wind year, fewest low-renewable days |
| 2 | 2010 | 0.176 | **0.192** | **8.2%** | **7.45** | 0.10 | Cold low-wind stress year: coldest winter, worst wind, 2.7× the dunkelflaute days of label 1 |
| 3 | 2016 | 0.179 | 0.221 | 4.7% | 5.83 | 0.11 | Mid-range year: moderate wind and solar, mild winter |
| 4 | 2017 | **0.173** | 0.246 | 4.7% | 5.51 | 0.08 | High-wind / low-solar year |
| 5 | 2018 | **0.193** | 0.239 | 4.4% | 5.57 | **0.22** | Hot high-solar year: best solar CF, by far the hottest summer |

Label `1` is also the **reference** year: it anchors the NL calibration, and `data.yaml` capacities and costs are NL **2025** values (`General.base_year`). **Dunkelflaute** here means the share of **days** whose *daily-mean* wind CF is below 0.10 **and** daily-mean solar CF below 0.05 — a sustained low-renewable day, not an isolated calm hour. Exact figures, medoid days, and weights are written to `Input/weather_scenario_summary.json`.

The set spans the two directions that matter for a VRES-plus-electrolysis system. The **wind/scarcity axis** runs from label 1 (benign: wind CF 0.28, dunkelflaute 3.0%) to label 2 (stressed: 0.19 and 8.2%) — a 32% swing in wind energy combined with nearly three times as many sustained low-renewable days, which is where an electrolyser's utilisation and a risk-averse agent's tail loss are decided. The **temperature axis** runs from label 2's cold winter to label 5's summer, and through the demand model below this moves the *level* of load, not just its shape, so the stressed weather year is also the high-demand year.

#### Raw weather data and capacity-factor conversion

Hourly weather is fetched from the **[Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)** (ERA5 reanalysis) at **52.09°N, 5.12°E** (central Netherlands). Three variables are used:

| Variable | Conversion to model column | Notes |
|---|---|---|
| `shortwave_radiation` (W/m²) | **SOLAR** | Hourly PV capacity factor = min(1, GHI / 1000 W/m²) |
| `wind_speed_100m` (km/h) | **WIND** | Standard turbine power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s), cubic between cut-in and rated |
| `temperature_2m` (°C) | **LOAD_E** | Heating/cooling degree-day model on an NL diurnal/seasonal shape (below) |

**Common calibration across all years.** The raw ERA5-to-CF conversion is generic and does not reproduce the NL fleet's real capacity factors (turbine hub heights, PV orientation and losses, siting). Two multiplicative constants are therefore derived **once** by bisection on the reference year, so that *its* annual means hit the NL CBS 2024 fleet averages of **18.2% solar CF and 28.0% wind CF** `[CBS-RE]`:

$$
m_{\mathrm{solar}} = 1.4707, \qquad m_{\mathrm{wind}} = 1.5711
$$

and the **same two constants are then applied to all five years** (with a clamp to `[0, 1]`). This is the important methodological point: no year is individually rescaled to a target. Inter-year differences in the table above are genuine ERA5 weather differences seen through one fixed NL fleet, which is what a scenario set should represent. Rescaling each year to its own annual target would have erased the very spread the scenarios exist to capture — every year would have shown 18%/28% by construction, leaving only shape differences and no meaningful energy risk.

#### Temperature-coupled electricity demand

Electricity demand is not a fixed shape reused across weather years — that would let a cold, still year look benign on the demand side, which is exactly backwards for a system where scarcity and demand peaks coincide. `LOAD_E` is generated per year from the same ERA5 temperature series that drives the capacity factors, so **both the shape and the level of demand move with the weather**.

The model is **multiplicative** in four factors (`nl_load_raw` in the driver): a base activity profile modulated by day type, season, and temperature.

$$
\mathrm{load}_h \;=\; \underbrace{s_{\mathrm{hod}}(h)}_{\text{hour of day}} \;\cdot\; \underbrace{f_{\mathrm{dow}}(d)}_{\text{weekday/weekend}} \;\cdot\; \underbrace{\left[1 + a_{\ell}\cos\!\left(\tfrac{2\pi (\mathrm{doy}-15)}{365}\right)\right]}_{\text{non-thermal seasonal}} \;\cdot\; \underbrace{\left[1 + a_{H}\,\mathrm{HDD}_h + a_{C}\,\mathrm{CDD}_h\right]}_{\text{thermal response}}
$$

$$
s_{\mathrm{hod}}(h) = 0.58 + 0.18\,e^{-\frac{1}{2}\left(\frac{h-8}{3.5}\right)^{2}} + 0.30\,e^{-\frac{1}{2}\left(\frac{h-19}{2.8}\right)^{2}}
$$

with degree-hours $\mathrm{HDD}_h = \max(0,\, T^{H}_{\mathrm{base}} - T_h)$ and $\mathrm{CDD}_h = \max(0,\, T_h - T^{C}_{\mathrm{base}})$.

A multiplicative thermal factor (rather than an additive offset) means the temperature response scales with the activity level, so a cold snap adds more absolute load during the evening peak than overnight — the behaviour observed in metered data, and the property that matters here because it is the *peak* that sizes capacity.

| Component | Value | Basis |
|---|---|---|
| Heating base $T^{H}_{\mathrm{base}}$ | 15.5 °C | Standard European HDD base (Eurostat degree-day convention) |
| Cooling base $T^{C}_{\mathrm{base}}$ | 22.0 °C | Standard European CDD base |
| Heating sensitivity $a_{H}$ | 0.010 /°C | NL space heating is still predominantly **gas**, so the *electrical* temperature response is modest — deliberately well below the ≈0.03–0.05 of electrically heated systems such as France or Sweden `[BF-2008]` |
| Cooling sensitivity $a_{C}$ | 0.008 /°C | Low NL air-conditioning penetration `[BF-2008]` |
| Weekend factor $f_{\mathrm{dow}}$ | 0.87 | NL weekend load ≈13% below weekday; applied on the **real calendar** weekday pattern of each source year `[ENTSOE-LOAD]` |
| Seasonal amplitude $a_{\ell}$ | 0.05 | ±5% non-thermal seasonal term (lighting, activity), peaking mid-January |
| Hour-of-day shape $s_{\mathrm{hod}}$ | base 0.58, morning bump +0.18 at 08:00, evening peak +0.30 at 19:00 | Amplitudes calibrated so the **reference year's load factor is 0.636**, matching NL 2024 (≈108.5 TWh net consumption against a 19.48 GW peak) `[CBS-EP]`, `[ENTSOE-PEAK-2024]`. A peakier shape understates annual energy for a given peak |

**Common normalisation.** All five years are divided by **one** shared constant — the maximum raw load across the whole scenario set — not by their own maxima. A per-year normalisation would force every year to peak at exactly 1.0 p.u. and silently delete the demand differences between them, which would defeat the purpose of coupling demand to weather at all. With the common divisor, the differences survive:

| Label | Peak (p.u.) | Mean (p.u.) | Load factor |
|---|---|---|---|
| 1 (2015, reference) | 0.963 | 0.659 | 0.685 |
| 2 (2010, cold) | **1.000** | **0.672** | 0.672 |
| 3 (2016) | 0.973 | 0.661 | 0.680 |
| 4 (2017) | 0.979 | 0.659 | 0.673 |
| 5 (2018, hot) | 0.987 | 0.660 | 0.669 |

(Common load normaliser across scenarios: peak raw load / **1.1246** ⇒ cold year 2 peaks at 1.0 p.u.; `PeakLoad` in `data.yaml` = 19,500 MW scales this to absolute MW.)

The cold year 2 sets the system peak and carries the highest mean load; the hot year 5 has a *lower* mean but a higher peak than the reference, because cooling load concentrates in summer afternoons. `PeakLoad` in `data.yaml` is therefore the system peak **across scenarios**, and milder years genuinely sit below it.

Gas-price scenarios do **not** shift electricity demand. The demand response to price is already endogenous through the elastic consumer (§9.3), so applying an exogenous shift as well would double-count it.

`LOAD_H` and `LOAD_EP` remain fixed normalised shapes (0.8 and 0.9) in all scenarios; absolute H₂ and EP demand is set in `data.yaml` via agent capacities and `Total_Demand`.

#### Representative-day selection (8760 h → 8 days) via RepresentativePeriodsFinder.jl

Full-year hourly profiles (365 × 24 h) are reduced to **`nReprDays = 8`** representative days using **[RepresentativePeriodsFinder.jl](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl)** (RPF) with the **clustering** method (hierarchical, medoid-based; Pineda & Morales 2018) — **not** the optimisation method. Selection is driven by a YAML config (`Input/rep_periods/config_template.yaml`) and orchestrated by `Input/rep_periods/generate_representative_days.jl`:

1. **Feature construction & normalisation**: for each clustered series (SOLAR, WIND, LOAD_E) RPF **min–max normalises** the full-year values to `[-1, 1]` and reshapes them into 365 daily vectors of 24 hourly values. The three are concatenated into one **72-element feature vector per day**, each series scaled by its configured **clustering weight** (SOLAR = WIND = 2, LOAD_E = 1) so the algorithm prioritises VRES diversity, including low-renewable days.
2. **Hierarchical clustering** (`representative_periods = 8`): RPF repeatedly merges the pair of clusters minimising the **Ward-style dissimilarity** $D_{ij} = \frac{2 n_{i} n_{j}}{n_{i} + n_{j}} \lVert \bar{x}_{i} - \bar{x}_{j} \rVert_{2}^{2}$, where $n_{i}$ is the number of days in cluster $i$ and $\bar{x}_{i}$ its centroid, until 8 clusters remain. The size factor makes merging two large clusters more costly than merging two small ones, which stops the partition collapsing onto one dominant group. After each merge the cluster's **medoid** — the actual calendar day closest to the new centroid — becomes its representative, so every representative day is a **real historical day**, never an average. Clustering needs **no MILP solver**, which is why this mode was chosen over RPF's optimisation mode.
3. **Weights**: for cluster `c`, `W[c] =` number of calendar days assigned to that cluster. Weights sum to **365** and are written to `decision_variables_short.csv` (`periods`, `weights`, `selected_periods`); the full `decision_variables.csv` lists all 365 days.
4. **Weight rebalancing** (post-processing, see below): the raw cluster-count weights are corrected so the weighted annual means reproduce the full-year annual means exactly.
5. **`ordering_variable.csv`**: 365 × 8 one-hot assignment matrix mapping each calendar day to its cluster's medoid (RPF native output; loaded by the model but **not** used in optimisation).
6. **Model export**: the driver translates RPF's `resulting_profiles.csv` into `timeseries_<label>.csv` (renaming the series columns and appending the constant `LOAD_H = 0.8`, `LOAD_EP = 0.9` shapes), and copies the decision-variable and ordering files into `Input/output_<label>/`.

Representative days are written in **ascending medoid-calendar-day order**, so row blocks 1–24, 25–48, … correspond to `jd = 1, 2, …, 8` and align row-for-row with `decision_variables_short.csv` (whose `jd`-th weight becomes `W[jd, jy]`).

**Why the weights are rebalanced.** RPF selects excellent *days*; the issue is what its cluster-count *weights* imply for annual energy. A medoid is the most central day of its cluster, but "central in 24-dimensional shape space" is not the same as "average in daily mean", and with only 8 clusters spanning 365 days the two diverge. Measured on the actual outputs, weighting RPF's medoids by raw cluster counts reproduces the full-year annual means with these errors:

| Label | Solar CF error | Wind CF error | Mean load error |
|---|---|---|---|
| 1 (2015) | **−13.3%** | −5.1% | −0.4% |
| 2 (2010) | −3.2% | −2.2% | −2.4% |
| 3 (2016) | −2.9% | **−12.1%** | −1.5% |
| 4 (2017) | **−14.3%** | −9.2% | −0.5% |
| 5 (2018) | −3.3% | −4.7% | +0.4% |

Two things are wrong here, and the second is the serious one.

The **level** error is large: understating solar output by 13–14% in two of the five years would materially distort the solar investment decision, since annual energy is what pays back capacity.

The **inconsistency across years** is worse. The errors are not a common bias that would partly cancel when comparing scenarios — they range from −2.9% to −14.3% on solar and −2.2% to −12.1% on wind, essentially at random. That corrupts precisely the inter-year differences the scenario set exists to represent, to the point of **reordering the years**:

- On solar, the true ranking is 5 > **1** > 3 > 2 > 4. Under raw weights it becomes 5 > 3 > 2 > **1** > 4 — label 1 falls from the second-sunniest year to the fourth.
- On wind, labels 4 and 5 swap: truly 4 (0.246) is windier than 5 (0.239), but raw weights make 5 (0.228) look windier than 4 (0.223).
- On demand, the cold stress year 2 has the highest true mean load (0.672) — under raw weights it drops to **fourth**, behind label 5, which inverts the entire point of the temperature-coupled demand model.

A risk-aware agent choosing capacity against a CVaR tail would be optimising against the wrong ordering of good and bad years. The driver therefore solves, per year, the minimum-adjustment problem

$$
\min_{w}\;\lVert w - w^{\mathrm{raw}} \rVert_2^2
\quad \text{s.t.} \quad
\sum_{d} w_d = 365,\qquad
\frac{1}{365}\sum_{d} w_d \bar{x}_{d,s} = \bar{X}_s \;\;\forall s \in \lbrace \text{SOLAR}, \text{WIND}, \text{LOAD\_E} \rbrace,\qquad
w_d \ge 1
$$

where $\bar{x}_{d,s}$ is representative day $d$'s daily mean of series $s$ and $\bar{X}_s$ the full-year annual mean. The objective keeps the adjustment as small as possible, so RPF's weights are the starting point and are moved only as far as the constraints require.

Naive least squares on the equality constraints alone produced **negative** weights, so the driver uses **Dykstra's alternating-projection algorithm** `[DYKSTRA-1983]`, projecting in turn onto the affine subspace (the four equalities) and onto the box $w \ge 1$. Dykstra's correction terms are what make the iteration converge to the true projection onto the intersection rather than to an arbitrary feasible point — plain alternating projection (von Neumann) would not, and a single active-set pass fails outright on label 5. If no feasible weighting exists the driver **keeps RPF's original weights** and emits a warning rather than silently shipping an infeasible result; `weights_rebalanced` in the summary JSON records which path was taken (currently `true` for all five years).

After rebalancing, `repr_solar_cf`, `repr_wind_cf`, and `repr_mean_load` in `Input/weather_scenario_summary.json` match their full-year counterparts to within 0.01% for all five years, and the year ordering on every series is restored.

**Cost of the correction (honest limitation).** Weights become **fractional day-equivalents** rather than day counts, and for two labels the redistribution is substantial:

| Label | RPF raw weights | Rebalanced | Weights at the 1-day floor | Largest shift |
|---|---|---|---|---|
| 1 | 10, 20, 71, 177, 12, 35, 31, 9 | 64.9, 33.8, 102.1, 137.4, 1.0, 4.9, 19.8, 1.0 | 2 of 8 | 54.9 days |
| 2 | 23, 28, 27, 62, 83, 59, 70, 13 | 1.0, 8.9, 1.0, 91.4, 109.8, 13.3, 81.7, 57.9 | 2 of 8 | 45.7 days |
| 3 | 19, 13, 115, 48, 19, 42, 65, 44 | 44.1, 7.9, 116.2, 52.5, 33.0, 35.3, 55.9, 20.0 | 0 | 25.1 days |
| 4 | 20, 14, 81, 113, 26, 24, 66, 21 | 24.3, 20.0, 60.6, 107.6, 58.2, 26.3, 34.7, 33.3 | 0 | 32.2 days |
| 5 | 102, 32, 67, 51, 49, 25, 26, 13 | 112.4, 28.5, 64.0, 49.3, 39.9, 27.6, 29.5, 13.8 | 0 | 10.4 days |

For labels 1 and 2 two medoids are pushed to the floor, so those years are effectively carried by ~6 heavily weighted days plus 2 near-vestigial ones. This trades some **duration-curve** fidelity for exact **annual-mean** fidelity. That is the right trade for this model — investment is driven by annual energy and by the scenario ranking, not by the fine structure of the load-duration curve, and there is no storage state to carry across days — but it is a genuine limitation to be aware of. Labels 3–5 need only mild adjustment and keep all eight days materially weighted.

If duration-curve fidelity ever becomes load-bearing (e.g. if storage or unit commitment is added), the cleaner fix is to **raise `nReprDays`**: more clusters shrink the raw bias at the source, so the rebalancing has less work to do. The cost is proportional solve time, since the model scales linearly in `nTimesteps × nReprDays × nYears`.

**Why RPF is vendored, and what was patched.** RPF is not installable from its upstream registry on current Julia: its `Project.toml` declares `julia = "1.3 - 1.6.7"`, so `Pkg.add` fails outright with a compat error on Julia 1.12. A copy therefore lives at `Input/rep_periods/RepresentativePeriodsFinder/` and is `dev`-ed into an **isolated sub-environment** (`Input/rep_periods/Project.toml`), keeping RPF's older dependency bounds from constraining the main model environment. Two minimal patches were applied, both recorded in the vendored source:

| Patch | File | Reason |
|---|---|---|
| `julia` compat widened `1.3 - 1.6.7` → `1.3 - 1.12` | `RepresentativePeriodsFinder/Project.toml` | Upstream bound predates current Julia; the package code itself is compatible |
| `_rpf_index_positions` shim replaces `ta[timestamps]` | `src/util/get.jl` | TimeSeries.jl ≥ 0.24 removed indexing a `TimeArray` by a vector of `DateTime`; the shim maps timestamps to integer positions and slices by index |

Neither patch touches the clustering algorithm, so results are those of upstream RPF 0.4.4.

**Regenerating the inputs.** One-time setup, then a full regeneration:

```bash
julia Input/rep_periods/setup_env.jl                                                  # instantiate the sub-environment (once)
julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl   # regenerate all weather labels
```

Pass specific labels (e.g. `... generate_representative_days.jl 1 5`) to regenerate a subset. Artefacts:

| Path | Contents |
|---|---|
| `weather_cache/open_meteo_<year>.json` | Raw ERA5 API responses for **all ten** candidate years, so the diversity search can be re-run without re-fetching |
| `weather_full/timeseries_full_<label>.csv` | 8760-hour calibrated series fed to RPF |
| `results/<label>/config_used.yaml` | The exact config RPF ran, for reproducibility |
| `results/<label>/*.csv` | RPF's **native, un-rebalanced** outputs — the reference against which the weight correction can always be audited |
| `Input/timeseries_<label>.csv`, `Input/output_<label>/` | Final model inputs (rebalanced weights) |
| `Input/_legacy_inputs_backup/` | One-time backup of the pre-RPF inputs |

Because `results/<label>/` keeps RPF's original weights while `Input/output_<label>/` holds the corrected ones, the effect of the rebalancing step is always reproducible from the repository — the tables above were computed by comparing the two.

#### Timeseries file layout and availability factors

Each `Input/timeseries_<label>.csv` contains **`nReprDays × nTimesteps = 8 × 24 = 192`** rows with columns:

`Time`, `SOLAR`, `LOAD_E`, `LOAD_H`, `LOAD_EP`, `WIND`

All renewable and load columns are **normalised capacity factors or shapes in [0, 1]** (peak load = 1.0).

In the Julia model (`define_power_parameters.jl`, `define_common_parameters.jl`):

| CSV column | Model array | Agent | Usage |
|---|---|---|---|
| `SOLAR` | `AF[jh, jd, jy]` | Solar VRES | Hourly **availability factor** × `Capacity` → max generation (MW) |
| `WIND` | `AF[jh, jd, jy]` | Wind VRES | Same |
| `LOAD_E` | `LOAD_E[jh, jd, jy]` | Consumer, GC demand | Normalised shape; × `PeakLoad` → demand (MW) |
| `LOAD_H` | `LOAD_H[jh, jd, jy]` | H₂ market | Normalised H₂ demand shape |
| `LOAD_EP` | `LOAD_EP[jh, jd, jy]` | EP market | × `Total_Demand` → ammonia demand (MW_EP) |

**Annual scaling**: any per-hour, per-representative-day cost or revenue term is multiplied by **`W[jd, jy]`** when aggregating to a scenario-year total, so an 8-day optimisation approximates a weighted 365-day year.

Conventional generation uses **`AF ≡ 1`** (fully dispatchable); only VRES availability varies with weather.

### 9.8 Gas Prices and the 15-Scenario Grid

Weather is one uncertainty axis; the **natural-gas price** is the other. Together they define the scenario set every risk-aware agent optimises over.

#### The grid

```yaml
Scenarios:
  weather_years: [1, 2, 3, 4, 5]
  gas_price_multipliers: [1.00, 1.10, 1.20]
```

`Source/define_scenarios.jl` expands this into the flat scenario index used everywhere else:

$$
n_{\mathrm{years}} = 5 \times 3 = 15,
\qquad
jy = (i_{\mathrm{weather}} - 1)\cdot 3 + i_{\mathrm{gas}}
$$

Gas varies **fastest**, so scenarios group by weather year: `jy = 1,2,3` are weather year 1 at 1.0/1.1/1.2 × gas, `jy = 4,5,6` are weather year 2, and so on to `jy = 13,14,15` for weather year 5. All 15 are equiprobable, `P[jy] = 1/15`. `build_scenario_grid` returns two lookups: `years[jy]` (the weather **file label**, so `timeseries_<label>.csv` is read once per label and reused by the three gas variants) and `gas_multiplier[jy]`. Every entry point prints the resulting grid at start-up, so a run's log always records the uncertainty set it optimised over:

```
Scenario grid: 5 weather year(s) [1, 2, 3, 4, 5] x 3 gas level(s) [1.0, 1.1, 1.2] = 15 scenarios (equal probability 0.0667 each)
```

If the `Scenarios` block is absent, the grid degenerates to weather years `1..nScenarioYears` at a single multiplier of 1.0, reproducing the pre-grid behaviour.

#### One gas price, two markets

The gas price is an **explicit model input**, not a number baked into each agent's cost. `Fuel` in `data.yaml` is the single source of truth:

```yaml
Fuel:
  GasPrice:  34.40     # €/MWh_th — TTF 2024 annual average
  CoalPrice: 12.54     # €/MWh_th — API2 2024 average
  BiomassPrice: 33.00  # €/MWh_th — NW-Europe industrial wood pellet 2024
  CO2Price:  64.79     # €/tCO₂ — EU-ETS EUA 2024 annual average
  GasEmissionFactor:     0.2016   # tCO₂/MWh_th
  CoalEmissionFactor:    0.3406
  BiomassEmissionFactor: 0.0
```

Both gas-consuming agents derive their variable cost from it, so a single price shock moves power and ammonia **together and consistently** — which is the physically correct behaviour and was not true of the previous hard-coded costs.

**Conventional generator** — each flat plant (or legacy merit-order stage) names a technology and its efficiency, and its short-run marginal cost is computed as

$$
\mathrm{SRMC} \;=\; \frac{p_{\mathrm{fuel}}}{\eta} \;+\; \frac{\mathrm{EF}}{\eta}\, p_{\mathrm{CO_2}} \;+\; \mathrm{VOM}
$$

**Grey ammonia offtaker** — SMR-based production, costed per MWh of end product:

$$
\mathrm{MC} \;=\; \mathrm{GasIntensity}\cdot p_{\mathrm{gas}} \;+\; \mathrm{CO_2 Intensity}\cdot p_{\mathrm{CO_2}} \;+\; \mathrm{VOM}
$$

with `GasIntensity = 1.720` MWh_th/MWh_EP (32 GJ_LHV per t NH₃ ÷ 5.167 MWh/t) and `CO2Intensity = 0.348` tCO₂/MWh_EP (1.8 tCO₂/t NH₃, process **plus** fuel). The CO₂ cost is charged in full: NL grey ammonia's process CO₂ is ETS-covered, and free allocation is a lump-sum transfer that does not change the marginal decision.

Only **gas** carries the scenario multiplier. The +10%/+20% scenarios represent a gas-market shock, so coal, biomass, and the ETS price are held fixed — which also means the shock genuinely reorders the merit order rather than shifting it uniformly.

| Scenario gas level | Gas price (€/MWh_th) | CCGT | Coal | Biomass | OCGT (benchmark)* | Grey NH₃ MC |
|---|---|---|---|---|---|---|
| 1.00 × | 34.40 | **83.3** | 85.4 | 89.8 | 127.9 | **85.7** €/MWh_EP (443 €/t) |
| 1.10 × | 37.84 | **89.3** | 85.4 | 89.8 | 136.9 | 91.6 €/MWh_EP (473 €/t) |
| 1.20 × | 41.28 | **95.2** | 85.4 | 89.8 | 146.0 | 97.6 €/MWh_EP (504 €/t) |

(Power values in €/MWh_e.) *OCGT is not a separate agent; the column is the peaking SRMC used for contract `ng_elec` benchmarks in `contract_strike.jl`. At 1.00 × gas, CCGT is cheapest; at 1.10 × gas, CCGT exceeds coal (merit order inverts); at 1.20 ×, CCGT exceeds both coal and biomass.

#### Why the 2024 fuel anchor

The gas price is anchored to the **TTF 2024 annual average of 34.40 €/MWh_th**, consistent with `General.base_year = 2025` (capacities) and with the ETS and fuel prices around it. Deriving both power and ammonia costs from one `Fuel` block keeps them coherent: the resulting grey cost of 443 €/t NH₃ sits near the 2024 NW Europe ammonia range ($528–581/t) `[OCI-NH3-2024]`, and the CCGT SRMC of ~83 €/MWh_e is a plausible gas-linked benchmark (2024 NL day-ahead averaged 77 €/MWh as VRES and imports often set price) `[NL-DA-2024]`.

#### What this changes downstream

Because costs now vary by scenario, the affected parameters became **scenario-indexed** rather than scalar, in both the ADMM and the planner paths:

| Parameter | Shape | Consumed by |
|---|---|---|
| Conventional `MarginalCostByYear` | `nYears` | `build_power_agent.jl`, `solve_power_agent.jl`, `compute_agent_objective.jl`, `compute_social_risk_metrics.jl` |
| `ConvStageBaseCost`, `ConvStageSlope` (legacy) | `3 × nYears` | as above, when `StageTechnologies` is set |
| `ConvFinalMarginalCost` (legacy) | `nYears` | as above |
| Grey offtaker `MarginalCostByYear` | `nYears` | `build_offtaker_agent.jl`, `solve_offtaker_agent.jl`, `compute_agent_objective.jl`, `compute_social_risk_metrics.jl` |

Flat `Technology` plants and legacy staged stacks can coexist in one model. The legacy scalar forms (`StageBaseCosts`, `FinalMarginalCost`, a scalar `MarginalCost`) are still honoured if the derived inputs are absent.

#### Choosing `beta` with 15 scenarios

CVaR at level $\beta$ averages the worst $(1-\beta)$ share of scenarios. With 15 equiprobable scenarios, one scenario is 6.7% of the mass, so the default `beta = 0.95` requests a 5% tail that is **narrower than a single scenario** — CVaR then collapses onto the single worst outcome. That is well defined but coarse, and the risk-metrics reporter now prints an explicit warning when it happens, together with the values that give a tail of `k` whole scenarios:

$$
\beta = 1 - \frac{k}{15} \quad\Rightarrow\quad \beta = 0.933\ (k{=}1),\; 0.867\ (k{=}2),\; 0.800\ (k{=}3),\; 0.667\ (k{=}5)
$$

`beta = 0.8` (worst 3 of 15) is a reasonable default for risk-averse runs.

## 10. Project Structure

```
Now/
├── market_exposure.jl          # Entry point: distributed ADMM simulation (5 markets)
├── me_pap.jl                    # Entry point: ADMM + PPA/HPA (pay-as-produced HPA)
├── me_top.jl                    # Entry point: ADMM + PPA/HPA (take-or-pay HPA)
├── me_sop.jl                    # Entry point: ADMM + PPA/HPA (send-or-pay HPA)
├── social_planner.jl           # Entry point: centralized benchmark
├── green_h2_social_planner.jl  # Entry point: electrolyzer + green offtaker coalition (§4.11)
├── green_social_planner.jl     # Entry point: VRES + electrolyzer + green offtaker coalition (§4.11)
├── Project.toml                # Julia project dependencies
├── Manifest.toml               # Julia dependency lock file
├── DOCUMENTATION.md            # This file
├── README.md                   # Quick-start guide (installation, running)
│
├── visualization/              # SP-vs-ME comparison plots (Python)
│   ├── visualize_results.py
│   ├── visualize_results.ipynb
│   └── figures/
│   └── data.yaml               # All configuration: agents, markets, ADMM settings
│
├── Input/
│   ├── timeseries_1.csv        # Representative-day hourly profiles (SOLAR, LOAD_E, LOAD_H, LOAD_EP, WIND)
│   ├── timeseries_2.csv        # (one per weather label 1..5; columns are normalized 0–1 profiles)
│   ├── ...
│   ├── output_1/
│   │   ├── decision_variables_short.csv   # Representative days: periods, weights, selected_periods
│   │   ├── decision_variables.csv         # All 365 days (weight 0 except medoids)
│   │   └── ordering_variable.csv          # 365×8 one-hot day→medoid assignment matrix
│   ├── output_2/
│   │   └── ...
│   ├── weather_scenario_summary.json      # Per-scenario source year, roles, annual CFs, medoid days/weights
│   └── rep_periods/            # Representative-day generation (RepresentativePeriodsFinder.jl)
│       ├── config_template.yaml            # RPF clustering config (hierarchical, 8 rep days)
│       ├── generate_representative_days.jl # Driver: ERA5 → CFs → RPF clustering → model inputs
│       ├── setup_env.jl                    # Instantiates the RPF sub-environment
│       ├── Project.toml / Manifest.toml    # Isolated env (keeps RPF's deps out of the model env)
│       ├── RepresentativePeriodsFinder/    # Vendored, compat-patched RPF package
│       ├── weather_cache/                  # Cached raw ERA5 API responses
│       ├── weather_full/                   # Full-year (8760 h) clustering inputs
│       └── results/<label>/                # RPF native outputs per scenario
│
├── Source/
│   ├── define_scenarios.jl               # Builds the weather × gas scenario grid; derives fuel-linked costs
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
│   ├── update_rho.jl                 # Adaptive ρ update (Boyd residual balancing, §6.3)
│   ├── save_results.jl               # Write market-exposure CSV outputs
│   ├── ADMM_contracts.jl             # ADMM loop with PPA + HPA pools
│   ├── ADMM_subroutine_contracts.jl  # Per-agent step with PPA/HPA g_bar/λ/ρ
│   ├── update_rho_contracts.jl      # Adaptive ρ on contract energy + cap-alignment residuals
│   ├── contract_capacity.jl         # Shared C init, fix caps, bargaining update, residuals
│   ├── contract_strike.jl           # Benchmarks, provisional K, finalize_contract_terms!
│   ├── contract_settlement.jl       # PaP / ToP / SoP payment terms
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
│   ├── Agent_Summary.csv             # Per-agent net positions, capacity & objective (same schema as SP)
│   ├── Agent_Objectives_Per_Timestep.csv
│   ├── Market_Prices.csv             # Final ADMM prices λ for all five markets
│   ├── Capacity_Consensus.csv        # Per-iteration capacity equality split (§6.4)
│   ├── Offtaker_GC_Diagnostics.csv   # GC compliance per offtaker
│   ├── H2_Producer_Diagnostics.csv   # H₂ GC-to-production ratio
│   ├── Risk_Metrics.csv              # Welfare / CVaR summary vs SP + demand/rest split
│   ├── Social_Welfare_Per_Year.csv   # Per-scenario welfare, loss, demand, ex-demand
│   ├── Welfare_By_Group_Per_Year.csv # Planner welfare by group (no transfers)
│   ├── Welfare_By_Agent_Per_Year.csv
│   ├── Private_CVaR_By_Agent.csv     # Per-agent private CVaR (written only when γ < 1)
│   └── TimerOutput.yaml              # Profiling data
│
├── me_pap_results/              # Output from me_pap.jl
├── me_top_results/              # Output from me_top.jl
├── me_sop_results/              # Output from me_sop.jl
│   ├── ADMM_Convergence.csv          # Same as market_exposure + PPA/HPA energy + shared-cap columns
│   ├── ADMM_Diagnostics.csv          # Same + PPA/HPA energy + shared-C motion
│   ├── Electricity_Market_History.csv
│   ├── Hydrogen_Market_History.csv
│   ├── Electricity_GC_Market_History.csv
│   ├── H2_GC_Market_History.csv
│   ├── End_Product_Market_History.csv
│   ├── Agent_Summary.csv             # Same structure as market_exposure (no contract columns)
│   ├── Market_Prices.csv             # Same + per-submarket PPA/HPA price columns
│   ├── Capacity_Consensus.csv
│   ├── Risk_Metrics.csv
│   ├── Social_Welfare_Per_Year.csv
│   ├── Welfare_By_Group_Per_Year.csv
│   ├── Welfare_By_Agent_Per_Year.csv
│   ├── PPAs.csv                      # Per-VRES PPA summary
│   ├── HPAs.csv                      # Per-GreenProducer HPA summary
│   └── Green_Agents_Detail.csv       # Detailed PPA breakdown for VRES and GreenProducer
│
├── green_h2_social_planner_results/  # Output from green_h2_social_planner.jl (ME layout)
├── green_social_planner_results/     # Output from green_social_planner.jl (ME layout)
│
└── social_planner_results/           # Output from social_planner.jl
    ├── Market_Prices.csv             # Equilibrium prices (balance duals / (W × μ); §7.2)
    ├── Agent_Summary.csv             # Per-agent net positions, capacity & objective (same schema as ME)
    ├── SP_Capacities.csv             # VRES / electrolyzer / green-offtaker capacity & investment
    ├── SP_Primal_Quantities.csv      # Full primal allocation per (jy, jd, jh)
    ├── Agent_Objectives_Per_Timestep.csv
    ├── Risk_Metrics.csv              # Written by compute_social_risk_metrics.jl
    ├── Social_Welfare_Per_Year.csv
    ├── Welfare_By_Group_Per_Year.csv
    └── Welfare_By_Agent_Per_Year.csv
```

---

## 11. File Reference

### 11.1 Runner Scripts

| File | Purpose |
|---|---|
| `market_exposure.jl` | Entry point for distributed ADMM. Sections 1–13: env, packages, dirs, source loading, data loading, results folder, agent init, market params, agent params, build models, run ADMM, save results. |
| `me_pap.jl` | ME + PPA (PaP) + HPA (PaP). Self-contained entry point; contract modules under `Source/`. Outputs to `me_pap_results/`. |
| `me_top.jl` | Same as `me_pap.jl` but HPA volume = take-or-pay. Outputs to `me_top_results/`. |
| `me_sop.jl` | Same as `me_pap.jl` but HPA volume = send-or-pay. Outputs to `me_sop_results/`. |
| `social_planner.jl` | Entry point for centralised benchmark. Sections 1–12: same structure as market_exposure but builds a single planner model instead of per-agent models + ADMM loop. Section 11 solves the planner as a convex QCP with IPOPT and requires direct dual availability. |

### 11.2 Parameter Definition Files

| File | Role |
|---|---|
| `define_scenarios.jl` | `build_scenario_grid` expands the `Scenarios` block into the flat index `jy = 1..nYears` with lookups `years[jy]` (weather file label) and `gas_multiplier[jy]`. `fuel_price_ef` and `thermal_srmc` turn the `Fuel` block into per-scenario commodity prices and thermal SRMCs; `describe_scenario_grid` prints the grid at start-up. |
| `define_common_parameters.jl` | Creates `mod.ext` dictionaries (sets, parameters, timeseries, variables, constraints, expressions). Fills JH/JD/JY, W, P, γ, β. Determines market participation from agent type. Pre-allocates ADMM placeholder arrays. |
| `define_power_parameters.jl` | VRES: capacity, AF profile. Conventional: capacity, AF=1; flat plant via `Technology` → `MarginalCostByYear`, or legacy 3-stage cost curve (`ConvStageCap`, `ConvStageBaseCost`, `ConvStageSlope`) from `StageTechnologies` + `Fuel`. Consumer: PeakLoad, LOAD_E profile, A_E, B_E. |
| `define_H2_parameters.jl` | Electrolyzer: Capacity_Electrolyzer, Capacity_H2_Output, SpecificConsumption, OperationalCost, η_elec_H2. |
| `define_offtaker_parameters.jl` | Copies all keys from agent block; sets gamma_GC = 0.42 (regulatory mandate). For the grey offtaker, derives `MarginalCostByYear` (length `nYears`) from `GasIntensity`, `CO2Intensity`, `VariableOM`, and the `Fuel` block; the scalar `MarginalCost` is retained at the base-scenario value for backward compatibility. |
| `define_elec_GC_demand_parameters.jl` | PeakLoad, Load_Column, A_GC, B_GC, LOAD_GC timeseries. |
| `define_EP_demand_parameters.jl` | Placeholder; copies EP_Demand block if present. |
| `define_*_market_parameters.jl` | Each market: name, initial_price, rho_initial, prices list. EP market also builds 3D demand array D_EP. |
| `define_results.jl` | Initialises results["λ"], per-agent quantity buffers, ADMM ρ lists, Imbalances, PriceHistory, ImbalanceMean, Residuals, Tolerance. |

### 11.3 Model Building Files

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

### 11.4 Solve Files

| File | Role |
|---|---|
| `solve_power_agent.jl` | Rebuilds objective with iteration-specific λ, ḡ, ρ. For VRES: recomputes loss expressions with iteration-specific λ, deletes and re-adds CVaR shortfall/linking constraints. For conventional: flat `MarginalCostByYear[jy]×g` or legacy 3-stage convex cost using scenario-indexed `stage_base[s, jy]` / `stage_slope[s, jy]`. Calls `optimize!`. |
| `solve_H2_agent.jl` | Rebuilds objective with iteration-specific λ, ḡ, ρ. Recomputes loss expressions with iteration-specific λ (4-market), deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_offtaker_agent.jl` | Rebuilds objective for green/grey/importer. For GreenOfftaker: recomputes loss expressions with iteration-specific λ, deletes and re-adds CVaR shortfall/linking constraints. Calls `optimize!`. |
| `solve_elec_GC_demand_agent.jl` | Rebuilds utility − expenditure + ADMM penalty; calls `optimize!`. |
| `solve_EP_demand_agent.jl` | Placeholder; just calls `optimize!`. |

### 11.5 ADMM Files

| File | Role |
|---|---|
| `ADMM.jl` | Main loop: iterate agents → imbalances → primal/dual residuals → scale-aware price update → adaptive ρ update → convergence check. Progress bar + summary printout. |
| `ADMM_subroutine.jl` | Per-agent step: update g_bar/λ/ρ on model → dispatch to solve_* → extract & record quantities. H₂-GC remains hourly (full 3D), consistent with other markets. |
| `ADMM_contracts.jl` | Same ADMM flow as `ADMM.jl` plus PPA + HPA energy/capacity consensuses and λ_ppa/λ_hpa updates. |
| `ADMM_subroutine_contracts.jl` | Per-agent step with PPA/HPA g_bar, λ, ρ updates and extraction. Dispatches to contracts solvers for power/H2/offtaker contract agents. |
| `update_rho.jl` | Minimal residual-balancing ρ update with market-specific rates/caps; includes per-agent capacity ρ update for the `x_cap = z_cap` split. |
| `update_rho_contracts.jl` | Residual-balancing ρ update for standard markets, per-agent capacity split, and ppa/hpa energy/capacity consensuses. |

### 11.6 Save Files

| File | Role |
|---|---|
| `save_results.jl` | Writes: ADMM_Convergence.csv, ADMM_Diagnostics.csv, Capacity_Consensus.csv, per-market history CSVs, Agent_Summary.csv, Agent_Objectives_Per_Timestep.csv, Offtaker_GC_Diagnostics.csv, H2_Producer_Diagnostics.csv, Market_Prices.csv. |
| `save_results_contracts.jl` | Writes the same major ADMM outputs as save_results (with PPA/HPA columns) plus: PPAs.csv, HPAs.csv, Green_Agents_Detail.csv. Agent_Summary matches market_exposure structure (no explicit contract columns). |
| `compute_social_risk_metrics.jl` | Post-processing: social CVaR, \(\mathbb{E}[\mathrm{SW}]\), demand vs ex-demand split, private CVaR sum, SP comparison; writes Risk_Metrics.csv and welfare decomposition CSVs. |
| `save_social_planner_results.jl` | Called after the direct QCP solve, with duals available. Writes: Market_Prices.csv, SP_Primal_Quantities.csv, SP_Capacities.csv, Agent_Summary.csv, Agent_Objectives_Per_Timestep.csv. (Risk_Metrics.csv and Social_Welfare_Per_Year.csv are written separately by `compute_social_risk_metrics.jl`, which also prints the risk summary.) |

---

## 12. Output Files

### 12.1 Market Exposure Results

| File | Contents |
|---|---|
| `ADMM_Convergence.csv` | Columns: `iter`, `{market}_primal`, `{market}_dual` for each of the 5 markets, plus `cap_primal` / `cap_dual` (aggregate L2 over agents) and **per-agent** `cap_primal_<m>` / `cap_dual_<m>` columns from the equality-split capacity ADMM (§6.4). One row per ADMM iteration. Used for convergence plots. |
| `ADMM_Diagnostics.csv` | Columns: `iter`, `{market}_rho`, `{market}_price_mean`, `{market}_imb_mean` for each flow market, plus per-agent `cap_rho_<m>` columns (one per cap-owning agent). |
| `Capacity_Consensus.csv` | Per-iteration, per-agent, per-year snapshot of the capacity equality split. Columns: `iter`, `AgentID`, `jy`, `x_cap`, `z_cap`, `lambda_cap`, `rho_cap`, `primal_local`, `dual_local`. Use this to identify the agent / year that gates capacity convergence; analogous to `{Market}_Market_History.csv` but at the (iter, agent, year) granularity that the per-agent split naturally produces. See §6.4 for the formal model. |
| `{Market}_Market_History.csv` | Per-market CSV with: `iter`, `rho`, `price_mean`, `imb_mean`, `primal_res`, `dual_res`. |
| `Agent_Summary.csv` | One row per agent — the main per-agent result table. Columns: `AgentID`, `Group`, `Type`, the five net positions `elec_net_sum`, `H2_net_sum`, `elec_GC_net_sum`, `H2_GC_net_sum`, `EP_net_sum` (summed over all hours, representative days and scenarios at the final iteration; **+ = sold into the market, − = bought**), plus `Capacity_Final_MW`, `Investment_Total_MW`, `Objective_Value`. The social planner writes the **same schema**, which is what makes the SP-vs-ME comparison in `visualization/` a direct column-by-column merge. |
| `Agent_Objectives_Per_Timestep.csv` | Per-agent objective contribution resolved to each (jy, jd, jh) slot; use it to see *when* an agent earns or loses money rather than only the annual total. |
| `Market_Prices.csv` | Final ADMM prices $\lambda$ for all five markets, one row per (jy, jd, jh). Same schema as the social planner's file (§12.3), so the two can be diffed directly. |
| `Offtaker_GC_Diagnostics.csv` | Columns: `AgentID`, `Type`, `EP_total`, `H2_in_total` (H₂ basis the mandate is measured on: bought H₂ for green, `ep/γ_NH3` for grey), `H2_GC_total`, `GC_share` (= GCs / H₂ basis), `GC_mandate` (= `γ_GC`), `GC_slack` (= share − mandate; > 0 ⇒ compliant). |
| `H2_Producer_Diagnostics.csv` | Columns: `AgentID`, `H2_total`, `H2_GC_total`, `GC_per_H2`. |
| `Risk_Metrics.csv` | `expected_social_welfare`, `social_CVaR`, `sum_private_CVaR`, gap vs SP, demand vs ex-demand split — §7.6. |
| `Social_Welfare_Per_Year.csv` | Per-year aggregate welfare, loss, `welfare_demand`, `welfare_ex_demand`. |
| `Welfare_By_Group_Per_Year.csv` | Per-year planner welfare by group (elec/GC demand vs VRES, conventional, H₂, offtakers). |
| `Welfare_By_Agent_Per_Year.csv` | Per-year planner welfare by agent id. |
| `Private_CVaR_By_Agent.csv` | Per-agent private CVaR (VRES, electrolyzer, green offtaker) when $\gamma<1$. |
| `TimerOutput.yaml` | Profiling: time spent in imbalances, residuals, capacity dual update, price updates, solve, etc. |

### 12.2 ME contract results (`me_pap_results/`, `me_top_results/`, `me_sop_results/`)

The ME contract entry points produce the same major ADMM outputs as market_exposure (ADMM_Convergence, ADMM_Diagnostics, `Capacity_Consensus.csv`, 5× Market_History, Agent_Summary, Market_Prices, Risk_Metrics, Social_Welfare_Per_Year, Welfare_By_Group_Per_Year, Welfare_By_Agent_Per_Year), with additional PPA/HPA energy columns and **shared-cap alignment** columns (`ppa_cap_*`, `hpa_cap_*`: $|C - q^{\mathrm{peak}}|$ and $|C^k - C^{k-1}|$) in convergence and diagnostics. Per-agent **investment** capacity columns and `Capacity_Consensus.csv` follow the equality-split structure in §6.4. Two ME-only files are **not** written by the contracts saver: `Offtaker_GC_Diagnostics.csv` and `H2_Producer_Diagnostics.csv`. Additional contract outputs:

| File | Contents |
|---|---|
| `PPAs.csv` | Per-VRES summary: `capacity_contracted_MW` (**shared** bilateral $C$ at convergence), `energy_transferred_MWh`, `ppa_price_EUR_per_MWh` (scalar strike $K^{\mathrm{PPA}}$). |
| `HPAs.csv` | Per-GreenProducer summary: `capacity_contracted_MW` (shared $C^{\mathrm{HPA}}$), `energy_transferred_MWh`, `hpa_price_EUR_per_MWh` (scalar $K^{\mathrm{HPA}}$). Volume mode (PaP/ToP/SoP) is set by entry point, not this file. |
| `Green_Agents_Detail.csv` | Per-agent PPA breakdown (VRES and GreenProducer): total capacity, contracted vs pool energy, and prices. |

### 12.3 Social Planner Results

Outputs from the **complete risk trading** benchmark (`social_planner.jl`; §4.8, §6).

| File | Contents |
|---|---|
| `Market_Prices.csv` | Columns: `Time`, `Elec_Price`, `H2_Price`, `Elec_GC_Price`, `H2_GC_Price`, `EP_Price`. One row per (jy, jd, jh) timestep. Prices = balance-constraint duals scaled per §7.2: `dual / (W[jd,jy] × μ[jy])`, where `μ[jy]` is the effective scenario weight from the epigraph dual (equals `P[jy]` at $\gamma=1$; includes CVaR tail weight at $\gamma<1$). Expected marginal social values at $\gamma=1$; **risk-adjusted** social shadow prices at $\gamma<1$. |
| `Risk_Metrics.csv` | **Long format**: columns `Metric`, `Value`, `Unit`, including demand vs ex-demand split. See §7.6. |
| `Social_Welfare_Per_Year.csv` | One row per scenario. Columns: `case`, `scenario_year`, `probability`, `social_welfare`, `social_loss`, `welfare_demand`, `welfare_ex_demand`. |
| `Welfare_By_Group_Per_Year.csv` | Per-year planner welfare by group. |
| `Welfare_By_Agent_Per_Year.csv` | Per-year planner welfare by agent. |
| `Agent_Summary.csv` | Identical schema to the ME file — see §12.1. |
| `SP_Capacities.csv` | Long format: `AgentID`, `jy`, `cap`. One row per capacity-owning agent per scenario. Because capacity is non-anticipative, `cap` is constant across `jy` for a given agent; the per-`jy` rows exist so the file can be joined against scenario-indexed results. |
| `SP_Primal_Quantities.csv` | The full primal allocation, one row per (jy, jd, jh) slot — 2,880 rows on the default grid ($24 \times 8 \times 15$). Columns: `Time`, `jy`, `jd`, `jh`, plus one `<AgentID>_<market>` column for every agent–market pair. |

---

## 13. Code Conventions

### 13.1 JuMP Model Storage

Each agent's JuMP model uses `mod.ext` dictionaries:
- `mod.ext[:sets]` — Index ranges (JH, JD, JY).
- `mod.ext[:parameters]` — Scalars and arrays (capacities, costs, ADMM λ/ḡ/ρ).
- `mod.ext[:timeseries]` — 3D hourly profiles (AF, LOAD_E, etc.).
- `mod.ext[:variables]` — JuMP decision variables.
- `mod.ext[:constraints]` — JuMP constraints.
- `mod.ext[:expressions]` — JuMP expressions (net positions, objective terms).

### 13.2 Anonymous Variables (Planner)

In the social planner, all variables use anonymous JuMP syntax with `base_name` to avoid naming conflicts when multiple agents share the same planner model:
```julia
q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")
```

### 13.3 Commenting Standard

Every `.jl` file follows this standard:
- **File header**: Purpose, arguments, side effects, context.
- **Section dividers**: `# ── Section Name ──` or `# ---` blocks.
- **Per-line/block comments**: Every non-trivial line explains WHAT it does and WHY.
- **Mathematical formulas**: Objectives and constraints are documented with their full mathematical form in comments above the code.

### 13.4 Data Flow

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

## 14. References

1. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.  
   Core references used here: epigraph reformulation, convex duality, and KKT optimality conditions.

1b. S. Boyd, N. Parikh, E. Chu, B. Peleato & J. Eckstein, “Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers,” *Foundations and Trends in Machine Learning*, 3(1), 1–122, 2011.  
   Canonical ADMM reference: sharing form, primal/dual residuals, adaptive $\rho$, stopping criteria — mapped to this implementation in §6.10.  
   PDF: [Stanford EE364b](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

2. R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk*, 2(3), 2000.  
   Foundational CVaR optimization reformulation used for the planner risk term.  
   Open copy: [University of Washington PDF](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)

3. IPOPT Documentation (latest release), nonlinear optimization and multiplier reporting.  
   Main docs: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)  
   (Used to justify direct QCP dual extraction for the social-planner benchmark in this project.)

4. A. Eichfelder, A. Schöbel, L. Schmitz, “A tutorial on properties of the epigraph reformulation,” *Optimization Online*, 2024.  
   Additional modern reference for epigraph reformulation properties and KKT interpretation.  
   PDF: [Optimization Online](https://optimization-online.org/wp-content/uploads/2024/10/Epigraph_reformulation-1.pdf)

5. G. de Maere d’Aertrycke, A. Ehrenmann, D. Ralph & Y. Smeers, “Risk trading in capacity equilibrium models,” EPRG Working Paper 1720 / Cambridge Working Paper in Economics 1757, 2018.  
   Source for the **complete vs incomplete risk trading** labels used in §1 and §4.8 (Table 1: risk-neutral optimisation [a]; risk-averse competitive with complete markets [b]; risk-averse competitive with no risk trading).  
   Stable URL: [JSTOR resrep30413](https://www.jstor.org/stable/resrep30413)

6. S. A. Gabriel, A. J. Conejo, J. D. Fuller, B. F. Hobbs, C. Ruiz, *Complementarity Modeling in Energy Markets*, Springer, 2013.  
   Taxonomy for MCP vs MPEC vs EPEC, spatial price equilibrium (nodal networks), and Cournot oligopoly models — see §4.3 for how this project maps to the perfect-competition MCP class (not SPE, not EPEC).

7. H. Höschle, H. Le Cadre, Y. Smeers, A. Papavasiliou & R. Belmans, “An ADMM-Based Method for Computing Risk-Averse Equilibrium in Capacity Markets,” *IEEE Transactions on Power Systems*, 33(5), 4819–4830, 2018.  
   Source for the **$\gamma$–CVaR objective** and ADMM for risk-averse equilibrium. Their case-study $\beta$ axis (1 → 0 at fixed $\gamma=0.5$) is **not** identical to Rockafellar $\beta$ in code — see §4.10.4.

### Data sources for the NL calibration (§9.6)

These source keys are referenced inline in `Data/data.yaml` (tag `[NL]`) and in the §9.6 calibration tables.

- **`[CBS-RE]`** — Statistics Netherlands (CBS), *Renewable electricity; production and capacity* (table 82610ENG). Installed capacity end-2025 nader voorlopig: **solar PV 25,881 MWp, wind 11,782 MW**; 2024 normalised fleet CFs **18.24% solar, 28.01% wind**. [cbs.nl/en-gb/figures/detail/82610ENG](https://www.cbs.nl/en-gb/figures/detail/82610ENG); longread [Hernieuwbare energie in Nederland 2024](https://www.cbs.nl/nl-nl/longread/rapportages/2025/hernieuwbare-energie-in-nederland-2024/4-windenergie)
- **`[CBS-EP]`** — CBS, *Electricity; production and means of production* (table 37823ENG) + *Electricity and heat by energy commodity* (80030ENG). NL **2024 net end-user consumption ≈108.5 TWh**; **2024 fossil generation ≈76% gas, 13% coal, 11% biomass**; central gas-fired electric capacity **≈17.9 GW**. [cbs.nl/en-gb/figures/detail/37823eng](https://www.cbs.nl/en-gb/figures/detail/37823eng); [cbs.nl/en-gb/figures/detail/80030eng](https://www.cbs.nl/en-gb/figures/detail/80030eng)
- **`[ENTSOE-PEAK-2024]`** — ENTSO-E *Statistical Factsheet 2024*: NL highest hourly load **19,477 MW** on 8 Jan 2024 16:00–17:00 UTC. [ENTSO-E Statistical Factsheet 2024](https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/Statistics/Factsheet/entsoe_sfs2024_web.pdf)
- **`[NL-DA-2024]`** — NL EPEX day-ahead annual average **77.29 €/MWh** (2024). [energy-charts.info NL prices](https://energy-charts.info/charts/price_average/chart.htm?c=NL&interval=year&year=2024); [Nationaal Energie Dashboard jaaroverzicht 2024](https://ned.nl/nl/achtergrond/jaaroverzicht-2024)
- **`[TTF-2024]`** — Dutch TTF natural-gas front-month futures: **2024 annual average ≈34.4 €/MWh_th** (Energy-Charts 34.39; AleaSoft ICE 34.65). Anchor for `Fuel.GasPrice = 34.40`.
- **`[ETS-2024]`** — EU ETS EUA front-year futures: **2024 average ≈64.8 €/tCO₂** (Energy-Charts 64.79; Veyt 66.5). Anchor for `Fuel.CO2Price = 64.79`.
- **`[API2-2024]`** — ARA (API2) steam-coal benchmark: **2024 contract average ≈102 €/t** (TÜV/Mainova certification of ICE API2 month futures) ⇒ ≈12.5 €/MWh_th at 8.14 MWh_th/t.
- **`[PELLET-2024]`** — NW-European industrial wood-pellet (I2) CIF ARA: **≈156 €/t** end-Q1 2024 (RBCN Biomass Market Update) ⇒ ≈33 €/MWh_th at 4.72 MWh_th/t (17 GJ/t).
- **`[IRENA-2024]`** — IRENA, *Renewable Power Generation Costs in 2024* (published 2025). Global utility-scale solar LCOE USD 0.043/kWh; EU offshore wind ~USD 0.080/kWh; NL solar LCOE range ≈45–58 €/MWh (third-party synthesis). Basis for updated VRES fixed costs. [irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024](https://www.irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024)
- **`[IEA-H2-2024]`** — IEA, *Global Hydrogen Review 2024* Assumptions annex: water electrolysis installed CAPEX **2160 USD/kWe** (global average 2023; China 1100), annual OPEX **3% of CAPEX**, efficiency 66% LHV. Basis for `FixedCost_per_MW_Electrolyzer = 262 k€/MW_e-yr` (2160 USD/kWe × 0.92 €/USD; CRF 8%/20 yr + 3% FOM). Nameplate convention is **electrical input**. [IEA GHR 2024 Assumptions annex](https://iea.blob.core.windows.net/assets/36017d2f-747b-4993-b06d-33209fe143fa/GHR24AssumptionsAnnex.pdf)
- **`[IEA-NH3-2024]`** — IEA (via PtX Hub *Power-to-Ammonia*, 2025): green-ammonia **synthesis loop + air separation** CAPEX **770 USD/(t NH₃/y)** in 2023, **excluding** the electrolyser. Converted at 5.167 MWh/t and 8760 h ⇒ 1.201 M€/MW_EP; CRF+3% FOM ⇒ 158 k€/MW_EP-yr. Ammonia plant nameplate is **product output**. [ptx-hub.org Power-to-Ammonia](https://ptx-hub.org/wp-content/uploads/2025/05/250425_Paper-Ammoniak_V2-2.pdf)
- **`[OCI-NH3-2024]`** — OCI Global H2/FY 2024 results: NW Europe ammonia **$528–581/t** (FY avg $528; H2 $581). Sanity check for derived grey MC ≈443 €/t at 2024 fuel prices.
- **`[Yara-NL]`** — Yara Nederland: Sluiskil ammonia production **≈1.8 Mt/yr**. [yara.nl productie-eenheid Sluiskil](https://www.yara.nl/over-yara/yara-in-de-benelux/yara-sluiskil/over-yara-sluiskil/productie-eenheid-sluiskil/)
- **`[TNO-2026]`** — TNO scenario study (COMPETES-TNO), Dutch dispatchable (gas-fired) capacity reference figures (~14.7 GW gas in the 2030 reference). [publications.tno.nl/publication/34645515](https://publications.tno.nl/publication/34645515/fLCICwBT/TNO-2026-R10080.pdf)
- **`[PBL-2019]`** — PBL/ECN, *Decarbonisation options for the Dutch fertiliser industry*, 2019. NL ammonia capacity: **Yara Sluiskil ≈1.8 Mt/yr + OCI Nitrogen Geleen ≈1.2 Mt/yr ≈ 3 Mt NH₃/yr**. [pbl.nl PDF](https://www.pbl.nl/uploads/default/downloads/pbl-2019-decarbonisation-options-for-the-dutch-fertiliser-industry_3657.pdf)
- **`[H2EU-2023]`** — Hydrogen Europe, *Clean Ammonia Report*, 2023. **Grey ammonia LCOA ≈534–891 €/t** (gas 40–80 €/MWh, CO₂ 75 €/t; range up to ~1,069 €/t at gas 110 €/MWh); green ammonia ≈2–6× grey; Haber–Bosch H₂↔NH₃ efficiency. [hydrogeneurope.eu PDF](https://hydrogeneurope.eu/wp-content/uploads/2023/03/2023.03_H2Europe_Clean_Ammonia_Report_DIGITAL_FINAL.pdf)
- **`[IEA-H2]`** — IEA, *Global Hydrogen Review* / electrolyser technology briefs. PEM electrolyser efficiency ≈67% (≈1.5 MWh_e/MWh_H₂); installed CAPEX in the ~1.0–1.5 M€/MW range in the early 2020s (noting CAPEX rose ~15% in 2023 vs 2021). [iea.org/reports/global-hydrogen-review-2023](https://www.iea.org/reports/global-hydrogen-review-2023)
- **`[DEA-2020]`** — Danish Energy Agency, *Technology Data for Renewable Fuels* (electrolysers). 100 MW alkaline installed CAPEX ≈1,200 €/kW (2020); corroborated by Fraunhofer ISE (2021): PEM ≈718 €/kW at 100 MW to ≈978 €/kW at 5 MW, fixed O&M ≈15–20 €/kW·yr. Basis for the ~130 k€/MW-yr annualised electrolyser fixed cost. [ens.dk technology data](https://ens.dk/en/our-services/technology-catalogues/technology-data-renewable-fuels)
- **`[IRENA-2022]`** — IRENA, *Renewable Power Generation Costs*, 2022 (superseded for VRES fixed costs by `[IRENA-2024]`). [irena.org/publications](https://www.irena.org/publications/2023/Aug/Renewable-Power-Generation-Costs-in-2022)
- **`[TTF-2021]`** — Dutch TTF 2021 annual average ≈47 €/MWh_th (historical anchor, superseded by `[TTF-2024]`).
- **`[ETS-2021]`** — EU ETS 2021 average ≈53 €/tCO₂ (historical anchor, superseded by `[ETS-2024]`).
- **`[API2-2021]`** — API2 2021 average ≈121 €/t (historical anchor, superseded by `[API2-2024]`).
- **`[PELLET-2021]`** — NW-European industrial wood-pellet ≈150–170 €/t in 2021 (historical anchor, superseded by `[PELLET-2024]`).
- **`[IHS-NH3-2021]`** — Ammonia CFR NW Europe 2021 average ≈557 $/t (historical sanity check, superseded by `[OCI-NH3-2024]`).
- **`[IPCC-EF]`** — IPCC 2006 Guidelines, default stationary-combustion emission factors: natural gas 56.1 kg CO₂/GJ ⇒ **0.2016 tCO₂/MWh_th**; hard coal 94.6 kg CO₂/GJ ⇒ **0.3406 tCO₂/MWh_th**; biomass ETS zero-rated. [ipcc-nggip.iges.or.jp](https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol2.html)
- **`[IEA-WEO]`** — IEA *World Energy Outlook* / power-plant technology assumptions: modern CCGT net LHV efficiency ≈58%, OCGT peaker ≈38%, hard-coal steam ≈42%; used for the conventional stage efficiencies.
- **`[FE-BAT]`** — European Commission JRC, *BAT Reference Document for the Manufacture of Large Volume Inorganic Chemicals — Ammonia*: modern SMR ammonia plants consume ≈32 GJ_LHV natural gas per tonne NH₃ and emit ≈1.8 tCO₂/t NH₃ (process plus fuel). Basis for `GasIntensity` and `CO2Intensity`.
- **`[DECHEMA-2030]`** — DECHEMA, *Technology Study: Low Carbon Energy and Feedstock for the European Chemical Industry*: corroborates the 30–34 GJ/t NH₃ SMR gas-intensity range.
- **`[RED-III]`** — Directive (EU) 2023/2413 (RED III): ≥42% of hydrogen used in industry must be renewable/RFNBO by 2030 → `gamma_GC = 0.42` mandate. [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/dir/2023/2413/oj)
- **`[GoO]`** — European Guarantees-of-Origin (GoO/CertiQ) market prices: historically <2 €/MWh, rising to ~5–9 €/MWh in 2022–2023; used for the GC demand WTP intercept.
- **`[LHV]`** — Ammonia lower heating value 18.6 MJ/kg ⇒ **5.167 MWh per tonne NH₃**, the conversion constant for all €/t ↔ €/MWh_EP and Mt ↔ TWh calculations.

### Weather and representative-day inputs (§9.7)

- **`[ERA5-OM]`** — Open-Meteo Historical Weather API (ERA5 reanalysis). Hourly GHI, 100 m wind speed, and 2 m temperature for central NL (52.09°N, 5.12°E). [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)
- **`[RPF]`** — RepresentativePeriodsFinder.jl (KU Leuven UCM), used with the hierarchical clustering method to select representative days. [gitlab.kuleuven.be/UCM/representativedaysfinder.jl](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl)
- **`[PM-2018]`** — S. Pineda and J. M. Morales, "Chronological time-period clustering for optimal capacity expansion planning with storage," *IEEE Trans. Power Syst.*, vol. 33, no. 6, pp. 7162–7170, 2018. (Clustering algorithm implemented in RPF.)
- **`[BF-2008]`** — M. Bessec and J. Fouquau, "The non-linear link between electricity consumption and temperature in Europe: A threshold panel approach," *Energy Economics*, vol. 30, no. 5, pp. 2705–2721, 2008. Establishes the heating/cooling degree-day threshold form and shows the temperature sensitivity of electricity demand is markedly **lower** in gas-heated northern countries than in electrically heated ones — the basis for the modest NL heating and cooling coefficients.
- **`[ENTSOE-LOAD]`** — ENTSO-E Transparency Platform, actual total load for the NL bidding zone. Basis for the weekday/weekend ratio and the NL load-factor target. [transparency.entsoe.eu](https://transparency.entsoe.eu/)
- **`[DYKSTRA-1983]`** — R. L. Dykstra, "An algorithm for restricted least squares regression," *J. Amer. Statist. Assoc.*, vol. 78, no. 384, pp. 837–842, 1983. The alternating-projection method with correction terms used to rebalance representative-day weights; unlike plain alternating projection it converges to the true projection onto the intersection of the constraint sets.
