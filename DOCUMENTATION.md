# Multi-Agent Energy Market Simulation — Technical Documentation

## Table of Contents

0. [Notation and Units](#0-notation-and-units)
1. [Overview](#1-overview)
2. [Markets](#2-markets) — incl. [Contract pools](#contract-pools-me-pap--me-top--me-sop), [Price: λ vs K](#price-clearing-λ-versus-settlement-k), [Risk allocation](#risk-allocation--who-bears-what), [Shared capacity $C$](#how-contract-capacity-is-determined), [ADMM iteration flow](#one-admm-iteration-contracts-case)
3. [Agents](#3-agents)
4. [Equilibrium Theory](#4-equilibrium-theory-mcp-structure-competition-and-objectives) — MCP, competition, objectives, risk institutions; [CVaR, γ, β](#410-risk-aversion-cvar-γ-and-β)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [ADMM Algorithm](#6-admm-algorithm) — [why ADMM](#60-why-admm-alternatives-and-literature), [Boyd mapping](#610-mapping-to-boyd-et-al-2011), [ρ controller](#63-adaptive-penalty-ρ), [warm-start](#66-warm-start-from-social-planner)
7. [Social Planner Benchmark](#7-social-planner-benchmark) — incl. [IPOPT tolerance (`ipopt_tol`)](#ipopt-settings-and-convergence-tolerance-ipopt_tol)
8. [Data and Indexing](#8-data-and-indexing)
9. [Configuration Reference (data.yaml)](#9-configuration-reference-datayaml) — incl. [NL Calibration and Data Sources](#96-nl-calibration-and-data-sources)
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
- $h \in \mathcal{H} = \{1,\dots,n_{\mathrm{timesteps}}\}$: hours within a representative day.
- $d \in \mathcal{D} = \{1,\dots,n_{\mathrm{reprDays}}\}$: representative days.
- $y \in \mathcal{Y} = \{1,\dots,n_{\mathrm{years}}\}$: scenario years.

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
- $\beta$: **CVaR confidence level** (Rockafellar–Uryasev / Hoschle et al.). $\mathrm{CVaR}_{\beta}$ averages the worst $(1-\beta)$ share of scenarios; **lower $\beta$ ⇒ more risk-averse** (e.g. $\beta=0.2$ ⇒ worst 80%; $\beta=0.8$ ⇒ worst 20%). At fixed $\gamma=0.5$, sweep $\beta$ over $0.2$, $0.4$, $0.6$, and $0.8$ for risk-aversion intensity (§4.10.4).
- $\alpha_i$, $u_i(y)$, $\mathrm{CVaR}_{i}$: Rockafellar–Uryasev auxiliaries for agent $i$ (VaR proxy, shortfall, conditional tail loss).

Full definitions, Hoschle-style calibration workflow, and equilibrium effects: **§4.10**. Social planner CVaR structure: **§7.4**. Reporting: **§7.6**.

---

## 1. Overview

This project implements a **multi-agent equilibrium model** for coupled electricity, hydrogen, green-certificate, and end-product markets, coordinated via **ADMM** (Alternating Direction Method of Multipliers). Each agent has its own JuMP optimization model; market-clearing is achieved by iteratively updating prices and penalty terms so that supply and demand balance in each market.

The project includes three entry points. Each has a **code name** (script / folder) and an **economic label** aligned with d’Aertrycke et al. (2018), *Risk trading in capacity equilibrium models* (see §4.8 and §14):

| Script | Code name | Economic case (competitive spot, capacity investment) |
|---|---|---|
| **`social_planner.jl`** | Social planner (SP) | **Complete risk trading** — centralised risk-averse welfare maximisation with a single social CVaR on aggregate welfare. |
| **`market_exposure.jl`** | Market exposure (ME) | **Incomplete risk trading** — decentralised equilibrium via ADMM; risk-averse agents hedge **private** tail losses with per-agent CVaR, without an explicit risk market. |
| **`me_pap.jl`** / **`me_top.jl`** / **`me_sop.jl`** | Market exposure with bilateral contracts | Same **incomplete risk trading** as ME, plus **PPA** (always PaP) and **HPA** (volume mode set by entry point). See §2 *Contract pools*. |
| **`green_h2_social_planner.jl`** | Green H₂ partial planner (GH2-SP) | **Partial complete risk trading** — electrolyzer + green offtaker merged; one coalition CVaR; ADMM + external spot markets unchanged. |
| **`green_social_planner.jl`** | Green partial planner (G-SP) | **Partial complete risk trading** — solar + wind + electrolyzer + green offtaker merged; one coalition CVaR; ADMM + external spot markets unchanged. |

**When $\gamma = 1$ (risk-neutral):** SP is the stochastic welfare-maximisation benchmark; ME (and ME+C) should converge to the same quantities and spot prices as SP in the limit of exact ADMM convergence (first welfare theorem).

**When $0 < \gamma < 1$ (risk-averse):** SP is the **centralised risk-pooling / complete-risk-trading** benchmark; ME and ME+C are **private-hedging / incomplete-risk-trading** equilibria. Quantities and prices need not (and generally should not) match SP — that divergence is expected theory, not a formulation error. SP balance duals remain valid **risk-adjusted social shadow prices** for each commodity (§4.8, §7.2); ADMM $\lambda$ are valid equilibrium prices for the decentralised case.

Entry-point details:

- **`market_exposure.jl`** — Distributed ADMM; five markets: electricity, elec GC, H₂, H₂ GC, end product.
- **`me_pap.jl`**, **`me_top.jl`**, **`me_sop.jl`** — Same risk architecture as ME, with bilateral **PPA** (always PaP) and **HPA** (PaP / take-or-pay / send-or-pay by entry point). See §2 *Contract pools*.
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

### Contract pools (me_pap / me_top / me_sop)

In **`me_pap.jl`**, **`me_top.jl`**, and **`me_sop.jl`**, two additional bilateral pools are cleared:

| Script | Label | HPA volume | Results folder |
|--------|-------|------------|----------------|
| `me_pap.jl` | **ME PaP** | Pay-as-produced | `me_pap_results/` |
| `me_top.jl` | **ME ToP** | Take-or-pay | `me_top_results/` |
| `me_sop.jl` | **ME SoP** | Send-or-pay | `me_sop_results/` |

All three share the same agent roster and spot markets as `market_exposure.jl`, plus bilateral **PPA** and **HPA** pools cleared by ADMM (`ADMM_contracts.jl`).

**PPAs are identical across all three entry points:** volume is always **pay-as-produced (PaP)**; price is a **scalar strike** $K^{\mathrm{PPA}}$ (€/MWh, uniform over all hours) derived from **negotiated** bilateral clearing at convergence.

**HPAs differ by entry point** (volume only). Price structure and benchmark are configured in `data.yaml` → `PPAs` / `HPAs` (see [Price: λ versus K](#price-clearing-λ-versus-settlement-k) and [HPA price structure](#hpa-price-structure-datyaml--hpas)).

| Market | Key | Description | Unit | Supply Side | Demand Side |
|---|---|---|---|---|---|
| **PPA** | `ppa` | Bilateral VRES–GreenProducer electricity flow (bundled with `elec_GC`) | MWh | VRES (`g_ppa`) | GreenProducer (`g_ppa_from`) |
| **HPA** | `hpa` | Bilateral GreenProducer–GreenOfftaker hydrogen flow (bundled with `H2_GC` equivalent) | MWh_H2 | GreenProducer (`h2_hpa`) | GreenOfftaker (`h2_hpa_from`) |

- **Contract capacity** $C$: scalar MW ceiling on contract flow at each hour; **one shared value per bilateral link** (PPA per VRES, HPA per green producer), updated **outside** agent subproblems — see [How contract capacity is determined](#how-contract-capacity-is-determined).
- **Contract energy** $q$: delivered MWh at each timestep (`g_ppa`, `h2_hpa`), cleared by bilateral $\lambda^{\mathrm{contract}}$; settled at strike $K$ (scalar €/MWh at convergence).
- **PPA volume**: always pay-as-produced. **HPA volume**: pay-as-produced (`me_pap`), take-or-pay (`me_top`), or send-or-pay (`me_sop`).

Each pool uses the same ADMM structure as spot markets for **energy**: bilateral imbalance must go to zero. **Capacity** $C$ is **not** chosen independently by buyer and seller; it follows the shared-capacity update in `Source/contract_capacity.jl`. ADMM details: §6.7.

#### Overview: real contracts vs this implementation

In real bilateral PPAs/HPAs, two firms sign **one package** before operations:

| Term | Meaning in practice |
|------|---------------------|
| **Price** $K$ | Strike or index rule (€/MWh) — fixed for the contract life |
| **Capacity** $C$ | Committed MW — same number for both parties |
| **Volume rule** | PaP, ToP, or SoP — who pays when $q \neq C$ |

The model mirrors that structure:

- **One shared $C$** per link (`ADMM["SharedCap"]`), not two opposing optimiser variables.
- **Scalar $K$** at convergence (uniform every hour), from negotiated clearing or an exogenous benchmark.
- **Private CVaR** when $\gamma < 1$ — contracts can hedge **tail** cash flows, but there is **no** separate financial risk market (same incomplete risk trading as `market_exposure.jl`).

**What is not modelled explicitly:** a formal “sign only if both better off than no contract” (individual-rationality) constraint. Mutual acceptance is **emergent**: if dispatch at converged $(K,C)$ routes volume through the contract (especially when $\gamma < 1$), $C$ rises with utilisation; if the pool dominates (typical at $\gamma = 1$), $C \to 0$.

#### Price: clearing λ versus settlement K

Three related objects appear in the code; keep them separate:

| Object | Role | When it moves |
|--------|------|----------------|
| $\lambda^{\mathrm{contract}}$ | ADMM **clearing price** for contract **energy** imbalance (3D field, like spot $\lambda$) | Every ADMM iteration (dual ascent on supply − demand) |
| $K$ | **Settlement strike** in agent cash flows (€/MWh, broadcast to all hours) | Each iteration: provisional $K$ from benchmark rule; **at convergence:** snapshotted scalar in `PPAs.csv` / `HPAs.csv` |
| $C$ | **Shared contract capacity** (MW) | Each iteration: `update_shared_contract_capacity!`; **at convergence:** snapshotted scalar |

**Default (`price_benchmark: negotiated`):** at convergence, $K$ equals the W-weighted mean of $\lambda^{\mathrm{contract}}$ over the horizon — “internally cleared” bilateral price. **Alternatives** in `data.yaml`: `electricity`, `ammonia`, `NG` (exogenous reference $B$); for HPAs also `price_structure: cfd` (fixed leg + index leg when $\gamma < 1$). PPAs always use negotiated clearing for $K^{\mathrm{PPA}}$.

Agents **pay/receive** using $K$ in objectives; $\lambda^{\mathrm{contract}}$ drives ADMM toward $q^{\mathrm{supply}} = q^{\mathrm{demand}}$.

#### Risk neutrality versus risk aversion (γ)

| Setting | Typical contract outcome | Interpretation |
|---------|--------------------------|----------------|
| **$\gamma = 1$** | $C \approx 0$, tiny or zero contract energy | Pool clearing is cheaper on **average**; no tail-risk motive → **no meaningful signing** (expected behaviour). |
| **$\gamma < 1$** | $C > 0$ possible if bilateral flows improve **private CVaR** | Firms may **commit capacity** and **lock price** $K$ to hedge bad-scenario cash flows; compare PaP / ToP / SoP for who bears volume risk. |

Compare `PPAs.csv`, `HPAs.csv` (columns `capacity_contracted_MW`, `energy_transferred_MWh`, strike price) across $\gamma$ and entry points. Compare ME contract runs to `social_planner.jl` for **institutional** risk gaps (§4.8), not only price levels.

#### Contract terms: two time axes

Agents enter bilateral contracts to **lock in a price and capacity** for bundled green inputs/outputs. Two axes must be kept separate:

| Axis | Behaviour |
|---|---|
| **ADMM iterations** | $\lambda^{\mathrm{contract}}$, provisional settlement strike $K$, and contracted capacity $C$ all **update each iteration** toward bilateral consensus (same as spot markets). |
| **Hourly timesteps** $(h,d,y)$ | At **convergence**, strike $K$ and capacity $C$ are **scalars** — the same €/MWh and MW every hour for the contract duration. Flows $q$ vary by hour; $K$ and $C$ do not. |

Implementation (`Source/contract_strike.jl`, `Source/contract_capacity.jl`):

1. **Each ADMM iteration:** $\lambda^{\mathrm{contract}}$ dual-updates from energy imbalances; provisional $K$ = W-weighted mean of the configured benchmark field (broadcast to all hours); shared $C$ is **fixed** on both parties’ `ppa_cap` / `hpa_cap` (`apply_shared_contract_caps!`), then updated once per iteration (`update_shared_contract_capacity!`).
2. **At convergence:** `finalize_contract_terms!` snapshots scalar $K^{\mathrm{PPA}}$, $K^{\mathrm{HPA}}$, $C^{\mathrm{PPA}}$, $C^{\mathrm{HPA}}$ into `ADMM["ContractStrikes"]` for CSV output and run summaries.

There is **no mid-run freeze** of strikes or capacities at a fixed ADMM iteration.

#### PPA — always pay-as-produced

- **Seller:** VRES $i$ — splits generation $g_i = g^{\mathrm{EOM}}_i + g^{\mathrm{PPA}}_i$.
- **Buyer:** Green producer — receives $g^{\mathrm{PPA}}_{i\leftarrow}$ from each VRES (bundled electricity + elec GC).
- **Capacity:** scalar $C^{\mathrm{PPA}}_i$ (MW); hourly $g^{\mathrm{PPA}}_i \le C^{\mathrm{PPA}}_i$.
- **Pool exclusion:** $g^{\mathrm{PPA}}$ is removed from electricity and elec-GC spot markets.

Settlement at each $(h,d,y)$:

$$
\text{Payment}_{i,h,d,y} = K^{\mathrm{PPA}}_i \cdot g^{\mathrm{PPA}}_{i,h,d,y}
$$

At convergence, $K^{\mathrm{PPA}}_i$ is the W-weighted mean of $\lambda^{\mathrm{PPA}}_{i,h,d,y}$ over the horizon (one scalar €/MWh, applied every hour).

#### HPA — three volume modes

Bundled **H₂ + H₂-GC equivalent**; contract flow removed from pool H₂ / H₂-GC markets. Let $q_{h,d,y}$ be contract delivery (MW_H2), $C$ contracted capacity (MW_H2), $K$ scalar strike (€/MWh_H2, uniform over hours at convergence).

**ME PaP (`me_pap.jl`) — pay-as-produced:**

$$
\pi^{\mathrm{buyer}} \mathrel{+}= K \cdot q_{h,d,y}, \qquad
\pi^{\mathrm{seller}} \mathrel{+}= K \cdot q_{h,d,y}
$$

**ME ToP (`me_top.jl`) — take-or-pay:** buyer pays for at least $C$ MW_H2 each hour:

$$
q^{\mathrm{pay}}_{h,d,y} = \max\bigl(q_{h,d,y},\, C\bigr), \qquad
\text{Payment}_{h,d,y} = K \cdot q^{\mathrm{pay}}_{h,d,y}
$$

(with auxiliary $s_{h,d,y} \ge C - q_{h,d,y}$, $s \ge 0$ in the JuMP model).

**ME SoP (`me_sop.jl`) — send-or-pay:** seller penalised for under-delivery vs $\min(C, h^{\mathrm{out}}_{h,d,y})$:

$$
o_{h,d,y} = \min\bigl(C,\, h^{\mathrm{out}}_{h,d,y}\bigr), \qquad
\pi^{\mathrm{seller}} \mathrel{+}= K \cdot q_{h,d,y} - K \cdot s_{h,d,y}, \quad s \ge o - q
$$

Buyer still pays PaP on received volume: $K \cdot q_{h,d,y}$.

#### Risk allocation — who bears what?

All contract cases share the same **price hedge** at convergence: strike $K$ is a scalar applied every hour, so bilateral energy settled at $K$ is insulated from hourly spot swings on the pool markets (`λ_elec`, `λ_H2`, `λ_EP`, etc.). **Contract capacity** $C$ is a negotiated MW ceiling on the bilateral pipe in every entry point; there is **no separate €/MW capacity tariff** cleared by ADMM. What differs by volume mode is **who bears utilization, production, and volume risk** when delivery $q$ is below $C$ or when spot prices move differently from $K$.

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
| **VRES (seller)** | Receives $K^{\mathrm{PPA}} \cdot g^{\mathrm{PPA}}$ | **Production risk:** revenue scales with renewable output; zero output ⇒ zero PPA revenue. $C^{\mathrm{PPA}}$ caps how much can be sold on the contract, but **unused headroom is not paid**. |
| **Green producer (buyer)** | Pays $K^{\mathrm{PPA}} \cdot g^{\mathrm{PPA}}$ | **Input cost risk:** pays only for MWh received; if VRES delivers little, PPA bill is low but the plant may need costlier pool electricity. No minimum offtake on the PPA leg. |

PPA shifts **renewable volume risk** to the VRES and **procurement / conversion risk** to the electrolyzer, while **locking the green-electricity price** at $K^{\mathrm{PPA}}$ for whatever is delivered.

**HPA — pay-as-produced (`me_pap.jl`):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **Green producer (seller)** | Receives $K \cdot q$ | **Production / dispatch risk:** revenue follows actual $q$; idle hours yield no contract revenue. Must still recover fixed H₂ CAPEX from total margins. |
| **Green offtaker (buyer)** | Pays $K \cdot q$ | **Uptake / sourcing risk:** pays only for MWh taken; if $q = 0$, no HPA energy bill—but must meet EP demand and GC rules from pool H₂ or curtail. **No penalty** for “under-using” contracted capacity $C$. |

PaP is the **neutral** volume mode: $C$ is a **rate limit**, not a minimum payment. Risk is symmetric in the sense that **cash flows only move with physical delivery**.

**HPA — take-or-pay (`me_top.jl`):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **Green offtaker (buyer)** | Pays $K \cdot \max(q, C) = K \cdot C$ when $q \le C$ | **Minimum offtake / utilization risk:** pays for contracted MW each hour even when $q \ll C$ (e.g. electrolyzer down). Cannot avoid the bill by taking pool H₂ instead for that contractual slice. |
| **Green producer (seller)** | Receives $K \cdot \max(q, C)$ | **Revenue floor:** receives at least $K \cdot C$ per hour while $q \le C$; production above $C$ is capped by $q \le C$ on the contract leg. |

ToP **shifts volume / utilization risk from seller to buyer**: the offtaker guarantees a **financial minimum** at the contracted rate, analogous to paying for reserved pipeline capacity at energy price $K$. This differs from PaP, where low $q$ implies low payment—not because $C$ is absent, but because $C$ does not enter the settlement formula.

**HPA — send-or-pay (`me_sop.jl`):**

| Party | Settlement | Main volume / utilization risk |
|-------|--------------|--------------------------------|
| **Green offtaker (buyer)** | Pays $K \cdot q$ (PaP on delivery) | Same delivery-linked payment as PaP; **no** minimum offtake premium. |
| **Green producer (seller)** | Receives $K \cdot q - K \cdot s$, with shortfall $s \ge \min(C, h^{\mathrm{out}}) - q$ | **Delivery / availability risk:** penalised when capable output exceeds contract delivery (under-shipping vs obligation $o = \min(C, h^{\mathrm{out}})$). |

SoP **shifts delivery risk to the producer**: the buyer is not forced to pay for undelivered molecules, but the seller pays a **strike-scaled penalty** when it could have delivered more under the contract.

**Summary — volume risk on the HPA leg:**

```text
                    Volume / utilization risk borne mainly by
                 Buyer (offtaker)          Seller (producer)
              ┌─────────────────────┬─────────────────────┐
   me_pap     │  sourcing if q low  │  revenue if q low   │
   me_top     │  pays K·C if q < C  │  floor K·C if q < C │
   me_sop     │  PaP on q only      │  penalty if under-deliver │
              └─────────────────────┴─────────────────────┘
```

**Relation to $\gamma < 1$:** At $\gamma = 1$, agents are risk-neutral and the table above is the full **average-cost** story — contracts are usually unused ($C \to 0$). At $\gamma < 1$, each agent’s **private CVaR** weights bad scenarios; bilateral cash flows at fixed $K$ and shared $C$ can **reduce tail loss**, so you may see **$C > 0$** and positive contract energy when hedging dominates. Volume modes change **which cash-flow tail** (buyer’s ToP bill vs seller’s SoP penalty vs PaP variability) enters whose CVaR. CfD (`HPAs.price_structure: cfd`) splits fixed vs benchmark-index legs for HPA when $\gamma < 1$. The social planner (`social_planner.jl`) still represents **complete** risk pooling — compare ME contract runs against SP for institutional gaps (§4.8), not only against `me_pap` at $\gamma = 1$.

#### HPA price structure (`data.yaml` → `HPAs`)

**Fixed price (default):** $\text{Settlement}_{h,d,y} = K \cdot q^{\mathrm{pay}}_{h,d,y}$ where $q^{\mathrm{pay}}$ is $q$ (PaP/SoP buyer), $\max(q,C)$ (ToP), etc. $K$ is a scalar €/MWh (uniform over hours) set at convergence from the chosen benchmark.

**CfD (contract for difference):** strike $K$ remains a scalar; settlement decomposes into reference leg $B_{h,d,y} \cdot q^{\mathrm{pay}}$ plus difference $(K - B_{h,d,y}) \cdot q^{\mathrm{pay}} = K \cdot q^{\mathrm{pay}}$ algebraically, but $B_{h,d,y}$ follows the selected benchmark while $K$ stays constant across hours — relevant when $\gamma < 1$. Set `HPAs.price_structure: cfd`.

| `price_benchmark` | Field $B$ at convergence | Source in model |
|-------------------|----------------------|-----------------|
| `negotiated` | $\lambda^{\mathrm{HPA}}$ | Bilateral ADMM clearing (default) |
| `electricity` | $\lambda^{\mathrm{elec}}$ | Electricity spot |
| `ammonia` | $\lambda^{\mathrm{EP}}$ | End-product / ammonia pool |
| `NG` | Grey-chain proxy | `Offtaker_Grey.MarginalCost / Alpha` → €/MWh_H2 |

At convergence: $K^{\mathrm{HPA}} = \text{W-weighted mean of } B$ over the horizon (scalar, uniform over hours).

#### How contract capacity is determined

There is **no separate capacity price** (no $\lambda$ for MW commitment). Each bilateral link has **one shared scalar** $C$ (MW):

| Pool | Storage key | Parties |
|------|-------------|---------|
| PPA (per VRES) | `ADMM["SharedCap"]["ppa"][vres_id]` | VRES seller ↔ green producer buyer |
| HPA (per H₂ producer) | `ADMM["SharedCap"]["hpa"][h2_id]` | green producer seller ↔ green offtaker buyer |

**Design rationale.** Real contracts specify **one** capacity both parties accept. An earlier formulation let each agent optimise its own `ppa_cap` / `hpa_cap` with ADMM penalties; under take-or-pay that produced a spurious buyer-vs-seller fight ($C \to 0$ vs $C \to \max$) and ADMM non-convergence. The shared-$C$ design applies to **all three entry points** (`me_pap`, `me_top`, `me_sop`).

**1. Agents take $C$ as given**

`ppa_cap` / `hpa_cap` remain JuMP variables but are **fixed** before each agent solve (`fix(cap, C)` in `apply_shared_contract_caps!`). Constraints $q_{h,d,y} \le C$ and settlement (PaP / ToP / SoP) use that value. Neither party optimises $C$ in its subproblem.

**2. Consensus / bargaining step (outside subproblems)**

After each ADMM iteration, `update_shared_contract_capacity!` (`Source/contract_capacity.jl`) sets:

$$
C^{\mathrm{target}} = \begin{cases}
\min\bigl(C_{\mathrm{phys}},\, C + \eta_{\uparrow}\,|\overline{\mathrm{imb}}|\bigr) & \text{if cap binds and energy imbalance wants more volume} \\
q^{\mathrm{peak}} & \text{otherwise}
\end{cases}
\qquad
C^{k+1} = (1-\tau)\, C^{k} + \tau\, C^{\mathrm{target}}
$$

where $q^{\mathrm{peak}} = \max_{h,d,y}(\text{contract supply}, \text{contract demand})$, $\overline{\mathrm{imb}}$ is mean energy imbalance on that link, $C_{\mathrm{phys}}$ is the minimum of relevant plant caps, and $(\tau, \eta_{\uparrow})$ come from `ADMM.contract_cap` in `data.yaml` (defaults: `relaxation: 0.35`, `expand_step: 2.0`).

**Intuition:** $C$ tracks **utilised** bilateral intent. No flow and no hedge motive → $C \to 0$. Sustained contract dispatch → $C$ rises toward $q^{\mathrm{peak}}$ (capped physically).

**3. Convergence and reporting**

Cap residuals monitor **alignment** $|C - q^{\mathrm{peak}}|$ (primal) and **stability** $|C^{k} - C^{k-1}|$ (dual). Contract **energy** clears via $\lambda^{\mathrm{contract}}$; **price** $K$ follows benchmark rules in [Price: λ versus K](#price-clearing-λ-versus-settlement-k).

History: `results["shared_ppa_cap"]`, `results["shared_hpa_cap"]`. At convergence, `finalize_contract_terms!` → `PPAs.csv` / `HPAs.csv` column `capacity_contracted_MW`.

#### One ADMM iteration (contracts case)

```text
For iteration k = 1, 2, …
  1. Fix shared C^k on all contract parties (apply_shared_contract_caps!)
  2. Update ḡ, λ, provisional K, z_cap; solve all subproblems (Gurobi)
  3. Record contract energy q; compute energy imbalances
  4. Update shared C^{k+1} (update_shared_contract_capacity!)
  5. Compute residuals; update λ^{k+1}; adapt ρ (update_rho_contracts!)
  6. Test convergence (energy + cap alignment + physical capacity + spot markets)
After loop: finalize_contract_terms! → scalar K, C for reporting
```

#### Contract implementation files

| File | Role |
|------|------|
| `Source/contract_strike.jl` | Benchmarks, per-iteration $K$ refresh, `finalize_contract_terms!` |
| `Source/contract_capacity.jl` | Shared $C$ init, bargaining update, fix caps before agent solves |
| `Source/contract_settlement.jl` | PaP / ToP / SoP payment terms |
| `Source/ADMM_contracts.jl` | Main loop; shared-cap update; `finalize_contract_terms!` after ADMM |
| `Source/ADMM_subroutine_contracts.jl` | Per-agent $\bar g$, $\lambda$, $K$, $B$ refresh; apply fixed $C$ |

During ADMM, $\lambda^{\mathrm{PPA/HPA}}$ and provisional $K$ update every iteration. At convergence, `ADMM["ContractStrikes"]` holds scalar $K$ and $C$ for reporting.

#### Recommended workflow

1. Run `social_planner.jl` (equilibrium benchmark).
2. Run `market_exposure.jl` (uncontracted ME).
3. Run `me_pap.jl`, `me_top.jl`, and/or `me_sop.jl` with the same `data.yaml` (SP warm-start for λ, primals, capacities).

**Suggested comparisons:**

| Question | What to run / read |
|----------|-------------------|
| Pool vs contracts at $\gamma=1$ | `market_exposure.jl` vs `me_pap.jl`; expect $C \approx 0$ |
| PaP vs ToP vs SoP volume risk | Same $\gamma$, three entry points; `HPAs.csv` + §2 *Risk allocation* |
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
- PPA couples VRES and GreenProducer: VRES splits generation `g_EOM + g_ppa`, and GreenProducer uses `e_in_pool + g_ppa_from`.
- HPA couples GreenProducer and GreenOfftaker: GreenProducer splits hydrogen pool sale vs contract (`h2_out - h2_hpa` to pool, `h2_hpa` to contract).

---

## 3. Agents

### 3.1 Power-Sector Agents

| Agent | Type | Description |
|---|---|---|
| `Gen_VRES_01` | `VRES` | Variable renewable (e.g. solar). Zero marginal cost. Produces both electricity and elec GCs (1:1). Constrained by hourly availability factor × **endogenous capacity**. Makes **one** installed-capacity and investment decision (`cap_VRES`, `inv_VRES`), incurring fixed annualised CAPEX `FixedCost_per_MW × cap_VRES` (same capacity in all weather scenarios). In contract entry points: splits generation into `g_EOM` (pool) and `g_ppa` (PPA); `g_ppa ≤ ppa_cap` where `ppa_cap` is the **shared** scalar $C^{\mathrm{PPA}}$ fixed each ADMM iteration; settlement at $K^{\mathrm{PPA}} \times g^{\mathrm{PPA}}$ (PaP). |
| `Gen_Conv_01` | `Conventional` | Dispatchable thermal fleet proxy. Constant availability (AF = 1). Uses a 3-stage increasing marginal-cost curve (coal-like, biomass-like, NG-like blocks) with configurable stage shares, stage-start marginal costs, and a final high-load marginal cost. No GC production. |
| `Cons_Elec_01` | `Consumer` | Elastic electricity demand. Quadratic utility `U(d) = A_E·d − ½B_E·d²` gives inverse demand `p(d) = A_E − B_E·d`. Bounded by `PeakLoad × load_profile`. |

### 3.2 Hydrogen-Sector Agent

| Agent | Type | Description |
|---|---|---|
| `Prod_H2_Green` | `GreenProducer` | PEM electrolyzer with **endogenous H₂ output capacity**. Converts electricity to H₂ with efficiency `η = 1/SpecificConsumption`. Buys elec + elec GCs; sells H₂ + H₂ GCs. Annual green-backing constraint ensures GCs purchased ≥ `(1/η) × GCs issued`. Makes **one** H₂ capacity and investment decision (`cap_H2_y`, `inv_H2_y`), incurring fixed annualised CAPEX `FixedCost_per_MW_Electrolyzer × cap_H2_y`. In contract entry points: receives `g_ppa_from` (PPA, settlement $K^{\mathrm{PPA}}$) and buys `e_in_pool`; sells `h2_hpa` under HPA (`h2_hpa ≤ hpa_cap` with shared $C^{\mathrm{HPA}}$; settlement at $K^{\mathrm{HPA}}$ per volume mode of the entry point). |

### 3.3 Offtaker Agents

| Agent | Type | Description |
|---|---|---|
| `Offtaker_Green` | `GreenOfftaker` | Buys green H₂ and converts it 1:1 (via `Alpha`) to end product. Must buy H₂ GCs for ≥ 42% of EP output (annual mandate `gamma_GC = 0.42`). Tight stoichiometric link: `ep = (1/α) × h2_in`. Has **endogenous EP output capacity** `cap_EP_y` (scalar, non-anticipative) with investment `inv_EP_y` and fixed annualised CAPEX `FixedCost_per_MW_EP_Out × cap_EP_y`. In contract entry points, buys `h2_hpa_from` under HPA (shared $C^{\mathrm{HPA}}$; settlement $K^{\mathrm{HPA}}$ per volume mode) in addition to pool H₂. |
| `Offtaker_Grey` | `GreyOfftaker` | Produces EP from conventional (grey) feedstock at `MarginalCost`. Must buy H₂ GCs for ≥ `gamma_GC × gamma_NH3 × ep` (only the H₂-feedstock fraction). |
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
- **Conventional:** convex **staged** variable cost (piecewise-quadratic stack) — aggregate merit-order approximation.
- **Electrolyzer:** buys elec + elec_GC; sells H₂ + H₂_GC; pays OPEX; CAPEX on H₂ capacity; **green-backing** links GC purchases to H₂ GC issuance.
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

**Transfers.** Revenue for one agent is expenditure for another; they **cancel in social welfare** (§6.3). The planner tracks **real** costs and utilities, not financial transfers.

### 4.5 Risk-adjusted competitive equilibrium ($\gamma < 1$)

With **private CVaR** (ME / ME+C), risk-averse agents minimise $\gamma_{i}\,\mathbb{E}[\ell_{i}] + (1-\gamma_{i})\,\mathrm{CVaR}_{i}(\ell_{i})$ with **full** loss $\ell_{i}$ (operational + $F_{\mathrm{cap}}\cdot\mathrm{cap}$) — still **price-taking**, but **incomplete risk trading** (d'Aertrycke et al.). The **social planner** pools tail risk via **one social CVaR** on aggregate welfare (**complete risk trading**). Definitions, $\gamma$ vs $\beta$, and effects on investment: **§4.10**. Labels: **§4.8**. Planner maths: **§7.4**.

### 4.6 Social planner as the welfare dual

The planner maximises (§7):

$$
\begin{aligned}
\max \;& \gamma \sum_y P_y\,\mathrm{sw}^{aux}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}} \\
\text{s.t.}\;& \text{all agent constraints + market clearing}
\end{aligned}
$$

Per-agent welfare contributions are **utility minus real cost** (no $\lambda$ terms). At $\gamma=1$, if the coupled problem is convex, any competitive equilibrium $(x^{\ast},\lambda^{\ast})$ solves the planner and vice versa (**first welfare theorem**). That is why SP and ME should agree at $\gamma=1$ when ADMM converges.

The planner is **not** an MCP solved directly as complementarity; it is a **mathematical program**. Its KKT multipliers on balance constraints are the **competitive prices** (§7.2).

### 4.7 Contracts case (ME+C): still competitive MCP

`me_pap.jl`, `me_top.jl`, or `me_sop.jl` adds **PPA** and **HPA** bilateral pools (§2):

- Same **price-taking** structure on pool markets and on contract clearing prices $\lambda_{\mathrm{ppa}}$, $\lambda_{\mathrm{hpa}}$.
- **Settlement** at scalar strike $K$ (PPA always PaP; HPA volume mode set by entry point). $K$ and shared $C$ evolve each ADMM iteration; both snapshotted at convergence.
- **Shared contract capacity** $C$ per link — updated outside agent subproblems (`contract_capacity.jl`), not via opposing per-agent cap optimisers.

Economically: extra **coupling constraints** (VRES splits generation; electrolyzer splits intake; etc.) and extra **clearing conditions** for contract energy. Capacity $C$ is coordinated by a **utilisation-based consensus step**, not a separate MW price or Nash bargaining game. At $\gamma < 1$, contracts can appear because **private CVaR** values fixed-price bilateral cash flows; at $\gamma = 1$, unused contracts ($C \approx 0$) are equilibrium-consistent.

### 4.8 Literature labels and price interpretation (d'Aertrycke et al.)

Mapping to d'Aertrycke, Ehrenmann, Ralph & Smeers (2018), *Risk trading in capacity equilibrium models* (see §14):

| Entry point | $\gamma = 1$ | $0 < \gamma < 1$ |
|---|---|---|
| **`social_planner.jl`** | Risk-neutral **competitive capacity equilibrium** (stochastic welfare max; duals = expected marginal social value) | **Competitive capacity equilibrium with complete risk trading** (social CVaR on aggregate welfare; duals = risk-adjusted social shadow prices) |
| **`market_exposure.jl`** | Risk-neutral decentralised competitive equilibrium (ADMM); should match SP | **Competitive capacity equilibrium with incomplete risk trading** (private per-agent CVaR; ADMM $\lambda$ = equilibrium commodity prices for that institution) |
| **`me_pap.jl`, `me_top.jl`, or `me_sop.jl`** | Same as ME, plus PPA/HPA pools | Same incomplete-risk-trading label as ME, with bilateral contract pools |

**Complete risk trading (SP, $\gamma < 1$):** one system-wide CVaR on aggregate welfare — centralised tail-risk pooling.

**Incomplete risk trading (ME / ME+C, $\gamma < 1$):** private CVaR per agent; no modelled risk market. ME+C adds **physical** contracts, not complete financial risk trading.

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

Weather scenarios $y \in JY$ (e.g. ten years 2021–2030 with distinct VRES profiles) create **uncertainty** in revenues and costs. For each risk-averse agent $i$, define **per-scenario loss** $\ell_{i,y}$ (€):

$$
\ell_{i,y} = \underbrace{\sum_{h,d} W_{d,y}\bigl(\mathrm{cost}_{i}(h,d,y) - \mathrm{rev}_{i}(h,d,y)\bigr)}_{\text{operational loss in year }y} + \underbrace{F_i^{\mathrm{cap}}\cdot \mathrm{cap}_i}_{\text{annualised CAPEX}}
$$

**Operational** loss depends on scenario (wind, sun, prices). **Capacity** `cap_i` is one scalar chosen **before** knowing which scenario occurs — so a large investment hurts in **every** scenario if revenues disappoint. CVaR must use this **full** $\ell_{i,y}$ (code: `loss_total[y]`); see §5.1.

A **risk-neutral** agent ($\gamma=1$) cares only about $\mathbb{E}[\ell_i] = \sum_y P_y \ell_{i,y}$. A **risk-averse** agent also dislikes **bad tail outcomes** — years where loss is much worse than average (e.g. low VRES output, high fuel prices, weak margins).

#### 4.10.2 VaR and CVaR — definitions

Fix confidence level $\beta \in (0,1)$.

- **Value-at-Risk (VaR)** at level $\beta$: a threshold $\alpha$ such that loss exceeds $\alpha$ with probability at most $1-\beta$ (in the discrete scenario case, a quantile of the loss distribution).

- **Conditional Value-at-Risk (CVaR)** at level $\beta$: the **expected loss in the worst $(1-\beta)$ fraction** of scenarios (tail average). Also called **Expected Shortfall**. Example: $\beta=0.8$ ⇒ average loss in the worst **20%** of weather years; $\beta=0.2$ ⇒ average in the worst **80%** — a much broader, more conservative tail.

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

**Social planner** maximises welfare with symmetric structure on **social loss** $L_y = -SW_y$:

$$
\max \;\; \gamma \sum_y P_y\,\mathrm{sw}^{aux}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}_\beta(-SW)
$$

$\gamma=1$: maximise expected social welfare only. $\gamma<1$: trade some expected welfare for **lower tail risk** on aggregate outcomes (complete risk trading).

**Why $\gamma=1$ is risk-neutral:** the CVaR term is multiplied by $(1-\gamma)$. At $\gamma=1$ it **drops out of the objective** entirely; optimality is driven only by expected profit/welfare.

#### 4.10.4 Hoschle-style calibration: $\gamma$ then $\beta$

This project follows the **two-step risk calibration** used in Hoschle et al. (2018) and related equilibrium literature:

| Step | Parameter | Setting | Role |
|---|---|---|---|
| **1 — Benchmark** | $\gamma$ | **$1$** | Risk-neutral: only $\mathbb{E}[\ell]$ / expected welfare. SP and ME should agree (§5.4.2). |
| **2 — Turn on CVaR** | $\gamma$ | **$0.5$** | Equal weight on **mean** and **CVaR** in the objective (Hoschle case-study default for risk-averse runs). |
| **3 — Risk-aversion intensity** | $\beta$ | **$0.2,\,0.4,\,0.6,\,0.8$** | At fixed $\gamma=0.5$, sweep $\beta$ to vary how aggressively agents penalise bad scenarios. |

**Lower $\beta$ ⇒ more risk-averse** (Hoschle et al. label $\beta$ as “risk aversion” in their sensitivity figures). Mechanism:

- $\mathrm{CVaR}_{\beta}$ averages loss over the worst $(1-\beta)$ share of scenarios.
- $\beta=0.8$ ⇒ worst **20%** only — mild tail focus within the risk-averse regime.
- $\beta=0.4$ ⇒ worst **60%** — penalises most below-median years.
- $\beta=0.2$ ⇒ worst **80%** — strongest tail penalty in the standard sweep (closest to worst-case among interior $\beta$ values).

**$\gamma$ and $\beta$ are complementary, not interchangeable:**

- **$\gamma$** switches risk aversion **on/off** and sets the **split** between $\mathbb{E}[\cdot]$ and $\mathrm{CVaR}_{\beta}(\cdot)$ in the objective.
- **$\beta$** defines **which part of the loss distribution** enters CVaR once $\gamma<1$. At fixed $\gamma=0.5$, varying $\beta$ is the main way to trace **increasing risk aversion** from mild ($\beta=0.8$) to strong ($\beta=0.2$).

Hoschle et al. sweep $\beta$ from $1$ (risk-neutral reference in their figures) down to $0.1$ with $\gamma=0.5$. This codebase uses **$\gamma=1$** for the risk-neutral benchmark (cleaner: CVaR term vanishes) and **$\beta \in \lbrace 0.2, 0.4, 0.6, 0.8 \rbrace$** at $\gamma=0.5$ for the risk-averse sensitivity — the same economic logic.

In `data.yaml`, both SP and ME read **`ADMM.gamma`** and **`ADMM.beta`** (global defaults; per-agent `gamma`/`beta` in agent blocks override for ADMM agents). Defaults: `gamma: 1.0`, `beta: 0.95` (placeholder when $\gamma=1$; set explicitly when running risk-averse cases).

**Multi-scenario requirement:** with `nYears=1` (one scenario), $\mathrm{CVaR}=\ell$ always — changing $\gamma$ has **no effect** on the optimum if `loss_total` is specified correctly (§5.1). Risk aversion is meaningful when **`nScenarioYears` > 1** (e.g. ten weather years).

#### 4.10.5 Complete vs incomplete risk trading (reminder)

| | **SP ($\gamma<1$)** | **ME ($\gamma<1$)** |
|---|---|---|
| **Who is risk-averse?** | Society once (social CVaR on $SW_y$) | VRES, electrolyzer, green offtaker **separately** (private CVaR) |
| **Risk market?** | Complete pooling (centralised) | No explicit risk trading |
| **Compare quantities to SP?** | SP is benchmark | Generally **no** (§4.8) |
| **Compare `Risk_Metrics.csv`?** | Yes — ex-post social CVaR gap (§7.6) | Yes |

#### 4.10.6 What changes in strategy when $\gamma < 1$ or $\beta$ falls?

Risk aversion reshapes **capacity**, **dispatch**, and **prices** because bad scenarios get **more weight** in the optimiser (directly via private CVaR in ME, via social CVaR in SP). **Within** $\gamma=0.5$ runs, **decreasing $\beta$** strengthens this effect: CVaR averages over a **larger** set of bad years, so capacity and dispatch shift further toward hedging tail losses (Hoschle et al. Fig. 6: installed capacity moves monotonically as $\beta$ decreases).

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

1. **Risk-neutral benchmark:** `gamma = 1` (any `beta`; inactive). Run SP then ME — verify convergence and quantity/price match (§5.4.2). Requires `nScenarioYears > 1` for meaningful multi-scenario dispatch; risk parameters only matter when $\gamma<1$.
2. **Risk-averse base case:** `gamma = 0.5` in the `ADMM` block (applies to SP and ME simultaneously). Ensure `nScenarioYears > 1` (e.g. 10 weather years).
3. **Risk-aversion sweep:** at fixed `gamma = 0.5`, run separate cases with `beta = 0.2, 0.4, 0.6, 0.8` — **lower `beta` = more risk-averse**. Compare capacities, prices, and `Risk_Metrics.csv` across the sweep.
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

Each agent minimises its **augmented Lagrangian** (possibly risk-averse for some agents):

$$
\begin{aligned}
\min \quad & \gamma_i \sum_{h,d,y} W_{d,y}\bigl(\mathrm{cost}_i(h,d,y) - \mathrm{rev}_i(h,d,y)\bigr) + F_i^{\mathrm{cap}} \\
& \quad + (1-\gamma_i)\,\mathrm{CVaR}_{i}(\ell_i) \\
& \quad + \sum_k \frac{\rho_k}{2}\sum_{h,d,y} W_{d,y}\bigl(g_i^k(h,d,y)-\bar{g}_i^k(h,d,y)\bigr)^2
\end{aligned}
$$

where (symbols map to code names in backticks):
- `cost_i − revenue_i` is the agent's private cost minus revenue across all markets.
- `g_i^k` is the agent's net position in market `k` (positive = supply, negative = demand).
- `ḡ_i^k` is the consensus target for agent `i` in market `k`.
- `ρ_k` is the penalty weight for market `k`.
- `W[d,y]` scales representative days to a full year.
- `γ_i` is a **per-agent risk weight** (`γ=1` → risk-neutral, `γ<1` → risk-averse). Non-trivial CVaR is used only for VRES, electrolyzer, and green offtaker.
- $\mathrm{CVaR}_{i}(\ell_i)$ is an agent-specific Conditional Value-at-Risk on yearly loss scenarios, with auxiliary variables $\alpha_i$, $u_i(y)$ over $y \in JY$, at confidence level $\beta$.

More explicitly:

- The **deterministic, expected-loss term**

$$
\sum_{h,d,y} W_{d,y}\,\bigl(\mathrm{cost}_i(h,d,y) - \mathrm{rev}_i(h,d,y)\bigr)
$$

contains fuel/operational costs, certificate purchases, and investment annuities on the **cost** side, and all market revenues (price × net position) on the **revenue** side.

- The **risk term** $\mathrm{CVaR}_{i}(\ell_i)$ captures the tail of the loss distribution over years $y$. It is only active when $\gamma_i < 1$; for $\gamma_i=1$ the CVaR part drops out and the agent becomes risk-neutral.

- The **quadratic ADMM penalties**

$$
\sum_k \frac{\rho_k}{2}\sum_{h,d,y} W_{d,y}\,\bigl(g_i^k(h,d,y)-\bar{g}_i^k(h,d,y)\bigr)^2
$$

ensure that, in equilibrium, each agent’s net position $g_i^k$ coincides with a consensus allocation $\bar{g}_i^k$ that satisfies market-clearing. Economically, this can be read as a **soft enforcement of market balance**: deviating from the consensus quantity becomes increasingly expensive as $\rho_k$ grows.

The ADMM penalties are **algorithmic** only (§4.2, §6); at convergence they vanish and the solution is the **risk-adjusted competitive MCP** of §4.

#### CVaR formulation (per agent)

For each risk-averse agent (VRES, electrolyzer, green offtaker), CVaR is linearised via:

**Important**: The loss that enters CVaR must be the **full** per-scenario loss, including the fixed capacity cost (`F_cap × cap`). Capacity `cap` is a **scalar** (non-anticipative: the same installed MW in every weather scenario). If only the operational loss is used, then when $\gamma < 1$ the fixed cost appears only in the $\gamma$-weighted term, so the effective weight on `F_cap` becomes $\gamma$ instead of $1$. With one scenario, changing $\gamma$ would then change the objective, breaking the equivalence between social planner and market exposure. The correct formulation uses `loss_total[y] = loss_operational[y] + F_cap × cap` in the CVaR shortfall constraints (same `cap` in every scenario). The $\gamma$-weighted expected term is `F_cap × cap + Σ_y P_y × loss_operational[y]`. With one scenario, $\mathrm{CVaR}_{i} = \ell_i$, so the objective reduces to total loss regardless of $\gamma$.
- `α_i` — VaR proxy (free variable, `≥ 0`)
- `u_i[jy]` — shortfall per scenario year (`≥ 0`)
- `cvar_i` — CVaR value (`≥ 0`)

Constraints (code names in backticks; mathematical form):

$$
\begin{aligned}
u_{i,y} &\ge \ell_{i,y} - \alpha_i \quad \forall y \in \mathcal{Y} \\
\mathrm{CVaR}_{i} &\ge \alpha_i + \frac{1}{1-\beta}\sum_{y \in \mathcal{Y}} P_y\, u_{i,y}
\end{aligned}
$$

**Dynamic constraint updates**: In ADMM, the loss expressions `loss_i[jy]` depend on iteration-specific market prices `λ` (which change every iteration). Because JuMP expressions bake in coefficient values at creation time, the CVaR shortfall and linking constraints must be **deleted and re-added** in every ADMM iteration with the freshly recomputed loss expressions. This happens in the `solve_*_agent!` functions.

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

**VRES in contracts case** (`build_power_agent_contracts.jl`): Splits generation into `g_EOM` (pool) and `g_ppa` (PPA). Loss includes `−K_ppa×g_ppa`; penalties add `(ρ_ppa/2)×Σ W×(g_ppa − ḡ_ppa)²`. Constraint `g_ppa ≤ ppa_cap`; `ppa_cap` **fixed** each iteration to shared $C$ (no `ρ_ppa_cap` penalty).

**Conventional generator (3-stage increasing cost):**

```text
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

**GreenProducer in contracts case** (`build_H2_agent_contracts.jl`): Uses `e_in_pool` (pool) and `g_ppa_from` (PPA). Loss includes `+K_ppa×g_ppa_from`; conversion `h2_out = η×(e_in_pool + g_ppa_from)`; penalties add `(ρ_ppa/2)×Σ W×(−g_ppa_from − ḡ_ppa)²` only (no `ρ_ppa_cap`; `ppa_cap` fixed to shared $C$). Sells `h2_hpa` under HPA with `−K_hpa×h2_hpa` and `(ρ_hpa/2)×Σ W×(h2_hpa − ḡ_hpa)²`; `hpa_cap` fixed to shared $C^{\mathrm{HPA}}$ (no `ρ_hpa_cap`).

**GreenOfftaker in contracts case** (`build_offtaker_agent_contracts.jl`): Buys `h2_hpa_from` under HPA (`h2_hpa_from ≤ hpa_cap` with shared $C^{\mathrm{HPA}}$ fixed each iteration). Settlement follows the entry-point volume mode (PaP / ToP / SoP) at $K^{\mathrm{HPA}}$; penalties add `(ρ_hpa/2)×Σ W×(−h2_hpa_from − ḡ_hpa)²` only.

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
| VRES capacity | `g ≤ AF × Capacity` | Per (h,d,y) | Generation limited by resource availability |
| Conventional staging | `g = Σ_s g_s`, `0 ≤ g_s ≤ cap_s` | Per (h,d,y) | Piecewise thermal stack with increasing stage costs and fixed total capacity |
| Consumer load | `d ≤ PeakLoad × load_profile` | Per (h,d,y) | Maximum consumption bound |
| H₂ conversion | `h2_out = η × e_in` | Per (h,d,y) | Stoichiometric mass/energy balance |
| H₂ GC physical limit | `gc_h2 ≤ h2_out` | Per (h,d,y) | Cannot certify more than produced |
| Green-backing (annual) | `Σ W×gc_elec ≥ (1/η)×Σ W×gc_h2` | Per year | Temporal flexibility in GC procurement |
| Green offtaker stoichiometry | `ep = (1/α) × h2_in` | Per (h,d,y) | No H₂ waste; tight conversion |
| GC mandate (green/grey) | `Σ W×gc_h2 ≥ γ_GC × Σ W×ep` | Per year | 42% renewable mandate |
| Grey GC mandate | `Σ W×gc_h2 ≥ γ_GC × γ_NH3 × Σ W×ep` | Per year | Only H₂-feedstock fraction |

### 5.3 Social Planner Objective

The social planner maximises **risk-adjusted social welfare** with a **single** social CVaR:


$$
\max \; \gamma \sum_y P_y\,\mathrm{sw^{aux}}_y \;-\; (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}
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
\gamma\,\mathbb{E}\bigl[SW\bigr] - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}(-SW),
$$

where $SW$ is aggregate welfare (including consumer utility and production/investment costs).
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

$$
\|r_k^t\|_2 = \Bigl(\sum_{h,d,y} r_k^t(h,d,y)^2\Bigr)^{1/2},\qquad \|s_k^t\|_2 = \rho_k^t\,\Bigl(\sum_{h,d,y} (\Delta z_k^t(h,d,y))^2\Bigr)^{1/2}.
$$

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

### 6.2 Consensus Formula (Sharing ADMM)

The consensus target for agent $i$ in a market with $n$ participants:

$$
\bar{g}_i^k = q_i^{k-1} - \frac{1}{n+1}\sum_j q_j^{k-1}
$$

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

5. **Layered stabilisers in `ADMM.jl` (beyond `update_rho!`):**
   - **η damping** on $\lambda$ updates near tolerance (§6.1 step 4);
   - **η_scale** per market — shrinks if merit worsens iteration-on-iteration;
   - **Local basin guards** — blend $\lambda$ and $\rho$ back toward per-market best merit if drift exceeds 25%;
   - **H2_GC price projection** — $\lambda_{H2\_GC}\ge 0$ (projected ADMM).

`ρ` initial values come from `data.yaml` (`rho_initial` per market, `rho_cap_initial` for capacity). They are **numerical knobs**, not calibrated economic parameters.

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

#### 6.4.2 Residuals

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

#### 6.4.3 Stopping rule

Convergence is checked **per agent**, not on the aggregate. For each capacity-owning agent the Boyd absolute + relative test:

```
ε_pri_m  = ε_abs · sqrt(n_yr) + ε_rel · ResidualScale_Primal_m
ε_dual_m = ε_abs · sqrt(n_yr) + ε_rel · ResidualScale_Dual_m
```

with `ResidualScale_*_m` initialised from the first non-zero observation per agent. Capacity is converged iff `r_m ≤ ε_pri_m` and `s_m ≤ ε_dual_m` for **every** `m`. *Why per-agent and not aggregate*: averaging residuals across agents can hide a single laggard whose split is still far from feasibility; an aggregate test would declare convergence even when one agent type (e.g. a strongly binding electrolyzer) has not satisfied the equality. The per-agent test is direction-correct: capacity is "done" when every agent's split is satisfied.

The optional knob `cap_tol_relax` (default **25** in the contracts case, **1** in plain ME) multiplies the right-hand side of the per-agent **physical investment capacity** test only; see §6.7. Contract capacity $C$ uses separate alignment residuals (§2, §6.7).

#### 6.4.4 Per-agent ρ controller

Each capacity-owning agent follows the same minimal residual-balancing rule as §6.3, applied per agent `m`:

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

Each ADMM iteration `k` for the capacity block runs:

1. **Derive `z^k`**: `ADMM_subroutine` computes `z_m^k` for every cap agent from realized flow histories (fallback to ADMM targets when history is not yet available) and pushes it to history (`ADMM["Capacity"]["z"][m]`).
   - `z` uses optional under-relaxation
     `z^k <- α·z_raw^k + (1-α)·z^{k-1}` with `α = cap_z_relax` (default 1.0 = off),
     then re-projects the agent's model-feasible minimum installed-capacity floor (from nonnegative investment).
2. **Set parameters on agent model**: `:z_cap = z_m^k`, `:λ_cap = λ_m^{k-1}` (read from history), `:ρ_cap = ρ_m^{k-1}` (read from history).
3. **Agent solves**: produces `x_m^k`.
4. **Dual ascent**: `λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k)`, pushed to history.
5. **Residuals**: `r_m^k`, `s_m^k` computed and pushed.
6. **Controller**: `update_rho!` updates `ρ_m^k` per agent using residual balancing.
7. **Convergence**: per-agent test (§6.4.3).

This ordering is identical for `market_exposure` and the ME contract cases (`me_pap`, `me_top`, `me_sop`); only the `z` derivation differs (the contracts case adds the PPA / HPA flow contributions when computing the peak of `g_bar + g_bar_ppa`, etc.).

**Why this choice (`z` under-relaxation):**

In tightly-coupled runs, raw `z` targets can jump sharply when flow consensus oscillates across markets. Because the capacity dual residual uses `Δz`, these jumps can produce very large `s_m` and trigger controller overreaction even when `x` is moving in the right direction. Under-relaxation damps target motion, reducing artificial dual spikes and improving monotonic progress toward the split fixed point.

`z` projection enforces feasibility against the model structure: minimum installed-capacity floor implied by nonnegative investment variables (scalar `cap` per agent).

### 6.5 Convergence Tolerances (Boyd-style)

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

The **ME contract cases** (`me_pap`, `me_top`, `me_sop`) have more coupled markets (standard flows + contract energy + contract capacity + capacity consensus) and stronger interdependence (VRES splits pool vs contract; electrolyzer does the same). As a result, convergence is slower and residuals tend to be larger than in `market_exposure`. To avoid running to `max_iter` without declaring convergence when results are already good enough, the contracts case uses a separate tolerance:

- **`epsilon`** — Used by `market_exposure`.
- **`epsilon_contracts`** — Used by `me_pap.jl` / `me_top.jl` / `me_sop.jl` when set in `data.yaml`. If not set, the contracts case falls back to `epsilon`.

Both cases use the same convergence logic; only the tolerance value differs. **`cap_tol_relax`** relaxes **physical investment capacity** consensus only (§6.4, §6.7); shared contract capacity $C$ is checked separately via $|C - q^{\mathrm{peak}}|$ and $|C^k - C^{k-1}|$.

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

- Primal warm-start requires **matching horizon**: `nTimesteps × nReprDays × nYears` rows in CSV must equal ADMM shape. Multi-scenario ME (`nScenarioYears=10`) needs SP outputs with the same year dimension (or primal warm-start is skipped with a warning).
- SP prices are loaded **as-is** (no H2_GC clamp at load); negative H2_GC at SP optimum would be inconsistent with equilibrium — the **floor** is applied after each λ update in `ADMM.jl`.
- Console message when all three load: `ADMM warm-start: λ from SP prices, primal quantities from SP, capacity seeds for N agents`.
- If `social_planner_results/` is missing, ADMM still runs from `initial_price` scalars — valid, but use for debugging only when benchmarking against SP.

### 6.7 Contract Pools ADMM (me_pap / me_top / me_sop)

The contracts loop (`ADMM_contracts.jl`) extends the standard ME loop. **All three entry points** use the same contract ADMM; only HPA settlement (PaP / ToP / SoP) differs.

**Per iteration:**

| Step | What happens |
|------|----------------|
| 1 | `apply_shared_contract_caps!` — fix `ppa_cap` / `hpa_cap` to `SharedCap` |
| 2 | `ADMM_subroutine_contracts!` per agent — refresh $\bar g$, $\lambda$, provisional $K$; solve (Gurobi) |
| 3 | Contract **energy** imbalances: `g_ppa` vs `g_ppa_from`; `h2_hpa` vs `h2_hpa_from` |
| 4 | `update_shared_contract_capacity!` — bargaining update for $C$; `record_shared_cap_residuals!` |
| 5 | $\lambda^{\mathrm{contract}}$ dual ascent (same logic as spot markets, with η damping) |
| 6 | `update_rho_contracts.jl` — adapt $\rho$ on energy and cap-alignment residuals |

**Markets tracked:**

- **PPA/HPA energy** (`ppa`, `hpa`): 3D imbalances; $\lambda^{\mathrm{ppa}}_i$, $\lambda^{\mathrm{hpa}}_j$ prices; Boyd residuals like spot markets.
- **Shared cap** (`ppa_cap_*`, `hpa_cap_*` in `ADMM_Convergence.csv`): scalar $|C - q^{\mathrm{peak}}|$ and $|C^k - C^{k-1}|$ — **not** supplier-cap minus buyer-cap.
- **Physical investment capacity** (`cap_*` agents): unchanged from `market_exposure.jl` (§6.4, §7.4).

**Tolerances** (contracts case is more coupled; see §6.5):

- **`epsilon_contracts`** — Base tolerance for flow markets (default 0.5 vs 0.1 for plain ME).
- **`cap_tol_relax`** — Multiplier for **physical** capacity consensus only (default 25 in `data.yaml`).

Full economic interpretation of $C$, $K$, and $\gamma$: §2 *Contract pools*. Configuration: `ADMM.contract_cap`, `PPAs`, `HPAs` in `data.yaml`.

### 6.8 Sign Convention

| Role | Net position sign | Example |
|---|---|---|
| Supplier / seller | **Positive** | VRES generation `+g`, H₂ sales `+h2_out` |
| Buyer / consumer | **Negative** | Electricity demand `−d`, H₂ purchase `−h2_in` |

Market imbalance = Σ (net positions). Positive imbalance = excess supply → price decreases. Negative imbalance = excess demand → price increases.

### 6.9 Practical Convergence Behavior and Monotonicity

With coupled multi-market ADMM (especially with endogenous investments and contract couplings), strict **per-iteration monotonic decrease** of every residual is generally not guaranteed. What the controller is designed to guarantee in practice is stronger **best-so-far progress** and anti-stall recovery:

- Residual merit is tracked in normalized form (relative to market-specific Boyd tolerances).
- If short-term residual motion worsens, per-market dual step scales are reduced automatically.
- If a long stall/worsening phase is detected, the algorithm restarts from the best checkpoint found so far and continues with smaller steps.

This design avoids the common "improve-then-wander" ADMM behavior in hard regimes while preserving convergence speed in easy regimes. In empirical runs, this yields:

- fast coarse convergence in early iterations,
- fewer late oscillation plateaus,
- improved ability to reach tighter `epsilon` values without inflating tolerance.

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
- **η damping** and basin guards — engineering stabilisers for tightly coupled energy markets; not in the original Boyd proof but standard in applied ADMM.
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
- **Risk-averse ($\gamma < 1$):** each dual is the **risk-adjusted marginal social value** — the marginal impact on the planner objective $\gamma \sum_y \mathrm{sw}^{aux}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}$ from relaxing that balance by one unit. These are the correct **commodity shadow prices** for the complete-risk-trading benchmark; they are **not** required to equal ADMM prices when agents use private CVaR (§4.8).

The social CVaR enters through `sw_aux` and linear shortfall constraints (§7.4); the balance constraints themselves are unchanged. The epigraph binds at optimality (`sw_aux[y] = social_welfare[y]` in code; $\mathrm{sw^{aux}}_y = \mathrm{socialWelfare}_y$ mathematically), so risk adjustment affects **levels** of prices and quantities, not the fact that balance duals are well-defined shadow prices.

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
- **Risk-averse ($\gamma < 1$):** $\mu_y = \gamma P_y + \xi_y$, where $\xi_y$ is the CVaR tail contribution (dual of the shortfall constraint $u_y \ge -\mathrm{sw}^{aux}_y - \alpha_{\mathrm{social}}$).

In code (`save_social_planner_results.jl`), $\mu_y$ is read as the dual of the epigraph constraint `sw_aux[y] ≤ social_welfare[y]`. At optimality this dual equals the full marginal weight on scenario $y$ in the planner objective ($\gamma P_y + \xi_y$). If that dual is numerically zero, the implementation falls back to $\gamma P_y$.

The epigraph supplies $\mu_y$ for this normalization as well as the linear CVaR structure (§7.4); it is part of the single QCP solve, not a separate reformulation step.

Why IPOPT (and not Gurobi) for SP duals:
- In this project’s large/scaled SP QCP instances, Gurobi can return primal-optimal status (`LOCALLY_SOLVED`) while still failing to expose usable QCP duals after tightened barrier settings.
- The social planner benchmark requires reliable dual multipliers for market-price comparison; IPOPT delivers these multipliers directly for the solved QCP in this workflow.
- ADMM subproblems use Gurobi; `social_planner.jl` uses IPOPT by default (`SocialPlanner.solver` in `data.yaml`).

#### IPOPT settings and convergence tolerance (`ipopt_tol`)

Solver options live in the `SocialPlanner` block of `data.yaml` and are applied in `social_planner.jl`. The default **`ipopt_tol: 1.0e-6`** is a deliberate choice for this model’s economic scales — not a loose “good enough for debugging” setting.

**What `tol` means.** IPOPT’s `tol` is a **KKT residual tolerance**: it bounds combined primal/dual infeasibility and complementarity at the reported solution. It is **not** a direct “price must be within X €/MWh” knob. Still, for a well-scaled convex QCP, a satisfied KKT tolerance of order $10^{-6}$ implies that reported shadow prices and welfare are accurate **far below** any resolution used in reporting or ADMM warm-start.

**Why `1e-6` is sufficient here.** Typical magnitudes from NL-calibrated risk-neutral and risk-averse social-planner runs ($\gamma=0.5$, $\beta\in\{0.2,\ldots,0.8\}$) and the implied absolute error if one (conservatively) treats `tol` as a relative scale on each quantity:

| Quantity | Typical magnitude | Order-of-magnitude error at `tol = 1e-6` |
|---|---|---|
| Electricity price | ~67 €/MWh | ~$7\times10^{-5}$ €/MWh (~0.007 cent/MWh) |
| H₂ price | ~110 €/MWh | ~$1.1\times10^{-4}$ €/MWh |
| EP price | ~173 €/MWh | ~$1.7\times10^{-4}$ €/MWh |
| Risk-adjusted social welfare | ~50 bn € | ~50 k€ |
| VRES installed capacity | ~20 GW | ~20 kW |

These residuals are negligible for:

- **Price CSVs and plots** (€/MWh, two significant figures in practice).
- **ADMM warm-start** from `Market_Prices.csv` and `SP_Capacities.csv` (§6.6).
- **Welfare and CVaR comparisons** between SP and ME at the bn-€ scale.

**Why not `1e-8` by default?** Tighter KKT tolerance mainly buys marginally cleaner dual multipliers. On the **stiff social CVaR QCP** (epigraph + tail constraints, especially at **low $\beta$** with few scenarios), `1e-8` often makes IPOPT declare `LOCALLY_INFEASIBLE` even when the primal allocation is already economically meaningful. That failure mode triggered unnecessary auxiliary solves and retries without improving reported prices in any material way.

**Default workflow (no extra warm-start).** At `ipopt_tol = 1e-6`, a **single IPOPT pass** usually suffices. CVaR variable seeds in `build_social_planner.jl` provide a feasible-ish starting point without a separate $\gamma=1$ auxiliary solve. Optional knobs:

| Parameter | Default | Role |
|---|---|---|
| `ipopt_tol` | `1.0e-6` | Primary KKT tolerance |
| `ipopt_max_iter` | `5000` | Iteration cap |
| `ipopt_print_level` | `0` | `0` = silent; `3`–`5` = IPOPT log |
| `risk_warmstart` | `false` | If `true` and $\gamma<1$: extra $\gamma=1$ solve, copy primals only |
| `risk_warmstart_beta` | `0.95` | $\beta$ for that auxiliary solve |
| `ipopt_retry_tol` | `1e-5` | Used only if the first solve fails (code default) |
| `ipopt_retry_max_iter` | `8000` | Iteration cap on retry (code default) |

Set `risk_warmstart: true` only for numerically difficult cases (e.g. very low $\beta$ with few scenarios). See also **§9.2.1**.

#### ADMM note on capacity tolerance scaling

In `market_exposure` ADMM, flow-market tolerances use Boyd-style horizon scaling (`ε_abs * sqrt(n_slots) + ε_rel * scale`), where `n_slots = nHours × nReprDays × nYears`.  
Capacity consensus is not a full flow tensor; it is low-dimensional (yearly scalar/vector). Therefore, capacity convergence is checked on a scalar basis (`sqrt(1)`), avoiding premature convergence declarations from over-loose `sqrt(n_slots)` scaling on the capacity channel.

### 7.3 Code Architecture

All problem definition lives in `Source/build_*.jl` files. Each file contains:

- `build_*_agent!()` — Builds the ADMM version (with `λ`, `ρ`, `ḡ` penalty terms and per-agent CVaR for risk-averse agents).
- `add_*_agent_to_planner!()` — Adds the same variables/constraints to the planner model **without** ADMM terms and **without** per-agent CVaR. Returns a `Dict{Int, Any}` of per-year welfare expressions.

`build_social_planner.jl` orchestrates the calls to all `add_*_to_planner!` functions, adds market-clearing constraints, aggregates per-year welfare into `social_welfare`, adds the epigraph formulation and single social CVaR, and sets the risk-adjusted objective.

### 7.4 Epigraph Formulation for Social CVaR

The social planner applies **one single CVaR** to the aggregate social welfare (not per-agent CVaR). This ensures risk aversion considers all welfare components (consumer utility, production costs, investment costs) holistically.

**Problem**: `social_welfare[y]` includes quadratic terms from both elastic demand utility (`A·d − B/2·d²`) and conventional stage costs (`base_s·q_s + 0.5·slope_s·q_s²`). Putting `−social_welfare[y]` directly inside the CVaR shortfall constraints would place those quadratics in the CVaR block.

**Solution — epigraph reformulation**: Introduce auxiliary variables `sw_aux[y]` (math: $\mathrm{sw}^{aux}_y$) with epigraph constraints:

$$
\begin{aligned}
& \mathrm{sw}^{aux}_y \le \mathrm{socialWelfare}_y \quad \forall y \in \mathcal{Y} \\
& \quad\text{(quadratic in $\mathrm{socialWelfare}_y$; convex QC)}
\end{aligned}
$$

The CVaR constraints then reference `sw_aux` instead of the quadratic `social_welfare`, making them purely linear:

$$
\begin{aligned}
u_y &\ge -\mathrm{sw}^{aux}_y - \alpha_{\mathrm{social}} \quad \forall y \in \mathcal{Y} \\
\mathrm{CVaR}^{\mathrm{social}} &\ge \alpha_{\mathrm{social}} + \frac{1}{1-\beta}\sum_{y \in \mathcal{Y}} P_y\, u_y
\end{aligned}
$$

**Important**: `α_social` and `cvar_social` must be **free** (no lower bound). When social welfare is positive, the social loss $-\mathrm{sw}^{aux}_y$ is negative. The optimal VaR $\alpha$ for CVaR of a negative loss is negative. With $\alpha \ge 0$, $\mathrm{CVaR}^{\mathrm{social}}$ would be forced $\ge 0$, so the objective would become $\gamma \sum_y \mathrm{sw}^{aux}_y$ instead of $\sum_y \mathrm{sw}^{aux}_y$ when $\gamma < 1$ — breaking SP/ME equivalence for `nYears = 1`. With $\alpha$ free, $\mathrm{CVaR}^{\mathrm{social}}$ equals social loss when there is only one scenario, so the objective reduces to $\sum_y \mathrm{sw}^{aux}_y$ regardless of $\gamma$.

The objective is also linear:

$$
\max \;\gamma \sum_{y \in \mathcal{Y}} \mathrm{sw}^{aux}_y - (1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}
$$

Since the objective maximises `sw_aux`, the epigraph constraint binds at optimality (`sw_aux[y] = social_welfare[y]` in code; $\mathrm{sw}^{aux}_y = \mathrm{socialWelfare}_y$ in math), making the formulation mathematically equivalent to applying CVaR directly to `social_welfare`.

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

- **Market exposure (ADMM)**: Each agent holds scalar `cap` in its decentralised model. Agents must agree on a **consensus capacity** via an ADMM penalty: `(ρ_cap/2) × (cap − z_cap)²`, where `z_cap` is the capacity implied by flow consensus over **all** scenarios (e.g. for VRES: `z_cap = max over (h,d,y) of g_bar[h,d,y] / AF[h,d,y]`). At convergence, all agents choose the same capacity and `z_cap` matches the agreed-upon level.

**Why warm-start matters for investment**: Without capacity warm-start from the SP, the first ADMM iteration has `z_cap` derived from zero flows (ḡ = 0), so `z_cap = 0`. Agents are then penalised toward zero capacity, which is far from the equilibrium. With SP capacity seeds (`set_start_value`) and primal warm-start (ḡ = SP), `z_cap` is consistent with SP flows and the capacity penalty pulls agents toward the SP investment levels from the first iteration. This dramatically speeds convergence of the investment consensus.

### 7.6 Risk metrics post-processing (CVaR reporting)

After each run, the project writes **`Risk_Metrics.csv`** and **`Social_Welfare_Per_Year.csv`** and prints a **risk metrics** block to the console. Implementation: `Source/compute_social_risk_metrics.jl`, called from `save_social_planner_results.jl`, `save_results.jl`, and `save_results_contracts.jl`.

#### What is reported

| Quantity | Social planner (SP) | ADMM (ME / ME+C) |
|---|---|---|
| **Expected social welfare** $E[SW]=\sum_y P_y\,SW_y$ | From solved `sw_aux` (binds to aggregate welfare) | **Ex-post**: recomputed from converged quantities using the same planner welfare accounting (no $\lambda$ transfers) |
| **Social CVaR** | Value of `CVaR_social` from the solved planner | **Ex-post social CVaR**: same Rockafellar formula as SP applied to $L_y=-SW_y$ from the ADMM allocation |
| **$\alpha$ (VaR proxy)** | `alpha_social` from the planner | From the ex-post CVaR calculation |
| **Sum of private agent CVaRs** | n/a | $\mathrm{CVaR}_{\mathrm{VRES}}+\mathrm{CVaR}_{\mathrm{H2}}+\mathrm{CVaR}_{\mathrm{Green}}$ at ADMM convergence (internal to agent problems) |
| **Gap vs SP** | 0 | `social_CVaR_gap_vs_SP` = ex-post ADMM social CVaR minus SP social CVaR (requires `social_planner_results/Risk_Metrics.csv` from a prior SP run) |

**Important distinctions:**

1. **Expected social welfare** is the **probability-weighted mean** $\sum_y P_y SW_y$, not the welfare in a single “most likely” year. With uniform $P_y=1/n_Y$, it is the arithmetic average across scenario years.

2. **Social CVaR** in the code is **CVaR of social loss** $L_y = -SW_y$ (tail of bad aggregate outcomes). The planner **minimizes** $(1-\gamma)\,\mathrm{CVaR}^{\mathrm{social}}$ in the objective (equivalently penalizes bad tails). **Lower social CVaR is better** (less tail risk).

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
| `Risk_Metrics.csv` | One row per metric (`expected_social_welfare`, `social_CVaR`, `alpha_social`, `sum_private_CVaR`, `social_CVaR_gap_vs_SP`, …) |
| `Social_Welfare_Per_Year.csv` | `scenario_year`, `probability`, `social_welfare`, `social_loss` per year |
| `Private_CVaR_By_Agent.csv` | ADMM only: per risk-averse agent `CVaR_private` and `alpha_private` |

#### Console log

Logs report **positive welfare** (bn EUR): expected welfare, **tail welfare** (= average welfare in the worst $(1-\beta)$ share of scenario years; higher is better), min scenario welfare, and spread. Internally the solver uses loss $L_y=-SW_y$; `Risk_Metrics.csv` still stores `social_CVaR` on that loss. ADMM runs add **SP tail welfare (benchmark)** and **tail welfare gap vs SP** (negative ⇒ ME worse on aggregate tail).

Example block:

```
------------------------------------------------------------------------
  Social planner risk metrics
------------------------------------------------------------------------
  Case:                    social_planner
  gamma:                   0.5000
  beta:                    0.8000  (tail = worst 20% of scenario years)
  E[social welfare]:               50.102 bn EUR
  Tail welfare (CVaR):             49.149 bn EUR  (higher = safer tail)
  Min scenario welfare:            48.593 bn EUR
  Welfare spread (max−min):         2.965 bn EUR
  Risk-adjusted objective:         49.626 bn EUR
  (Placeholder scenario years — if welfare is similar across years, risk shifts stay small.)
```

---

## 8. Data and Indexing

### 8.1 Temporal Dimensions

| Dimension | Set | Size | Description |
|---|---|---|---|
| Hours | `JH = 1:nTimesteps` | 24 | Hours within each representative day |
| Representative days | `JD = 1:nReprDays` | 8 | Representative days (configured in `data.yaml`) |
| Weather scenarios | `JY = 1:nYears` | typically 10 | Parallel weather scenarios (`ADMM.nScenarioYears` when set, for both SP and ME); **not** sequential investment years |

### 8.2 Representative-Day Weights

`W[jd, jy]` = number of real calendar days that representative day `jd` stands for in year `jy`. Used to scale per-representative-day objective values to a full-year total. Weights are read from `Input/output_<scenario>/decision_variables_short.csv` and always sum to **365** per scenario. See §9.7 for how they are computed.

### 8.3 Scenario Labels (JY mapping)

`years = Dict(1 => 2021, 2 => 2022, ...)` maps scenario index to a **scenario label** used to load CSV files (`timeseries_<label>.csv`, `output_<label>/…`). These labels are **not calendar dates**: scenario `2021` is the baseline weather scenario (and matches `base_year` for installed-capacity calibration), while `2022`–`2030` are nine additional weather scenarios with distinct VRES profiles. **Investment is decided once** before operations; scenarios differ only in availability factors, dispatch, and scenario-weighted expected profit / CVaR. See §7.5 and §9.7.

### 8.4 3D Arrays

All prices, quantities, and imbalances are stored as 3D arrays `[jh, jd, jy]`. Scalar diagnostics (mean price, mean imbalance) are computed per iteration for CSV output.

---

## 9. Configuration Reference (data.yaml)

### 9.1 General

| Parameter | Value | Description |
|---|---|---|
| `nTimesteps` | 24 | Hours per representative day (hourly resolution) |
| `nReprDays` | 8 | Representative days (trade-off: speed vs. accuracy) |
| `nYears` | 1 | Base-year horizon used by `social_planner.jl` |
| `base_year` | 2021 | `[NL]` Scenario label for baseline timeseries + installed capacities; see §9.6–§9.7 |

### 9.2 ADMM

| Parameter | Value | Description |
|---|---|---|
| `rho_initial` | 1.0 | Default penalty weight (neutral starting point) |
| `nScenarioYears` | 10 | Scenario years used by `market_exposure*.jl` (e.g., 2021..2030) |
| `max_iter` | 200 | Maximum ADMM iterations |
| `epsilon` | 0.2 | Convergence tolerance for `market_exposure`; see §6.5 for accuracy/speed trade-off. |
| `epsilon_contracts` | 0.5 | [me_pap / me_top / me_sop] Contracts tolerance; looser than ME (`epsilon` 0.1). |
| `cap_tol_relax` | 25 | [me_pap / me_top / me_sop only] Multiplier for **physical investment capacity** consensus (§6.4). Does **not** apply to shared contract $C$. |
| `contract_cap.relaxation` | 0.35 | Damping $\tau$ in shared-$C$ bargaining update (§2). |
| `contract_cap.expand_step` | 2.0 | MW expansion per unit mean contract energy imbalance when cap binds. |
| `contract_cap.bind_tol` | 1e-6 | Numerical tolerance: treat $q^{\mathrm{peak}} \approx C$ as binding. |
| `rho_cap_initial` | 0.1 | Initial per-agent capacity penalty for the equality split (§6.4). |
| `rho_cap_inc_factor` | 1.05 | Per-agent capacity controller increase factor; decrease factor is the reciprocal. See §6.4.4. |
| `rho_cap_max` | 30 | Per-agent capacity penalty upper bound. See §6.4.4 for justification. |
| `cap_z_relax` | 1.0 | Under-relaxation factor for capacity target update `z^k <- α z_raw^k + (1-α) z^{k-1}`. `1.0` disables damping (default). Use `0.2–0.8` only if target oscillations cause large `Δz` dual spikes. See §6.4.8. |
| `gamma` | 1.0 | Risk weight on expected loss vs CVaR ($\gamma=1$ risk-neutral; $\gamma=0.5$ risk-averse base case). Shared by SP and ME. See §4.10. |
| `beta` | 0.95 | CVaR confidence level; **lower $\beta$ = more risk-averse** at fixed $\gamma<1$. Sensitivity sweep: $0.2,0.4,0.6,0.8$ at $\gamma=0.5$ (§4.10.4). Inactive when $\gamma=1$. |

### 9.2.1 SocialPlanner (`SocialPlanner` block)

Used **only** by `social_planner.jl`. ADMM subproblems continue to use Gurobi.

| Parameter | Default | Description |
|---|---|---|
| `solver` | `ipopt` | SP solver (`ipopt` recommended; `gurobi` optional) |
| `ipopt_tol` | `1.0e-6` | IPOPT KKT tolerance. Justification and magnitude table: **§7.2** |
| `ipopt_max_iter` | `5000` | IPOPT iteration cap |
| `ipopt_print_level` | `0` | `0` = silent; `3`–`5` = iteration log |
| `risk_warmstart` | `false` | Optional $\gamma=1$ auxiliary solve before risk-averse run |
| `risk_warmstart_beta` | `0.95` | $\beta$ for auxiliary warm-start only |
| `ipopt_retry_tol` | `1e-5` | Retry tolerance if first solve fails (code default) |
| `risk_warmstart_beta` | `0.95` | $\beta$ for auxiliary warm-start only |
| `ipopt_retry_tol` | `1e-5` | Retry tolerance if first solve fails (code default) |
| `ipopt_retry_max_iter` | `8000` | Retry iteration cap (code default) |

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
| `elec_market` | 90.0 €/MWh | 1.0 | Seed near NL 2021 wholesale with realistic VRES/gas costs |
| `elec_GC_market` | 3.0 €/MWh_GC | 0.3 | Seed near realistic GoO clearing (abundant VRES ⇒ low premium) |
| `H2_market` | 120.0 €/MWh_H2 | 0.5 | Seed ≈ 4 €/kg green H₂ at realistic power prices |
| `H2_GC_market` | 25.0 €/MWh_GC | 1.0 | Seed near green-H₂ certificate value under the 42% mandate |
| `EP_market` | 194.0 €/MWh_EP | 3.0 | NL-calibrated (~1000 €/t NH₃); `Total_Demand` ≈ 1970 → ~3 Mt/y via `LOAD_EP` |

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
| `ADMM.contract_cap` | `relaxation`, `expand_step`, `bind_tol` | Shared bilateral capacity $C$ bargaining (§2, `contract_capacity.jl`) |
| `PPAs` | `initial_price`, `rho_initial` | Seed for `λ_ppa` and ADMM energy penalty on PPA flow |
| `HPAs` | `initial_price`, `rho_initial` | Seed for `λ_hpa` and ADMM energy penalty on HPA flow |
| `HPAs` | `price_structure` | `fixed` or `cfd` (§2 *HPA price structure*) |
| `HPAs` | `price_benchmark` | Benchmark $B$ for strike at convergence and CfD reference leg |

Per-agent overrides under `PPAs.Gen_VRES_Solar`, `HPAs.Prod_H2_Green`, etc.

PPA and HPA use **one shared scalar capacity** $C$ (MW) per bilateral link — stored in `ADMM["SharedCap"]`, updated by `contract_capacity.jl`, **not** chosen independently by buyer and seller. There is no separate MW clearing price for $C$. Settlement strike $K$ is a scalar €/MWh (uniform over hours) snapshotted at convergence; during ADMM, provisional $K$ tracks the configured benchmark each iteration while $\lambda^{\mathrm{contract}}$ clears **energy** on the contract pool.

### 9.5 Agent Parameters

See `Data/data.yaml` for the full annotated configuration. Key parameters:

- **VRES**: `Capacity`, `Profile_Column`, `MarginalCost`
- **Conventional**: `Capacity`, `StageCapacityShares`, `StageBaseCosts`, `FinalMarginalCost` (`MarginalCost` optional if stage inputs are omitted)
- **Consumer**: `PeakLoad`, `Load_Column`, `A_E`, `B_E` (quadratic utility)
- **Electrolyzer**: `Capacity_Electrolyzer`, `Capacity_H2_Output`, `SpecificConsumption`, `OperationalCost`
- **Green offtaker**: `Capacity_H2_In`, `Capacity_EP_Out`, `Alpha`, `ProcessingCost`
- **Grey offtaker**: `Capacity`, `MarginalCost`, `gamma_NH3`
- **EP importer**: `Capacity`, `ImportCost`
- **GC demand**: `PeakLoad`, `Load_Column`, `A_GC`, `B_GC`

### 9.6 NL Calibration and Data Sources

This model is calibrated to the **Netherlands, base year 2021**. Every input that represents Dutch reality is tagged `[NL]` in `Data/data.yaml` with a short source key; the keys are resolved here and in §14. Endogenous variables (installed capacities after investment, all market prices) are **outputs**, not inputs — the values below are the *inputs/seeds* the optimiser starts from or is bounded by.

**Unit convention.** Ammonia (the end product, EP) is accounted on an energy basis using its lower heating value **LHV = 18.6 MJ/kg ⇒ 5.167 MWh_EP per tonne NH₃** `[LHV]`. All €/t ↔ €/MWh_EP and Mt ↔ TWh conversions use this factor (e.g. 1000 €/t ÷ 5.167 ≈ 194 €/MWh_EP; 3 Mt/yr × 5.167 ≈ 15.5 TWh/yr).

#### NL-calibrated inputs (sourced)

| Parameter (agent) | Value | Basis / derivation | Source |
|---|---|---|---|
| `base_year` | 2021 | Year with complete CBS capacity + generation data; ME horizon 2021–2030 | `[CBS-RE]`, `[CBS-EP]` |
| Solar `Capacity` | 14,823 MW | CBS installed solar PV at end-2021 (14,823 MWp) | `[CBS-RE]` |
| Wind `Capacity` | 7,700 MW | CBS installed wind (on+offshore) at end-2021 | `[CBS-RE]` |
| Solar `FixedCost_per_MW` | 90,000 €/MW-yr | ≈0.75 M€/MW CAPEX+FOM, 25 yr, 8% WACC ⇒ LCOE ≈43 €/MWh | `[IRENA-2022]` |
| Wind `FixedCost_per_MW` | 170,000 €/MW-yr | ≈2.0 M€/MW (offshore-leaning) CAPEX+FOM ⇒ LCOE ≈57 €/MWh | `[IRENA-2022]` |
| Conventional `Capacity` | 16,000 MW | NL dispatchable fossil fleet ~16 GW (mostly gas) | `[CBS-EP]`, `[TNO-2026]` |
| Conventional `StageCapacityShares` | [0.08, 0.12, 0.80] | NL fossil generation ≈80% gas, small coal/biomass residual | `[CBS-EP]` |
| Conventional `StageBaseCosts` / `FinalMarginalCost` | [70, 80, 95] / 160 €/MWh | 2021 SRMC: TTF gas ÷ η + EU-ETS CO₂ (coal heavier CO₂); peaking gas tail | `[TTF-2021]`, `[ETS-2021]` |
| Consumer `PeakLoad` | 20,000 MW | NL system peak ~20 GW; annual load ≈117 TWh (2021) | `[CBS-EP]` |
| Electrolyser `SpecificConsumption` | 1.5 MWh_e/MWh_H₂ | PEM efficiency ≈67% | `[IEA-H2]` |
| Electrolyser `FixedCost_per_MW_Electrolyzer` | 130,000 €/MW-yr | ≈1.1 M€/MW installed (2020–21, ~100 MW), 8% WACC / 20 yr ⇒ ~112 k€ annuity + ~18 k€ fixed O&M | `[DEA-2020]`, `[IEA-H2]` |
| EP demand `Total_Demand` | 1,970 | ≈3 Mt NH₃/yr (Yara Sluiskil ~1.8 Mt + OCI Geleen ~1.2 Mt) ⇒ ~15.5 TWh/yr | `[PBL-2019]`, `[LHV]` |
| EP `initial_price` | 194 €/MWh_EP | ~1000 €/t NH₃, realistic 2021 high-gas cost | `[H2EU-2023]`, `[LHV]` |
| Green offtaker `Capacity_H2_In` (seed) | 533 MW_H₂ | ≈20% of NL ammonia nameplate via green/electrolysis route | `[PBL-2019]` |
| Green offtaker `Alpha`, Grey `gamma_NH3` | 0.75 | H₂↔EP conversion ≈70–80% Haber–Bosch LHV efficiency | `[H2EU-2023]` |
| Grey offtaker `Capacity` | 1,570 MW_EP | ≈80% of NL ammonia nameplate (domestic conventional + ex-import share) | `[PBL-2019]` |
| Grey offtaker `MarginalCost` | 180 €/MWh_EP (≈930 €/t) | SMR gas (~33 GJ/t) + EU-ETS CO₂ (~1.8 t/t) at 2021 prices; grey LCOA range 534–891 €/t (gas 40–80 €/MWh) | `[H2EU-2023]`, `[TTF-2021]`, `[ETS-2021]` |
| GC mandate `gamma_GC` (in code) | 0.42 | EU RED III RFNBO-in-industry target (≥42% renewable H₂ by 2030) | `[RED-III]` |
| GC demand `PeakLoad` | 18,000 MW_GC | Most NL consumption seeks green certification (~system load) | `[CBS-EP]` |
| GC demand `A_GC` | 10 €/MWh_GC | Max WTP ≈ recent EU GoO price peak (clears lower) | `[GoO]` |

#### Engineering estimates and modeling choices (not direct NL measurements)

These are physically reasonable but not pulled from a single NL statistic; flagged in `data.yaml` so they are not mistaken for empirical data:

- **Green offtaker `ProcessingCost` = 12 €/MWh_EP (≈62 €/t)** and **`FixedCost_per_MW_EP_Out` = 120,000 €/MW-yr** (~1.2 M€/MW NH₃ synthesis plant) — engineering estimates for Haber–Bosch + air separation + compression.
- **Electrolyser `OperationalCost` = 3 €/MWh_H₂** — variable O&M estimate.
- **Consumer `A_E` = 500 €/MWh, `B_E` = 0.0025; GC demand `B_GC` = 0.0005** — quadratic-utility shape parameters; chosen to keep electricity demand near-inelastic at the ~20 GW scale and the GC market well-scaled (not NL price observations).
- **Importer `ImportCost` = 250 €/MWh_EP** — inactive (`Capacity = 0`); retained for re-enabling imports.
- **Risk parameters `gamma` = 0.5, `beta` = 0.95; all `rho_initial`, tolerances, `max_iter`** — algorithmic/risk-preference settings, not NL data.

### 9.7 Weather Scenarios, Representative Days, and Availability Factors

This section documents how the hourly input files are built, how they enter the model, and what each scenario year represents.

#### Scenario labels vs calendar dates

The model uses **10 scenario labels** `2021`–`2030` (`ADMM.nScenarioYears = 10`). These are **names for distinct weather scenarios**, not claims about calendar time:

- **`2021`** is the **baseline / reference** weather scenario. It matches `General.base_year`; installed capacities in `data.yaml` are NL 2021 values; the weather profile is calibrated to NL-realistic annual VRES capacity factors (see below).
- **`2022`–`2030`** are **nine additional weather scenarios** with different solar/wind hourly shapes and annual capacity factors. They enter the **same single operating year**: the optimiser chooses one investment level, then evaluates expected profit and CVaR over all scenarios with probability `P[jy] = 1/nScenarioYears`. They are **not** years in which the agent reinvests each period.
- A label **does not have to equal the calendar year of the underlying weather**. For example, scenario `2021` is built from ERA5 reanalysis at a central NL location for calendar **2015**, then rescaled to CBS-like annual CFs; scenario `2022` uses calendar **2010** weather, and so on. The mapping is fixed in `Input/generate_weather_scenarios.py` and summarised in `Input/weather_scenario_summary.json`.

| Scenario label | Source weather year (ERA5) | Role |
|---|---|---|
| 2021 | 2015 | Baseline NL reference; solar CF → **18%**, wind CF → **28%** (CBS 2021 fleet averages) |
| 2022 | 2010 | Low-wind / dunkelflaute-prone (~14% wind CF) |
| 2023 | 2012 | Average mixed VRES |
| 2024 | 2013 | High-solar summer emphasis |
| 2025 | 2014 | High-wind year |
| 2026 | 2016 | Windy winter |
| 2027 | 2017 | Calm summer / lower wind |
| 2028 | 2018 | Strong solar summer |
| 2029 | 2019 | Cold winter (higher load, moderate VRES) |
| 2030 | 2011 | Alternative tail scenario |

Annual capacity factors for all scenarios are listed in `Input/weather_scenario_summary.json`.

#### Raw weather data and capacity-factor conversion

Hourly weather is fetched from the **[Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)** (ERA5 reanalysis) at **52.09°N, 5.12°E** (central Netherlands). Three variables are used:

| Variable | Conversion to model column | Notes |
|---|---|---|
| `shortwave_radiation` (W/m²) | **SOLAR** | Hourly PV capacity factor = min(1, GHI / 1000 W/m²) |
| `wind_speed_100m` (km/h) | **WIND** | Standard turbine power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s), cubic between cut-in and rated |
| `temperature_2m` (°C) | **LOAD_E** (indirect) | Drives mild heating sensitivity on a fixed NL diurnal/seasonal load shape |

For the **baseline scenario only** (`2021`), hourly SOLAR and WIND profiles are **rescaled** (preserving hourly shape) so that the annual mean matches NL CBS 2021 fleet averages: **~18% solar CF, ~28% wind CF** `[CBS-RE]`. Other scenarios use unscaled CFs from their source weather year, giving a realistic spread (~12–19% solar, ~14–19% wind in the current mapping).

**LOAD_H** and **LOAD_EP** are fixed normalised shapes (0.8 and 0.9) in all scenarios; absolute H₂ and EP demand is set in `data.yaml` via agent capacities and `Total_Demand`.

#### Representative-day selection (8760 h → 8 days)

Full-year hourly profiles (365 × 24 h) are reduced to **`nReprDays = 8`** representative days for tractability:

1. **Daily feature vector** (72 dimensions): concatenate the 24 hourly SOLAR, 24 hourly WIND, and 24 hourly LOAD_E values for each calendar day. SOLAR and WIND entries are **double-weighted** relative to load so clustering prioritises VRES diversity (including low-renewable days).
2. **Standardisation**: each of the 72 features is z-scored across the 365 days.
3. **k-medoids clustering** (`k = 8`): partition the 365 days; the **medoid** of each cluster is the actual calendar day whose profile best represents that cluster (PAM-style, implemented in `Input/generate_weather_scenarios.py`).
4. **Weights**: for cluster `c`, `W[c] =` number of calendar days assigned to that cluster. Weights sum to **365** and are stored in `decision_variables_short.csv`.
5. **Medoid day index** (`periods`): day-of-year (1–365) of the medoid, stored alongside its weight in `decision_variables_short.csv`. The full `decision_variables.csv` lists all 365 days with weight zero except the eight medoids.
6. **`ordering_variable.csv`**: 365 × 8 matrix of row-normalised inverse Euclidean distances from each calendar day to each medoid profile (output of the representative-day selection script; loaded but not used in optimisation).

Representative days are sorted by medoid calendar day before writing `timeseries_<label>.csv`, so row blocks 1–24, 25–48, … correspond to `jd = 1, 2, …, 8`.

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

## 10. Project Structure

```
Now/
├── market_exposure.jl          # Entry point: distributed ADMM simulation (5 markets)
├── me_pap.jl                    # Entry point: ADMM + PPA/HPA (pay-as-produced HPA)
├── me_top.jl                    # Entry point: ADMM + PPA/HPA (take-or-pay HPA)
├── me_sop.jl                    # Entry point: ADMM + PPA/HPA (send-or-pay HPA)
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
│   ├── Agent_Summary.csv             # Agent group membership and ADMM objective value
│   ├── Agent_Quantities_Final.csv    # Final-iteration net quantities per agent
│   ├── Offtaker_GC_Diagnostics.csv   # GC compliance per offtaker
│   ├── H2_Producer_Diagnostics.csv   # H₂ GC-to-production ratio
│   ├── Capacity_Investments.csv      # VRES/electrolyzer/green offtaker capacity & investment (one row per agent; ADMM)
│   └── TimerOutput.yaml              # Profiling data
│
├── me_pap_results/              # Output from me_pap.jl
├── me_top_results/              # Output from me_top.jl
├── me_sop_results/              # Output from me_sop.jl
│   ├── ADMM_Convergence.csv          # Same as market_exposure + PPA/HPA energy + shared-cap columns
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
    ├── Market_Prices.csv             # Equilibrium prices (balance duals / (W × μ); §7.2)
    ├── Agent_Summary.csv             # Per-agent quantity & ADMM-style objective value
    └── Capacity_Investments_Planner.csv  # VRES/electrolyzer/green offtaker capacity & investment (planner)
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
| `define_common_parameters.jl` | Creates `mod.ext` dictionaries (sets, parameters, timeseries, variables, constraints, expressions). Fills JH/JD/JY, W, P, γ, β. Determines market participation from agent type. Pre-allocates ADMM placeholder arrays. |
| `define_power_parameters.jl` | VRES: capacity, AF profile. Conventional: capacity, AF=1, 3-stage cost curve (`ConvStageCap`, `ConvStageBaseCost`, `ConvStageSlope`) built from share + absolute stage-cost inputs (coal/biomass/NG style), with no global average-cost rescaling. Consumer: PeakLoad, LOAD_E profile, A_E, B_E. |
| `define_H2_parameters.jl` | Electrolyzer: Capacity_Electrolyzer, Capacity_H2_Output, SpecificConsumption, OperationalCost, η_elec_H2. |
| `define_offtaker_parameters.jl` | Copies all keys from agent block; sets gamma_GC = 0.42 (regulatory mandate). |
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
| `solve_power_agent.jl` | Rebuilds objective with iteration-specific λ, ḡ, ρ. For VRES: recomputes loss expressions with iteration-specific λ, deletes and re-adds CVaR shortfall/linking constraints. For conventional: applies the 3-stage convex variable cost (single linear `MarginalCost` only if stage inputs are absent). Calls `optimize!`. |
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
| `save_results.jl` | Writes: ADMM_Convergence.csv, ADMM_Diagnostics.csv, per-market history CSVs, Agent_Summary.csv, Agent_Quantities_Final.csv, Offtaker_GC_Diagnostics.csv, H2_Producer_Diagnostics.csv. |
| `save_results_contracts.jl` | Writes the same major ADMM outputs as save_results (with PPA/HPA columns) plus: PPAs.csv, HPAs.csv, Green_Agents_Detail.csv. Agent_Summary matches market_exposure structure (no explicit contract columns). |
| `compute_social_risk_metrics.jl` | Post-processing: social CVaR, $E[SW]$, private CVaR sum, SP comparison; writes Risk_Metrics.csv. |
| `save_social_planner_results.jl` | Called after direct QCP solve with duals available. Writes: Market_Prices.csv, Agent_Summary.csv, Capacity_Investments_Planner.csv, Risk_Metrics.csv, Social_Welfare_Per_Year.csv; prints risk summary. |

---

## 12. Output Files

### 12.1 Market Exposure Results

| File | Contents |
|---|---|
| `ADMM_Convergence.csv` | Columns: `iter`, `{market}_primal`, `{market}_dual` for each of the 5 markets, plus `cap_primal` / `cap_dual` (aggregate L2 over agents) and **per-agent** `cap_primal_<m>` / `cap_dual_<m>` columns from the equality-split capacity ADMM (§6.4). One row per ADMM iteration. Used for convergence plots. |
| `ADMM_Diagnostics.csv` | Columns: `iter`, `{market}_rho`, `{market}_price_mean`, `{market}_imb_mean` for each flow market, plus per-agent `cap_rho_<m>` columns (one per cap-owning agent). |
| `Capacity_Consensus.csv` | Per-iteration, per-agent, per-year snapshot of the capacity equality split. Columns: `iter`, `AgentID`, `jy`, `x_cap`, `z_cap`, `lambda_cap`, `rho_cap`, `primal_local`, `dual_local`. Use this to identify the agent / year that gates capacity convergence; analogous to `{Market}_Market_History.csv` but at the (iter, agent, year) granularity that the per-agent split naturally produces. See §6.4 for the formal model. |
| `{Market}_Market_History.csv` | Per-market CSV with: `iter`, `rho`, `price_mean`, `imb_mean`, `primal_res`, `dual_res`. |
| `Agent_Summary.csv` | Columns: `AgentID`, `Group`. Group membership table. |
| `Agent_Quantities_Final.csv` | Columns: `AgentID`, `Group`, `elec_net_sum`, `H2_net_sum`, `elec_GC_net_sum`, `H2_GC_net_sum`, `EP_net_sum`. Sum of final-iteration 3D quantities. |
| `Offtaker_GC_Diagnostics.csv` | Columns: `AgentID`, `Type`, `EP_total`, `H2_in_total`, `H2_GC_total`, `GC_share`, `GC_mandate`, `GC_slack`. |
| `H2_Producer_Diagnostics.csv` | Columns: `AgentID`, `H2_total`, `H2_GC_total`, `GC_per_H2`. |
| `Risk_Metrics.csv` | `expected_social_welfare`, `social_CVaR`, `sum_private_CVaR`, gap vs SP — §7.6. |
| `Social_Welfare_Per_Year.csv` | Per-year aggregate welfare and loss at the ADMM allocation. |
| `Private_CVaR_By_Agent.csv` | Per-agent private CVaR (VRES, electrolyzer, green offtaker) when $\gamma<1$. |
| `TimerOutput.yaml` | Profiling: time spent in imbalances, residuals, capacity dual update, price updates, solve, etc. |

### 12.2 ME contract results (`me_pap_results/`, `me_top_results/`, `me_sop_results/`)

The ME contract entry points produce the same major ADMM outputs as market_exposure (ADMM_Convergence, ADMM_Diagnostics, `Capacity_Consensus.csv`, 5× Market_History, Agent_Summary, Market_Prices), with additional PPA/HPA energy columns and **shared-cap alignment** columns (`ppa_cap_*`, `hpa_cap_*`: $|C - q^{\mathrm{peak}}|$ and $|C^k - C^{k-1}|$) in convergence and diagnostics. Per-agent **investment** capacity columns and `Capacity_Consensus.csv` follow the equality-split structure in §6.4. Additional contract outputs:

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
| `Risk_Metrics.csv` | Social CVaR, expected social welfare, $\alpha$, and (for ADMM) gap vs SP — see §7.6. |
| `Social_Welfare_Per_Year.csv` | Per scenario year: `social_welfare`, `social_loss`, `probability`. |
| `Agent_Summary.csv` | Columns: `Agent`, `Type`, `Total_Quantity`, `Welfare_Contribution`. |
| `Capacity_Investments_Planner.csv` | Per-agent scalar capacity and investment for VRES, electrolyzer, and green offtaker. |

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
   Source for the **$\gamma$–CVaR objective** and **$\beta$ sensitivity** (decreasing $\beta$ = increasing risk aversion at fixed $\gamma$); ADMM for risk-averse equilibrium — see §4.10.4 and §6.

### Data sources for the NL calibration (§9.6)

These source keys are referenced inline in `Data/data.yaml` (tag `[NL]`) and in the §9.6 calibration tables.

- **`[CBS-RE]`** — Statistics Netherlands (CBS), *Renewable electricity; production and capacity* (table 82610ENG). Installed capacity end-2021: **solar PV 14,823 MWp, wind 7,700 MW**. [cbs.nl/en-gb/figures/detail/82610ENG](https://www.cbs.nl/en-gb/figures/detail/82610ENG)
- **`[CBS-EP]`** — CBS, *Electricity; production and means of production* + NL electricity-sector statistics. NL **2021 consumption ≈117 TWh, peak demand ~20 GW**, fossil generation ≈80% gas, dispatchable fossil fleet on the order of ~16–24 GW. [cbs.nl/en-gb/figures/detail/37823eng](https://www.cbs.nl/en-gb/figures/detail/37823eng); overview: [Electricity sector in the Netherlands (Wikipedia)](https://en.wikipedia.org/wiki/Electricity_sector_in_the_Netherlands)
- **`[TNO-2026]`** — TNO scenario study (COMPETES-TNO), Dutch dispatchable (gas-fired) capacity reference figures (~14.7 GW gas in the 2030 reference, used to corroborate the ~16 GW dispatchable proxy). [publications.tno.nl/publication/34645515](https://publications.tno.nl/publication/34645515/fLCICwBT/TNO-2026-R10080.pdf)
- **`[PBL-2019]`** — PBL/ECN, *Decarbonisation options for the Dutch fertiliser industry*, 2019. NL ammonia capacity: **Yara Sluiskil ≈1.8 Mt/yr + OCI Nitrogen Geleen ≈1.2 Mt/yr ≈ 3 Mt NH₃/yr**. [pbl.nl PDF](https://www.pbl.nl/uploads/default/downloads/pbl-2019-decarbonisation-options-for-the-dutch-fertiliser-industry_3657.pdf)
- **`[H2EU-2023]`** — Hydrogen Europe, *Clean Ammonia Report*, 2023. **Grey ammonia LCOA ≈534–891 €/t** (gas 40–80 €/MWh, CO₂ 75 €/t; range up to ~1,069 €/t at gas 110 €/MWh); green ammonia ≈2–6× grey; Haber–Bosch H₂↔NH₃ efficiency. [hydrogeneurope.eu PDF](https://hydrogeneurope.eu/wp-content/uploads/2023/03/2023.03_H2Europe_Clean_Ammonia_Report_DIGITAL_FINAL.pdf)
- **`[IEA-H2]`** — IEA, *Global Hydrogen Review* / electrolyser technology briefs. PEM electrolyser efficiency ≈67% (≈1.5 MWh_e/MWh_H₂); installed CAPEX in the ~1.0–1.5 M€/MW range in the early 2020s (noting CAPEX rose ~15% in 2023 vs 2021). [iea.org/reports/global-hydrogen-review-2023](https://www.iea.org/reports/global-hydrogen-review-2023)
- **`[DEA-2020]`** — Danish Energy Agency, *Technology Data for Renewable Fuels* (electrolysers). 100 MW alkaline installed CAPEX ≈1,200 €/kW (2020); corroborated by Fraunhofer ISE (2021): PEM ≈718 €/kW at 100 MW to ≈978 €/kW at 5 MW, fixed O&M ≈15–20 €/kW·yr. Basis for the ~130 k€/MW-yr annualised electrolyser fixed cost. [ens.dk technology data](https://ens.dk/en/our-services/technology-catalogues/technology-data-renewable-fuels)
- **`[IRENA-2022]`** — IRENA, *Renewable Power Generation Costs*, 2022. Utility solar and on/offshore wind CAPEX and LCOE ranges used for the VRES fixed costs. [irena.org/publications](https://www.irena.org/publications/2023/Aug/Renewable-Power-Generation-Costs-in-2022)
- **`[TTF-2021]`** — Dutch TTF natural-gas day-ahead: 2021 annual average ≈46 €/MWh with strong Q4 escalation; used for conventional SRMC and grey-ammonia gas cost.
- **`[ETS-2021]`** — EU ETS CO₂ allowance price, 2021 average ≈53–55 €/tCO₂; used for conventional SRMC and grey-ammonia CO₂ cost.
- **`[RED-III]`** — Directive (EU) 2023/2413 (RED III): ≥42% of hydrogen used in industry must be renewable/RFNBO by 2030 → `gamma_GC = 0.42` mandate. [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/dir/2023/2413/oj)
- **`[GoO]`** — European Guarantees-of-Origin (GoO/CertiQ) market prices: historically <2 €/MWh, rising to ~5–9 €/MWh in 2022–2023; used for the GC demand WTP intercept.
- **`[LHV]`** — Ammonia lower heating value 18.6 MJ/kg ⇒ **5.167 MWh per tonne NH₃**, the conversion constant for all €/t ↔ €/MWh_EP and Mt ↔ TWh calculations.

### Weather and representative-day inputs (§9.7)

- **`[ERA5-OM]`** — Open-Meteo Historical Weather API (ERA5 reanalysis). Hourly GHI, 100 m wind speed, and 2 m temperature for central NL (52.09°N, 5.12°E). [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)
