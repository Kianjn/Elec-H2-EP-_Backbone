# ==============================================================================
# build_power_agent.jl — JuMP model for power-sector agents
# ==============================================================================
#
# PURPOSE:
#   Constructs variables, constraints, and objective for one power-sector agent
#   (VRES, Conventional, or Consumer) in the ADMM formulation, and adds the
#   corresponding block to the social planner model.
#
#   - VRES: variable renewable generator with endogenous yearly capacity
#     cap_VRES[jy] and investment inv_VRES[jy]. Supplies electricity and
#     electricity GCs 1:1. Net positions: g_net_elec = +g, g_net_elec_GC = +g.
#     ADMM objective = generation cost − elec revenue − GC revenue
#     + ADMM penalties + fixed annualised CAPEX on cap_VRES + optional CVaR
#     term (γ·β_VRES). Capacity constraint: g ≤ AF × cap_VRES[jy].
#
#   - Conventional: dispatchable generator with fixed Capacity and marginal
#     cost. Net position: g_net_elec = +g. ADMM objective = generation cost −
#     elec revenue + ADMM penalties.
#
#   - Consumer: elastic electricity demand with quadratic utility. Net position:
#     g_net_elec = −d. ADMM objective = expenditure − utility + ADMM penalty.
#
#   No solve is performed here; solve_power_agent! re-sets the objective with
#   current λ/ρ/g_bar (and risk variables) and calls optimize!.
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model (parameters and sets already set by define_*_parameters!).
#   elec_market, elec_GC_market — Not read here; nAgents etc. used in ADMM.
#
# ==============================================================================

function build_power_agent!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict)
    # ── Index sets ────────────────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W   = mod.ext[:parameters][:W]    # W[jd,jy] = weight (number of real days
                                      #   represented by this representative day)

    # ── ADMM parameters for the ELECTRICITY market ────────────────────────
    λ_elec     = mod.ext[:parameters][:λ_elec]       # λ = ADMM price (Lagrange
                                      # multiplier for the electricity market-
                                      # clearing constraint, updated each ADMM
                                      # iteration)
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]   # ḡ = consensus target
                                      # (average of all agents' net positions
                                      # in the electricity market; agents are
                                      # driven toward this shared schedule)
    ρ_elec     = mod.ext[:parameters][:ρ_elec]       # ρ = ADMM penalty weight
                                      # (quadratic coefficient that drives each
                                      # agent's position toward the consensus ḡ;
                                      # larger ρ → faster convergence but harder
                                      # sub-problems)

    # ── ADMM parameters for the ELECTRICITY-GC market ─────────────────────
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]       # λ_GC = Lagrange
                                      # multiplier for the elec-GC market
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]   # ḡ_GC = consensus
                                      # target in the elec-GC market
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]          # ρ_GC = penalty
                                      # weight for elec-GC market
    agent_type = mod.ext[:parameters][:Type]

    if agent_type == "VRES"
        cap_initial = mod.ext[:parameters][:Capacity]   # initial installed capacity (MW in first model year)
        AF  = mod.ext[:timeseries][:AF]          # AF[jh,jd,jy] = hour-specific
                                                 #   availability factor (0–1);
                                                 #   reflects wind/solar resource
        MC  = mod.ext[:parameters][:MarginalCost]  # marginal cost (€/MWh)
        # Annualised fixed investment cost per MW of installed capacity (€/MW-year).
        # Default 0.0 preserves original behaviour if not provided.
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        # Risk parameters (CVaR skeleton; γ = 0 ⇒ risk-neutral by default).
        gamma = get(mod.ext[:parameters], :γ, 1.0)
        beta_cvar = get(mod.ext[:parameters], :β, 0.95)   # confidence level τ
        P = mod.ext[:parameters][:P]
        n_years = length(JY)   # apply the annualised charge once per model year

        # ── Capacity and investment decision variables (per year) ──────────────
        # cap_VRES[jy] = available VRES capacity (MW) in year jy.
        # inv_VRES[jy] = new VRES capacity investment (MW) in year jy.
        cap_VRES = mod.ext[:variables][:cap_VRES] = @variable(mod, [jy in JY], lower_bound = 0, base_name = "cap_VRES")
        inv_VRES = mod.ext[:variables][:inv_VRES] = @variable(mod, [jy in JY], lower_bound = 0, base_name = "inv_VRES")

        # Capacity evolution over years: cumulative investment on top of initial capacity.
        JY_vec = collect(JY)
        first_jy = JY_vec[1]
        # First year: initial capacity plus first-year investment.
        mod.ext[:constraints][:cap_VRES_init] = @constraint(mod, cap_VRES[first_jy] == cap_initial + inv_VRES[first_jy])
        # Subsequent years: add new investment on top of previous year's capacity.
        for (k, jy) in enumerate(JY_vec)
            k == 1 && continue
            prev_jy = JY_vec[k - 1]
            cname = Symbol("cap_VRES_dyn_", jy)
            mod.ext[:constraints][cname] = @constraint(mod, cap_VRES[jy] == cap_VRES[prev_jy] + inv_VRES[jy])
        end

        # Generation variable g ≥ 0 (MWh produced in each hour/day/year).
        g = mod.ext[:variables][:g] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gen")

        # Net positions — VRES output counts as positive supply in BOTH the
        # electricity market AND the electricity-GC market, because all VRES
        # output is inherently renewable-certified (each MWh of wind/solar
        # automatically generates one green certificate).
        mod.ext[:expressions][:g_net_elec]    = @expression(mod, g)   # g_net_elec    = +g
        mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g)   # g_net_elec_GC = +g

        # ── Risk variables (agent-level CVaR) ───────────────────────────────
        # α_VRES: VaR proxy; β_VRES: CVaR; u_VRES[jy]: shortfall per scenario year.
        alpha_VRES = mod.ext[:variables][:alpha_VRES] = @variable(mod, lower_bound = 0, base_name = "alpha_VRES_$(m)")
        beta_VRES  = mod.ext[:variables][:beta_VRES]  = @variable(mod, lower_bound = 0, base_name = "beta_VRES_$(m)")
        u_VRES     = mod.ext[:variables][:u_VRES]     = @variable(mod, [jy in JY], lower_bound = 0, base_name = "u_VRES_$(m)")

        # Per-year economic loss (cost − revenue) excluding ADMM penalties; prices
        # λ are updated each ADMM iteration via mod.ext[:parameters][:λ_*].
        loss_VRES = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_VRES[jy] = @expression(mod,
                sum(W[jd, jy] * (MC * g[jh, jd, jy]
                    - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                    - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        mod.ext[:expressions][:loss_VRES] = loss_VRES

        # CVaR shortfall constraints: u_VRES[jy] ≥ loss_VRES[jy] − α.
        mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
            u_VRES[jy] >= loss_VRES[jy] - alpha_VRES
        )

        # CVaR definition: β_VRES ≥ α_VRES + (1/(1−τ)) * Σ P[jy]*u_VRES[jy].
        one_minus_tau = max(1e-6, 1.0 - beta_cvar)
        mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
            beta_VRES >= alpha_VRES + (1 / one_minus_tau) * sum(P[jy] * u_VRES[jy] for jy in JY)
        )

        # Objective:
        #   min  Σ_{h,d,y} W[d,y]·( MC·g − λ_elec·g − λ_GC·g )       ← (1)
        #      + Σ_{h,d,y} (ρ_elec/2)·W[d,y]·(g − ḡ_elec)²           ← (2)
        #      + Σ_{h,d,y} (ρ_GC /2)·W[d,y]·(g − ḡ_GC)²             ← (3)
        #
        # (1) Production cost minus revenue from the electricity market
        #     minus revenue from the elec-GC market.  The agent earns λ_elec
        #     and λ_GC per MWh of generation g.
        # (2) ADMM augmented-Lagrangian penalty pushing g toward the
        #     electricity-market consensus ḡ_elec.
        # (3) ADMM augmented-Lagrangian penalty pushing g toward the
        #     elec-GC-market consensus ḡ_GC.
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * g[jh, jd, jy]
                - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec_GC/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            # Fixed annualised investment cost: F_cap [€/MW-year] × capacity in each model year.
            + F_cap * sum(cap_VRES[jy] for jy in JY)
        )

        # Capacity constraint: generation limited by the hour-specific
        # availability factor (AF) times installed capacity.  For VRES this
        # captures the physical resource limit (e.g. wind speed, irradiance).
        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], g[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES[jy])

    elseif agent_type == "Conventional"
        cap = mod.ext[:parameters][:Capacity]        # installed capacity (MW)
        MC  = mod.ext[:parameters][:MarginalCost]    # marginal cost (€/MWh)

        # Generation variable g ≥ 0 for conventional (thermal) plant.
        g = mod.ext[:variables][:g] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "gen")

        # Net position in the electricity market only — conventional generation
        # does NOT participate in the elec-GC market (output is not renewable-
        # certified), so there is no g_net_elec_GC expression.
        mod.ext[:expressions][:g_net_elec] = @expression(mod, g)   # g_net_elec = +g

        # Objective — same structure as VRES but without GC market terms:
        #   min  Σ W·(MC·g − λ_elec·g)     ← production cost minus elec revenue
        #      + Σ (ρ_elec/2)·W·(g − ḡ)²   ← ADMM penalty toward consensus
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

        # Capacity constraint: generation ≤ installed capacity (no AF for
        # dispatchable plants — they can produce up to full capacity at will).
        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], g[jh, jd, jy] <= cap)

    elseif agent_type == "Consumer"
        peak = mod.ext[:parameters][:PeakLoad]   # peak demand (MW)
        A_E  = mod.ext[:parameters][:A_E]         # intercept of inverse demand
        B_E  = mod.ext[:parameters][:B_E]         # slope of inverse demand
        load = mod.ext[:timeseries][:LOAD_E]      # LOAD_E[jh,jd,jy] = normalized
                                                  #   hourly load profile (0–1)

        # Demand variable d ≥ 0 (MWh consumed).
        d = mod.ext[:variables][:d] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "demand")

        # Net position: demand is NEGATIVE supply in the electricity market.
        mod.ext[:expressions][:g_net_elec] = @expression(mod, -d)   # g_net_elec = −d

        # Objective:
        #   min  Σ W·( λ_elec·d  −  U(d) )           ← (1)
        #      + Σ (ρ_elec/2)·W·(−d − ḡ_elec)²       ← (2)
        #
        # U(d) = A_E·d − (B_E/2)·d²  is the quadratic consumer utility
        #   (area under the inverse demand curve).
        # (1) The agent minimizes expenditure (λ·d) minus utility U(d),
        #     which is equivalent to maximizing consumer surplus U(d) − λ·d.
        # (2) ADMM penalty on the net position (−d) toward consensus ḡ_elec.
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

        # Load constraint: demand bounded by peak load × hourly load profile,
        # representing the physical maximum consumption in each hour.
        mod.ext[:constraints][:load] = @constraint(mod, [jh in JH, jd in JD, jy in JY], d[jh, jd, jy] <= peak * load[jh, jd, jy])
    end

    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add power agent block to shared planner model (no ADMM terms).
#
# Unlike the ADMM build above, the planner optimizes ALL agents jointly in a
# single model.  Therefore there are NO ADMM penalty terms: no λ (prices emerge
# as dual variables of market-clearing constraints added elsewhere), no ρ
# (penalty weight), and no ḡ (consensus target).  Each agent contributes only
# its physical constraints and its welfare expression (utility for consumers,
# negative cost for generators) to the planner's objective.
#
# Returns: the agent's contribution to total social welfare as a JuMP expression.
# Side-effect: stores the agent's decision variables in var_dict so that the
# caller can build market-clearing constraints across all agents.
# ------------------------------------------------------------------------------

function add_power_agent_to_planner!(planner::Model, id::String, mod::Model,
                                     var_dict::Dict, W::AbstractArray)::JuMP.AbstractJuMPScalar
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    _p(m, k) = get(m.ext[:parameters], k, nothing)
    _ts(m, k) = m.ext[:timeseries][k]
    agent_type = String(_p(mod, :Type))

    # W_dict is a nested Dict{year => Dict{rep_day => weight}} that enables
    # convenient indexed access inside JuMP @expression macros (which require
    # scalar look-ups rather than array slicing).
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    if agent_type == "Consumer"
        # Quadratic utility parameters (defaults guard against missing data).
        A_E = _p(mod, :A_E) !== nothing ? _p(mod, :A_E) : 500.0
        B_E = _p(mod, :B_E) !== nothing ? _p(mod, :B_E) : 0.5
        # D_bar = peak × load_profile: physical upper bound on consumption.
        D_bar = _ts(mod, :LOAD_E) .* _p(mod, :PeakLoad)

        d_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_E_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], d_E[jh, jd, jy] <= D_bar[jh, jd, jy])

        # Welfare contribution = consumer utility U(d) = A_E·d − (B_E/2)·d².
        # No expenditure term: the planner internalizes prices via market-
        # clearing duals, so only the utility (= willingness to pay) matters.
        obj = @expression(planner,
            sum(W_dict[jy][jd] * ((A_E * d_E[jh, jd, jy]) - 0.5 * B_E * d_E[jh, jd, jy]^2)
                for jh in JH, jd in JD, jy in JY)
        )
        var_dict[:power_d_E][id] = d_E
        return obj

    elseif agent_type == "VRES"
        # Availability factors and costs.
        AF = _ts(mod, :AF)
        C = _p(mod, :MarginalCost)
        gamma = get(mod.ext[:parameters], :γ, 1.0)
        beta_cvar = get(mod.ext[:parameters], :β, 0.95)
        P = mod.ext[:parameters][:P]
        # Annualised fixed investment cost per MW of installed VRES capacity (€/MW-year).
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        cap_initial = _p(mod, :Capacity)

        # Yearly capacity and investment variables.
        cap_VRES = @variable(planner, [jy in JY], lower_bound=0, base_name="cap_VRES_$(id)")
        inv_VRES = @variable(planner, [jy in JY], lower_bound=0, base_name="inv_VRES_$(id)")

        # Capacity evolution over years.
        JY_vec = collect(JY)
        first_jy = JY_vec[1]
        @constraint(planner, cap_VRES[first_jy] == cap_initial + inv_VRES[first_jy])
        for (k, jy) in enumerate(JY_vec)
            k == 1 && continue
            prev_jy = JY_vec[k - 1]
            @constraint(planner, cap_VRES[jy] == cap_VRES[prev_jy] + inv_VRES[jy])
        end

        q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], q_E[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES[jy])

        # Risk variables for planner CVaR (cost-based, no prices in planner objective).
        alpha_VRES = @variable(planner, lower_bound=0, base_name="alpha_VRES_$(id)")
        beta_VRES  = @variable(planner, lower_bound=0, base_name="beta_VRES_$(id)")
        u_VRES     = @variable(planner, [jy in JY], lower_bound=0, base_name="u_VRES_$(id)")

        # Per-year cost (positive) = production cost + fixed capacity cost in that year.
        loss_VRES = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_VRES[jy] = @expression(planner,
                sum(W_dict[jy][jd] * (C * q_E[jh, jd, jy]) for jh in JH, jd in JD)
                + F_cap * cap_VRES[jy]
            )
        end

        # Shortfall constraints and CVaR link.
        @constraint(planner, [jy in JY], u_VRES[jy] >= loss_VRES[jy] - alpha_VRES)
        one_minus_tau = max(1e-6, 1.0 - beta_cvar)
        @constraint(planner,
            beta_VRES >= alpha_VRES + (1 / one_minus_tau) * sum(P[iy] * u_VRES[jy] for (iy, jy) in enumerate(JY))
        )

        # Welfare contribution = −(production cost).  Negative because cost
        # reduces total welfare; revenue cancels out in the planner's
        # aggregate (it is a transfer between agents).
        obj = @expression(planner,
            -sum(W_dict[jy][jd] * (C * q_E[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            - F_cap * sum(cap_VRES[jy] for jy in JY)   # fixed annualised investment cost (capacity-only term)
            # CVaR risk penalty; γ = 0 ⇒ risk-neutral.
            - gamma * beta_VRES
        )
        var_dict[:power_q_E][id] = q_E
        var_dict[:power_cap_VRES][id] = cap_VRES
        var_dict[:power_inv_VRES][id] = inv_VRES
        return obj

    else  # Conventional
        cap = _p(mod, :Capacity)
        C = _p(mod, :MarginalCost)

        q_E = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="q_E_$(id)")
        # Dispatchable: capacity limit is constant (no availability factor).
        @constraint(planner, [jh in JH, jd in JD, jy in JY], q_E[jh, jd, jy] <= cap)

        # Welfare contribution = −(production cost), same logic as VRES.
        obj = @expression(planner,
            -sum(W_dict[jy][jd] * (C * q_E[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
        )
        var_dict[:power_q_E][id] = q_E
        return obj
    end
end
