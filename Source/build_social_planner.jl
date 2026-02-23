# ==============================================================================
# build_social_planner.jl — Build centralized social planner from Source blocks
# ==============================================================================
#
# PURPOSE:
#   Constructs the social planner optimization problem by calling add_*_to_planner!
#   from each build_* module. Uses the same parameter definitions (define_*_parameters!)
#   and agent types as the market-exposure ADMM case. No duplication of objectives
#   or constraints: they all live in the build_* files.
#
#   Changes to objectives or constraints in build_* automatically propagate to
#   both market_exposure and social_planner.
#
# ARGUMENTS:
#   mdict — Dict of agent models with ext[:parameters], ext[:sets], ext[:timeseries].
#   agents — Dict of agent lists (power, H2, offtaker, elec_GC_demand, EP_demand).
#   elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market — Market dicts.
#
# RETURNS:
#   planner — JuMP model (convex QCP with epigraph formulation; ready to optimize).
#   planner_state — Dict with:
#     :var_dict, :agent_welfare (total per agent), :agent_welfare_per_year,
#     :social_welfare (aggregate per year), :sw_aux (epigraph proxy variables),
#     :gamma, :beta, :demand_var_keys (var_dict keys for quadratic-utility
#       demand variables — needed for QCP→LP dual recovery in social_planner.jl),
#     :elec_balance, :elec_GC_balance, :H2_balance, :H2_GC_balance, :EP_balance,
#     :power_consumers, :power_vres, :power_conv, :H2_producers, :H2_consumers,
#     :offtaker_green, :offtaker_grey, :offtaker_import, :JH, :JD, :JY, :W.
#
# ==============================================================================

# env: optional Gurobi environment for license reuse. When provided, the
# planner model shares the same Gurobi license token as the caller, avoiding
# the overhead of acquiring a new license. If nothing, a fresh Env is created.
function build_social_planner!(mdict::Dict, agents::Dict, elec_market::Dict, H2_market::Dict,
                               elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
                               env::Union{Gurobi.Env, Nothing} = nothing)
    if isempty(agents[:all])
        error("No agents defined in data.yaml; cannot build social planner problem.")
    end

    planner = env !== nothing ? Model(() -> Gurobi.Optimizer(env)) : Model(Gurobi.Optimizer)
    set_silent(planner)

    m0 = mdict[agents[:all][1]]
    JH = collect(m0.ext[:sets][:JH])
    JD = collect(m0.ext[:sets][:JD])
    JY = collect(m0.ext[:sets][:JY])
    W = m0.ext[:parameters][:W]
    # W_dict: nested Dict {year => {day => weight}} that mirrors the W matrix.
    # WHY: JuMP @expression macros are more readable with W_dict[jy][jd] than
    # W[jd, jy], and the nested structure naturally matches the (jy, jd) loop
    # nesting used in the objective and constraint definitions below.
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    # var_dict: stores all agent decision variables indexed by agent ID and
    # organized by market role (e.g. :power_q_E for electricity supply).
    # WHY: market-clearing constraints need to sum over all participants in a
    # given market. Collecting variables here lets us iterate over the relevant
    # subset without revisiting each agent's internal model.
    var_dict = Dict{Symbol, Dict{String, Any}}(
        :power_d_E => Dict{String, Any}(),
        :power_q_E => Dict{String, Any}(),
        :power_cap_VRES => Dict{String, Any}(),
        :power_inv_VRES => Dict{String, Any}(),
        :H2_e_buy => Dict{String, Any}(),
        :H2_gc_e_buy => Dict{String, Any}(),
        :H2_h_sell => Dict{String, Any}(),
        :H2_gc_h_sell => Dict{String, Any}(),
        :H2_cap_elec => Dict{String, Any}(),
        :H2_inv_elec => Dict{String, Any}(),
        :H2_d_H => Dict{String, Any}(),
        :offtaker_h_buy => Dict{String, Any}(),
        :offtaker_gc_h_buy => Dict{String, Any}(),
        :offtaker_ep_sell => Dict{String, Any}(),
        :offtaker_cap_EP_green => Dict{String, Any}(),
        :offtaker_inv_EP_green => Dict{String, Any}(),
        :offtaker_gc_h_buy_G => Dict{String, Any}(),
        :offtaker_ep_sell_import => Dict{String, Any}(),
        :elec_GC_demand_d_GC_E => Dict{String, Any}(),
        :EP_demand_d_EP => Dict{String, Any}(),
    )

    # agent_welfare_per_year: per-year welfare Dict from each add_*_to_planner!.
    # agent_welfare: total welfare per agent (sum over years); kept for
    # backward compatibility with save_social_planner_results!.
    agent_welfare_per_year = Dict{String, Dict{Int, Any}}()
    agent_welfare = Dict{String, Any}()

    # ── Classify agents by sub-type ──────────────────────────────────────
    _hasp(m, k) = haskey(m.ext[:parameters], k)
    power_consumers = String[]
    power_vres = String[]
    power_conv = String[]

    for id in agents[:power]
        m = mdict[id]
        atype = String(get(m.ext[:parameters], :Type, ""))
        if atype == "Consumer"
            push!(power_consumers, id)
        elseif atype == "VRES"
            push!(power_vres, id)
        else
            push!(power_conv, id)
        end
    end

    H2_producers = String[]
    H2_consumers = String[]

    for id in agents[:H2]
        m = mdict[id]
        if _hasp(m, :Capacity_Electrolyzer) || (_hasp(m, :E_bar) && _hasp(m, :H_bar))
            push!(H2_producers, id)
        elseif _hasp(m, :D_H_bar)
            push!(H2_consumers, id)
        end
    end

    offtaker_green = String[]
    offtaker_grey = String[]
    offtaker_import = String[]

    for id in agents[:offtaker]
        m = mdict[id]
        atype = String(get(m.ext[:parameters], :Type, ""))
        if atype == "GreenOfftaker"
            push!(offtaker_green, id)
        elseif atype == "EPImporter"
            push!(offtaker_import, id)
        else
            push!(offtaker_grey, id)
        end
    end

    # ── Add agents (each returns per-year welfare Dict) ──────────────────
    # Each add_*_to_planner! function creates the agent's decision variables
    # and physical constraints inside the shared `planner` model, stores
    # variables in `var_dict` for use in market-clearing constraints, and
    # returns a Dict{Int, Any} mapping year jy → JuMP expression for that
    # agent's welfare contribution in year jy (utility for consumers,
    # negative cost for producers). Market transfers (λ·q) cancel in the
    # aggregate and are therefore excluded from welfare expressions.
    #
    # agent_welfare_per_year[id][jy] is needed for the social CVaR, which
    # operates on the per-year aggregate. agent_welfare[id] = Σ_y wpy[jy]
    # is kept for backward compatibility with save_social_planner_results!.

    for id in agents[:power]
        wpy = add_power_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
    end

    for id in H2_producers
        wpy = add_H2_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
    end

    # H₂ consumers: inelastic demand with linear utility (no quadratic
    # term). D_H_bar[jh,jd,jy] = LOAD_H × D_H_bar is the maximum H₂
    # demand in each hour. Per-year welfare = Utility × total H₂ consumed.
    for id in H2_consumers
        m = mdict[id]
        p = m.ext[:parameters]
        ts = m.ext[:timeseries]
        D_H_bar = ts[:LOAD_H] .* p[:D_H_bar]
        utility_val = get(p, :Utility, 0.0)

        d_H = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_H_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], d_H[jh, jd, jy] <= D_H_bar[jh, jd, jy])

        # Per-year welfare = linear utility of H₂ consumption.
        # No expenditure: H₂ payments cancel in the planner aggregate.
        wpy = Dict{Int, Any}()
        for jy in JY
            wpy[jy] = @expression(planner,
                sum(W_dict[jy][jd] * (utility_val * d_H[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
        var_dict[:H2_d_H][id] = d_H
    end

    for id in agents[:offtaker]
        wpy = add_offtaker_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
    end

    for id in agents[:elec_GC_demand]
        wpy = add_elec_GC_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
    end

    for id in agents[:EP_demand]
        wpy = add_EP_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
        agent_welfare_per_year[id] = wpy
        agent_welfare[id] = @expression(planner, sum(wpy[jy] for jy in JY))
    end

    # ── Market-clearing constraints ──────────────────────────────────────
    # In the planner, prices are NOT explicit decision variables — they
    # emerge as shadow prices (dual values) of the market-clearing
    # constraints. Each constraint enforces hourly supply = demand for one
    # market. The haskey() guards handle agents that participate in a
    # market subset (e.g. only VRES supply elec-GCs; conventional does not).

    # Electricity market: VRES + Conventional supply = Consumer demand + H₂ producer intake.
    elec_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:power_q_E], id) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:power_d_E], id) ? var_dict[:power_d_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_e_buy], id) ? var_dict[:H2_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) == 0.0
    )

    # Electricity-GC market: VRES GC supply = H₂ producer GC purchases + GC demand.
    elec_GC_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum((haskey(var_dict[:power_q_E], id) && id in power_vres) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_gc_e_buy], id) ? var_dict[:H2_gc_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:elec_GC_demand_d_GC_E], id) ? var_dict[:elec_GC_demand_d_GC_E][id][jh, jd, jy] : 0.0 for id in agents[:elec_GC_demand]) == 0.0
    )

    # H₂ market: H₂ producer output = H₂ consumer demand + GreenOfftaker purchases.
    H2_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:H2_h_sell], id) ? var_dict[:H2_h_sell][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:H2_d_H], id) ? var_dict[:H2_d_H][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:offtaker_h_buy], id) ? var_dict[:offtaker_h_buy][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) == 0.0
    )

    # H₂-GC market: H₂ producer GC issuance = GreenOfftaker + GreyOfftaker GC purchases.
    H2_GC_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:H2_gc_h_sell], id) ? var_dict[:H2_gc_h_sell][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:offtaker_gc_h_buy], id) ? var_dict[:offtaker_gc_h_buy][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) -
        sum(haskey(var_dict[:offtaker_gc_h_buy_G], id) ? var_dict[:offtaker_gc_h_buy_G][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) == 0.0
    )

    # EP market: GreenOfftaker + EPImporter supply = fixed inelastic demand D_EP + elastic EP demand.
    D_EP = EP_market["D_EP"]
    EP_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:offtaker_ep_sell], id) ? var_dict[:offtaker_ep_sell][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) +
        sum(haskey(var_dict[:offtaker_ep_sell_import], id) ? var_dict[:offtaker_ep_sell_import][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) -
        D_EP[jh, jd, jy] -
        sum(haskey(var_dict[:EP_demand_d_EP], id) ? var_dict[:EP_demand_d_EP][id][jh, jd, jy] : 0.0 for id in agents[:EP_demand]) == 0.0
    )

    # ── Single social CVaR ───────────────────────────────────────────────
    # Instead of per-agent CVaR terms (which incorrectly produced 3 separate
    # risk-aversion terms), the social planner applies a SINGLE CVaR to the
    # aggregate social welfare. γ and β are read from any agent (all agents
    # share the same values). When γ=1 (risk-neutral), the CVaR term
    # vanishes and the planner reduces to the standard welfare maximization,
    # matching the ADMM risk-neutral equilibrium.

    gamma = mdict[agents[:all][1]].ext[:parameters][:γ]
    beta_conf = mdict[agents[:all][1]].ext[:parameters][:β]
    P = mdict[agents[:all][1]].ext[:parameters][:P]

    # ── Aggregate social welfare per year ─────────────────────────────────
    # social_welfare[jy] sums ALL agents' per-year welfare, including
    # quadratic consumer utility terms (A·d − B/2·d²).
    all_ids = collect(keys(agent_welfare_per_year))
    social_welfare = Dict{Int, Any}()
    for jy in JY
        social_welfare[jy] = @expression(planner,
            sum(agent_welfare_per_year[id][jy] for id in all_ids)
        )
    end

    # ── Epigraph variables for social welfare ────────────────────────────
    # social_welfare[jy] contains quadratic consumer utility terms
    # (−B/2·d² from elastic demand agents). If used directly in the CVaR
    # shortfall constraint  u ≥ −social_welfare − α,  the +B/2·d² on the
    # RHS makes it a quadratic constraint (QC), turning the model into a
    # QCP. Gurobi does not reliably provide dual variables for QCPs.
    #
    # Reformulation: introduce epigraph variables sw_aux[jy] constrained
    # by sw_aux[jy] ≤ social_welfare[jy]. In Gurobi's standard form this
    # becomes B/2·d² + sw_aux ≤ A·d + … (PSD Q on the ≤ side), which is
    # the canonical convex QC form. Since the objective maximizes sw_aux,
    # the epigraph constraint is binding at optimality (sw_aux = welfare),
    # so the solution is mathematically identical to applying CVaR
    # directly to social_welfare.
    #
    # The CVaR constraints reference sw_aux instead of the quadratic
    # social_welfare, making them purely LINEAR. The only QC constraints
    # are the epigraph inequalities in standard convex form.
    #
    # DUAL RECOVERY: Gurobi cannot provide duals (Pi) for QCP models.
    # After solving the QCP for optimal quantities, the caller
    # (social_planner.jl) fixes the demand variables (d_E, d_GC_E, d_EP)
    # at their optimal values, converting all QC constraints to linear.
    # A second optimize! call then solves the resulting LP, for which
    # Gurobi provides full dual variables. See social_planner.jl §11.
    sw_aux = @variable(planner, [jy in JY], base_name = "sw_aux")

    # Epigraph constraints: sw_aux[jy] ≤ social_welfare[jy].
    # These are the ONLY quadratic constraints in the model, and they are
    # in standard convex form (PSD on the ≤ side).
    @constraint(planner, social_welfare_epigraph[jy in JY],
        sw_aux[jy] <= social_welfare[jy]
    )

    # ── Social CVaR auxiliary variables ──────────────────────────────────
    # The social loss in year jy is defined as −sw_aux[jy] (≡ −welfare
    # at optimality). CVaR is applied via the standard linear
    # reformulation:
    #   alpha_social = VaR proxy (threshold of the loss distribution)
    #   u_social[jy] = max(0, social_loss[jy] − alpha_social) — shortfall
    #   cvar_social  ≥ alpha_social + (1/(1−β)) · Σ P[jy]·u_social[jy]
    # lower_bound = 0 on all three is consistent with the individual ADMM
    # agent formulations. All constraints below are purely LINEAR.
    alpha_social = @variable(planner, lower_bound = 0, base_name = "alpha_social")
    cvar_social  = @variable(planner, lower_bound = 0, base_name = "CVaR_social")
    u_social     = @variable(planner, [jy in JY], lower_bound = 0, base_name = "u_social")

    # Shortfall: u[jy] ≥ −sw_aux[jy] − α (purely LINEAR).
    @constraint(planner, [jy in JY],
        u_social[jy] >= -sw_aux[jy] - alpha_social
    )
    # CVaR linking: cvar_social ≥ α + (1/(1−β)) · Σ P[jy]·u_social[jy].
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    @constraint(planner,
        cvar_social >= alpha_social + (1 / one_minus_beta) * sum(P[jy] * u_social[jy] for jy in JY)
    )

    # ── Planner objective ─────────────────────────────────────────────────
    #   max  γ · Σ_y sw_aux[y]             ← (1) expected social welfare
    #      − (1−γ) · CVaR_social           ← (2) tail-risk penalty
    #
    # (1) sw_aux[jy] = social_welfare[jy] at optimality (epigraph binds
    #     because we maximize). This keeps the objective LINEAR; the
    #     quadratic consumer utility terms live only in the epigraph QC.
    # (2) CVaR of the social loss. When γ=1 (risk-neutral) this term
    #     vanishes, and the planner maximizes expected welfare — matching
    #     the ADMM risk-neutral equilibrium by the first welfare theorem.
    @objective(planner, Max,
        gamma * sum(sw_aux[jy] for jy in JY)
        - (1 - gamma) * cvar_social
    )

    # ── Return planner model and state dictionary ─────────────────────────
    # planner_state carries all information that save_social_planner_results!
    # and post-processing scripts need:
    #   :var_dict              — agent decision variables by market role
    #   :agent_welfare         — total welfare per agent (Σ_y), for results
    #   :agent_welfare_per_year — per-year welfare Dict (for social CVaR)
    #   :social_welfare        — aggregate social welfare per year
    #   :*_balance             — market-clearing constraints (duals = prices)
    #   :power_*/H2_*/offtaker_* — agent ID lists by sub-type
    #   :JH, :JD, :JY, :W     — index sets and weights
    #   :gamma, :beta          — risk parameters (shared by all agents)
    #   :demand_var_keys       — var_dict keys for demand agents whose
    #                            variables must be fixed for dual recovery
    planner_state = Dict(
        :var_dict => var_dict,
        :agent_welfare => agent_welfare,
        :agent_welfare_per_year => agent_welfare_per_year,
        :social_welfare => social_welfare,
        :elec_balance => elec_balance,
        :elec_GC_balance => elec_GC_balance,
        :H2_balance => H2_balance,
        :H2_GC_balance => H2_GC_balance,
        :EP_balance => EP_balance,
        :power_consumers => power_consumers,
        :power_vres => power_vres,
        :power_conv => power_conv,
        :H2_producers => H2_producers,
        :H2_consumers => H2_consumers,
        :offtaker_green => offtaker_green,
        :offtaker_grey => offtaker_grey,
        :offtaker_import => offtaker_import,
        :JH => JH,
        :JD => JD,
        :JY => JY,
        :W => W,
        :gamma => gamma,
        :beta => beta_conf,
        :demand_var_keys => [:power_d_E, :elec_GC_demand_d_GC_E, :EP_demand_d_EP],
        :sw_aux => sw_aux,
    )

    return planner, planner_state
end
