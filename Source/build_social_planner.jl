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
#   planner — JuMP model (ready to optimize).
#   planner_state — Dict with :var_dict, :agent_welfare, :elec_balance, :H2_balance,
#     :power_consumers, :power_vres, :power_conv, :H2_producers, :H2_consumers,
#     :offtaker_green, :offtaker_grey, :offtaker_import for save_results.
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
        :H2_e_buy => Dict{String, Any}(),
        :H2_gc_e_buy => Dict{String, Any}(),
        :H2_h_sell => Dict{String, Any}(),
        :H2_gc_h_sell => Dict{String, Any}(),
        :H2_d_H => Dict{String, Any}(),
        :offtaker_h_buy => Dict{String, Any}(),
        :offtaker_gc_h_buy => Dict{String, Any}(),
        :offtaker_ep_sell => Dict{String, Any}(),
        :offtaker_gc_h_buy_G => Dict{String, Any}(),
        :offtaker_ep_sell_import => Dict{String, Any}(),
        :elec_GC_demand_d_GC_E => Dict{String, Any}(),
        :EP_demand_d_EP => Dict{String, Any}(),
    )

    agent_welfare = Dict{String, JuMP.AbstractJuMPScalar}()

    # ── Classify agents by sub-type ──────────────────────────────────────
    # Separate lists by type because market-clearing constraints need to sum
    # over specific subsets. For example, only VRES agents contribute to the
    # electricity-GC supply side, and only H₂ producers appear in the H₂
    # supply term. Building these lists once avoids repeated type checks
    # inside the constraint macros.
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

    # Add power agents
    for id in agents[:power]
        agent_welfare[id] = add_power_agent_to_planner!(planner, id, mdict[id], var_dict, W)
    end

    # Add H2 producers
    for id in H2_producers
        agent_welfare[id] = add_H2_agent_to_planner!(planner, id, mdict[id], var_dict, W)
    end

    # Add H2 consumers (if any)
    for id in H2_consumers
        m = mdict[id]
        p = m.ext[:parameters]
        ts = m.ext[:timeseries]
        D_H_bar = ts[:LOAD_H] .* p[:D_H_bar]
        utility_val = get(p, :Utility, 0.0)

        d_H = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="d_H_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], d_H[jh, jd, jy] <= D_H_bar[jh, jd, jy])

        obj = @expression(planner,
            sum(W_dict[jy][jd] * (utility_val * d_H[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
        )
        agent_welfare[id] = obj
        var_dict[:H2_d_H][id] = d_H
    end

    # Add offtakers
    for id in agents[:offtaker]
        agent_welfare[id] = add_offtaker_agent_to_planner!(planner, id, mdict[id], var_dict, W)
    end

    # Add electricity GC demand
    for id in agents[:elec_GC_demand]
        agent_welfare[id] = add_elec_GC_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
    end

    # Add EP demand (if any)
    for id in agents[:EP_demand]
        agent_welfare[id] = add_EP_demand_agent_to_planner!(planner, id, mdict[id], var_dict, W)
    end

    # ── Market-clearing constraints ──────────────────────────────────────
    # Each constraint enforces supply = demand for the corresponding market
    # at each timestep. The haskey checks handle agents that do not
    # participate in a given market (returning 0.0 so they are neutral).
    #
    # Electricity balance: generation − consumption − electrolyzer elec buy = 0
    elec_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:power_q_E], id) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:power_d_E], id) ? var_dict[:power_d_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_e_buy], id) ? var_dict[:H2_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) == 0.0
    )

    # Electricity GC balance: VRES generation (= GC supply) − electrolyzer
    # GC purchases − GC demand agents = 0
    elec_GC_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum((haskey(var_dict[:power_q_E], id) && id in power_vres) ? var_dict[:power_q_E][id][jh, jd, jy] : 0.0 for id in agents[:power]) -
        sum(haskey(var_dict[:H2_gc_e_buy], id) ? var_dict[:H2_gc_e_buy][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:elec_GC_demand_d_GC_E], id) ? var_dict[:elec_GC_demand_d_GC_E][id][jh, jd, jy] : 0.0 for id in agents[:elec_GC_demand]) == 0.0
    )

    # Hydrogen balance: H₂ production − H₂ consumption − offtaker H₂ buys = 0
    H2_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:H2_h_sell], id) ? var_dict[:H2_h_sell][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:H2_d_H], id) ? var_dict[:H2_d_H][id][jh, jd, jy] : 0.0 for id in agents[:H2]) -
        sum(haskey(var_dict[:offtaker_h_buy], id) ? var_dict[:offtaker_h_buy][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) == 0.0
    )

    # H₂ GC balance — *annual* (indexed only by jy, not jh/jd) because H₂
    # green-certificate trading is settled on a yearly basis, not hourly.
    # The weighted sum (W_dict[jy][jd] * ...) aggregates hourly GC flows
    # into a single annual quantity per year.
    H2_GC_balance = @constraint(planner, [jy in JY],
        sum(sum(W_dict[jy][jd] * (haskey(var_dict[:H2_gc_h_sell], id) ? var_dict[:H2_gc_h_sell][id][jh, jd, jy] : 0.0) for jh in JH, jd in JD) for id in agents[:H2]) -
        sum(sum(W_dict[jy][jd] * (haskey(var_dict[:offtaker_gc_h_buy], id) ? var_dict[:offtaker_gc_h_buy][id][jh, jd, jy] : 0.0) for jh in JH, jd in JD) for id in agents[:offtaker]) -
        sum(sum(W_dict[jy][jd] * (haskey(var_dict[:offtaker_gc_h_buy_G], id) ? var_dict[:offtaker_gc_h_buy_G][id][jh, jd, jy] : 0.0) for jh in JH, jd in JD) for id in agents[:offtaker]) == 0.0
    )

    # End-product balance: offtaker EP supply + EP importer supply
    #                    − D_EP (fixed inelastic demand, not a decision variable)
    #                    − EP demand agents = 0
    # D_EP is subtracted directly because it represents exogenous / fixed
    # demand that is not modeled as an optimizable agent decision.
    D_EP = EP_market["D_EP"]
    EP_balance = @constraint(planner, [jy in JY, jh in JH, jd in JD],
        sum(haskey(var_dict[:offtaker_ep_sell], id) ? var_dict[:offtaker_ep_sell][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) +
        sum(haskey(var_dict[:offtaker_ep_sell_import], id) ? var_dict[:offtaker_ep_sell_import][id][jh, jd, jy] : 0.0 for id in agents[:offtaker]) -
        D_EP[jh, jd, jy] -
        sum(haskey(var_dict[:EP_demand_d_EP], id) ? var_dict[:EP_demand_d_EP][id][jh, jd, jy] : 0.0 for id in agents[:EP_demand]) == 0.0
    )

    # Objective: maximize total welfare = Σ (each agent's surplus or cost
    # contribution). This is the social planner's goal: find the allocation
    # that maximizes the sum of consumer surplus and producer profit across
    # all agents, subject to market-clearing and individual constraints.
    @objective(planner, Max, sum(agent_welfare[id] for id in keys(agent_welfare)))

    # planner_state: collects all data that save_social_planner_results!
    # needs (variable dicts, welfare expressions, balance constraints, agent
    # classification lists, and index sets) so we can pass a single Dict
    # instead of many separate arguments.
    planner_state = Dict(
        :var_dict => var_dict,
        :agent_welfare => agent_welfare,
        :elec_balance => elec_balance,
        :H2_balance => H2_balance,
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
    )

    return planner, planner_state
end
