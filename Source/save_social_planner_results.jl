# ==============================================================================
# save_social_planner_results.jl — Write social planner outputs to CSV
# ==============================================================================
#
# PURPOSE:
#   Saves Market_Prices.csv (dual of balance constraints) and Agent_Summary.csv
#   from the solved social planner model.
#
# ARGUMENTS:
#   planner — Solved JuMP model.
#   planner_state — Dict from build_social_planner! (:var_dict, :agent_welfare,
#     :elec_balance, :H2_balance, :power_consumers, :power_vres, :power_conv,
#     :H2_producers, :H2_consumers, :offtaker_green, :offtaker_grey, :offtaker_import,
#     :JH, :JD, :JY).
#   agents — Dict of agent lists.
#   results_folder — Path to social_planner_results directory.
#
# ==============================================================================

function save_social_planner_results!(planner::Model, planner_state::Dict, agents::Dict,
                                      results_folder::String)
    var_dict = planner_state[:var_dict]
    agent_welfare = planner_state[:agent_welfare]
    power_consumers = planner_state[:power_consumers]
    power_vres = planner_state[:power_vres]
    power_conv = planner_state[:power_conv]
    H2_producers = planner_state[:H2_producers]
    H2_consumers = planner_state[:H2_consumers]
    offtaker_green = planner_state[:offtaker_green]
    offtaker_grey = planner_state[:offtaker_grey]
    offtaker_import = planner_state[:offtaker_import]
    JH = planner_state[:JH]
    JD = planner_state[:JD]
    JY = planner_state[:JY]

    elec_balance = planner_state[:elec_balance]
    H2_balance = planner_state[:H2_balance]

    # ── Market_Prices.csv — Equilibrium prices from dual variables ──────
    # By LP/QP duality, the shadow price (dual value) of a market-clearing
    # constraint equals the equilibrium price in that market at that
    # timestep. Extracting duals is the standard way to recover prices from
    # a centralized welfare-maximization problem.
    prices_rows = Vector{NamedTuple{(:Time, :Elec_Price, :H2_Price), Tuple{Int, Float64, Float64}}}()

    # Iterate over (jy, jd, jh) — this ordering must match the constraint
    # indexing used in build_social_planner! so that each dual maps to the
    # correct timestep.
    t_index = 1
    for jy in JY, jd in JD, jh in JH
        push!(prices_rows, (
            Time = t_index,
            Elec_Price = dual(elec_balance[jy, jh, jd]),
            H2_Price = dual(H2_balance[jy, jh, jd])
        ))
        t_index += 1
    end

    prices_df = DataFrame(prices_rows)
    CSV.write(joinpath(results_folder, "Market_Prices.csv"), prices_df)

    # ── Agent_Summary.csv — Per-agent quantity and welfare contribution ──
    # For every agent, record:
    #   Total_Quantity        — sum of the agent's primary decision variable
    #                           values across all (jh, jd, jy) timesteps. This
    #                           represents the total energy produced or consumed
    #                           over the full modeled horizon.
    #   Welfare_Contribution  — value of the agent's welfare expression as
    #                           evaluated at the optimal solution. For demand
    #                           agents this is consumer surplus; for supply
    #                           agents it is negative cost (profit contribution).
    summary = DataFrame(
        Agent = String[],
        Type = String[],
        Total_Quantity = Float64[],
        Welfare_Contribution = Float64[]
    )

    for id in agents[:all]
        ag_type = "Unknown"
        qty = 0.0

        if id in power_consumers
            ag_type = "PowerCons"
            d = var_dict[:power_d_E][id]
            qty = sum(value.(d))
        elseif id in power_vres || id in power_conv
            ag_type = "PowerGen"
            q = var_dict[:power_q_E][id]
            qty = sum(value.(q))
        elseif id in H2_producers
            ag_type = "H2Prod"
            h = var_dict[:H2_h_sell][id]
            qty = sum(value.(h))
        elseif id in H2_consumers
            ag_type = "H2Cons"
            dH = var_dict[:H2_d_H][id]
            qty = sum(value.(dH))
        elseif id in offtaker_green || id in offtaker_grey || id in offtaker_import
            ag_type = "Offtaker"
            ep = haskey(var_dict[:offtaker_ep_sell], id) ? var_dict[:offtaker_ep_sell][id] :
                  haskey(var_dict[:offtaker_ep_sell_import], id) ? var_dict[:offtaker_ep_sell_import][id] : nothing
            if ep !== nothing
                qty = sum(value.(ep))
            end
        elseif id in agents[:elec_GC_demand]
            ag_type = "GC_Demand"
            dgc = var_dict[:elec_GC_demand_d_GC_E][id]
            qty = sum(value.(dgc))
        elseif id in agents[:EP_demand]
            ag_type = "EP_Demand"
            dep = var_dict[:EP_demand_d_EP][id]
            qty = sum(value.(dep))
        end

        welfare_val = haskey(agent_welfare, id) ? value(agent_welfare[id]) : 0.0
        push!(summary, (id, ag_type, qty, welfare_val))
    end

    CSV.write(joinpath(results_folder, "Agent_Summary.csv"), summary)
    return nothing
end
