# ==============================================================================
# compute_agent_objective.jl — Economic objective computation for agents
# ==============================================================================
#
# PURPOSE:
#   Computes the ADMM-style economic objective (cost − revenue) for each agent
#   type, used by save_results.jl and save_social_planner_results.jl to report
#   per-agent objective values that match the underlying JuMP model formulation.
#
#   - compute_agent_objective_economic: returns the total scalar objective
#     (sum over timesteps, weighted by W, plus fixed CAPEX where applicable).
#   - compute_agent_objective_contributions: returns a 3D array of per-timestep
#     contributions (variable/operational part only; fixed CAPEX is in the
#     total). Used for Agent_Objectives_Per_Timestep.csv.
#
#   Agent types: :power_consumer, :power_vres, :power_conv, :H2_producer,
#   :H2_consumer, :offtaker_green, :offtaker_grey, :offtaker_import,
#   :elec_GC_demand, :EP_demand.
#
# ==============================================================================

"""
    compute_agent_objective_economic(agent_type, quantities, prices, params; JH, JD, JY)

Compute the total economic objective (cost − revenue) for an agent of the given type.

- `quantities`: Dict with keys like :g, :d, :e_in, :q_elec_gc, :h2_out, :q_h2gc,
  :cap_VRES, :cap_H2_y, :cap_EP_y (as needed per agent type).
- `prices`: Dict with :λ_elec, :λ_H2, :λ_elec_GC, :λ_H2_GC, :λ_EP (3D arrays).
- `params`: Dict with W, A_E, B_E, MarginalCost, etc.
- Returns: Float64 (total objective).
"""
function compute_agent_objective_economic(agent_type::Symbol, quantities::Dict, prices::Dict, params::Dict; JH, JD, JY)
    W = params[:W]
    obj = 0.0

    if agent_type == :power_consumer
        d = quantities[:d]
        λ_elec = prices[:λ_elec]
        A_E = get(params, :A_E, 500.0)
        B_E = get(params, :B_E, 0.5)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2))
        end

    elseif agent_type == :power_vres
        g = quantities[:g]
        λ_elec = prices[:λ_elec]
        λ_elec_GC = prices[:λ_elec_GC]
        MC = get(params, :MarginalCost, 0.0)
        F_cap = get(params, :FixedCost_per_MW, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy] - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy])
        end
        if haskey(quantities, :cap_VRES)
            for jy in JY
                obj += F_cap * quantities[:cap_VRES][jy]
            end
        end

    elseif agent_type == :power_conv
        g = quantities[:g]
        λ_elec = prices[:λ_elec]
        MC = get(params, :MarginalCost, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy])
        end

    elseif agent_type == :H2_producer
        e_in = quantities[:e_in]
        q_elec_gc = quantities[:q_elec_gc]
        h2_out = quantities[:h2_out]
        q_h2gc = quantities[:q_h2gc]
        λ_elec = prices[:λ_elec]
        λ_elec_GC = prices[:λ_elec_GC]
        λ_H2 = prices[:λ_H2]
        λ_H2_GC = prices[:λ_H2_GC]
        op_cost = get(params, :OperationalCost, 0.0)
        F_cap = get(params, :FixedCost_per_MW_Electrolyzer, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (
                λ_elec[jh, jd, jy] * e_in[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy] * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
            )
        end
        if haskey(quantities, :cap_H2_y)
            for jy in JY
                obj += F_cap * quantities[:cap_H2_y][jy]
            end
        end

    elseif agent_type == :H2_consumer
        d_H = quantities[:d_H]
        λ_H2 = prices[:λ_H2]
        utility_val = get(params, :Utility, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (λ_H2[jh, jd, jy] * d_H[jh, jd, jy] - utility_val * d_H[jh, jd, jy])
        end

    elseif agent_type == :offtaker_green
        h2_in = quantities[:h2_in]
        q_h2gc = quantities[:q_h2gc]
        ep = quantities[:ep]
        λ_H2 = prices[:λ_H2]
        λ_H2_GC = prices[:λ_H2_GC]
        λ_EP = prices[:λ_EP]
        proc_cost = get(params, :ProcessingCost, 0.0)
        F_cap = get(params, :FixedCost_per_MW_EP_Out, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (
                λ_H2[jh, jd, jy] * h2_in[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
            )
        end
        if haskey(quantities, :cap_EP_y)
            for jy in JY
                obj += F_cap * quantities[:cap_EP_y][jy]
            end
        end

    elseif agent_type == :offtaker_grey
        ep = quantities[:ep]
        q_h2gc = quantities[:q_h2gc]
        λ_H2_GC = prices[:λ_H2_GC]
        λ_EP = prices[:λ_EP]
        MC = get(params, :MarginalCost, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (MC * ep[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy])
        end

    elseif agent_type == :offtaker_import
        ep = quantities[:ep]
        λ_EP = prices[:λ_EP]
        imp_cost = get(params, :ImportCost, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy])
        end

    elseif agent_type == :elec_GC_demand
        d_gc = quantities[:d_gc]
        λ_elec_GC = prices[:λ_elec_GC]
        A_GC = get(params, :A_GC, 20.0)
        B_GC = get(params, :B_GC, 0.5)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (λ_elec_GC[jh, jd, jy] * d_gc[jh, jd, jy] - (A_GC * d_gc[jh, jd, jy] - B_GC/2 * d_gc[jh, jd, jy]^2))
        end

    elseif agent_type == :EP_demand
        d_EP = quantities[:d_EP]
        λ_EP = prices[:λ_EP]
        A_EP = get(params, :A_EP, 150.0)
        B_EP = get(params, :B_EP, 0.5)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (λ_EP[jh, jd, jy] * d_EP[jh, jd, jy] - (A_EP * d_EP[jh, jd, jy] - B_EP/2 * d_EP[jh, jd, jy]^2))
        end

    # ── PPA variants (used only by market_exposure_contracts) ─────────
    elseif agent_type == :power_vres_ppa
        g_EOM = quantities[:g_EOM]
        g_ppa = quantities[:g_ppa]
        λ_elec = prices[:λ_elec]
        λ_elec_GC = prices[:λ_elec_GC]
        λ_ppa = prices[:λ_ppa]
        MC = get(params, :MarginalCost, 0.0)
        F_cap = get(params, :FixedCost_per_MW, 0.0)
        for jh in JH, jd in JD, jy in JY
            obj += W[jd, jy] * (
                MC * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_ppa[jh, jd, jy] * g_ppa[jh, jd, jy]
            )
        end
        if haskey(quantities, :cap_VRES)
            for jy in JY
                obj += F_cap * quantities[:cap_VRES][jy]
            end
        end

    elseif agent_type == :H2_producer_ppa
        e_in_pool = quantities[:e_in_pool]
        g_ppa_from = quantities[:g_ppa_from]
        q_elec_gc = quantities[:q_elec_gc]
        h2_out = quantities[:h2_out]
        q_h2gc = quantities[:q_h2gc]
        λ_elec = prices[:λ_elec]
        λ_elec_GC = prices[:λ_elec_GC]
        λ_ppa = prices[:λ_ppa]  # Dict(vres_id => 3D array)
        λ_H2 = prices[:λ_H2]
        λ_H2_GC = prices[:λ_H2_GC]
        op_cost = get(params, :OperationalCost, 0.0)
        F_cap = get(params, :FixedCost_per_MW_Electrolyzer, 0.0)
        for jh in JH, jd in JD, jy in JY
            contract_term = 0.0
            for v in keys(g_ppa_from)
                contract_term += λ_ppa[v][jh, jd, jy] * g_ppa_from[v][jh, jd, jy]
            end
            obj += W[jd, jy] * (
                λ_elec[jh, jd, jy] * e_in_pool[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + contract_term
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy] * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
            )
        end
        if haskey(quantities, :cap_H2_y)
            for jy in JY
                obj += F_cap * quantities[:cap_H2_y][jy]
            end
        end

    else
        return 0.0
    end

    return obj
end

"""
    compute_agent_objective_contributions(agent_type, quantities, prices_dict, params; JH, JD, JY)

Compute per-timestep economic objective contributions (variable/operational part only).
Returns a 3D array indexed by (jh, jd, jy). Fixed CAPEX is excluded (handled in
compute_agent_objective_economic).
"""
function compute_agent_objective_contributions(agent_type::Symbol, quantities::Dict, prices_dict::Dict, params::Dict; JH, JD, JY)
    JH_vec = collect(JH)
    JD_vec = collect(JD)
    JY_vec = collect(JY)
    n_h = length(JH_vec)
    n_d = length(JD_vec)
    n_y = length(JY_vec)
    contrib = zeros(n_h, n_d, n_y)

    if agent_type == :power_consumer
        d = quantities[:d]
        λ_elec = prices_dict[:λ_elec]
        A_E = get(params, :A_E, 500.0)
        B_E = get(params, :B_E, 0.5)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2)
        end

    elseif agent_type == :power_vres
        g = quantities[:g]
        λ_elec = prices_dict[:λ_elec]
        λ_elec_GC = prices_dict[:λ_elec_GC]
        MC = get(params, :MarginalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy] - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]
        end

    elseif agent_type == :power_conv
        g = quantities[:g]
        λ_elec = prices_dict[:λ_elec]
        MC = get(params, :MarginalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy]
        end

    elseif agent_type == :H2_producer
        e_in = quantities[:e_in]
        q_elec_gc = quantities[:q_elec_gc]
        h2_out = quantities[:h2_out]
        q_h2gc = quantities[:q_h2gc]
        λ_elec = prices_dict[:λ_elec]
        λ_elec_GC = prices_dict[:λ_elec_GC]
        λ_H2 = prices_dict[:λ_H2]
        λ_H2_GC = prices_dict[:λ_H2_GC]
        op_cost = get(params, :OperationalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = (
                λ_elec[jh, jd, jy] * e_in[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy] * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
            )
        end

    elseif agent_type == :H2_consumer
        d_H = quantities[:d_H]
        λ_H2 = prices_dict[:λ_H2]
        utility_val = get(params, :Utility, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = λ_H2[jh, jd, jy] * d_H[jh, jd, jy] - utility_val * d_H[jh, jd, jy]
        end

    elseif agent_type == :offtaker_green
        h2_in = quantities[:h2_in]
        q_h2gc = quantities[:q_h2gc]
        ep = quantities[:ep]
        λ_H2 = prices_dict[:λ_H2]
        λ_H2_GC = prices_dict[:λ_H2_GC]
        λ_EP = prices_dict[:λ_EP]
        proc_cost = get(params, :ProcessingCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = (
                λ_H2[jh, jd, jy] * h2_in[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
            )
        end

    elseif agent_type == :offtaker_grey
        ep = quantities[:ep]
        q_h2gc = quantities[:q_h2gc]
        λ_H2_GC = prices_dict[:λ_H2_GC]
        λ_EP = prices_dict[:λ_EP]
        MC = get(params, :MarginalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = MC * ep[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
        end

    elseif agent_type == :offtaker_import
        ep = quantities[:ep]
        λ_EP = prices_dict[:λ_EP]
        imp_cost = get(params, :ImportCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
        end

    elseif agent_type == :elec_GC_demand
        d_gc = quantities[:d_gc]
        λ_elec_GC = prices_dict[:λ_elec_GC]
        A_GC = get(params, :A_GC, 20.0)
        B_GC = get(params, :B_GC, 0.5)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = λ_elec_GC[jh, jd, jy] * d_gc[jh, jd, jy] - (A_GC * d_gc[jh, jd, jy] - B_GC/2 * d_gc[jh, jd, jy]^2)
        end

    elseif agent_type == :EP_demand
        d_EP = quantities[:d_EP]
        λ_EP = prices_dict[:λ_EP]
        A_EP = get(params, :A_EP, 150.0)
        B_EP = get(params, :B_EP, 0.5)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = λ_EP[jh, jd, jy] * d_EP[jh, jd, jy] - (A_EP * d_EP[jh, jd, jy] - B_EP/2 * d_EP[jh, jd, jy]^2)
        end

    # ── PPA variants (used only by market_exposure_contracts) ─────────
    elseif agent_type == :power_vres_ppa
        g_EOM = quantities[:g_EOM]
        g_ppa = quantities[:g_ppa]
        λ_elec = prices_dict[:λ_elec]
        λ_elec_GC = prices_dict[:λ_elec_GC]
        λ_ppa = prices_dict[:λ_ppa]
        MC = get(params, :MarginalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contrib[ih, id, iy] = (
                MC * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_ppa[jh, jd, jy] * g_ppa[jh, jd, jy]
            )
        end

    elseif agent_type == :H2_producer_ppa
        e_in_pool = quantities[:e_in_pool]
        g_ppa_from = quantities[:g_ppa_from]
        q_elec_gc = quantities[:q_elec_gc]
        h2_out = quantities[:h2_out]
        q_h2gc = quantities[:q_h2gc]
        λ_elec = prices_dict[:λ_elec]
        λ_elec_GC = prices_dict[:λ_elec_GC]
        λ_ppa = prices_dict[:λ_ppa]  # Dict(vres_id => 3D array)
        λ_H2 = prices_dict[:λ_H2]
        λ_H2_GC = prices_dict[:λ_H2_GC]
        op_cost = get(params, :OperationalCost, 0.0)
        for (iy, jy) in enumerate(JY_vec), (id, jd) in enumerate(JD_vec), (ih, jh) in enumerate(JH_vec)
            contract_term = 0.0
            for v in keys(g_ppa_from)
                contract_term += λ_ppa[v][jh, jd, jy] * g_ppa_from[v][jh, jd, jy]
            end
            contrib[ih, id, iy] = (
                λ_elec[jh, jd, jy] * e_in_pool[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy] * q_elec_gc[jh, jd, jy]
                + contract_term
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy] * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
            )
        end

    else
        return contrib
    end

    return contrib
end
