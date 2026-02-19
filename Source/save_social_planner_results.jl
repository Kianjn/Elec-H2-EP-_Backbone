# ==============================================================================
# save_social_planner_results.jl — Write social planner outputs to CSV
# ==============================================================================
#
# PURPOSE:
#   Saves Market_Prices.csv (dual of balance constraints) and Agent_Summary.csv
#   from the solved social planner model.
#
#   POST-PROCESSING: The social planner maximizes welfare (no prices in objective);
#   prices emerge as dual variables. To match the ADMM Agent_Summary format
#   (which reports cost − revenue per agent), we post-process: after extracting
#   equilibrium prices from the duals, we compute for each agent the ADMM-style
#   objective (cost − revenue) evaluated at the optimal quantities and those
#   prices. This ensures direct comparability with market_exposure_results.
#
# ARGUMENTS:
#   planner — Solved JuMP model.
#   planner_state — Dict from build_social_planner! (:var_dict, :agent_welfare,
#     :elec_balance, :H2_balance, :power_consumers, :power_vres, :power_conv,
#     :H2_producers, :H2_consumers, :offtaker_green, :offtaker_grey, :offtaker_import,
#     :JH, :JD, :JY).
#   agents — Dict of agent lists.
#   mdict — Dict of parameter-container models (one per agent); needed to read
#     cost/utility parameters for ADMM-style objective computation.
#   results_folder — Path to social_planner_results directory.
#
# ==============================================================================

function save_social_planner_results!(planner::Model, planner_state::Dict, agents::Dict,
                                      mdict::Dict, results_folder::String)
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
    elec_GC_balance = planner_state[:elec_GC_balance]
    H2_balance = planner_state[:H2_balance]
    H2_GC_balance = planner_state[:H2_GC_balance]
    EP_balance = planner_state[:EP_balance]
    W = planner_state[:W]   # Representative-day weights W[jd, jy]

    # ── Check solver status before extracting results ─────────────────
    status = termination_status(planner)
    if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
        @warn "Social planner did NOT solve to optimality (status: $status). " *
              "Cannot extract duals or variable values."
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            @error "Model is INFEASIBLE. Check that demand ≤ total supply capacity " *
                   "for all markets (especially EP: D_EP = Total_Demand × LOAD_EP ≤ Σ EP capacities)."
        end
        return
    end

    # ── Market_Prices.csv — Equilibrium prices from dual variables ──────
    # By LP/QP duality, the shadow price (dual value) of a market-clearing
    # constraint equals the equilibrium price in that market at that
    # timestep. Extracting duals is the standard way to recover prices from
    # a centralized welfare-maximization problem. Includes all 5 markets.
    #
    # IMPORTANT: The raw duals of per-timestep constraints include the
    # representative-day weight W[jd,jy] as a scaling factor, because the
    # objective sums W * welfare but the constraints are unweighted:
    #   raw_dual = W * true_price.
    # We divide by W[jd,jy] to recover the actual equilibrium price.
    #
    # H2_GC is now hourly (same as other markets), so we divide by W to recover
    # the true price from the raw dual.
    prices_rows = []
    t_index = 1
    for jy in JY, jd in JD, jh in JH
        w = W[jd, jy]
        push!(prices_rows, (
            Time = t_index,
            Elec_Price = dual(elec_balance[jy, jh, jd]) / w,
            H2_Price = dual(H2_balance[jy, jh, jd]) / w,
            Elec_GC_Price = dual(elec_GC_balance[jy, jh, jd]) / w,
            H2_GC_Price = dual(H2_GC_balance[jy, jh, jd]) / w,
            EP_Price = dual(EP_balance[jy, jh, jd]) / w,
        ))
        t_index += 1
    end

    prices_df = DataFrame(prices_rows)
    CSV.write(joinpath(results_folder, "Market_Prices.csv"), prices_df)

    # Print equilibrium prices to the output log (same format as market exposure)
    # These duals are the THEORETICAL BENCHMARK: true equilibrium prices from
    # centralized welfare maximization. Run market_exposure.jl to compare ADMM
    # and FOC-extracted prices against this benchmark.
    println()
    println("Social planner optimization completed.")
    println("Equilibrium prices (from dual variables, saved to Market_Prices.csv):")
    println("  Electricity     mean = ", round(mean(prices_df.Elec_Price), digits=6))
    println("  Hydrogen        mean = ", round(mean(prices_df.H2_Price), digits=6))
    println("  Electricity_GC  mean = ", round(mean(prices_df.Elec_GC_Price), digits=6))
    println("  H2_GC           mean = ", round(mean(prices_df.H2_GC_Price), digits=6))
    println("  End_Product     mean = ", round(mean(prices_df.EP_Price), digits=6))

    # ── Build 3D price arrays [jh, jd, jy] for ADMM-style objective computation ─
    # The duals are indexed [jy, jh, jd]. We build λ[jh, jd, jy] to match the
    # ADMM agent objective formulas.
    n_jh, n_jd, n_jy = length(JH), length(JD), length(JY)
    λ_elec    = zeros(n_jh, n_jd, n_jy)
    λ_H2      = zeros(n_jh, n_jd, n_jy)
    λ_elec_GC = zeros(n_jh, n_jd, n_jy)
    λ_H2_GC   = zeros(n_jh, n_jd, n_jy)
    λ_EP      = zeros(n_jh, n_jd, n_jy)
    for (iy, jy) in enumerate(JY), (id, jd) in enumerate(JD), (ih, jh) in enumerate(JH)
        w = W[jd, jy]
        λ_elec[ih, id, iy]    = dual(elec_balance[jy, jh, jd]) / w
        λ_H2[ih, id, iy]      = dual(H2_balance[jy, jh, jd]) / w
        λ_elec_GC[ih, id, iy] = dual(elec_GC_balance[jy, jh, jd]) / w
        λ_H2_GC[ih, id, iy]   = dual(H2_GC_balance[jy, jh, jd]) / w
        λ_EP[ih, id, iy]      = dual(EP_balance[jy, jh, jd]) / w
    end

    # Helper: compute ADMM-style objective (cost − revenue) for an agent.
    # Uses optimal quantities and equilibrium prices. Matches the formulas in
    # build_*_agent.jl (without ADMM penalty terms, which vanish at equilibrium).
    function _admm_objective(id, ag_type)
        m = mdict[id]
        p = m.ext[:parameters]
        _get(m, k, def) = get(p, k, def)
        obj = 0.0
        if id in power_consumers
            d = var_dict[:power_d_E][id]
            A_E = _get(m, :A_E, 500.0)
            B_E = _get(m, :B_E, 0.5)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                dv = value(d[jh, jd, jy])
                w = W[jd, jy]
                # ADMM: min λ·d − U(d) = λ·d − (A_E·d − B_E/2·d²)
                obj += w * (λ_elec[ih, id_d, iy] * dv - (A_E * dv - B_E/2 * dv^2))
            end
        elseif id in power_vres
            q = var_dict[:power_q_E][id]
            cap_VRES = haskey(var_dict, :power_cap_VRES) && haskey(var_dict[:power_cap_VRES], id) ? var_dict[:power_cap_VRES][id] : nothing
            MC = _get(m, :MarginalCost, 0.0)
            F_cap = _get(m, :FixedCost_per_MW, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                qv = value(q[jh, jd, jy])
                w = W[jd, jy]
                # ADMM: min MC·g − λ_elec·g − λ_GC·g
                obj += w * (MC * qv - λ_elec[ih, id_d, iy] * qv - λ_elec_GC[ih, id_d, iy] * qv)
            end
            # Fixed capacity cost (no W weighting): F_cap [€/MW-year] × capacity per year.
            if cap_VRES !== nothing
                for (iy, jy) in enumerate(JY)
                    obj += F_cap * value(cap_VRES[jy])
                end
            end
        elseif id in power_conv
            q = var_dict[:power_q_E][id]
            MC = _get(m, :MarginalCost, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                qv = value(q[jh, jd, jy])
                w = W[jd, jy]
                # ADMM: min MC·g − λ_elec·g
                obj += w * (MC * qv - λ_elec[ih, id_d, iy] * qv)
            end
        elseif id in H2_producers
            e_buy = var_dict[:H2_e_buy][id]
            gc_e_buy = var_dict[:H2_gc_e_buy][id]
            h_sell = var_dict[:H2_h_sell][id]
            gc_h_sell = var_dict[:H2_gc_h_sell][id]
            cap_H2_y = haskey(var_dict, :H2_cap_elec) && haskey(var_dict[:H2_cap_elec], id) ? var_dict[:H2_cap_elec][id] : nothing
            op_cost = _get(m, :OperationalCost, 0.0)
            F_cap = _get(m, :FixedCost_per_MW_Electrolyzer, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                w = W[jd, jy]
                ev = value(e_buy[jh, jd, jy])
                gcev = value(gc_e_buy[jh, jd, jy])
                hv = value(h_sell[jh, jd, jy])
                gchv = value(gc_h_sell[jh, jd, jy])
                obj += w * (λ_elec[ih, id_d, iy] * ev + λ_elec_GC[ih, id_d, iy] * gcev
                    + op_cost * hv - λ_H2[ih, id_d, iy] * hv - λ_H2_GC[ih, id_d, iy] * gchv)
            end
            if cap_H2_y !== nothing
                for (iy, jy) in enumerate(JY)
                    obj += F_cap * value(cap_H2_y[jy])
                end
            end
        elseif id in H2_consumers
            # H2 consumer: utility - λ·d; ADMM min is λ·d - U(d)
            dH = var_dict[:H2_d_H][id]
            utility_val = _get(m, :Utility, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                dv = value(dH[jh, jd, jy])
                w = W[jd, jy]
                obj += w * (λ_H2[ih, id_d, iy] * dv - utility_val * dv)
            end
        elseif id in offtaker_green
            h_buy = var_dict[:offtaker_h_buy][id]
            gc_h_buy = var_dict[:offtaker_gc_h_buy][id]
            ep_sell = var_dict[:offtaker_ep_sell][id]
            cap_EP_y = haskey(var_dict, :offtaker_cap_EP_green) && haskey(var_dict[:offtaker_cap_EP_green], id) ? var_dict[:offtaker_cap_EP_green][id] : nothing
            proc_cost = _get(m, :ProcessingCost, 0.0)
            F_cap = _get(m, :FixedCost_per_MW_EP_Out, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                w = W[jd, jy]
                hv = value(h_buy[jh, jd, jy])
                gcv = value(gc_h_buy[jh, jd, jy])
                epv = value(ep_sell[jh, jd, jy])
                obj += w * (λ_H2[ih, id_d, iy] * hv + λ_H2_GC[ih, id_d, iy] * gcv
                    + proc_cost * epv - λ_EP[ih, id_d, iy] * epv)
            end
            if cap_EP_y !== nothing
                for (iy, jy) in enumerate(JY)
                    obj += F_cap * value(cap_EP_y[jy])
                end
            end
        elseif id in offtaker_grey
            ep_sell = var_dict[:offtaker_ep_sell][id]
            gc_h_buy = var_dict[:offtaker_gc_h_buy_G][id]
            MC = _get(m, :MarginalCost, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                w = W[jd, jy]
                epv = value(ep_sell[jh, jd, jy])
                gcv = value(gc_h_buy[jh, jd, jy])
                obj += w * (MC * epv + λ_H2_GC[ih, id_d, iy] * gcv - λ_EP[ih, id_d, iy] * epv)
            end
        elseif id in offtaker_import
            ep_sell = var_dict[:offtaker_ep_sell_import][id]
            imp_cost = _get(m, :ImportCost, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                w = W[jd, jy]
                epv = value(ep_sell[jh, jd, jy])
                obj += w * (imp_cost * epv - λ_EP[ih, id_d, iy] * epv)
            end
        elseif id in agents[:elec_GC_demand]
            dgc = var_dict[:elec_GC_demand_d_GC_E][id]
            A_GC = _get(m, :A_GC, 0.0)
            B_GC = _get(m, :B_GC, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                dv = value(dgc[jh, jd, jy])
                w = W[jd, jy]
                # ADMM: min λ_GC·d − U(d)
                obj += w * (λ_elec_GC[ih, id_d, iy] * dv - (A_GC * dv - B_GC/2 * dv^2))
            end
        elseif id in agents[:EP_demand]
            dep = var_dict[:EP_demand_d_EP][id]
            A_EP = _get(m, :A_EP, 0.0)
            B_EP = _get(m, :B_EP, 0.0)
            for (ih, jh) in enumerate(JH), (id_d, jd) in enumerate(JD), (iy, jy) in enumerate(JY)
                dv = value(dep[jh, jd, jy])
                w = W[jd, jy]
                obj += w * (λ_EP[ih, id_d, iy] * dv - (A_EP * dv - B_EP/2 * dv^2))
            end
        else
            return haskey(agent_welfare, id) ? value(agent_welfare[id]) : 0.0  # fallback
        end
        return obj
    end

    # ── Agent_Summary.csv — Per-agent quantity and ADMM-style objective value ─
    # For every agent, record:
    #   Total_Quantity  — sum of the agent's primary decision variable values
    #                     across all (jh, jd, jy) timesteps.
    #   Objective_Value — ADMM-style objective (cost − revenue) evaluated at the
    #                     optimal quantities and equilibrium prices (duals).
    #                     This matches the market_exposure Agent_Summary format
    #                     and enables direct comparison.
    summary = DataFrame(
        Agent = String[],
        Type = String[],
        Total_Quantity = Float64[],
        Objective_Value = Float64[]
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

        obj_val = _admm_objective(id, ag_type)
        push!(summary, (id, ag_type, qty, obj_val))
    end

    CSV.write(joinpath(results_folder, "Agent_Summary.csv"), summary)

    # --------------------------------------------------------------------------
    # Capacity_Investments_Planner.csv — YEARLY CAPACITY & INVESTMENT (PLANNER)
    # --------------------------------------------------------------------------
    # Mirror of the ADMM Capacity_Investments.csv but for the centralized
    # social planner solution. Uses the planner's cap/investment variables in
    # var_dict built by build_social_planner!.
    cap_rows = Vector{NamedTuple{(:AgentID, :Group, :Type, :YearIndex, :Capacity_MW, :Investment_MW),Tuple{String,String,String,Int,Float64,Float64}}}()

    # Helper: append one row per year for a given agent, if the capacity
    # and investment variables exist in var_dict under the provided keys.
    function _append_cap_rows_planner!(rows,
                                       id::String,
                                       group::String,
                                       atype::String,
                                       cap_dict::Union{Dict{String,Any},Nothing},
                                       inv_dict::Union{Dict{String,Any},Nothing},
                                       JY)
        cap_dict === nothing && return
        !haskey(cap_dict, id) && return
        cap_var = cap_dict[id]
        inv_vec = Float64[]
        if inv_dict !== nothing && haskey(inv_dict, id)
            inv_var = inv_dict[id]
            for jy in JY
                push!(inv_vec, value(inv_var[jy]))
            end
        else
            for _ in JY
                push!(inv_vec, 0.0)
            end
        end
        iy = 0
        for jy in JY
            iy += 1
            cap_val = value(cap_var[jy])
            inv_val = inv_vec[iy]
            push!(rows, (AgentID = String(id),
                         Group = group,
                         Type = atype,
                         YearIndex = iy,
                         Capacity_MW = cap_val,
                         Investment_MW = inv_val))
        end
    end

    # VRES capacities (power_vres agents).
    for id in power_vres
        _append_cap_rows_planner!(cap_rows,
                                  id,
                                  "power",
                                  "VRES",
                                  get(var_dict, :power_cap_VRES, nothing),
                                  get(var_dict, :power_inv_VRES, nothing),
                                  JY)
    end

    # H₂ producer capacities (H2_producers).
    for id in H2_producers
        _append_cap_rows_planner!(cap_rows,
                                  id,
                                  "H2",
                                  "H2Prod",
                                  get(var_dict, :H2_cap_elec, nothing),   # stores H2 capacity in the current formulation
                                  get(var_dict, :H2_inv_elec, nothing),
                                  JY)
    end

    # Green offtaker EP capacities.
    for id in offtaker_green
        _append_cap_rows_planner!(cap_rows,
                                  id,
                                  "offtaker",
                                  "GreenOfftaker",
                                  get(var_dict, :offtaker_cap_EP_green, nothing),
                                  get(var_dict, :offtaker_inv_EP_green, nothing),
                                  JY)
    end

    if !isempty(cap_rows)
        cap_df = DataFrame(cap_rows)
        CSV.write(joinpath(results_folder, "Capacity_Investments_Planner.csv"), cap_df)
    end

    return nothing
end
