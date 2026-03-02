# ==============================================================================
# save_results_contracts.jl — Write results for market_exposure_contracts
# ==============================================================================
#
# PURPOSE:
#   Writes to market_exposure_contracts_results/. Keeps the same major ADMM CSVs
#   as market_exposure (ADMM_Convergence, ADMM_Diagnostics, 5× Market_History,
#   Agent_Summary, Market_Prices) plus contract columns where relevant.
#   Adds focal contract outputs:
#   - Contracts.csv: capacity_contracted_MW, energy_transferred_MWh, contract_price_EUR_per_MWh
#   - Green_Agents_Detail.csv: per-agent breakdown (VRES, electrolyzer) — total capacity,
#     contracted vs pool energy, and prices
#
# ARGUMENTS:
#   mdict, elec_market, H2_market, elec_GC_market, H2_GC_market — Same as save_results.
#   contract_market — Contract market dict (initial_price, rho_initial).
#   ADMM_state — ADMM state with contract/contract_cap keys.
#   results — Results with contract, contract_cap, λ["contract"].
#   agents — Agent lists including agents[:contract_market].
#
# ==============================================================================

function save_results_contracts!(mdict::Dict, elec_market::Dict, H2_market::Dict,
                                 elec_GC_market::Dict, H2_GC_market::Dict,
                                 contract_market::Dict, ADMM_state::Dict, results::Dict, agents::Dict)
    results_dir = joinpath(@__DIR__, "..", "market_exposure_contracts_results")
    isdir(results_dir) || mkdir(results_dir)

    n_it = length(ADMM_state["Imbalances"]["elec"])

    # ── ADMM_Convergence.csv (with contract columns) ─────────────────────────
    conv_df = DataFrame(
        iter          = 1:n_it,
        elec_primal   = ADMM_state["Residuals"]["Primal"]["elec"],
        elec_dual     = ADMM_state["Residuals"]["Dual"]["elec"],
        H2_primal     = ADMM_state["Residuals"]["Primal"]["H2"],
        H2_dual       = ADMM_state["Residuals"]["Dual"]["H2"],
        elec_GC_primal = ADMM_state["Residuals"]["Primal"]["elec_GC"],
        elec_GC_dual   = ADMM_state["Residuals"]["Dual"]["elec_GC"],
        H2_GC_primal   = ADMM_state["Residuals"]["Primal"]["H2_GC"],
        H2_GC_dual     = ADMM_state["Residuals"]["Dual"]["H2_GC"],
        EP_primal      = ADMM_state["Residuals"]["Primal"]["EP"],
        EP_dual        = ADMM_state["Residuals"]["Dual"]["EP"],
        contract_primal      = ADMM_state["Residuals"]["Primal"]["contract"],
        contract_dual        = ADMM_state["Residuals"]["Dual"]["contract"],
        contract_cap_primal  = ADMM_state["Residuals"]["Primal"]["contract_cap"],
        contract_cap_dual    = ADMM_state["Residuals"]["Dual"]["contract_cap"],
    )
    CSV.write(joinpath(results_dir, "ADMM_Convergence.csv"), conv_df)

    # ── ADMM_Diagnostics.csv (with contract columns) ───────────────────────
    diag_df = DataFrame(
        iter             = 1:n_it,
        elec_rho         = ADMM_state["ρ"]["elec"][1:n_it],
        elec_price_mean  = ADMM_state["PriceHistory"]["elec"],
        elec_imb_mean    = ADMM_state["ImbalanceMean"]["elec"],
        H2_rho           = ADMM_state["ρ"]["H2"][1:n_it],
        H2_price_mean    = ADMM_state["PriceHistory"]["H2"],
        H2_imb_mean      = ADMM_state["ImbalanceMean"]["H2"],
        elec_GC_rho        = ADMM_state["ρ"]["elec_GC"][1:n_it],
        elec_GC_price_mean = ADMM_state["PriceHistory"]["elec_GC"],
        elec_GC_imb_mean   = ADMM_state["ImbalanceMean"]["elec_GC"],
        H2_GC_rho          = ADMM_state["ρ"]["H2_GC"][1:n_it],
        H2_GC_price_mean   = ADMM_state["PriceHistory"]["H2_GC"],
        H2_GC_imb_mean     = ADMM_state["ImbalanceMean"]["H2_GC"],
        EP_rho             = ADMM_state["ρ"]["EP"][1:n_it],
        EP_price_mean      = ADMM_state["PriceHistory"]["EP"],
        EP_imb_mean        = ADMM_state["ImbalanceMean"]["EP"],
        contract_rho         = ADMM_state["ρ"]["contract"][1:n_it],
        contract_price_mean = ADMM_state["PriceHistory"]["contract"],
        contract_imb_mean   = ADMM_state["ImbalanceMean"]["contract"],
        contract_cap_rho         = ADMM_state["ρ"]["contract_cap"][1:n_it],
        contract_cap_imb_mean   = ADMM_state["ImbalanceMean"]["contract_cap"],
    )
    CSV.write(joinpath(results_dir, "ADMM_Diagnostics.csv"), diag_df)

    # ── Per-market history (5 base markets only; contract focal info in Contracts.csv) ─
    markets = Dict(
        "elec"    => "Electricity",
        "H2"      => "Hydrogen",
        "elec_GC" => "Electricity_GC",
        "H2_GC"   => "H2_GC",
        "EP"      => "End_Product",
    )
    for (key, name) in markets
        df = DataFrame(
            iter       = 1:n_it,
            rho        = ADMM_state["ρ"][key][1:n_it],
            price_mean = ADMM_state["PriceHistory"][key],
            imb_mean   = ADMM_state["ImbalanceMean"][key],
            primal_res = ADMM_state["Residuals"]["Primal"][key],
            dual_res   = ADMM_state["Residuals"]["Dual"][key],
        )
        CSV.write(joinpath(results_dir, string(name, "_Market_History.csv")), df)
    end

    # ── Contracts.csv — Single focal output: capacity contracted, energy transferred, price ─
    # ── Green_Agents_Detail.csv — VRES and electrolyzer breakdown: capacity, contract vs pool ─
    contract_agents = get(agents, :contract_market, String[])
    λ_contract_final = isempty(contract_agents) ? nothing : results["λ"]["contract"][end]
    λ_elec_final = isempty(contract_agents) ? nothing : results["λ"]["elec"][end]
    λ_contract_mean = λ_contract_final === nothing ? 0.0 : mean(λ_contract_final)
    λ_elec_mean = λ_elec_final === nothing ? 0.0 : mean(λ_elec_final)

    vres_contract = [id for id in contract_agents if String(mdict[id].ext[:parameters][:Type]) == "VRES"]
    if !isempty(vres_contract)
        vres_id = first(vres_contract)
        cap_list = results["contract_cap"][vres_id]
        g_list = results["contract"][vres_id]
        cap_MW = isempty(cap_list) ? 0.0 : abs(cap_list[end])
        energy_MWh = isempty(g_list) ? 0.0 : abs(sum(g_list[end]))
        contracts_df = DataFrame(
            capacity_contracted_MW = [cap_MW],
            energy_transferred_MWh = [energy_MWh],
            contract_price_EUR_per_MWh = [λ_contract_mean],
        )
        CSV.write(joinpath(results_dir, "Contracts.csv"), contracts_df)
    end

    # Green agents detail: VRES (total cap, contracted share, pool share) and electrolyzer (same)
    if !isempty(contract_agents)
        detail_ids = String[]
        detail_types = String[]
        total_capacity = Float64[]
        contracted_capacity_MW = Float64[]
        energy_from_contract_MWh = Float64[]
        energy_from_pool_MWh = Float64[]
        contract_price_EUR_per_MWh = Float64[]
        electricity_price_EUR_per_MWh = Float64[]

        for id in contract_agents
            m = mdict[id]
            atype = String(get(m.ext[:parameters], :Type, ""))
            push!(detail_ids, id)
            push!(detail_types, atype == "VRES" ? "VRES" : "Electrolyzer")

            cap_tot = 0.0
            if atype == "VRES"
                ch = get(results["Cap_VRES"], id, [])
                cap_tot = (isempty(ch) || isempty(ch[end])) ? 0.0 : ch[end][end]
            else
                ch = get(results["Cap_Elec_H2"], id, [])
                cap_tot = (isempty(ch) || isempty(ch[end])) ? 0.0 : ch[end][end]
            end
            push!(total_capacity, cap_tot)

            cc = get(results["contract_cap"], id, [])
            cap_contracted = isempty(cc) ? 0.0 : abs(cc[end])
            push!(contracted_capacity_MW, cap_contracted)

            gc = get(results["contract"], id, [])
            g_contract_sum = isempty(gc) ? 0.0 : abs(sum(gc[end]))
            push!(energy_from_contract_MWh, g_contract_sum)

            g_elec = get(results["g"], id, [])
            g_pool_sum = isempty(g_elec) ? 0.0 : (atype == "VRES" ? sum(g_elec[end]) : -sum(g_elec[end]))
            push!(energy_from_pool_MWh, max(0.0, g_pool_sum))

            push!(contract_price_EUR_per_MWh, λ_contract_mean)
            push!(electricity_price_EUR_per_MWh, λ_elec_mean)
        end

        # For VRES: total_capacity = cap_VRES (MW). For Electrolyzer: total_capacity = cap_H2_y (MW_H2 output).
        detail_df = DataFrame(
            AgentID = detail_ids,
            Type = detail_types,
            total_capacity_MW = total_capacity,
            contracted_capacity_MW = contracted_capacity_MW,
            energy_from_contract_MWh = energy_from_contract_MWh,
            energy_from_pool_MWh = energy_from_pool_MWh,
            contract_price_EUR_per_MWh = contract_price_EUR_per_MWh,
            electricity_price_EUR_per_MWh = electricity_price_EUR_per_MWh,
        )
        CSV.write(joinpath(results_dir, "Green_Agents_Detail.csv"), detail_df)
    end

    # ── Agent classification (mirrors save_results) ─────────────────────────
    power_consumers = String[]
    power_vres = String[]
    power_conv = String[]
    for id in get(agents, :power, String[])
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
    for id in get(agents, :H2, String[])
        m = mdict[id]
        p = m.ext[:parameters]
        if haskey(p, :Capacity_Electrolyzer) || (haskey(p, :E_bar) && haskey(p, :H_bar))
            push!(H2_producers, id)
        elseif haskey(p, :D_H_bar)
            push!(H2_consumers, id)
        end
    end

    offtaker_green = String[]
    offtaker_grey = String[]
    offtaker_import = String[]
    for id in get(agents, :offtaker, String[])
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

    contract_market_agents = get(agents, :contract_market, String[])

    # ── Prices (including contract) ───────────────────────────────────────
    λ_elec_final    = results["λ"]["elec"][end]
    λ_H2_final      = results["λ"]["H2"][end]
    λ_elec_GC_final = results["λ"]["elec_GC"][end]
    λ_H2_GC_final   = results["λ"]["H2_GC"][end]
    λ_EP_final      = results["λ"]["EP"][end]
    λ_contract_final = results["λ"]["contract"][end]

    prices = Dict(
        :λ_elec => λ_elec_final, :λ_H2 => λ_H2_final, :λ_elec_GC => λ_elec_GC_final,
        :λ_H2_GC => λ_H2_GC_final, :λ_EP => λ_EP_final,
        :λ_contract => λ_contract_final,
    )

    # ── _admm_objective_economic (contract-aware) ───────────────────────────
    function _admm_objective_economic(id::String)
        m = mdict[id]
        p = m.ext[:parameters]
        sets = m.ext[:sets]
        vars = m.ext[:variables]
        W = p[:W]
        JH = sets[:JH]
        JD = sets[:JD]
        JY = sets[:JY]
        params = merge(Dict(:W => W), Dict(k => v for (k, v) in p))
        quantities = Dict{Symbol, Any}()

        if id in power_consumers
            quantities[:d] = [value(vars[:d][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_consumer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in power_vres
            if id in contract_market_agents && haskey(vars, :g_EOM)
                quantities[:g_EOM] = [value(vars[:g_EOM][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:g_contract] = [value(vars[:g_contract][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:contract_cap] = value(vars[:contract_cap])
                quantities[:cap_VRES] = [value(vars[:cap_VRES][jy]) for jy in JY]
                return compute_agent_objective_economic(:power_vres_contracts, quantities, prices, params; JH=JH, JD=JD, JY=JY)
            else
                quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                if haskey(vars, :cap_VRES)
                    quantities[:cap_VRES] = [value(vars[:cap_VRES][jy]) for jy in JY]
                end
                return compute_agent_objective_economic(:power_vres, quantities, prices, params; JH=JH, JD=JD, JY=JY)
            end
        elseif id in power_conv
            quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_conv, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in H2_producers
            if id in contract_market_agents && haskey(vars, :e_in_pool)
                quantities[:e_in_pool] = [value(vars[:e_in_pool][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:g_contract] = [value(vars[:g_contract][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_elec_gc] = [value(vars[:q_elec_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:h2_out] = [value(vars[:h2_out][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:contract_cap] = value(vars[:contract_cap])
                quantities[:cap_H2_y] = [value(vars[:cap_H2_y][jy]) for jy in JY]
                return compute_agent_objective_economic(:H2_producer_contracts, quantities, prices, params; JH=JH, JD=JD, JY=JY)
            else
                quantities[:e_in] = [value(vars[:e_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_elec_gc] = [value(vars[:q_elec_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:h2_out] = [value(vars[:h2_out][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                if haskey(vars, :cap_H2_y)
                    quantities[:cap_H2_y] = [value(vars[:cap_H2_y][jy]) for jy in JY]
                end
                return compute_agent_objective_economic(:H2_producer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
            end
        elseif id in H2_consumers
            quantities[:d_H] = [value(vars[:d_H][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:H2_consumer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_green
            quantities[:h2_in] = [value(vars[:h2_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(vars, :cap_EP_y)
                quantities[:cap_EP_y] = [value(vars[:cap_EP_y][jy]) for jy in JY]
            end
            return compute_agent_objective_economic(:offtaker_green, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_grey
            quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:offtaker_grey, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in offtaker_import
            quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:offtaker_import, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in get(agents, :elec_GC_demand, String[])
            quantities[:d_gc] = [value(vars[:d_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:elec_GC_demand, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        else
            return 0.0
        end
    end

    # ── _final_net_quantities (base markets only; contract info in Contracts.csv) ─
    function _final_net_quantities(id::String)
        g_list   = results["g"][id]
        h2_list  = results["h2"][id]
        egc_list = results["elec_GC"][id]
        hgc_list = results["H2_GC"][id]
        ep_list  = results["EP"][id]
        elec_q    = isempty(g_list)   ? 0.0 : sum(g_list[end])
        H2_q      = isempty(h2_list)  ? 0.0 : sum(h2_list[end])
        elec_GC_q = isempty(egc_list) ? 0.0 : sum(egc_list[end])
        H2_GC_q   = isempty(hgc_list) ? 0.0 : sum(hgc_list[end])
        EP_q      = isempty(ep_list)  ? 0.0 : sum(ep_list[end])
        return elec_q, H2_q, elec_GC_q, H2_GC_q, EP_q
    end

    # ── _capacity_summary_admm (same as market_exposure) ─────────────────────
    function _capacity_summary_admm(id::String)
        cap_final = 0.0
        inv_total = 0.0
        if id in power_vres
            cap_hist = results["Cap_VRES"][id]
            inv_hist = results["Inv_VRES"][id]
            if !isempty(cap_hist)
                cap_vec = cap_hist[end]
                cap_final = isempty(cap_vec) ? 0.0 : cap_vec[end]
            end
            if !isempty(inv_hist)
                inv_vec = inv_hist[end]
                inv_total = sum(inv_vec)
            end
        elseif id in H2_producers
            cap_hist = results["Cap_Elec_H2"][id]
            inv_hist = results["Inv_Elec_H2"][id]
            if !isempty(cap_hist)
                cap_vec = cap_hist[end]
                cap_final = isempty(cap_vec) ? 0.0 : cap_vec[end]
            end
            if !isempty(inv_hist)
                inv_vec = inv_hist[end]
                inv_total = sum(inv_vec)
            end
        elseif id in offtaker_green
            cap_hist = results["Cap_EP_Green"][id]
            inv_hist = results["Inv_EP_Green"][id]
            if !isempty(cap_hist)
                cap_vec = cap_hist[end]
                cap_final = isempty(cap_vec) ? 0.0 : cap_vec[end]
            end
            if !isempty(inv_hist)
                inv_vec = inv_hist[end]
                inv_total = sum(inv_vec)
            end
        end
        return cap_final, inv_total
    end

    # ── Agent_Summary.csv (same structure as market_exposure; no contract columns) ─
    agent_ids_sum = String[]
    group_sum     = String[]
    type_sum      = String[]
    elec_sum      = Float64[]
    H2_sum        = Float64[]
    elec_GC_sum   = Float64[]
    H2_GC_sum     = Float64[]
    EP_sum        = Float64[]
    cap_final_sum = Float64[]
    inv_total_sum = Float64[]
    obj_sum       = Float64[]

    for k in (:power, :H2, :offtaker, :elec_GC_demand)
        haskey(agents, k) || continue
        for id in agents[k]
            push!(agent_ids_sum, String(id))
            push!(group_sum, String(k))
            type_label = if id in power_consumers
                "PowerCons"
            elseif id in power_vres || id in power_conv
                "PowerGen"
            elseif id in H2_producers
                "H2Prod"
            elseif id in H2_consumers
                "H2Cons"
            elseif id in offtaker_green || id in offtaker_grey || id in offtaker_import
                "Offtaker"
            elseif k == :elec_GC_demand
                "GC_Demand"
            else
                "Unknown"
            end
            push!(type_sum, type_label)
            e_q, h_q, egc_q, hgc_q, ep_q = _final_net_quantities(String(id))
            push!(elec_sum, e_q)
            push!(H2_sum, h_q)
            push!(elec_GC_sum, egc_q)
            push!(H2_GC_sum, hgc_q)
            push!(EP_sum, ep_q)
            cap_f, inv_t = _capacity_summary_admm(String(id))
            push!(cap_final_sum, cap_f)
            push!(inv_total_sum, inv_t)
            push!(obj_sum, _admm_objective_economic(String(id)))
        end
    end

    agents_df = DataFrame(
        AgentID = agent_ids_sum,
        Group = group_sum,
        Type = type_sum,
        elec_net_sum = elec_sum,
        H2_net_sum = H2_sum,
        elec_GC_net_sum = elec_GC_sum,
        H2_GC_net_sum = H2_GC_sum,
        EP_net_sum = EP_sum,
        Capacity_Final_MW = cap_final_sum,
        Investment_Total_MW = inv_total_sum,
        Objective_Value = obj_sum,
    )
    CSV.write(joinpath(results_dir, "Agent_Summary.csv"), agents_df)

    # ── Market_Prices.csv (with Contract_Price) ──────────────────────────────
    λ_elec    = results["λ"]["elec"][end]
    λ_H2      = results["λ"]["H2"][end]
    λ_elec_GC = results["λ"]["elec_GC"][end]
    λ_H2_GC   = results["λ"]["H2_GC"][end]
    λ_EP      = results["λ"]["EP"][end]
    λ_contract = results["λ"]["contract"][end]

    shp = size(λ_elec)
    n_ts, n_rd, n_yr = shp[1], shp[2], shp[3]

    prices_rows = []
    t_index = 1
    for jy in 1:n_yr, jd in 1:n_rd, jh in 1:n_ts
        push!(prices_rows, (
            Time = t_index,
            Elec_Price = λ_elec[jh, jd, jy],
            H2_Price = λ_H2[jh, jd, jy],
            Elec_GC_Price = λ_elec_GC[jh, jd, jy],
            H2_GC_Price = λ_H2_GC[jh, jd, jy],
            EP_Price = λ_EP[jh, jd, jy],
            Contract_Price = λ_contract[jh, jd, jy],
        ))
        t_index += 1
    end
    prices_df = DataFrame(prices_rows)
    CSV.write(joinpath(results_dir, "Market_Prices.csv"), prices_df)

    println()
    println("Equilibrium prices (ADMM λ, saved to Market_Prices.csv):")
    println("  Electricity     mean = ", round(mean(λ_elec), digits=6))
    println("  Hydrogen        mean = ", round(mean(λ_H2), digits=6))
    println("  Electricity_GC  mean = ", round(mean(λ_elec_GC), digits=6))
    println("  H2_GC           mean = ", round(mean(λ_H2_GC), digits=6))
    println("  End_Product     mean = ", round(mean(λ_EP), digits=6))
    println("  Contract        mean = ", round(mean(λ_contract), digits=6))

    return nothing
end
