# ==============================================================================
# save_results.jl — Write all result and diagnostic CSVs
# ==============================================================================
#
# PURPOSE:
#   Writes into market_exposure_results/: (1) ADMM_Convergence.csv — per-iteration
#   primal and dual residuals for every market. (2) ADMM_Diagnostics.csv — per-
#   iteration rho, mean price, and mean imbalance for every market. (3) One
#   *_Market_History.csv per market (Electricity, Hydrogen, Electricity_GC,
#   H2_GC, End_Product) with iter, rho, price_mean, imb_mean, primal_res, dual_res.
#   (4) Agent_Summary.csv — one row per agent with AgentID, Group, Objective_Value.
#   (5) Market_Prices.csv — per-timestep ADMM λ equilibrium prices.
#
# ARGUMENTS:
#   mdict — Dict of JuMP models; used to extract per-agent objective values.
#   elec_market, H2_market, ... — Market dicts (for reference; most data comes
#     from ADMM_state and results).
#   ADMM_state — Contains ρ, PriceHistory, ImbalanceMean, Imbalances, Residuals.
#   results — Contains λ and agents/markets refs.
#   agents — Dict of agent lists for the summary table.
#
# ==============================================================================

function save_results(mdict::Dict, elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                     H2_GC_market::Dict, ADMM_state::Dict, results::Dict, agents::Dict)
    results_dir = joinpath(@__DIR__, "..", "market_exposure_results")
    isdir(results_dir) || mkdir(results_dir)

    # Determine the number of ADMM iterations actually performed.
    # WHY length(Imbalances): each iteration appends exactly one imbalance value,
    # so the list length equals the iteration count.
    # NOTE: ρ has length n_it+1 because it includes the initial value set *before*
    # the first iteration plus one update per iteration. When building per-iteration
    # DataFrames below we therefore slice ρ as [1:n_it] to align with the other
    # per-iteration vectors (prices, imbalances, residuals) that have length n_it.
    n_it = length(ADMM_state["Imbalances"]["elec"])

    # ── ADMM_Convergence.csv ─────────────────────────────────────────────
    # One row per ADMM iteration with primal and dual residuals for every
    # market. Used to generate convergence plots that show whether the ADMM
    # algorithm is approaching feasibility (primal) and stationarity (dual).
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
        cap_primal     = ADMM_state["Residuals"]["Primal"]["cap"],
        cap_dual       = ADMM_state["Residuals"]["Dual"]["cap"],
    )
    CSV.write(joinpath(results_dir, "ADMM_Convergence.csv"), conv_df)

    # ── ADMM_Diagnostics.csv ──────────────────────────────────────────────
    # One row per ADMM iteration with ρ (penalty parameter), mean price, and
    # mean imbalance for each market. Used to understand how prices and
    # imbalances evolve across iterations and to diagnose oscillation or
    # divergence. ρ is sliced [1:n_it] because its vector is one element
    # longer than the other per-iteration vectors (see n_it note above).
    # PriceHistory may have length n_it+1 (initial + 1 per iteration); slice to
    # [2:end] so we get "price after iteration i" for i=1..n_it.
    ph(mkt) = ADMM_state["PriceHistory"][mkt]
    ph_slice(mkt) = length(ph(mkt)) == n_it + 1 ? ph(mkt)[2:end] : ph(mkt)
    diag_df = DataFrame(
        iter             = 1:n_it,
        elec_rho         = ADMM_state["ρ"]["elec"][1:n_it],
        elec_price_mean  = ph_slice("elec"),
        elec_imb_mean    = ADMM_state["ImbalanceMean"]["elec"],
        H2_rho           = ADMM_state["ρ"]["H2"][1:n_it],
        H2_price_mean    = ph_slice("H2"),
        H2_imb_mean      = ADMM_state["ImbalanceMean"]["H2"],
        elec_GC_rho        = ADMM_state["ρ"]["elec_GC"][1:n_it],
        elec_GC_price_mean = ph_slice("elec_GC"),
        elec_GC_imb_mean   = ADMM_state["ImbalanceMean"]["elec_GC"],
        H2_GC_rho          = ADMM_state["ρ"]["H2_GC"][1:n_it],
        H2_GC_price_mean   = ph_slice("H2_GC"),
        H2_GC_imb_mean     = ADMM_state["ImbalanceMean"]["H2_GC"],
        EP_rho             = ADMM_state["ρ"]["EP"][1:n_it],
        EP_price_mean      = ph_slice("EP"),
        EP_imb_mean        = ADMM_state["ImbalanceMean"]["EP"],
        cap_rho            = ADMM_state["ρ"]["cap"][1:n_it],
    )
    CSV.write(joinpath(results_dir, "ADMM_Diagnostics.csv"), diag_df)

    # ── Per-market history CSVs ──────────────────────────────────────────
    # Same convergence + diagnostic data reorganized into one CSV per market
    # (e.g. Electricity_Market_History.csv). This makes it easier to plot or
    # analyse a single market without filtering the combined tables above.
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
            price_mean = ph_slice(key),
            imb_mean   = ADMM_state["ImbalanceMean"][key],
            primal_res = ADMM_state["Residuals"]["Primal"][key],
            dual_res   = ADMM_state["Residuals"]["Dual"][key],
        )
        CSV.write(joinpath(results_dir, string(name, "_Market_History.csv")), df)
    end

    # --------------------------------------------------------------------------
    # Agent_Summary.csv — GROUP MEMBERSHIP + CLEAN (NO-PENALTY) OBJECTIVE VALUE
    # --------------------------------------------------------------------------
    # One row per agent recording which sector / group it belongs to (power,
    # H2, offtaker, elec_GC_demand) and an objective value that matches the
    # ADMM-style cost − revenue metric used for the social planner benchmark.
    # IMPORTANT: we intentionally EXCLUDE ADMM quadratic penalty terms here so
    # that in the risk-neutral case (γ = 1) the per-agent objectives are
    # comparable between market_exposure and social_planner.

    # Classify agents by sub-type (mirrors build_social_planner.jl).
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

    # Final ADMM prices (used to value cost − revenue, matching the planner).
    λ_elec_final    = results["λ"]["elec"][end]
    λ_H2_final      = results["λ"]["H2"][end]
    λ_elec_GC_final = results["λ"]["elec_GC"][end]
    λ_H2_GC_final   = results["λ"]["H2_GC"][end]
    λ_EP_final      = results["λ"]["EP"][end]

    prices = Dict(
        :λ_elec => λ_elec_final, :λ_H2 => λ_H2_final, :λ_elec_GC => λ_elec_GC_final,
        :λ_H2_GC => λ_H2_GC_final, :λ_EP => λ_EP_final,
    )

    # Helper: compute the agent's economic objective using the shared
    # compute_agent_objective_economic so objectives match social_planner.
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
            quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(vars, :cap_VRES)
                quantities[:cap_VRES] = [value(vars[:cap_VRES][jy]) for jy in JY]
            end
            return compute_agent_objective_economic(:power_vres, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in power_conv
            quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_conv, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in H2_producers
            quantities[:e_in] = [value(vars[:e_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_elec_gc] = [value(vars[:q_elec_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:h2_out] = [value(vars[:h2_out][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            if haskey(vars, :cap_H2_y)
                quantities[:cap_H2_y] = [value(vars[:cap_H2_y][jy]) for jy in JY]
            end
            return compute_agent_objective_economic(:H2_producer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
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

    # Helper: per-agent net market quantities at final iteration (same sign convention
    # as Agent_Quantities_Final below).
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

    # Helper: capacity summary from final ADMM iteration (final-year capacity and
    # total investment over the horizon) for capacity-expanding agents.
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

    # Unified Agent_Summary.csv for market_exposure: quantities, investment, objectives.
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

    # --------------------------------------------------------------------------
    # Agent_Objectives_Per_Timestep.csv — Per-hour prices, quantities, objective contributions
    # --------------------------------------------------------------------------
    # One row per timestep (jh, jd, jy). Column order: Time, jh, jd, jy, W; all prices;
    # all quantities (VRES, CONV, elec demand, elec GC demand, H2 prod, green off, grey off, …);
    # all objective values (same agent order). Same structure as planner for direct comparison.
    all_agent_ids = get(agents, :all, vcat(get.(Ref(agents), [:power, :H2, :offtaker, :elec_GC_demand, :EP_demand], Ref(String[]))...))
    ordered_agents = vcat(
        power_vres, power_conv, power_consumers,
        get(agents, :elec_GC_demand, String[]),
        H2_producers, offtaker_green, offtaker_grey, offtaker_import,
        H2_consumers, get(agents, :EP_demand, String[]),
    )
    ordered_agents = [id for id in ordered_agents if id in all_agent_ids]
    if !isempty(ordered_agents)
        m0 = mdict[ordered_agents[1]]
        JH = collect(m0.ext[:sets][:JH])
        JD = collect(m0.ext[:sets][:JD])
        JY = collect(m0.ext[:sets][:JY])
        λ_elec = results["λ"]["elec"][end]
        λ_H2 = results["λ"]["H2"][end]
        λ_elec_GC = results["λ"]["elec_GC"][end]
        λ_H2_GC = results["λ"]["H2_GC"][end]
        λ_EP = results["λ"]["EP"][end]
        W_mat = m0.ext[:parameters][:W]
        qvars = Dict(:power_vres => [:g], :power_conv => [:g], :power_consumer => [:d], :elec_GC_demand => [:d_gc],
                     :H2_producer => [:e_in, :q_elec_gc, :h2_out, :q_h2gc], :offtaker_green => [:h2_in, :q_h2gc, :ep],
                     :offtaker_grey => [:ep, :q_h2gc], :offtaker_import => [:ep], :H2_consumer => [:d_H], :EP_demand => [:d_EP])
        type_of(id) = id in power_vres ? :power_vres : id in power_conv ? :power_conv : id in power_consumers ? :power_consumer :
                      id in get(agents, :elec_GC_demand, []) ? :elec_GC_demand : id in H2_producers ? :H2_producer :
                      id in offtaker_green ? :offtaker_green : id in offtaker_grey ? :offtaker_grey :
                      id in offtaker_import ? :offtaker_import : id in H2_consumers ? :H2_consumer :
                      id in get(agents, :EP_demand, []) ? :EP_demand : :unknown

        function _admm_agent_data(id)
            m = mdict[id]
            p = m.ext[:parameters]
            vars = m.ext[:variables]
            params = merge(Dict(:W => p[:W]), Dict(k => v for (k, v) in p))
            prices_dict = Dict(:λ_elec => λ_elec, :λ_H2 => λ_H2, :λ_elec_GC => λ_elec_GC,
                              :λ_H2_GC => λ_H2_GC, :λ_EP => λ_EP)
            quantities = Dict{Symbol, Any}()
            agent_type = nothing
            if id in power_consumers
                agent_type = :power_consumer
                quantities[:d] = [value(vars[:d][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in power_vres
                agent_type = :power_vres
                quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                haskey(vars, :cap_VRES) && (quantities[:cap_VRES] = [value(vars[:cap_VRES][jy]) for jy in JY])
            elseif id in power_conv
                agent_type = :power_conv
                quantities[:g] = [value(vars[:g][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in H2_producers
                agent_type = :H2_producer
                quantities[:e_in] = [value(vars[:e_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_elec_gc] = [value(vars[:q_elec_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:h2_out] = [value(vars[:h2_out][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                haskey(vars, :cap_H2_y) && (quantities[:cap_H2_y] = [value(vars[:cap_H2_y][jy]) for jy in JY])
            elseif id in H2_consumers
                agent_type = :H2_consumer
                quantities[:d_H] = [value(vars[:d_H][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in offtaker_green
                agent_type = :offtaker_green
                quantities[:h2_in] = [value(vars[:h2_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                haskey(vars, :cap_EP_y) && (quantities[:cap_EP_y] = [value(vars[:cap_EP_y][jy]) for jy in JY])
            elseif id in offtaker_grey
                agent_type = :offtaker_grey
                quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in offtaker_import
                agent_type = :offtaker_import
                quantities[:ep] = [value(vars[:ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in get(agents, :elec_GC_demand, String[])
                agent_type = :elec_GC_demand
                quantities[:d_gc] = [value(vars[:d_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif id in get(agents, :EP_demand, String[])
                agent_type = :EP_demand
                quantities[:d_EP] = [value(vars[:q_ep][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            end
            contrib = agent_type !== nothing ? compute_agent_objective_contributions(agent_type, quantities, prices_dict, params; JH=JH, JD=JD, JY=JY) : nothing
            return agent_type, quantities, contrib
        end

        col_order = [:Time, :jh, :jd, :jy, :W, :Elec_Price, :H2_Price, :Elec_GC_Price, :H2_GC_Price, :EP_Price]
        for id in ordered_agents
            t = type_of(id)
            for v in get(qvars, t, [])
                push!(col_order, Symbol(id * "_" * string(v)))
            end
        end
        for id in ordered_agents
            push!(col_order, Symbol(id * "_obj"))
        end

        admm_agent_data = Dict(id => _admm_agent_data(id) for id in ordered_agents)
        ts_rows = []
        t_idx = 1
        for jy in JY, jd in JD, jh in JH
            w = W_mat[jd, jy]
            row = Dict(:Time => t_idx, :jh => jh, :jd => jd, :jy => jy, :W => w,
                       :Elec_Price => λ_elec[jh, jd, jy], :H2_Price => λ_H2[jh, jd, jy],
                       :Elec_GC_Price => λ_elec_GC[jh, jd, jy], :H2_GC_Price => λ_H2_GC[jh, jd, jy],
                       :EP_Price => λ_EP[jh, jd, jy])
            for id in ordered_agents
                atype, q, c = admm_agent_data[id]
                row[Symbol(id * "_obj")] = c !== nothing ? c[jh, jd, jy] : 0.0
                if q !== nothing
                    for (k, v) in q
                        if v isa AbstractArray && ndims(v) == 3
                            row[Symbol(id * "_" * string(k))] = v[jh, jd, jy]
                        end
                    end
                end
            end
            push!(ts_rows, row)
            t_idx += 1
        end
        ts_df = DataFrame(ts_rows)
        present = [c for c in col_order if c in propertynames(ts_df)]
        ts_df = select(ts_df, present)
        CSV.write(joinpath(results_dir, "Agent_Objectives_Per_Timestep.csv"), ts_df)
    end

    # --------------------------------------------------------------------------
    # Agent_Quantities_Final.csv — PER-AGENT NET QUANTITIES AT FINAL ITERATION
    # --------------------------------------------------------------------------
    # Builds a compact summary showing each agent's total energy traded in
    # every market at the last ADMM iteration.
    #
    # Net position sign convention (consistent with the rest of the model):
    #   +  = supply (selling into the market)
    #   −  = demand (buying from the market)
    #
    # _total_last helper: takes the history dict for a given market, selects
    # the 3D array from the *last* ADMM iteration (arr_dict[id][end]), and
    # sums over all (jh, jd, jy) entries to collapse it into a single scalar.
    # This scalar is the total energy traded by that agent across the full
    # modeled year (all hours × representative days × scenario years).

    # --------------------------------------------------------------------------
    # Offtaker_GC_Diagnostics.csv — GREEN-CERTIFICATE COMPLIANCE PER OFFTAKER
    # --------------------------------------------------------------------------
    # For each offtaker agent, exports total end-product (EP) output, total H₂
    # consumed, total H₂ green certificates (GCs) consumed, the resulting GC
    # share, the regulatory mandate (γ_GC), and the slack (share − mandate).
    #
    # Sign convention: offtakers *buy* H₂ and H₂ GCs, so their net positions
    # in those markets are negative. We negate h2_net and h2gc_net below to
    # obtain the positive quantity consumed, which is more intuitive for
    # compliance reporting.
    #
    # gc_share = H₂ GCs consumed / EP produced — fraction of output backed by
    #            green certificates.
    # gc_slack = gc_share − γ_GC mandate. Positive → compliant; negative → short.

    if haskey(agents, :offtaker)
        off_ids    = agents[:offtaker]
        off_agent  = String[]
        off_type   = String[]
        ep_total   = Float64[]
        h2_in_tot  = Float64[]
        h2_gc_tot  = Float64[]
        gc_share   = Float64[]
        gc_mandate = Float64[]
        gc_slack   = Float64[]

        for id in off_ids
            m = mdict[id]
            t = String(get(m.ext[:parameters], :Type, ""))
            γ = get(m.ext[:parameters], :gamma_GC, 0.42)

            ep_list   = results["EP"][id]
            h2_list   = results["h2"][id]
            h2gc_list = results["H2_GC"][id]

            ep_sum   = isempty(ep_list)   ? 0.0 : sum(ep_list[end])
            h2_net   = isempty(h2_list)   ? 0.0 : sum(h2_list[end])      # < 0 for offtakers (they buy H₂)
            h2gc_net = isempty(h2gc_list) ? 0.0 : sum(h2gc_list[end])    # < 0 for offtakers (they buy H₂ GCs)

            # Negate to convert negative net positions into positive consumed
            # quantities, which are easier to interpret in a compliance context.
            h2_in_sum  = -h2_net       # total H₂ consumed (positive)
            h2_gc_sum  = -h2gc_net     # total H₂ GCs consumed (positive)
            # gc_share: fraction of EP output backed by green H₂ certificates
            share      = (ep_sum > 0 && h2_gc_sum > 0) ? h2_gc_sum / ep_sum : 0.0
            # slack > 0 means the offtaker exceeds its green mandate
            slack      = share - γ

            push!(off_agent,  String(id))
            push!(off_type,   t)
            push!(ep_total,   ep_sum)
            push!(h2_in_tot,  h2_in_sum)
            push!(h2_gc_tot,  h2_gc_sum)
            push!(gc_share,   share)
            push!(gc_mandate, γ)
            push!(gc_slack,   slack)
        end

        off_df = DataFrame(
            AgentID      = off_agent,
            Type         = off_type,
            EP_total     = ep_total,
            H2_in_total  = h2_in_tot,
            H2_GC_total  = h2_gc_tot,
            GC_share     = gc_share,
            GC_mandate   = gc_mandate,
            GC_slack     = gc_slack,
        )
        CSV.write(joinpath(results_dir, "Offtaker_GC_Diagnostics.csv"), off_df)
    end

    # --------------------------------------------------------------------------
    # H2_Producer_Diagnostics.csv — ELECTROLYZER GREEN-FRACTION SUMMARY
    # --------------------------------------------------------------------------
    # For each hydrogen-producing agent, summarizes total H₂ production,
    # total H₂ green certificates (GCs) issued, and the ratio GC/H₂.
    # The ratio indicates what fraction of this producer's hydrogen output
    # is certified green (i.e. backed by renewable electricity GCs).

    if haskey(agents, :H2)
        h2_ids        = agents[:H2]
        el_agent      = String[]
        h2_prod_total = Float64[]
        h2_gc_total   = Float64[]
        gc_to_h2      = Float64[]

        for id in h2_ids
            h2_list   = results["h2"][id]
            h2gc_list = results["H2_GC"][id]

            h2_sum   = isempty(h2_list)   ? 0.0 : sum(h2_list[end])      # > 0 for producers (they sell H₂)
            h2gc_sum = isempty(h2gc_list) ? 0.0 : sum(h2gc_list[end])    # > 0 when issuing H₂ GCs
            # Ratio of H₂ GCs issued to total H₂ produced: what fraction of
            # this producer's output is certified as green hydrogen.
            ratio    = h2_sum != 0.0 ? h2gc_sum / h2_sum : 0.0

            push!(el_agent,      String(id))
            push!(h2_prod_total, h2_sum)
            push!(h2_gc_total,   h2gc_sum)
            push!(gc_to_h2,      ratio)
        end

        el_df = DataFrame(
            AgentID       = el_agent,
            H2_total      = h2_prod_total,
            H2_GC_total   = h2_gc_total,
            GC_per_H2     = gc_to_h2,
        )
        CSV.write(joinpath(results_dir, "H2_Producer_Diagnostics.csv"), el_df)
    end

    # --------------------------------------------------------------------------
    # Market_Prices.csv — Equilibrium prices from ADMM Lagrange multipliers
    # --------------------------------------------------------------------------
    # The ADMM λ values are the standard equilibrium price output of the
    # distributed market-clearing algorithm. At convergence (primal and dual
    # residuals → 0), they equal the true market-clearing prices. These are
    # the per-timestep prices that agents respond to, and should converge
    # toward the social planner dual prices.
    #
    # Format matches the social planner Market_Prices.csv for direct comparison.

    λ_elec    = results["λ"]["elec"][end]
    λ_H2      = results["λ"]["H2"][end]
    λ_elec_GC = results["λ"]["elec_GC"][end]
    λ_H2_GC   = results["λ"]["H2_GC"][end]
    λ_EP      = results["λ"]["EP"][end]

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
        ))
        t_index += 1
    end
    prices_df = DataFrame(prices_rows)
    CSV.write(joinpath(results_dir, "Market_Prices.csv"), prices_df)

    # Print equilibrium prices (ADMM λ) to the output log
    println()
    println("Equilibrium prices (ADMM λ, saved to Market_Prices.csv):")
    println("  Electricity     mean = ", round(mean(λ_elec), digits=6))
    println("  Hydrogen        mean = ", round(mean(λ_H2), digits=6))
    println("  Electricity_GC  mean = ", round(mean(λ_elec_GC), digits=6))
    println("  H2_GC           mean = ", round(mean(λ_H2_GC), digits=6))
    println("  End_Product     mean = ", round(mean(λ_EP), digits=6))

    # --------------------------------------------------------------------------
    # Comparison with social planner benchmark (if available)
    # --------------------------------------------------------------------------
    sp_path = joinpath(@__DIR__, "..", "social_planner_results", "Market_Prices.csv")
    if isfile(sp_path)
        sp_df = CSV.read(sp_path, DataFrame)
        println()
        println("Comparison with social planner benchmark:")
        println("  Market          | Social planner |   ADMM λ mean")
        println("  " * repeat("-", 52))
        @printf("  %-14s | %14.6f | %14.6f\n", "Electricity",    mean(sp_df.Elec_Price),    mean(λ_elec))
        @printf("  %-14s | %14.6f | %14.6f\n", "Hydrogen",       mean(sp_df.H2_Price),      mean(λ_H2))
        @printf("  %-14s | %14.6f | %14.6f\n", "Electricity_GC", mean(sp_df.Elec_GC_Price), mean(λ_elec_GC))
        @printf("  %-14s | %14.6f | %14.6f\n", "H2_GC",          mean(sp_df.H2_GC_Price),   mean(λ_H2_GC))
        @printf("  %-14s | %14.6f | %14.6f\n", "End_Product",    mean(sp_df.EP_Price),      mean(λ_EP))
    end

    return nothing
end
