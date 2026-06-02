# ==============================================================================
# save_results_contracts.jl — Write results for market_exposure_contracts
# ==============================================================================
#
# PURPOSE:
#   Writes to market_exposure_contracts_results/. Keeps the same major ADMM CSVs
#   as market_exposure (ADMM_Convergence, ADMM_Diagnostics, 5× Market_History,
#   Agent_Summary, Market_Prices) plus contract columns where relevant.
#   Adds focal contract outputs:
#   - PPAs.csv: per-VRES contracted capacity/energy/price
#   - HPAs.csv: per-GreenProducer contracted capacity/energy/price
#   - Green_Agents_Detail.csv: detailed PPA breakdown for VRES + GreenProducer
#
# ARGUMENTS:
#   mdict, elec_market, H2_market, elec_GC_market, H2_GC_market — Same as save_results.
#   ppa_market, hpa_market — Contract market dicts.
#   ADMM_state — ADMM state with ppa/hpa sub-dicts.
#   results — Results including ppa/hpa quantities and λ_ppa/λ_hpa.
#   agents — Agent lists including contract participants.
#
# ==============================================================================

include(joinpath(@__DIR__, "print_run_summary.jl"))
include(joinpath(@__DIR__, "compute_social_risk_metrics.jl"))

function save_results_contracts!(mdict::Dict, elec_market::Dict, H2_market::Dict,
                                 elec_GC_market::Dict, H2_GC_market::Dict,
                                 ppa_market::Dict, hpa_market::Dict, ADMM_state::Dict, results::Dict, agents::Dict)
    results_dir = joinpath(@__DIR__, "..", "market_exposure_contracts_results")
    isdir(results_dir) || mkdir(results_dir)

    n_it = length(ADMM_state["Imbalances"]["elec"])
    ppa_vres = get(ppa_market, "ppa_vres", String[])
    hpa_h2 = get(hpa_market, "hpa_h2", String[])
    C = get(ADMM_state, "ppa", Dict())
    C_hpa = get(ADMM_state, "hpa", Dict())

    # Ensure deterministic column order with iteration/time first.
    function _move_first!(df::DataFrame, col::Symbol)
        cols = Symbol.(names(df))
        if col in cols
            ordered = vcat([col], [c for c in cols if c != col])
            select!(df, ordered)
        end
        return df
    end

    # PriceHistory may have length n_it+1 (initial + 1 per iteration); slice to [2:end]
    # so all diagnostic columns have length n_it (same as save_results.jl).
    ph(mkt) = ADMM_state["PriceHistory"][mkt]
    ph_slice(mkt) = length(ph(mkt)) == n_it + 1 ? ph(mkt)[2:end] : ph(mkt)

    # ── ADMM_Convergence.csv (with per-VRES contract columns) ─────────────────
    # In addition to the 5 base markets and the PPA/HPA contract columns, we
    # now also expose aggregate (Σ over agents) and PER-AGENT capacity
    # residuals from the new equality-split formulation; see
    # DOCUMENTATION.md §5.4.
    conv_cols = Dict(
        :iter => 1:n_it,
        :elec_primal => ADMM_state["Residuals"]["Primal"]["elec"],
        :elec_dual => ADMM_state["Residuals"]["Dual"]["elec"],
        :H2_primal => ADMM_state["Residuals"]["Primal"]["H2"],
        :H2_dual => ADMM_state["Residuals"]["Dual"]["H2"],
        :elec_GC_primal => ADMM_state["Residuals"]["Primal"]["elec_GC"],
        :elec_GC_dual => ADMM_state["Residuals"]["Dual"]["elec_GC"],
        :H2_GC_primal => ADMM_state["Residuals"]["Primal"]["H2_GC"],
        :H2_GC_dual => ADMM_state["Residuals"]["Dual"]["H2_GC"],
        :EP_primal => ADMM_state["Residuals"]["Primal"]["EP"],
        :EP_dual => ADMM_state["Residuals"]["Dual"]["EP"],
        :cap_primal => ADMM_state["Residuals"]["Primal"]["cap"],
        :cap_dual   => ADMM_state["Residuals"]["Dual"]["cap"],
    )
    cap_state_save = get(ADMM_state, "Capacity", Dict())
    cap_agents_save = get(cap_state_save, "agents", String[])
    for m in cap_agents_save
        rp = get(get(cap_state_save, "Primal", Dict()), m, Float64[])
        rd = get(get(cap_state_save, "Dual",   Dict()), m, Float64[])
        rp_aligned = length(rp) >= n_it ? rp[1:n_it] : vcat(rp, fill(NaN, n_it - length(rp)))
        rd_aligned = length(rd) >= n_it ? rd[1:n_it] : vcat(rd, fill(NaN, n_it - length(rd)))
        conv_cols[Symbol("cap_primal_$(m)")] = rp_aligned
        conv_cols[Symbol("cap_dual_$(m)")]   = rd_aligned
    end
    C = get(ADMM_state, "ppa", Dict())
    for vres_id in ppa_vres
        haskey(C, "Primal") || break
        conv_cols[Symbol("ppa_$(vres_id)_primal")] = C["Primal"][vres_id]
        conv_cols[Symbol("ppa_$(vres_id)_dual")] = C["Dual"][vres_id]
        conv_cols[Symbol("ppa_cap_$(vres_id)_primal")] = C["Primal_cap"][vres_id]
        conv_cols[Symbol("ppa_cap_$(vres_id)_dual")] = C["Dual_cap"][vres_id]
    end
    for h2_id in hpa_h2
        haskey(C_hpa, "Primal") || break
        conv_cols[Symbol("hpa_$(h2_id)_primal")] = C_hpa["Primal"][h2_id]
        conv_cols[Symbol("hpa_$(h2_id)_dual")] = C_hpa["Dual"][h2_id]
        conv_cols[Symbol("hpa_cap_$(h2_id)_primal")] = C_hpa["Primal_cap"][h2_id]
        conv_cols[Symbol("hpa_cap_$(h2_id)_dual")] = C_hpa["Dual_cap"][h2_id]
    end
    conv_df = DataFrame(conv_cols)
    _move_first!(conv_df, :iter)
    CSV.write(joinpath(results_dir, "ADMM_Convergence.csv"), conv_df)

    # ── ADMM_Diagnostics.csv (with per-VRES contract columns) ─────────────────
    ph_ppa(vid) = C["PriceHistory"][vid]
    ph_ppa_slice(vid) = length(ph_ppa(vid)) == n_it + 1 ? ph_ppa(vid)[2:end] : ph_ppa(vid)
    ph_hpa(vid) = C_hpa["PriceHistory"][vid]
    ph_hpa_slice(vid) = length(ph_hpa(vid)) == n_it + 1 ? ph_hpa(vid)[2:end] : ph_hpa(vid)
    diag_cols = Dict(
        :iter => 1:n_it,
        :elec_rho => ADMM_state["ρ"]["elec"][1:n_it],
        :elec_price_mean => ph_slice("elec"),
        :elec_imb_mean => ADMM_state["ImbalanceMean"]["elec"],
        :H2_rho => ADMM_state["ρ"]["H2"][1:n_it],
        :H2_price_mean => ph_slice("H2"),
        :H2_imb_mean => ADMM_state["ImbalanceMean"]["H2"],
        :elec_GC_rho => ADMM_state["ρ"]["elec_GC"][1:n_it],
        :elec_GC_price_mean => ph_slice("elec_GC"),
        :elec_GC_imb_mean => ADMM_state["ImbalanceMean"]["elec_GC"],
        :H2_GC_rho => ADMM_state["ρ"]["H2_GC"][1:n_it],
        :H2_GC_price_mean => ph_slice("H2_GC"),
        :H2_GC_imb_mean => ADMM_state["ImbalanceMean"]["H2_GC"],
        :EP_rho => ADMM_state["ρ"]["EP"][1:n_it],
        :EP_price_mean => ph_slice("EP"),
        :EP_imb_mean => ADMM_state["ImbalanceMean"]["EP"],
    )
    # Per-agent capacity ρ_m columns (new equality-split formulation).
    for m in cap_agents_save
        ρhist = get(get(cap_state_save, "ρ", Dict()), m, Float64[])
        ρ_aligned = length(ρhist) >= n_it ? ρhist[1:n_it] :
                    vcat(ρhist, fill(isempty(ρhist) ? NaN : ρhist[end], n_it - length(ρhist)))
        diag_cols[Symbol("cap_rho_$(m)")] = ρ_aligned
    end
    for vres_id in ppa_vres
        haskey(C, "ρ") || break
        diag_cols[Symbol("ppa_$(vres_id)_rho")] = C["ρ"][vres_id][1:n_it]
        diag_cols[Symbol("ppa_$(vres_id)_price_mean")] = ph_ppa_slice(vres_id)
        diag_cols[Symbol("ppa_$(vres_id)_imb_mean")] = C["ImbalanceMean"][vres_id]
        diag_cols[Symbol("ppa_cap_$(vres_id)_rho")] = C["ρ_cap"][vres_id][1:n_it]
        diag_cols[Symbol("ppa_cap_$(vres_id)_imb_mean")] = C["ImbalanceMean_cap"][vres_id]
    end
    for h2_id in hpa_h2
        haskey(C_hpa, "ρ") || break
        diag_cols[Symbol("hpa_$(h2_id)_rho")] = C_hpa["ρ"][h2_id][1:n_it]
        diag_cols[Symbol("hpa_$(h2_id)_price_mean")] = ph_hpa_slice(h2_id)
        diag_cols[Symbol("hpa_$(h2_id)_imb_mean")] = C_hpa["ImbalanceMean"][h2_id]
        diag_cols[Symbol("hpa_cap_$(h2_id)_rho")] = C_hpa["ρ_cap"][h2_id][1:n_it]
        diag_cols[Symbol("hpa_cap_$(h2_id)_imb_mean")] = C_hpa["ImbalanceMean_cap"][h2_id]
    end
    diag_df = DataFrame(diag_cols)
    _move_first!(diag_df, :iter)
    CSV.write(joinpath(results_dir, "ADMM_Diagnostics.csv"), diag_df)

    # ── Capacity_Consensus.csv ──────────────────────────────────────────
    # Per-iteration, per-agent, per-year snapshot of the capacity ADMM split.
    # Mirrors save_results.jl; see DOCUMENTATION.md §5.4 and §11.
    if !isempty(cap_agents_save)
        cap_rows = NamedTuple[]
        for m in cap_agents_save
            xhist = if !isempty(get(results["Cap_VRES"], m, []))
                results["Cap_VRES"][m]
            elseif !isempty(get(results["Cap_Elec_H2"], m, []))
                results["Cap_Elec_H2"][m]
            elseif !isempty(get(results["Cap_EP_Green"], m, []))
                results["Cap_EP_Green"][m]
            else
                Vector{Vector{Float64}}()
            end
            zhist = get(get(cap_state_save, "z", Dict()), m, Vector{Vector{Float64}}())
            λhist = get(get(cap_state_save, "λ", Dict()), m, Vector{Vector{Float64}}())
            ρhist = get(get(cap_state_save, "ρ", Dict()), m, Float64[])
            rp_hist = get(get(cap_state_save, "Primal", Dict()), m, Float64[])
            rd_hist = get(get(cap_state_save, "Dual",   Dict()), m, Float64[])
            nrec = min(length(xhist), length(zhist))
            for i in 1:nrec
                xvec = xhist[i]
                zvec = zhist[i]
                λvec = i <= length(λhist) ? λhist[i] : zeros(length(xvec))
                ρ_i  = i <= length(ρhist) ? ρhist[i] : (isempty(ρhist) ? NaN : ρhist[end])
                rp_i = i <= length(rp_hist) ? rp_hist[i] : NaN
                rd_i = i <= length(rd_hist) ? rd_hist[i] : NaN
                for jy in 1:length(xvec)
                    push!(cap_rows, (
                        iter         = i,
                        AgentID      = m,
                        jy           = jy,
                        x_cap        = xvec[jy],
                        z_cap        = zvec[jy],
                        lambda_cap   = jy <= length(λvec) ? λvec[jy] : 0.0,
                        rho_cap      = ρ_i,
                        primal_local = rp_i,
                        dual_local   = rd_i,
                    ))
                end
            end
        end
        if !isempty(cap_rows)
            CSV.write(joinpath(results_dir, "Capacity_Consensus.csv"), DataFrame(cap_rows))
        end
    end

    # ── Per-market history (5 base markets only; contract focal info in PPAs/HPAs CSVs) ─
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
        _move_first!(df, :iter)
        CSV.write(joinpath(results_dir, string(name, "_Market_History.csv")), df)
    end

    # ── PPAs.csv — One row per VRES: capacity contracted, energy transferred, bundled price (elec+elec_GC) ─
    # ── Green_Agents_Detail.csv — VRES and electrolyzer breakdown with per-VRES PPA columns ─
    ppa_agents = get(agents, :ppa_market, String[])
    λ_elec_final = isempty(ppa_agents) ? nothing : results["λ"]["elec"][end]
    λ_elec_mean = λ_elec_final === nothing ? 0.0 : mean(λ_elec_final)

    vres_ppa = [id for id in ppa_agents if String(mdict[id].ext[:parameters][:Type]) == "VRES"]
    if !isempty(vres_ppa)
        vres_ids = String[]
        cap_MWs = Float64[]
        energy_MWhs = Float64[]
        prices_EUR = Float64[]
        for vres_id in vres_ppa
            push!(vres_ids, vres_id)
            cap_list = get(results["ppa_cap"], vres_id, [])
            g_list = get(results["ppa"], vres_id, [])
            cap_MW = isempty(cap_list) ? 0.0 : abs(cap_list[end])
            energy_MWh = isempty(g_list) ? 0.0 : abs(sum(g_list[end]))
            λ_vres = get(results["λ_ppa"], vres_id, [fill(0.0, 1, 1, 1)])
            price_mean = isempty(λ_vres) ? 0.0 : mean(λ_vres[end])
            push!(cap_MWs, cap_MW)
            push!(energy_MWhs, energy_MWh)
            push!(prices_EUR, price_mean)
        end
        ppas_df = DataFrame(
            VRES_AgentID = vres_ids,
            capacity_contracted_MW = cap_MWs,
            energy_transferred_MWh = energy_MWhs,
            ppa_price_EUR_per_MWh = prices_EUR,  # bundled elec + elec_GC
        )
        CSV.write(joinpath(results_dir, "PPAs.csv"), ppas_df)
    end

    # ── HPAs.csv — One row per GreenProducer: contracted H2 capacity, transferred H2, bundled H2+H2_GC price ─
    if !isempty(hpa_h2)
        h2_ids = String[]
        cap_MWs = Float64[]
        energy_MWhs = Float64[]
        prices_EUR = Float64[]
        for h2_id in hpa_h2
            push!(h2_ids, h2_id)
            cap_list = get(results["hpa_cap"], h2_id, [])
            g_list = get(results["hpa"], h2_id, [])
            cap_MW = isempty(cap_list) ? 0.0 : abs(cap_list[end])
            energy_MWh = isempty(g_list) ? 0.0 : abs(sum(g_list[end]))
            λ_h = get(results["λ_hpa"], h2_id, [fill(0.0, 1, 1, 1)])
            price_mean = isempty(λ_h) ? 0.0 : mean(λ_h[end])
            push!(cap_MWs, cap_MW)
            push!(energy_MWhs, energy_MWh)
            push!(prices_EUR, price_mean)
        end
        hpas_df = DataFrame(
            H2_Producer_AgentID = h2_ids,
            capacity_contracted_MW = cap_MWs,
            energy_transferred_MWh = energy_MWhs,
            hpa_price_EUR_per_MWh = prices_EUR,
        )
        CSV.write(joinpath(results_dir, "HPAs.csv"), hpas_df)
    end

    # Green agents detail: VRES (total cap, PPA share, pool share) and electrolyzer (per-VRES breakdown)
    if !isempty(ppa_agents)
        detail_ids = String[]
        detail_types = String[]
        total_capacity = Float64[]
        ppa_capacity_MW = Float64[]
        energy_from_ppa_MWh = Float64[]
        energy_from_pool_MWh = Float64[]
        ppa_price_EUR_per_MWh = Float64[]
        electricity_price_EUR_per_MWh = Float64[]
        # Per-VRES columns for electrolyzer (energy_from_ppa_Gen_VRES_Solar_MWh, etc.)
        vres_energy_cols = Dict(v => Float64[] for v in ppa_vres)

        for id in ppa_agents
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

            if atype == "VRES"
                cc = get(results["ppa_cap"], id, [])
                cap_contracted = isempty(cc) ? 0.0 : abs(cc[end])
                gc = get(results["ppa"], id, [])
                g_contract_sum = isempty(gc) ? 0.0 : abs(sum(gc[end]))
                λ_mean = haskey(results["λ_ppa"], id) && !isempty(results["λ_ppa"][id]) ?
                    mean(results["λ_ppa"][id][end]) : 0.0
                for v in ppa_vres
                    push!(vres_energy_cols[v], v == id ? g_contract_sum : 0.0)
                end
            else
                # Electrolyzer: sum over VRES for total; per-VRES from contract_from
                cap_contracted = 0.0
                g_contract_sum = 0.0
                cf = get(results["ppa_from"], id, Dict())
                ccf = get(results["ppa_cap_from"], id, Dict())
                for v in ppa_vres
                    gv = get(cf, v, [])
                    ev = isempty(gv) ? 0.0 : abs(sum(gv[end]))
                    push!(vres_energy_cols[v], ev)
                    g_contract_sum += ev
                    capv = get(ccf, v, [])
                    cap_contracted += isempty(capv) ? 0.0 : abs(capv[end])
                end
                λ_mean = 0.0
                if g_contract_sum > 0 && !isempty(ppa_vres)
                    wsum = 0.0
                    for v in ppa_vres
                        ev = vres_energy_cols[v][end]
                        if ev > 0 && haskey(results["λ_ppa"], v) && !isempty(results["λ_ppa"][v])
                            λ_mean += ev * mean(results["λ_ppa"][v][end])
                            wsum += ev
                        end
                    end
                    λ_mean = wsum > 0 ? λ_mean / wsum : 0.0
                end
            end
            push!(ppa_capacity_MW, cap_contracted)
            push!(energy_from_ppa_MWh, g_contract_sum)
            push!(ppa_price_EUR_per_MWh, λ_mean)

            g_elec = get(results["g"], id, [])
            g_pool_sum = isempty(g_elec) ? 0.0 : (atype == "VRES" ? sum(g_elec[end]) : -sum(g_elec[end]))
            push!(energy_from_pool_MWh, max(0.0, g_pool_sum))
            push!(electricity_price_EUR_per_MWh, λ_elec_mean)
        end

        # Build detail_df with per-VRES energy columns
        detail_cols = Dict(
            :AgentID => detail_ids,
            :Type => detail_types,
            :total_capacity_MW => total_capacity,
            :ppa_capacity_MW => ppa_capacity_MW,
            :energy_from_ppa_MWh => energy_from_ppa_MWh,
            :energy_from_pool_MWh => energy_from_pool_MWh,
            :ppa_price_EUR_per_MWh => ppa_price_EUR_per_MWh,
            :electricity_price_EUR_per_MWh => electricity_price_EUR_per_MWh,
        )
        for v in ppa_vres
            detail_cols[Symbol("energy_from_ppa_$(v)_MWh")] = vres_energy_cols[v]
        end
        detail_df = DataFrame(detail_cols)
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

    ppa_market_agents = get(agents, :ppa_market, String[])

    # ── Base prices (contract λ handled per sub-market in _admm_objective_economic) ─
    λ_elec_final    = results["λ"]["elec"][end]
    λ_H2_final      = results["λ"]["H2"][end]
    λ_elec_GC_final = results["λ"]["elec_GC"][end]
    λ_H2_GC_final   = results["λ"]["H2_GC"][end]
    λ_EP_final      = results["λ"]["EP"][end]

    # ── _admm_objective_economic (contract-aware, per-VRES prices) ───────────
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
        # Build agent-specific prices (λ_ppa: VRES gets 3D array; electrolyzer gets Dict{vres_id => 3D array})
        prices = Dict{Symbol, Any}(
            :λ_elec => λ_elec_final, :λ_H2 => λ_H2_final, :λ_elec_GC => λ_elec_GC_final,
            :λ_H2_GC => λ_H2_GC_final, :λ_EP => λ_EP_final,
        )

        if id in power_consumers
            quantities[:d] = [value(vars[:d][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            return compute_agent_objective_economic(:power_consumer, quantities, prices, params; JH=JH, JD=JD, JY=JY)
        elseif id in power_vres
            if id in ppa_market_agents && haskey(vars, :g_EOM)
                quantities[:g_EOM] = [value(vars[:g_EOM][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:g_ppa] = [value(vars[:g_ppa][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:ppa_cap] = value(vars[:ppa_cap])
                quantities[:cap_VRES] = [value(vars[:cap_VRES][jy]) for jy in JY]
                prices[:λ_ppa] = haskey(results["λ_ppa"], id) && !isempty(results["λ_ppa"][id]) ?
                    results["λ_ppa"][id][end] : zeros(length(sets[:JH]), length(sets[:JD]), length(sets[:JY]))
                return compute_agent_objective_economic(:power_vres_ppa, quantities, prices, params; JH=JH, JD=JD, JY=JY)
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
            if id in ppa_market_agents && haskey(vars, :e_in_pool)
                quantities[:e_in_pool] = [value(vars[:e_in_pool][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                g_cf = vars[:g_ppa_from]
                quantities[:g_ppa_from] = Dict(v => [value(g_cf[v][jh, jd, jy]) for jh in JH, jd in JD, jy in JY] for v in keys(g_cf))
                quantities[:q_elec_gc] = [value(vars[:q_elec_gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:h2_out] = [value(vars[:h2_out][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:q_h2gc] = [value(vars[:q_h2gc][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
                quantities[:cap_H2_y] = [value(vars[:cap_H2_y][jy]) for jy in JY]
                prices[:λ_ppa] = Dict(v => (haskey(results["λ_ppa"], v) && !isempty(results["λ_ppa"][v]) ?
                    results["λ_ppa"][v][end] : zeros(length(JH), length(JD), length(JY))) for v in keys(g_cf))
                return compute_agent_objective_economic(:H2_producer_ppa, quantities, prices, params; JH=JH, JD=JD, JY=JY)
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
            if haskey(vars, :h2_in)
                quantities[:h2_in] = [value(vars[:h2_in][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            elseif haskey(vars, :h2_in_pool)
                quantities[:h2_in] = [value(vars[:h2_in_pool][jh, jd, jy]) for jh in JH, jd in JD, jy in JY]
            else
                quantities[:h2_in] = zeros(length(JH), length(JD), length(JY))
            end
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

    # ── _final_net_quantities (base markets only; contract info in PPAs/HPAs CSVs) ─
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

    # ── Market_Prices.csv (with per-VRES Contract_Price columns) ─────────────
    λ_elec    = results["λ"]["elec"][end]
    λ_H2      = results["λ"]["H2"][end]
    λ_elec_GC = results["λ"]["elec_GC"][end]
    λ_H2_GC   = results["λ"]["H2_GC"][end]
    λ_EP      = results["λ"]["EP"][end]
    λ_ppa_dict = get(results, "λ_ppa", Dict{String, Vector}())
    λ_hpa_dict = get(results, "λ_hpa", Dict{String, Vector}())

    shp = size(λ_elec)
    n_ts, n_rd, n_yr = shp[1], shp[2], shp[3]

    prices_rows = []
    t_index = 1
    for jy in 1:n_yr, jd in 1:n_rd, jh in 1:n_ts
        row = Dict{Symbol, Any}(
            :Time => t_index,
            :Elec_Price => λ_elec[jh, jd, jy],
            :H2_Price => λ_H2[jh, jd, jy],
            :Elec_GC_Price => λ_elec_GC[jh, jd, jy],
            :H2_GC_Price => λ_H2_GC[jh, jd, jy],
            :EP_Price => λ_EP[jh, jd, jy],
        )
        for vres_id in ppa_vres
            λv = get(λ_ppa_dict, vres_id, [])
            row[Symbol("Contract_Price_$(vres_id)")] = isempty(λv) ? 0.0 : λv[end][jh, jd, jy]
        end
        for h2_id in hpa_h2
            λh = get(λ_hpa_dict, h2_id, [])
            row[Symbol("HPA_Price_$(h2_id)")] = isempty(λh) ? 0.0 : λh[end][jh, jd, jy]
        end
        push!(prices_rows, row)
        t_index += 1
    end
    prices_df = DataFrame(prices_rows)
    _move_first!(prices_df, :Time)
    CSV.write(joinpath(results_dir, "Market_Prices.csv"), prices_df)

    risk_metrics = write_admm_risk_outputs!(mdict, agents, results_dir;
                                            case_label = "market_exposure_contracts")
    print_risk_metrics_summary!(risk_metrics; title = "ADMM+contracts risk metrics (ex-post social CVaR)")

    print_admm_run_summary!(ADMM_state, results, agents;
                            results_dir=results_dir,
                            ppa_market=ppa_market,
                            hpa_market=hpa_market)

    return nothing
end
