# ==============================================================================
# ADMM_subroutine_contracts.jl — Per-agent step with bilateral contract support
# ==============================================================================
#
# PURPOSE:
#   Extends ADMM_subroutine! for market_exposure_contracts.
#   Updates PPA/HPA market params (λ_ppa/λ_hpa, g_bar_*, g_bar_*_cap),
#   calls contracts solvers, and extracts contract energy/capacity quantities.
#
# ==============================================================================

function ADMM_subroutine_contracts!(m::String, data::Dict, results::Dict, ADMM_state::Dict,
                                    elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                    H2_GC_market::Dict, EP_market::Dict, ppa_market::Dict, hpa_market::Dict,
                                    mdict::Dict, agents::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    zeros_shp = zeros(n_ts, n_rd, n_yr)
    mod = mdict[m]

    function _to_float(x, default=0.0)
        if x === nothing
            return default
        elseif x isa Number
            return Float64(x)
        end
        try
            return parse(Float64, string(x))
        catch
            return default
        end
    end

    function _contract_strike(cfg::Dict, λ_contract::Array{Float64,3})
        mode = String(get(cfg, "pricing_mode", "endogenous_clearing"))
        K = copy(λ_contract)
        if mode == "fixed"
            fixed = _to_float(get(cfg, "fixed_strike", mean(λ_contract)), mean(λ_contract))
            K .= fixed
        elseif mode == "indexed"
            terms = get(cfg, "index_terms", Dict())
            K .= _to_float(get(terms, "constant", 0.0), 0.0)
            K .+= _to_float(get(terms, "elec", 0.0), 0.0) .* results["λ"]["elec"][end]
            K .+= _to_float(get(terms, "elec_GC", 0.0), 0.0) .* results["λ"]["elec_GC"][end]
            K .+= _to_float(get(terms, "H2", 0.0), 0.0) .* results["λ"]["H2"][end]
            K .+= _to_float(get(terms, "H2_GC", 0.0), 0.0) .* results["λ"]["H2_GC"][end]
            K .+= _to_float(get(terms, "EP", 0.0), 0.0) .* results["λ"]["EP"][end]
        end
        floor_v = _to_float(get(cfg, "price_floor", 0.0), 0.0)
        cap_v = _to_float(get(cfg, "price_cap", 1.0e9), 1.0e9)
        K .= clamp.(K, floor_v, cap_v)
        return K
    end

    # ------------------------------------------------------------------
    # Update ADMM parameters for standard markets (same as base)
    # ------------------------------------------------------------------
    @timeit TO "Update ADMM params" begin
        if mod.ext[:parameters][:in_elec_market]
            n = elec_market["nAgents"]
            prev_g = isempty(results["g"][m]) ? zeros_shp : results["g"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec"]) ? zeros_shp : ADMM_state["Imbalances"]["elec"][end]
            mod.ext[:parameters][:g_bar_elec] = prev_g .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_elec]    = results["λ"]["elec"][end]
            mod.ext[:parameters][:ρ_elec]    = ADMM_state["ρ"]["elec"][end]
        end
        if mod.ext[:parameters][:in_H2_market]
            n = H2_market["nAgents"]
            prev = isempty(results["h2"][m]) ? zeros_shp : results["h2"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2"]) ? zeros_shp : ADMM_state["Imbalances"]["H2"][end]
            mod.ext[:parameters][:g_bar_H2] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_H2]    = results["λ"]["H2"][end]
            mod.ext[:parameters][:ρ_H2]    = ADMM_state["ρ"]["H2"][end]
        end
        if mod.ext[:parameters][:in_elec_GC_market]
            n = elec_GC_market["nAgents"]
            prev = isempty(results["elec_GC"][m]) ? zeros_shp : results["elec_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["elec_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["elec_GC"][end]
            mod.ext[:parameters][:g_bar_elec_GC] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_elec_GC]     = results["λ"]["elec_GC"][end]
            mod.ext[:parameters][:ρ_elec_GC]     = ADMM_state["ρ"]["elec_GC"][end]
        end
        if mod.ext[:parameters][:in_H2_GC_market]
            n = H2_GC_market["nAgents"]
            prev = isempty(results["H2_GC"][m]) ? zeros_shp : results["H2_GC"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["H2_GC"]) ? zeros_shp : ADMM_state["Imbalances"]["H2_GC"][end]
            mod.ext[:parameters][:g_bar_H2_GC] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_H2_GC] = results["λ"]["H2_GC"][end]
            mod.ext[:parameters][:ρ_H2_GC] = ADMM_state["ρ"]["H2_GC"][end]
        end
        if mod.ext[:parameters][:in_EP_market]
            n = EP_market["nAgents"]
            prev = isempty(results["EP"][m]) ? zeros_shp : results["EP"][m][end]
            imb = isempty(ADMM_state["Imbalances"]["EP"]) ? zeros_shp : ADMM_state["Imbalances"]["EP"][end]
            mod.ext[:parameters][:g_bar_EP] = prev .- (1.0 / (n + 1)) .* imb
            mod.ext[:parameters][:λ_EP]    = results["λ"]["EP"][end]
            mod.ext[:parameters][:ρ_EP]    = ADMM_state["ρ"]["EP"][end]
        end

        # Contract market (per-VRES energy + capacity) — set BEFORE z_cap so VRES z_cap can use g_bar_ppa
        if get(mod.ext[:parameters], :in_ppa_market, false)
            ppa_vres = get(ppa_market, "ppa_vres", String[])
            atype = String(get(mod.ext[:parameters], :Type, ""))
            C = ADMM_state["ppa"]

            if atype == "VRES"
                # VRES m: single contract (its own sub-market)
                n_contract = 2  # VRES + electrolyzer
                prev_contract = isempty(results["ppa"][m]) ? zeros_shp : results["ppa"][m][end]
                imb_contract = isempty(C["Imbalances"][m]) ? zeros_shp : C["Imbalances"][m][end]
                mod.ext[:parameters][:g_bar_ppa] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
                mod.ext[:parameters][:λ_ppa]     = results["λ_ppa"][m][end]
                vres_cfg = get(get(ppa_market, "per_vres", Dict()), m, Dict())
                mod.ext[:parameters][:K_ppa]     = _contract_strike(vres_cfg, results["λ_ppa"][m][end])
                mod.ext[:parameters][:ρ_ppa]     = C["ρ"][m][end]

                prev_net_cap = isempty(results["ppa_cap"][m]) ? 0.0 : results["ppa_cap"][m][end]
                imb_cap  = isempty(C["Imbalances_cap"][m]) ? 0.0 : C["Imbalances_cap"][m][end]
                mod.ext[:parameters][:g_bar_ppa_cap] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                mod.ext[:parameters][:ρ_ppa_cap]     = C["ρ_cap"][m][end]
            else
                # Electrolyzer: per-VRES params
                for vres_id in ppa_vres
                    n_contract = 2
                    # Buyer net position is negative of demanded contract flow.
                    prev_contract = isempty(results["ppa_from"][m][vres_id]) ? zeros_shp : .-results["ppa_from"][m][vres_id][end]
                    imb_contract = isempty(C["Imbalances"][vres_id]) ? zeros_shp : C["Imbalances"][vres_id][end]
                    mod.ext[:parameters][:g_bar_ppa][vres_id] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
                    mod.ext[:parameters][:λ_ppa][vres_id]     = results["λ_ppa"][vres_id][end]
                    vres_cfg = get(get(ppa_market, "per_vres", Dict()), vres_id, Dict())
                    mod.ext[:parameters][:K_ppa][vres_id]     = _contract_strike(vres_cfg, results["λ_ppa"][vres_id][end])
                    mod.ext[:parameters][:ρ_ppa][vres_id]     = C["ρ"][vres_id][end]

                    prev_net_cap = isempty(results["ppa_cap_from"][m][vres_id]) ? 0.0 : results["ppa_cap_from"][m][vres_id][end]
                    imb_cap  = isempty(C["Imbalances_cap"][vres_id]) ? 0.0 : C["Imbalances_cap"][vres_id][end]
                    mod.ext[:parameters][:g_bar_ppa_cap][vres_id] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                    mod.ext[:parameters][:ρ_ppa_cap][vres_id]     = C["ρ_cap"][vres_id][end]
                end
            end
        end

        # HPA market (per-GreenProducer energy + capacity)
        if get(mod.ext[:parameters], :in_hpa_market, false)
            hpa_h2 = get(hpa_market, "hpa_h2", String[])
            atype = String(get(mod.ext[:parameters], :Type, ""))
            C_hpa = ADMM_state["hpa"]

            if atype == "GreenProducer"
                n_contract = 2  # GreenProducer + GreenOfftaker
                prev_contract = isempty(results["hpa"][m]) ? zeros_shp : results["hpa"][m][end]
                imb_contract = isempty(C_hpa["Imbalances"][m]) ? zeros_shp : C_hpa["Imbalances"][m][end]
                mod.ext[:parameters][:g_bar_hpa] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
                mod.ext[:parameters][:λ_hpa] = results["λ_hpa"][m][end]
                h2_cfg = get(get(hpa_market, "per_h2", Dict()), m, Dict())
                mod.ext[:parameters][:K_hpa] = _contract_strike(h2_cfg, results["λ_hpa"][m][end])
                mod.ext[:parameters][:ρ_hpa] = C_hpa["ρ"][m][end]

                prev_net_cap = isempty(results["hpa_cap"][m]) ? 0.0 : results["hpa_cap"][m][end]
                imb_cap = isempty(C_hpa["Imbalances_cap"][m]) ? 0.0 : C_hpa["Imbalances_cap"][m][end]
                mod.ext[:parameters][:g_bar_hpa_cap] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                mod.ext[:parameters][:ρ_hpa_cap] = C_hpa["ρ_cap"][m][end]
            elseif atype == "GreenOfftaker"
                for h2_id in hpa_h2
                    n_contract = 2
                    # Buyer net position is negative of demanded contract flow.
                    prev_contract = isempty(results["hpa_from"][m][h2_id]) ? zeros_shp : .-results["hpa_from"][m][h2_id][end]
                    imb_contract = isempty(C_hpa["Imbalances"][h2_id]) ? zeros_shp : C_hpa["Imbalances"][h2_id][end]
                    mod.ext[:parameters][:g_bar_hpa][h2_id] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
                    mod.ext[:parameters][:λ_hpa][h2_id] = results["λ_hpa"][h2_id][end]
                    h2_cfg = get(get(hpa_market, "per_h2", Dict()), h2_id, Dict())
                    mod.ext[:parameters][:K_hpa][h2_id] = _contract_strike(h2_cfg, results["λ_hpa"][h2_id][end])
                    mod.ext[:parameters][:ρ_hpa][h2_id] = C_hpa["ρ"][h2_id][end]

                    prev_net_cap = isempty(results["hpa_cap_from"][m][h2_id]) ? 0.0 : results["hpa_cap_from"][m][h2_id][end]
                    imb_cap = isempty(C_hpa["Imbalances_cap"][h2_id]) ? 0.0 : C_hpa["Imbalances_cap"][h2_id][end]
                    mod.ext[:parameters][:g_bar_hpa_cap][h2_id] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                    mod.ext[:parameters][:ρ_hpa_cap][h2_id] = C_hpa["ρ_cap"][h2_id][end]
                end
            end
        end

        # ----------------------------------------------------------------
        # Capacity consensus parameter refresh (per-agent ADMM equality split)
        #
        # Same per-agent split as in ADMM_subroutine.jl, but z_cap derivation
        # accounts for both the pool flow consensus AND the contract flow
        # consensus (since VRES generation in contracts splits between EOM
        # and PPA pools, and H₂ production splits between H₂ pool and HPA).
        #
        # See ADMM_subroutine.jl and DOCUMENTATION.md §5.4 for the formal
        # derivation, residual definitions, and units check.
        # ----------------------------------------------------------------
        if haskey(mod.ext[:parameters], :z_cap)
            cap_state = ADMM_state["Capacity"]
            # On the first ADMM iteration, if z was warm-started from SP
            # capacities, use that target directly to keep x and z aligned.
            if get(ADMM_state, "n_iter", 0) == 0 && !isempty(cap_state["z"][m])
                z_raw = cap_state["z"][m][end]
                z_cap = z_raw isa Real ? Float64(z_raw) :
                        (isempty(z_raw) ? 0.0 : Float64(maximum(z_raw)))
                mod.ext[:parameters][:z_cap] = z_cap
                λ_raw = cap_state["λ"][m][end]
                mod.ext[:parameters][:λ_cap] = λ_raw isa Real ? Float64(λ_raw) :
                    (isempty(λ_raw) ? 0.0 : Float64(λ_raw[1]))
                mod.ext[:parameters][:ρ_cap] = cap_state["ρ"][m][end]
            else
                agent_type = String(get(mod.ext[:parameters], :Type, ""))
                JY = mod.ext[:sets][:JY]
                cap_floor = if agent_type == "VRES"
                    get(mod.ext[:parameters], :Capacity, 0.0)
                elseif agent_type == "GreenProducer"
                    get(mod.ext[:parameters], :Capacity_H2_Output, 0.0)
                elseif agent_type == "GreenOfftaker"
                    get(mod.ext[:parameters], :Capacity_EP_Out, 0.0)
                else
                    0.0
                end
                z_cap = cap_floor
                if agent_type == "VRES"
                    flow_eom = isempty(get(results["g"], m, [])) ?
                               mod.ext[:parameters][:g_bar_elec] : results["g"][m][end]
                    AF = mod.ext[:timeseries][:AF]
                    flow_ppa = isempty(get(results["ppa"], m, [])) ?
                               get(mod.ext[:parameters], :g_bar_ppa, zeros_shp) :
                               results["ppa"][m][end]
                    flow_total = flow_eom .+ flow_ppa
                    for jy in JY
                        for jh in 1:n_ts, jd in 1:n_rd
                            af = AF[jh, jd, jy]
                            if af > 1e-9
                                z_cap = max(z_cap, max(0.0, flow_total[jh, jd, jy] / af))
                            end
                        end
                    end
                elseif agent_type == "GreenProducer"
                    flow_pool = isempty(get(results["h2"], m, [])) ?
                                mod.ext[:parameters][:g_bar_H2] : results["h2"][m][end]
                    flow_hpa = isempty(get(results["hpa"], m, [])) ?
                               get(mod.ext[:parameters], :g_bar_hpa, zeros_shp) :
                               results["hpa"][m][end]
                    flow_total = flow_pool .+ flow_hpa
                    for jy in JY
                        z_cap = max(z_cap, max(0.0, maximum(flow_total[:, :, jy])))
                    end
                elseif agent_type == "GreenOfftaker"
                    flow_ref = isempty(get(results["EP"], m, [])) ?
                               mod.ext[:parameters][:g_bar_EP] : results["EP"][m][end]
                    for jy in JY
                        z_cap = max(z_cap, max(0.0, maximum(flow_ref[:, :, jy])))
                    end
                end

                z_alpha = get(get(data, "ADMM", Dict()), "cap_z_relax", 1.0)
                z_alpha = min(1.0, max(0.05, z_alpha))
                if !isempty(cap_state["z"][m])
                    z_prev = cap_state["z"][m][end]
                    z_prev_scalar = z_prev isa Real ? Float64(z_prev) :
                        (isempty(z_prev) ? z_cap : Float64(z_prev[1]))
                    z_cap = z_alpha * z_cap + (1.0 - z_alpha) * z_prev_scalar
                end
                z_cap = max(z_cap, cap_floor)
                push!(cap_state["z"][m], z_cap)

                mod.ext[:parameters][:z_cap] = z_cap
                λ_raw = cap_state["λ"][m][end]
                mod.ext[:parameters][:λ_cap] = λ_raw isa Real ? Float64(λ_raw) :
                    (isempty(λ_raw) ? 0.0 : Float64(λ_raw[1]))
                mod.ext[:parameters][:ρ_cap] = cap_state["ρ"][m][end]
            end
        end
    end

    # ------------------------------------------------------------------
    # Solve dispatch: use contracts versions for power and H2
    # ------------------------------------------------------------------
    @timeit TO "Solve agent" begin
        if m in agents[:power]
            solve_power_agent_contracts!(m, mod, elec_market, elec_GC_market, ppa_market)
        elseif m in agents[:H2]
            solve_H2_agent_contracts!(m, mod, H2_market, H2_GC_market, ppa_market)
        elseif m in agents[:offtaker]
            solve_offtaker_agent_contracts!(m, mod, EP_market, H2_market, H2_GC_market, hpa_market)
        elseif m in agents[:elec_GC_demand]
            solve_elec_GC_demand_agent!(m, mod, elec_GC_market)
        end
    end

    # ------------------------------------------------------------------
    # Result extraction (same as base + contract quantities)
    # ------------------------------------------------------------------
    @timeit TO "Query results" begin
        if mod.ext[:parameters][:in_elec_market]
            g = collect(value.(mod.ext[:expressions][:g_net_elec]))
            push!(results["g"][m], g)
        end
        if mod.ext[:parameters][:in_H2_market]
            h2 = collect(value.(mod.ext[:expressions][:g_net_H2]))
            push!(results["h2"][m], h2)
        end
        if mod.ext[:parameters][:in_elec_GC_market]
            gc = collect(value.(mod.ext[:expressions][:g_net_elec_GC]))
            push!(results["elec_GC"][m], gc)
        end
        if mod.ext[:parameters][:in_H2_GC_market]
            h2gc = collect(value.(mod.ext[:expressions][:g_net_H2_GC]))
            push!(results["H2_GC"][m], h2gc)
        end
        if mod.ext[:parameters][:in_EP_market]
            ep = collect(value.(mod.ext[:expressions][:g_net_EP]))
            push!(results["EP"][m], ep)
        end

        # Contract market quantities (per-VRES)
        if get(mod.ext[:parameters], :in_ppa_market, false)
            atype = String(get(mod.ext[:parameters], :Type, ""))
            if atype == "VRES"
                g_contract = collect(value.(mod.ext[:expressions][:g_net_ppa]))
                push!(results["ppa"][m], g_contract)
                cap_val = value(mod.ext[:variables][:ppa_cap])
                push!(results["ppa_cap"][m], cap_val)  # VRES: +supply
            else
                g_contract_from = mod.ext[:expressions][:g_net_ppa_from]
                ppa_cap_var = mod.ext[:variables][:ppa_cap]
                for vres_id in keys(g_contract_from)
                    g_from = collect(value.(g_contract_from[vres_id]))
                    push!(results["ppa_from"][m][vres_id], g_from)
                    cap_val = value(ppa_cap_var[vres_id])
                    push!(results["ppa_cap_from"][m][vres_id], -cap_val)  # Electrolyzer: -demand
                end
            end
        end

        if get(mod.ext[:parameters], :in_hpa_market, false)
            atype = String(get(mod.ext[:parameters], :Type, ""))
            if atype == "GreenProducer"
                g_contract = collect(value.(mod.ext[:expressions][:g_net_hpa]))
                push!(results["hpa"][m], g_contract)
                cap_val = value(mod.ext[:variables][:hpa_cap])
                push!(results["hpa_cap"][m], cap_val)
            elseif atype == "GreenOfftaker"
                g_contract_from = mod.ext[:expressions][:g_net_hpa_from]
                hpa_cap_var = mod.ext[:variables][:hpa_cap]
                for h2_id in keys(g_contract_from)
                    g_from = collect(value.(g_contract_from[h2_id]))
                    push!(results["hpa_from"][m][h2_id], g_from)
                    cap_val = value(hpa_cap_var[h2_id])
                    push!(results["hpa_cap_from"][m][h2_id], -cap_val)
                end
            end
        end

        agent_type = String(get(mod.ext[:parameters], :Type, ""))
        if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
            cap_vec = [value(mod.ext[:variables][:cap_VRES])]
            inv_vec = [value(mod.ext[:variables][:inv_VRES])]
            push!(results["Cap_VRES"][m], cap_vec)
            push!(results["Inv_VRES"][m], inv_vec)
        end
        if (haskey(mod.ext[:variables], :cap_H2_y) && haskey(mod.ext[:variables], :inv_cap_H2))
            cap_vec = [value(mod.ext[:variables][:cap_H2_y])]
            inv_vec = [value(mod.ext[:variables][:inv_cap_H2])]
            push!(results["Cap_Elec_H2"][m], cap_vec)
            push!(results["Inv_Elec_H2"][m], inv_vec)
        end
        if agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
            cap_vec = [value(mod.ext[:variables][:cap_EP_y])]
            inv_vec = [value(mod.ext[:variables][:inv_EP])]
            push!(results["Cap_EP_Green"][m], cap_vec)
            push!(results["Inv_EP_Green"][m], inv_vec)
        end
    end
    return nothing
end
