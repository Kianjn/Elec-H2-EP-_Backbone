# ==============================================================================
# ADMM_subroutine_contracts.jl — Per-agent step with bilateral contract support
# ==============================================================================
#
# PURPOSE:
#   Extends ADMM_subroutine! for market_exposure_contracts. For VRES and
#   electrolyzer: updates contract market params (λ_contract, g_bar_contract,
#   g_bar_contract_cap), calls solve_*_contracts!, and
#   extracts contract and contract_cap quantities.
#
# ==============================================================================

function ADMM_subroutine_contracts!(m::String, data::Dict, results::Dict, ADMM_state::Dict,
                                    elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                    H2_GC_market::Dict, EP_market::Dict, ppa_market::Dict,
                                    mdict::Dict, agents::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    zeros_shp = zeros(n_ts, n_rd, n_yr)
    mod = mdict[m]

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

        # Contract market (per-VRES energy + capacity) — set BEFORE cap_bar so VRES cap_bar can use g_bar_ppa
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
                mod.ext[:parameters][:ρ_ppa]     = C["ρ"][m][end]

                prev_net_cap = isempty(results["ppa_cap"][m]) ? 0.0 : results["ppa_cap"][m][end]
                imb_cap  = isempty(C["Imbalances_cap"][m]) ? 0.0 : C["Imbalances_cap"][m][end]
                mod.ext[:parameters][:g_bar_ppa_cap] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                mod.ext[:parameters][:ρ_ppa_cap]     = C["ρ_cap"][m][end]
            else
                # Electrolyzer: per-VRES params
                for vres_id in ppa_vres
                    n_contract = 2
                    prev_contract = isempty(results["ppa_from"][m][vres_id]) ? zeros_shp : results["ppa_from"][m][vres_id][end]
                    imb_contract = isempty(C["Imbalances"][vres_id]) ? zeros_shp : C["Imbalances"][vres_id][end]
                    mod.ext[:parameters][:g_bar_ppa][vres_id] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
                    mod.ext[:parameters][:λ_ppa][vres_id]     = results["λ_ppa"][vres_id][end]
                    mod.ext[:parameters][:ρ_ppa][vres_id]     = C["ρ"][vres_id][end]

                    prev_net_cap = isempty(results["ppa_cap_from"][m][vres_id]) ? 0.0 : results["ppa_cap_from"][m][vres_id][end]
                    imb_cap  = isempty(C["Imbalances_cap"][vres_id]) ? 0.0 : C["Imbalances_cap"][vres_id][end]
                    mod.ext[:parameters][:g_bar_ppa_cap][vres_id] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
                    mod.ext[:parameters][:ρ_ppa_cap][vres_id]     = C["ρ_cap"][vres_id][end]
                end
            end
        end

        # Investment consensus: cap_bar = capacity needed to support flow consensus (same as base ADMM_subroutine).
        # For VRES in contracts: total generation = g_EOM + g_ppa, so cap_bar uses g_bar_elec + g_bar_ppa.
        if haskey(mod.ext[:parameters], :cap_bar)
            agent_type = String(get(mod.ext[:parameters], :Type, ""))
            JY = mod.ext[:sets][:JY]
            cap_bar = zeros(length(JY))
            if agent_type == "VRES"
                g_bar = mod.ext[:parameters][:g_bar_elec]
                AF = mod.ext[:timeseries][:AF]
                g_bar_ppa = get(mod.ext[:parameters], :g_bar_ppa, zeros_shp)
                g_bar_total = g_bar .+ g_bar_ppa
                for (iy, jy) in enumerate(JY)
                    mx = 0.0
                    for jh in 1:n_ts, jd in 1:n_rd
                        af = AF[jh, jd, jy]
                        mx = max(mx, af > 1e-9 ? max(0.0, g_bar_total[jh, jd, jy] / af) : 0.0)
                    end
                    cap_bar[iy] = mx
                end
            elseif agent_type == "GreenProducer"
                g_bar = mod.ext[:parameters][:g_bar_H2]
                for (iy, jy) in enumerate(JY)
                    cap_bar[iy] = max(0.0, maximum(g_bar[:, :, jy]))
                end
            elseif agent_type == "GreenOfftaker"
                g_bar = mod.ext[:parameters][:g_bar_EP]
                for (iy, jy) in enumerate(JY)
                    cap_bar[iy] = max(0.0, maximum(g_bar[:, :, jy]))
                end
            end
            mod.ext[:parameters][:cap_bar] = cap_bar
            mod.ext[:parameters][:ρ_cap]  = ADMM_state["ρ"]["cap"][end]
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
            solve_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)
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

        agent_type = String(get(mod.ext[:parameters], :Type, ""))
        if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
            cap_vec = collect(value.(mod.ext[:variables][:cap_VRES]))
            inv_vec = collect(value.(mod.ext[:variables][:inv_VRES]))
            push!(results["Cap_VRES"][m], cap_vec)
            push!(results["Inv_VRES"][m], inv_vec)
        end
        if (haskey(mod.ext[:variables], :cap_H2_y) && haskey(mod.ext[:variables], :inv_cap_H2))
            cap_vec = collect(value.(mod.ext[:variables][:cap_H2_y]))
            inv_vec = collect(value.(mod.ext[:variables][:inv_cap_H2]))
            push!(results["Cap_Elec_H2"][m], cap_vec)
            push!(results["Inv_Elec_H2"][m], inv_vec)
        end
        if agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
            cap_vec = collect(value.(mod.ext[:variables][:cap_EP_y]))
            inv_vec = collect(value.(mod.ext[:variables][:inv_EP]))
            push!(results["Cap_EP_Green"][m], cap_vec)
            push!(results["Inv_EP_Green"][m], inv_vec)
        end
    end
    return nothing
end
