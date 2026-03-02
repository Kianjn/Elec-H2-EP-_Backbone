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
                                    H2_GC_market::Dict, EP_market::Dict, contract_market::Dict,
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

        # Contract market (energy + capacity) — only for VRES and electrolyzer
        if get(mod.ext[:parameters], :in_contract_market, false)
            n_contract = contract_market["nAgents"]
            prev_contract = isempty(results["contract"][m]) ? zeros_shp : results["contract"][m][end]
            imb_contract = isempty(ADMM_state["Imbalances"]["contract"]) ? zeros_shp : ADMM_state["Imbalances"]["contract"][end]
            mod.ext[:parameters][:g_bar_contract] = prev_contract .- (1.0 / (n_contract + 1)) .* imb_contract
            mod.ext[:parameters][:λ_contract]     = results["λ"]["contract"][end]
            mod.ext[:parameters][:ρ_contract]     = ADMM_state["ρ"]["contract"][end]

            prev_net_cap = isempty(results["contract_cap"][m]) ? 0.0 : results["contract_cap"][m][end]
            imb_cap  = isempty(ADMM_state["Imbalances"]["contract_cap"]) ? 0.0 : ADMM_state["Imbalances"]["contract_cap"][end]
            mod.ext[:parameters][:g_bar_contract_cap] = prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
            mod.ext[:parameters][:ρ_contract_cap]     = ADMM_state["ρ"]["contract_cap"][end]
        end
    end

    # ------------------------------------------------------------------
    # Solve dispatch: use contracts versions for power and H2
    # ------------------------------------------------------------------
    @timeit TO "Solve agent" begin
        if m in agents[:power]
            solve_power_agent_contracts!(m, mod, elec_market, elec_GC_market, contract_market)
        elseif m in agents[:H2]
            solve_H2_agent_contracts!(m, mod, H2_market, H2_GC_market, contract_market)
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

        # Contract market quantities
        if get(mod.ext[:parameters], :in_contract_market, false)
            g_contract = collect(value.(mod.ext[:expressions][:g_net_contract]))
            push!(results["contract"][m], g_contract)
            cap_val = value(mod.ext[:variables][:contract_cap])
            # Net position: VRES supplies +contract_cap, electrolyzer demands -contract_cap
            atype = String(get(mod.ext[:parameters], :Type, ""))
            net_cap = (atype == "VRES") ? cap_val : -cap_val
            push!(results["contract_cap"][m], net_cap)
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
