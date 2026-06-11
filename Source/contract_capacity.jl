# ==============================================================================
# contract_capacity.jl — Shared bilateral contract capacity (PPA / HPA)
# ==============================================================================

using JuMP
#
# Contracted capacity C (MW) is a single scalar per bilateral link, updated
# outside agent subproblems after each ADMM iteration. Neither party optimises
# a separate copy of C (which caused opposing ToP incentives). Agents receive
# C as a fixed parameter (JuMP fix on cap variables) when solving dispatch.
#
# Update rule (bargaining / consensus step):
#   - Track peak hourly contract flow q_peak from both sides.
#   - If the cap binds and energy imbalance wants more volume, expand C.
#   - Otherwise relax C toward q_peak (→ 0 at risk neutrality when q ≈ 0).
#
# ==============================================================================

"""Initialize shared contract capacity state and history buffers."""
function init_shared_contract_capacity!(ADMM_state::Dict, results::Dict,
                                        ppa_market::Dict, hpa_market::Dict)
    ppa_vres = get(ppa_market, "ppa_vres", String[])
    hpa_h2 = get(hpa_market, "hpa_h2", String[])
    ADMM_state["SharedCap"] = Dict(
        "ppa" => Dict(v => 0.0 for v in ppa_vres),
        "hpa" => Dict(h => 0.0 for h in hpa_h2),
    )
    ADMM_state["SharedCapPrev"] = Dict(
        "ppa" => Dict(v => 0.0 for v in ppa_vres),
        "hpa" => Dict(h => 0.0 for h in hpa_h2),
    )
    results["shared_ppa_cap"] = Dict(v => Float64[] for v in ppa_vres)
    results["shared_hpa_cap"] = Dict(h => Float64[] for h in hpa_h2)
    return nothing
end

function _latest_scalar_cap(results::Dict, cap_key::String, agent_id::String)
    hist = get(results, cap_key, Dict())
    haskey(hist, agent_id) || return 0.0
    isempty(hist[agent_id]) && return 0.0
    v = hist[agent_id][end]
    if v isa Real
        return Float64(v)
    elseif v isa AbstractVector
        return isempty(v) ? 0.0 : Float64(maximum(v))
    end
    return 0.0
end

function _physical_max_ppa(results::Dict, mdict::Dict, vres_id::String, h2_ids)
    cap_v = Inf
    if haskey(mdict, vres_id) && haskey(mdict[vres_id].ext[:variables], :cap_VRES)
        cap_v = _latest_scalar_cap(results, "Cap_VRES", vres_id)
        cap_v = cap_v > 0 ? cap_v : get(mdict[vres_id].ext[:parameters], :Capacity, cap_v)
    end
    cap_h2 = Inf
    η = 1.0
    for h2_id in h2_ids
        haskey(mdict, h2_id) || continue
        mod = mdict[h2_id]
        η = Float64(get(mod.ext[:parameters], :SpecificConsumption, η))
        ch = _latest_scalar_cap(results, "Cap_Elec_H2", h2_id)
        ch = ch > 0 ? ch : Float64(get(mod.ext[:parameters], :Capacity_H2_Output, ch))
        cap_h2 = min(cap_h2, ch)
    end
    η = max(η, 1e-9)
    return min(cap_v, cap_h2 / η)
end

function _physical_max_hpa(results::Dict, mdict::Dict, h2_id::String, offtaker_ids)
    cap_h2 = Inf
    if haskey(mdict, h2_id)
        mod = mdict[h2_id]
        ch = _latest_scalar_cap(results, "Cap_Elec_H2", h2_id)
        cap_h2 = ch > 0 ? ch : Float64(get(mod.ext[:parameters], :Capacity_H2_Output, 0.0))
    end
    cap_ep = Inf
    α = 1.0
    for off_id in offtaker_ids
        haskey(mdict, off_id) || continue
        mod = mdict[off_id]
        α = Float64(get(mod.ext[:parameters], :Alpha, α))
        ce = _latest_scalar_cap(results, "Cap_EP_Green", off_id)
        ce = ce > 0 ? ce : Float64(get(mod.ext[:parameters], :Capacity_EP_Out, 0.0))
        cap_ep = min(cap_ep, ce)
    end
    α = max(α, 1e-9)
    return min(cap_h2, cap_ep / α)
end

function _contract_flow_peak(supply, demand, shp)
    q_s = supply isa AbstractArray ? maximum(Float64.(supply)) : 0.0
    q_d = demand isa AbstractArray ? maximum(Float64.(demand)) : 0.0
    return max(q_s, q_d, 0.0)
end

"""
    apply_shared_contract_caps!(mod, agent_id, ADMM_state, ppa_market, hpa_market)

Fix bilateral cap JuMP variables to the shared scalars (not optimised in subproblems).
"""
function apply_shared_contract_caps!(mod::Model, agent_id::String, ADMM_state::Dict,
                                     ppa_market::Dict, hpa_market::Dict)
    shared = ADMM_state["SharedCap"]
    atype = String(get(mod.ext[:parameters], :Type, ""))

    if get(mod.ext[:parameters], :in_ppa_market, false)
        if atype == "VRES" && haskey(mod.ext[:variables], :ppa_cap)
            C = get(shared["ppa"], agent_id, 0.0)
            fix(mod.ext[:variables][:ppa_cap], C; force=true)
        elseif haskey(mod.ext[:variables], :ppa_cap)
            ppa_vres = get(ppa_market, "ppa_vres", String[])
            for vres_id in ppa_vres
                haskey(mod.ext[:variables][:ppa_cap], vres_id) || continue
                C = get(shared["ppa"], vres_id, 0.0)
                fix(mod.ext[:variables][:ppa_cap][vres_id], C; force=true)
            end
        end
    end

    if get(mod.ext[:parameters], :in_hpa_market, false)
        if atype == "GreenProducer" && haskey(mod.ext[:variables], :hpa_cap)
            C = get(shared["hpa"], agent_id, 0.0)
            fix(mod.ext[:variables][:hpa_cap], C; force=true)
        elseif atype == "GreenOfftaker" && haskey(mod.ext[:variables], :hpa_cap)
            hpa_h2 = get(hpa_market, "hpa_h2", String[])
            for h2_id in hpa_h2
                haskey(mod.ext[:variables][:hpa_cap], h2_id) || continue
                C = get(shared["hpa"], h2_id, 0.0)
                fix(mod.ext[:variables][:hpa_cap][h2_id], C; force=true)
            end
        end
    end
    return nothing
end

"""
    update_shared_contract_capacity!(ADMM_state, results, data, mdict, agents,
                                     ppa_market, hpa_market, shp)

Bargaining / consensus step for shared contract capacities after agent solves.
Returns Dict with q_peak per id for cap residual logging.
"""
function update_shared_contract_capacity!(ADMM_state::Dict, results::Dict, data::Dict,
                                          mdict::Dict, agents::Dict,
                                          ppa_market::Dict, hpa_market::Dict, shp::Tuple)
    cfg = get(get(data, "ADMM", Dict()), "contract_cap", Dict())
    τ = Float64(get(cfg, "relaxation", 0.35))
    η_up = Float64(get(cfg, "expand_step", 2.0))
    bind_tol = Float64(get(cfg, "bind_tol", 1e-6))

    shared = ADMM_state["SharedCap"]
    prev = ADMM_state["SharedCapPrev"]
    peaks = Dict{String, Dict{String, Float64}}("ppa" => Dict(), "hpa" => Dict())

    ppa_vres = get(ppa_market, "ppa_vres", String[])
    h2_ids = agents[:H2]
    C_ppa = ADMM_state["ppa"]
    for vres_id in ppa_vres
        supply = isempty(get(results["ppa"], vres_id, [])) ? zeros(shp...) : results["ppa"][vres_id][end]
        demand = zeros(shp...)
        for h2_id in h2_ids
            if !isempty(get(get(results["ppa_from"], h2_id, Dict()), vres_id, []))
                demand .+= results["ppa_from"][h2_id][vres_id][end]
            end
        end
        q_peak = _contract_flow_peak(supply, demand, shp)
        peaks["ppa"][vres_id] = q_peak

        C_old = shared["ppa"][vres_id]
        prev["ppa"][vres_id] = C_old
        imb_mean = isempty(C_ppa["ImbalanceMean"][vres_id]) ? 0.0 : C_ppa["ImbalanceMean"][vres_id][end]
        C_phys = _physical_max_ppa(results, mdict, vres_id, h2_ids)

        if q_peak >= C_old - bind_tol && imb_mean > 0
            C_target = min(C_phys, C_old + η_up * abs(imb_mean))
        else
            C_target = q_peak
        end
        C_new = (1 - τ) * C_old + τ * C_target
        shared["ppa"][vres_id] = clamp(C_new, 0.0, C_phys)
        push!(results["shared_ppa_cap"][vres_id], shared["ppa"][vres_id])
    end

    hpa_h2 = get(hpa_market, "hpa_h2", String[])
    off_ids = agents[:offtaker]
    C_hpa = ADMM_state["hpa"]
    for h2_id in hpa_h2
        supply = isempty(get(results["hpa"], h2_id, [])) ? zeros(shp...) : results["hpa"][h2_id][end]
        demand = zeros(shp...)
        for off_id in off_ids
            if !isempty(get(get(results["hpa_from"], off_id, Dict()), h2_id, []))
                demand .+= results["hpa_from"][off_id][h2_id][end]
            end
        end
        q_peak = _contract_flow_peak(supply, demand, shp)
        peaks["hpa"][h2_id] = q_peak

        C_old = shared["hpa"][h2_id]
        prev["hpa"][h2_id] = C_old
        imb_mean = isempty(C_hpa["ImbalanceMean"][h2_id]) ? 0.0 : C_hpa["ImbalanceMean"][h2_id][end]
        C_phys = _physical_max_hpa(results, mdict, h2_id, off_ids)

        if q_peak >= C_old - bind_tol && imb_mean > 0
            C_target = min(C_phys, C_old + η_up * abs(imb_mean))
        else
            C_target = q_peak
        end
        C_new = (1 - τ) * C_old + τ * C_target
        shared["hpa"][h2_id] = clamp(C_new, 0.0, C_phys)
        push!(results["shared_hpa_cap"][h2_id], shared["hpa"][h2_id])
    end

    return peaks
end

"""Record shared-cap utilisation residuals into ADMM contract-cap markets."""
function record_shared_cap_residuals!(ADMM_state::Dict, peaks::Dict, iter::Int)
    shared = ADMM_state["SharedCap"]
    prev = ADMM_state["SharedCapPrev"]
    for pool in ("ppa", "hpa")
        C = ADMM_state[pool]
        ids = pool == "ppa" ? keys(shared["ppa"]) : keys(shared["hpa"])
        for id in ids
            C_val = shared[pool][id]
            q_peak = get(peaks[pool], id, 0.0)
            rp = abs(C_val - q_peak)
            C_prev = prev[pool][id]
            rd = iter <= 1 ? Inf : abs(C_val - C_prev)
            push!(C["Imbalances_cap"][id], C_val - q_peak)
            push!(C["ImbalanceMean_cap"][id], abs(C_val - q_peak))
            push!(C["Primal_cap"][id], rp)
            push!(C["Dual_cap"][id], rd)
            if C["ResidualScale_Primal_cap"][id] == 0.0 && rp > 0.0
                C["ResidualScale_Primal_cap"][id] = rp
            end
            if iter > 1 && C["ResidualScale_Dual_cap"][id] == 0.0 && isfinite(rd) && rd > 0.0
                C["ResidualScale_Dual_cap"][id] = rd
            end
        end
    end
    return nothing
end

"""Return finalized shared contract capacity for reporting."""
function shared_contract_capacity(results::Dict, pool::Symbol, id::String)
    key = pool == :ppa ? "shared_ppa_cap" : "shared_hpa_cap"
    hist = get(results, key, Dict())
    haskey(hist, id) || return 0.0
    return isempty(hist[id]) ? 0.0 : Float64(hist[id][end])
end
