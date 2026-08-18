# ==============================================================================
# contract_capacity.jl — Shared bilateral contract capacity (PPA / HPA)
# ==============================================================================

using JuMP
using Statistics
#
# Contracted capacity C (MW) is a single scalar per bilateral link, updated
# outside agent subproblems after each ADMM iteration. Neither party optimises
# a separate copy of C (which caused opposing ToP incentives). Agents receive
# C as a fixed parameter (JuMP fix on cap variables) when solving dispatch.
#
# Update rule (bargaining / consensus step):
#   - Seed C > 0 so q ≤ C is not presolved away (Gurobi duals exist).
#   - Expand C only when BOTH sides want more capacity: shadow of q ≤ C
#     (and/or reduced cost of the fixed cap) is positive, or — if duals are
#     missing — both sides use the full slice. Energy imbalance is NOT used:
#     when both are rationed at the same C, imb = 0 even though they want more.
#   - Expand size is remaining headroom to the physical bottleneck, not a
#     fixed 25 MW crawl, and is applied undamped (so expand_step is not
#     silently scaled by relaxation τ). Floor expand_step, cap expand_max.
#   - Otherwise HOLD at current C. Snap to zero only on a sustained idle
#     utilisation streak (unused slice). Do not default-shrink to 0 whenever
#     someone stops asking for more — that bang-bangs and never settles.
#
# ==============================================================================

"""Read `ADMM.contract_cap` from either nested `data["ADMM"]` or flattened admm_data."""
function _contract_cap_cfg(data::Dict)
    nested = get(get(data, "ADMM", Dict()), "contract_cap", nothing)
    nested isa Dict && return nested
    flat = get(data, "contract_cap", nothing)
    return flat isa Dict ? flat : Dict()
end

"""Initialize shared contract capacity state and history buffers."""
function init_shared_contract_capacity!(ADMM_state::Dict, results::Dict,
                                        ppa_market::Dict, hpa_market::Dict,
                                        data::Dict=Dict())
    cfg = _contract_cap_cfg(data)
    C0 = max(0.0, Float64(get(cfg, "initial", 10.0)))
    ppa_vres = get(ppa_market, "ppa_vres", String[])
    hpa_h2 = get(hpa_market, "hpa_h2", String[])
    ADMM_state["SharedCap"] = Dict(
        "ppa" => Dict(v => C0 for v in ppa_vres),
        "hpa" => Dict(h => C0 for h in hpa_h2),
    )
    ADMM_state["SharedCapPrev"] = Dict(
        "ppa" => Dict(v => 0.0 for v in ppa_vres),
        "hpa" => Dict(h => 0.0 for h in hpa_h2),
    )
    ADMM_state["CapSignal"] = Dict(
        "ppa" => Dict(v => Dict("up" => 0, "down" => 0) for v in ppa_vres),
        "hpa" => Dict(h => Dict("up" => 0, "down" => 0) for h in hpa_h2),
    )
    ADMM_state["CapMove"] = Dict(
        "ppa" => Dict{String, Symbol}(),
        "hpa" => Dict{String, Symbol}(),
    )
    results["shared_ppa_cap"] = Dict(v => Float64[] for v in ppa_vres)
    results["shared_hpa_cap"] = Dict(h => Float64[] for h in hpa_h2)
    results["ppa_q_peak"] = Dict(v => Float64[] for v in ppa_vres)
    results["hpa_q_peak"] = Dict(h => Float64[] for h in hpa_h2)
    results["ppa_util_ratio"] = Dict(v => Float64[] for v in ppa_vres)
    results["hpa_util_ratio"] = Dict(h => Float64[] for h in hpa_h2)
    return nothing
end

function _confirmed_move!(sig::Dict, id::String, wants_up::Bool, up_confirm::Int, down_confirm::Int)
    st = get!(sig, id, Dict("up" => 0, "down" => 0))
    if wants_up
        st["up"] = Int(get(st, "up", 0)) + 1
        st["down"] = 0
    else
        st["down"] = Int(get(st, "down", 0)) + 1
        st["up"] = 0
    end
    if st["up"] >= up_confirm
        return :expand
    elseif st["down"] >= down_confirm
        return :shrink
    else
        return :hold
    end
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
        # Grey / import offtakers are not HPA counterparties; including them
        # would set cap_ep = 0 and clamp a positive C seed back to zero.
        String(get(mod.ext[:parameters], :Type, "")) == "GreenOfftaker" || continue
        α = Float64(get(mod.ext[:parameters], :Alpha, α))
        ce = _latest_scalar_cap(results, "Cap_EP_Green", off_id)
        ce = ce > 0 ? ce : Float64(get(mod.ext[:parameters], :Capacity_EP_Out, 0.0))
        cap_ep = min(cap_ep, ce)
    end
    α = max(α, 1e-9)
    return min(cap_h2, cap_ep / α)
end

function _contract_flow_peaks(supply, demand)
    q_s = supply isa AbstractArray && !isempty(supply) ? maximum(Float64.(supply)) : 0.0
    q_d = demand isa AbstractArray && !isempty(demand) ? maximum(Float64.(demand)) : 0.0
    return max(q_s, 0.0), max(q_d, 0.0), max(q_s, q_d, 0.0)
end

"""
Average mutual flow across all slots (energy-like utilisation signal).

Peak-based checks can stay non-zero due to tiny spikes; this average reflects
whether the contracted slice is materially used over the full horizon.
"""
function _contract_flow_avg_mutual(supply, demand)
    if !(supply isa AbstractArray) || isempty(supply) || !(demand isa AbstractArray) || isempty(demand)
        return 0.0
    end
    qs = max.(Float64.(supply), 0.0)
    qd = max.(Float64.(demand), 0.0)
    return mean(min.(qs, qd))
end

"""W-weighted sum of positive duals of an hourly q ≤ C constraint (Min problem)."""
function _weighted_pos_dual(cref, W::AbstractMatrix)
    acc = 0.0
    try
        ax = axes(cref)
        length(ax) == 3 || return NaN
        for jh in ax[1], jd in ax[2], jy in ax[3]
            d = dual(cref[jh, jd, jy])
            isfinite(d) || continue
            acc += Float64(W[jd, jy]) * max(Float64(d), 0.0)
        end
        return acc
    catch
        return NaN
    end
end

"""Reduced cost of a JuMP-fixed cap. Min problem: RC < 0 ⇒ increasing C helps."""
function _fixed_var_reduced_cost(x)
    try
        rc = reduced_cost(x)
        return isfinite(rc) ? Float64(rc) : NaN
    catch
        try
            rc = dual(FixRef(x))
            return isfinite(rc) ? Float64(rc) : NaN
        catch
            return NaN
        end
    end
end

"""
Shadow of extra contract MW for one party.

Returns (want_more, duals_available, shadow).
`shadow` is €/MW-year value of relaxing C (positive ⇒ wants more).
"""
function _party_cap_shadow(mod, cap_var, cap_limit_constr, W::AbstractMatrix, dual_tol::Float64)
    (mod === nothing || cap_var === nothing) && return (false, false, NaN)
    try
        has_values(mod) || return (false, false, NaN)
    catch
        return (false, false, NaN)
    end

    shadow_rc = NaN
    rc = _fixed_var_reduced_cost(cap_var)
    if isfinite(rc)
        shadow_rc = max(-rc, 0.0)   # Min: negative RC = wants more C
    end

    shadow_μ = NaN
    duals_ok = false
    try
        duals_ok = has_duals(mod)
    catch
        duals_ok = false
    end
    if duals_ok && cap_limit_constr !== nothing
        μ = _weighted_pos_dual(cap_limit_constr, W)
        shadow_μ = μ
    end

    shadows = filter(isfinite, (shadow_rc, shadow_μ))
    isempty(shadows) && return (false, false, NaN)
    shadow = maximum(shadows)
    return (shadow > dual_tol, true, shadow)
end

function _lookup_var(mod, key)
    (mod === nothing || !haskey(mod.ext[:variables], key)) && return nothing
    return mod.ext[:variables][key]
end

function _lookup_constr(mod, key)
    (mod === nothing || !haskey(mod.ext[:constraints], key)) && return nothing
    return mod.ext[:constraints][key]
end

function _index_lookup(container, id)
    container === nothing && return nothing
    try
        return container[id]
    catch
        return nothing
    end
end

"""Pick W from any solved agent model."""
function _admm_weights(mdict::Dict, agents::Dict)
    for ids in (get(agents, :all, String[]), get(agents, :H2, String[]),
                get(agents, :power, String[]), collect(keys(mdict)))
        for id in ids
            haskey(mdict, id) || continue
            W = get(mdict[id].ext[:parameters], :W, nothing)
            W isa AbstractMatrix && return W
        end
    end
    error("No agent model has a W weight matrix for contract-cap duals.")
end

const IDLE_SNAP_ITERS = 3
const IDLE_UTIL_FRAC  = 0.02

"""
MW to add when both sides want more C.

Not a fixed crawl: take `expand_frac` of remaining headroom to the physical
bottleneck, at least `expand_step`, at most `expand_max`. Applied undamped so
yaml `expand_step: 25` is not silently turned into 8.75 MW by τ=0.35.
"""
function _expand_delta(C_old, C_phys, cfg::Dict)
    headroom = max(0.0, C_phys - C_old)
    η_min = Float64(get(cfg, "expand_step", 25.0))
    frac  = Float64(get(cfg, "expand_frac", 0.2))
    η_max = Float64(get(cfg, "expand_max", 500.0))
    raw = max(η_min, frac * headroom)
    return min(headroom, raw, η_max)
end

"""Next shared C after a confirmed move (expand / hold / idle snap)."""
function _next_shared_C(C_old, C_phys, move::Symbol, idle::Int, cfg::Dict)
    if move == :expand
        return clamp(C_old + _expand_delta(C_old, C_phys, cfg), 0.0, C_phys)
    elseif idle >= IDLE_SNAP_ITERS
        return 0.0
    else
        return C_old
    end
end

"""Count consecutive recent iterations where utilisation ratio stayed low."""
function _idle_streak(util_hist::Dict, id::String, idle_frac::Float64)
    haskey(util_hist, id) || return 0
    uvec = util_hist[id]
    n = 0
    for i in length(uvec):-1:1
        uvec[i] <= idle_frac || break
        n += 1
    end
    return n
end

function _cap_hist_flat(hist, settle_tol::Float64, n_hold::Int)
    length(hist) < n_hold && return false
    for i in 0:(n_hold - 2)
        abs(Float64(hist[end - i]) - Float64(hist[end - i - 1])) > settle_tol && return false
    end
    return true
end

"""
    shared_contract_capacity_settled(ADMM_state, results, data) -> Bool

True when every bilateral link's shared capacity C has been flat for
`settle_iters` consecutive iterations, is not expanding, and has no
pending CapSignal-up streak.

Fails closed: missing SharedCap or short history ⇒ not settled. Flow-market
Boyd residuals can pass while C is still crawling; do not declare ADMM
convergence in that case.
"""
function shared_contract_capacity_settled(ADMM_state::Dict, results::Dict, data::Dict)
    cfg = _contract_cap_cfg(data)
    settle_tol = Float64(get(cfg, "settle_tol", 0.5))
    n_hold = Int(get(cfg, "settle_iters", 3))
    shared = get(ADMM_state, "SharedCap", nothing)
    shared === nothing && return false
    prev = get(ADMM_state, "SharedCapPrev", Dict())
    signal = get(ADMM_state, "CapSignal", Dict())
    moves = get(ADMM_state, "CapMove", Dict())

    for pool in ("ppa", "hpa")
        haskey(shared, pool) || continue
        hist_key = pool == "ppa" ? "shared_ppa_cap" : "shared_hpa_cap"
        hists = get(results, hist_key, Dict())
        for id in keys(shared[pool])
            C_now = Float64(shared[pool][id])
            C_was = Float64(get(get(prev, pool, Dict()), id, C_now))
            if abs(C_now - C_was) > settle_tol
                return false
            end
            move = get(get(moves, pool, Dict()), id, :hold)
            move == :expand && return false
            st = get(get(signal, pool, Dict()), id, Dict())
            Int(get(st, "up", 0)) > 0 && return false
            hist = get(hists, id, Float64[])
            _cap_hist_flat(hist, settle_tol, n_hold) || return false
        end
    end
    return true
end

"""
    apply_shared_contract_caps!(mod, agent_id, ADMM_state, ppa_market, hpa_market)

Fix bilateral cap JuMP variables to the shared scalars (not optimised in subproblems).
"""
function apply_shared_contract_caps!(mod::Model, agent_id::String, ADMM_state::Dict,
                                     ppa_market::Dict, hpa_market::Dict)
    haskey(ADMM_state, "SharedCap") || return nothing
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
    cfg = _contract_cap_cfg(data)
    bind_tol = Float64(get(cfg, "bind_tol", 1e-6))
    dual_tol = Float64(get(cfg, "dual_tol", 1e-4))
    up_confirm = Int(get(cfg, "up_confirm_iters", 3))
    down_confirm = Int(get(cfg, "down_confirm_iters", 3))
    W = _admm_weights(mdict, agents)

    shared = ADMM_state["SharedCap"]
    prev = ADMM_state["SharedCapPrev"]
    signal = ADMM_state["CapSignal"]
    moves = ADMM_state["CapMove"]
    peaks = Dict{String, Dict{String, Float64}}("ppa" => Dict(), "hpa" => Dict())

    ppa_vres = get(ppa_market, "ppa_vres", String[])
    h2_ids = agents[:H2]
    for vres_id in ppa_vres
        supply = isempty(get(results["ppa"], vres_id, [])) ? zeros(shp...) : results["ppa"][vres_id][end]
        demand = zeros(shp...)
        for h2_id in h2_ids
            if !isempty(get(get(results["ppa_from"], h2_id, Dict()), vres_id, []))
                demand .+= results["ppa_from"][h2_id][vres_id][end]
            end
        end
        q_s, q_d, q_peak = _contract_flow_peaks(supply, demand)
        q_avg_mut = _contract_flow_avg_mutual(supply, demand)
        peaks["ppa"][vres_id] = q_peak

        C_old = shared["ppa"][vres_id]
        prev["ppa"][vres_id] = C_old
        C_phys = _physical_max_ppa(results, mdict, vres_id, h2_ids)

        seller_mod = get(mdict, vres_id, nothing)
        s_want, s_duals, _ = _party_cap_shadow(
            seller_mod,
            _lookup_var(seller_mod, :ppa_cap),
            _lookup_constr(seller_mod, :ppa_cap_limit),
            W, dual_tol)

        b_want, b_duals = false, false
        for h2_id in h2_ids
            bmod = get(mdict, h2_id, nothing)
            bw, bd, _ = _party_cap_shadow(
                bmod,
                _index_lookup(_lookup_var(bmod, :ppa_cap), vres_id),
                _lookup_constr(bmod, Symbol("ppa_cap_limit_", vres_id)),
                W, dual_tol)
            b_want |= bw
            b_duals |= bd
        end

        wants_up = (s_duals && b_duals) ? (s_want && b_want) :
                   ((C_old > bind_tol) && (q_s >= C_old - bind_tol) && (q_d >= C_old - bind_tol))
        move = _confirmed_move!(signal["ppa"], vres_id, wants_up, up_confirm, down_confirm)
        moves["ppa"][vres_id] = move
        push!(results["ppa_q_peak"][vres_id], q_peak)
        util = C_old > 0.0 ? (q_avg_mut / C_old) : 0.0
        push!(results["ppa_util_ratio"][vres_id], util)
        idle = _idle_streak(results["ppa_util_ratio"], vres_id, IDLE_UTIL_FRAC)
        C_new = _next_shared_C(C_old, C_phys, move, idle, cfg)
        shared["ppa"][vres_id] = C_new
        push!(results["shared_ppa_cap"][vres_id], shared["ppa"][vres_id])
    end

    hpa_h2 = get(hpa_market, "hpa_h2", String[])
    off_ids = agents[:offtaker]
    for h2_id in hpa_h2
        supply = isempty(get(results["hpa"], h2_id, [])) ? zeros(shp...) : results["hpa"][h2_id][end]
        demand = zeros(shp...)
        for off_id in off_ids
            if !isempty(get(get(results["hpa_from"], off_id, Dict()), h2_id, []))
                demand .+= results["hpa_from"][off_id][h2_id][end]
            end
        end
        q_s, q_d, q_peak = _contract_flow_peaks(supply, demand)
        q_avg_mut = _contract_flow_avg_mutual(supply, demand)
        peaks["hpa"][h2_id] = q_peak

        C_old = shared["hpa"][h2_id]
        prev["hpa"][h2_id] = C_old
        C_phys = _physical_max_hpa(results, mdict, h2_id, off_ids)

        seller_mod = get(mdict, h2_id, nothing)
        s_want, s_duals, _ = _party_cap_shadow(
            seller_mod,
            _lookup_var(seller_mod, :hpa_cap),
            _lookup_constr(seller_mod, :hpa_cap_limit),
            W, dual_tol)

        b_want, b_duals = false, false
        for off_id in off_ids
            bmod = get(mdict, off_id, nothing)
            bw, bd, _ = _party_cap_shadow(
                bmod,
                _index_lookup(_lookup_var(bmod, :hpa_cap), h2_id),
                _lookup_constr(bmod, Symbol("hpa_cap_limit_", h2_id)),
                W, dual_tol)
            b_want |= bw
            b_duals |= bd
        end

        wants_up = (s_duals && b_duals) ? (s_want && b_want) :
                   ((C_old > bind_tol) && (q_s >= C_old - bind_tol) && (q_d >= C_old - bind_tol))
        move = _confirmed_move!(signal["hpa"], h2_id, wants_up, up_confirm, down_confirm)
        moves["hpa"][h2_id] = move
        push!(results["hpa_q_peak"][h2_id], q_peak)
        util = C_old > 0.0 ? (q_avg_mut / C_old) : 0.0
        push!(results["hpa_util_ratio"][h2_id], util)
        idle = _idle_streak(results["hpa_util_ratio"], h2_id, IDLE_UTIL_FRAC)
        C_new = _next_shared_C(C_old, C_phys, move, idle, cfg)
        shared["hpa"][h2_id] = C_new
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
