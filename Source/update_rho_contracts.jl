# ==============================================================================
# update_rho_contracts.jl — Adaptive penalty update (with contract markets)
# ==============================================================================
#
# PURPOSE:
#   Extends update_rho! for market_exposure_contracts. Applies the same
#   three-regime Boyd rule to the five standard markets plus the contract
#   energy and contract capacity markets.
#
#   Contract markets use conservative parameters (inc_factor 1.05, ρ_max 500)
#   and the same history-aware safeguards as base ADMM.
#
#   Tolerance basis:
#     For the five standard markets, market_tol uses the SAME Boyd-style scaled
#     tolerance basis as convergence checks:
#       ε_abs * sqrt(n_slots) + ε_rel * residual_scale.
#     For scalar contract-capacity consensuses, sqrt(1) is used.
#
# ==============================================================================

function update_rho_contracts!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return
    ctrl = get!(ADMM_state, "RhoControllerContracts", Dict{String,Any}())
    prev_merit = get!(ctrl, "PrevMerit", Dict{String,Float64}())
    last_dir = get!(ctrl, "LastDir", Dict{String,Int}())
    step_scale_map = get!(ctrl, "StepScale", Dict{String,Float64}())
    eps_abs = get(ADMM_state, "EpsilonAbs", 1.0)
    eps_rel = get(ADMM_state, "EpsilonRel", 0.0)
    n_slots = get(ADMM_state, "n_slots", 1)
    sqrt_n = sqrt(max(1, n_slots))
    # Capacity consensus is handled separately at the end of this function by
    # the PER-AGENT controller; see DOCUMENTATION.md §5.4 for justification.
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        isempty(ADMM_state["Residuals"]["Primal"][key]) && continue
        isempty(ADMM_state["Residuals"]["Dual"][key]) && continue
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        ρ  = ADMM_state["ρ"][key][end]

        best_pr = ADMM_state["BestResidual"]["Primal"][key]
        best_du = ADMM_state["BestResidual"]["Dual"][key]
        if rp < best_pr
            ADMM_state["BestResidual"]["Primal"][key] = rp
            best_pr = rp
        end
        if rd < best_du
            ADMM_state["BestResidual"]["Dual"][key] = rd
            best_du = rd
        end
        R = rp + rd
        push!(ADMM_state["R_hist"][key], R)

        if ADMM_state["ρ_frozen"][key]
            push!(ADMM_state["ρ"][key], ρ)
            continue
        end

        if key in ("elec", "elec_GC")
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 5_000.0
        elseif key == "H2_GC"
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 100.0
        else
            inc_factor = 1.01
            dec_factor = 1.0 / 1.01
            ρ_max = 100.0
        end

        scale_pr = max(get(ADMM_state["ResidualScale"]["Primal"], key, 1.0), 1.0)
        scale_du = max(get(ADMM_state["ResidualScale"]["Dual"], key, 1.0), 1.0)
        eps_pr = eps_abs * sqrt_n + eps_rel * scale_pr
        eps_du = eps_abs * sqrt_n + eps_rel * scale_du
        market_tol = max(eps_pr, eps_du)
        balance_threshold = 1.2
        mid_resid_factor = 5.0
        high_resid_factor = 2.0
        window_len = 10
        improve_tol = 1.02

        R_hist = ADMM_state["R_hist"][key]
        diverging = length(R_hist) >= 3 && R_hist[end] > R_hist[end-1] > R_hist[end-2]

        dir = 0
        if diverging
            if rp > balance_threshold * rd
                dir = +1
            elseif rd > balance_threshold * rp
                dir = -1
            else
                dir = -1
            end
        elseif rp > balance_threshold * rd
            can_increase = true
            if length(R_hist) >= window_len
                R_now = R_hist[end]
                R_past = R_hist[end - window_len + 1]
                if R_now > improve_tol * R_past
                    can_increase = false
                end
            end
            if can_increase
                dir = +1
            end
        elseif rd > balance_threshold * rp
            dir = -1
        else
            if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                if close_to_best
                    ADMM_state["ρ_frozen"][key] = true
                end
            elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                mild_inc = 1.01
                can_increase = true
                R_hist = ADMM_state["R_hist"][key]
                if length(R_hist) >= window_len
                    R_now = R_hist[end]
                    R_past = R_hist[end - window_len + 1]
                    if R_now > improve_tol * R_past
                        can_increase = false
                    end
                end
                if can_increase
                    dir = +1
                end
            end
        end

        merit = max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))
        prev = get(prev_merit, key, Inf)
        last = get(last_dir, key, 0)
        step_scale = get(step_scale_map, key, 1.0)
        forced_backoff = false
        if isfinite(prev)
            if merit > 1.01 * prev
                step_scale = max(0.5, 0.8 * step_scale)
                if last != 0 && (dir == 0 || dir == last)
                    dir = -last
                end
            elseif merit < 0.995 * prev
                if dir != 0 && dir == last
                    step_scale = min(1.5, 1.05 * step_scale)
                else
                    step_scale = min(1.5, 1.02 * step_scale)
                end
            end
        end
        if dir > 0
            eff_inc = 1.0 + (inc_factor - 1.0) * step_scale
            push!(ADMM_state["ρ"][key], min(ρ_max, eff_inc * ρ))
        elseif dir < 0
            eff_dec = 1.0 - (1.0 - dec_factor) * step_scale
            push!(ADMM_state["ρ"][key], max(1e-4, eff_dec * ρ))
        else
            push!(ADMM_state["ρ"][key], ρ)
        end
        prev_merit[key] = merit
        last_dir[key] = dir
        step_scale_map[key] = step_scale
    end

    # ----------------------------------------------------------------------
    # PER-AGENT capacity ρ controller — identical structure to update_rho.jl
    # (same three-regime Boyd rule, same guards). Contract markets handled
    # in the following block; capacity here is the pool-level cap consensus.
    # See update_rho.jl and DOCUMENTATION.md §5.4 for full justification.
    # ----------------------------------------------------------------------
    cap_state = get(ADMM_state, "Capacity", nothing)
    if cap_state !== nothing
        cap_prev_merit = get!(ctrl, "PrevMeritCap", Dict{String,Float64}())
        cap_last_dir   = get!(ctrl, "LastDirCap",   Dict{String,Int}())
        cap_step_scale = get!(ctrl, "StepScaleCap", Dict{String,Float64}())
        cap_best_pr    = get!(ctrl, "BestPrimalCap", Dict{String,Float64}())
        cap_best_du    = get!(ctrl, "BestDualCap",   Dict{String,Float64}())

        n_yr   = get(ADMM_state, "n_yr", 1)
        sqrt_y = sqrt(max(1, n_yr))

        inc_factor_cap = get(ADMM_state, "rho_cap_inc_factor", 1.05)
        dec_factor_cap = 1.0 / inc_factor_cap
        ρ_max_cap      = get(ADMM_state, "rho_cap_max", 30.0)
        ρ_min_cap      = 0.05

        balance_threshold = 1.2
        mid_resid_factor  = 5.0
        high_resid_factor = 2.0
        window_len = 10
        improve_tol = 1.02

        for m in get(cap_state, "agents", String[])
            isempty(cap_state["Primal"][m]) && continue
            isempty(cap_state["Dual"][m])   && continue
            rp = cap_state["Primal"][m][end]
            rd = cap_state["Dual"][m][end]
            ρ  = cap_state["ρ"][m][end]

            # First iteration: dual residual is undefined (no z^{k-1}); keep
            # ρ fixed for one step.
            if !isfinite(rd)
                push!(cap_state["ρ"][m], ρ)
                push!(cap_state["R_hist"][m], isfinite(rp) ? rp : 0.0)
                continue
            end

            best_pr = get(cap_best_pr, m, Inf)
            best_du = get(cap_best_du, m, Inf)
            if isfinite(rp) && rp < best_pr; best_pr = rp; end
            if isfinite(rd) && rd < best_du; best_du = rd; end
            cap_best_pr[m] = best_pr
            cap_best_du[m] = best_du

            R = (isfinite(rp) ? rp : 0.0) + (isfinite(rd) ? rd : 0.0)
            push!(cap_state["R_hist"][m], R)

            if cap_state["ρ_frozen"][m]
                push!(cap_state["ρ"][m], ρ)
                continue
            end

            scale_pr = max(cap_state["ResidualScale_Primal"][m], 1.0)
            scale_du = max(cap_state["ResidualScale_Dual"][m],   1.0)
            eps_pr = eps_abs * sqrt_y + eps_rel * scale_pr
            eps_du = eps_abs * sqrt_y + eps_rel * scale_du
            market_tol = max(eps_pr, eps_du)

            R_hist_m = cap_state["R_hist"][m]
            diverging = length(R_hist_m) >= 3 && R_hist_m[end] > R_hist_m[end-1] > R_hist_m[end-2]

            dir = 0
            if diverging
                if rp > balance_threshold * rd
                    dir = +1
                elseif rd > balance_threshold * rp
                    dir = -1
                else
                    dir = -1
                end
            elseif rp > balance_threshold * rd
                can_increase = true
                if length(R_hist_m) >= window_len
                    R_now  = R_hist_m[end]
                    R_past = R_hist_m[end - window_len + 1]
                    if R_now > improve_tol * R_past
                        can_increase = false
                    end
                end
                if can_increase
                    dir = +1
                end
            elseif rd > balance_threshold * rp
                dir = -1
            else
                if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                    close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                    if close_to_best
                        cap_state["ρ_frozen"][m] = true
                    end
                elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                    can_increase = true
                    if length(R_hist_m) >= window_len
                        R_now  = R_hist_m[end]
                        R_past = R_hist_m[end - window_len + 1]
                        if R_now > improve_tol * R_past
                            can_increase = false
                        end
                    end
                    if can_increase && !diverging
                        dir = +1
                    end
                end
            end

            merit = max(rp / max(eps_pr, 1e-9), isfinite(rd) ? rd / max(eps_du, 1e-9) : 0.0)
            prev = get(cap_prev_merit, m, Inf)
            last = get(cap_last_dir, m, 0)
            step_scale = get(cap_step_scale, m, 1.0)
            if isfinite(prev)
                if merit > 1.01 * prev
                    step_scale = max(0.5, 0.8 * step_scale)
                    if last != 0 && (dir == 0 || dir == last)
                        dir = -last
                    end
                elseif merit < 0.995 * prev
                    if dir != 0 && dir == last
                        step_scale = min(1.5, 1.05 * step_scale)
                    else
                        step_scale = min(1.5, 1.02 * step_scale)
                    end
                end
            end
            # Cap-specific gain cap (kinked CAPEX investment): see update_rho.jl
            step_scale = min(step_scale, 1.0)

            if dir > 0
                eff_inc = 1.0 + (inc_factor_cap - 1.0) * step_scale
                push!(cap_state["ρ"][m], min(ρ_max_cap, eff_inc * ρ))
            elseif dir < 0
                eff_dec = 1.0 - (1.0 - dec_factor_cap) * step_scale
                push!(cap_state["ρ"][m], max(ρ_min_cap, eff_dec * ρ))
            else
                push!(cap_state["ρ"][m], ρ)
            end
            cap_prev_merit[m] = merit
            cap_last_dir[m]   = dir
            cap_step_scale[m] = step_scale
        end
    end

    # Per-submarket contract pools: PPA and HPA
    for contract_key in ("ppa", "hpa")
        haskey(ADMM_state, contract_key) || continue
        C = ADMM_state[contract_key]
        for vres_id in keys(C["ρ"])
        isempty(C["Primal"][vres_id]) && continue
        isempty(C["Dual"][vres_id]) && continue
        rp = C["Primal"][vres_id][end]
        rd = C["Dual"][vres_id][end]
        ρ  = C["ρ"][vres_id][end]

        best_pr = C["BestPrimal"][vres_id]
        best_du = C["BestDual"][vres_id]
        if rp < best_pr
            C["BestPrimal"][vres_id] = rp
            best_pr = rp
        end
        if rd < best_du
            C["BestDual"][vres_id] = rd
            best_du = rd
        end
        R = rp + rd
        push!(C["R_hist"][vres_id], R)

        if C["ρ_frozen"][vres_id]
            push!(C["ρ"][vres_id], ρ)
        else
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 500.0
            eps_pr = eps_abs * sqrt_n + eps_rel * max(get(C["ResidualScale_Primal"], vres_id, 1.0), 1.0)
            eps_du = eps_abs * sqrt_n + eps_rel * max(get(C["ResidualScale_Dual"], vres_id, 1.0), 1.0)
            market_tol = max(eps_pr, eps_du)
            balance_threshold = 1.2
            mid_resid_factor = 5.0
            high_resid_factor = 2.0
            window_len = 10
            improve_tol = 1.02

            R_hist = C["R_hist"][vres_id]
            diverging = length(R_hist) >= 3 && R_hist[end] > R_hist[end-1] > R_hist[end-2]

            if diverging
                push!(C["ρ"][vres_id], max(1e-4, dec_factor * ρ))
            elseif rp > balance_threshold * rd
                can_increase = true
                R_hist = C["R_hist"][vres_id]
                if length(R_hist) >= window_len
                    R_now = R_hist[end]
                    R_past = R_hist[end - window_len + 1]
                    if R_now > improve_tol * R_past
                        can_increase = false
                    end
                end
                if can_increase
                    push!(C["ρ"][vres_id], min(ρ_max, inc_factor * ρ))
                else
                    push!(C["ρ"][vres_id], ρ)
                end
            elseif rd > balance_threshold * rp
                push!(C["ρ"][vres_id], max(1e-4, dec_factor * ρ))
            else
                if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                    close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                    if close_to_best
                        C["ρ_frozen"][vres_id] = true
                    end
                    push!(C["ρ"][vres_id], ρ)
            elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                mild_inc = 1.01
                can_increase = true
                if length(R_hist) >= window_len
                    R_now = R_hist[end]
                    R_past = R_hist[end - window_len + 1]
                    if R_now > improve_tol * R_past
                        can_increase = false
                    end
                end
                if can_increase && !diverging
                    push!(C["ρ"][vres_id], min(ρ_max, mild_inc * ρ))
                else
                    push!(C["ρ"][vres_id], ρ)
                end
            else
                push!(C["ρ"][vres_id], ρ)
            end
            end
        end

        # contract_cap
        isempty(C["Primal_cap"][vres_id]) && continue
        isempty(C["Dual_cap"][vres_id]) && continue
        rp = C["Primal_cap"][vres_id][end]
        rd = C["Dual_cap"][vres_id][end]
        ρ  = C["ρ_cap"][vres_id][end]
        best_pr = C["BestPrimal_cap"][vres_id]
        best_du = C["BestDual_cap"][vres_id]
        if rp < best_pr; C["BestPrimal_cap"][vres_id] = rp; best_pr = rp; end
        if rd < best_du; C["BestDual_cap"][vres_id] = rd; best_du = rd; end
        R = rp + rd
        push!(C["R_hist_cap"][vres_id], R)
        if C["ρ_frozen_cap"][vres_id]
            push!(C["ρ_cap"][vres_id], ρ)
        else
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 500.0
            eps_pr = eps_abs * 1.0 + eps_rel * max(get(C["ResidualScale_Primal_cap"], vres_id, 1.0), 1.0)
            eps_du = eps_abs * 1.0 + eps_rel * max(get(C["ResidualScale_Dual_cap"], vres_id, 1.0), 1.0)
            market_tol = max(eps_pr, eps_du)
            balance_threshold = 1.2
            mid_resid_factor = 5.0
            high_resid_factor = 2.0
            window_len = 10
            improve_tol = 1.02

            R_hist_cap = C["R_hist_cap"][vres_id]
            diverging_cap = length(R_hist_cap) >= 3 && R_hist_cap[end] > R_hist_cap[end-1] > R_hist_cap[end-2]

            if diverging_cap
                push!(C["ρ_cap"][vres_id], max(1e-4, dec_factor * ρ))
            elseif rp > balance_threshold * rd
                can_increase = true
                if length(R_hist_cap) >= window_len
                    R_now = R_hist_cap[end]
                    R_past = R_hist_cap[end - window_len + 1]
                    if R_now > improve_tol * R_past
                        can_increase = false
                    end
                end
                push!(C["ρ_cap"][vres_id], can_increase ? min(ρ_max, inc_factor * ρ) : ρ)
            elseif rd > balance_threshold * rp
                push!(C["ρ_cap"][vres_id], max(1e-4, dec_factor * ρ))
            else
                if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                    if (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                        C["ρ_frozen_cap"][vres_id] = true
                    end
                    push!(C["ρ_cap"][vres_id], ρ)
                elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                    can_increase = true
                    if length(R_hist_cap) >= window_len
                        R_now = R_hist_cap[end]
                        R_past = R_hist_cap[end - window_len + 1]
                        if R_now > improve_tol * R_past
                            can_increase = false
                        end
                    end
                    push!(C["ρ_cap"][vres_id], can_increase && !diverging_cap ? min(ρ_max, 1.01 * ρ) : ρ)
                else
                    push!(C["ρ_cap"][vres_id], ρ)
                end
            end
        end
        end
    end
    return nothing
end
