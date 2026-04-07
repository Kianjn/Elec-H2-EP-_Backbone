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
    eps_abs = get(ADMM_state, "EpsilonAbs", 1.0)
    eps_rel = get(ADMM_state, "EpsilonRel", 0.0)
    n_slots = get(ADMM_state, "n_slots", 1)
    sqrt_n = sqrt(max(1, n_slots))
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP", "cap")
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
        elseif key == "cap"
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

        if diverging
            push!(ADMM_state["ρ"][key], max(1e-4, dec_factor * ρ))
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
                push!(ADMM_state["ρ"][key], min(ρ_max, inc_factor * ρ))
            else
                push!(ADMM_state["ρ"][key], ρ)
            end
        elseif rd > balance_threshold * rp
            push!(ADMM_state["ρ"][key], max(1e-4, dec_factor * ρ))
        else
            if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                if close_to_best
                    ADMM_state["ρ_frozen"][key] = true
                end
                push!(ADMM_state["ρ"][key], ρ)
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
                    push!(ADMM_state["ρ"][key], min(ρ_max, mild_inc * ρ))
                else
                    push!(ADMM_state["ρ"][key], ρ)
                end
            else
                push!(ADMM_state["ρ"][key], ρ)
            end
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
