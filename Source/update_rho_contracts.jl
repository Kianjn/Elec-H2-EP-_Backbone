# ==============================================================================
# update_rho_contracts.jl — Adaptive penalty update (with contract markets)
# ==============================================================================
#
# PURPOSE:
#   Extends update_rho! for market_exposure_contracts. Applies the same
#   three-regime Boyd rule to the five standard markets plus the contract
#   energy and contract capacity markets.
#
#   Contract markets use conservative parameters (inc_factor 1.01, ρ_max 100)
#   to avoid oscillation in the thin bilateral contract pool.
#
# ==============================================================================

function update_rho_contracts!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return
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

        market_tol = get(ADMM_state["Tolerance"], key, 1.0)
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

    # Per-VRES PPA and ppa_cap (from ADMM_state["ppa"])
    haskey(ADMM_state, "ppa") || return
    C = ADMM_state["ppa"]
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
            market_tol = get(C["Tolerance"], vres_id, 1.0)
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
            market_tol = get(C["Tolerance_cap"], vres_id, 1.0)
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
    return nothing
end
