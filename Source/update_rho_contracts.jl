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
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP", "contract", "contract_cap")
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
        elseif key in ("contract", "contract_cap")
            # Same as H2/EP: moderate adaptation; thin bilateral pool benefits from
            # responsive ρ like other thin markets (H2, EP) rather than very slow 1.01
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 500.0
        else
            inc_factor = 1.01
            dec_factor = 1.0 / 1.01
            ρ_max = 100.0
        end

        market_tol = get(ADMM_state["Tolerance"], key, 1.0)
        balance_threshold = 1.2
        mid_resid_factor = 2.0
        high_resid_factor = 2.0
        window_len = 5
        improve_tol = 1.05

        if rp > balance_threshold * rd
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
    return nothing
end
