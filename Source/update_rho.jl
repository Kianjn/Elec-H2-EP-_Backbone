# ==============================================================================
# update_rho.jl — Adaptive penalty parameter update (Boyd et al.)
# ==============================================================================
#
# PURPOSE:
#   After primal and dual residuals are computed for the current iteration,
#   adapt the ADMM penalty ρ per market in THREE regimes:
#
#   (1) Normal Boyd updates (coarse balancing of primal vs dual):
#       - If rp >> rd, increase ρ (feasibility too loose).
#       - If rd >> rp, decrease ρ (steps too aggressive).
#       This is the classic Boyd rule and dominates when rp and rd differ
#       by more than a factor 'balance_threshold' (≈2×).
#
#   (2) Gentle push far from tolerance (anti-stall mechanism):
#       - If rp ≈ rd but BOTH are still much larger than the market's
#         tolerance (rp, rd > high_resid_factor * tol), slowly increase ρ.
#       - This fixes the situation where ρ stops adapting even though both
#         residuals are large and similar (rp ≈ rd >> tol), which previously
#         caused ADMM to "stall" far from convergence.
#
#   (3) Fixed-ρ near convergence (stability zone):
#       - If rp ≈ rd and BOTH are within a modest band around tolerance
#         (rp, rd ≤ mid_resid_factor * tol), we STOP adapting ρ for that
#         market and keep it fixed.
#       - This prevents the gentle push / Boyd updates from continually
#         perturbing ρ once we are already in the near-solution region,
#         which is important now that endogenous investment and tight
#         capacity constraints make the problem more kinked and prone to
#         limit cycles around the optimum.
#
#   Per-market parameters:
#     • elec / elec_GC — inc/dec factor 1.10, ρ_max = 100,000
#     • H2 / H2_GC / EP — inc/dec factor 1.01 (or 1.05 for H2_GC), ρ_max = 100
#       (more conservative for tightly price-coupled markets to avoid
#        oscillation when capacities/investments are binding).
#
# ARGUMENTS:
#   ADMM_state — Must contain Residuals["Primal"] and ["Dual"] per market, and
#     ρ[key] as a list (we use the last element and push a new one).
#   iter — Current iteration index.
#
# ==============================================================================

function update_rho!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        isempty(ADMM_state["Residuals"]["Primal"][key]) && continue
        isempty(ADMM_state["Residuals"]["Dual"][key]) && continue
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        ρ  = ADMM_state["ρ"][key][end]

        # Update best-seen residuals (hysteresis anchor) and residual history.
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

        # If ρ has been frozen for this market, keep it fixed forever.
        if ADMM_state["ρ_frozen"][key]
            push!(ADMM_state["ρ"][key], ρ)
            continue
        end

        # Per-market parameters
        if key in ("elec", "elec_GC")
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 5_000.0
        elseif key == "H2_GC"
            # H2_GC is hourly but still a thin certificate market.
            # Moderate adaptation avoids destabilizing the tightly-coupled
            # electrolyzer (which also participates in elec, elec_GC, H2).
            inc_factor = 1.05
            dec_factor = 1.0 / 1.05
            ρ_max = 100.0
        else  # H2, EP
            inc_factor = 1.01
            dec_factor = 1.0 / 1.01
            ρ_max = 100.0
        end

        # Per-market convergence scale and thresholds.
        market_tol = get(ADMM_state["Tolerance"], key, 1.0)
        # When rp and rd differ by more than this factor, we are clearly in
        # regime (1) "normal Boyd updates".
        balance_threshold = 1.2
        # When BOTH rp and rd are below this multiple of tol, we consider
        # the market to be in the near-convergence stability zone.
        mid_resid_factor = 2.0
        # When BOTH rp and rd are above this multiple of tol, and roughly
        # balanced, we apply the gentle push (regime (2)).
        high_resid_factor = 2.0

        # Window length and tolerance for deciding whether increasing ρ has
        # recently helped or hurt residuals.
        window_len = 5
        improve_tol = 1.05

        # === Regime (1): normal Boyd updates when rp and rd are imbalanced ===
        if rp > balance_threshold * rd
            # Primal >> dual: increase ρ to enforce feasibility more strongly,
            # but only if doing so has not been worsening residuals over the
            # recent history window.
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
            # Dual >> primal: decrease ρ to avoid overshooting.
            push!(ADMM_state["ρ"][key], max(1e-4, dec_factor * ρ))
        else
            # rp and rd are of comparable magnitude.
            if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                # === Regime (3): near-convergence — fix ρ for stability ===
                # Once BOTH residuals are within a modest multiple of tol and
                # close to the best residuals observed so far, we freeze ρ for
                # this market permanently. This hysteresis prevents later
                # updates from kicking the algorithm out of a good basin.
                close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                if close_to_best
                    ADMM_state["ρ_frozen"][key] = true
                end
                push!(ADMM_state["ρ"][key], ρ)
            elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                # === Regime (2): far from tol but rp≈rd — gentle push ===
                # We are clearly far from convergence (both residuals large) but
                # the classic Boyd rule sees them as "balanced" and would freeze
                # ρ. To avoid a true stall, we nudge ρ upward only slightly.
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
                # Intermediate band: rp≈rd but not too far from tol and not yet
                # clearly in the near-convergence zone. Keep ρ unchanged so that
                # the algorithm behaves like fixed-ρ ADMM while residuals
                # naturally decay.
                push!(ADMM_state["ρ"][key], ρ)
            end
        end
    end
    return nothing
end
