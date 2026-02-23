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

        # Per-market parameters
        if key in ("elec", "elec_GC")
            inc_factor = 1.10
            dec_factor = 1.0 / 1.10
            ρ_max = 100_000.0
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
        balance_threshold = 2.0
        # When BOTH rp and rd are below this multiple of tol, we consider
        # the market to be in the near-convergence stability zone and fix ρ
        # (regime (3)).
        mid_resid_factor = 3.0
        # When BOTH rp and rd are above this multiple of tol, and roughly
        # balanced, we apply the gentle push (regime (2)).
        high_resid_factor = 10.0

        # === Regime (1): normal Boyd updates when rp and rd are imbalanced ===
        if rp > balance_threshold * rd
            # Primal >> dual: increase ρ to enforce feasibility more strongly.
            push!(ADMM_state["ρ"][key], min(ρ_max, inc_factor * ρ))
        elseif rd > balance_threshold * rp
            # Dual >> primal: decrease ρ to avoid overshooting.
            push!(ADMM_state["ρ"][key], max(1e-4, dec_factor * ρ))
        else
            # rp and rd are of comparable magnitude.
            if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                # === Regime (3): near-convergence — fix ρ for stability ===
                # Once BOTH residuals are within a modest multiple of tol, we stop
                # adapting ρ for this market. This prevents small oscillations in
                # tight, kinked problems (e.g. with investment and capacity) from
                # being amplified by continual ρ changes.
                push!(ADMM_state["ρ"][key], ρ)
            elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                # === Regime (2): far from tol but rp≈rd — gentle push ===
                # We are clearly far from convergence (both residuals large) but
                # the classic Boyd rule sees them as "balanced" and would freeze
                # ρ. To avoid a true stall, we nudge ρ upward only slightly.
                mild_inc = key in ("H2", "EP") ? 1.01 : 1.02
                push!(ADMM_state["ρ"][key], min(ρ_max, mild_inc * ρ))
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
