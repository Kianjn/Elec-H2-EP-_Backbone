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
#   Per-market parameters (current implementation):
#     • elec / elec_GC — inc/dec factor 1.05, ρ_max = 5,000
#     • H2 / EP        — inc/dec factor 1.01, ρ_max = 100
#     • H2_GC          — inc/dec factor 1.05, ρ_max = 100
#       (more conservative in tightly coupled or thin markets to avoid
#        oscillation when capacities/investments are binding).
#
#   Capacity ADMM:
#     Capacity consensus uses a PER-AGENT equality split (x_cap = z_cap) with
#     per-agent ρ_m, λ_m, residuals. The single old "cap" market entry has
#     been removed from this controller. A separate per-agent loop at the
#     END of this function applies the SAME three-regime rule to each
#     capacity-owning agent's ρ_m using its own r_m, s_m, best-residuals
#     and R-history (all stored in ADMM_state["Capacity"]). See
#     DOCUMENTATION.md §5.4 for the full justification.
#
#   Tolerance basis:
#     market_tol is computed from the SAME Boyd-style scaled tolerances used
#     by convergence checks (ε_abs * sqrt(n_slots) + ε_rel * residual_scale),
#     so ρ adaptation stays consistent when nYears/horizon size increases.
#
# ARGUMENTS:
#   ADMM_state — Must contain Residuals["Primal"] and ["Dual"] per market, and
#     ρ[key] as a list (we use the last element and push a new one).
#     For capacity, ADMM_state["Capacity"] holds per-agent state.
#   iter — Current iteration index.
#
# ==============================================================================

function update_rho!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return
    ctrl = get!(ADMM_state, "RhoController", Dict{String,Any}())
    prev_merit = get!(ctrl, "PrevMerit", Dict{String,Float64}())
    last_dir = get!(ctrl, "LastDir", Dict{String,Int}())
    step_scale_map = get!(ctrl, "StepScale", Dict{String,Float64}())
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
        # Use the SAME Boyd-style scaled tolerance basis as ADMM convergence:
        #   eps = eps_abs * sqrt(n_slots) + eps_rel * residual_scale.
        # This keeps rho-adaptation behavior consistent when nYears grows.
        eps_abs = get(ADMM_state, "EpsilonAbs", get(get(ADMM_state, "Tolerance", Dict()), key, 1.0))
        eps_rel = get(ADMM_state, "EpsilonRel", 0.0)
        n_slots = get(ADMM_state, "n_slots", 1)
        sqrt_n = sqrt(max(1, n_slots))
        scale_pr = max(get(ADMM_state["ResidualScale"]["Primal"], key, 1.0), 1.0)
        scale_du = max(get(ADMM_state["ResidualScale"]["Dual"], key, 1.0), 1.0)
        eps_pr = eps_abs * sqrt_n + eps_rel * scale_pr
        eps_du = eps_abs * sqrt_n + eps_rel * scale_du
        market_tol = max(eps_pr, eps_du)
        # When rp and rd differ by more than this factor, we are clearly in
        # regime (1) "normal Boyd updates".
        balance_threshold = 1.2
        # When BOTH rp and rd are below this multiple of tol, we consider
        # the market to be in the near-convergence stability zone and freeze ρ.
        # Increased from 2.0 to 5.0 so we freeze earlier (within ~5×ε) and avoid
        # ρ updates that can kick the algorithm out of a good basin.
        mid_resid_factor = 5.0
        # When BOTH rp and rd are above this multiple of tol, and roughly
        # balanced, we apply the gentle push (regime (2)).
        high_resid_factor = 2.0

        # Window length and tolerance for deciding whether increasing ρ has
        # recently helped or hurt residuals.
        window_len = 10
        improve_tol = 1.02

        # Divergence detection: if R has increased for 3+ consecutive iters,
        # we are overshooting. Force a ρ decrease to break the cycle.
        R_hist = ADMM_state["R_hist"][key]
        diverging = length(R_hist) >= 3 && R_hist[end] > R_hist[end-1] > R_hist[end-2]

        # Decide a direction first (increase/decrease/hold), then pass it
        # through a merit-aware controller that keeps changes which improve the
        # normalized residual score and dampens/reverses harmful moves.
        dir = 0
        if diverging
            # Residuals increasing: use residual-balance direction instead of
            # always decreasing rho (which can be harmful when rp >> rd).
            if rp > balance_threshold * rd
                dir = +1
            elseif rd > balance_threshold * rp
                dir = -1
            else
                dir = -1
            end
        elseif rp > balance_threshold * rd
            # Primal >> dual: increase ρ to enforce feasibility more strongly,
            # but only if doing so has not been worsening residuals over the
            # recent history window.
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
            # Dual >> primal: decrease ρ to avoid overshooting.
            dir = -1
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
            elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                # === Regime (2): far from tol but rp≈rd — gentle push ===
                # We are clearly far from convergence (both residuals large) but
                # the classic Boyd rule sees them as "balanced" and would freeze
                # ρ. To avoid a true stall, we nudge ρ upward only slightly.
                # Skip increase if R has been worsening (prevents improve→stall→diverge).
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
                # Last move likely harmful: damp and reverse if we would repeat it.
                step_scale = max(0.5, 0.8 * step_scale)
                if last != 0 && (dir == 0 || dir == last)
                    dir = -last
                end
            elseif merit < 0.995 * prev
                # Improvement: allow slightly larger steps in the same direction.
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
    # PER-AGENT capacity ρ controller (equality-split formulation)
    #
    # Each capacity-owning agent m has its own ρ_m, its own (r_m, s_m), and
    # its own R-history / best-residual anchors. We apply EXACTLY the same
    # three-regime Boyd rule used for flow markets above, but evaluated per
    # agent. Two implementation choices, both justified in §5.4:
    #
    #   (a) inc/dec factor = 1.05 and ρ_max = 30 (configurable via data.yaml
    #       keys `rho_cap_inc_factor`, `rho_cap_max`). Capacity penalties
    #       interact with binding investment limits; aggressive growth can
    #       cause limit cycles around the kinked capacity bound.
    #   (b) tolerance basis uses sqrt(n_yr) (not sqrt(n_slots)) because
    #       the cap residual has length n_yr (years), not nh·nd·n_yr.
    # ----------------------------------------------------------------------
    cap_state = get(ADMM_state, "Capacity", nothing)
    if cap_state !== nothing
        cap_prev_merit = get!(ctrl, "PrevMeritCap", Dict{String,Float64}())
        cap_last_dir   = get!(ctrl, "LastDirCap",   Dict{String,Int}())
        cap_step_scale = get!(ctrl, "StepScaleCap", Dict{String,Float64}())
        cap_best_pr    = get!(ctrl, "BestPrimalCap", Dict{String,Float64}())
        cap_best_du    = get!(ctrl, "BestDualCap",   Dict{String,Float64}())

        eps_abs = get(ADMM_state, "EpsilonAbs", 1.0)
        eps_rel = get(ADMM_state, "EpsilonRel", 0.0)
        n_yr    = get(ADMM_state, "n_yr", 1)
        sqrt_y  = sqrt(max(1, n_yr))

        # Defaults match the flow-market H2_GC settings (moderate adaptation,
        # tight upper bound) since capacity has the same "thin" character
        # (few decision variables per year, strong CAPEX coupling).
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
            # ρ fixed for one step so the controller has a finite reference.
            if !isfinite(rd)
                push!(cap_state["ρ"][m], ρ)
                push!(cap_state["R_hist"][m], isfinite(rp) ? rp : 0.0)
                continue
            end

            # Anchor best-seen residuals (used by the freeze guard).
            best_pr = get(cap_best_pr, m, Inf)
            best_du = get(cap_best_du, m, Inf)
            if isfinite(rp) && rp < best_pr; best_pr = rp; end
            if isfinite(rd) && rd < best_du; best_du = rd; end
            cap_best_pr[m] = best_pr
            cap_best_du[m] = best_du

            R = (isfinite(rp) ? rp : 0.0) + (isfinite(rd) ? rd : 0.0)
            push!(cap_state["R_hist"][m], R)

            # Frozen: keep ρ_m fixed for the rest of the run.
            if cap_state["ρ_frozen"][m]
                push!(cap_state["ρ"][m], ρ)
                continue
            end

            # Boyd-style scaled tolerances at the AGENT level.
            scale_pr = max(cap_state["ResidualScale_Primal"][m], 1.0)
            scale_du = max(cap_state["ResidualScale_Dual"][m],   1.0)
            eps_pr = eps_abs * sqrt_y + eps_rel * scale_pr
            eps_du = eps_abs * sqrt_y + eps_rel * scale_du
            market_tol = max(eps_pr, eps_du)

            R_hist = cap_state["R_hist"][m]
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
                    R_now  = R_hist[end]
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
                # Balanced residuals.
                if rp <= mid_resid_factor * market_tol && rd <= mid_resid_factor * market_tol
                    # Near-convergence freeze (regime 3).
                    close_to_best = (rp <= improve_tol * best_pr) && (rd <= improve_tol * best_du)
                    if close_to_best
                        cap_state["ρ_frozen"][m] = true
                    end
                elseif rp > high_resid_factor * market_tol && rd > high_resid_factor * market_tol
                    # Gentle push (regime 2) — only if R-window not worsening.
                    can_increase = true
                    if length(R_hist) >= window_len
                        R_now  = R_hist[end]
                        R_past = R_hist[end - window_len + 1]
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
            # Cap-specific gain cap: investment is stiff (kinked CAPEX). Limit
            # the per-step controller gain to 1.0 to avoid limit cycles.
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

    return nothing
end
