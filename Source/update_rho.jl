# ==============================================================================
# update_rho.jl — Adaptive penalty parameter update (Boyd et al.)
# ==============================================================================
#
# PURPOSE:
#   After primal and dual residuals are computed for the current iteration,
#   adjust ρ per market to balance primal and dual progress: if primal residual
#   is much larger than dual (rp > 2*rd), increase ρ (up to 100000) to penalize
#   feasibility more; if dual is much larger (rd > 2*rp), decrease ρ to avoid
#   overshooting. Otherwise leave ρ unchanged. The new ρ is pushed onto
#   ADMM_state["ρ"][key] so the next iteration uses it for price update and
#   agent penalties.
#
# ARGUMENTS:
#   ADMM_state — Must contain Residuals["Primal"] and ["Dual"] per market, and
#     ρ[key] as a list (we use the last element and push a new one).
#   iter — Current iteration index; we run every iteration (mod(iter,1)==0 always true).
#
# ==============================================================================

function update_rho!(ADMM_state::Dict, iter::Int)
    # mod(iter, 1) == 0 is always true (every integer mod 1 = 0).
    # It is a placeholder so the update frequency can be changed easily,
    # e.g. mod(iter, 10) == 0 would update rho every 10 iterations.
    mod(iter, 1) == 0 || return
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        isempty(ADMM_state["Residuals"]["Primal"][key]) && continue
        isempty(ADMM_state["Residuals"]["Dual"][key]) && continue
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        ρ = ADMM_state["ρ"][key][end]
        # ----- Market-specific adaptation factors -----
        # H2 / H2_GC / EP markets: gentler factors (1.01 increase,
        # 1/1.01 decrease) and a low rho_max = 1.0. These markets are
        # tightly coupled through the hydrogen value chain, so aggressive
        # rho changes cause large oscillations and prevent convergence.
        # The low cap prevents the penalty from dominating the economic
        # objective.
        #
        # elec / elec_GC markets: standard factors (1.10 increase,
        # 1/1.10 decrease) and a high rho_max = 100000. These markets
        # are less coupled and converge faster, so more aggressive
        # adaptation is safe.
        if key in ("H2", "H2_GC", "EP")
            inc_factor = 1.01
            dec_factor = 1.0 / 1.01
            ρ_max = 1.0
        else
            inc_factor = 1.10
            dec_factor = 1.0 / 1.10
            ρ_max = 100_000.0
        end
        # Boyd rule: balance primal vs dual progress.
        #   primal >> dual  =>  rho too small (agents not penalized enough
        #                       for infeasibility), so increase rho.
        #   dual >> primal  =>  rho too large (agents over-penalized,
        #                       causing oscillation), so decrease rho.
        #   otherwise       =>  rho is balanced, keep it unchanged.
        if rp > 2 * rd
            push!(ADMM_state["ρ"][key], min(ρ_max, inc_factor * ρ))
        elseif rd > 2 * rp
            push!(ADMM_state["ρ"][key], dec_factor * ρ)
        else
            push!(ADMM_state["ρ"][key], ρ)
        end
    end
    return nothing
end
