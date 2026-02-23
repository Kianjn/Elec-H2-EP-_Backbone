# ==============================================================================
# solve_power_agent.jl — Re-set objective and solve power-sector agent
# ==============================================================================
#
# PURPOSE:
#   Called from ADMM_subroutine after the model's λ, g_bar, and ρ have been
#   updated from the current ADMM state. This function re-builds the objective
#   (min cost - revenue + ADMM penalties) using those updated parameters and
#   then calls optimize!(mod).
#
#   For VRES agents with CVaR, the per-year loss expressions (which depend on
#   λ_elec and λ_elec_GC) must also be recomputed, because JuMP expressions
#   bake in coefficient values at creation time. The CVaR shortfall and linking
#   constraints that reference these loss expressions are therefore deleted and
#   re-added each iteration with the fresh losses.
#
#   For Conventional and Consumer agents no CVaR logic is needed; only the
#   objective is rebuilt.
#
# ARGUMENTS:
#   m — Agent ID (used only for dispatch; parameters are on mod).
#   mod — JuMP model (ext[:parameters] already updated by ADMM_subroutine).
#   elec_market, elec_GC_market — Passed for interface consistency; nAgents used elsewhere.
#
# ==============================================================================

function solve_power_agent!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict)
    # Only the objective is rebuilt each ADMM iteration. Variables and
    # constraints were defined once in build_power_agent! and are invariant;
    # only the prices (lambda), consensus targets (g_bar), and penalty
    # parameter (rho) change between iterations.
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W   = mod.ext[:parameters][:W]
    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    agent_type = mod.ext[:parameters][:Type]

    if agent_type == "VRES"
        # ── VRES parameters and variables ─────────────────────────────────
        gamma      = get(mod.ext[:parameters], :γ, 1.0)   # risk weight (1 = risk-neutral)
        F_cap      = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        MC         = mod.ext[:parameters][:MarginalCost]
        cap_VRES   = mod.ext[:variables][:cap_VRES]
        g          = mod.ext[:variables][:g]

        # CVaR auxiliary variables (created once in build_power_agent!):
        # alpha_VRES = VaR proxy, cvar_VRES = CVaR of loss, u_VRES[jy] = shortfall.
        alpha_VRES = mod.ext[:variables][:alpha_VRES]
        cvar_VRES  = mod.ext[:variables][:CVaR_VRES]
        u_VRES     = mod.ext[:variables][:u_VRES]
        beta_conf  = get(mod.ext[:parameters], :β, 0.95)   # CVaR confidence level
        P          = mod.ext[:parameters][:P]               # scenario probabilities

        # ── Recompute per-year loss expressions with current λ ────────────
        # loss_VRES[jy] = Σ_{h,d} W[d,y]·( MC·g − λ_elec·g − λ_GC·g )
        #   = per-year economic loss (production cost minus market revenues).
        # JuMP expressions bake in coefficient values at creation time, so
        # the loss expressions built in build_power_agent! contain the old λ
        # from the previous (or initial) ADMM iteration. Since λ updates
        # every iteration, we must rebuild these expressions from scratch.
        loss_VRES = Dict{Int,JuMP.AffExpr}()
        for jy in JY
            loss_VRES[jy] = @expression(mod,
                sum(W[jd, jy] * (MC * g[jh, jd, jy]
                    - λ_elec[jh, jd, jy] * g[jh, jd, jy]
                    - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD)
            )
        end
        mod.ext[:expressions][:loss_VRES] = loss_VRES

        # ── Risk-adjusted objective ───────────────────────────────────────
        #   min  γ · ( Σ_y loss_VRES[y] + F_cap · Σ_y cap_VRES[y] )   ← (1)
        #      + (1−γ) · CVaR_VRES                                      ← (2)
        #      + (ρ_elec/2) · Σ W·(g − ḡ_elec)²                        ← (3)
        #      + (ρ_GC /2)  · Σ W·(g − ḡ_GC)²                          ← (4)
        #
        # (1) Expected cost: weighted sum of per-year losses (production
        #     cost minus revenue) plus annualised fixed capacity cost.
        # (2) Tail-risk penalty: CVaR of the loss distribution. When γ=1
        #     (risk-neutral) this term vanishes, recovering the standard
        #     expected-cost objective used in the build function.
        # (3) ADMM augmented-Lagrangian penalty for the electricity market.
        # (4) ADMM augmented-Lagrangian penalty for the elec-GC market.
        mod.ext[:objective] = @objective(mod, Min,
            gamma * (
                sum(loss_VRES[jy] for jy in JY)
                + F_cap * sum(cap_VRES[jy] for jy in JY)
            )
            + (1 - gamma) * cvar_VRES
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec_GC/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

        # ── Delete stale CVaR constraints and re-add with fresh losses ────
        # The shortfall constraints u_VRES[jy] ≥ loss_VRES[jy] − α_VRES
        # and the linking constraint CVaR_VRES ≥ α + (1/(1−β))·Σ P·u
        # both reference the loss expressions. Since those expressions
        # changed (new λ coefficients), we must delete the old constraints
        # and create new ones.
        for jy in JY
            delete(mod, mod.ext[:constraints][:CVaR_VRES_shortfall][jy])
        end
        delete(mod, mod.ext[:constraints][:CVaR_VRES_link])

        # Shortfall constraints: u_VRES[jy] ≥ loss_VRES[jy] − α_VRES.
        mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
            u_VRES[jy] >= loss_VRES[jy] - alpha_VRES
        )
        # CVaR linking: CVaR_VRES ≥ α_VRES + (1/(1−β)) · Σ P[jy]·u_VRES[jy].
        one_minus_beta = max(1e-6, 1.0 - beta_conf)
        mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
            cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY)
        )

    elseif agent_type == "Conventional"
        # Conventional generator objective: same structure as VRES but
        # WITHOUT the green certificate (GC) revenue/penalty terms,
        # since conventional plants do not earn GCs.
        #   min  sum W * (MC*g - lambda_elec*g)           [cost - revenue]
        #      + rho_elec/2 * sum W * (g - g_bar_elec)^2  [ADMM elec penalty]
        g  = mod.ext[:variables][:g]
        MC = mod.ext[:parameters][:MarginalCost]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    elseif agent_type == "Consumer"
        # Consumer objective:
        #   min  sum W * (lambda*d - U(d))                  [cost - utility]
        #      + rho/2 * sum W * (-d - g_bar_elec)^2        [ADMM penalty]
        # where U(d) = A_E*d - B_E/2*d^2 is the quadratic utility function.
        # The net market position is -d (negative because consumer is a buyer),
        # hence the penalty uses (-d - g_bar_elec)^2.
        d   = mod.ext[:variables][:d]
        A_E = mod.ext[:parameters][:A_E]
        B_E = mod.ext[:parameters][:B_E]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (λ_elec[jh, jd, jy] * d[jh, jd, jy] - (A_E * d[jh, jd, jy] - B_E/2 * d[jh, jd, jy]^2)) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * ((-d[jh, jd, jy]) - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    end
    optimize!(mod)
    return nothing
end
