# ==============================================================================
# solve_power_agent.jl — Re-set objective and solve power-sector agent
# ==============================================================================
#
# PURPOSE:
#   Called from ADMM_subroutine after the model's λ, g_bar, and ρ have been
#   updated from the current ADMM state. This function re-builds the objective
#   (min cost - revenue + ADMM penalties) using those updated parameters and
#   then calls optimize!(mod). Variables and constraints are unchanged from
#   build_power_agent!; only the objective coefficients change each iteration.
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
        # VRES objective:
        #   min  sum W * (MC*g - lambda_elec*g - lambda_GC*g)     [cost - revenue]
        #      + rho_elec/2 * sum W * (g - g_bar_elec)^2          [ADMM elec penalty]
        #      + rho_GC/2   * sum W * (g - g_bar_elec_GC)^2       [ADMM GC penalty]
        # VRES earns from both the electricity AND green certificate markets,
        # so both lambda terms appear with negative sign (revenue).
        g  = mod.ext[:variables][:g]
        MC = mod.ext[:parameters][:MarginalCost]
        # Annualised fixed investment cost per MW-year; capacity is year-specific.
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
        cap_VRES = mod.ext[:variables][:cap_VRES]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * g[jh, jd, jy] - λ_elec[jh, jd, jy] * g[jh, jd, jy] - λ_elec_GC[jh, jd, jy] * g[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_elec_GC/2 * W[jd, jy] * (g[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            # Fixed annualised investment cost, summed over model years (no W weighting).
            + F_cap * sum(cap_VRES[jy] for jy in JY))
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
