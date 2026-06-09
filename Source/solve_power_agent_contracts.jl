# ==============================================================================
# solve_power_agent_contracts.jl — Re-set objective and solve power agents (contracts)
# ==============================================================================
#
# PURPOSE:
#   For market_exposure_contracts: re-builds the objective with current λ, g_bar, ρ
#   and calls optimize!. For VRES with contract: includes contract energy and
#   contract capacity terms; recomputes CVaR loss with contract revenue.
#   For Conventional and Consumer: delegates to solve_power_agent!.
#
# ==============================================================================

function solve_power_agent_contracts!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict,
                                     ppa_market::Dict)
    agent_type = mod.ext[:parameters][:Type]

    if agent_type in ("Conventional", "Consumer")
        return solve_power_agent!(m, mod, elec_market, elec_GC_market)
    end

    # ── VRES with contract ──────────────────────────────────────────────────
    if agent_type != "VRES"
        return
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]

    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    λ_ppa     = mod.ext[:parameters][:λ_ppa]
    K_ppa     = get(mod.ext[:parameters], :K_ppa, λ_ppa)
    g_bar_ppa = mod.ext[:parameters][:g_bar_ppa]
    ρ_ppa     = mod.ext[:parameters][:ρ_ppa]
    g_bar_ppa_cap = mod.ext[:parameters][:g_bar_ppa_cap]
    ρ_ppa_cap     = mod.ext[:parameters][:ρ_ppa_cap]

    gamma     = get(mod.ext[:parameters], :γ, 1.0)
    F_cap     = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
    MC        = mod.ext[:parameters][:MarginalCost]
    cap_VRES  = mod.ext[:variables][:cap_VRES]
    g_EOM     = mod.ext[:variables][:g_EOM]
    g_ppa = mod.ext[:variables][:g_ppa]
    ppa_cap = mod.ext[:variables][:ppa_cap]

    alpha_VRES = mod.ext[:variables][:alpha_VRES]
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]
    u_VRES     = mod.ext[:variables][:u_VRES]
    beta_conf  = get(mod.ext[:parameters], :β, 0.95)
    P          = mod.ext[:parameters][:P]

    # Recompute per-year loss with contract revenue; loss_total for CVaR (includes F_cap)
    loss_VRES = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_VRES[jy] = @expression(mod,
            sum(W[jd, jy] * (
                MC * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g_EOM[jh, jd, jy]
                - K_ppa[jh, jd, jy] * g_ppa[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
        loss_total[jy] = @expression(mod, loss_VRES[jy] + F_cap * cap_VRES)
    end
    mod.ext[:expressions][:loss_VRES] = loss_VRES

    z_cap   = get(mod.ext[:parameters], :z_cap, 0.0)
    λ_cap   = get(mod.ext[:parameters], :λ_cap, 0.0)
    ρ_cap   = get(mod.ext[:parameters], :ρ_cap, 0.1)
    cap_pen = haskey(mod.ext[:parameters], :z_cap) ?
        λ_cap * (cap_VRES - z_cap) + ρ_cap/2 * (cap_VRES - z_cap)^2 : 0.0

    mod.ext[:objective] = @objective(mod, Min,
        gamma * (F_cap * cap_VRES + sum(P[jy] * loss_VRES[jy] for jy in JY))
        + (1 - gamma) * cvar_VRES
        + sum(ρ_elec/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_ppa/2 * W[jd, jy] * (g_ppa[jh, jd, jy] - g_bar_ppa[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + (ρ_ppa_cap/2) * (ppa_cap - g_bar_ppa_cap)^2
        + cap_pen
    )

    # Re-add CVaR constraints with fresh loss expressions (use loss_total for γ-invariance when nYears=1)
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_VRES_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_VRES_link])

    mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
        u_VRES[jy] >= loss_total[jy] - alpha_VRES)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
        cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
