# ==============================================================================
# solve_H2_agent_contracts.jl — Re-set objective and solve electrolyzer (contracts)
# ==============================================================================
#
# PURPOSE:
#   For market_exposure_contracts: re-builds the electrolyzer objective with
#   current λ, g_bar, ρ (including per-VRES contract energy and capacity)
#   and calls optimize!. Recomputes CVaR loss with contract cost.
#
# ==============================================================================

function solve_H2_agent_contracts!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict,
                                   ppa_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]
    op_cost = mod.ext[:parameters][:OperationalCost]

    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2  = mod.ext[:parameters][:g_bar_H2]
    ρ_H2      = mod.ext[:parameters][:ρ_H2]
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC     = mod.ext[:parameters][:ρ_H2_GC]
    λ_ppa     = mod.ext[:parameters][:λ_ppa]
    g_bar_ppa = mod.ext[:parameters][:g_bar_ppa]
    ρ_ppa     = mod.ext[:parameters][:ρ_ppa]
    g_bar_ppa_cap = mod.ext[:parameters][:g_bar_ppa_cap]
    ρ_ppa_cap     = mod.ext[:parameters][:ρ_ppa_cap]
    ppa_vres = collect(keys(λ_ppa))

    e_in_pool       = mod.ext[:variables][:e_in_pool]
    g_ppa_from = mod.ext[:variables][:g_ppa_from]
    h2_out          = mod.ext[:variables][:h2_out]
    q_elec_gc       = mod.ext[:variables][:q_elec_gc]
    q_h2gc          = mod.ext[:variables][:q_h2gc]
    cap_H2_y        = mod.ext[:variables][:cap_H2_y]
    ppa_cap    = mod.ext[:variables][:ppa_cap]

    gamma     = get(mod.ext[:parameters], :γ, 1.0)
    F_cap     = get(mod.ext[:parameters], :FixedCost_per_MW_Electrolyzer, 0.0)
    alpha_H2  = mod.ext[:variables][:alpha_H2]
    cvar_H2   = mod.ext[:variables][:CVaR_H2]
    u_H2      = mod.ext[:variables][:u_H2]
    beta_conf = get(mod.ext[:parameters], :β, 0.95)
    P         = mod.ext[:parameters][:P]

    # Recompute per-year loss with per-VRES contract cost; loss_total for CVaR
    loss_H2 = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_H2[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy]       * e_in_pool[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
                + sum(λ_ppa[v][jh, jd, jy] * g_ppa_from[v][jh, jd, jy] for v in ppa_vres)
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
        loss_total[jy] = @expression(mod, loss_H2[jy] + F_cap * cap_H2_y[jy])
    end
    mod.ext[:expressions][:loss_H2] = loss_H2

    obj_ppa = sum(
        sum(ρ_ppa[v]/2 * W[jd, jy] * ((-g_ppa_from[v][jh, jd, jy]) - g_bar_ppa[v][jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + (ρ_ppa_cap[v]/2) * ((-ppa_cap[v]) - g_bar_ppa_cap[v])^2
        for v in ppa_vres
    )

    mod.ext[:objective] = @objective(mod, Min,
        gamma * sum(loss_total[jy] for jy in JY)
        + (1 - gamma) * cvar_H2
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in_pool[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy]         - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + obj_ppa
    )

    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_H2_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_H2_link])

    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_total[jy] - alpha_H2)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
