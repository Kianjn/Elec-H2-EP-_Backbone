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
                                     contract_market::Dict)
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
    λ_contract     = mod.ext[:parameters][:λ_contract]
    g_bar_contract = mod.ext[:parameters][:g_bar_contract]
    ρ_contract     = mod.ext[:parameters][:ρ_contract]
    g_bar_contract_cap = mod.ext[:parameters][:g_bar_contract_cap]
    ρ_contract_cap     = mod.ext[:parameters][:ρ_contract_cap]

    gamma     = get(mod.ext[:parameters], :γ, 1.0)
    F_cap     = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
    MC        = mod.ext[:parameters][:MarginalCost]
    cap_VRES  = mod.ext[:variables][:cap_VRES]
    g_EOM     = mod.ext[:variables][:g_EOM]
    g_contract = mod.ext[:variables][:g_contract]
    contract_cap = mod.ext[:variables][:contract_cap]

    alpha_VRES = mod.ext[:variables][:alpha_VRES]
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]
    u_VRES     = mod.ext[:variables][:u_VRES]
    beta_conf  = get(mod.ext[:parameters], :β, 0.95)
    P          = mod.ext[:parameters][:P]

    # Recompute per-year loss with contract revenue
    loss_VRES = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_VRES[jy] = @expression(mod,
            sum(W[jd, jy] * (
                MC * (g_EOM[jh, jd, jy] + g_contract[jh, jd, jy])
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * (g_EOM[jh, jd, jy] + g_contract[jh, jd, jy])
                - λ_contract[jh, jd, jy] * g_contract[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
    end
    mod.ext[:expressions][:loss_VRES] = loss_VRES

    mod.ext[:objective] = @objective(mod, Min,
        gamma * (
            sum(loss_VRES[jy] for jy in JY)
            + F_cap * sum(cap_VRES[jy] for jy in JY)
        )
        + (1 - gamma) * cvar_VRES
        + sum(ρ_elec/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((g_EOM[jh, jd, jy] + g_contract[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_contract/2 * W[jd, jy] * (g_contract[jh, jd, jy] - g_bar_contract[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + (ρ_contract_cap/2) * (contract_cap - g_bar_contract_cap)^2
    )

    # Re-add CVaR constraints with fresh loss expressions
    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_VRES_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_VRES_link])

    mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
        u_VRES[jy] >= loss_VRES[jy] - alpha_VRES)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
        cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

    optimize!(mod)
    return nothing
end
