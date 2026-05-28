 # ==============================================================================
 # solve_offtaker_agent_contracts.jl — Re-set objective and solve GreenOfftaker
 # ==============================================================================
 #
 # PURPOSE:
 #   Rebuilds the GreenOfftaker objective for market_exposure_contracts.jl with
 #   current λ/g_bar/ρ for standard markets and HPA terms, then optimize!.
 #
 # NOTES:
 #   - Non-GreenOfftaker agents delegate to solve_offtaker_agent!.
 #   - HPA objective terms mirror PPA logic: energy priced by λ_hpa and
 #     capacity aligned via scalar hpa_cap consensus penalties.
 #
 # ==============================================================================

function solve_offtaker_agent_contracts!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict,
                                         hpa_market::Dict)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    if agent_type != "GreenOfftaker"
        return solve_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]

    λ_H2 = mod.ext[:parameters][:λ_H2]
    g_bar_H2 = mod.ext[:parameters][:g_bar_H2]
    ρ_H2 = mod.ext[:parameters][:ρ_H2]
    λ_H2_GC = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC = mod.ext[:parameters][:ρ_H2_GC]
    λ_EP = mod.ext[:parameters][:λ_EP]
    g_bar_EP = mod.ext[:parameters][:g_bar_EP]
    ρ_EP = mod.ext[:parameters][:ρ_EP]

    λ_hpa = mod.ext[:parameters][:λ_hpa]
    g_bar_hpa = mod.ext[:parameters][:g_bar_hpa]
    ρ_hpa = mod.ext[:parameters][:ρ_hpa]
    g_bar_hpa_cap = mod.ext[:parameters][:g_bar_hpa_cap]
    ρ_hpa_cap = mod.ext[:parameters][:ρ_hpa_cap]
    hpa_h2 = collect(keys(λ_hpa))

    h2_in_pool = mod.ext[:variables][:h2_in_pool]
    h2_hpa_from = mod.ext[:variables][:h2_hpa_from]
    q_h2gc = mod.ext[:variables][:q_h2gc]
    ep = mod.ext[:variables][:ep]
    hpa_cap = mod.ext[:variables][:hpa_cap]
    cap_EP_y = mod.ext[:variables][:cap_EP_y]

    gamma_G = get(mod.ext[:parameters], :γ, 1.0)
    proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)
    F_cap = get(mod.ext[:parameters], :FixedCost_per_MW_EP_Out, 0.0)
    alpha_G = mod.ext[:variables][:alpha_GreenOfftaker]
    cvar_G = mod.ext[:variables][:CVaR_GreenOfftaker]
    u_G = mod.ext[:variables][:u_GreenOfftaker]
    beta_conf = get(mod.ext[:parameters], :β, 0.95)
    P = mod.ext[:parameters][:P]

    loss_G = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_G[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_H2[jh, jd, jy] * h2_in_pool[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
                + sum(λ_hpa[v][jh, jd, jy] * h2_hpa_from[v][jh, jd, jy] for v in hpa_h2)
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
        loss_total[jy] = @expression(mod, loss_G[jy] + F_cap * cap_EP_y[jy])
    end
    mod.ext[:expressions][:loss_GreenOfftaker] = loss_G

    # Capacity ADMM equality split (per-agent): same augmented Lagrangian
    # form as the non-contracts solver. See DOCUMENTATION.md §5.4.
    z_cap   = get(mod.ext[:parameters], :z_cap, zeros(length(JY)))
    λ_cap   = get(mod.ext[:parameters], :λ_cap, zeros(length(JY)))
    ρ_cap   = get(mod.ext[:parameters], :ρ_cap, 0.1)
    cap_pen = haskey(mod.ext[:parameters], :z_cap) ?
        sum(λ_cap[jy] * (cap_EP_y[jy] - z_cap[jy]) +
            ρ_cap/2 * (cap_EP_y[jy] - z_cap[jy])^2 for jy in JY) : 0.0
    obj_hpa = sum(
        sum(ρ_hpa[v]/2 * W[jd, jy] * ((-h2_hpa_from[v][jh, jd, jy]) - g_bar_hpa[v][jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + (ρ_hpa_cap[v]/2) * ((-hpa_cap[v]) - g_bar_hpa_cap[v])^2
        for v in hpa_h2
    )
    mod.ext[:objective] = @objective(mod, Min,
        gamma_G * sum(loss_total[jy] for jy in JY)
        + (1 - gamma_G) * cvar_G
        + sum(ρ_H2/2 * W[jd, jy] * ((-h2_in_pool[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + obj_hpa
        + cap_pen
    )

    for jy in JY
        delete(mod, mod.ext[:constraints][:CVaR_Green_shortfall][jy])
    end
    delete(mod, mod.ext[:constraints][:CVaR_Green_link])

    mod.ext[:constraints][:CVaR_Green_shortfall] = @constraint(mod, [jy in JY], u_G[jy] >= loss_total[jy] - alpha_G)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_Green_link] = @constraint(mod,
        cvar_G >= alpha_G + (1 / one_minus_beta) * sum(P[jy] * u_G[jy] for jy in JY)
    )

    optimize!(mod)
    return nothing
end
