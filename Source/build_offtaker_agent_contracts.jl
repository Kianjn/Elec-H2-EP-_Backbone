 # ==============================================================================
 # build_offtaker_agent_contracts.jl — GreenOfftaker with HPAs (per-GreenProducer)
 # ==============================================================================
 #
 # PURPOSE:
 #   Builds the GreenOfftaker model for market_exposure_contracts.jl with HPA
 #   buy-side variables in addition to the standard H2/H2_GC/EP markets.
 #
 # HPA (pay-as-produced, bundled H2 + H2_GC equivalent):
 #   - GreenOfftaker receives h2_hpa_from[h2_id] from each GreenProducer h2_id.
#   - Payment is K_hpa[h2_id] × h2_hpa_from[h2_id].
 #   - Contract flow is capped by hpa_cap[h2_id].
 #   - HPA-delivered H2 contributes to green-backing (bundled certificate logic).
 #
 # NOTE:
 #   Non-GreenOfftaker agents delegate to build_offtaker_agent! unchanged.
 #
 # ==============================================================================

function build_offtaker_agent_contracts!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict,
                                         hpa_market::Dict)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    if agent_type != "GreenOfftaker"
        return build_offtaker_agent!(m, mod, EP_market, H2_market, H2_GC_market)
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]

    gamma_GC = get(mod.ext[:parameters], :gamma_GC, 0.42)
    cap_ep_initial = mod.ext[:parameters][:Capacity_EP_Out]
    alpha = get(mod.ext[:parameters], :Alpha, 1.0)
    proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)
    F_cap = get(mod.ext[:parameters], :FixedCost_per_MW_EP_Out, 0.0)
    gamma = get(mod.ext[:parameters], :γ, 1.0)
    beta_conf = get(mod.ext[:parameters], :β, 0.95)
    P = mod.ext[:parameters][:P]

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
    K_hpa = get(mod.ext[:parameters], :K_hpa, λ_hpa)
    g_bar_hpa = mod.ext[:parameters][:g_bar_hpa]
    ρ_hpa = mod.ext[:parameters][:ρ_hpa]
    g_bar_hpa_cap = mod.ext[:parameters][:g_bar_hpa_cap]
    ρ_hpa_cap = mod.ext[:parameters][:ρ_hpa_cap]
    hpa_h2 = collect(keys(λ_hpa))

    h2_in_pool = mod.ext[:variables][:h2_in_pool] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_in_pool")
    h2_hpa_from = mod.ext[:variables][:h2_hpa_from] = Dict(
        v => @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_hpa_$(v)")
        for v in hpa_h2
    )
    hpa_cap = mod.ext[:variables][:hpa_cap] = Dict(
        v => @variable(mod, lower_bound = 0, base_name = "hpa_cap_$(v)")
        for v in hpa_h2
    )
    q_h2gc = mod.ext[:variables][:q_h2gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC")
    ep = mod.ext[:variables][:ep] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep")

    cap_EP_y = mod.ext[:variables][:cap_EP_y] = @variable(mod, lower_bound = 0, base_name = "cap_EP")
    inv_EP = mod.ext[:variables][:inv_EP] = @variable(mod, lower_bound = 0, base_name = "inv_EP")
    mod.ext[:constraints][:cap_EP_init] = @constraint(mod, cap_EP_y == cap_ep_initial + inv_EP)

    mod.ext[:expressions][:g_net_H2] = @expression(mod, -h2_in_pool)
    mod.ext[:expressions][:g_net_H2_GC] = @expression(mod, -q_h2gc)
    mod.ext[:expressions][:g_net_EP] = @expression(mod, ep)
    mod.ext[:expressions][:g_net_hpa_from] = h2_hpa_from

    h2_hpa_total = sum(h2_hpa_from[v] for v in hpa_h2)
    mod.ext[:constraints][:ep_from_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        ep[jh, jd, jy] == alpha * (h2_in_pool[jh, jd, jy] + sum(h2_hpa_from[v][jh, jd, jy] for v in hpa_h2)))
    mod.ext[:constraints][:cap_ep] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_EP_y)
    mod.ext[:constraints][:gc_cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        q_h2gc[jh, jd, jy] <= h2_in_pool[jh, jd, jy] + sum(h2_hpa_from[v][jh, jd, jy] for v in hpa_h2))
    mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD)
        + sum(W[jd, jy] * h2_hpa_from[v][jh, jd, jy] for v in hpa_h2, jh in JH, jd in JD)
        >= gamma_GC * (
            sum(W[jd, jy] * h2_in_pool[jh, jd, jy] for jh in JH, jd in JD)
            + sum(W[jd, jy] * h2_hpa_from[v][jh, jd, jy] for v in hpa_h2, jh in JH, jd in JD)
        )
    )
    for v in hpa_h2
        mod.ext[:constraints][Symbol("hpa_cap_limit_", v)] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
            h2_hpa_from[v][jh, jd, jy] <= hpa_cap[v])
    end

    alpha_G = mod.ext[:variables][:alpha_GreenOfftaker] = @variable(mod, lower_bound = 0, base_name = "alpha_GreenOfftaker_$(m)")
    cvar_G = mod.ext[:variables][:CVaR_GreenOfftaker] = @variable(mod, lower_bound = 0, base_name = "CVaR_GreenOfftaker_$(m)")
    u_G = mod.ext[:variables][:u_GreenOfftaker] = @variable(mod, [jy in JY], lower_bound = 0, base_name = "u_GreenOfftaker_$(m)")

    loss_G = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_G[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_H2[jh, jd, jy] * h2_in_pool[jh, jd, jy]
                + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
                + sum(K_hpa[v][jh, jd, jy] * h2_hpa_from[v][jh, jd, jy] for v in hpa_h2)
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
        loss_total[jy] = @expression(mod, loss_G[jy] + F_cap * cap_EP_y)
    end
    mod.ext[:expressions][:loss_GreenOfftaker] = loss_G

    mod.ext[:constraints][:CVaR_Green_shortfall] = @constraint(mod, [jy in JY], u_G[jy] >= loss_total[jy] - alpha_G)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_Green_link] = @constraint(mod,
        cvar_G >= alpha_G + (1 / one_minus_beta) * sum(P[jy] * u_G[jy] for jy in JY)
    )

    obj_hpa = sum(
        sum(ρ_hpa[v]/2 * W[jd, jy] * ((-h2_hpa_from[v][jh, jd, jy]) - g_bar_hpa[v][jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + (ρ_hpa_cap[v]/2) * ((-hpa_cap[v]) - g_bar_hpa_cap[v])^2
        for v in hpa_h2
    )

    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_H2[jh, jd, jy] * h2_in_pool[jh, jd, jy]
            + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy]
            + sum(K_hpa[v][jh, jd, jy] * h2_hpa_from[v][jh, jd, jy] for v in hpa_h2)
            + proc_cost * ep[jh, jd, jy]
            - λ_EP[jh, jd, jy] * ep[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * ((-h2_in_pool[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + obj_hpa
        + F_cap * cap_EP_y
    )

    return mod
end
