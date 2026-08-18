# ==============================================================================
# build_H2_agent_contracts.jl — GreenProducer with bilateral PPA + HPA
# ==============================================================================
#
# PURPOSE:
#   Builds the GreenProducer model for me_pap.jl, me_top.jl, me_sop.jl.
#   Extends the base model with:
#   - PPA buy-side (receiving electricity+elec_GC from each VRES), and
#   - HPA sell-side (delivering hydrogen bundled with H2_GC equivalent
#     to green offtaker side).
#
#   PER-VRES PPAs (pay-as-produced CfD, bundled elec+elec_GC):
#   - The electrolyzer receives g_ppa_from[vres_id] (MWh) from each VRES.
#   - Payment: (K − λ_elec − λ_elec_GC) × g_ppa_from per VRES.
#   - Total physical electricity input remains e_in_pool.
#   - g_ppa_from is a hedge quantity and does not alter physical conversion.
#
# HPA (pay-as-produced CfD, bundled H2 + H2_GC):
#   - GreenProducer supplies hedge h2_hpa; payoff (K − λ_H2 − λ_H2_GC) · q
#     plus SoP/ToP volume adjustments.
#   - h2_hpa <= hpa_cap at each hour; it is a hedge quantity (not physical carve-out).
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model.
#   H2_market, H2_GC_market — Market dicts.
#   ppa_market — PPA market dict (per_vres, ppa_vres).
#   hpa terms are read from mod.ext[:parameters] placeholders.
#
# ==============================================================================

function build_H2_agent_contracts!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict,
                                  ppa_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]

    η  = mod.ext[:parameters][:η_elec_H2]
    cap_H2_initial = mod.ext[:parameters][:Capacity_H2_Output]
    op_cost  = mod.ext[:parameters][:OperationalCost]
    F_cap    = electrolyzer_h2_annuity(mod.ext[:parameters])
    gamma    = get(mod.ext[:parameters], :γ, 1.0)
    beta_conf = get(mod.ext[:parameters], :β, 0.95)
    P        = mod.ext[:parameters][:P]

    # ADMM parameters — standard markets
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
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]

    # Per-VRES PPA params (electrolyzer has Dict vres_id => array)
    λ_ppa     = mod.ext[:parameters][:λ_ppa]
    K_ppa     = get(mod.ext[:parameters], :K_ppa, λ_ppa)
    g_bar_ppa = mod.ext[:parameters][:g_bar_ppa]
    ρ_ppa     = mod.ext[:parameters][:ρ_ppa]
    g_bar_ppa_cap = mod.ext[:parameters][:g_bar_ppa_cap]
    ρ_ppa_cap     = mod.ext[:parameters][:ρ_ppa_cap]
    ppa_vres = collect(keys(λ_ppa))

    λ_hpa     = mod.ext[:parameters][:λ_hpa]
    K_hpa     = get(mod.ext[:parameters], :K_hpa, λ_hpa)
    g_bar_hpa = mod.ext[:parameters][:g_bar_hpa]
    ρ_hpa     = mod.ext[:parameters][:ρ_hpa]
    g_bar_hpa_cap = mod.ext[:parameters][:g_bar_hpa_cap]
    ρ_hpa_cap     = mod.ext[:parameters][:ρ_hpa_cap]

    # ── Decision variables ─────────────────────────────────────────────────
    e_in_pool = mod.ext[:variables][:e_in_pool] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_pool")
    g_ppa_from = mod.ext[:variables][:g_ppa_from] = Dict(
        v => @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_ppa_$(v)")
        for v in ppa_vres
    )
    h2_hpa = mod.ext[:variables][:h2_hpa] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_hpa")

    h2_out    = mod.ext[:variables][:h2_out]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_out")
    q_elec_gc = mod.ext[:variables][:q_elec_gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_GC")
    q_h2gc    = mod.ext[:variables][:q_h2gc]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_GC_prod")

    cap_H2_y   = mod.ext[:variables][:cap_H2_y]   = @variable(mod, lower_bound=0, base_name="cap_H2")
    inv_cap_H2 = mod.ext[:variables][:inv_cap_H2] = @variable(mod, lower_bound=0, base_name="inv_cap_H2")
    mod.ext[:constraints][:cap_H2_init] = @constraint(mod, cap_H2_y == cap_H2_initial + inv_cap_H2)

    # PPA capacity (MW) per VRES: upper bound on g_ppa_from at each hour.
    ppa_cap = mod.ext[:variables][:ppa_cap] = Dict(
        v => @variable(mod, lower_bound=0, base_name="ppa_cap_$(v)")
        for v in ppa_vres
    )
    hpa_cap = mod.ext[:variables][:hpa_cap] = @variable(mod, lower_bound=0, base_name="hpa_cap")

    # ── Net market positions ────────────────────────────────────────────────
    # Electricity market: electrolyzer buys e_in_pool (negative = buyer).
    mod.ext[:expressions][:g_net_elec] = @expression(mod, -e_in_pool)

    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, -q_elec_gc)
    mod.ext[:expressions][:g_net_H2]      = @expression(mod, h2_out)
    mod.ext[:expressions][:g_net_H2_GC]   = @expression(mod, q_h2gc)

    # PPA market: electrolyzer demands g_ppa_from[v] per VRES (for result extraction).
    mod.ext[:expressions][:g_net_ppa_from] = g_ppa_from
    mod.ext[:expressions][:g_net_hpa] = @expression(mod, h2_hpa)

    # ── Physical constraints ────────────────────────────────────────────────
    # Physical conversion remains in the pool only (contracts hedge cashflows).
    mod.ext[:constraints][:h2_from_elec] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_out[jh, jd, jy] == η * e_in_pool[jh, jd, jy])

    # Pool H2-GC is backed by physical hydrogen output.
    mod.ext[:constraints][:gc_phys_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        q_h2gc[jh, jd, jy] <= h2_out[jh, jd, jy])

    mod.ext[:constraints][:cap_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_out[jh, jd, jy] <= cap_H2_y)
    # Hedge volume bounded by production and contract cap.
    mod.ext[:constraints][:hpa_prod_link] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_hpa[jh, jd, jy] <= h2_out[jh, jd, jy])
    mod.ext[:constraints][:hpa_cap_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_hpa[jh, jd, jy] <= hpa_cap)
    mod.ext[:constraints][:hpa_cap_plant] = @constraint(mod, hpa_cap <= cap_H2_y)

    # PPA hedge volume bounds per VRES.
    for v in ppa_vres
        mod.ext[:constraints][Symbol("ppa_cap_limit_", v)] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
            g_ppa_from[v][jh, jd, jy] <= ppa_cap[v])
        mod.ext[:constraints][Symbol("ppa_phys_bound_", v)] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
            g_ppa_from[v][jh, jd, jy] <= e_in_pool[jh, jd, jy])
        # Per-VRES contract capacity bounded by electrolyzer electricity input (MW_elec).
        mod.ext[:constraints][Symbol("ppa_cap_electro_limit_", v)] = @constraint(mod, ppa_cap[v] <= cap_H2_y / η)
    end

    raw_mode = lowercase(String(get(mod.ext[:parameters], :hpa_volume_mode, "sop")))
    mode = raw_mode == "pap" ? "sop" : raw_mode
    mode in ("sop", "top") || error("Unsupported hpa_volume_mode=$(raw_mode). Use sop or top.")
    mod.ext[:parameters][:hpa_volume_mode] = mode
    add_hpa_volume_variables!(mod; role=:producer)

    # Annual green-backing based on physical pool production only.
    mod.ext[:constraints][:gc_backing_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_elec_gc[jh, jd, jy] for jh in JH, jd in JD) >=
        (1 / η) * sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD)
    )

    # ── Risk variables (CVaR) ──────────────────────────────────────────────
    alpha_H2 = mod.ext[:variables][:alpha_H2] = @variable(mod, base_name="alpha_H2_$(m)")
    cvar_H2  = mod.ext[:variables][:CVaR_H2]  = @variable(mod, base_name="CVaR_H2_$(m)")
    u_H2     = mod.ext[:variables][:u_H2]     = @variable(mod, [jy in JY], lower_bound=0, base_name="u_H2_$(m)")

    # Per-year loss: pool cost + PPA cost + op cost − H2/H2_GC pool revenue − HPA bundled revenue
    loss_H2 = Dict{Int,JuMP.AffExpr}()
    loss_total = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_H2[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy]       * e_in_pool[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
            ) for jh in JH, jd in JD)
            + sum_ppa_buyer_cost_jy(mod, ppa_vres, jy, W, JH, JD)
            - sum_hpa_seller_revenue_jy(mod, jy, W, JH, JD)
        )
        loss_total[jy] = @expression(mod, loss_H2[jy] + F_cap * cap_H2_y)
    end
    mod.ext[:expressions][:loss_H2] = loss_H2

    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_total[jy] - alpha_H2)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY))

    # ── Objective ───────────────────────────────────────────────────────────
    obj_ppa = sum(
        sum(
            ρ_ppa[v]/2 * W[jd, jy] * ((-g_ppa_from[v][jh, jd, jy]) - g_bar_ppa[v][jh, jd, jy])^2
            for jh in JH, jd in JD, jy in JY
        )
        for v in ppa_vres
    )
    obj_hpa = sum(ρ_hpa/2 * W[jd, jy] * (h2_hpa[jh, jd, jy] - g_bar_hpa[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_elec[jh, jd, jy]       * e_in_pool[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
            + op_cost * h2_out[jh, jd, jy]
            - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
            - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum_ppa_buyer_cost(mod, ppa_vres, W, JH, JD, JY)
        - sum_hpa_seller_revenue(mod, W, JH, JD, JY)
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in_pool[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy] - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + obj_ppa
        + obj_hpa
        + F_cap * cap_H2_y
    )

    return mod
end
