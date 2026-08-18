# ==============================================================================
# build_power_agent_contracts.jl — Power agents with PPAs (VRES only)
# ==============================================================================
#
# PURPOSE:
#   Builds power-sector agents for the me_pap / me_top / me_sop entry points.
#   For VRES: extends the model with PPA variables and constraints.
#   For Conventional and Consumer: delegates to build_power_agent! (unchanged).
#
#   VRES PPAs (bundled elec + elec_GC):
#   - VRES keeps physical dispatch in the pool (g_EOM). PPA quantity g_ppa is a
#     bilateral hedge: CfD (K − λ_elec − λ_elec_GC) · g_ppa on top of pool sales.
#   - PPA capacity ppa_cap (MW) is the maximum hedge volume.
#   - Physical generation: g_EOM <= AF × cap_VRES.
#   - Hedge volume is bounded by contract and availability: g_ppa <= ppa_cap and
#     g_ppa <= AF × cap_VRES.
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model (parameters and sets from define_*_parameters!).
#   elec_market, elec_GC_market — Market dicts.
#   ppa_market — PPA market dict (initial_price, rho_initial, per_vres).
#
# ==============================================================================

function build_power_agent_contracts!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict,
                                     ppa_market::Dict)
    agent_type = mod.ext[:parameters][:Type]

    # Conventional and Consumer are unchanged — delegate to original build.
    if agent_type in ("Conventional", "Consumer")
        return build_power_agent!(m, mod, elec_market, elec_GC_market)
    end

    # ── VRES with bilateral contract ───────────────────────────────────────
    if agent_type != "VRES"
        return mod
    end

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]

    # ADMM parameters — electricity and elec-GC markets
    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]

    # ADMM parameters — PPA hedge (CfD at K_ppa; quantity consensus via g_bar_ppa)
    g_bar_ppa = mod.ext[:parameters][:g_bar_ppa]
    ρ_ppa     = mod.ext[:parameters][:ρ_ppa]
    g_bar_ppa_cap = mod.ext[:parameters][:g_bar_ppa_cap]
    ρ_ppa_cap     = mod.ext[:parameters][:ρ_ppa_cap]

    cap_initial = mod.ext[:parameters][:Capacity]
    AF          = mod.ext[:timeseries][:AF]
    MC          = mod.ext[:parameters][:MarginalCost]
    F_cap       = get(mod.ext[:parameters], :FixedCost_per_MW, 0.0)
    gamma       = get(mod.ext[:parameters], :γ, 1.0)
    beta_conf   = get(mod.ext[:parameters], :β, 0.95)
    P           = mod.ext[:parameters][:P]

    # ── Capacity and investment variables (same as base VRES) ───────────────
    cap_VRES = mod.ext[:variables][:cap_VRES] = @variable(mod, lower_bound=0, base_name="cap_VRES")
    inv_VRES = mod.ext[:variables][:inv_VRES] = @variable(mod, lower_bound=0, base_name="inv_VRES")
    mod.ext[:constraints][:cap_VRES_init] = @constraint(mod, cap_VRES == cap_initial + inv_VRES)

    # ── Physical pool dispatch vs financial PPA hedge ───────────────────────
    # g_EOM = electricity sold to the spot market (physical; binds AF × cap).
    # g_ppa = bilateral hedge quantity (financial; does not consume extra MW).
    g_EOM     = mod.ext[:variables][:g_EOM]     = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_EOM")
    g_ppa = mod.ext[:variables][:g_ppa] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_ppa")

    # PPA capacity (MW): maximum that can flow under the bilateral agreement.
    # At each hour, g_ppa <= ppa_cap. Pay-as-produced: when AF=0
    # (e.g. night for solar), g_ppa=0, so nothing delivered and nothing paid.
    ppa_cap = mod.ext[:variables][:ppa_cap] = @variable(mod, lower_bound=0, base_name="ppa_cap")

    # ── Net positions ──────────────────────────────────────────────────────
    # Electricity market: physical pool dispatch only.
    mod.ext[:expressions][:g_net_elec] = @expression(mod, g_EOM)

    # Elec-GC market follows physical pool dispatch only.
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g_EOM)

    # PPA market: VRES supplies g_ppa (positive = seller).
    mod.ext[:expressions][:g_net_ppa] = @expression(mod, g_ppa)

    # ── Physical constraints ──────────────────────────────────────────────
    # Physical generation limited by availability × capacity.
    mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_EOM[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES)

    # Financial hedge volume bounds (no hard physical must-deliver).
    mod.ext[:constraints][:ppa_cap_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_ppa[jh, jd, jy] <= ppa_cap)
    mod.ext[:constraints][:ppa_phys_bound] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_ppa[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES)
    mod.ext[:constraints][:ppa_cap_plant] = @constraint(mod, ppa_cap <= cap_VRES)

    # ── Risk variables (CVaR, same structure as base VRES) ──────────────────
    alpha_VRES = mod.ext[:variables][:alpha_VRES] = @variable(mod, base_name="alpha_VRES_$(m)")
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]  = @variable(mod, base_name="CVaR_VRES_$(m)")
    u_VRES     = mod.ext[:variables][:u_VRES]     = @variable(mod, [jy in JY], lower_bound=0, base_name="u_VRES_$(m)")

    # Per-year loss: cost − revenue. Pool: λ_elec/λ_elec_GC on g_EOM.
    # PPA is a bundled CfD: (K − λ_elec − λ_elec_GC) · g_ppa.
    loss_VRES = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_VRES[jy] = @expression(mod,
            sum(W[jd, jy] * (
                MC * g_EOM[jh, jd, jy]
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g_EOM[jh, jd, jy]
            ) for jh in JH, jd in JD)
            - sum_ppa_seller_revenue_jy(mod, jy, W, JH, JD)
        )
    end
    mod.ext[:expressions][:loss_VRES] = loss_VRES

    mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
        u_VRES[jy] >= loss_VRES[jy] - alpha_VRES)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
        cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

    # ── Objective: cost − revenue + ADMM penalties ──────────────────────────
    # Penalties: elec (g_EOM), elec_GC (g_EOM only — PPA removed), contract (g_ppa).
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            MC * g_EOM[jh, jd, jy]
            - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
            - λ_elec_GC[jh, jd, jy] * g_EOM[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        - sum_ppa_seller_revenue(mod, W, JH, JD, JY)
        + sum(ρ_elec/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_ppa/2 * W[jd, jy] * (g_ppa[jh, jd, jy] - g_bar_ppa[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + F_cap * cap_VRES
    )

    return mod
end
