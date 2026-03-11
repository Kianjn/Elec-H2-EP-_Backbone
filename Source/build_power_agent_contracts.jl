# ==============================================================================
# build_power_agent_contracts.jl — Power agents with PPAs (VRES only)
# ==============================================================================
#
# PURPOSE:
#   Builds power-sector agents for the market_exposure_ppa entry point.
#   For VRES: extends the model with PPA variables and constraints.
#   For Conventional and Consumer: delegates to build_power_agent! (unchanged).
#
#   VRES PPAs (bundled elec + elec_GC):
#   - VRES splits generation into g_EOM (sold to electricity/elec_GC markets) and
#     g_ppa (delivered under PPA to electrolyzer). PPA flow is REMOVED from
#     both EOM and elec_GC — real-world: buyer receives elec+GC as a package.
#   - PPA capacity ppa_cap (MW) is the maximum that can flow.
#   - Pay-as-produced: VRES earns λ_ppa per MWh actually delivered (bundled price).
#   - Total generation: g_EOM + g_ppa <= AF × cap_VRES.
#   - PPA delivery: g_ppa <= ppa_cap at each hour.
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

    # ADMM parameters — PPA pool (g_ppa at λ_ppa; ppa_cap consensus)
    λ_ppa     = mod.ext[:parameters][:λ_ppa]
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
    cap_VRES = mod.ext[:variables][:cap_VRES] = @variable(mod, [jy in JY], lower_bound=0, base_name="cap_VRES")
    inv_VRES = mod.ext[:variables][:inv_VRES] = @variable(mod, [jy in JY], lower_bound=0, base_name="inv_VRES")

    JY_vec = collect(JY)
    first_jy = JY_vec[1]
    mod.ext[:constraints][:cap_VRES_init] = @constraint(mod, cap_VRES[first_jy] == cap_initial + inv_VRES[first_jy])
    for (k, jy) in enumerate(JY_vec)
        k == 1 && continue
        prev_jy = JY_vec[k - 1]
        mod.ext[:constraints][Symbol("cap_VRES_dyn_", jy)] = @constraint(mod, cap_VRES[jy] == cap_VRES[prev_jy] + inv_VRES[jy])
    end

    # ── Generation split: EOM (pool) vs contract ────────────────────────────
    # g_EOM     = electricity sold to the day-ahead / spot market (MWh)
    # g_contract = electricity delivered under bilateral contract (MWh)
    # Total generation = g_EOM + g_contract
    g_EOM     = mod.ext[:variables][:g_EOM]     = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_EOM")
    g_ppa = mod.ext[:variables][:g_ppa] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_ppa")

    # PPA capacity (MW): maximum that can flow under the bilateral agreement.
    # At each hour, g_ppa <= ppa_cap. Pay-as-produced: when AF=0
    # (e.g. night for solar), g_ppa=0, so nothing delivered and nothing paid.
    ppa_cap = mod.ext[:variables][:ppa_cap] = @variable(mod, lower_bound=0, base_name="ppa_cap")

    # ── Net positions ──────────────────────────────────────────────────────
    # Electricity market: only g_EOM (pool sales). PPA flow bypasses the pool.
    mod.ext[:expressions][:g_net_elec] = @expression(mod, g_EOM)

    # Elec-GC market: only g_EOM contributes. PPA electricity is BUNDLED with
    # its GC and delivered directly to the H2 producer — it is REMOVED from
    # both EOM and elec_GC market. Real-world PPAs: buyer receives elec+GC
    # as a package; VRES cannot sell that capacity in either market.
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g_EOM)

    # PPA market: VRES supplies g_ppa (positive = seller).
    mod.ext[:expressions][:g_net_ppa] = @expression(mod, g_ppa)

    # ── Physical constraints ──────────────────────────────────────────────
    # Total generation limited by availability × capacity.
    mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES[jy])

    # PPA delivery cannot exceed PPA capacity at any hour.
    mod.ext[:constraints][:ppa_cap_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_ppa[jh, jd, jy] <= ppa_cap)

    # ── Risk variables (CVaR, same structure as base VRES) ──────────────────
    alpha_VRES = mod.ext[:variables][:alpha_VRES] = @variable(mod, lower_bound=0, base_name="alpha_VRES_$(m)")
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]  = @variable(mod, lower_bound=0, base_name="CVaR_VRES_$(m)")
    u_VRES     = mod.ext[:variables][:u_VRES]     = @variable(mod, [jy in JY], lower_bound=0, base_name="u_VRES_$(m)")

    # Per-year loss: cost − revenue. PPA is bundled (elec+GC); λ_contract is the
    # bundled price. Pool revenue: λ_elec * g_EOM, λ_elec_GC * g_EOM only
    # (PPA flow removed from both markets).
    loss_VRES = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_VRES[jy] = @expression(mod,
            sum(W[jd, jy] * (
                MC * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
                - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_elec_GC[jh, jd, jy] * g_EOM[jh, jd, jy]
                - λ_ppa[jh, jd, jy] * g_ppa[jh, jd, jy]
            ) for jh in JH, jd in JD)
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
            MC * (g_EOM[jh, jd, jy] + g_ppa[jh, jd, jy])
            - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
            - λ_elec_GC[jh, jd, jy] * g_EOM[jh, jd, jy]
            - λ_ppa[jh, jd, jy] * g_ppa[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_ppa/2 * W[jd, jy] * (g_ppa[jh, jd, jy] - g_bar_ppa[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        # ppa_cap consensus: both parties must agree (penalty only, no separate price).
        + (ρ_ppa_cap/2) * (ppa_cap - g_bar_ppa_cap)^2
        + F_cap * sum(cap_VRES[jy] for jy in JY)
    )

    return mod
end
