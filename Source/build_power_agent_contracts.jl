# ==============================================================================
# build_power_agent_contracts.jl — Power agents with bilateral contract (VRES only)
# ==============================================================================
#
# PURPOSE:
#   Builds power-sector agents for the market_exposure_contracts entry point.
#   For VRES: extends the model with bilateral contract variables and constraints.
#   For Conventional and Consumer: delegates to build_power_agent! (unchanged).
#
#   VRES BILATERAL CONTRACT:
#   - VRES splits generation into g_EOM (sold to electricity market) and
#     g_contract (delivered under bilateral contract to electrolyzer).
#   - Contract capacity contract_cap (MW) is the maximum that can flow.
#   - Pay-as-produced: VRES earns λ_contract per MWh actually delivered.
#   - Capacity removed from EOM: only g_EOM goes to the electricity market;
#     g_contract is delivered directly to the electrolyzer, so it does NOT
#     appear in the electricity market balance.
#   - Total generation: g_EOM + g_contract <= AF × cap_VRES.
#   - Contract delivery: g_contract <= contract_cap at each hour.
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model (parameters and sets from define_*_parameters!).
#   elec_market, elec_GC_market — Market dicts.
#   contract_market — Contract market dict (initial_price, rho_initial, shape).
#
# ==============================================================================

function build_power_agent_contracts!(m::String, mod::Model, elec_market::Dict, elec_GC_market::Dict,
                                     contract_market::Dict)
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

    # ADMM parameters — contract pool (g_contract at λ_contract; contract_cap consensus)
    λ_contract     = mod.ext[:parameters][:λ_contract]
    g_bar_contract = mod.ext[:parameters][:g_bar_contract]
    ρ_contract     = mod.ext[:parameters][:ρ_contract]
    g_bar_contract_cap = mod.ext[:parameters][:g_bar_contract_cap]
    ρ_contract_cap     = mod.ext[:parameters][:ρ_contract_cap]

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
    g_contract = mod.ext[:variables][:g_contract] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gen_contract")

    # Contract capacity (MW): maximum that can flow under the bilateral agreement.
    # At each hour, g_contract <= contract_cap. Pay-as-produced: when AF=0
    # (e.g. night for solar), g_contract=0, so nothing delivered and nothing paid.
    contract_cap = mod.ext[:variables][:contract_cap] = @variable(mod, lower_bound=0, base_name="contract_cap")

    # ── Net positions ──────────────────────────────────────────────────────
    # Electricity market: only g_EOM (pool sales). Contract flow bypasses the pool.
    mod.ext[:expressions][:g_net_elec] = @expression(mod, g_EOM)

    # Elec-GC market: total VRES generation produces GCs (1:1). Both g_EOM and
    # g_contract are renewable, so total GC supply = g_EOM + g_contract.
    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, g_EOM + g_contract)

    # Contract market: VRES supplies g_contract (positive = seller).
    mod.ext[:expressions][:g_net_contract] = @expression(mod, g_contract)

    # ── Physical constraints ──────────────────────────────────────────────
    # Total generation limited by availability × capacity.
    mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_EOM[jh, jd, jy] + g_contract[jh, jd, jy] <= AF[jh, jd, jy] * cap_VRES[jy])

    # Contract delivery cannot exceed contracted capacity at any hour.
    mod.ext[:constraints][:contract_cap_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_contract[jh, jd, jy] <= contract_cap)

    # ── Risk variables (CVaR, same structure as base VRES) ──────────────────
    alpha_VRES = mod.ext[:variables][:alpha_VRES] = @variable(mod, lower_bound=0, base_name="alpha_VRES_$(m)")
    cvar_VRES  = mod.ext[:variables][:CVaR_VRES]  = @variable(mod, lower_bound=0, base_name="CVaR_VRES_$(m)")
    u_VRES     = mod.ext[:variables][:u_VRES]     = @variable(mod, [jy in JY], lower_bound=0, base_name="u_VRES_$(m)")

    # Per-year loss: cost − revenue. Revenue now includes contract (λ_contract * g_contract).
    # Pool revenue: λ_elec * g_EOM, λ_elec_GC * (g_EOM + g_contract).
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

    mod.ext[:constraints][:CVaR_VRES_shortfall] = @constraint(mod, [jy in JY],
        u_VRES[jy] >= loss_VRES[jy] - alpha_VRES)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_VRES_link] = @constraint(mod,
        cvar_VRES >= alpha_VRES + (1 / one_minus_beta) * sum(P[jy] * u_VRES[jy] for jy in JY))

    # ── Objective: cost − revenue + ADMM penalties ──────────────────────────
    # Penalties: elec (g_EOM), elec_GC (g_EOM + g_contract), contract (g_contract).
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            MC * (g_EOM[jh, jd, jy] + g_contract[jh, jd, jy])
            - λ_elec[jh, jd, jy] * g_EOM[jh, jd, jy]
            - λ_elec_GC[jh, jd, jy] * (g_EOM[jh, jd, jy] + g_contract[jh, jd, jy])
            - λ_contract[jh, jd, jy] * g_contract[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec/2 * W[jd, jy] * (g_EOM[jh, jd, jy] - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((g_EOM[jh, jd, jy] + g_contract[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_contract/2 * W[jd, jy] * (g_contract[jh, jd, jy] - g_bar_contract[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        # contract_cap consensus: both parties must agree (penalty only, no separate price).
        + (ρ_contract_cap/2) * (contract_cap - g_bar_contract_cap)^2
        + F_cap * sum(cap_VRES[jy] for jy in JY)
    )

    return mod
end
