# ==============================================================================
# build_H2_agent_contracts.jl — Electrolyzer with bilateral contract
# ==============================================================================
#
# PURPOSE:
#   Builds the electrolyzer (GreenProducer) model for market_exposure_contracts.
#   Extends the base model to receive electricity from a bilateral contract
#   with VRES in addition to the electricity market.
#
#   BILATERAL CONTRACT (pay-as-produced):
#   - The electrolyzer receives g_contract (MWh) from VRES at each timestep.
#   - Payment: λ_contract × g_contract (only for energy actually delivered).
#   - When VRES has no output (e.g. night for solar), g_contract = 0,
#     so nothing is delivered and nothing is paid.
#   - Total electricity input: e_in_pool + g_contract.
#   - Green-backing: contract electricity is inherently green (from VRES),
#     so it counts toward the annual green-backing constraint.
#
# ARGUMENTS:
#   m — Agent ID.
#   mod — JuMP model.
#   H2_market, H2_GC_market — Market dicts.
#   contract_market — Contract market dict.
#
# ==============================================================================

function build_H2_agent_contracts!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict,
                                  contract_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]

    η  = mod.ext[:parameters][:η_elec_H2]
    cap_H2_initial = mod.ext[:parameters][:Capacity_H2_Output]
    op_cost  = mod.ext[:parameters][:OperationalCost]
    F_cap    = get(mod.ext[:parameters], :FixedCost_per_MW_Electrolyzer, 0.0)
    gamma    = get(mod.ext[:parameters], :γ, 1.0)
    beta_conf = get(mod.ext[:parameters], :β, 0.95)
    P        = mod.ext[:parameters][:P]

    # ADMM parameters — all markets including contract
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
    λ_contract     = mod.ext[:parameters][:λ_contract]
    g_bar_contract = mod.ext[:parameters][:g_bar_contract]
    ρ_contract     = mod.ext[:parameters][:ρ_contract]
    g_bar_contract_cap = mod.ext[:parameters][:g_bar_contract_cap]
    ρ_contract_cap     = mod.ext[:parameters][:ρ_contract_cap]

    # ── Decision variables ─────────────────────────────────────────────────
    # e_in_pool = electricity purchased from the electricity market (MWh)
    # g_contract = electricity received from bilateral contract (MWh)
    # Total input = e_in_pool + g_contract
    e_in_pool  = mod.ext[:variables][:e_in_pool]  = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_pool")
    g_contract = mod.ext[:variables][:g_contract]  = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_contract")

    h2_out    = mod.ext[:variables][:h2_out]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_out")
    q_elec_gc = mod.ext[:variables][:q_elec_gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="elec_GC")
    q_h2gc    = mod.ext[:variables][:q_h2gc]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h2_GC_prod")

    cap_H2_y   = mod.ext[:variables][:cap_H2_y]   = @variable(mod, [jy in JY], lower_bound=0, base_name="cap_H2")
    inv_cap_H2 = mod.ext[:variables][:inv_cap_H2] = @variable(mod, [jy in JY], lower_bound=0, base_name="inv_cap_H2")

    # Contract capacity (MW): upper bound on g_contract at each hour.
    # Must match VRES's contract_cap (enforced by ADMM contract market clearing).
    contract_cap = mod.ext[:variables][:contract_cap] = @variable(mod, lower_bound=0, base_name="contract_cap")

    # Capacity evolution
    JY_vec = collect(JY)
    first_jy = JY_vec[1]
    mod.ext[:constraints][:cap_H2_init] = @constraint(mod, cap_H2_y[first_jy] == cap_H2_initial + inv_cap_H2[first_jy])
    for (k, jy) in enumerate(JY_vec)
        k == 1 && continue
        prev_jy = JY_vec[k - 1]
        mod.ext[:constraints][Symbol("cap_H2_dyn_", jy)] = @constraint(mod, cap_H2_y[jy] == cap_H2_y[prev_jy] + inv_cap_H2[jy])
    end

    # ── Net market positions ────────────────────────────────────────────────
    # Electricity market: electrolyzer buys e_in_pool (negative = buyer).
    mod.ext[:expressions][:g_net_elec] = @expression(mod, -e_in_pool)

    mod.ext[:expressions][:g_net_elec_GC] = @expression(mod, -q_elec_gc)
    mod.ext[:expressions][:g_net_H2]      = @expression(mod, h2_out)
    mod.ext[:expressions][:g_net_H2_GC]   = @expression(mod, q_h2gc)

    # Contract market: electrolyzer demands g_contract (negative = buyer).
    mod.ext[:expressions][:g_net_contract] = @expression(mod, -g_contract)

    # ── Physical constraints ────────────────────────────────────────────────
    # Conversion: h2_out = η × (e_in_pool + g_contract)
    mod.ext[:constraints][:h2_from_elec] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_out[jh, jd, jy] == η * (e_in_pool[jh, jd, jy] + g_contract[jh, jd, jy]))

    mod.ext[:constraints][:gc_phys_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        q_h2gc[jh, jd, jy] <= h2_out[jh, jd, jy])

    mod.ext[:constraints][:cap_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        h2_out[jh, jd, jy] <= cap_H2_y[jy])

    # Contract delivery cannot exceed contracted capacity.
    mod.ext[:constraints][:contract_cap_limit] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
        g_contract[jh, jd, jy] <= contract_cap)

    # Annual green-backing: elec GCs purchased + contract electricity (inherently
    # green from VRES) must be enough to back all H2 GCs issued.
    #   sum(q_elec_gc) + sum(g_contract) >= (1/η) × sum(q_h2gc)
    mod.ext[:constraints][:gc_backing_yearly] = @constraint(mod, [jy in JY],
        sum(W[jd, jy] * q_elec_gc[jh, jd, jy] for jh in JH, jd in JD) +
        sum(W[jd, jy] * g_contract[jh, jd, jy] for jh in JH, jd in JD) >=
        (1 / η) * sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD)
    )

    # ── Risk variables (CVaR) ──────────────────────────────────────────────
    alpha_H2 = mod.ext[:variables][:alpha_H2] = @variable(mod, lower_bound=0, base_name="alpha_H2_$(m)")
    cvar_H2  = mod.ext[:variables][:CVaR_H2]  = @variable(mod, lower_bound=0, base_name="CVaR_H2_$(m)")
    u_H2     = mod.ext[:variables][:u_H2]     = @variable(mod, [jy in JY], lower_bound=0, base_name="u_H2_$(m)")

    # Per-year loss: pool cost + contract cost + op cost − H2 revenue − H2_GC revenue
    loss_H2 = Dict{Int,JuMP.AffExpr}()
    for jy in JY
        loss_H2[jy] = @expression(mod,
            sum(W[jd, jy] * (
                λ_elec[jh, jd, jy]       * e_in_pool[jh, jd, jy]
                + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
                + λ_contract[jh, jd, jy] * g_contract[jh, jd, jy]
                + op_cost * h2_out[jh, jd, jy]
                - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
                - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
            ) for jh in JH, jd in JD)
        )
    end
    mod.ext[:expressions][:loss_H2] = loss_H2

    mod.ext[:constraints][:CVaR_H2_shortfall] = @constraint(mod, [jy in JY],
        u_H2[jy] >= loss_H2[jy] - alpha_H2)
    one_minus_beta = max(1e-6, 1.0 - beta_conf)
    mod.ext[:constraints][:CVaR_H2_link] = @constraint(mod,
        cvar_H2 >= alpha_H2 + (1 / one_minus_beta) * sum(P[jy] * u_H2[jy] for jy in JY))

    # ── Objective ───────────────────────────────────────────────────────────
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_elec[jh, jd, jy]       * e_in_pool[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
            + λ_contract[jh, jd, jy] * g_contract[jh, jd, jy]
            + op_cost * h2_out[jh, jd, jy]
            - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
            - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in_pool[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy]         - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_contract/2 * W[jd, jy] * ((-g_contract[jh, jd, jy]) - g_bar_contract[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        # contract_cap consensus: both parties must agree (penalty only, no separate price).
        + (ρ_contract_cap/2) * ((-contract_cap) - g_bar_contract_cap)^2
        + F_cap * sum(cap_H2_y[jy] for jy in JY)
    )

    return mod
end
