# ==============================================================================
# build_offtaker_agent.jl — JuMP models for offtakers (green, grey, importer)
# ==============================================================================
#
# PURPOSE:
#   GreenOfftaker: h2_in, q_h2gc, ep; ep == (1/α)*h2_in (no H₂ waste); q_h2gc >= 0.42*ep.
#   GreyOfftaker: ep, q_h2gc >= 0.42*ep; no H₂ purchase. EPImporter: ep <= cap, import cost.
#   All report g_net_EP = ep; green reports g_net_H2 = -h2_in, g_net_H2_GC = -q_h2gc;
#   grey reports g_net_H2_GC = -q_h2gc. Objectives: cost - revenue + ADMM penalties.
#
# ==============================================================================

function build_offtaker_agent!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)
    # ── Index sets & weights ──────────────────────────────────────────────
    JH = mod.ext[:sets][:JH]          # hours within each representative day
    JD = mod.ext[:sets][:JD]          # representative days
    JY = mod.ext[:sets][:JY]          # years in the horizon
    W = mod.ext[:parameters][:W]      # W[jd,jy] = representative-day weight

    # gamma_GC = 0.42 (42%) is the green certificate mandate fraction,
    # derived from EU renewable fuel targets (e.g. RED II/III).  It specifies
    # the minimum share of output that must be backed by green H2 certificates.
    gamma_GC = get(mod.ext[:parameters], :gamma_GC, 0.42)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))

    # ── ADMM parameters — H₂ market ──────────────────────────────────────
    λ_H2     = mod.ext[:parameters][:λ_H2]       # Lagrange multiplier (price)
    g_bar_H2 = mod.ext[:parameters][:g_bar_H2]   # consensus target
    ρ_H2     = mod.ext[:parameters][:ρ_H2]       # penalty weight

    # ── ADMM parameters — H₂-GC market ───────────────────────────────────
    # H₂-GC price is annual (one price per year); penalties remain on full 3D grid.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC  = mod.ext[:parameters][:ρ_H2_GC]

    # ── ADMM parameters — EP (energy product) market ─────────────────────
    λ_EP     = mod.ext[:parameters][:λ_EP]
    g_bar_EP = mod.ext[:parameters][:g_bar_EP]
    ρ_EP     = mod.ext[:parameters][:ρ_EP]

    # ══════════════════════════════════════════════════════════════════════
    # GreenOfftaker: buys H₂ and H₂-GCs, sells energy product (EP).
    # alpha = H₂-to-EP conversion ratio (alpha=1 means 1 MWh_H2 -> 1 MWh_EP).
    # ══════════════════════════════════════════════════════════════════════
    if agent_type == "GreenOfftaker"
        cap_h2  = mod.ext[:parameters][:Capacity_H2_In]    # max H2 intake (MW_H2)
        cap_ep  = mod.ext[:parameters][:Capacity_EP_Out]   # max EP output (MW_EP)
        alpha   = get(mod.ext[:parameters], :Alpha, 1.0)   # H2-to-EP conversion
                                      # ratio: alpha=1 means 1 MWh_H2 -> 1 MWh_EP
        proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)  # processing cost (EUR/MWh_EP)

        # Decision variables.
        h2_in  = mod.ext[:variables][:h2_in]  = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_in")
        q_h2gc = mod.ext[:variables][:q_h2gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC")
        ep     = mod.ext[:variables][:ep]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep")

        # Net positions: H2 purchased (negative), H2-GCs purchased (negative),
        # EP sold (positive).
        mod.ext[:expressions][:g_net_H2]    = @expression(mod, -h2_in)     # buyer on H2 market
        mod.ext[:expressions][:g_net_H2_GC]  = @expression(mod, -q_h2gc)   # buyer on H2-GC market
        mod.ext[:expressions][:g_net_EP]     = @expression(mod, ep)         # seller on EP market

        # Tight stoichiometric link: ep == (1/alpha) * h2_in.  ALL purchased
        # H₂ must be converted to EP (no H₂ waste).  EP output is exactly
        # proportional to H₂ input via the conversion ratio alpha.
        mod.ext[:constraints][:ep_from_h2] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] == (1/alpha) * h2_in[jh, jd, jy])

        # Annual GC mandate: at least gamma_GC (42%) fraction of EP output
        # must be backed by green H₂ certificates, computed on a weighted
        # yearly basis (more realistic than hourly matching — allows temporal
        # flexibility in GC procurement within the year).
        mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
            sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * sum(W[jd, jy] * ep[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Capacity limits on H₂ intake and EP output.
        mod.ext[:constraints][:cap_h2]    = @constraint(mod, [jh in JH, jd in JD, jy in JY], h2_in[jh, jd, jy] <= cap_h2)
        mod.ext[:constraints][:cap_ep]    = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_ep)

        # Objective — min(cost - revenue + ADMM penalties):
        #   cost    = H2 purchase (lambda_H2 * h2_in) + GC purchase (lambda_H2GC * gc)
        #             + processing cost
        #   revenue = EP sales (lambda_EP * ep)
        #   penalties use net positions: -h2_in (H2), -gc (H2-GC), +ep (EP)
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (
                λ_H2[jh, jd, jy]        * h2_in[jh, jd, jy]
                + λ_H2_GC[jy]           * q_h2gc[jh, jd, jy]
                + proc_cost * ep[jh, jd, jy]
                - λ_EP[jh, jd, jy]      * ep[jh, jd, jy]
            ) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2/2 * W[jd, jy] * ((-h2_in[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

    # ══════════════════════════════════════════════════════════════════════
    # GreyOfftaker: EP seller using conventional feedstock.  Does NOT buy
    # H₂ on the market, but a fraction of its EP output requires H₂ as
    # feedstock internally (e.g. ammonia synthesis).  Must still purchase
    # H₂-GCs to certify the green share of that internal H₂ usage.
    # ══════════════════════════════════════════════════════════════════════
    elseif agent_type == "GreyOfftaker"
        cap_ep = mod.ext[:parameters][:Capacity]        # max EP output (MW_EP)
        MC     = mod.ext[:parameters][:MarginalCost]    # marginal cost (EUR/MWh_EP)

        ep     = mod.ext[:variables][:ep]    = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep")
        q_h2gc = mod.ext[:variables][:q_h2gc] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "h2_GC")

        # Net positions: H2-GCs purchased (negative), EP sold (positive).
        mod.ext[:expressions][:g_net_H2_GC] = @expression(mod, -q_h2gc)  # buyer on H2-GC market
        mod.ext[:expressions][:g_net_EP]    = @expression(mod, ep)        # seller on EP market

        # Annual GC mandate for grey offtaker.  gamma_NH3 = fraction of EP
        # output that requires H₂ as feedstock (e.g. 0.18 for ammonia).
        # The GC mandate applies to the H₂-equivalent portion of EP output:
        #   gc_h2 >= gamma_GC * gamma_NH3 * ep
        # i.e., at least 42% of the H₂-using share must be green-certified.
        mod.ext[:constraints][:gc_mandate_yearly] = @constraint(mod, [jy in JY],
            sum(W[jd, jy] * q_h2gc[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * mod.ext[:parameters][:gamma_NH3] *
            sum(W[jd, jy] * ep[jh, jd, jy] for jh in JH, jd in JD)
        )
        mod.ext[:constraints][:cap_ep]     = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap_ep)

        # Objective — min(cost - revenue + ADMM penalties):
        #   cost    = production cost (MC * ep) + GC purchase (lambda_H2GC * gc)
        #   revenue = EP sales (lambda_EP * ep)
        #   penalties use net positions: -gc (H2-GC, purchase), +ep (EP, sale)
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * ep[jh, jd, jy] + λ_H2_GC[jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )

    # ══════════════════════════════════════════════════════════════════════
    # EPImporter: simple price-taking EP supplier via imports.  No H₂
    # purchase and no GC involvement — just imports EP at a fixed cost.
    # ══════════════════════════════════════════════════════════════════════
    else  # EPImporter
        cap   = mod.ext[:parameters][:Capacity]       # max import capacity (MW_EP)
        imp_cost = mod.ext[:parameters][:ImportCost]   # import cost (EUR/MWh_EP)

        ep = mod.ext[:variables][:ep] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound = 0, base_name = "ep_import")
        mod.ext[:expressions][:g_net_EP] = @expression(mod, ep)  # seller on EP market (positive)
        mod.ext[:constraints][:cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY], ep[jh, jd, jy] <= cap)

        # Objective — min(import cost - EP revenue + ADMM penalty):
        #   Only participates in the EP market; net position = +ep (sale).
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        )
    end

    return mod
end

# ------------------------------------------------------------------------------
# Social planner: add offtaker block to shared planner model.
#
# Same physical constraints as the ADMM build (stoichiometric conversion,
# GC mandates, capacity limits) but WITHOUT any ADMM terms — no prices (lambda),
# no penalty weights (rho), no consensus targets (g_bar).  The planner optimizes
# all agents jointly; market prices emerge as duals of clearing constraints.
#
# Returns: welfare contribution = -(processing/production/import cost).
# Market revenues and expenditures cancel in the planner's aggregate.
# ------------------------------------------------------------------------------

function add_offtaker_agent_to_planner!(planner::Model, id::String, mod::Model,
                                        var_dict::Dict, W::AbstractArray)::JuMP.AbstractJuMPScalar
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    p = mod.ext[:parameters]
    gamma_GC = get(p, :gamma_GC, 0.42)   # 42% green certificate mandate
    agent_type = String(get(p, :Type, ""))
    # Nested Dict for convenient indexed access in JuMP @expression macros.
    W_dict = Dict(y => Dict(r => W[r, y] for r in JD) for y in JY)

    # ── GreenOfftaker (planner) ──────────────────────────────────────────
    if agent_type == "GreenOfftaker"
        H_buy_bar = p[:Capacity_H2_In]
        EP_sell_bar = p[:Capacity_EP_Out]
        alpha = get(p, :Alpha, 1.0)
        C_proc = get(p, :ProcessingCost, 0.0)

        h_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="h_buy_$(id)")
        gc_h_buy = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_h_buy_$(id)")
        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_sell_$(id)")

        # Capacity limits.
        @constraint(planner, [jh in JH, jd in JD, jy in JY], h_buy[jh, jd, jy] <= H_buy_bar)
        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= EP_sell_bar)
        # Stoichiometric link: EP = alpha * H2 (tight, no waste).
        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] == alpha * h_buy[jh, jd, jy])

        # Annual GC mandate: gamma_GC share of EP must be green-backed.
        @constraint(planner, [jy in JY],
            sum(W_dict[jy][jd] * gc_h_buy[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * sum(W_dict[jy][jd] * ep_sell[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Welfare = -(processing cost); market transfers cancel in aggregate.
        obj = @expression(planner,
            -sum(W_dict[jy][jd] * (C_proc * ep_sell[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
        )
        var_dict[:offtaker_h_buy][id] = h_buy
        var_dict[:offtaker_gc_h_buy][id] = gc_h_buy
        var_dict[:offtaker_ep_sell][id] = ep_sell
        return obj

    # ── GreyOfftaker (planner) ───────────────────────────────────────────
    elseif agent_type == "GreyOfftaker"
        EP_sell_bar = p[:Capacity]
        gamma_NH3 = p[:gamma_NH3]    # fraction of EP requiring H2 feedstock
        C_proc = p[:MarginalCost]

        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_sell_$(id)")
        gc_h_buy_G = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="gc_h_buy_G_$(id)")

        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= EP_sell_bar)

        # GC mandate on H2-equivalent portion: gc >= gamma_GC * gamma_NH3 * ep.
        @constraint(planner, [jy in JY],
            sum(W_dict[jy][jd] * gc_h_buy_G[jh, jd, jy] for jh in JH, jd in JD) >=
            gamma_GC * gamma_NH3 * sum(W_dict[jy][jd] * ep_sell[jh, jd, jy] for jh in JH, jd in JD)
        )

        # Welfare = -(production cost).
        obj = @expression(planner,
            -sum(W_dict[jy][jd] * (C_proc * ep_sell[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
        )
        var_dict[:offtaker_ep_sell][id] = ep_sell
        var_dict[:offtaker_gc_h_buy_G][id] = gc_h_buy_G
        return obj

    # ── EPImporter (planner) ─────────────────────────────────────────────
    # Simple price-taking EP supplier; no H2 or GC involvement.
    else  # EPImporter
        EP_sell_bar = p[:Capacity]
        C_proc = p[:ImportCost]       # import cost (EUR/MWh_EP)

        ep_sell = @variable(planner, [jh in JH, jd in JD, jy in JY], lower_bound=0, base_name="ep_import_$(id)")
        @constraint(planner, [jh in JH, jd in JD, jy in JY], ep_sell[jh, jd, jy] <= EP_sell_bar)

        # Welfare = -(import cost).
        obj = @expression(planner,
            -sum(W_dict[jy][jd] * (C_proc * ep_sell[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
        )
        var_dict[:offtaker_ep_sell_import][id] = ep_sell
        return obj
    end
end
