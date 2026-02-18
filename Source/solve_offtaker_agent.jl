# ==============================================================================
# solve_offtaker_agent.jl — Re-set objective and solve offtaker (green/grey/importer)
# ==============================================================================
#
# PURPOSE:
#   Re-builds the offtaker objective with current λ, g_bar, ρ and calls
#   optimize!(mod). Green: cost (H2, H2_GC, processing) - revenue (EP) + penalties.
#   Grey: cost (MC*ep, H2_GC) - revenue (EP) + penalties. Importer: cost (import) - revenue (EP) + penalty.
#
# ==============================================================================

function solve_offtaker_agent!(m::String, mod::Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W = mod.ext[:parameters][:W]
    gamma_GC = get(mod.ext[:parameters], :gamma_GC, 0.42)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2 = mod.ext[:parameters][:g_bar_H2]
    ρ_H2     = mod.ext[:parameters][:ρ_H2]
    # Hourly H2-GC price (full 3D), same as all other markets.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]
    λ_EP     = mod.ext[:parameters][:λ_EP]
    g_bar_EP = mod.ext[:parameters][:g_bar_EP]
    ρ_EP     = mod.ext[:parameters][:ρ_EP]

    if agent_type == "GreenOfftaker"
        # GreenOfftaker objective:
        #   min  sum W * ( lambda_H2*h2_in              [H2 purchase cost]
        #                + lambda_H2_GC*q_h2gc          [H2 GC purchase cost]
        #                + proc_cost*ep                 [processing / conversion cost]
        #                - lambda_EP*ep )                [end-product revenue]
        #      + rho_H2/2    * sum W * (-h2_in - g_bar_H2)^2       [ADMM H2 penalty; buyer => -h2_in]
        #      + rho_H2_GC/2 * sum W * (-q_h2gc - g_bar_H2_GC)^2  [ADMM H2 GC penalty; buyer => -q_h2gc]
        #      + rho_EP/2    * sum W * (ep - g_bar_EP)^2            [ADMM EP penalty; seller => +ep]
        h2_in  = mod.ext[:variables][:h2_in]
        q_h2gc = mod.ext[:variables][:q_h2gc]
        ep     = mod.ext[:variables][:ep]
        cap_EP_y = mod.ext[:variables][:cap_EP_y]
        proc_cost = get(mod.ext[:parameters], :ProcessingCost, 0.0)
        F_cap = get(mod.ext[:parameters], :FixedCost_per_MW_EP_Out, 0.0)
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (λ_H2[jh, jd, jy] * h2_in[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] + proc_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2/2 * W[jd, jy] * ((-h2_in[jh, jd, jy]) - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            # Fixed annualised investment cost, summed over model years (no W weighting).
            + F_cap * sum(cap_EP_y[jy] for jy in JY))
    elseif agent_type == "GreyOfftaker"
        # GreyOfftaker objective:
        #   min  sum W * ( MC*ep                        [marginal production cost]
        #                + lambda_H2_GC*q_h2gc          [H2 GC purchase cost (compliance)]
        #                - lambda_EP*ep )                [end-product revenue]
        #      + rho_H2_GC/2 * sum W * (-q_h2gc - g_bar_H2_GC)^2  [ADMM H2 GC penalty; buyer]
        #      + rho_EP/2    * sum W * (ep - g_bar_EP)^2            [ADMM EP penalty; seller]
        # Note: GreyOfftaker does NOT buy physical H2 (no H2 market penalty),
        # but must purchase H2 GCs for regulatory compliance.
        ep     = mod.ext[:variables][:ep]
        q_h2gc = mod.ext[:variables][:q_h2gc]
        MC     = mod.ext[:parameters][:MarginalCost]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (MC * ep[jh, jd, jy] + λ_H2_GC[jh, jd, jy] * q_h2gc[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_H2_GC/2 * W[jd, jy] * ((-q_h2gc[jh, jd, jy]) - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    else
        # Importer objective (fallback for any non-Green/Grey offtaker):
        #   min  sum W * ( imp_cost*ep                  [import cost per unit of EP]
        #                - lambda_EP*ep )                [end-product revenue]
        #      + rho_EP/2 * sum W * (ep - g_bar_EP)^2   [ADMM EP penalty; seller]
        # The importer participates only in the EP market (no H2 or H2_GC).
        ep = mod.ext[:variables][:ep]
        imp_cost = mod.ext[:parameters][:ImportCost]
        mod.ext[:objective] = @objective(mod, Min,
            sum(W[jd, jy] * (imp_cost * ep[jh, jd, jy] - λ_EP[jh, jd, jy] * ep[jh, jd, jy]) for jh in JH, jd in JD, jy in JY)
            + sum(ρ_EP/2 * W[jd, jy] * (ep[jh, jd, jy] - g_bar_EP[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY))
    end
    optimize!(mod)
    return nothing
end
