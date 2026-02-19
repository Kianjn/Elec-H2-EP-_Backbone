# ==============================================================================
# solve_H2_agent.jl — Re-set objective and solve electrolyzer
# ==============================================================================
#
# PURPOSE:
#   After ADMM_subroutine has updated λ, g_bar, ρ for elec, elec_GC, H2, H2_GC
#   on the model, this re-builds the objective (cost - revenue + penalties) and
#   calls optimize!(mod). Variables and constraints from build_H2_agent! are unchanged.
#
# ==============================================================================

function solve_H2_agent!(m::String, mod::Model, H2_market::Dict, H2_GC_market::Dict)
    # Only the objective is rebuilt each ADMM iteration. Variables and
    # constraints were defined once in build_H2_agent! and are invariant;
    # only prices (lambda), consensus targets (g_bar), and penalty (rho) change.
    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    W  = mod.ext[:parameters][:W]
    op_cost = mod.ext[:parameters][:OperationalCost]
    λ_elec     = mod.ext[:parameters][:λ_elec]
    g_bar_elec = mod.ext[:parameters][:g_bar_elec]
    ρ_elec     = mod.ext[:parameters][:ρ_elec]
    λ_elec_GC     = mod.ext[:parameters][:λ_elec_GC]
    g_bar_elec_GC = mod.ext[:parameters][:g_bar_elec_GC]
    ρ_elec_GC  = mod.ext[:parameters][:ρ_elec_GC]
    λ_H2     = mod.ext[:parameters][:λ_H2]
    g_bar_H2  = mod.ext[:parameters][:g_bar_H2]
    ρ_H2      = mod.ext[:parameters][:ρ_H2]
    # Hourly H2-GC price (full 3D), like all other markets.
    λ_H2_GC     = mod.ext[:parameters][:λ_H2_GC]
    g_bar_H2_GC = mod.ext[:parameters][:g_bar_H2_GC]
    ρ_H2_GC    = mod.ext[:parameters][:ρ_H2_GC]

    e_in      = mod.ext[:variables][:e_in]
    h2_out    = mod.ext[:variables][:h2_out]
    q_elec_gc = mod.ext[:variables][:q_elec_gc]
    q_h2gc    = mod.ext[:variables][:q_h2gc]
    cap_H2_y   = mod.ext[:variables][:cap_H2_y]
    beta_H2  = mod.ext[:variables][:beta_H2]
    gamma = get(mod.ext[:parameters], :γ, 1.0)
    F_cap = get(mod.ext[:parameters], :FixedCost_per_MW_Electrolyzer, 0.0)

    # Electrolyzer objective breakdown:
    #   + lambda_elec * e_in          electricity cost (buyer pays spot price)
    #   + lambda_elec_GC * q_elec_gc  green certificate cost (buyer purchases GCs)
    #   + op_cost * h2_out            operational cost of H2 production
    #   - lambda_H2 * h2_out          H2 revenue (seller receives H2 spot price)
    #   - lambda_H2_GC * q_h2gc       H2 GC revenue (seller earns H2 certificates)
    #   + four ADMM penalty terms     (one per market the electrolyzer participates in)
    #
    # Penalty net positions use negative signs for buyer markets:
    #   (-e_in - g_bar_elec)^2     electrolyzer is a BUYER of electricity, so
    #                               its net position is -e_in (demand is negative).
    #   (-q_elec_gc - g_bar_GC)^2  likewise a BUYER of elec GCs.
    #   (h2_out - g_bar_H2)^2      SELLER of H2 (positive net position).
    #   (q_h2gc - g_bar_H2_GC)^2   SELLER of H2 GCs (positive net position).
    mod.ext[:objective] = @objective(mod, Min,
        sum(W[jd, jy] * (
            λ_elec[jh, jd, jy]       * e_in[jh, jd, jy]
            + λ_elec_GC[jh, jd, jy]  * q_elec_gc[jh, jd, jy]
            + op_cost * h2_out[jh, jd, jy]
            - λ_H2[jh, jd, jy]       * h2_out[jh, jd, jy]
            - λ_H2_GC[jh, jd, jy]   * q_h2gc[jh, jd, jy]
        ) for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec/2 * W[jd, jy] * ((-e_in[jh, jd, jy])      - g_bar_elec[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_elec_GC/2 * W[jd, jy] * ((-q_elec_gc[jh, jd, jy]) - g_bar_elec_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2/2 * W[jd, jy] * (h2_out[jh, jd, jy]         - g_bar_H2[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        + sum(ρ_H2_GC/2 * W[jd, jy] * (q_h2gc[jh, jd, jy]      - g_bar_H2_GC[jh, jd, jy])^2 for jh in JH, jd in JD, jy in JY)
        # Fixed annualised investment cost, summed over model years (no W weighting).
        + F_cap * sum(cap_H2_y[jy] for jy in JY)
        # CVaR risk term (γ * β); γ = 0 ⇒ risk-neutral.
        + gamma * beta_H2)
    optimize!(mod)
    return nothing
end
