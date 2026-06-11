# ==============================================================================
# contract_settlement.jl — Volume and price-structure settlement for PPAs/HPAs
# ==============================================================================
#
# PPA (all ME contract entry points): pay-as-produced at scalar strike K (€/MWh,
# uniform over hours). K tracks bilateral λ each ADMM iteration; snapshotted at
# convergence for reporting.
#
# HPA volume modes (selected by entry point me_pap / me_top / me_sop):
#   pap — pay-as-produced:     buyer pays K·q, seller receives K·q
#   top — take-or-pay:         buyer pays K·max(q, cap), seller receives K·max(q, cap)
#   sop — send-or-pay:         buyer pays K·q; seller receives K·q − K·s, s = shortfall
#                              vs min(cap, available production)
#
# HPA price structure (data.yaml):
#   fixed — settlement uses locked strike K only
#   cfd   — decomposed as B·q + (K−B)·q (net K·q; B exposes benchmark risk in CVaR)
#
# ==============================================================================

"""Add auxiliary variables/constraints for ToP / SoP HPA volume logic."""
function add_hpa_volume_variables!(mod::Model; role::Symbol)
    p = mod.ext[:parameters]
    volume_mode = String(get(p, :hpa_volume_mode, "pap"))
    volume_mode == "pap" && return mod

    JH = mod.ext[:sets][:JH]
    JD = mod.ext[:sets][:JD]
    JY = mod.ext[:sets][:JY]
    vars = mod.ext[:variables]
    cons = mod.ext[:constraints]

    if role == :buyer
        hpa_h2 = collect(keys(mod.ext[:parameters][:λ_hpa]))
        h2_hpa_from = vars[:h2_hpa_from]
        hpa_cap = vars[:hpa_cap]
        if volume_mode == "top"
            vars[:hpa_top_shortfall] = Dict{String, Any}()
            for v in hpa_h2
                s = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0,
                              base_name="hpa_top_short_$(v)")
                vars[:hpa_top_shortfall][v] = s
                cons[Symbol("hpa_top_shortfall_lb_", v)] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                    s[jh, jd, jy] >= hpa_cap[v] - h2_hpa_from[v][jh, jd, jy])
                cons[Symbol("hpa_top_shortfall_ub_", v)] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                    s[jh, jd, jy] <= hpa_cap[v] - h2_hpa_from[v][jh, jd, jy])
            end
        end
    elseif role == :producer
        h2_hpa = vars[:h2_hpa]
        hpa_cap = vars[:hpa_cap]
        if volume_mode == "top"
            # s = cap - q (q <= cap from hpa_cap_limit); payment K·(q+s) = K·cap hourly ToP.
            vars[:hpa_top_shortfall] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0,
                                                 base_name="hpa_top_shortfall")
            cons[:hpa_top_shortfall_lb] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                vars[:hpa_top_shortfall][jh, jd, jy] >= hpa_cap - h2_hpa[jh, jd, jy])
            cons[:hpa_top_shortfall_ub] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                vars[:hpa_top_shortfall][jh, jd, jy] <= hpa_cap - h2_hpa[jh, jd, jy])
        elseif volume_mode == "sop"
            h2_out = vars[:h2_out]
            vars[:hpa_sop_oblig] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0,
                                             base_name="hpa_sop_oblig")
            vars[:hpa_sop_shortfall] = @variable(mod, [jh in JH, jd in JD, jy in JY], lower_bound=0,
                                                 base_name="hpa_sop_shortfall")
            cons[:hpa_sop_oblig_cap] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                vars[:hpa_sop_oblig][jh, jd, jy] <= hpa_cap)
            cons[:hpa_sop_oblig_prod] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                vars[:hpa_sop_oblig][jh, jd, jy] <= h2_out[jh, jd, jy])
            cons[:hpa_sop_shortfall] = @constraint(mod, [jh in JH, jd in JD, jy in JY],
                vars[:hpa_sop_shortfall][jh, jd, jy] >= vars[:hpa_sop_oblig][jh, jd, jy] - h2_hpa[jh, jd, jy])
        end
    end
    return mod
end

"""PPA payment on delivered energy (always PaP): +K·g for seller, cost for buyer."""
function ppa_payment_term(K::AbstractArray, q::JuMP.AbstractJuMPScalar, W, jh, jd, jy)
    return K[jh, jd, jy] * q
end

"""
    hpa_settlement_terms(p, K, B, q, W, jh, jd, jy; side=:buyer)

Return (fixed_leg, cfd_leg) contributions to agent loss at one slot.
Buyer: positive = cost. Seller: returned values are subtracted in caller (revenue).
"""
function hpa_settlement_terms(p, K::AbstractArray, B::Union{Nothing, AbstractArray},
                              q::JuMP.AbstractJuMPScalar, W, jh, jd, jy; side::Symbol=:buyer,
                              top_shortfall=nothing)
    volume_mode = String(get(p, :hpa_volume_mode, "pap"))
    structure = String(get(p, :hpa_price_structure, "fixed"))
    Buse = B === nothing ? K : B

    q_pay = q
    if volume_mode == "top" && top_shortfall !== nothing
        q_pay = q + top_shortfall[jh, jd, jy]
    end

    fixed_leg = K[jh, jd, jy] * q_pay
    if structure == "cfd"
        cfd_leg = (K[jh, jd, jy] - Buse[jh, jd, jy]) * q_pay
        return fixed_leg, cfd_leg
    end
    return fixed_leg, 0.0
end

"""Seller-side HPA revenue including SoP penalty."""
function hpa_seller_revenue_terms(p, K::AbstractArray, B::Union{Nothing, AbstractArray},
                                  h2_hpa, W, jh, jd, jy; top_shortfall=nothing, sop_shortfall=nothing)
    volume_mode = String(get(p, :hpa_volume_mode, "pap"))
    structure = String(get(p, :hpa_price_structure, "fixed"))
    Buse = B === nothing ? K : B

    q = h2_hpa[jh, jd, jy]
    q_pay = q
    if volume_mode == "top" && top_shortfall !== nothing
        q_pay = q + top_shortfall[jh, jd, jy]
    end

    rev = K[jh, jd, jy] * q_pay
    if structure == "cfd"
        rev += (K[jh, jd, jy] - Buse[jh, jd, jy]) * q_pay
    end
    if volume_mode == "sop" && sop_shortfall !== nothing
        rev -= K[jh, jd, jy] * sop_shortfall[jh, jd, jy]
    end
    return rev
end

"""Sum W-weighted PPA buyer cost for electrolyzer (per VRES leg)."""
function sum_ppa_buyer_cost(mod, ppa_vres, W, JH, JD, JY)
    K_ppa = mod.ext[:parameters][:K_ppa]
    g_ppa_from = mod.ext[:variables][:g_ppa_from]
    return sum(W[jd, jy] * sum(K_ppa[v][jh, jd, jy] * g_ppa_from[v][jh, jd, jy] for v in ppa_vres)
               for jh in JH, jd in JD, jy in JY)
end

"""Sum W-weighted PPA seller revenue for VRES."""
function sum_ppa_seller_revenue(mod, W, JH, JD, JY)
    K_ppa = mod.ext[:parameters][:K_ppa]
    g_ppa = mod.ext[:variables][:g_ppa]
    return sum(W[jd, jy] * K_ppa[jh, jd, jy] * g_ppa[jh, jd, jy] for jh in JH, jd in JD, jy in JY)
end

"""Sum W-weighted HPA buyer cost for GreenOfftaker (one scenario year jy)."""
function sum_hpa_buyer_cost_jy(mod, hpa_h2, jy, W, JH, JD)
    p = mod.ext[:parameters]
    h2_hpa_from = mod.ext[:variables][:h2_hpa_from]
    top_sf = get(mod.ext[:variables], :hpa_top_shortfall, nothing)
    cost = 0.0
    for v in hpa_h2
        K = p[:K_hpa][v]
        B = haskey(p, :B_hpa) ? p[:B_hpa][v] : K
        ts = top_sf === nothing ? nothing : get(top_sf, v, nothing)
        for jd in JD, jh in JH
            fl, cl = hpa_settlement_terms(p, K, B, h2_hpa_from[v][jh, jd, jy], W, jh, jd, jy;
                                          side=:buyer, top_shortfall=ts)
            cost += W[jd, jy] * (fl + cl)
        end
    end
    return cost
end

"""Sum W-weighted HPA seller revenue for GreenProducer (one scenario year jy)."""
function sum_hpa_seller_revenue_jy(mod, jy, W, JH, JD)
    p = mod.ext[:parameters]
    h2_hpa = mod.ext[:variables][:h2_hpa]
    K = p[:K_hpa]
    B = get(p, :B_hpa, K)
    top_sf = get(mod.ext[:variables], :hpa_top_shortfall, nothing)
    sop_sf = get(mod.ext[:variables], :hpa_sop_shortfall, nothing)
    rev = 0.0
    for jd in JD, jh in JH
        rev += W[jd, jy] * hpa_seller_revenue_terms(p, K, B, h2_hpa, W, jh, jd, jy;
            top_shortfall=top_sf, sop_shortfall=sop_sf)
    end
    return rev
end

"""Sum W-weighted PPA buyer cost (one scenario year jy)."""
function sum_ppa_buyer_cost_jy(mod, ppa_vres, jy, W, JH, JD)
    K_ppa = mod.ext[:parameters][:K_ppa]
    g_ppa_from = mod.ext[:variables][:g_ppa_from]
    return sum(W[jd, jy] * sum(K_ppa[v][jh, jd, jy] * g_ppa_from[v][jh, jd, jy] for v in ppa_vres)
               for jh in JH, jd in JD)
end

"""Sum W-weighted PPA seller revenue (one scenario year jy)."""
function sum_ppa_seller_revenue_jy(mod, jy, W, JH, JD)
    K_ppa = mod.ext[:parameters][:K_ppa]
    g_ppa = mod.ext[:variables][:g_ppa]
    return sum(W[jd, jy] * K_ppa[jh, jd, jy] * g_ppa[jh, jd, jy] for jh in JH, jd in JD)
end

"""Sum W-weighted HPA buyer cost for GreenOfftaker (full horizon)."""
function sum_hpa_buyer_cost(mod, hpa_h2, W, JH, JD, JY)
    return sum(sum_hpa_buyer_cost_jy(mod, hpa_h2, jy, W, JH, JD) for jy in JY)
end

"""Sum W-weighted HPA seller revenue for GreenProducer (full horizon)."""
function sum_hpa_seller_revenue(mod, W, JH, JD, JY)
    return sum(sum_hpa_seller_revenue_jy(mod, jy, W, JH, JD) for jy in JY)
end
