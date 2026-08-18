# ==============================================================================
# update_rho_contracts.jl — Residual-balancing rho update (contracts)
# ==============================================================================
#
# PURPOSE:
#   Apply the same minimal residual-balancing rule used in update_rho.jl to:
#     - the five standard markets
#     - per-agent capacity split penalties
#     - contract submarket penalties (energy and capacity)
#
# RULE:
#     if rp > μ*rd -> increase ρ
#     if rd > μ*rp -> decrease ρ
#     else         -> keep ρ
#
# ==============================================================================

function update_rho_contracts!(ADMM_state::Dict, iter::Int)
    mod(iter, 1) == 0 || return

    μ = get(ADMM_state, "rho_balance_threshold", 1.2)
    eps_nonzero = 1e-9

    # Standard flow markets
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        isempty(ADMM_state["Residuals"]["Primal"][key]) && continue
        isempty(ADMM_state["Residuals"]["Dual"][key]) && continue
        rp = ADMM_state["Residuals"]["Primal"][key][end]
        rd = ADMM_state["Residuals"]["Dual"][key][end]
        ρ  = ADMM_state["ρ"][key][end]

        if !isfinite(rp) || !isfinite(rd)
            push!(ADMM_state["ρ"][key], ρ)
            continue
        end

        if key in ("elec", "elec_GC", "H2_GC")
            τ = 1.05
            ρ_max = 30.0
        else
            τ = 1.01
            ρ_max = 30.0
        end
        ρ_min = 1e-4

        if rp > μ * max(rd, eps_nonzero)
            push!(ADMM_state["ρ"][key], min(ρ_max, τ * ρ))
        elseif rd > μ * max(rp, eps_nonzero)
            push!(ADMM_state["ρ"][key], max(ρ_min, ρ / τ))
        else
            push!(ADMM_state["ρ"][key], ρ)
        end
    end

    # Capacity per-agent equality split (x_cap = z_cap)
    cap_state = get(ADMM_state, "Capacity", nothing)
    if cap_state !== nothing
        τ_cap = get(ADMM_state, "rho_cap_inc_factor", 1.05)
        ρ_max_cap = get(ADMM_state, "rho_cap_max", 30.0)
        ρ_min_cap = 0.10
        for m in get(cap_state, "agents", String[])
            isempty(cap_state["Primal"][m]) && continue
            isempty(cap_state["Dual"][m]) && continue
            rp = cap_state["Primal"][m][end]
            rd = cap_state["Dual"][m][end]
            ρ  = cap_state["ρ"][m][end]

            if !isfinite(rp) || !isfinite(rd)
                push!(cap_state["ρ"][m], ρ)
                continue
            end

            if rp > μ * max(rd, eps_nonzero)
                push!(cap_state["ρ"][m], min(ρ_max_cap, τ_cap * ρ))
            elseif rd > μ * max(rp, eps_nonzero)
                push!(cap_state["ρ"][m], max(ρ_min_cap, ρ / τ_cap))
            else
                push!(cap_state["ρ"][m], ρ)
            end
        end
    end

    # Contract submarkets (ppa/hpa): energy + capacity
    for contract_key in ("ppa", "hpa")
        haskey(ADMM_state, contract_key) || continue
        C = ADMM_state[contract_key]
        τ = 1.05
        ρ_max = 30.0
        ρ_min = 1e-4
        for id in keys(C["ρ"])
            isempty(C["Primal"][id]) && continue
            isempty(C["Dual"][id]) && continue
            rp = C["Primal"][id][end]
            rd = C["Dual"][id][end]
            ρ  = C["ρ"][id][end]

            if !isfinite(rp) || !isfinite(rd)
                push!(C["ρ"][id], ρ)
            elseif rp > μ * max(rd, eps_nonzero)
                push!(C["ρ"][id], min(ρ_max, τ * ρ))
            elseif rd > μ * max(rp, eps_nonzero)
                push!(C["ρ"][id], max(ρ_min, ρ / τ))
            else
                push!(C["ρ"][id], ρ)
            end

            isempty(C["Primal_cap"][id]) && continue
            isempty(C["Dual_cap"][id]) && continue
            rp_c = C["Primal_cap"][id][end]
            rd_c = C["Dual_cap"][id][end]
            ρ_c  = C["ρ_cap"][id][end]

            if !isfinite(rp_c) || !isfinite(rd_c)
                push!(C["ρ_cap"][id], ρ_c)
            elseif rp_c > μ * max(rd_c, eps_nonzero)
                push!(C["ρ_cap"][id], min(ρ_max, τ * ρ_c))
            elseif rd_c > μ * max(rp_c, eps_nonzero)
                push!(C["ρ_cap"][id], max(ρ_min, ρ_c / τ))
            else
                push!(C["ρ_cap"][id], ρ_c)
            end
        end
    end

    return nothing
end
