# ==============================================================================
# cap_admm_helpers.jl — Shared helpers for scalar capacity ADMM (x = z split)
# ==============================================================================
#
# Capacity consensus stores z-history as Vector{Vector{Float64}} where each
# snapshot is a length-1 (or legacy length-nYears) vector. Always push via
# _cap_z_push! so a bare Float64 is never appended to z history.

"""Extract scalar capacity / dual value from scalar or legacy vector storage."""
_cap_scalar(x) = x isa Real ? Float64(x) : (isempty(x) ? 0.0 : Float64(x[1]))

"""Normalize a capacity target to Vector{Float64} for z-history storage."""
function _cap_z_vec(val)
    val isa Real && return [Float64(val)]
    return Float64.(collect(val))
end

"""Append one capacity-target snapshot to z history."""
function _cap_z_push!(hist::Vector{Vector{Float64}}, val)
    push!(hist, _cap_z_vec(val))
    return hist
end

"""Read installed capacity vector from ADMM results (coalition or legacy single-cap)."""
function _agent_cap_vec_from_results(m::String, results::Dict)
    if haskey(results, "Cap_Merged") && !isempty(get(results["Cap_Merged"], m, []))
        return copy(results["Cap_Merged"][m][end])
    end
    if haskey(results, "Cap_VRES") && !isempty(get(results["Cap_VRES"], m, []))
        return copy(results["Cap_VRES"][m][end])
    end
    if haskey(results, "Cap_Elec_H2") && !isempty(get(results["Cap_Elec_H2"], m, []))
        return copy(results["Cap_Elec_H2"][m][end])
    end
    if haskey(results, "Cap_EP_Green") && !isempty(get(results["Cap_EP_Green"], m, []))
        return copy(results["Cap_EP_Green"][m][end])
    end
    return Float64[]
end

"""Primal capacity residual (L2 for multi-cap coalitions)."""
function _cap_primal_residual(cap_vec::AbstractVector{<:Real}, z_vec::AbstractVector{<:Real})
    isempty(cap_vec) && return 0.0
    length(cap_vec) == 1 && return abs(_cap_scalar(cap_vec) - _cap_scalar(z_vec))
    d = Float64.(cap_vec) .- Float64.(z_vec)
    return sqrt(sum(abs2, d))
end

"""Dual ascent update for scalar or vector capacity split."""
function _cap_dual_ascent(λ_prev, ρ_m::Real, cap_vec::AbstractVector{<:Real}, z_vec::AbstractVector{<:Real})
    isempty(cap_vec) && return _cap_z_vec(λ_prev)
    if length(cap_vec) == 1
        return [_cap_scalar(λ_prev) + ρ_m * (_cap_scalar(cap_vec) - _cap_scalar(z_vec))]
    end
    λ0 = λ_prev isa AbstractVector ? Float64.(λ_prev) : fill(_cap_scalar(λ_prev), length(cap_vec))
    length(λ0) != length(cap_vec) && (λ0 = resize!(copy(λ0), length(cap_vec)))
    return [λ0[i] + ρ_m * (Float64(cap_vec[i]) - Float64(z_vec[i])) for i in eachindex(cap_vec)]
end

"""Read one scalar installed capacity from SP_Capacities.csv rows for an agent."""
function _sp_cap_scalar(rows::DataFrame)
    nrow(rows) == 0 && return nothing
    v = rows.cap[1]
    v isa Real && return Float64(v)
    v isa AbstractVector && !isempty(v) && return Float64(v[1])
    return Float64(v)
end

"""Vector z warm-start for merged agents: one SP capacity per member slot."""
function merged_cap_z_from_sp(cap_df::DataFrame, member_ids, n_cap::Int)
    z = zeros(n_cap)
    for (i, mid) in enumerate(member_ids)
        i > n_cap && break
        rows = cap_df[cap_df.AgentID .== String(mid), :]
        val = _sp_cap_scalar(rows)
        val !== nothing && (z[i] = val)
    end
    return z
end
