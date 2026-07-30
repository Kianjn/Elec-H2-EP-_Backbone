# ==============================================================================
# contract_strike.jl — Contract strikes, benchmarks, and convergence snapshots
# ==============================================================================
#
# Bilateral PPAs and HPAs are cleared by ADMM alongside spot markets. Each ADMM
# iteration updates:
#   - λ_contract (dual / clearing price for contract energy imbalances)
#   - provisional settlement strike K (scalar €/MWh, broadcast to all hours)
#   - contract capacity C: single shared scalar per bilateral link, updated
#     outside agent subproblems (see contract_capacity.jl)
#
# "Fixed" contract terms mean fixed across hourly timesteps (jh, jd, jy), not
# fixed at a mid-run ADMM iteration. K and C may change from one ADMM iteration
# to the next until global convergence. At convergence, finalize_contract_terms!
# snapshots scalar K and C into ADMM["ContractStrikes"] for reporting.
#
# ==============================================================================

if !isdefined(@__MODULE__, :shared_contract_capacity)
    include(joinpath(@__DIR__, "contract_capacity.jl"))
end

"""W-weighted mean over representative days and hours → one scalar €/MWh."""
function _contract_strike_scalar(arr::AbstractArray{<:Real}, W::AbstractMatrix)
    n_ts, n_rd, n_yr = size(arr)
    total_w = 0.0
    acc = 0.0
    for jy in 1:n_yr, jd in 1:n_rd
        w = Float64(W[jd, jy])
        block = Float64.(arr[:, jd, jy])
        acc += w * sum(block)
        total_w += w * n_ts
    end
    return total_w > 0 ? acc / total_w : mean(arr)
end

"""Broadcast a scalar strike to a uniform 3D field (same €/MWh every hour)."""
function _strike_field(K_scalar::Real, shp::Tuple)
    fill(Float64(K_scalar), shp...)
end

"""
Read NG / grey-chain proxy prices (EUR/MWh) used by the `NG` contract benchmarks.

A contract strike is a single scalar, but the gas price now varies across the
scenario grid (§9.8). The benchmark therefore uses the EXPECTED gas level — the
mean of `Scenarios.gas_price_multipliers` — so the reference price sits at the
centre of the uncertainty set rather than at one arbitrary corner of it.

Both scalars are derived from the `Fuel` block via the same formulas the agents
themselves use, so the benchmark cannot drift away from the costs it references.
Legacy hard-coded `FinalMarginalCost` / `MarginalCost` entries are still honoured
when the derived inputs are absent.
"""
function _ng_benchmark_scalars(data::Dict)
    power = get(data, "Power", Dict())
    off = get(data, "Hydrogen_Offtaker", Dict())
    fuel = get(data, "Fuel", Dict())

    mults = Float64.(get(get(data, "Scenarios", Dict()), "gas_price_multipliers", [1.0]))
    gas_mult = isempty(mults) ? 1.0 : sum(mults) / length(mults)

    conv_mc = 100.0
    for (_, blk) in power
        String(get(blk, "Type", "")) == "Conventional" || continue
        peak = get(blk, "PeakTechnology", nothing)
        conv_mc = if peak !== nothing && !isempty(fuel)
            thermal_srmc(peak, fuel, gas_mult)   # OCGT tail: the price-setting unit
        else
            Float64(get(blk, "FinalMarginalCost", get(blk, "MarginalCost", conv_mc)))
        end
        break
    end

    grey_ep = 180.0
    grey_alpha = 1.0
    for (_, blk) in off
        String(get(blk, "Type", "")) == "GreyOfftaker" || continue
        grey_alpha = Float64(get(blk, "Alpha", 1.0))
        grey_ep = if haskey(blk, "GasIntensity") && !isempty(fuel)
            Float64(blk["GasIntensity"]) * Float64(get(fuel, "GasPrice", 0.0)) * gas_mult +
            Float64(get(blk, "CO2Intensity", 0.0)) * Float64(get(fuel, "CO2Price", 0.0)) +
            Float64(get(blk, "VariableOM", 0.0))
        else
            Float64(get(blk, "MarginalCost", grey_ep))
        end
        break
    end
    grey_alpha = max(grey_alpha, 1e-9)
    return (
        elec_ng = conv_mc,
        h2_ng = grey_ep / grey_alpha,
        ep_grey = grey_ep,
    )
end

"""
    contract_benchmark_field(benchmark, results, data, shp; λ_clearing=nothing)

Return 3D benchmark price field B for strike settlement and CfD floating leg.
"""
function contract_benchmark_field(benchmark::String, results::Dict, data::Dict, shp::Tuple;
                                  λ_clearing::Union{Nothing, AbstractArray{<:Real}}=nothing)
    benchmark = lowercase(strip(benchmark))
    ng = _ng_benchmark_scalars(data)
    if benchmark in ("negotiated", "internal", "endogenous")
        λ_clearing === nothing && return fill(0.0, shp...)
        return copy(λ_clearing)
    elseif benchmark in ("electricity", "elec")
        return copy(results["λ"]["elec"][end])
    elseif benchmark in ("ammonia", "ep", "end_product")
        return copy(results["λ"]["EP"][end])
    elseif benchmark in ("ng", "natural_gas", "gas")
        return fill(ng.h2_ng, shp...)
    elseif benchmark in ("ng_electricity", "ng_elec")
        return fill(ng.elec_ng, shp...)
    else
        @warn "Unknown contract price_benchmark '$benchmark'; falling back to negotiated clearing."
        λ_clearing === nothing && return fill(0.0, shp...)
        return copy(λ_clearing)
    end
end

"""ADMM consensus target ḡ_cap for bilateral contract capacity (scalar). Legacy — unused with shared C."""
function contract_cap_g_bar(prev_net_cap::Real, imb_cap::Real, n_contract::Int)
    return prev_net_cap - (1.0 / (n_contract + 1)) * imb_cap
end

function _consensus_contract_capacity(results::Dict, cap_key::String, id::String)
    pool = cap_key == "ppa_cap" ? :ppa : :hpa
    return shared_contract_capacity(results, pool, id)
end

"""
    init_contract_strike_state!(ADMM, admm_data, ppa_market, hpa_market)

Initialize ContractStrikes storage (filled by finalize_contract_terms! at end of ADMM).
"""
function init_contract_strike_state!(ADMM::Dict, admm_data::Dict, ppa_market::Dict, hpa_market::Dict)
    ADMM["ContractStrikes"] = Dict(
        "K_ppa" => Dict{String, Float64}(),
        "K_hpa" => Dict{String, Float64}(),
        "C_ppa" => Dict{String, Float64}(),
        "C_hpa" => Dict{String, Float64}(),
        "finalized" => false,
        "final_iter" => 0,
    )
    return nothing
end

"""
    finalize_contract_terms!(ADMM_state, results, data, ppa_market, hpa_market, shp, W)

Snapshot converged scalar strikes K and capacities C from the final ADMM iterate.
"""
function finalize_contract_terms!(ADMM_state::Dict, results::Dict, data::Dict,
                                  ppa_market::Dict, hpa_market::Dict, shp::Tuple,
                                  W::AbstractMatrix)
    strikes = ADMM_state["ContractStrikes"]
    strikes["K_ppa"] = Dict{String, Float64}()
    strikes["K_hpa"] = Dict{String, Float64}()
    strikes["C_ppa"] = Dict{String, Float64}()
    strikes["C_hpa"] = Dict{String, Float64}()

    for vres_id in get(ppa_market, "ppa_vres", String[])
        λ = results["λ_ppa"][vres_id][end]
        B = contract_benchmark_field("negotiated", results, data, shp; λ_clearing=λ)
        strikes["K_ppa"][vres_id] = _contract_strike_scalar(B, W)
        strikes["C_ppa"][vres_id] = _consensus_contract_capacity(results, "ppa_cap", vres_id)
    end

    for h2_id in get(hpa_market, "hpa_h2", String[])
        cfg = get(get(hpa_market, "per_h2", Dict()), h2_id, Dict())
        bench = String(get(cfg, "price_benchmark", get(hpa_market, "price_benchmark", "negotiated")))
        λ = results["λ_hpa"][h2_id][end]
        B = contract_benchmark_field(bench, results, data, shp; λ_clearing=λ)
        strikes["K_hpa"][h2_id] = _contract_strike_scalar(B, W)
        strikes["C_hpa"][h2_id] = _consensus_contract_capacity(results, "hpa_cap", h2_id)
    end

    strikes["finalized"] = true
    strikes["final_iter"] = get(ADMM_state, "n_iter", 0)
    return nothing
end

"""Return finalized strike K for reporting, or nothing if ADMM has not finished."""
function final_contract_strike(ADMM_state::Dict, pool::Symbol, id::String)
    strikes = get(ADMM_state, "ContractStrikes", Dict())
    get(strikes, "finalized", false) || return nothing
    key = pool == :ppa ? "K_ppa" : "K_hpa"
    haskey(strikes, key) || return nothing
    haskey(strikes[key], id) || return nothing
    return strikes[key][id]
end

"""Return finalized contract capacity C (MW) for reporting."""
function final_contract_capacity(ADMM_state::Dict, pool::Symbol, id::String)
    strikes = get(ADMM_state, "ContractStrikes", Dict())
    get(strikes, "finalized", false) || return nothing
    key = pool == :ppa ? "C_ppa" : "C_hpa"
    haskey(strikes, key) || return nothing
    haskey(strikes[key], id) || return nothing
    return strikes[key][id]
end

"""Update K_ppa for one VRES or one electrolyzer vres leg (scalar, uniform over hours)."""
function update_ppa_strike!(mod::Model, vres_id::String, vres_cfg::Dict,
                            results::Dict, data::Dict, ADMM_state::Dict, shp::Tuple, W::AbstractMatrix)
    p = mod.ext[:parameters]
    λ = results["λ_ppa"][vres_id][end]
    B = contract_benchmark_field("negotiated", results, data, shp; λ_clearing=λ)
    K = _strike_field(_contract_strike_scalar(B, W), shp)
    if p[:K_ppa] isa Dict
        p[:K_ppa][vres_id] = K
    else
        p[:K_ppa] = K
    end
    return nothing
end

"""Update K_hpa / B_hpa for one HPA leg (scalar strike, uniform over hours)."""
function update_hpa_strike!(mod::Model, h2_id::String, h2_cfg::Dict,
                            results::Dict, data::Dict, ADMM_state::Dict, shp::Tuple, W::AbstractMatrix)
    p = mod.ext[:parameters]
    bench = String(get(h2_cfg, "price_benchmark", "negotiated"))
    structure = String(get(h2_cfg, "price_structure", "fixed"))
    λ = results["λ_hpa"][h2_id][end]
    B = contract_benchmark_field(bench, results, data, shp; λ_clearing=λ)
    K = _strike_field(_contract_strike_scalar(B, W), shp)
    if p[:K_hpa] isa Dict
        p[:K_hpa][h2_id] = K
        haskey(p, :B_hpa) || (p[:B_hpa] = Dict{String, Array{Float64, 3}}())
        p[:B_hpa][h2_id] = B
    else
        p[:K_hpa] = K
        p[:B_hpa] = B
    end
    p[:hpa_price_structure] = structure
    return nothing
end
