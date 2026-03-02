# ==============================================================================
# define_contract_parameters.jl — Add bilateral contract parameters to agents
# ==============================================================================
#
# PURPOSE:
#   Called for VRES and electrolyzer agents when running market_exposure_contracts.
#   Adds contract-market participation flags and ADMM placeholder arrays
#   (λ_contract, g_bar_contract, ρ_contract) to mod.ext[:parameters].
#   Pushes the agent ID into agents[:contract_market] for imbalance aggregation.
#
#   Only VRES and GreenProducer (electrolyzer) participate in the contract pool.
#
# ARGUMENTS:
#   m      — Agent ID (string).
#   mod    — JuMP model (we write into mod.ext[:parameters]).
#   data   — Merged dict with nTimesteps, nReprDays, nYears.
#   agents — Dict; we push! m into agents[:contract_market] if applicable.
#
# ==============================================================================

function define_contract_parameters!(m::String, mod::Model, data::Dict, agents::Dict)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    in_contract = agent_type in ("VRES", "GreenProducer")

    mod.ext[:parameters][:in_contract_market] = in_contract

    if in_contract
        push!(agents[:contract_market], m)

        n_ts = data["nTimesteps"]
        n_rd = data["nReprDays"]
        n_yr = data["nYears"]
        shp  = (n_ts, n_rd, n_yr)

        # ADMM placeholders for contract pool (3D): g_contract cleared at λ_contract (€/MWh).
        # Pay-as-produced: H2 producer pays per MWh delivered. contract_cap (MW) is the
        # capacity commitment; both parties agree via penalty (no separate capacity price).
        mod.ext[:parameters][:λ_contract]     = zeros(shp)
        mod.ext[:parameters][:g_bar_contract] = zeros(shp)
        mod.ext[:parameters][:ρ_contract]     = 1.0

        # Consensus on contract_cap: both must agree (penalty only, no price).
        mod.ext[:parameters][:g_bar_contract_cap] = 0.0
        mod.ext[:parameters][:ρ_contract_cap]     = 1.0
    end

    return mod, agents
end
