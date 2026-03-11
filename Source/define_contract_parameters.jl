# ==============================================================================
# define_contract_parameters.jl — Add PPA (Power Purchase Agreement) parameters
# ==============================================================================
#
# PURPOSE:
#   Called for VRES and electrolyzer agents when running market_exposure_ppa.
#   Adds PPA-market participation flags and ADMM placeholder arrays.
#   Pushes the agent ID into agents[:ppa_market] for imbalance aggregation.
#
#   PER-VRES PPA MARKETS: Each VRES has its own PPA with the electrolyzer.
#   VRES gets single λ_ppa (its own); electrolyzer gets λ_ppa[vres_id],
#   g_bar_ppa[vres_id], etc. for each vres_id in agents[:ppa_vres].
#
# ARGUMENTS:
#   m      — Agent ID (string).
#   mod    — JuMP model (we write into mod.ext[:parameters]).
#   data   — Merged dict with nTimesteps, nReprDays, nYears.
#   agents — Dict; we push! m into agents[:ppa_market] and agents[:ppa_vres].
#
# ==============================================================================

function define_contract_parameters!(m::String, mod::Model, data::Dict, agents::Dict)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    in_ppa = agent_type in ("VRES", "GreenProducer")

    mod.ext[:parameters][:in_ppa_market] = in_ppa

    if in_ppa
        push!(agents[:ppa_market], m)
        if agent_type == "VRES"
            if !haskey(agents, :ppa_vres)
                agents[:ppa_vres] = String[]
            end
            push!(agents[:ppa_vres], m)
        end

        n_ts = data["nTimesteps"]
        n_rd = data["nReprDays"]
        n_yr = data["nYears"]
        shp  = (n_ts, n_rd, n_yr)

        if agent_type == "VRES"
            mod.ext[:parameters][:λ_ppa]     = zeros(shp)
            mod.ext[:parameters][:g_bar_ppa] = zeros(shp)
            mod.ext[:parameters][:ρ_ppa]     = 1.0
            mod.ext[:parameters][:g_bar_ppa_cap] = 0.0
            mod.ext[:parameters][:ρ_ppa_cap]     = 1.0
        else
            ppa_vres = get(agents, :ppa_vres, String[])
            mod.ext[:parameters][:λ_ppa]     = Dict(v => zeros(shp) for v in ppa_vres)
            mod.ext[:parameters][:g_bar_ppa] = Dict(v => zeros(shp) for v in ppa_vres)
            mod.ext[:parameters][:ρ_ppa]     = Dict(v => 1.0 for v in ppa_vres)
            mod.ext[:parameters][:g_bar_ppa_cap] = Dict(v => 0.0 for v in ppa_vres)
            mod.ext[:parameters][:ρ_ppa_cap]     = Dict(v => 1.0 for v in ppa_vres)
        end
    end

    return mod, agents
end
