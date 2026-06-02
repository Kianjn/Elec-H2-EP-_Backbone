# ==============================================================================
# define_contract_parameters.jl — Add bilateral contract placeholders (PPA + HPA)
# ==============================================================================
#
# PURPOSE:
#   Called by market_exposure_contracts.jl.
#   Adds bilateral-contract participation flags and ADMM placeholder arrays.
#   Pushes agent IDs into agents[:ppa_market]/[:hpa_market] for imbalance
#   aggregation in ADMM_contracts.jl.
#
#   PER-VRES PPA MARKETS: Each VRES has its own PPA with the electrolyzer.
#   VRES gets single λ_ppa (its own); electrolyzer gets λ_ppa[vres_id],
#   g_bar_ppa[vres_id], etc. for each vres_id in agents[:ppa_vres].
#
#   PER-H2-PRODUCER HPA MARKETS: Each GreenProducer has its own HPA with the
#   GreenOfftaker side. GreenProducer gets single λ_hpa (its own); GreenOfftaker
#   gets λ_hpa[h2_id], g_bar_hpa[h2_id], etc. for each h2_id in agents[:hpa_h2].
#
# ARGUMENTS:
#   m      — Agent ID (string).
#   mod    — JuMP model (we write into mod.ext[:parameters]).
#   data   — Merged dict with nTimesteps, nReprDays, nYears.
#   agents — Dict; we push! m into agents[:ppa_market]/[:hpa_market] and
#            subtype lists (:ppa_vres, :hpa_h2).
#
# ==============================================================================

function define_contract_parameters!(m::String, mod::Model, data::Dict, agents::Dict)
    agent_type = String(get(mod.ext[:parameters], :Type, ""))
    in_ppa = agent_type in ("VRES", "GreenProducer")
    in_hpa = agent_type in ("GreenProducer", "GreenOfftaker")

    mod.ext[:parameters][:in_ppa_market] = in_ppa
    mod.ext[:parameters][:in_hpa_market] = in_hpa

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
            mod.ext[:parameters][:K_ppa]     = zeros(shp)
            mod.ext[:parameters][:g_bar_ppa] = zeros(shp)
            mod.ext[:parameters][:ρ_ppa]     = 1.0
            mod.ext[:parameters][:g_bar_ppa_cap] = 0.0
            mod.ext[:parameters][:ρ_ppa_cap]     = 1.0
        else
            ppa_vres = get(agents, :ppa_vres, String[])
            mod.ext[:parameters][:λ_ppa]     = Dict(v => zeros(shp) for v in ppa_vres)
            mod.ext[:parameters][:K_ppa]     = Dict(v => zeros(shp) for v in ppa_vres)
            mod.ext[:parameters][:g_bar_ppa] = Dict(v => zeros(shp) for v in ppa_vres)
            mod.ext[:parameters][:ρ_ppa]     = Dict(v => 1.0 for v in ppa_vres)
            mod.ext[:parameters][:g_bar_ppa_cap] = Dict(v => 0.0 for v in ppa_vres)
            mod.ext[:parameters][:ρ_ppa_cap]     = Dict(v => 1.0 for v in ppa_vres)
        end
    end

    if in_hpa
        push!(agents[:hpa_market], m)
        if agent_type == "GreenProducer"
            if !haskey(agents, :hpa_h2)
                agents[:hpa_h2] = String[]
            end
            push!(agents[:hpa_h2], m)
        end

        n_ts = data["nTimesteps"]
        n_rd = data["nReprDays"]
        n_yr = data["nYears"]
        shp  = (n_ts, n_rd, n_yr)

        if agent_type == "GreenProducer"
            mod.ext[:parameters][:λ_hpa]     = zeros(shp)
            mod.ext[:parameters][:K_hpa]     = zeros(shp)
            mod.ext[:parameters][:g_bar_hpa] = zeros(shp)
            mod.ext[:parameters][:ρ_hpa]     = 1.0
            mod.ext[:parameters][:g_bar_hpa_cap] = 0.0
            mod.ext[:parameters][:ρ_hpa_cap]     = 1.0
        else
            hpa_h2 = get(agents, :hpa_h2, String[])
            mod.ext[:parameters][:λ_hpa]     = Dict(v => zeros(shp) for v in hpa_h2)
            mod.ext[:parameters][:K_hpa]     = Dict(v => zeros(shp) for v in hpa_h2)
            mod.ext[:parameters][:g_bar_hpa] = Dict(v => zeros(shp) for v in hpa_h2)
            mod.ext[:parameters][:ρ_hpa]     = Dict(v => 1.0 for v in hpa_h2)
            mod.ext[:parameters][:g_bar_hpa_cap] = Dict(v => 0.0 for v in hpa_h2)
            mod.ext[:parameters][:ρ_hpa_cap]     = Dict(v => 1.0 for v in hpa_h2)
        end
    end

    return mod, agents
end
