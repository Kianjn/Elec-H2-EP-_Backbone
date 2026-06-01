# ==============================================================================
# MARKET EXPOSURE (CONTRACTS) SCRIPT: ADMM WITH BILATERAL PPA + HPA
# By Kian Jafarinejad - PhD Researcher at TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
#
# PURPOSE:
#   Third entry point alongside market_exposure.jl and social_planner.jl.
#   Identical to market_exposure.jl except that it adds two bilateral
#   pay-as-produced contract pools:
#
#   - PPA (VRES -> GreenProducer electricity+elec_GC):
#     VRES commits ppa_cap (MW) of capacity; that capacity is REMOVED from
#     BOTH the electricity market (EOM) and elec_GC market. Real-world PPAs
#     bundle electricity + elec_GC — buyer receives both as a package.
#     The electrolyzer buys that capacity pay-as-produced: receives g_ppa
#     at each timestep, pays λ_ppa per MWh actually delivered (bundled price).
#   - When VRES has no output (e.g. night for solar), g_ppa = 0, so nothing
#     delivered and nothing paid.
#   - HPA (GreenProducer -> GreenOfftaker hydrogen+H2_GC equivalent):
#     GreenProducer commits hpa_cap (MW_H2) and delivers h2_hpa
#     pay-as-produced at λ_hpa per MWh_H2 delivered.
#     Contracted H2 capacity is removed from pool H2/H2_GC sales.
#   - ADMM clears both energy (3D) and capacity (scalar consensus) for each pool.
#
#   social_planner.jl and market_exposure.jl are UNCHANGED. This script uses
#   PPA/HPA-specific build/solve/ADMM/save modules.
#
# HOW TO RUN:
#   From the project root:  julia market_exposure_contracts.jl
#
# OUTPUT:
#   market_exposure_contracts_results/ — same major ADMM CSVs as market_exposure_results
#   plus PPAs.csv, HPAs.csv, Green_Agents_Detail.csv, and contract columns in
#   convergence/diagnostics CSVs.
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

using Pkg
Pkg.activate(@__DIR__)

# ------------------------------------------------------------------------------
# SECTION 2: PACKAGE LOADING
# ------------------------------------------------------------------------------

using JuMP
using Gurobi
using DataFrames
using CSV
using YAML
using DataStructures
using ProgressBars
using Printf
using TimerOutputs
using ArgParse
using Statistics
using Base.Threads: @spawn
using Base: split

const GUROBI_ENV = Gurobi.Env()

# ------------------------------------------------------------------------------
# SECTION 3: DIRECTORY SETUP
# ------------------------------------------------------------------------------

const home_dir = @__DIR__

# ------------------------------------------------------------------------------
# SECTION 4: FUNCTION LOADING (SOURCE FILES)
# ------------------------------------------------------------------------------

# Parameter definitions (same as market_exposure)
include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))

# Contracts-specific: adds PPA/HPA flags and placeholders
include(joinpath(home_dir, "Source", "define_contract_parameters.jl"))
include(joinpath(home_dir, "Source", "define_contract_market_parameters.jl"))

# Market definitions
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))

# Model building: use contracts versions for power/H2/offtaker where needed
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))

# ADMM and solving: contracts versions (include PPA + HPA markets)
include(joinpath(home_dir, "Source", "define_results.jl"))
include(joinpath(home_dir, "Source", "define_results_contracts.jl"))
include(joinpath(home_dir, "Source", "ADMM.jl"))
include(joinpath(home_dir, "Source", "ADMM_contracts.jl"))
include(joinpath(home_dir, "Source", "ADMM_subroutine.jl"))
include(joinpath(home_dir, "Source", "ADMM_subroutine_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_power_agent.jl"))
include(joinpath(home_dir, "Source", "solve_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_H2_agent.jl"))
include(joinpath(home_dir, "Source", "solve_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "solve_offtaker_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "solve_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "solve_EP_demand_agent.jl"))
include(joinpath(home_dir, "Source", "update_rho.jl"))
include(joinpath(home_dir, "Source", "update_rho_contracts.jl"))
include(joinpath(home_dir, "Source", "compute_agent_objective.jl"))
include(joinpath(home_dir, "Source", "save_results_contracts.jl"))

# ------------------------------------------------------------------------------
# SECTION 5: DATA LOADING
# ------------------------------------------------------------------------------

data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))

ts = Dict()
order_matrix = Dict()
repr_days = Dict()

gen = data["General"]
base_year = haskey(gen, "base_year") ? gen["base_year"] : 2021
n_years  = haskey(data["ADMM"], "nScenarioYears") ? data["ADMM"]["nScenarioYears"] :
           (haskey(gen, "nYears") ? gen["nYears"] : 1)
run_general = merge(gen, Dict("nYears" => n_years))
data_run = copy(data)
data_run["General"] = run_general
years = Dict(i => base_year + (i - 1) for i in 1:n_years)

for y in values(years)
    ts[y] = CSV.read(joinpath(home_dir, "Input", "timeseries_$(y).csv"), DataFrame)
    order_matrix[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "ordering_variable.csv"), delim=",", DataFrame)
    repr_days[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "decision_variables_short.csv"), delim=",", DataFrame)
end

# ------------------------------------------------------------------------------
# SECTION 6: RESULTS FOLDER
# ------------------------------------------------------------------------------

if !isdir(joinpath(home_dir, "market_exposure_contracts_results"))
    mkdir(joinpath(home_dir, "market_exposure_contracts_results"))
end

# ------------------------------------------------------------------------------
# SECTION 7: AGENT INITIALIZATION
# ------------------------------------------------------------------------------

agents = Dict()

agents[:power] = [id for id in keys(data["Power"])]
agents[:H2] = [id for id in keys(data["Hydrogen"])]
agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"])]
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ? [id for id in keys(data["Electricity_GC_Demand"])] : []

agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand])

agents[:elec_market] = []
agents[:H2_market] = []
agents[:elec_GC_market] = []
agents[:H2_GC_market] = []
agents[:EP_market] = []
# PPA market: VRES and electrolyzer only (populated by define_contract_parameters!)
agents[:ppa_market] = []
agents[:hpa_market] = []

mdict = Dict(i => Model(Gurobi.Optimizer) for i in agents[:all])

for m in values(mdict)
    set_silent(m)
end

# ------------------------------------------------------------------------------
# SECTION 8: MARKET PARAMETER DEFINITION
# ------------------------------------------------------------------------------

elec_market = Dict{String,Any}()
H2_market = Dict{String,Any}()
elec_GC_market = Dict{String,Any}()
H2_GC_market = Dict{String,Any}()
EP_market = Dict{String,Any}()
ppa_market = Dict{String,Any}()
hpa_market = Dict{String,Any}()

elec_market["nAgents"] = length(agents[:elec_market])
H2_market["nAgents"] = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])
EP_market["nAgents"] = length(agents[:offtaker])

define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts, repr_days)

# PPA/HPA markets: per-asset initial prices and rho from data.yaml PPAs/HPAs blocks
define_contract_market_parameters!(ppa_market, hpa_market, data_run, agents)

# ------------------------------------------------------------------------------
# SECTION 9: AGENT PARAMETER DEFINITION
# ------------------------------------------------------------------------------

for m in agents[:power]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Power"][m], data["ADMM"]), ts, repr_days, agents)
    define_power_parameters!(m, mdict[m], merge(run_general, data["Power"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], merge(run_general, data["Power"][m]), agents)
end

for m in agents[:H2]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m], data["ADMM"]), ts, repr_days, agents)
    define_H2_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m]), agents)
end

for m in agents[:offtaker]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m], data["ADMM"]), ts, repr_days, agents)
    define_offtaker_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m]), agents)
end

for m in agents[:elec_GC_demand]
    define_common_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m], data["ADMM"]), ts, repr_days, agents)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m]), ts, repr_days)
end

# Agents with endogenous capacity (VRES, H2 producer, GreenOfftaker) for investment consensus.
agents[:cap_agents] = [m for m in agents[:all] if haskey(mdict[m].ext[:parameters], :z_cap)]

# Set market participant counts
elec_market["nAgents"]    = length(agents[:elec_market])
H2_market["nAgents"]     = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"]  = length(agents[:H2_GC_market])
EP_market["nAgents"]     = length(agents[:offtaker])
ppa_market["nAgents"] = length(agents[:ppa_market])
hpa_market["nAgents"] = length(agents[:hpa_market])

# ------------------------------------------------------------------------------
# SECTION 10: BUILD OPTIMIZATION MODELS
# ------------------------------------------------------------------------------

# Use PPA build for power and H2 (VRES and electrolyzer get PPA variables)
for m in agents[:power]
    build_power_agent_contracts!(m, mdict[m], elec_market, elec_GC_market, ppa_market)
end

for m in agents[:H2]
    build_H2_agent_contracts!(m, mdict[m], H2_market, H2_GC_market, ppa_market)
end

for m in agents[:offtaker]
    build_offtaker_agent_contracts!(m, mdict[m], EP_market, H2_market, H2_GC_market, hpa_market)
end

for m in agents[:elec_GC_demand]
    build_elec_GC_demand_agent!(m, mdict[m], elec_GC_market)
end

# ------------------------------------------------------------------------------
# SECTION 10b: CAPACITY WARM-START FROM SP (optional, same as market_exposure)
# ------------------------------------------------------------------------------
n_cap_warmstart = 0
sp_cap_file = joinpath(home_dir, "social_planner_results", "SP_Capacities.csv")
if isfile(sp_cap_file)
    try
        sp_cap_df = CSV.read(sp_cap_file, DataFrame)
        jy_set = collect(mdict[agents[:all][1]].ext[:sets][:JY])
        for m in agents[:cap_agents]
            mod = mdict[m]
            agent_type = String(get(mod.ext[:parameters], :Type, ""))
            cap_var = nothing
            if agent_type == "VRES" && haskey(mod.ext[:variables], :cap_VRES)
                cap_var = mod.ext[:variables][:cap_VRES]
            elseif agent_type == "GreenProducer" && haskey(mod.ext[:variables], :cap_H2_y)
                cap_var = mod.ext[:variables][:cap_H2_y]
            elseif agent_type == "GreenOfftaker" && haskey(mod.ext[:variables], :cap_EP_y)
                cap_var = mod.ext[:variables][:cap_EP_y]
            end
            if cap_var !== nothing
                for jy in jy_set
                    row = sp_cap_df[(sp_cap_df.AgentID .== m) .& (sp_cap_df.jy .== jy), :]
                    if nrow(row) == 0
                        # If SP was solved for one year, reuse jy=1 capacity for all scenario years.
                        row = sp_cap_df[(sp_cap_df.AgentID .== m) .& (sp_cap_df.jy .== 1), :]
                    end
                    if nrow(row) >= 1
                        set_start_value(cap_var[jy], row.cap[1])
                    end
                end
                global n_cap_warmstart += 1
            end
        end
    catch e
        @warn "Could not load SP capacities ($sp_cap_file): $e"
    end
end

# ------------------------------------------------------------------------------
# SECTION 11: RUN ADMM (PPA VERSION)
# ------------------------------------------------------------------------------

results = Dict()
ADMM = Dict()
TO = TimerOutput()

sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
sp_primal_file = joinpath(home_dir, "social_planner_results", "SP_Primal_Quantities.csv")
# Use epsilon_contracts if set (more relaxed for higher complexity); else same epsilon as market_exposure.
admm_data = merge(run_general, data["ADMM"])
if haskey(data["ADMM"], "epsilon_contracts")
    admm_data = merge(admm_data, Dict("epsilon" => data["ADMM"]["epsilon_contracts"], "epsilon_abs" => data["ADMM"]["epsilon_contracts"]))
end
define_results_contracts!(admm_data, results, ADMM, agents, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, ppa_market, hpa_market;
    sp_prices_file=sp_prices_file, sp_primal_file=sp_primal_file, sp_cap_file=sp_cap_file, use_primal_warmstart=true)

# Single consolidated warm-start message (same as market_exposure)
ws = results["warmstart"]
parts = String[]
ws["λ"] && push!(parts, "λ from SP prices")
ws["primal"] && push!(parts, "primal quantities from SP")
n_cap_warmstart > 0 && push!(parts, "capacity seeds for $n_cap_warmstart agents")
if !isempty(parts)
    @info "ADMM warm-start: $(join(parts, ", "))"
end

ADMM_contracts!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, ppa_market, hpa_market, mdict, agents, data_run, TO)

ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ------------------------------------------------------------------------------
# SECTION 12: SAVE RESULTS (PPA VERSION)
# ------------------------------------------------------------------------------

save_results_contracts!(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, ppa_market, hpa_market, ADMM, results, agents)

YAML.write_file(joinpath(home_dir, "market_exposure_contracts_results", "TimerOutput.yaml"), TO)
