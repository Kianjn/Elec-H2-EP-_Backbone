# ==============================================================================
# MARKET EXPOSURE (CONTRACTS) SCRIPT: ADMM WITH BILATERAL VRES–ELECTROLYZER CONTRACTS
# By Kian Jafarinejad - PhD Researcher at TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
#
# PURPOSE:
#   Third entry point alongside market_exposure.jl and social_planner.jl.
#   Identical to market_exposure.jl except that VRES and the electrolyzer can
#   enter a bilateral contract:
#
#   - VRES commits contract_cap (MW) of capacity; that capacity is REMOVED
#     from the electricity market (EOM). VRES sells only g_EOM to the pool.
#   - The electrolyzer buys that capacity pay-as-produced: receives g_contract
#     at each timestep, pays λ_contract per MWh actually delivered.
#   - When VRES has no output (e.g. night for solar), g_contract = 0, so
#     nothing is delivered and nothing is paid.
#   - A contract pool clears both contract energy (3D) and contract capacity
#     (scalar) via ADMM.
#
#   social_planner.jl and market_exposure.jl are UNCHANGED. This script uses contract-specific
#   build/solve/ADMM/save modules.
#
# HOW TO RUN:
#   From the project root:  julia market_exposure_contracts.jl
#
# OUTPUT:
#   market_exposure_contracts_results/ — same major ADMM CSVs as market_exposure_results
#   plus Contracts.csv, Green_Agents_Detail.csv, and contract columns in convergence/
#   diagnostics CSVs.
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

# Contract-specific: adds in_contract_market, λ_contract, g_bar_contract, etc.
include(joinpath(home_dir, "Source", "define_contract_parameters.jl"))
include(joinpath(home_dir, "Source", "define_contract_market_parameters.jl"))

# Market definitions
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))

# Model building: use CONTRACTS versions for power and H2 (VRES/electrolyzer)
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))

# ADMM and solving: CONTRACTS versions (include contract market)
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
n_years  = haskey(gen, "nYears") ? gen["nYears"] : 1
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
# Contract market: VRES and electrolyzer only (populated by define_contract_parameters!)
agents[:contract_market] = []

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
contract_market = Dict{String,Any}()

elec_market["nAgents"] = length(agents[:elec_market])
H2_market["nAgents"] = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])
EP_market["nAgents"] = length(agents[:offtaker])

define_electricity_market_parameters!(elec_market, merge(data["General"], data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(data["General"], data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(data["General"], data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(data["General"], data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(data["General"], data["ADMM"], data["EP_market"]), ts, repr_days)

# Contract market: initial price and rho from data.yaml Contracts block
contract_data = haskey(data, "Contracts") ? merge(data["General"], data["ADMM"], data["Contracts"]) : merge(data["General"], data["ADMM"], Dict("initial_price" => 60.0, "rho_initial" => 0.5))
define_contract_market_parameters!(contract_market, contract_data)

# ------------------------------------------------------------------------------
# SECTION 9: AGENT PARAMETER DEFINITION
# ------------------------------------------------------------------------------

for m in agents[:power]
    define_common_parameters!(m, mdict[m], merge(data["General"], data["Power"][m], data["ADMM"]), ts, repr_days, agents)
    define_power_parameters!(m, mdict[m], merge(data["General"], data["Power"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], merge(data["General"], data["Power"][m]), agents)
end

for m in agents[:H2]
    define_common_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen"][m], data["ADMM"]), ts, repr_days, agents)
    define_H2_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen"][m]), ts, repr_days)
    define_contract_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen"][m]), agents)
end

for m in agents[:offtaker]
    define_common_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen_Offtaker"][m], data["ADMM"]), ts, repr_days, agents)
    define_offtaker_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen_Offtaker"][m]), ts, repr_days)
end

for m in agents[:elec_GC_demand]
    define_common_parameters!(m, mdict[m], merge(data["General"], data["Electricity_GC_Demand"][m], data["ADMM"]), ts, repr_days, agents)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(data["General"], data["Electricity_GC_Demand"][m]), ts, repr_days)
end

# Set market participant counts
elec_market["nAgents"]    = length(agents[:elec_market])
H2_market["nAgents"]     = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"]  = length(agents[:H2_GC_market])
EP_market["nAgents"]     = length(agents[:offtaker])
contract_market["nAgents"] = length(agents[:contract_market])

# ------------------------------------------------------------------------------
# SECTION 10: BUILD OPTIMIZATION MODELS
# ------------------------------------------------------------------------------

# Use CONTRACTS build for power and H2 (VRES and electrolyzer get contract variables)
for m in agents[:power]
    build_power_agent_contracts!(m, mdict[m], elec_market, elec_GC_market, contract_market)
end

for m in agents[:H2]
    build_H2_agent_contracts!(m, mdict[m], H2_market, H2_GC_market, contract_market)
end

for m in agents[:offtaker]
    build_offtaker_agent!(m, mdict[m], EP_market, H2_market, H2_GC_market)
end

for m in agents[:elec_GC_demand]
    build_elec_GC_demand_agent!(m, mdict[m], elec_GC_market)
end

# ------------------------------------------------------------------------------
# SECTION 11: RUN ADMM (CONTRACTS VERSION)
# ------------------------------------------------------------------------------

results = Dict()
ADMM = Dict()
TO = TimerOutput()

sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
define_results_contracts!(merge(data["General"], data["ADMM"]), results, ADMM, agents, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, contract_market; sp_prices_file=sp_prices_file)

ADMM_contracts!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, contract_market, mdict, agents, data, TO)

ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ------------------------------------------------------------------------------
# SECTION 12: SAVE RESULTS (CONTRACTS VERSION)
# ------------------------------------------------------------------------------

save_results_contracts!(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, contract_market, ADMM, results, agents)

YAML.write_file(joinpath(home_dir, "market_exposure_contracts_results", "TimerOutput.yaml"), TO)
