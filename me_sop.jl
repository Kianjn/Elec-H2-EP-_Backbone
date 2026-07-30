# ==============================================================================
# ME SoP — Market exposure with send-or-pay HPAs
# ==============================================================================
# HOW TO RUN:  julia me_sop.jl
# OUTPUT:      me_sop_results/
# PPAs: always pay-as-produced. Settlement strike K is a scalar €/MWh (uniform
# over hours) derived from bilateral λ_ppa; updated each ADMM iteration until
# convergence, then snapshotted for reporting.
# HPAs: seller penalised K × shortfall when under-delivering vs min(cap, production).
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)

using JuMP
using Gurobi
using DataFrames
using CSV
using YAML
using DataStructures
using ProgressBars
using Printf
using TimerOutputs
using Statistics

const GUROBI_ENV = Gurobi.Env()
const home_dir = @__DIR__
const HPA_VOLUME_MODE = "sop"
const CASE_LABEL = "me_sop"
const RESULTS_SUBDIR = "me_sop_results"
const CASE_RESULTS_DIR = joinpath(home_dir, RESULTS_SUBDIR)

# ------------------------------------------------------------------------------
# SOURCE FILES
# ------------------------------------------------------------------------------

include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_contract_parameters.jl"))
include(joinpath(home_dir, "Source", "define_contract_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))
include(joinpath(home_dir, "Source", "contract_strike.jl"))
include(joinpath(home_dir, "Source", "contract_capacity.jl"))
include(joinpath(home_dir, "Source", "contract_settlement.jl"))
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_power_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent_contracts.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))
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
# DATA LOADING
# ------------------------------------------------------------------------------

data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))
ts = Dict()
order_matrix = Dict()
repr_days = Dict()

gen = data["General"]
n_years = haskey(data["ADMM"], "nScenarioYears") ? data["ADMM"]["nScenarioYears"] :
          (haskey(gen, "nYears") ? gen["nYears"] : 1)
run_general = merge(gen, Dict("nYears" => n_years))
data_run = copy(data)
data_run["General"] = run_general
years = Dict(i => i for i in 1:n_years)

for y in values(years)
    ts[y] = CSV.read(joinpath(home_dir, "Input", "timeseries_$(y).csv"), DataFrame)
    order_matrix[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "ordering_variable.csv"), delim=",", DataFrame)
    repr_days[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "decision_variables_short.csv"), delim=",", DataFrame)
end

isdir(CASE_RESULTS_DIR) || mkdir(CASE_RESULTS_DIR)

# ------------------------------------------------------------------------------
# AGENTS
# ------------------------------------------------------------------------------

agents = Dict()
agents[:power] = [id for id in keys(data["Power"])]
agents[:H2] = [id for id in keys(data["Hydrogen"])]
agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"])]
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ?
    [id for id in keys(data["Electricity_GC_Demand"])] : String[]
agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand])
agents[:elec_market] = String[]
agents[:H2_market] = String[]
agents[:elec_GC_market] = String[]
agents[:H2_GC_market] = String[]
agents[:EP_market] = String[]
agents[:ppa_market] = String[]
agents[:hpa_market] = String[]
agents[:hpa_volume_mode] = HPA_VOLUME_MODE

mdict = Dict(i => Model(Gurobi.Optimizer) for i in agents[:all])
for m in values(mdict)
    set_silent(m)
end

# ------------------------------------------------------------------------------
# MARKETS AND PARAMETERS
# ------------------------------------------------------------------------------

elec_market = Dict{String, Any}()
H2_market = Dict{String, Any}()
elec_GC_market = Dict{String, Any}()
H2_GC_market = Dict{String, Any}()
EP_market = Dict{String, Any}()
ppa_market = Dict{String, Any}()
hpa_market = Dict{String, Any}()

define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts, repr_days)
define_contract_market_parameters!(ppa_market, hpa_market, data_run, agents)

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

agents[:cap_agents] = [m for m in agents[:all] if haskey(mdict[m].ext[:parameters], :z_cap)]
elec_market["nAgents"] = length(agents[:elec_market])
H2_market["nAgents"] = length(agents[:H2_market])
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])
EP_market["nAgents"] = length(agents[:offtaker])
ppa_market["nAgents"] = length(agents[:ppa_market])
hpa_market["nAgents"] = length(agents[:hpa_market])

# ------------------------------------------------------------------------------
# BUILD MODELS
# ------------------------------------------------------------------------------

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
# CAPACITY WARM-START FROM SP (optional)
# ------------------------------------------------------------------------------

n_cap_warmstart = 0
sp_cap_file = joinpath(home_dir, "social_planner_results", "SP_Capacities.csv")
if isfile(sp_cap_file)
    try
        sp_cap_df = CSV.read(sp_cap_file, DataFrame)
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
                row = sp_cap_df[sp_cap_df.AgentID .== m, :]
                cap_val = _sp_cap_scalar(row)
                cap_val !== nothing && set_start_value(cap_var, cap_val)
                global n_cap_warmstart += 1
            end
        end
    catch e
        @warn "Could not load SP capacities ($sp_cap_file): $e"
    end
end

# ------------------------------------------------------------------------------
# RUN ADMM
# ------------------------------------------------------------------------------

results = Dict()
ADMM = Dict()
TO = TimerOutput()

sp_prices_file = joinpath(home_dir, "social_planner_results", "Market_Prices.csv")
sp_primal_file = joinpath(home_dir, "social_planner_results", "SP_Primal_Quantities.csv")
admm_data = merge(run_general, data["ADMM"])
if haskey(data["ADMM"], "epsilon_contracts")
    admm_data = merge(admm_data, Dict("epsilon" => data["ADMM"]["epsilon_contracts"],
        "epsilon_abs" => data["ADMM"]["epsilon_contracts"]))
end

define_results_contracts!(admm_data, results, ADMM, agents, elec_market, H2_market,
    elec_GC_market, H2_GC_market, EP_market, ppa_market, hpa_market;
    sp_prices_file=sp_prices_file, sp_primal_file=sp_primal_file,
    sp_cap_file=sp_cap_file, use_primal_warmstart=true)

ws = results["warmstart"]
parts = String[]
ws["λ"] && push!(parts, "λ from SP prices")
ws["primal"] && push!(parts, "primal quantities from SP")
n_cap_warmstart > 0 && push!(parts, "capacity seeds for $n_cap_warmstart agents")
@info CASE_LABEL hpa_volume=HPA_VOLUME_MODE
!isempty(parts) && @info "ADMM warm-start: $(join(parts, ", "))"

ADMM_contracts!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market,
    EP_market, ppa_market, hpa_market, mdict, agents, data_run, TO)
ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ------------------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------------------

save_results_contracts!(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market,
    ppa_market, hpa_market, ADMM, results, agents;
    results_dir=CASE_RESULTS_DIR, case_label=CASE_LABEL)

YAML.write_file(joinpath(CASE_RESULTS_DIR, "TimerOutput.yaml"), TO)
