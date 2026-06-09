# ==============================================================================
# SOCIAL PLANNER SCRIPT: CENTRALIZED BENCHMARK — RUNNER ONLY
# By Kian Jafarinejad - PhD Researcher at TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
#
# PURPOSE:
#   Entry point for the social planner (centralized welfare-maximization)
#   benchmark. Loads configuration and time series, builds per-agent parameter
#   models (identical to market_exposure), constructs the single centralized
#   planner model from Source/build_* functions, solves it, and writes results
#   to the "social_planner_results" folder.
#
#   The social planner maximizes risk-adjusted social welfare:
#     max  γ · Σ_y sw_aux[y]  −  (1−γ) · CVaR_social
#   where sw_aux[y] is an epigraph proxy for aggregate social welfare
#   (including quadratic consumer utility), and CVaR_social is the
#   Conditional Value-at-Risk of the social loss across scenario years.
#   When γ=1 (risk-neutral), the CVaR term vanishes and the planner
#   reduces to standard welfare maximization — matching the ADMM
#   risk-neutral equilibrium by the first welfare theorem.
#
#   PRICE RECOVERY:
#   We solve the planner as a convex QCP and extract equilibrium prices from
#   solver duals of the market-clearing constraints. By default this script
#   uses IPOPT for the social planner because in large/scaled instances Gurobi
#   can return primal-optimal QCP points but fail to provide usable QCP duals.
#   ADMM remains on Gurobi.
#
#   All problem definition (objectives, constraints, variables) lives in Source/.
#   Changes to build_* files propagate automatically to both market_exposure
#   and social_planner — no duplication of problem logic.
#
# HOW TO RUN:
#   From the project root:  julia social_planner.jl
#
# RESULTS:
#   Written to "social_planner_results/":
#     - Market_Prices.csv                 — Equilibrium prices from dual variables of balance
#                                           constraints (electricity and hydrogen).
#     - Agent_Summary.csv                 — Per-agent total quantity and welfare contribution.
#     - Capacity_Investments_Planner.csv  — Per-agent total capacity installed during the run.
#
# FLOW:
#   1. Environment and packages
#   2. Load Source/*.jl (parameter definitions, model builders, planner builder,
#      results saver)
#   3. Load Data/data.yaml and Input (timeseries, representative days)
#   4. Create results folder
#   5. Initialize agents dict and parameter-container JuMP models (mdict)
#   6. Define market parameter dicts (initial prices, rho, EP demand profile)
#   7. Define agent parameters (common + type-specific) via define_*_parameters!
#   8. Build centralized planner model via build_social_planner!
#   9. Solve the planner (convex QCP — epigraph formulation); check optimality
#  10. Extract duals directly from QCP and verify availability
#  11. Save results (prices + agent summary) via save_social_planner_results!
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

using Pkg
# Activate the project environment in the same directory as this script so that
# Project.toml / Manifest.toml dictate package versions.
Pkg.activate(@__DIR__)

# ------------------------------------------------------------------------------
# SECTION 2: PACKAGE LOADING
# ------------------------------------------------------------------------------

using JuMP
# JuMP: algebraic modeling for optimization; we use it to build the centralized
# planner model (variables, constraints, objective) and to query duals/values.

using Gurobi
# Gurobi: still used by ADMM and available as optional SP solver.

using Ipopt
# Ipopt: default solver for social_planner.jl to obtain reliable QCP duals
# on this model family when Gurobi QCP dual recovery is numerically fragile.

using DataFrames
# Tabular data; used when reading CSVs (timeseries, representative days) and
# when writing result DataFrames to CSV.

using CSV
# Read/write CSV files (inputs and output CSVs in social_planner_results).

using Statistics
# mean, etc., used in the summary print of equilibrium prices.

using Printf
# @sprintf / @printf in save_social_planner_results! and print_run_summary.jl

using YAML
# Parse data.yaml: General, ADMM, market blocks, and agent blocks (Power,
# Hydrogen, Hydrogen_Offtaker, Electricity_GC_Demand, EP_Demand).

# MathOptInterface: imported to access termination status constants (e.g.
# MOI.OPTIMAL). Needed for the post-solve check that verifies the solver
# found an optimal solution; without it we cannot compare against MOI.OPTIMAL.
import MathOptInterface as MOI

# Lazy Gurobi environment. Created only if SP solver is set to Gurobi.
const GUROBI_ENV_REF = Ref{Union{Nothing, Gurobi.Env}}(nothing)
function get_gurobi_env()
    if GUROBI_ENV_REF[] === nothing
        GUROBI_ENV_REF[] = Gurobi.Env()
    end
    return GUROBI_ENV_REF[]::Gurobi.Env
end

# ------------------------------------------------------------------------------
# SECTION 3: DIRECTORY SETUP
# ------------------------------------------------------------------------------

# Root directory of the project; all paths (Data/, Input/, Source/, results) are
# built from this.
const home_dir = @__DIR__

# ------------------------------------------------------------------------------
# SECTION 4: FUNCTION LOADING (SOURCE FILES)
# ------------------------------------------------------------------------------

# Parameter definition: attach to each agent model sets, weights, ADMM arrays,
# and market participation flags; fill agent-specific parameters and timeseries.
include(joinpath(home_dir, "Source", "define_common_parameters.jl"))
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_demand_parameters.jl"))

# Market definitions: initial prices, rho, and for EP_market the fixed demand
# profile D_EP. These are shared with market_exposure; the social planner uses
# EP_market["D_EP"] in its market-clearing constraint.
include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))

# Model building: the add_*_to_planner! functions inside each build_* file add
# agent-specific variables, constraints, and welfare expressions to the
# centralized planner model. build_social_planner! orchestrates all of them.
include(joinpath(home_dir, "Source", "build_power_agent.jl"))
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))
include(joinpath(home_dir, "Source", "build_EP_demand_agent.jl"))

# Social planner orchestrator: calls add_*_to_planner! for each agent, adds
# market-clearing balance constraints, and sets Max(total welfare) objective.
include(joinpath(home_dir, "Source", "build_social_planner.jl"))

# Shared objective computation (used by save_social_planner_results! and save_results).
include(joinpath(home_dir, "Source", "compute_agent_objective.jl"))

# Result writer: extracts dual prices and agent quantities/welfare from the
# solved planner model and writes Market_Prices.csv + Agent_Summary.csv.
include(joinpath(home_dir, "Source", "save_social_planner_results.jl"))

# ------------------------------------------------------------------------------
# SECTION 5: DATA LOADING
# ------------------------------------------------------------------------------

# Load the single configuration file: General (nTimesteps, nReprDays, nYears,
# base_year), ADMM (max_iter, epsilon, rho_initial — used here only because
# define_common_parameters! expects ADMM keys for placeholder arrays), per-market
# blocks (initial_price, rho_initial; EP_market also Demand_Column, Total_Demand),
# and per-agent blocks under Power, Hydrogen, Hydrogen_Offtaker,
# Electricity_GC_Demand, EP_Demand.
data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))

# Time series: keyed by calendar year (e.g. 2021). Each value is a DataFrame
# with columns such as SOLAR, LOAD_E, LOAD_H, LOAD_EP (normalized 0–1 profiles).
# Named ts_dict here (vs. ts in market_exposure.jl) to distinguish the social-
# planner script's local scope, but it holds identical data.
ts_dict = Dict()

# Ordering matrix: loaded for completeness — it is used by the upstream
# representative-day selection algorithm (outside this script) but is NOT
# directly used in the optimization itself.
order_matrix = Dict()

# Representative days: keyed by year. Each value is a DataFrame with columns
# periods (day index 1–365), weights (frequency), selected_periods.
repr_days = Dict()

# Determine modeled years for the social planner.
# We intentionally align SP and ADMM scenario horizons by default so benchmark
# comparisons are apples-to-apples in multi-scenario studies. SP uses:
#   ADMM.nScenarioYears (if provided), else General.nYears.
# For example:
#   base_year = 2021, nYears = 1  -> {1 => 2021}
#   base_year = 2021, nYears = 5  -> {1 => 2021, 2 => 2022, ..., 5 => 2025}
# years Dict: maps scenario index (1, 2, ...) to calendar year (2021, 2022, ...).
# WHY: timeseries and repr_days are keyed by calendar year, while the model
# uses integer scenario indices (JY). This mapping bridges the two.
run_general = merge(data["General"])
run_general["nYears"] = get(data["ADMM"], "nScenarioYears", get(data["General"], "nYears", 1))
gen = run_general
base_year = haskey(gen, "base_year") ? gen["base_year"] : 2021
n_years = haskey(gen, "nYears") ? gen["nYears"] : 1
years = Dict(i => base_year + (i - 1) for i in 1:n_years)

# Full-year hourly time series and representative days for each modeled year.
# Input files are expected to follow the pattern:
#   Input/timeseries_<year>.csv
#   Input/output_<year>/ordering_variable.csv
#   Input/output_<year>/decision_variables_short.csv
for y in values(years)
    ts_dict[y] = CSV.read(joinpath(home_dir, "Input", "timeseries_$(y).csv"), DataFrame)
    order_matrix[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "ordering_variable.csv"), delim=",", DataFrame)
    repr_days[y] = CSV.read(joinpath(home_dir, "Input", "output_$(y)", "decision_variables_short.csv"), delim=",", DataFrame)
end

# ------------------------------------------------------------------------------
# SECTION 6: RESULTS FOLDER
# ------------------------------------------------------------------------------

results_folder = joinpath(home_dir, "social_planner_results")
if !isdir(results_folder)
    mkdir(results_folder)
end

# ------------------------------------------------------------------------------
# SECTION 7: AGENT INITIALIZATION
# ------------------------------------------------------------------------------

agents = Dict{Symbol, Any}()

# List of agent IDs that belong to the power sector (VRES, conventional, consumer).
agents[:power] = [id for id in keys(data["Power"])]

# List of hydrogen-sector agent IDs (e.g. electrolyzer).
agents[:H2] = [id for id in keys(data["Hydrogen"])]

# List of offtaker agent IDs (green, grey, importer).
agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"])]

# Electricity GC demand agents; empty if the block is missing in data.yaml.
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ? [id for id in keys(data["Electricity_GC_Demand"])] : String[]

# EP demand agents; empty if the block is missing in data.yaml (currently
# EP demand is inelastic, defined via EP_market["D_EP"]).
agents[:EP_demand] = haskey(data, "EP_Demand") ? [id for id in keys(data["EP_Demand"])] : String[]

# Union of all agents: used to create parameter-container models and to iterate
# when calling define_*_parameters!.
agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand], agents[:EP_demand])

# These lists are filled by define_common_parameters! when each agent's type is
# known; they indicate which agents participate in which market (used by
# build_social_planner! for market-clearing constraint construction).
agents[:elec_market] = []
agents[:H2_market] = []
agents[:elec_GC_market] = []
agents[:H2_GC_market] = []
agents[:EP_market] = []

# ------------------------------------------------------------------------------
# SECTION 8: MARKET PARAMETER DEFINITION
# ------------------------------------------------------------------------------

elec_market = Dict{String, Any}()
H2_market = Dict{String, Any}()
elec_GC_market = Dict{String, Any}()
H2_GC_market = Dict{String, Any}()
EP_market = Dict{String, Any}()

# Fill market dicts with initial_price, rho_initial, and for EP_market also
# Demand_Column, Total_Demand, and the 3D demand array D_EP. The social planner
# uses EP_market["D_EP"] in the end-product balance constraint; other fields
# (initial_price, rho_initial) are populated for interface consistency with
# define_*_parameters! but are not used by the planner optimization itself.
define_electricity_market_parameters!(elec_market, merge(run_general, data["ADMM"], data["elec_market"]), ts_dict, repr_days)
define_H2_market_parameters!(H2_market, merge(run_general, data["ADMM"], data["H2_market"]), ts_dict, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(run_general, data["ADMM"], data["elec_GC_market"]), ts_dict, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(run_general, data["ADMM"], data["H2_GC_market"]), ts_dict, repr_days)
define_EP_market_parameters!(EP_market, merge(run_general, data["ADMM"], data["EP_market"]), ts_dict, repr_days)

# ------------------------------------------------------------------------------
# SECTION 9: AGENT PARAMETER DEFINITION
# ------------------------------------------------------------------------------

# mdict: creates one empty JuMP Model per agent to serve as a parameter
# container (ext[:parameters], ext[:sets], ext[:timeseries]). These models are
# NOT used for optimization — the centralized planner model is built separately
# by build_social_planner!. We reuse the same define_*_parameters! functions as
# market_exposure, which expect a JuMP Model with ext storage.
mdict = Dict{String, JuMP.Model}()
for id in agents[:all]
    mdict[id] = Model()
end

for m in agents[:power]
    # Common: sets (JY, JD, JH), weights W, P, γ, β, market flags, ADMM arrays.
    define_common_parameters!(m, mdict[m], merge(run_general, data["Power"][m], data["ADMM"]), ts_dict, repr_days, agents)
    # Power-specific: capacity, profile column, costs, or consumer utility/load.
    define_power_parameters!(m, mdict[m], merge(run_general, data["Power"][m]), ts_dict, repr_days)
end

for m in agents[:H2]
    # Common + H2-specific: electrolyzer capacity, H2 output capacity,
    # specific consumption, operational cost, efficiency η.
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m], data["ADMM"]), ts_dict, repr_days, agents)
    define_H2_parameters!(m, mdict[m], merge(run_general, data["Hydrogen"][m]), ts_dict, repr_days)
end

for m in agents[:offtaker]
    # Common + offtaker-specific: type (Green/Grey/Importer), capacities,
    # alpha, processing cost, marginal cost, gamma_GC, gamma_NH3, import cost.
    define_common_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m], data["ADMM"]), ts_dict, repr_days, agents)
    define_offtaker_parameters!(m, mdict[m], merge(run_general, data["Hydrogen_Offtaker"][m]), ts_dict, repr_days)
end

for m in agents[:elec_GC_demand]
    # Common + GC demand-specific: peak load, load column, A_GC, B_GC
    # (quadratic utility for GC demand).
    define_common_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m], data["ADMM"]), ts_dict, repr_days, agents)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(run_general, data["Electricity_GC_Demand"][m]), ts_dict, repr_days)
end

for m in agents[:EP_demand]
    # Common + EP demand-specific: placeholder for future elastic EP demand.
    define_common_parameters!(m, mdict[m], merge(run_general, data["EP_Demand"][m], data["ADMM"]), ts_dict, repr_days, agents)
    define_EP_demand_parameters!(m, mdict[m], merge(run_general, data["EP_Demand"][m]), ts_dict, repr_days)
end

# ------------------------------------------------------------------------------
# SECTION 10: BUILD CENTRALIZED PLANNER MODEL
# ------------------------------------------------------------------------------

# build_social_planner! orchestrates the construction of the single centralized
# convex QCP model (epigraph formulation for full-welfare CVaR):
#   1. For each agent, calls the corresponding add_*_to_planner! function from
#      the build_* files. Each function adds the agent's decision variables,
#      physical constraints, and per-year welfare expression (utility or
#      negative cost) to the shared planner model — with NO ADMM penalty terms
#      and NO per-agent CVaR (CVaR is applied once to aggregate social welfare).
#   2. Adds market-clearing balance constraints (electricity, elec-GC, H₂,
#      H₂-GC, end-product) that enforce supply = demand in every market.
#   3. Aggregates per-year social welfare and adds epigraph variables (sw_aux),
#      quadratic epigraph constraints, linear CVaR constraints, and a linear
#      risk-adjusted objective: max γ·Σ sw_aux − (1−γ)·CVaR_social.
#
# Returns:
#   planner       — JuMP model (convex QCP) ready to optimize.
#   planner_state — Dict collecting variable dicts, welfare expressions,
#                   balance constraints, agent classification lists, index
#                   sets, risk parameters, demand_var_keys, and sw_aux
#                   needed by direct QCP solve and save_social_planner_results!.
# Social planner solver selection (SP only).
sp_cfg = get(data, "SocialPlanner", Dict{String, Any}())
sp_solver = lowercase(String(get(sp_cfg, "solver", "ipopt")))
optimizer_factory = sp_solver == "gurobi" ? Gurobi.Optimizer : Ipopt.Optimizer
sp_env = sp_solver == "gurobi" ? get_gurobi_env() : nothing
planner, planner_state = build_social_planner!(mdict, agents, elec_market, H2_market,
                                              elec_GC_market, H2_GC_market, EP_market,
                                              data; env = sp_env, optimizer_factory = optimizer_factory)

# ------------------------------------------------------------------------------
# SECTION 11: DIRECT QCP SOLVE + DIRECT DUAL EXTRACTION
# ------------------------------------------------------------------------------
#
# The planner is solved directly as a convex QCP and market prices are read from
# duals of the market-clearing constraints.
# ------------------------------------------------------------------------------
qcp_status = MOI.OTHER_ERROR
duals_ok = false
if sp_solver == "gurobi"
    # Gurobi path: retry with tighter barrier settings if QCP duals are missing.
    admm_cfg = get(data, "ADMM", Dict{String, Any}())
    base_tol = Float64(get(admm_cfg, "BarQCPConvTol", 1e-8))
    tol_candidates = unique([base_tol, min(base_tol, 1e-9), 1e-10])
    status_hist = String[]
    let _status = MOI.OTHER_ERROR, _duals = false
        for (attempt, tol) in enumerate(tol_candidates)
            set_optimizer_attribute(planner, "QCPDual", 1)
            set_optimizer_attribute(planner, "Method", 2)      # barrier
            set_optimizer_attribute(planner, "Crossover", 0)   # keep barrier point
            set_optimizer_attribute(planner, "NumericFocus", min(3, attempt))
            set_optimizer_attribute(planner, "BarQCPConvTol", tol)

            optimize!(planner)
            _status = termination_status(planner)
            _duals = has_duals(planner)
            push!(status_hist, "attempt=$(attempt), BarQCPConvTol=$(tol), status=$(_status), has_duals=$(_duals)")
            if (_status == MOI.OPTIMAL || _status == MOI.LOCALLY_SOLVED) && _duals
                @info "Gurobi QCP duals available (attempt=$(attempt), BarQCPConvTol=$(tol))."
                break
            end
        end
        qcp_status = _status
        duals_ok = _duals
    end

    if qcp_status != MOI.OPTIMAL && qcp_status != MOI.LOCALLY_SOLVED
        @error("Social planner QCP solve failed. Attempts: " * join(status_hist, " | "))
        error("Social planner QCP solve failed (see attempt log in output).")
    end
    if !duals_ok
        error("Gurobi solved SP QCP but did not return duals after retries. " *
              "Attempts: " * join(status_hist, " | ") * ". " *
              "Set SocialPlanner.solver=ipopt to obtain SP prices from QCP duals.")
    end
else
    # IPOPT path (default): solve QCP directly and use returned multipliers.
    set_optimizer_attribute(planner, "tol", Float64(get(sp_cfg, "ipopt_tol", 1e-8)))
    set_optimizer_attribute(planner, "max_iter", Int(get(sp_cfg, "ipopt_max_iter", 5000)))
    set_optimizer_attribute(planner, "print_level", 0)
    optimize!(planner)
    qcp_status = termination_status(planner)
    duals_ok = has_duals(planner)
    if qcp_status != MOI.OPTIMAL && qcp_status != MOI.LOCALLY_SOLVED
        error("IPOPT social planner solve failed with status $qcp_status")
    end
    if !duals_ok
        error("IPOPT solved SP QCP but duals are unavailable.")
    end
    @info "IPOPT QCP solve complete with duals available."
end

# ------------------------------------------------------------------------------
# SECTION 12: SAVE RESULTS
# ------------------------------------------------------------------------------

# Write Market_Prices.csv (equilibrium prices from QCP dual variables of balance
# constraints) and Agent_Summary.csv (per-agent quantities and objective values)
# to the social_planner_results folder.
save_social_planner_results!(planner, planner_state, agents, mdict, results_folder)
