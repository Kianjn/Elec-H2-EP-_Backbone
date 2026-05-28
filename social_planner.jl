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
#   DUAL RECOVERY: The epigraph formulation creates a convex QCP
#   (quadratically constrained program). Gurobi solves convex QCPs to
#   global optimality but does not provide dual variables (prices).
#   A two-step solve is used: (1) solve QCP for optimal quantities,
#   (2) fix all variables that appear quadratically in social welfare
#   (elastic-demand vars + conventional stage-dispatch vars), replace
#   QC constraints with linear equivalents, then re-solve the resulting
#   LP to obtain duals.
#   The duals of market-clearing constraints are equilibrium prices.
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
#  10. Dual recovery: fix all quadratic-welfare variables at optimal values → LP re-solve
#  11. Save results (prices + agent summary) via save_social_planner_results!
#  12. Unfix LP-fixed variables (cleanup)
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
# Gurobi: QP solver for the social planner problem. Must be installed and
# licensed. A single shared Env is created below to avoid multiple license
# tokens.

using DataFrames
# Tabular data; used when reading CSVs (timeseries, representative days) and
# when writing result DataFrames to CSV.

using CSV
# Read/write CSV files (inputs and output CSVs in social_planner_results).

using Statistics
# mean, etc., used in the summary print of equilibrium prices.

using YAML
# Parse data.yaml: General, ADMM, market blocks, and agent blocks (Power,
# Hydrogen, Hydrogen_Offtaker, Electricity_GC_Demand, EP_Demand).

# MathOptInterface: imported to access termination status constants (e.g.
# MOI.OPTIMAL). Needed for the post-solve check that verifies the solver
# found an optimal solution; without it we cannot compare against MOI.OPTIMAL.
import MathOptInterface as MOI

# Single shared Gurobi environment for the entire process. WHY: each Gurobi
# Env consumes a license token; sharing one Env across the planner model (and
# any parameter-container models) avoids acquiring multiple licenses and
# significantly reduces solver-startup overhead.
const GUROBI_ENV = Gurobi.Env()

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
#                   needed by the two-step solve and save_social_planner_results!.
planner, planner_state = build_social_planner!(mdict, agents, elec_market, H2_market,
                                              elec_GC_market, H2_GC_market, EP_market,
                                              data; env = GUROBI_ENV)

# ------------------------------------------------------------------------------
# SECTION 11: TWO-STEP SOLVE WITH DUAL RECOVERY
# ------------------------------------------------------------------------------
#
# The epigraph formulation (sw_aux[jy] ≤ social_welfare[jy]) makes the model
# a convex QCP: the epigraph constraints are quadratic (consumer utility terms
# contain −B/2·d²). Gurobi solves convex QCPs to global optimality but does
# NOT provide dual variables (Pi attribute) for QCP models.
#
# DUAL RECOVERY STRATEGY:
#   Step 1 — Solve the QCP to obtain optimal primal values (quantities).
#            Accept LOCALLY_SOLVED because for a convex QCP, local = global.
#   Step 2 — Fix all variables that appear quadratically in social welfare
#            (elastic-demand vars and conventional stage-dispatch vars) at
#            their optimal values. With squared terms replaced by constants,
#            the epigraph QC constraints become linear → the entire model
#            reduces to an LP.
#   Step 3 — Re-solve the LP. Gurobi provides full dual variables for LPs.
#            The duals of the market-clearing constraints are the equilibrium
#            prices at the risk-averse optimal allocation.
#   Step 4 — Save results (primals from step 1, duals from step 3).
#   Step 5 — Unfix LP-fixed quadratic-term variables (cleanup / restore original model).
#
# This approach is exact: the LP has the same optimal allocation as the QCP
# (demand variables are fixed at QCP-optimal values), so the duals are the
# correct shadow prices at that allocation.
# ------------------------------------------------------------------------------

# ── Step 1: Solve QCP for optimal primal values ─────────────────────────
set_optimizer_attribute(planner, "NumericFocus", 1)
optimize!(planner)

qcp_status = termination_status(planner)
if qcp_status != MOI.OPTIMAL && qcp_status != MOI.LOCALLY_SOLVED
    @error("QCP solve failed with status $qcp_status. Cannot proceed with dual recovery.")
    error("Social planner QCP solve failed (status: $qcp_status)")
end
@info "Step 1 complete: QCP solved with status $qcp_status — primal values available."

# ── Step 2: Convert QCP → LP (fix demand vars + replace QC constraints) ──
# demand_var_keys lists the var_dict keys for elastic demand agents whose
# utility functions contain quadratic terms (A·d − B/2·d²). Fixing these
# variables turns the quadratic terms into constants. However, Gurobi still
# classifies the epigraph constraints as QC based on their structural form,
# so the model remains a QCP in Gurobi's view and Pi is still unavailable.
#
# Full conversion to LP requires TWO modifications:
#   (a) Fix demand variables at QCP-optimal values.
#   (b) Delete the quadratic epigraph constraints and re-add them as LINEAR
#       constraints. The supply-side welfare terms are already linear; we
#       replace the demand-side quadratic welfare with its evaluated constant.
# After both modifications the model is a true LP — Gurobi provides duals.
var_dict        = planner_state[:var_dict]
demand_var_keys = planner_state[:demand_var_keys]
JY              = planner_state[:JY]
sw_aux          = planner_state[:sw_aux]
agent_welfare_per_year = planner_state[:agent_welfare_per_year]

# Identify agents with quadratic welfare terms that must be fixed for LP dual recovery:
# - demand agents (quadratic utility),
# - conventional generator (stagewise convex variable cost).
demand_agent_ids = Set{String}()
union!(demand_agent_ids, planner_state[:power_consumers])
union!(demand_agent_ids, agents[:elec_GC_demand])
union!(demand_agent_ids, agents[:EP_demand])
quadratic_welfare_agent_ids = Set{String}(demand_agent_ids)
union!(quadratic_welfare_agent_ids, planner_state[:power_conv])
all_welfare_ids   = collect(keys(agent_welfare_per_year))
linear_welfare_agent_ids  = filter(id -> !(id in quadratic_welfare_agent_ids), all_welfare_ids)

# ── Phase A: Query ALL optimal values BEFORE any model modification ──────
# The first fix() call invalidates JuMP's solution cache, so every value()
# query must happen here.

# (a) Variable optimal values to be fixed before LP re-solve.
#     Start with elastic-demand variables.
demand_vars_and_vals = Tuple{JuMP.VariableRef, Float64}[]
for key in demand_var_keys
    if haskey(var_dict, key)
        vars = var_dict[key]
        for v in values(vars)
            if v isa JuMP.VariableRef
                push!(demand_vars_and_vals, (v, value(v)))
            elseif v isa AbstractArray
                for vi in v
                    push!(demand_vars_and_vals, (vi, value(vi)))
                end
            end
        end
    end
end

# Also fix conventional stage variables because they appear quadratically
# in the social welfare epigraph after introducing staged convex cost.
if haskey(var_dict, :power_q_E_stage)
    for id in planner_state[:power_conv]
        if haskey(var_dict[:power_q_E_stage], id)
            qstage = var_dict[:power_q_E_stage][id]
            if qstage isa AbstractArray
                for vi in qstage
                    push!(demand_vars_and_vals, (vi, value(vi)))
                end
            elseif qstage isa JuMP.VariableRef
                push!(demand_vars_and_vals, (qstage, value(qstage)))
            end
        end
    end
end

# Deduplicate fix targets (a variable can appear through multiple containers).
fix_val_by_var = Dict{JuMP.VariableRef, Float64}()
for (v, val) in demand_vars_and_vals
    if !haskey(fix_val_by_var, v)
        fix_val_by_var[v] = val
    end
end

# ── Phase B: Modify the model ────────────────────────────────────────────

# (b1) Fix quadratic-term variables at QCP-optimal values.
# If tiny numerical noise puts a value slightly outside variable bounds,
# temporarily relax that bound to preserve feasibility of the fixed-point slice.
fixed_vars = JuMP.VariableRef[]
# Save original bounds for cleanup: (var, :lb/:ub, old_bound_value)
bound_patches = Tuple{JuMP.VariableRef, Symbol, Float64}[]
bound_relax_count = Ref(0)
max_bound_relax = Ref(0.0)
for (v, val_raw) in fix_val_by_var
    if !isfinite(val_raw)
        error("Non-finite QCP value encountered while fixing $(name(v)); cannot build LP dual-recovery model.")
    end
    val = val_raw
    if JuMP.has_lower_bound(v)
        lb = JuMP.lower_bound(v)
        if val < lb
            push!(bound_patches, (v, :lb, lb))
            JuMP.set_lower_bound(v, val)
            bound_relax_count[] += 1
            max_bound_relax[] = max(max_bound_relax[], lb - val)
        end
    end
    if JuMP.has_upper_bound(v)
        ub = JuMP.upper_bound(v)
        if val > ub
            push!(bound_patches, (v, :ub, ub))
            JuMP.set_upper_bound(v, val)
            bound_relax_count[] += 1
            max_bound_relax[] = max(max_bound_relax[], val - ub)
        end
    end
    fix_val_by_var[v] = val
    fix(v, val; force = true)
    push!(fixed_vars, v)
end
if bound_relax_count[] > 0
    @info "Step 2 note: relaxed $(bound_relax_count[]) variable bounds to match QCP fixed values (max relax = $(max_bound_relax[]))."
end

# (b2) Per-year quadratic welfare terms evaluated at the ACTUAL fixed values.
# This avoids inconsistency between fixed-point values and epigraph constants.
quadratic_welfare_const = Dict{Int, Float64}()
for jy in JY
    quadratic_welfare_const[jy] = sum(
        JuMP.value(v -> fix_val_by_var[v], agent_welfare_per_year[id][jy]) for id in quadratic_welfare_agent_ids;
        init = 0.0
    )
end

# (b3) Delete quadratic epigraph constraints (the ONLY QC in the model).
epigraph_refs = planner[:social_welfare_epigraph]
for jy in JY
    delete(planner, epigraph_refs[jy])
end
unregister(planner, :social_welfare_epigraph)

# (b4) Re-add epigraph as purely LINEAR constraints:
#   sw_aux[jy] ≤ Σ(linear welfare)[jy] + quadratic_welfare_const[jy]
# where quadratic welfare (demand utility + conventional staged cost) has been
# evaluated at the QCP optimum and absorbed into constants.
# Gurobi now classifies the model as a pure LP.
@constraint(planner, social_welfare_epigraph_lp[jy in JY],
    sw_aux[jy] <= sum(agent_welfare_per_year[id][jy] for id in linear_welfare_agent_ids)
                + quadratic_welfare_const[jy]
)

@info "Step 2 complete: Fixed $(length(fixed_vars)) quadratic-term variables, " *
      "replaced QC epigraph with linear constraints — model is now LP."

# ── Step 3: Re-solve as LP for dual variables ────────────────────────────
set_optimizer_attribute(planner, "FeasibilityTol", 1e-5)
optimize!(planner)

lp_status = termination_status(planner)
if lp_status == MOI.INFEASIBLE_OR_UNBOUNDED
    @warn("LP re-solve returned INFEASIBLE_OR_UNBOUNDED; retrying with DualReductions=0 to disambiguate.")
    set_optimizer_attribute(planner, "DualReductions", 0)
    optimize!(planner)
    lp_status = termination_status(planner)
end
if lp_status != MOI.OPTIMAL
    @warn("LP re-solve returned status $lp_status (expected OPTIMAL). Duals may be unavailable.")
end
@info "Step 3 complete: LP solved with status $lp_status — dual variables available."

# ------------------------------------------------------------------------------
# SECTION 12: SAVE RESULTS
# ------------------------------------------------------------------------------

# Write Market_Prices.csv (equilibrium prices from dual variables of balance
# constraints) and Agent_Summary.csv (per-agent total quantity and welfare
# contribution) to the social_planner_results folder.
save_social_planner_results!(planner, planner_state, agents, mdict, results_folder)

# ── Step 5: Restore original QCP model (cleanup) ─────────────────────────
# Restore the model to its original QCP form so it can be re-used (e.g.
# with different gamma values) without rebuilding from scratch.

# (a) Unfix LP-fixed variables.
for v in fixed_vars
    unfix(v)
end

# Restore temporary bound relaxations (if any).
for (v, btype, oldb) in bound_patches
    if btype == :lb
        JuMP.set_lower_bound(v, oldb)
    else
        JuMP.set_upper_bound(v, oldb)
    end
end

# (b) Delete the linear epigraph constraints and restore the original
#     quadratic epigraph constraints (sw_aux[jy] ≤ social_welfare[jy]).
lp_epigraph = planner[:social_welfare_epigraph_lp]
for jy in JY
    delete(planner, lp_epigraph[jy])
end
unregister(planner, :social_welfare_epigraph_lp)

social_welfare = planner_state[:social_welfare]
@constraint(planner, social_welfare_epigraph[jy in JY],
    sw_aux[jy] <= social_welfare[jy]
)

set_optimizer_attribute(planner, "DualReductions", 1)
set_optimizer_attribute(planner, "FeasibilityTol", 1e-6)

@info "Step 5 complete: Demand variables unfixed, QC epigraph restored — model back to QCP form."
