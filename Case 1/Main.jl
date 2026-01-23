# ==============================================================================
# MAIN SCRIPT: MULTI-AGENT ENERGY MARKET SIMULATION USING ADMM
# by Kian Jafarinejad - PhD researcher at TPM Faculty of TU Delft (K.Jafarinejad@tudelft.nl)
# ==============================================================================
# This is the main entry point for the multi-agent energy system optimization.
# The script:
#   1. Loads and activates the Julia environment
#   2. Loads all required packages
#   3. Reads configuration and input data
#   4. Initializes agents and markets
#   5. Builds optimization models for all agents
#   6. Runs the ADMM algorithm to find market equilibrium
#   7. Saves results to CSV files
#
# System Overview:
#   - Agents: VRES generators, Conventional generators, Electricity consumers,
#             Electrolytic H2 producers, Ammonia producers (green & grey),
#             Electricity GC demand agents
#   - Markets: Electricity, Electricity GC, Hydrogen, Hydrogen GC, End Product
#   - Algorithm: ADMM (Alternating Direction Method of Multipliers)
#   - Solver: Gurobi (commercial optimization solver)
# ==============================================================================

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ==============================================================================

# Activate the environment relative to this file's location
# This ensures that the correct package versions are used for this project
using Pkg  # Load the Julia Package Manager to handle dependencies
Pkg.activate(@__DIR__)  # Activate the project environment located in the same directory as this script
# @__DIR__ is a Julia macro that returns the directory containing the current file

# ==============================================================================
# SECTION 2: PACKAGE LOADING
# ==============================================================================

# Load all required packages for the simulation
# These packages provide functionality for optimization, data handling, and I/O

using JuMP  # Load JuMP for mathematical optimization modeling
            # JuMP provides a high-level interface for building optimization models

using Gurobi  # Load Gurobi solver (must be installed and licensed)
              # Gurobi is a commercial optimization solver (MIP, QP, etc.)
              # Alternative: Can use open-source solvers like HiGHS, Clp, etc.

using DataFrames  # Load DataFrames for handling tabular data
                  # Used for reading CSV files and organizing time series data

using CSV  # Load CSV for reading and writing comma-separated files
           # Used for input/output of time series and results

using YAML  # Load YAML for reading configuration files
            # Used to read agent and market configurations from data.yaml

using DataStructures  # Load DataStructures for advanced collections
                      # Provides additional data structures beyond base Julia

using ProgressBars  # Load ProgressBars for iteration tracking
                   # Can be used to show progress during long computations

using Printf  # Load Printf for formatted output strings
               # Used for formatted printing of convergence metrics

using TimerOutputs  # Load TimerOutputs for performance profiling
                     # Tracks execution time of different code sections

using ArgParse  # Load ArgParse for parsing command-line arguments
                # Can be used to make the script configurable via command line

using Base.Threads: @spawn  # Import threading macro for parallel execution
                            # Can be used to parallelize agent subproblem solves

using Base: split  # Import split function from standard library
                   # Utility function for string manipulation

# Create a global Gurobi environment
# This is a single environment instance that can be reused across multiple models
# Helps with license management and performance
const GUROBI_ENV = Gurobi.Env()

# Print confirmation message to indicate successful package loading
println("Environment successfully activated and packages loaded.")

# ==============================================================================
# SECTION 3: DIRECTORY SETUP
# ==============================================================================

# Home directory: Define root directory based on script location
# This is used throughout the script to construct file paths
# @__DIR__ returns the directory containing Main.jl
const home_dir = @__DIR__

# ==============================================================================
# SECTION 4: FUNCTION LOADING
# ==============================================================================
# Load all source files containing function definitions
# These files define the optimization models, ADMM algorithm, and helper functions

# --- PARAMETER DEFINITION FUNCTIONS ---
# These functions define parameters for agents and markets from configuration data

include(joinpath(home_dir, "Source", "define_common_parameters.jl"))  # Define common agent parameters (sets, weights, ADMM settings)
include(joinpath(home_dir, "Source", "define_power_parameters.jl"))  # Define Power Sector agent parameters (VRES, Conventional, Consumer)
include(joinpath(home_dir, "Source", "define_H2_parameters.jl"))  # Define Hydrogen Sector agent parameters (Electrolytic Producer, Consumer)
include(joinpath(home_dir, "Source", "define_offtaker_parameters.jl"))  # Define Offtaker agent parameters (Green & Grey Ammonia Producers)
include(joinpath(home_dir, "Source", "define_elec_GC_demand_parameters.jl"))  # Define Electricity GC Demand Agent parameters

# --- MARKET PARAMETER DEFINITION FUNCTIONS ---
# These functions initialize market structures (prices, balances, rho)

include(joinpath(home_dir, "Source", "define_electricity_market_parameters.jl"))  # Define Electricity Market parameters
include(joinpath(home_dir, "Source", "define_H2_market_parameters.jl"))  # Define Hydrogen Market parameters
include(joinpath(home_dir, "Source", "define_electricity_GC_market_parameters.jl"))  # Define Electricity GC Market parameters
include(joinpath(home_dir, "Source", "define_H2_GC_market_parameters.jl"))  # Define Hydrogen GC Market parameters
include(joinpath(home_dir, "Source", "define_EP_market_parameters.jl"))  # Define End Product coordination parameters

# --- AGENT MODEL BUILDING FUNCTIONS ---
# These functions construct JuMP optimization models for each agent type

include(joinpath(home_dir, "Source", "build_power_agent.jl"))  # Build Power Agent optimization models
include(joinpath(home_dir, "Source", "build_H2_agent.jl"))  # Build Hydrogen Agent optimization models
include(joinpath(home_dir, "Source", "build_offtaker_agent.jl"))  # Build Offtaker Agent optimization models
include(joinpath(home_dir, "Source", "build_elec_GC_demand_agent.jl"))  # Build Electricity GC Demand Agent optimization models

# --- ADMM AND SOLVING FUNCTIONS ---
# These functions implement the ADMM algorithm and agent subproblem solving

include(joinpath(home_dir, "Source", "define_results.jl"))  # Initialize result storage structures
include(joinpath(home_dir, "Source", "ADMM.jl"))  # Main ADMM algorithm loop
include(joinpath(home_dir, "Source", "ADMM_subroutine.jl"))  # ADMM helper functions (balances, residuals, price updates)
include(joinpath(home_dir, "Source", "solve_power_agent.jl"))  # Solve Power Agent subproblem
include(joinpath(home_dir, "Source", "solve_H2_agent.jl"))  # Solve Hydrogen Agent subproblem
include(joinpath(home_dir, "Source", "solve_offtaker_agent.jl"))  # Solve Offtaker Agent subproblem
include(joinpath(home_dir, "Source", "solve_elec_GC_demand_agent.jl"))  # Solve Electricity GC Demand Agent subproblem
include(joinpath(home_dir, "Source", "update_rho.jl"))  # Update ADMM penalty parameter (adaptive strategy)
include(joinpath(home_dir, "Source", "save_results.jl"))  # Save simulation results to CSV files

# ==============================================================================
# SECTION 5: DATA LOADING
# ==============================================================================

# Load configuration file containing agent definitions and market parameters
# data.yaml contains:
#   - General settings (nTimesteps, nReprDays, etc.)
#   - ADMM parameters (max_iter, epsilon, rho_initial)
#   - Market initial prices
#   - Agent configurations (types, capacities, costs, etc.)
data = YAML.load_file(joinpath(home_dir, "Data", "data.yaml"))

# Initialize dictionaries for time series and representative day data
ts = Dict()  # Dictionary for time series data, keyed by year (e.g., ts[2021])
             # Contains hourly profiles: SOLAR, LOAD_E, LOAD_H, LOAD_EP, etc.

order_matrix = Dict()  # Dictionary for ordering variables (not currently used, kept for compatibility)
                       # Could be used for representative day selection algorithms

repr_days = Dict()  # Dictionary for representative days, keyed by year
                   # Contains: periods (actual day numbers), weights (frequencies), selected_periods

years = Dict(1 => 2021)  # Define the year mapping (Scenario 1 -> 2021)
                          # Allows for multi-scenario analysis in the future

# Load time series data for 2021
# This CSV contains hourly profiles for the entire year (8760 hours)
# Columns: Time, SOLAR, LOAD_E, LOAD_H, LOAD_EP, WIND_ONSHORE, etc.
# Each column contains normalized profiles (0-1) representing availability/demand patterns
ts[2021] = CSV.read(joinpath(home_dir, "Input", "timeseries_2021.csv"), DataFrame)

# Load ordering variable for 2021
# This file contains information about representative day selection
# Used by representative day selection algorithms (not directly used in optimization)
order_matrix[2021] = CSV.read(joinpath(home_dir, "Input", "output_2021", "ordering_variable.csv"), delim=",", DataFrame)

# Load representative days data for 2021
# This CSV contains the selected representative days and their weights
# Columns: periods (actual day number 1-365), weights (frequency), selected_periods (boolean)
# The weights represent how many days each representative day represents in the full year
repr_days[2021] = CSV.read(joinpath(home_dir, "Input", "output_2021", "decision_variables_short.csv"), delim=",", DataFrame)

# ==============================================================================
# SECTION 6: RESULTS FOLDER CREATION
# ==============================================================================

# Create Results Folder
# Folder name format: "Results_{nReprDays}_repr_days" (e.g., "Results_3_repr_days")
# This allows multiple runs with different representative day configurations
# Check if results folder exists (isdir returns 1 if exists, 0 if not)
if isdir(joinpath(home_dir, string("Results_", data["General"]["nReprDays"], "_repr_days"))) != 1
    # Create results folder if it doesn't exist
    # This folder will contain all output CSV files and performance profiles
    mkdir(joinpath(home_dir, string("Results_", data["General"]["nReprDays"], "_repr_days")))
end

# ==============================================================================
# SECTION 7: AGENT INITIALIZATION
# ==============================================================================

# Initialize dictionary for agent IDs organized by type
# This structure allows easy iteration over agent types during ADMM
agents = Dict()

# Get list of Power Sector agent IDs from configuration
# Power agents include: VRES generators, Conventional generators, Electricity consumers
agents[:power] = [id for id in keys(data["Power"])]

# Get list of Hydrogen Sector agent IDs from configuration
# Hydrogen agents include: Electrolytic H2 producers, Hydrogen consumers
agents[:H2] = [id for id in keys(data["Hydrogen"])]

# Get list of Offtaker agent IDs from configuration
# Offtaker agents include: Electrolytic Ammonia Producers (green), Grey Ammonia Producers
agents[:offtaker] = [id for id in keys(data["Hydrogen_Offtaker"])]

# Get list of Electricity GC Demand Agent IDs (if exists in data)
# These agents are optional - check if the section exists in data.yaml
# If it exists, extract agent IDs; otherwise, use empty list
agents[:elec_GC_demand] = haskey(data, "Electricity_GC_Demand") ? [id for id in keys(data["Electricity_GC_Demand"])] : []

# Create combined list of all agents
# This is used for creating JuMP models and for result storage
agents[:all] = union(agents[:power], agents[:H2], agents[:offtaker], agents[:elec_GC_demand])

# ==============================================================================
# SECTION 8: MARKET INITIALIZATION
# ==============================================================================

# Initialize lists for agents in each market (currently not used, kept for compatibility)
# These could be used to track which agents participate in which markets
agents[:elec_market] = []  # Initialize list for agents in Electricity Market
agents[:H2_market] = []  # Initialize list for agents in Hydrogen Market
agents[:elec_GC_market] = []  # Initialize list for agents in Electricity GC Market
agents[:H2_GC_market] = []  # Initialize list for agents in Hydrogen GC Market
agents[:EP_market] = []  # Initialize list for agents in End Product coordination

# Create JuMP models for all agents
# Each agent gets its own optimization model container
# Gurobi.Optimizer specifies that Gurobi will be used as the solver
# Structure: mdict[agent_id] = JuMP.Model
mdict = Dict(i => Model(Gurobi.Optimizer) for i in agents[:all])

# ==============================================================================
# SECTION 9: MARKET PARAMETER DEFINITION
# ==============================================================================

# Initialize market parameter dictionaries
# These will be populated by the market definition functions
elec_market = Dict{String,Any}()  # Initialize Electricity Market parameters
H2_market = Dict{String,Any}()  # Initialize Hydrogen Market parameters
elec_GC_market = Dict{String,Any}()  # Initialize Electricity GC Market parameters
H2_GC_market = Dict{String,Any}()  # Initialize Hydrogen GC Market parameters
EP_market = Dict{String,Any}()  # Initialize End Product coordination parameters

# Store agent counts for each market (currently not used, kept for compatibility)
# These could be used for market-specific logic or reporting
elec_market["nAgents"] = length(agents[:elec_market])  # Store agent count for Electricity Market
H2_market["nAgents"] = length(agents[:H2_market])  # Store agent count for Hydrogen Market
elec_GC_market["nAgents"] = length(agents[:elec_GC_market])  # Store agent count for Electricity GC Market
H2_GC_market["nAgents"] = length(agents[:H2_GC_market])  # Store agent count for Hydrogen GC Market
EP_market["nAgents"] = length(agents[:offtaker])  # Store agent count for End Product coordination

# Define market parameters by calling market definition functions
# merge() combines General, ADMM, and market-specific settings
# This ensures all markets have access to common parameters (nTimesteps, nReprDays, rho_initial)
define_electricity_market_parameters!(elec_market, merge(data["General"], data["ADMM"], data["elec_market"]), ts, repr_days)
define_H2_market_parameters!(H2_market, merge(data["General"], data["ADMM"], data["H2_market"]), ts, repr_days)
define_electricity_GC_market_parameters!(elec_GC_market, merge(data["General"], data["ADMM"], data["elec_GC_market"]), ts, repr_days)
define_H2_GC_market_parameters!(H2_GC_market, merge(data["General"], data["ADMM"], data["H2_GC_market"]), ts, repr_days)
define_EP_market_parameters!(EP_market, merge(data["General"], data["ADMM"], data["EP_market"]), ts, repr_days)

# ==============================================================================
# SECTION 10: AGENT PARAMETER DEFINITION
# ==============================================================================

# Define parameters for each agent type
# This sets up the problem structure (sets, weights, agent-specific parameters)
# Parameters are stored in the JuMP models for use during model building

# --- Power Sector Agents ---
# Iterate over Power Sector agents and define their parameters
for m in agents[:power]
    # Define common parameters (sets, weights, ADMM settings)
    # merge() combines General, ADMM, and agent-specific settings
    define_common_parameters!(m, mdict[m], merge(data["General"], data["ADMM"], data["Power"][m]), ts, repr_days, agents)
    
    # Define Power Sector-specific parameters (capacity profiles, costs, utility parameters)
    define_power_parameters!(m, mdict[m], merge(data["General"], data["Power"][m]), ts, repr_days)
end

# --- Hydrogen Sector Agents ---
# Iterate over Hydrogen Sector agents and define their parameters
for m in agents[:H2]
    # Define common parameters (sets, weights, ADMM settings)
    define_common_parameters!(m, mdict[m], merge(data["General"], data["ADMM"], data["Hydrogen"][m]), ts, repr_days, agents)
    
    # Define Hydrogen Sector-specific parameters (capacities, efficiency, costs)
    define_H2_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen"][m]), ts, repr_days)
end

# --- Offtaker Agents (Ammonia Producers) ---
# Iterate over Offtaker agents and define their parameters
for m in agents[:offtaker]
    # Define common parameters (sets, weights, ADMM settings)
    define_common_parameters!(m, mdict[m], merge(data["General"], data["ADMM"], data["Hydrogen_Offtaker"][m]), ts, repr_days, agents)
    
    # Define Offtaker-specific parameters (conversion factors, capacities, costs, gamma_NH3)
    define_offtaker_parameters!(m, mdict[m], merge(data["General"], data["Hydrogen_Offtaker"][m]), ts, repr_days)
end

# --- Electricity GC Demand Agents ---
# Iterate over Electricity GC Demand agents and define their parameters
for m in agents[:elec_GC_demand]
    # Define common parameters (sets, weights, ADMM settings)
    define_common_parameters!(m, mdict[m], merge(data["General"], data["ADMM"], data["Electricity_GC_Demand"][m]), ts, repr_days, agents)
    
    # Define Electricity GC Demand-specific parameters (utility parameters, demand profiles)
    define_elec_GC_demand_parameters!(m, mdict[m], merge(data["General"], data["Electricity_GC_Demand"][m]), ts, repr_days)
end

# ==============================================================================
# SECTION 11: BUILD OPTIMIZATION MODELS
# ==============================================================================

# Build JuMP optimization models for each agent type
# This creates the variables, constraints, and objective functions for each agent
# The models are ready to be solved once market prices are updated

# --- Power Sector Agents ---
# Iterate over Power Sector agents and build their optimization models
for m in agents[:power]
    # Build optimization model for Power Agent
    # Passes electricity and GC markets so agents can see current prices
    build_power_agent!(m, mdict[m], elec_market, elec_GC_market)
end

# --- Hydrogen Sector Agents ---
# Iterate over Hydrogen Sector agents and build their optimization models
for m in agents[:H2]
    # Build optimization model for Hydrogen Agent
    # Passes H2 and H2 GC markets so agents can see current prices
    build_H2_agent!(m, mdict[m], H2_market, H2_GC_market)
end

# --- Offtaker Agents (Ammonia Producers) ---
# Iterate over Offtaker agents and build their optimization models
for m in agents[:offtaker]
    # Build optimization model for Offtaker Agent
    # Passes EP_market so they can see the End Product price signal
    # Also passes H2 and H2 GC markets for green offtakers
    build_offtaker_agent!(m, mdict[m], EP_market, H2_market, H2_GC_market)
end

# --- Electricity GC Demand Agents ---
# Iterate over Electricity GC Demand agents and build their optimization models
for m in agents[:elec_GC_demand]
    # Build optimization model for Electricity GC Demand Agent
    # Passes elec_GC_market so agents can see current GC prices
    build_elec_GC_demand_agent!(m, mdict[m], elec_GC_market)
end

# ==============================================================================
# SECTION 12: RUN ADMM ALGORITHM
# ==============================================================================

# Initialize result storage structures
results = Dict()  # Initialize results dictionary for storing final agent results
ADMM = Dict()  # Initialize ADMM metrics dictionary for storing convergence history
TO = TimerOutput()  # Initialize Timer for performance profiling

# Initialize result structures
# This creates the data structures that will store convergence metrics and agent results
# Added EP_market for tracking End Product market results
define_results!(merge(data["General"], data["ADMM"]), results, ADMM, agents, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market)

# Execute the main ADMM loop
# This iteratively solves agent subproblems, updates market prices, and checks convergence
# The algorithm continues until convergence or max_iter is reached
ADMM!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, mdict, agents, data, TO)

# Calculate total wall-clock time of the simulation
# TimerOutputs.tottime(TO) returns total time in nanoseconds
# Convert to minutes: multiply by 10^-9 (nanoseconds to seconds) then divide by 60 (seconds to minutes)
ADMM["walltime"] = TimerOutputs.tottime(TO) * 10^-9 / 60

# ==============================================================================
# SECTION 13: SAVE RESULTS
# ==============================================================================

# Save simulation results to CSV files
# Creates three main output files:
#   1. ADMM_Convergence.csv: Convergence history (iterations, residuals)
#   2. Market_Prices.csv: Final market prices (electricity, hydrogen)
#   3. Agent_Summary.csv: Summary statistics for each agent
save_results(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, ADMM, results, agents)

# Save performance profile to YAML file
# This contains detailed timing information for each code section
# Useful for identifying performance bottlenecks
YAML.write_file(joinpath(home_dir, string("Results_", data["General"]["nReprDays"], "_repr_days"), string("TimerOutput.yaml")), TO)
