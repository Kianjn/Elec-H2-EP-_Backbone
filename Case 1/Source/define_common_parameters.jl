# ==============================================================================
# DEFINE COMMON PARAMETERS FOR ALL AGENTS
# ==============================================================================
# This function initializes parameters that are common to all agent models.
# It sets up the fundamental sets (time steps, representative days, years) and
# calculates representative day weights for the optimization problem.
#
# The function is called for every agent during the initialization phase to
# establish the problem structure before agent-specific parameters are defined.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Configuration dictionary containing general settings (nTimesteps, nReprDays, rho_initial)
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year (e.g., ts[2021])
#   - repr_days::Dict: Dictionary of representative day DataFrames, keyed by year
#                      Contains columns: periods, weights, selected_periods
#   - agents::Dict: Dictionary containing lists of all agent IDs (for reference)
#
# Returns:
#   - Modifies the JuMP model in-place by storing sets and weights
# ==============================================================================
function define_common_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict, agents::Dict)
    
    # --- 1. DEFINE SETS ---
    # Sets define the dimensions and indexing structure of the optimization problem
    
    # T: Time steps within a representative day
    # Creates a range from 1 to nTimesteps (e.g., 1:24 for hourly resolution)
    # This represents the hours within a single day
    T = 1:data["nTimesteps"]
    
    # R: Representative days
    # Creates a range from 1 to nReprDays (e.g., 1:3 for 3 representative days)
    # Representative days are selected days that capture the full year's variability
    # This reduces computational complexity from 8760 hours to nReprDays × nTimesteps
    R = 1:data["nReprDays"]
    
    # Y: Years/scenarios
    # Extracts the keys from the time series dictionary to define years/scenarios
    # Example: If ts has key 2021, then Y = [2021]
    # This allows for multi-year or multi-scenario analysis
    Y = keys(ts)

    # --- 2. STORE SETS IN MODEL ---
    # Store the sets in the JuMP model for easy access during constraint and objective building
    # These will be used extensively in @variable, @constraint, and @objective macros
    
    # Store the time step set T in the JuMP model
    # Used for indexing variables and constraints by time (e.g., q_E[t,r,y])
    model[:T] = T
    
    # Store the representative day set R in the JuMP model
    # Used for indexing variables and constraints by representative day
    model[:R] = R
    
    # Store the year/scenario set Y in the JuMP model
    # Used for indexing variables and constraints by year/scenario
    model[:Y] = Y

    # --- 3. CALCULATE REPRESENTATIVE DAY WEIGHTS ---
    # Weights represent the frequency/probability of each representative day in the full year
    # They are used to scale objective function terms to account for how often each
    # representative day occurs in the actual year
    
    # Initialize a dictionary to store the weights (probability/frequency) of each representative day
    # Structure: W[year][representative_day_index] = weight_value
    W = Dict()
    
    # Loop through each year in the available data
    for jy in Y
        # Map each representative day index 'jd' to its weight found in the CSV data
        # repr_days[jy][!,:weights][jd] extracts the weight for representative day jd in year jy
        # The weights column in the CSV contains the frequency (number of days) that each
        # representative day represents in the full year
        # Example: If repr_day 1 represents 50 days, weight = 50
        W[jy] = Dict(jd => repr_days[jy][!,:weights][jd] for jd in R)
    end
    
    # Store the weight dictionary W in the JuMP model for use in objective functions
    # These weights are multiplied in objective function terms to properly account for
    # the temporal frequency of each representative day
    # Example: sum(W[y][r] * profit[t,r,y] for t in T, r in R, y in Y)
    model[:W] = W

    # --- 4. INITIALIZE ADMM PARAMETERS ---
    # Set the initial penalty parameter rho from the configuration data
    # This value will be dynamically updated later by the ADMM algorithm (see update_rho.jl)
    # rho controls the strength of the penalty for deviating from market consensus
    # Higher rho = stronger penalty = faster constraint satisfaction
    # Lower rho = weaker penalty = slower constraint satisfaction but better dual convergence
    # Typical initial value: 1.0
    model[:rho] = data["rho_initial"]

    # Print a confirmation message to the console for debugging purposes
    # This helps track which agents have been initialized during the setup phase
    println("Defined common parameters for agent: $agent_id")
end
