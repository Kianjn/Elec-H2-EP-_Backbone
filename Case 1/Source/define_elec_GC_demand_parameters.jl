# ==============================================================================
# DEFINE ELECTRICITY GC DEMAND AGENT PARAMETERS
# ==============================================================================
# This function defines parameters for Electricity Green Certificate (GC) Demand Agents.
# These agents have a demand for green certificates with a quadratic utility function,
# representing environmental preferences or regulatory compliance requirements.
#
# The function processes configuration data and time series to create:
#   - Utility function parameters (A_GC, B_GC)
#   - Demand profile limits (D_GC_E_bar)
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Demand_GC_Elec_01")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Agent-specific configuration dictionary from data.yaml
#                 Contains: PeakLoad, Load_Column, A_GC, B_GC
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year
#               Contains hourly profiles (e.g., LOAD_E column)
#   - repr_days::Dict: Dictionary of representative day DataFrames, keyed by year
#                      Contains mapping from repr day index to actual day number
#
# Returns:
#   - Modifies the JuMP model in-place by storing agent-specific parameters
# ==============================================================================
function define_elec_GC_demand_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict)
    
    # --- 1. EXTRACT DIMENSIONS ---
    # Retrieve the number of time steps per day (e.g., 24 for hourly resolution)
    # This is used for indexing into the time series data
    nTimesteps = data["nTimesteps"]
    
    # Retrieve the number of representative days (e.g., 3)
    # This determines how many representative days are used to represent the full year
    nReprDays = data["nReprDays"]

    # --- 2. UTILITY FUNCTION PARAMETERS ---
    # These parameters define the quadratic utility function for GC demand
    # Utility function: U(d) = A_GC * d - (1/2) * B_GC * d^2
    # This is a concave quadratic function representing diminishing marginal utility
    
    # A_GC: Intercept of inverse demand curve (willingness to pay intercept)
    #       Higher A_GC = higher willingness to pay = more inelastic demand
    #       Represents the base valuation of green certificates
    #       Units: EUR/Certificate
    #       Default: 20.0 if not specified (typical value)
    model[:A_GC] = get(data, "A_GC", 20.0)
    
    # B_GC: Slope of inverse demand curve (price sensitivity)
    #       Higher B_GC = more price-sensitive = more elastic demand
    #       Controls the curvature of the utility function
    #       Units: EUR/Certificate²
    #       Default: 0.5 if not specified (typical value)
    model[:B_GC] = get(data, "B_GC", 0.5)

    # --- 3. DEMAND PROFILE ---
    # Initialize a dictionary to store time-dependent GC demand limits
    # Structure: D_GC_E_bar[year] = matrix of size (nTimesteps × nReprDays)
    # Each element D_GC_E_bar[y][t,r] represents the maximum GC demand at time t, day r, year y
    # The demand profile typically follows a similar pattern to electricity demand
    D_GC_E_bar = Dict()
    
    # Loop through each year available in the time series data
    for jy in keys(ts)
        # Use the same load profile as electricity demand (similar pattern)
        # GC demand often correlates with electricity demand patterns
        # If Load_Column is specified in data.yaml, use it; otherwise use LOAD_E as default
        profile_name = Symbol(get(data, "Load_Column", "LOAD_E"))
        
        # Calculate the GC demand profile for every time step and representative day
        # Logic: Multiply Peak Load by the normalized load profile value from the time series
        # 
        # The indexing logic maps the representative day index back to the specific row in the annual time series:
        #   - repr_days[jy][!,:periods][jd] gives the actual day number (1-365) for representative day jd
        #   - nTimesteps*(periods[jd]-1) calculates the starting hour index for that day
        #   - + jh gives the specific hour within that day
        #   - round(Int, ...) converts to integer index for DataFrame access
        #
        # Formula: D_GC_E_bar[y][t,r] = PeakLoad × LoadProfile[t,r]
        # This creates a time-dependent demand limit matrix
        # PeakLoad default: 10.0 if not specified (typical value in MW)
        D_GC_E_bar[jy] = [
            get(data, "PeakLoad", 10.0) * ts[jy][!, profile_name][round(Int, nTimesteps*(repr_days[jy][!,:periods][jd]-1) + jh)] 
            for jh in 1:nTimesteps, jd in 1:nReprDays
        ]
    end
    
    # Store the calculated GC demand profile in the model
    # This will be used in build_elec_GC_demand_agent! to set demand constraints
    model[:D_GC_E_bar] = D_GC_E_bar

    # Print a confirmation message indicating parameters have been defined for this agent
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise
end
