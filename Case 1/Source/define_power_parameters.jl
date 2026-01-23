# ==============================================================================
# DEFINE POWER SECTOR AGENT PARAMETERS
# ==============================================================================
# This function defines parameters specific to Power Sector agents.
# It handles three types of agents:
#   1. VRES (Variable Renewable Energy Source) Generators: Solar, Wind, etc.
#   2. Conventional Generators: Gas turbines, coal plants, etc.
#   3. Electricity Consumers: Elastic demand with quadratic utility
#
# The function processes configuration data and time series to create agent-specific
# parameters such as capacity profiles, demand profiles, and cost parameters.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Gen_VRES_01")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Agent-specific configuration dictionary from data.yaml
#                 Contains: Type, Capacity, Profile_Column, MarginalCost, etc.
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year
#               Contains hourly profiles (e.g., SOLAR, LOAD_E columns)
#   - repr_days::Dict: Dictionary of representative day DataFrames, keyed by year
#                      Contains mapping from repr day index to actual day number
#
# Returns:
#   - Modifies the JuMP model in-place by storing agent-specific parameters
# ==============================================================================
function define_power_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict)
    
    # --- 1. EXTRACT DIMENSIONS ---
    # Retrieve the number of time steps per day (e.g., 24 for hourly resolution)
    # This is used for indexing into the time series data
    nTimesteps = data["nTimesteps"]
    
    # Retrieve the number of representative days (e.g., 3)
    # This determines how many representative days are used to represent the full year
    nReprDays = data["nReprDays"]
    
    # --- 2. VRES GENERATOR LOGIC (Variable Renewable Energy Source) ---
    # VRES generators have time-dependent capacity that varies with weather conditions
    # Examples: Solar PV (varies with irradiance), Wind (varies with wind speed)
    # Check if the agent type is "VRES" (e.g., Solar or Wind)
    if data["Type"] == "VRES"
        # Initialize a dictionary to store time-dependent capacity limits (Q_bar) for each year
        # Structure: Q_bar[year] = matrix of size (nTimesteps × nReprDays)
        # Each element Q_bar[y][t,r] represents the maximum available capacity at time t, day r, year y
        Q_bar = Dict()
        
        # Loop through each year/scenario available in the time series data
        for jy in keys(ts)
            # Identify the column name in the CSV (e.g., :SOLAR) corresponding to this agent's profile
            # The Profile_Column in data.yaml specifies which time series column to use
            # This column contains normalized capacity factors (0-1) representing availability
            profile_name = Symbol(data["Profile_Column"])
            
            # Calculate the capacity limit for every time step (jh) and representative day (jd)
            # Logic: Multiply Installed Capacity by the normalized profile value (0-1) from the time series
            # 
            # The indexing logic maps the representative day index back to the specific row in the annual time series:
            #   - repr_days[jy][!,:periods][jd] gives the actual day number (1-365) for representative day jd
            #   - nTimesteps*(periods[jd]-1) calculates the starting hour index for that day
            #   - + jh gives the specific hour within that day
            #   - round(Int, ...) converts to integer index for DataFrame access
            #
            # Example: If repr_day 1 corresponds to day 12, and jh=5 (5th hour):
            #   Index = round(Int, 24*(12-1) + 5) = round(Int, 269) = 269
            #   This accesses hour 269 of the year (which is hour 5 of day 12)
            #
            # Formula: Q_bar[y][t,r] = Capacity × Profile[t,r]
            # This creates a time-dependent capacity matrix
            Q_bar[jy] = [
                data["Capacity"] * ts[jy][!, profile_name][round(Int, nTimesteps*(repr_days[jy][!,:periods][jd]-1) + jh)] 
                for jh in 1:nTimesteps, jd in 1:nReprDays
            ]
        end
        
        # Store the calculated time-dependent capacity dictionary in the JuMP model
        # This will be used in build_power_agent! to set capacity constraints
        model[:Q_bar] = Q_bar
        
        # Store the marginal cost parameter (typically close to 0 for VRES) in the model
        # Marginal cost for renewables is very low (near zero) since there's no fuel cost
        # Only includes variable O&M costs
        # Units: EUR/MWh
        model[:C] = data["MarginalCost"]

    # --- 3. CONVENTIONAL GENERATOR LOGIC ---
    # Conventional generators have constant capacity (not weather-dependent)
    # Examples: Gas turbines, coal plants, nuclear
    # Check if the agent type is "Conventional" (e.g., Gas Turbine)
    elseif data["Type"] == "Conventional"
        # Store the constant capacity limit (scalar value) directly in the model
        # Unlike VRES, conventional generators have fixed nameplate capacity
        # This is a single value, not a time-dependent matrix
        # Units: MW
        model[:Q_bar] = data["Capacity"]
        
        # Store the marginal cost parameter (e.g., fuel + CO2 cost) in the model
        # Marginal cost for conventional generators includes:
        #   - Fuel costs (natural gas, coal, etc.)
        #   - CO2 emission costs (carbon taxes, ETS prices)
        #   - Variable O&M costs
        # Units: EUR/MWh
        model[:C] = data["MarginalCost"]

    # --- 4. ELECTRICITY CONSUMER LOGIC ---
    # Consumers have elastic demand with a quadratic utility function
    # Check if the agent type is "Consumer"
    elseif data["Type"] == "Consumer"
        # Initialize a dictionary to store time-dependent demand limits (D_bar)
        # Structure: D_bar[year] = matrix of size (nTimesteps × nReprDays)
        # Each element D_bar[y][t,r] represents the maximum demand at time t, day r, year y
        D_bar = Dict()
        
        # Loop through each year in the time series data
        for jy in keys(ts)
            # Identify the column name for the load profile (e.g., :LOAD_E)
            # The Load_Column in data.yaml specifies which time series column to use
            # This column contains normalized load factors (0-1) representing demand patterns
            profile_name = Symbol(data["Load_Column"])
            
            # Calculate the demand profile for every time step and representative day
            # Logic: Multiply Peak Load by the normalized load profile value from the time series
            # 
            # Similar indexing logic as VRES:
            #   - Maps representative day index to actual day number
            #   - Accesses the specific hour in the annual time series
            #   - Multiplies peak load by normalized profile (0-1)
            #
            # Formula: D_bar[y][t,r] = PeakLoad × LoadProfile[t,r]
            # This creates a time-dependent demand limit matrix
            D_bar[jy] = [
                data["PeakLoad"] * ts[jy][!, profile_name][round(Int, nTimesteps*(repr_days[jy][!,:periods][jd]-1) + jh)] 
                for jh in 1:nTimesteps, jd in 1:nReprDays
            ]
        end
        
        # Store the calculated demand profile dictionary in the model
        # This will be used in build_power_agent! to set demand constraints
        model[:D_bar] = D_bar
        
        # Store quadratic utility function parameters
        # These define the consumer's willingness to pay for electricity
        # Utility function: U(d) = A_E * d - (1/2) * B_E * d^2
        # 
        # A_E: Intercept of inverse demand curve (willingness to pay intercept)
        #      Higher A_E = higher willingness to pay = more inelastic demand
        #      Units: EUR/MWh
        #      Default: 500.0 if not specified
        model[:A_E] = get(data, "A_E", 500.0)
        
        # B_E: Slope of inverse demand curve (price sensitivity)
        #      Higher B_E = more price-sensitive = more elastic demand
        #      Units: EUR/MWh²
        #      Default: 0.5 if not specified
        model[:B_E] = get(data, "B_E", 0.5)
    end

    # Print a confirmation message indicating parameters have been defined for this agent
    # This helps track the initialization progress during setup
    println("Defined power parameters for agent: $agent_id")
end
