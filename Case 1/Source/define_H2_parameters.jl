# ==============================================================================
# DEFINE HYDROGEN SECTOR AGENT PARAMETERS
# ==============================================================================
# This function defines parameters specific to Hydrogen Sector agents.
# It handles two types of agents:
#   1. Electrolytic Hydrogen Producer: Uses electricity to produce green hydrogen
#   2. Hydrogen Consumer: Consumes hydrogen with elastic demand
#
# The function processes configuration data to create agent-specific parameters
# such as capacity limits, efficiency factors, and cost parameters.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Prod_H2_Green")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Agent-specific configuration dictionary from data.yaml
#                 Contains: Type, Capacity_Electrolyzer, SpecificConsumption, etc.
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year
#               Used for hydrogen consumer demand profiles
#   - repr_days::Dict: Dictionary of representative day DataFrames, keyed by year
#                      Used for mapping representative days to actual days
#
# Returns:
#   - Modifies the JuMP model in-place by storing agent-specific parameters
# ==============================================================================
function define_H2_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict)
    
    # --- 1. EXTRACT DIMENSIONS ---
    # Retrieve the number of time steps per day (e.g., 24 for hourly resolution)
    # This is used for indexing into the time series data
    nTimesteps = data["nTimesteps"]
    
    # Retrieve the number of representative days (e.g., 3)
    # This determines how many representative days are used to represent the full year
    nReprDays = data["nReprDays"]

    # --- 2. ELECTROLYTIC HYDROGEN PRODUCER LOGIC ---
    # Electrolytic producers convert electricity into hydrogen using electrolyzers
    # Check if the agent type is "GreenProducer" (Electrolyzer)
    if data["Type"] == "GreenProducer"
        # Store the electricity consumption per unit of Hydrogen (MWh_e / MWh_H2)
        # This is the inverse of efficiency (e.g., if efficiency eta=67%, e_H ~ 1.5)
        # It represents how much electricity is needed to produce one unit of hydrogen
        # Typical values: 1.4-1.6 MWh_e per MWh_H2 (depending on electrolyzer technology)
        # Units: MWh_electricity / MWh_hydrogen
        model[:e_H] = data["SpecificConsumption"]
        
        # Store the Input Capacity Limit (Maximum Electricity Consumption in MW)
        # This is the maximum power that the electrolyzer can consume
        # It represents the nameplate capacity of the electrolyzer
        # Units: MW (electricity input)
        model[:E_bar] = data["Capacity_Electrolyzer"]
        
        # Store the Output Capacity Limit (Maximum Hydrogen Production in MW)
        # This is the maximum hydrogen production rate
        # It is related to E_bar through the efficiency: H_bar ≈ E_bar / e_H
        # Units: MW (hydrogen output)
        model[:H_bar] = data["Capacity_H2_Output"]
        
        # Store the variable operational cost
        # This represents variable O&M costs for the electrolyzer
        # Note: In the LaTeX, this is C_H(e) = c_{H,0} * e + (1/2) * c_{H,1} * e^2
        # For linear version: C_H(e) = c_{H,0} * e_buy
        # The cost is based on electricity input, not hydrogen output
        # Units: EUR/MWh_electricity (for linear cost function)
        model[:C_H] = data["OperationalCost"]

    # --- 3. HYDROGEN CONSUMER LOGIC ---
    # Consumers have elastic demand for hydrogen
    # Check if the agent type is "Consumer"
    elseif data["Type"] == "Consumer"
        # Initialize a dictionary to store time-dependent Hydrogen demand limits
        # Structure: D_H_bar[year] = matrix of size (nTimesteps × nReprDays)
        # Each element D_H_bar[y][t,r] represents the maximum demand at time t, day r, year y
        D_H_bar = Dict()
        
        # Loop through each year available in the time series data
        for jy in keys(ts)
            # Identify the column name for the hydrogen load profile (e.g., :LOAD_H)
            # The Load_Column in data.yaml specifies which time series column to use
            # This column contains normalized load factors (0-1) representing demand patterns
            profile_name = Symbol(data["Load_Column"])
            
            # Calculate the hydrogen demand profile for every time step and representative day
            # Logic: Multiply Peak Load by the normalized load profile value from the time series
            # 
            # The indexing logic maps the representative day index back to the specific row in the annual time series:
            #   - repr_days[jy][!,:periods][jd] gives the actual day number (1-365) for representative day jd
            #   - nTimesteps*(periods[jd]-1) calculates the starting hour index for that day
            #   - + jh gives the specific hour within that day
            #   - round(Int, ...) converts to integer index for DataFrame access
            #
            # Formula: D_H_bar[y][t,r] = PeakLoad × LoadProfile[t,r]
            # This creates a time-dependent demand limit matrix
            D_H_bar[jy] = [
                data["PeakLoad"] * ts[jy][!, profile_name][round(Int, nTimesteps*(repr_days[jy][!,:periods][jd]-1) + jh)] 
                for jh in 1:nTimesteps, jd in 1:nReprDays
            ]
        end
        
        # Store the calculated hydrogen demand profile in the model
        # This will be used in build_H2_agent! to set demand constraints
        model[:D_H_bar] = D_H_bar
        
        # Check if the "WillingnessToPay" parameter exists (for elastic demand logic)
        # This represents the consumer's utility per unit of hydrogen consumed
        # If present, it enables elastic demand behavior
        # Units: EUR/MWh_hydrogen
        if haskey(data, "WillingnessToPay")
            # Store the willingness to pay parameter
            # This will be used in the objective function as a linear utility term
            model[:Utility] = data["WillingnessToPay"]
        end
    end

    # Print a confirmation message indicating parameters have been defined for this agent
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise
end
