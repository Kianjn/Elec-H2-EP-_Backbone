# ==============================================================================
# DEFINE END PRODUCT (EP) MARKET PARAMETERS
# ==============================================================================
# This function initializes the End Product Market structure with prices, balances,
# fixed demand profile, and ADMM penalty parameters. The EP market coordinates
# competition between Green and Grey Ammonia Producers to meet a fixed demand.
#
# Market Structure:
#   - Supply: Electrolytic Ammonia Producers (green), Grey Ammonia Producers (grey)
#   - Demand: Fixed demand profile (exogenous, not an agent)
#   - Clearing: Total Supply = Fixed Demand (enforced via ADMM)
#
# This is a coordination market (not a true market) where the shadow price (lambda_EP)
# coordinates the competition between green and grey producers to meet the fixed demand.
#
# Arguments:
#   - market::Dict: Dictionary to be populated with market parameters (modified in-place)
#   - data::Dict: Configuration dictionary containing market settings
#                 Contains: initial_price, rho_initial, Demand_Column, Total_Demand, nTimesteps, nReprDays
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year
#               Contains hourly profiles (e.g., LOAD_EP column for demand profile)
#   - repr_days::Dict: Dictionary of representative day DataFrames, keyed by year
#                      Contains mapping from repr day index to actual day number
#
# Returns:
#   - Modifies the market dictionary in-place with price, balance, demand, and rho structures
# ==============================================================================
function define_EP_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    
    # --- 1. EXTRACT DIMENSIONS ---
    # Retrieve the number of time steps per day (e.g., 24 for hourly resolution)
    # This determines the temporal resolution of the market
    nTimesteps = data["nTimesteps"]
    
    # Retrieve the number of representative days (e.g., 3)
    # This determines how many representative days are used to represent the full year
    nReprDays = data["nReprDays"]
    
    # Extract the keys from the time series dictionary to define the Years/Scenarios set
    # Example: If ts has key 2021, then Y = [2021]
    Y = keys(ts)

    # --- 2. INITIALIZE CONTAINERS ---
    # Initialize the dictionary to store the End Product shadow prices (Lambda)
    # This price coordinates the competition between Green and Grey offtakers
    # The shadow price represents the marginal value of meeting the fixed demand
    # Structure: market["price"][year] = matrix of size (nTimesteps × nReprDays)
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances (Total Supply - Fixed Demand)
    # Balance represents the net imbalance: positive = excess supply, negative = excess demand
    # At equilibrium, balance should be zero (Total Supply = Fixed Demand)
    # Structure: market["balance"][year] = matrix of size (nTimesteps × nReprDays)
    market["balance"] = Dict()
    
    # Initialize the dictionary to store the fixed demand profile for the End Product
    # This is the exogenous demand that green and grey producers compete to meet
    # Structure: market["demand"][year] = matrix of size (nTimesteps × nReprDays)
    # Each element represents the fixed demand at time t, day r, year y
    market["demand"] = Dict()
    
    # Initialize Rho (ADMM penalty parameter)
    # The solver (ADMM!) expects every market dictionary to have a "rho" key for the penalty parameter
    # We retrieve this from the passed 'data' dictionary (which includes ADMM settings via merge in main.jl)
    # If not found, we default to 1.0 to prevent crashes
    market["rho"] = get(data, "rho_initial", 1.0)

    # --- 3. POPULATE DATA PER YEAR ---
    # Loop through each year available in the time series data
    for y in Y
        # --- INITIALIZE PRICE ---
        # Retrieve the initial price guess from the data
        # This acts as the starting signal for the ADMM algorithm
        # The shadow price will be updated iteratively based on supply-demand imbalances
        # Typical End Product (Ammonia) prices: 300-600 EUR/MWh (depending on market)
        # Defaulting to 500.0 EUR/MWh if "initial_price" is not specified in YAML
        p_init = get(data, "initial_price", 500.0)
        
        # Create a matrix of initial prices filled with the seed value
        # Dimensions are (nTimesteps × nReprDays)
        # All time steps and representative days start with the same initial price
        # Prices will become time-dependent as ADMM updates them based on imbalances
        market["price"][y] = fill(Float64(p_init), (nTimesteps, nReprDays))

        # --- INITIALIZE BALANCE ---
        # Initialize the balance matrix with zeros
        # This will later hold the residual (Total Supply - Fixed Demand)
        # At the start, we assume no imbalance
        # The balance will be calculated in update_market_balances! during ADMM iterations
        # Dimensions are (nTimesteps × nReprDays)
        market["balance"][y] = zeros(Float64, nTimesteps, nReprDays)

        # --- LOAD FIXED DEMAND PROFILE ---
        # Identify the column name for the End Product load profile in the time series CSV
        # This column contains normalized load factors (0-1) representing demand patterns
        # Defaults to "LOAD_EP" if not specified in data.yaml
        col_name = Symbol(get(data, "Demand_Column", "LOAD_EP"))
        
        # Retrieve the total peak demand for the system (MW)
        # This is the maximum demand that the system must meet
        # The actual demand at each time step is: PeakLoad × NormalizedProfile[t]
        # Defaults to 10.0 MW if not specified
        peak_load = get(data, "Total_Demand", 10.0)

        # Calculate the specific demand profile for every time step and representative day
        # Logic: Multiply Peak Load by the normalized load profile value from the time series
        # 
        # The indexing maps representative days back to the full year time series:
        #   - repr_days[y][!,:periods][jd] gives the actual day number (1-365) for representative day jd
        #   - nTimesteps*(periods[jd]-1) calculates the starting hour index for that day
        #   - + jh gives the specific hour within that day
        #   - round(Int, ...) converts to integer index for DataFrame access
        #
        # Formula: demand[y][t,r] = PeakLoad × LoadProfile[t,r]
        # This creates a time-dependent demand matrix
        # The demand is fixed (exogenous) and green/grey producers compete to meet it
        market["demand"][y] = [
            peak_load * ts[y][!, col_name][round(Int, nTimesteps*(repr_days[y][!,:periods][jd]-1) + jh)] 
            for jh in 1:nTimesteps, jd in 1:nReprDays
        ]
    end

    # Print a confirmation message indicating the End Product coordination parameters are set
    # This helps track the initialization progress during setup
    println("Defined End Product Coordination parameters.")
end
