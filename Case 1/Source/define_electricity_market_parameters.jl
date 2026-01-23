# ==============================================================================
# DEFINE ELECTRICITY MARKET PARAMETERS
# ==============================================================================
# This function initializes the Electricity Market structure with prices, balances,
# and ADMM penalty parameters. The electricity market coordinates supply and demand
# between generators (VRES, Conventional) and consumers.
#
# Market Structure:
#   - Supply: VRES generators, Conventional generators
#   - Demand: Electricity consumers, Electrolytic H2 producers (electricity input)
#   - Clearing: Supply = Demand (enforced via ADMM)
#
# Arguments:
#   - market::Dict: Dictionary to be populated with market parameters (modified in-place)
#   - data::Dict: Configuration dictionary containing market settings
#                 Contains: initial_price, rho_initial, nTimesteps, nReprDays
#   - ts::Dict: Dictionary of time series DataFrames, keyed by year (used to determine years)
#   - repr_days::Dict: Dictionary of representative day DataFrames (not used here, kept for consistency)
#
# Returns:
#   - Modifies the market dictionary in-place with price, balance, and rho structures
# ==============================================================================
function define_electricity_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    
    # --- 1. EXTRACT DIMENSIONS ---
    # Retrieve the number of time steps per day (e.g., 24 for hourly resolution)
    # This determines the temporal resolution of the market
    nTimesteps = data["nTimesteps"]
    
    # Retrieve the number of representative days (e.g., 3)
    # This determines how many representative days are used to represent the full year
    nReprDays = data["nReprDays"]
    
    # Extract the keys from the time series dictionary to define the Years/Scenarios set
    # Example: If ts has key 2021, then Y = [2021]
    # This allows for multi-year or multi-scenario analysis
    Y = keys(ts)

    # --- 2. INITIALIZE CONTAINERS ---
    # Initialize the dictionary to store electricity prices (p^E) for each year
    # Prices are dual variables (Lagrange multipliers) that will be updated by ADMM
    # Structure: market["price"][year] = matrix of size (nTimesteps × nReprDays)
    # We use a Dict for years to handle potentially non-sequential year keys (e.g., 2025, 2030)
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances (Total Supply - Total Demand)
    # Balance represents the net imbalance: positive = excess supply, negative = excess demand
    # At equilibrium, balance should be zero (Supply = Demand)
    # Structure: market["balance"][year] = matrix of size (nTimesteps × nReprDays)
    market["balance"] = Dict()
    
    # Initialize Rho (ADMM penalty parameter)
    # This controls the strength of the penalty for market imbalances
    # Higher rho = stronger penalty = faster constraint satisfaction
    # Lower rho = weaker penalty = slower constraint satisfaction but better dual convergence
    # Default: 1.0 if not specified
    market["rho"] = get(data, "rho_initial", 1.0)

    # --- 3. POPULATE DATA PER YEAR ---
    # Loop through each year available in the time series data
    for y in Y
        # Retrieve the initial price guess from the data, defaulting to 50.0 if not specified
        # This is the starting value for the dual variable (Lagrange multiplier)
        # The ADMM algorithm will update this iteratively based on market imbalances
        # Typical electricity prices: 30-80 EUR/MWh (depending on market)
        # Default: 50.0 EUR/MWh if not specified
        p_init = get(data, "initial_price", 50.0)
        
        # Create a matrix of initial prices filled with the seed value
        # Dimensions are (nTimesteps × nReprDays)
        # All time steps and representative days start with the same initial price
        # Prices will become time-dependent as ADMM updates them based on imbalances
        market["price"][y] = fill(Float64(p_init), (nTimesteps, nReprDays))

        # Initialize the balance matrix with zeros (assuming equilibrium start or zero imbalance)
        # At the start, we assume no imbalance (Supply = Demand)
        # The balance will be calculated in update_market_balances! during ADMM iterations
        # Dimensions are (nTimesteps × nReprDays)
        market["balance"][y] = zeros(Float64, nTimesteps, nReprDays)
    end

    # Print a confirmation message indicating the electricity market parameters are set
    # This helps track the initialization progress during setup
    println("Defined Electricity Market parameters.")
end
