# ==============================================================================
# DEFINE HYDROGEN MARKET PARAMETERS
# ==============================================================================
# This function initializes the Hydrogen Market structure with prices, balances,
# and ADMM penalty parameters. The hydrogen market coordinates supply and demand
# between producers (Electrolytic) and consumers (Electrolytic Ammonia Producers).
#
# Market Structure:
#   - Supply: Electrolytic H2 producers (sell hydrogen)
#   - Demand: Electrolytic Ammonia Producers (buy hydrogen as feedstock)
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
function define_H2_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    
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
    # Initialize the dictionary to store hydrogen prices (p^H) for each year
    # Prices are dual variables (Lagrange multipliers) that will be updated by ADMM
    # Structure: market["price"][year] = matrix of size (nTimesteps × nReprDays)
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances (Sum(H2 Supply) - Sum(H2 Demand))
    # Balance represents the net imbalance: positive = excess supply, negative = excess demand
    # At equilibrium, balance should be zero (Supply = Demand)
    # Structure: market["balance"][year] = matrix of size (nTimesteps × nReprDays)
    market["balance"] = Dict()
    
    # Initialize Rho (ADMM penalty parameter)
    # This controls the strength of the penalty for market imbalances
    # Default: 1.0 if not specified
    market["rho"] = get(data, "rho_initial", 1.0)

    # Store a scaling factor for price updates to account for representative days.
    # Scaling = (nTimesteps * nReprDays) / 8760.0 (fraction of full-year hours represented).
    # This keeps ADMM penalty updates comparable to a full-year hourly model.
    market["scale"] = (nTimesteps * nReprDays) / 8760.0

    # --- 3. POPULATE DATA PER YEAR ---
    # Loop through each year available in the time series data
    for y in Y
        # Retrieve the initial price guess
        # Hydrogen prices are typically higher than electricity due to:
        #   - Energy conversion losses (electrolysis efficiency ~67%)
        #   - Additional processing and storage costs
        # Typical hydrogen prices: 100-200 EUR/MWh (depending on market conditions)
        # We default to 150.0 EUR/MWh if "initial_price" is not specified in the YAML
        p_init = get(data, "initial_price", 150.0)
        
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

    # Print a confirmation message indicating the hydrogen market parameters are set
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise
end
