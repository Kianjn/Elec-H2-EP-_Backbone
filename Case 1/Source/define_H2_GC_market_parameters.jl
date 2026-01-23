# ==============================================================================
# DEFINE HYDROGEN GREEN CERTIFICATE (GC) MARKET PARAMETERS
# ==============================================================================
# This function initializes the Hydrogen GC Market structure with prices, balances,
# and ADMM penalty parameters. The H2 GC market coordinates supply and demand for
# green certificates that certify renewable hydrogen production.
#
# Market Structure:
#   - Supply: Electrolytic H2 producers (produce H2 GCs when using green electricity)
#   - Demand: Electrolytic Ammonia Producers (buy H2 GCs for green ammonia),
#             Grey Ammonia Producers (buy H2 GCs to meet 42% policy mandate)
#   - Clearing: Supply = Demand (enforced via ADMM)
#
# Green Certificate Chain:
#   - Electrolytic H2 producers buy Elec GCs and sell H2 GCs
#   - Green backing constraint: gc_e_buy >= e_H * gc_h_sell (ensures green chain)
#   - Both green and grey ammonia producers must buy H2 GCs (42% mandate)
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
function define_H2_GC_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    
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
    # Initialize the dictionary to store Hydrogen GC prices (p^{GC,H}) for each year
    # Prices are dual variables (Lagrange multipliers) that will be updated by ADMM
    # These represent the price premium for green-certified hydrogen
    # Structure: market["price"][year] = matrix of size (nTimesteps × nReprDays)
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances
    # Balance = H2 GC Sold (by Electrolytic H2 producers) - H2 GC Bought (by Ammonia producers)
    # Balance represents the net imbalance: positive = excess supply, negative = excess demand
    # At equilibrium, balance should be zero (Supply = Demand)
    # Structure: market["balance"][year] = matrix of size (nTimesteps × nReprDays)
    market["balance"] = Dict()
    
    # Initialize Rho (ADMM penalty parameter)
    # This controls the strength of the penalty for market imbalances
    # Default: 1.0 if not specified
    market["rho"] = get(data, "rho_initial", 1.0)

    # --- 3. POPULATE DATA PER YEAR ---
    # Loop through each year available in the time series data
    for y in Y
        # Retrieve the initial price guess
        # Default initialization to 5.0 EUR/Certificate if not specified in YAML
        # Green certificate prices are typically lower than commodity prices
        # They represent a premium for environmental attributes
        # Typical H2 GC prices: 1-20 EUR/Certificate (depending on policy and market conditions)
        # The price may be higher than Elec GCs due to the additional green backing requirement
        p_init = get(data, "initial_price", 5.0)
        
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

    # Print a confirmation message indicating the Hydrogen GC market parameters are set
    # This helps track the initialization progress during setup
    println("Defined Hydrogen GC Market parameters.")
end
