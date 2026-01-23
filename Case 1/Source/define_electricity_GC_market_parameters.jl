# ==============================================================================
# DEFINE ELECTRICITY GREEN CERTIFICATE (GC) MARKET PARAMETERS
# ==============================================================================
# This function initializes the Electricity GC Market structure with prices, balances,
# and ADMM penalty parameters. The GC market coordinates supply and demand for
# green certificates that certify renewable electricity generation.
#
# Market Structure:
#   - Supply: VRES generators (produce 1 GC per 1 MWh of renewable electricity)
#   - Demand: Electrolytic H2 producers (buy GCs to certify green hydrogen),
#             Electricity GC Demand Agents (buy GCs for environmental compliance)
#   - Clearing: Supply = Demand (enforced via ADMM)
#
# Green Certificate Chain:
#   - VRES generates electricity + GCs (1:1 ratio)
#   - Electrolytic H2 producers buy Elec GCs to certify their hydrogen as green
#   - The green backing constraint ensures: gc_e_buy >= e_H * gc_h_sell
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
function define_electricity_GC_market_parameters!(market::Dict, data::Dict, ts::Dict, repr_days::Dict)
    
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
    # Initialize the dictionary to store Electricity GC prices (p^{GC,E}) for each year
    # Prices are dual variables (Lagrange multipliers) that will be updated by ADMM
    # These represent the price premium for green-certified electricity
    # Structure: market["price"][year] = matrix of size (nTimesteps × nReprDays)
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances
    # Balance = VRES Generation (GC Supply) - GC Consumed - GC Demand Agent purchases
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
        # GC prices might start low or at a policy floor (e.g., 5.0 EUR/Certificate)
        # Green certificate prices are typically lower than commodity prices
        # They represent a premium for environmental attributes
        # Typical GC prices: 1-20 EUR/Certificate (depending on policy and market conditions)
        # We default to 5.0 EUR/Certificate if "initial_price" is not specified in the YAML
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

    # Print a confirmation message indicating the electricity GC market parameters are set
    # This helps track the initialization progress during setup
    println("Defined Electricity GC Market parameters.")
end
