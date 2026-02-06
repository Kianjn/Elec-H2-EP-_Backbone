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
#   - Clearing: Supply = Demand (enforced via ADMM) - ANNUAL CLEARING
#
# IMPORTANT: This market clears ANNUALLY, not hourly.
#   - H2 producers produce GCs hourly (for backing constraint) but sell them annually (aggregate)
#   - Ammonia producers buy GCs annually (aggregate) to meet annual mandate
#   - This allows "GC banking" and eliminates structural conflict between hourly clearing and annual constraints
#
# Green Certificate Chain:
#   - Electrolytic H2 producers buy Elec GCs hourly (for backing) and sell H2 GCs annually
#   - Green backing constraint: gc_e_buy >= e_H * gc_h_sell (ensures green chain) - ANNUAL
#   - Both green and grey ammonia producers must buy H2 GCs (42% mandate) - ANNUAL
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
    # Initialize the dictionary to store Hydrogen GC prices (p^{GC,H}) for each year.
    # Prices are dual variables (Lagrange multipliers) updated by ADMM and represent
    # the price premium for green-certified hydrogen.
    # Prices are modeled as annual scalars, while GC production and purchases are hourly.
    # Structure: market["price"][year] = scalar (Float64).
    market["price"] = Dict()
    
    # Initialize the dictionary to store market balances.
    # Balance = H2 GC Sold (by Electrolytic H2 producers) - H2 GC Bought (by Ammonia producers).
    # Balance represents the net imbalance: positive = excess supply, negative = excess demand.
    # At equilibrium, balance should be zero (Supply = Demand).
    # Balance is modeled as an annual scalar aggregated across all hours and days.
    # Structure: market["balance"][year] = scalar (Float64).
    market["balance"] = Dict()
    
    # Initialize Rho (ADMM penalty parameter)
    # This controls the strength of the penalty for market imbalances
    # Default: 1.0 if not specified
    market["rho"] = get(data, "rho_initial", 1.0)

    # Store a scaling factor for price updates to account for representative days.
    # For the annual H2 GC market, we use the same fraction of full-year hours that
    # the representative days cover, to keep step sizes consistent with hourly markets.
    market["scale"] = (nTimesteps * nReprDays) / 8760.0

    # --- 3. POPULATE DATA PER YEAR ---
    # Loop through each year available in the time series data
    for y in Y
        # Retrieve the initial price guess.
        # Default initialization is 5.0 EUR/Certificate if not specified in YAML.
        # Green certificates represent a premium for environmental attributes; typical
        # H2 GC prices might be in the 1–20 EUR/Certificate range depending on policy.
        p_init = get(data, "initial_price", 5.0)
        
        # Initialize the price for this year as a single annual scalar rather than a time-dependent matrix.
        # This reflects the annual clearing nature of the H2 GC market; ADMM updates this
        # value based on annual aggregate imbalances.
        market["price"][y] = Float64(p_init)

        # Initialize the annual H2 GC balance with zero (Supply = Demand at start).
        # The balance will later be calculated in update_market_balances! by aggregating
        # hourly GC flows with representative-day weights.
        market["balance"][y] = 0.0
    end

    # Print a confirmation message indicating the Hydrogen GC market parameters are set
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise in large simulations
end
