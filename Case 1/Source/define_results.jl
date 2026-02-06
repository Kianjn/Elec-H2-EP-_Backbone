# ==============================================================================
# DEFINE RESULTS DATA STRUCTURES
# ==============================================================================
# This function initializes data structures for storing ADMM convergence history
# and final agent results. These structures will be populated during and after
# the optimization loop.
#
# The function creates:
#   1. ADMM convergence tracking: iteration numbers, residuals, wall time
#   2. Price history: Evolution of market prices (optional, for debugging)
#   3. Agent results containers: Nested dictionaries for each agent's final values
#
# Arguments:
#   - data::Dict: Configuration dictionary containing general settings (read-only)
#   - results::Dict: Dictionary to be populated with result structures (modified in-place)
#   - ADMM::Dict: Dictionary to be populated with ADMM metrics (modified in-place)
#   - agents::Dict: Dictionary containing lists of agent IDs by type
#   - elec_market::Dict: Electricity market dictionary (read-only, used for structure)
#   - H2_market::Dict: Hydrogen market dictionary (read-only, used for structure)
#   - elec_GC_market::Dict: Electricity GC market dictionary (read-only, used for structure)
#   - H2_GC_market::Dict: Hydrogen GC market dictionary (read-only, used for structure)
#   - EP_market::Dict: End Product market dictionary (read-only, used for structure)
#
# Returns:
#   - Modifies results and ADMM dictionaries in-place
# ==============================================================================
function define_results!(data::Dict, results::Dict, ADMM::Dict, agents::Dict, 
                         elec_market::Dict, H2_market::Dict, 
                         elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict)
    
    # --- 1. ADMM CONVERGENCE HISTORY ---
    # Initialize lists to store metrics for every iteration.
    # This allows us to plot convergence graphs later to visualize algorithm performance.
    
    # List to store iteration numbers (1, 2, 3, ...)
    # This tracks which iteration each metric corresponds to
    ADMM["iter"] = []
    
    # List to store the Primal Residual (max imbalance) at each step
    # Primal residual measures how far markets are from equilibrium (Supply = Demand)
    # Should decrease over iterations as the algorithm converges
    ADMM["primal_residual"] = []
    
    # List to store the Dual Residual (price update magnitude) at each step
    # Dual residual measures solution stability (how much prices are changing)
    # Should decrease over iterations as the solution stabilizes
    ADMM["dual_residual"] = []
    
    # List to store diagnostic information about where the maximum imbalance occurs
    # Each element is a dictionary containing: market, year, time, repr_day, imbalance, abs_imbalance
    # This helps identify which market and time period is causing convergence issues
    ADMM["diagnostics"] = []
    
    # Variable to store total wall-clock time of the simulation
    # This will be calculated after the ADMM loop completes
    # Units: minutes (converted from nanoseconds in Main.jl)
    ADMM["walltime"] = 0.0
    
    # --- PRICE HISTORY (OPTIONAL BUT RECOMMENDED FOR DEBUGGING) ---
    # We store the evolution of prices to see if they oscillate or converge smoothly.
    # This helps diagnose convergence issues and understand market dynamics.
    # Currently initialized as empty dictionaries - can be populated if needed
    ADMM["prices_E"] = Dict()   # Electricity Prices evolution
    ADMM["prices_H2"] = Dict()  # Hydrogen Prices evolution
    ADMM["prices_EP"] = Dict()  # End Product Prices (Shadow Prices) evolution

    # --- 2. FINAL RESULTS CONTAINERS ---
    # Initialize nested dictionaries for every agent to store their final variable values
    # (e.g., generation, consumption, costs) once the algorithm finishes.
    # Structure: results["AgentType"][agent_id] = Dict() (to be populated later)
    
    # Power Sector Agents
    # Creates a dictionary entry for each power agent ID
    # Example: results["Power"]["Gen_VRES_01"] = Dict()
    results["Power"] = Dict(id => Dict() for id in agents[:power])
    
    # Hydrogen Sector Agents
    # Creates a dictionary entry for each hydrogen agent ID
    # Example: results["Hydrogen"]["Prod_H2_Green"] = Dict()
    results["Hydrogen"] = Dict(id => Dict() for id in agents[:H2])
    
    # Offtaker Agents (Green and Grey)
    # Creates a dictionary entry for each offtaker agent ID
    # Example: results["Offtaker"]["Offtaker_Green"] = Dict()
    results["Offtaker"] = Dict(id => Dict() for id in agents[:offtaker])
    
    # Note: Electricity GC Demand Agents are not included in results structure
    # They can be added here if detailed results are needed for these agents

    # Print a confirmation message indicating structures are ready
    # This helps track the initialization progress during setup
    # Initialization print removed to reduce console noise
end
