# ==============================================================================
# ADMM SUBROUTINES
# ==============================================================================
# This file contains the helper functions used within the main ADMM loop to:
#   1. Update market balances (Supply - Demand) from agent decisions
#   2. Update market prices (Dual variables) based on imbalances
#   3. Calculate convergence residuals (Primal and Dual)
#
# These functions are called iteratively during the ADMM algorithm to coordinate
# the distributed optimization and drive the system toward market equilibrium.
# ==============================================================================

# ==============================================================================
# UPDATE MARKET BALANCES
# ==============================================================================
# This function calculates the net imbalance (Supply - Demand) for all markets
# based on the current agent decisions. It aggregates contributions from all
# agents and updates the balance matrices for each market.
#
# Market Balance Calculation:
#   Balance = Total Supply - Total Demand
#   - Positive balance = excess supply (prices should decrease)
#   - Negative balance = excess demand (prices should increase)
#   - Zero balance = market equilibrium (Supply = Demand)
#
# The function processes all agent types:
#   - Power Sector: Generators (supply), Consumers (demand)
#   - Hydrogen Sector: Electrolytic Producers (supply), Consumers (demand)
#   - Offtakers: Ammonia Producers (EP supply, H2 demand, GC demand)
#   - Electricity GC Demand: GC demand agents
#
# Arguments:
#   - agents::Dict: Dictionary containing lists of agent IDs by type
#   - mdict::Dict: Dictionary of JuMP models, keyed by agent ID
#   - elec::Dict: Electricity market dictionary (modified in-place)
#   - h2::Dict: Hydrogen market dictionary (modified in-place)
#   - elec_gc::Dict: Electricity GC market dictionary (modified in-place)
#   - h2_gc::Dict: Hydrogen GC market dictionary (modified in-place)
#   - ep::Dict: End Product market dictionary (modified in-place)
#   - data::Dict: Configuration dictionary (used for dimensions, not modified)
#
# Returns:
#   - Modifies all market balance dictionaries in-place
# ==============================================================================
function update_market_balances!(agents, mdict, elec, h2, elec_gc, h2_gc, ep, data)
    
    # --- 1. RESET BALANCES TO INITIAL STATES ---
    # Iterate over each year/scenario to reset all balance matrices to zero
    # This ensures we start fresh for each ADMM iteration
    for y in keys(elec["balance"])
        # Reset Electricity Market Balance to 0.0
        # This will accumulate: Generation - Consumption - Electrolytic H2 input
        elec["balance"][y] .= 0.0
        
        # Reset Hydrogen Market Balance to 0.0
        # This will accumulate: H2 Production - H2 Consumption
        h2["balance"][y] .= 0.0
        
        # Reset Electricity GC Market Balance to 0.0
        # This will accumulate: VRES GC Production - GC Consumption - GC Demand Agent purchases
        elec_gc["balance"][y] .= 0.0
        
        # Reset Hydrogen GC Market Balance to 0.0
        # This will accumulate: H2 GC Production - H2 GC Consumption (green + grey offtakers)
        # 
        # STRUCTURAL NOTE: Annual constraints (green backing, 42% mandate) allow temporal flexibility,
        # meaning agents can buy/sell GCs at different times as long as annual totals match.
        # However, ADMM enforces hourly market clearing (Supply = Demand at each hour).
        # This creates a fundamental trade-off: annual compliance vs. perfect hourly clearing.
        # As a result, small persistent hourly imbalances (~1-10 MWh) may remain even when
        # annual constraints are satisfied. This is expected behavior and not a bug.
        h2_gc["balance"][y] .= 0.0
        
        # Reset End Product Balance
        # Since Balance = Total Supply - Total Demand, we initialize it with negative Demand.
        # This represents the "hole" that agents need to fill with supply.
        # The fixed demand is exogenous (not an agent), so we subtract it here
        # Then we add supply from green and grey ammonia producers
        # At equilibrium: Supply - Demand = 0, so balance = 0
        ep["balance"][y] .= -ep["demand"][y]
    end

    # --- 2. ADD CONTRIBUTIONS FROM POWER SECTOR AGENTS ---
    # Loop through all agents identified as Power Sector
    # Power agents include: VRES generators, Conventional generators, Electricity consumers
    for id in agents[:power]
        # Retrieve the JuMP model for this agent
        # The model contains the solved optimization variables with their optimal values
        m = mdict[id]
        
        # Loop through years to process each year's contributions
        for y in keys(elec["balance"])
            # CRITICAL: Check for Consumer FIRST (before Generator check)
            # Consumers have both :d_E (actual variable) and :q_E (alias), so we must check :d_E first
            # to correctly identify them as consumers, not generators
            if haskey(m, :d_E)
                 # This is a Consumer (has demand variable d_E)
                 # Consumers have elastic demand with quadratic utility function
                 # Subtract consumption from Electricity Balance (Demand)
                 # This reduces the balance (negative contribution to balance)
                 elec["balance"][y] .-= Array(value.(m[:d_E][:,:,y]))
            # Check if agent is a Generator (has generation variable q_E, but NOT d_E)
            # Generators include: VRES (solar, wind) and Conventional (gas, coal)
            elseif haskey(m, :q_E)
                 # Add generation to Electricity Balance (Supply)
                 # value.(m[:q_E][:,:,y]) extracts the optimal generation values
                 # This is a JuMP DenseAxisArray, so we convert to standard Array to avoid broadcasting errors
                 # The .+= operator adds the generation matrix element-wise to the balance
                 elec["balance"][y] .+= Array(value.(m[:q_E][:,:,y]))
                 
                 # If VRES, they also produce Green Certificates
                 # VRES generators produce 1 GC per 1 MWh of electricity (1:1 ratio)
                 # Check if the model has gc_E variable (VRES indicator)
                 if haskey(m, :gc_E)
                     # Add GCs to Electricity GC Balance (Supply)
                     # For VRES: gc_E = q_E (1 MWh electricity = 1 GC)
                     elec_gc["balance"][y] .+= Array(value.(m[:gc_E][:,:,y]))
                 end
            end
        end
    end

    # --- 3. ADD CONTRIBUTIONS FROM HYDROGEN SECTOR AGENTS ---
    # Loop through all Hydrogen Sector agents
    # Hydrogen agents include: Electrolytic H2 producers, Hydrogen consumers
    for id in agents[:H2]
        # Retrieve the JuMP model for this agent
        m = mdict[id]
        
        # Loop through years to process each year's contributions
        for y in keys(elec["balance"])
            # Check if Electrolytic Producer (buys Elec, sells H2)
            # Identified by presence of e_buy variable (electricity purchase)
            if haskey(m, :e_buy)
                # Subtract electricity consumption (Demand)
                # Electrolytic H2 producers consume electricity as input
                elec["balance"][y] .-= Array(value.(m[:e_buy][:,:,y]))
                
                # Subtract Elec GC consumption (Demand)
                # Electrolytic H2 producers buy Elec GCs to certify their hydrogen as green
                # Required by the green backing constraint
                elec_gc["balance"][y] .-= Array(value.(m[:gc_e_buy][:,:,y]))
                
                # Add Hydrogen production (Supply)
                # Electrolytic H2 producers sell hydrogen to the market
                h2["balance"][y] .+= Array(value.(m[:h_sell][:,:,y]))
                
                # Add Hydrogen GC production (Supply)
                # Electrolytic H2 producers sell H2 GCs when using green electricity
                h2_gc["balance"][y] .+= Array(value.(m[:gc_h_sell][:,:,y]))
            # Note: Only Electrolytic Producers exist now (no conventional H2 producers)
            # Check if Hydrogen Consumer
            # Identified by presence of d_H variable (hydrogen demand)
            elseif haskey(m, :d_H)
                # Subtract Hydrogen consumption (Demand)
                # Hydrogen consumers reduce the H2 market balance
                h2["balance"][y] .-= Array(value.(m[:d_H][:,:,y]))
            end
        end
    end

    # --- 4. ADD CONTRIBUTIONS FROM OFFTAKER AGENTS (AMMONIA PRODUCERS) ---
    # Loop through Offtaker agents (Green and Grey)
    # Offtakers include: Electrolytic Ammonia Producers, Grey Ammonia Producers
    for id in agents[:offtaker]
        # Retrieve the JuMP model for this agent
        m = mdict[id]
        
        # Loop through years to process each year's contributions
        for y in keys(elec["balance"])
            # Check if Green Offtaker (Electrolytic Ammonia Producer - buys H2)
            # Identified by presence of h_buy variable (hydrogen purchase)
            if haskey(m, :h_buy)
                # Subtract Hydrogen consumption (Demand)
                # Green offtakers buy hydrogen as feedstock for ammonia production
                h2["balance"][y] .-= Array(value.(m[:h_buy][:,:,y]))
                
                # Subtract Hydrogen GC consumption (Demand)
                # Green offtakers buy H2 GCs to certify their ammonia as green
                # Required by the 42% policy mandate
                h2_gc["balance"][y] .-= Array(value.(m[:gc_h_buy][:,:,y]))
            end
            
            # Check if Grey Ammonia Producer (buys H2 GCs directly)
            # Identified by presence of gc_h_buy_G variable (H2 GC purchase for grey producer)
            # Grey producers don't buy physical H2 (they produce it internally via SMR)
            # But they must buy H2 GCs to meet the 42% policy mandate
            if haskey(m, :gc_h_buy_G)
                # Subtract Hydrogen GC consumption (Demand)
                # Grey producers buy H2 GCs based on their internal H2 consumption equivalent
                h2_gc["balance"][y] .-= Array(value.(m[:gc_h_buy_G][:,:,y]))
            end
            
            # Check if selling End Product (Both Green and Grey do this)
            # Identified by presence of ep_sell variable (End Product sales)
            if haskey(m, :ep_sell)
                # Add End Product production (Supply)
                # This fills the "hole" created by the fixed demand in step 1
                # Both green and grey ammonia producers contribute to meeting the fixed demand
                ep["balance"][y] .+= Array(value.(m[:ep_sell][:,:,y]))
            end
        end
    end

    # --- 5. ADD CONTRIBUTIONS FROM ELECTRICITY GC DEMAND AGENTS ---
    # Loop through Electricity GC Demand agents (if any exist)
    # These agents have demand for green certificates with quadratic utility
    # Check if the agent list exists and is non-empty (these agents are optional)
    if haskey(agents, :elec_GC_demand)
        for id in agents[:elec_GC_demand]
            # Retrieve the JuMP model for this agent
            m = mdict[id]
            
            # Loop through years to process each year's contributions
            for y in keys(elec["balance"])
                # Check if agent has GC demand variable
                # All Electricity GC Demand agents should have this variable
                if haskey(m, :d_GC_E)
                    # Subtract GC consumption from Electricity GC Balance (Demand)
                    # These agents purchase GCs for environmental compliance or preferences
                    elec_gc["balance"][y] .-= Array(value.(m[:d_GC_E][:,:,y]))
                end
            end
        end
    end
end

# ==============================================================================
# UPDATE MARKET PRICES (DUAL VARIABLES)
# ==============================================================================
# This function updates the dual variables (market prices) based on the calculated
# market imbalances. This is the dual update step in the ADMM algorithm.
#
# Dual Ascent Update Rule (for Maximization problem):
#   Lambda_new = Lambda_old - Rho * Balance
#
# Interpretation:
#   - If balance > 0 (excess supply): price decreases (encourages more demand)
#   - If balance < 0 (excess demand): price increases (encourages more supply)
#   - The step size is controlled by rho (penalty parameter)
#   - Higher rho = larger price adjustments = faster convergence
#
# Arguments:
#   - elec::Dict: Electricity market dictionary (modified in-place)
#   - h2::Dict: Hydrogen market dictionary (modified in-place)
#   - elec_gc::Dict: Electricity GC market dictionary (modified in-place)
#   - h2_gc::Dict: Hydrogen GC market dictionary (modified in-place)
#   - ep::Dict: End Product market dictionary (modified in-place)
#
# Returns:
#   - Modifies all market price dictionaries in-place
# ==============================================================================
function update_prices!(elec, h2, elec_gc, h2_gc, ep)
    # Loop through years to update prices for each year/scenario
    for y in keys(elec["price"])
        # Update Electricity Price
        # Price decreases if excess supply, increases if excess demand
        # The .-= operator performs element-wise subtraction
        # elec["rho"] is a scalar, so it's broadcast across the entire balance matrix
        elec["price"][y] .-= elec["rho"] .* elec["balance"][y]
        
        # Update Hydrogen Price
        # Same logic: price adjusts based on H2 market imbalance
        h2["price"][y] .-= h2["rho"] .* h2["balance"][y]
        
        # Update Electricity GC Price
        # Price adjusts based on Elec GC market imbalance
        elec_gc["price"][y] .-= elec_gc["rho"] .* elec_gc["balance"][y]
        
        # Update Hydrogen GC Price
        # Price adjusts based on H2 GC market imbalance
        h2_gc["price"][y] .-= h2_gc["rho"] .* h2_gc["balance"][y]
        
        # Update End Product Shadow Price
        # This coordinates the competition between green and grey ammonia producers
        # Price adjusts based on EP market imbalance (Supply - Fixed Demand)
        ep["price"][y] .-= ep["rho"] .* ep["balance"][y]
    end
end

# ==============================================================================
# CALCULATE CONVERGENCE RESIDUALS
# ==============================================================================
# This function calculates the convergence residuals used to determine if the
# ADMM algorithm has reached equilibrium.
#
# Primal Residual:
#   Maximum absolute imbalance across all markets and all time steps
#   Represents how far the markets are from equilibrium (Supply = Demand)
#   Should approach zero as the algorithm converges
#
# Dual Residual:
#   Proxy for stationarity (how stable the solution is)
#   Currently set equal to Primal Residual for simplicity
#   In a more sophisticated implementation, this would measure the change in
#   dual variables (prices) between iterations
#
# Convergence Criterion:
#   Both residuals must be below epsilon threshold for convergence
#   This ensures both market balance (primal) and solution stability (dual)
#
# Arguments:
#   - elec::Dict: Electricity market dictionary (read-only)
#   - h2::Dict: Hydrogen market dictionary (read-only)
#   - elec_gc::Dict: Electricity GC market dictionary (read-only)
#   - h2_gc::Dict: Hydrogen GC market dictionary (read-only)
#   - ep::Dict: End Product market dictionary (read-only)
#   - data::Dict: Configuration dictionary (read-only, used for dimensions)
#
# Returns:
#   - primal_res::Float64: Maximum absolute imbalance across all markets
#   - dual_res::Float64: Dual residual (currently equal to primal)
#   - diagnostics::Dict: Dictionary containing detailed information about the maximum imbalance:
#       - "market": String name of the market with maximum imbalance
#       - "year": Year/scenario with maximum imbalance
#       - "time": Time step with maximum imbalance
#       - "repr_day": Representative day with maximum imbalance
#       - "imbalance": The actual imbalance value (can be positive or negative)
#       - "abs_imbalance": Absolute value of the imbalance
# ==============================================================================
function calculate_residuals(elec, h2, elec_gc, h2_gc, ep, data)
    # Initialize primal residual tracking
    # Start at 0.0, will be updated to the maximum imbalance found
    p_res = 0.0
    
    # Initialize diagnostic tracking variables
    # These will store information about where the maximum imbalance occurs
    max_market = "Unknown"
    max_year = nothing
    max_time = 0
    max_repr_day = 0
    max_imbalance = 0.0
    
    # Iterate through years to find the maximum violation across all markets
    for y in keys(elec["balance"])
        # Check max imbalance in Electricity Market
        # findmax(...) finds both the maximum value and its index (CartesianIndex)
        # This allows us to track exactly where the maximum imbalance occurs
        elec_max, elec_idx = findmax(abs.(elec["balance"][y]))
        if elec_max > p_res
            p_res = elec_max
            max_market = "Electricity"
            max_year = y
            max_time, max_repr_day = elec_idx[1], elec_idx[2]
            max_imbalance = elec["balance"][y][elec_idx]
        end
        
        # Check max imbalance in Electricity GC Market
        # Same logic: find maximum absolute imbalance in GC market
        elec_gc_max, elec_gc_idx = findmax(abs.(elec_gc["balance"][y]))
        if elec_gc_max > p_res
            p_res = elec_gc_max
            max_market = "Electricity_GC"
            max_year = y
            max_time, max_repr_day = elec_gc_idx[1], elec_gc_idx[2]
            max_imbalance = elec_gc["balance"][y][elec_gc_idx]
        end
        
        # Check max imbalance in Hydrogen Market
        # Find maximum absolute imbalance in H2 market
        h2_max, h2_idx = findmax(abs.(h2["balance"][y]))
        if h2_max > p_res
            p_res = h2_max
            max_market = "Hydrogen"
            max_year = y
            max_time, max_repr_day = h2_idx[1], h2_idx[2]
            max_imbalance = h2["balance"][y][h2_idx]
        end
        
        # Check max imbalance in Hydrogen GC Market
        # Find maximum absolute imbalance in H2 GC market
        h2_gc_max, h2_gc_idx = findmax(abs.(h2_gc["balance"][y]))
        if h2_gc_max > p_res
            p_res = h2_gc_max
            max_market = "Hydrogen_GC"
            max_year = y
            max_time, max_repr_day = h2_gc_idx[1], h2_gc_idx[2]
            max_imbalance = h2_gc["balance"][y][h2_gc_idx]
        end
        
        # Check max imbalance in End Product Market (Supply vs Fixed Demand)
        # Find maximum absolute imbalance in EP market
        # This checks if total supply matches the fixed demand
        ep_max, ep_idx = findmax(abs.(ep["balance"][y]))
        if ep_max > p_res
            p_res = ep_max
            max_market = "End_Product"
            max_year = y
            max_time, max_repr_day = ep_idx[1], ep_idx[2]
            max_imbalance = ep["balance"][y][ep_idx]
        end
    end
    
    # Set Dual Residual equal to Primal Residual (Simplified checking)
    # In a more sophisticated implementation, dual residual would measure:
    #   - Change in dual variables (prices) between iterations
    #   - Stationarity of the solution
    # For now, we use this simplified approach where both residuals must be small
    d_res = p_res
    
    # Create diagnostics dictionary with detailed information about the maximum imbalance
    # This helps identify which market and time period is causing convergence issues
    diagnostics = Dict(
        "market" => max_market,
        "year" => max_year,
        "time" => max_time,
        "repr_day" => max_repr_day,
        "imbalance" => max_imbalance,
        "abs_imbalance" => p_res
    )
    
    # Return both residuals and diagnostics
    # These will be compared against epsilon threshold in the main ADMM loop
    return p_res, d_res, diagnostics
end

# ==============================================================================
# UPDATE REFERENCES (PLACEHOLDER)
# ==============================================================================
# Placeholder function for updating reference quantities (q_bar) in standard ADMM.
# In this specific ADMM implementation (Exchange ADMM on net balance), 
# references are handled implicitly via the balance updates.
#
# In Exchange ADMM:
#   - References are always set to zero
#   - The penalty term becomes: (rho/2) * variable^2
#   - This is simpler than standard ADMM where references track consensus values
#
# This function is kept for interface consistency but does nothing.
#
# Arguments:
#   - agents::Dict: Dictionary containing lists of agent IDs (not used)
#   - mdict::Dict: Dictionary of JuMP models (not used)
#   - elec::Dict: Electricity market dictionary (not used)
#   - h2::Dict: Hydrogen market dictionary (not used)
#   - elec_gc::Dict: Electricity GC market dictionary (not used)
#   - h2_gc::Dict: Hydrogen GC market dictionary (not used)
#   - ep::Dict: End Product market dictionary (not used)
#
# Returns:
#   - Nothing (function is a placeholder)
# ==============================================================================
function update_references!(agents, mdict, elec, h2, elec_gc, h2_gc, ep)
    return
end
