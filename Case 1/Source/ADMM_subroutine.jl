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
        
        # DEBUG: Initialize tracking variables for Electricity market (hourly matrix)
        # Track supply and demand separately for diagnostics
        if !haskey(elec, "debug_supply")
            elec["debug_supply"] = Dict()
            elec["debug_demand"] = Dict()
        end
        if !haskey(elec["debug_supply"], y)
            elec["debug_supply"][y] = zeros(size(elec["balance"][y]))
            elec["debug_demand"][y] = zeros(size(elec["balance"][y]))
        else
            elec["debug_supply"][y] .= 0.0
            elec["debug_demand"][y] .= 0.0
        end
        
        # Reset Hydrogen Market Balance to 0.0
        # This will accumulate: H2 Production - H2 Consumption
        h2["balance"][y] .= 0.0
        
        # DEBUG: Initialize tracking variables for Hydrogen market (hourly matrix)
        if !haskey(h2, "debug_supply")
            h2["debug_supply"] = Dict()
            h2["debug_demand"] = Dict()
        end
        if !haskey(h2["debug_supply"], y)
            h2["debug_supply"][y] = zeros(size(h2["balance"][y]))
            h2["debug_demand"][y] = zeros(size(h2["balance"][y]))
        else
            h2["debug_supply"][y] .= 0.0
            h2["debug_demand"][y] .= 0.0
        end
        
        # Reset Electricity GC Market Balance to 0.0
        # This will accumulate: VRES GC Production - GC Consumption - GC Demand Agent purchases
        elec_gc["balance"][y] .= 0.0
        
        # DEBUG: Initialize tracking variables for Electricity GC market (hourly matrix)
        if !haskey(elec_gc, "debug_supply")
            elec_gc["debug_supply"] = Dict()
            elec_gc["debug_demand"] = Dict()
        end
        if !haskey(elec_gc["debug_supply"], y)
            elec_gc["debug_supply"][y] = zeros(size(elec_gc["balance"][y]))
            elec_gc["debug_demand"][y] = zeros(size(elec_gc["balance"][y]))
        else
            elec_gc["debug_supply"][y] .= 0.0
            elec_gc["debug_demand"][y] .= 0.0
        end
        
        # Reset Hydrogen GC Market Balance to 0.0.
        # This will accumulate annual H2 GC Production - H2 GC Consumption (green + grey offtakers).
        # The H2 GC market is modeled with an annual scalar balance; GC decisions are hourly but aggregated.
        h2_gc["balance"][y] = 0.0
        
        # DEBUG: Initialize tracking variables for H2 GC market (annual aggregation)
        # These will help diagnose aggregation issues
        if !haskey(h2_gc, "debug_supply")
            h2_gc["debug_supply"] = Dict()
            h2_gc["debug_demand"] = Dict()
        end
        h2_gc["debug_supply"][y] = 0.0
        h2_gc["debug_demand"][y] = 0.0
        
        # Reset End Product Balance
        # For inelastic (fixed) EP demand, we start from negative demand and add:
        #   + Supply from ammonia producers and importer
        # Fixed demand is exogenous and does not respond directly to price.
        # At equilibrium: Supply - Demand = 0, so balance = 0
        ep["balance"][y] .= -ep["demand"][y]
        
        # DEBUG: Initialize tracking variables for End Product market (hourly matrix)
        # Track supply and fixed demand separately for diagnostics
        if !haskey(ep, "debug_supply")
            ep["debug_supply"] = Dict()
            ep["debug_demand"] = Dict()
        end
        if !haskey(ep["debug_supply"], y)
            ep["debug_supply"][y] = zeros(size(ep["balance"][y]))
            ep["debug_demand"][y] = copy(ep["demand"][y])
        else
            ep["debug_supply"][y] .= 0.0
            ep["debug_demand"][y] .= ep["demand"][y]
        end
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
                 demand_values = Array(value.(m[:d_E][:,:,y]))
                 elec["balance"][y] .-= demand_values
                 # DEBUG: Track demand contribution
                 elec["debug_demand"][y] .+= demand_values
            # Check if agent is a Generator (has generation variable q_E, but NOT d_E)
            # Generators include: VRES (solar, wind) and Conventional (gas, coal)
            elseif haskey(m, :q_E)
                 # Add generation to Electricity Balance (Supply)
                 # value.(m[:q_E][:,:,y]) extracts the optimal generation values
                 # This is a JuMP DenseAxisArray, so we convert to standard Array to avoid broadcasting errors
                 # The .+= operator adds the generation matrix element-wise to the balance
                 gen_values = Array(value.(m[:q_E][:,:,y]))
                 elec["balance"][y] .+= gen_values
                 # DEBUG: Track supply contribution
                 elec["debug_supply"][y] .+= gen_values
                 
                 # If VRES, they also produce Green Certificates
                 # VRES generators produce 1 GC per 1 MWh of electricity (1:1 ratio)
                 # Check if the model has gc_E variable (VRES indicator)
                 if haskey(m, :gc_E)
                     # Add GCs to Electricity GC Balance (Supply)
                     # For VRES: gc_E = q_E (1 MWh electricity = 1 GC)
                     gc_values = Array(value.(m[:gc_E][:,:,y]))
                     elec_gc["balance"][y] .+= gc_values
                     # DEBUG: Track supply contribution
                     elec_gc["debug_supply"][y] .+= gc_values
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
                e_buy_values = Array(value.(m[:e_buy][:,:,y]))
                elec["balance"][y] .-= e_buy_values
                # DEBUG: Track demand contribution
                elec["debug_demand"][y] .+= e_buy_values
                
                # Subtract Elec GC consumption (Demand)
                # Electrolytic H2 producers buy Elec GCs to certify their hydrogen as green
                # Required by the green backing constraint
                gc_e_buy_values = Array(value.(m[:gc_e_buy][:,:,y]))
                elec_gc["balance"][y] .-= gc_e_buy_values
                # DEBUG: Track demand contribution
                elec_gc["debug_demand"][y] .+= gc_e_buy_values
                
                # Add Hydrogen production (Supply)
                # Electrolytic H2 producers sell hydrogen to the market
                h_sell_values = Array(value.(m[:h_sell][:,:,y]))
                h2["balance"][y] .+= h_sell_values
                # DEBUG: Track supply contribution
                h2["debug_supply"][y] .+= h_sell_values
                
                # Add Hydrogen GC production (Supply), aggregated annually across hours and representative days.
                # Electrolytic H2 producers sell H2 GCs when using green electricity.
                gc_h_sell_values = Array(value.(m[:gc_h_sell][:,:,y]))
                h_sell_values = Array(value.(m[:h_sell][:,:,y]))  # DEBUG: Also get H2 sales for comparison
                W = m[:W]  # Representative day weights
                T = m[:T]  # Time steps
                R = m[:R]  # Representative days
                annual_raw_supply = sum(W[y][r] * gc_h_sell_values[t, r] for t in T, r in R)
                annual_h2_sales = sum(W[y][r] * h_sell_values[t, r] for t in T, r in R)  # DEBUG: Weighted annual H2 sales
                h2_gc["balance"][y] += annual_raw_supply
                # DEBUG: Track GC supply from producer
                h2_gc["debug_supply"][y] += annual_raw_supply
                # DEBUG: Store H2 sales for comparison (to verify gc_h_sell <= h_sell constraint)
                if !haskey(h2_gc, "debug_h2_sales")
                    h2_gc["debug_h2_sales"] = Dict()
                end
                h2_gc["debug_h2_sales"][y] = annual_h2_sales
            # Note: Only Electrolytic Producers exist now (no conventional H2 producers)
            # Check if Hydrogen Consumer
            # Identified by presence of d_H variable (hydrogen demand)
            elseif haskey(m, :d_H)
                # Subtract Hydrogen consumption (Demand)
                # Hydrogen consumers reduce the H2 market balance
                d_H_values = Array(value.(m[:d_H][:,:,y]))
                h2["balance"][y] .-= d_H_values
                # DEBUG: Track demand contribution
                h2["debug_demand"][y] .+= d_H_values
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
                h_buy_values = Array(value.(m[:h_buy][:,:,y]))
                h2["balance"][y] .-= h_buy_values
                # DEBUG: Track demand contribution
                h2["debug_demand"][y] .+= h_buy_values
                
                # Subtract Hydrogen GC consumption (Demand), aggregated annually across hours and representative days.
                # Green offtakers buy H2 GCs to certify their ammonia as green
                # as required by the 42% policy mandate.
                gc_h_buy_values = Array(value.(m[:gc_h_buy][:,:,y]))
                W = m[:W]  # Representative day weights
                T = m[:T]  # Time steps
                R = m[:R]  # Representative days
                annual_demand_green = sum(W[y][r] * gc_h_buy_values[t, r] for t in T, r in R)
                h2_gc["balance"][y] -= annual_demand_green
                # DEBUG: Track demand contribution from green offtaker
                h2_gc["debug_demand"][y] += annual_demand_green
            end
            
            # Check if Grey Ammonia Producer (buys H2 GCs directly)
            # Identified by presence of gc_h_buy_G variable (H2 GC purchase for grey producer)
            # Grey producers don't buy physical H2 (they produce it internally via SMR)
            # But they must buy H2 GCs to meet the 42% policy mandate
            if haskey(m, :gc_h_buy_G)
                # Subtract Hydrogen GC consumption (Demand), aggregated annually across hours and representative days.
                # Grey producers buy H2 GCs based on their internal H2 consumption equivalent.
                gc_h_buy_G_values = Array(value.(m[:gc_h_buy_G][:,:,y]))
                W = m[:W]  # Representative day weights
                T = m[:T]  # Time steps
                R = m[:R]  # Representative days
                annual_demand_grey = sum(W[y][r] * gc_h_buy_G_values[t, r] for t in T, r in R)
                h2_gc["balance"][y] -= annual_demand_grey
                # DEBUG: Track demand contribution from grey offtaker
                h2_gc["debug_demand"][y] += annual_demand_grey
            end
            
            # Check if selling End Product (Both Green and Grey do this)
            # Identified by presence of ep_sell variable (End Product sales)
            if haskey(m, :ep_sell)
                # Add End Product production (Supply)
                # This fills the "hole" created by the fixed demand in step 1
                # Both green and grey ammonia producers contribute to meeting the fixed demand
                ep_sell_values = Array(value.(m[:ep_sell][:,:,y]))
                ep["balance"][y] .+= ep_sell_values
                # DEBUG: Track supply contribution
                ep["debug_supply"][y] .+= ep_sell_values
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
                    d_GC_E_values = Array(value.(m[:d_GC_E][:,:,y]))
                    elec_gc["balance"][y] .-= d_GC_E_values
                    # DEBUG: Track demand contribution
                    elec_gc["debug_demand"][y] .+= d_GC_E_values
                end
            end
        end
    end

    # --- 6. ADD CONTRIBUTIONS FROM EP DEMAND AGENTS (ELASTIC AMMONIA DEMAND) ---
    if haskey(agents, :EP_demand)
        for id in agents[:EP_demand]
            m = mdict[id]
            for y in keys(ep["balance"])
                if haskey(m, :d_EP)
                    d_EP_values = Array(value.(m[:d_EP][:,:,y]))
                    # Subtract EP demand from EP Balance
                    ep["balance"][y] .-= d_EP_values
                    # DEBUG: Track EP demand
                    ep["debug_demand"][y] .+= d_EP_values
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
        # Use a scaling factor to account for representative days (fraction of full-year hours).
        local elec_scale = haskey(elec, "scale") ? elec["scale"] : 1.0
        elec["price"][y] .-= (elec["rho"] * elec_scale) .* elec["balance"][y]
        # DEBUG: Track price history (store average price for hourly markets)
        if !haskey(elec, "price_history")
            elec["price_history"] = Dict()
            elec["balance_history"] = Dict()
        end
        if !haskey(elec["price_history"], y)
            elec["price_history"][y] = Float64[]
            elec["balance_history"][y] = Float64[]
        end
        push!(elec["price_history"][y], sum(elec["price"][y]) / length(elec["price"][y]))  # Average price across all hours
        push!(elec["balance_history"][y], sum(elec["balance"][y]))  # Total balance
        
        # Update Hydrogen Price
        # Same logic: price adjusts based on H2 market imbalance
        local h2_scale = haskey(h2, "scale") ? h2["scale"] : 1.0
        h2["price"][y] .-= (h2["rho"] * h2_scale) .* h2["balance"][y]
        # DEBUG: Track price history (store average price for hourly markets)
        if !haskey(h2, "price_history")
            h2["price_history"] = Dict()
            h2["balance_history"] = Dict()
        end
        if !haskey(h2["price_history"], y)
            h2["price_history"][y] = Float64[]
            h2["balance_history"][y] = Float64[]
        end
        push!(h2["price_history"][y], sum(h2["price"][y]) / length(h2["price"][y]))  # Average price across all hours
        push!(h2["balance_history"][y], sum(h2["balance"][y]))  # Total balance
        
        # Update Electricity GC Price
        # Price adjusts based on Elec GC market imbalance
        local elec_gc_scale = haskey(elec_gc, "scale") ? elec_gc["scale"] : 1.0
        elec_gc["price"][y] .-= (elec_gc["rho"] * elec_gc_scale) .* elec_gc["balance"][y]
        # DEBUG: Track price history (store average price for hourly markets)
        if !haskey(elec_gc, "price_history")
            elec_gc["price_history"] = Dict()
            elec_gc["balance_history"] = Dict()
        end
        if !haskey(elec_gc["price_history"], y)
            elec_gc["price_history"][y] = Float64[]
            elec_gc["balance_history"][y] = Float64[]
        end
        push!(elec_gc["price_history"][y], sum(elec_gc["price"][y]) / length(elec_gc["price"][y]))  # Average price across all hours
        push!(elec_gc["balance_history"][y], sum(elec_gc["balance"][y]))  # Total balance
        
        # Update Hydrogen GC Price.
        # H2 GC price is an annual scalar (one value per year) rather than an hourly matrix.
        # The same rho * scale * balance logic is used as in other markets.
        local h2_gc_scale = haskey(h2_gc, "scale") ? h2_gc["scale"] : 1.0
        h2_gc["price"][y] -= (h2_gc["rho"] * h2_gc_scale) * h2_gc["balance"][y]
        # DEBUG: Track price evolution
        if !haskey(h2_gc, "price_history")
            h2_gc["price_history"] = Dict()
            h2_gc["balance_history"] = Dict()
        end
        if !haskey(h2_gc["price_history"], y)
            h2_gc["price_history"][y] = Float64[]
            h2_gc["balance_history"][y] = Float64[]
        end
        push!(h2_gc["price_history"][y], h2_gc["price"][y])
        push!(h2_gc["balance_history"][y], h2_gc["balance"][y])
        
        # Update End Product Shadow Price
        # This coordinates the competition between green and grey ammonia producers
        # Price adjusts based on EP market imbalance (Supply - Fixed Demand)
        local ep_scale = haskey(ep, "scale") ? ep["scale"] : 1.0
        ep["price"][y] .-= (ep["rho"] * ep_scale) .* ep["balance"][y]
        # DEBUG: Track price history (store average price for hourly markets)
        if !haskey(ep, "price_history")
            ep["price_history"] = Dict()
            ep["balance_history"] = Dict()
        end
        if !haskey(ep["price_history"], y)
            ep["price_history"][y] = Float64[]
            ep["balance_history"][y] = Float64[]
        end
        push!(ep["price_history"][y], sum(ep["price"][y]) / length(ep["price"][y]))  # Average price across all hours
        push!(ep["balance_history"][y], sum(ep["balance"][y]))  # Total balance
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
        
        # Check max imbalance in Hydrogen GC Market.
        # The H2 GC balance is an annual scalar, so we simply use the absolute value of that scalar.
        h2_gc_abs = abs(h2_gc["balance"][y])
        if h2_gc_abs > p_res
            p_res = h2_gc_abs
            max_market = "Hydrogen_GC"
            max_year = y
            max_time = 0  # Annual market has no specific time step
            max_repr_day = 0  # Annual market has no specific representative day
            max_imbalance = h2_gc["balance"][y]
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
