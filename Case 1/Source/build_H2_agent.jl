# ==============================================================================
# BUILD HYDROGEN AGENT MODEL
# ==============================================================================
# This function constructs the JuMP optimization model for Hydrogen Sector agents.
# It handles two types of agents:
#   1. Electrolytic Hydrogen Producer: Uses electricity to produce green hydrogen
#   2. Hydrogen Consumer: Consumes hydrogen with elastic demand
#
# The function defines decision variables, operational constraints, and objective
# functions according to the mathematical formulation in the LaTeX document.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Prod_H2_Green")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - H2_market::Dict: Dictionary containing hydrogen market parameters (prices, balances)
#   - H2_GC_market::Dict: Dictionary containing hydrogen green certificate market parameters
#
# Returns:
#   - Modifies the JuMP model in-place by adding variables, constraints, and objective
# ==============================================================================
function build_H2_agent!(agent_id::String, model::JuMP.Model, H2_market::Dict, H2_GC_market::Dict)

    # --- 1. EXTRACT SETS AND COMMON PARAMETERS ---
    # These sets define the dimensions of the optimization problem
    # T: Time steps within a representative day (e.g., 1:24 for hourly resolution)
    T = model[:T]
    # R: Representative days (e.g., 1:3 for 3 representative days)
    R = model[:R]
    # Y: Years/scenarios (e.g., [2021] for a single year)
    Y = model[:Y]
    # W: Representative day weights (probability/frequency of each representative day)
    W = model[:W]

    # --- 2. DEFINE ADMM PARAMETERS ---
    # These parameters are placeholders that will be updated iteratively by the ADMM solver.
    # They represent market prices (dual variables) and penalty parameters for the
    # augmented Lagrangian formulation.

    # -- Electricity Market Parameters (Input for Electrolytic Producer) --
    # Initialize electricity prices (lambda_E) to zero matrices
    # These will be updated in solve_H2_agent! with current market prices
    model[:lambda_E] = Dict(y => zeros(length(T), length(R)) for y in Y)
    # Initialize the ADMM penalty factor (rho) for electricity market coupling
    # This penalizes deviations from market consensus in the augmented Lagrangian
    model[:rho_E] = model[:rho]
    # Initialize reference quantities for ADMM tracking (Exchange ADMM formulation)
    # These are set to zero in the Exchange ADMM variant used here
    model[:E_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)
    
    # -- Electricity Green Certificate (GC) Market Parameters (Input) --
    # Initialize Elec GC prices to zero (will be updated by ADMM solver)
    model[:lambda_GC_E] = Dict(y => zeros(length(T), length(R)) for y in Y)
    # Initialize the penalty factor for Elec GC market coupling
    model[:rho_GC_E] = model[:rho]
    # Initialize reference quantities for Elec GC ADMM tracking
    model[:GC_E_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # -- Hydrogen Market Parameters (Output for Producers, Input for Consumers) --
    # Initialize H2 prices using the global market initialization values
    # These represent the dual variables (Lagrange multipliers) for hydrogen market clearing
    model[:lambda_H] = Dict(y => H2_market["price"][y] for y in Y)
    # Initialize the penalty factor for Hydrogen market coupling
    model[:rho_H] = model[:rho]
    # Initialize reference quantities for Hydrogen ADMM tracking
    model[:H_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # -- Hydrogen Green Certificate (GC) Market Parameters (Output) --
    # Initialize H2 GC prices from global market
    # These represent the price premium for green-certified hydrogen
    model[:lambda_GC_H] = Dict(y => H2_GC_market["price"][y] for y in Y)
    # Initialize the penalty factor for H2 GC market coupling
    model[:rho_GC_H] = model[:rho]
    # Initialize reference quantities for H2 GC ADMM tracking
    model[:GC_H_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)


    # --- 3. BUILD VARIABLES AND OBJECTIVES BASED ON AGENT TYPE ---
    # The agent type is determined by checking which parameters exist in the model
    # This allows flexible agent configuration without explicit type flags

    # ==========================================================================
    # CASE A: ELECTROLYTIC HYDROGEN PRODUCER
    # ==========================================================================
    # Identified by presence of Electrolyzer Capacity (:Capacity_Electrolyzer) 
    # or generic Input/Output Capacities (:E_bar and :H_bar)
    # This agent converts electricity (and Elec GCs) into hydrogen (and H2 GCs)
    # ==========================================================================
    if haskey(model, :Capacity_Electrolyzer) || (haskey(model, :E_bar) && haskey(model, :H_bar))
        
        # -- DECISION VARIABLES --
        # All variables are non-negative and indexed by time (t), representative day (r), and year (y)
        
        # e_buy: Electricity purchased from the electricity market (MWh)
        # This is the primary input for electrolysis
        @variable(model, 0 <= e_buy[t=T, r=R, y=Y])       
        
        # gc_e_buy: Electricity Green Certificates purchased (Certificates)
        # Required to certify the hydrogen as "green" (renewable-based)
        @variable(model, 0 <= gc_e_buy[t=T, r=R, y=Y])    
        
        # h_sell: Hydrogen sold to the hydrogen market (MWh)
        # This is the primary output from electrolysis
        @variable(model, 0 <= h_sell[t=T, r=R, y=Y])      
        
        # gc_h_sell: Hydrogen Green Certificates sold (Certificates)
        # These certify that the hydrogen was produced using renewable electricity
        @variable(model, 0 <= gc_h_sell[t=T, r=R, y=Y])   

        # -- OPERATIONAL CONSTRAINTS --
        
        # Constraint 1: Capacity Limits
        # Restrict electricity input to installed electrolyzer capacity
        # E_bar: Maximum electricity consumption capacity (MW)
        @constraint(model, [t=T, r=R, y=Y], e_buy[t,r,y] <= model[:E_bar])
        # Restrict hydrogen output to installed production capacity
        # H_bar: Maximum hydrogen production capacity (MW)
        @constraint(model, [t=T, r=R, y=Y], h_sell[t,r,y] <= model[:H_bar])
        
        # Constraint 2: Production Efficiency / Energy Balance
        # Energy Balance: Electricity Input >= Specific Consumption * Hydrogen Output
        # This enforces the physical relationship: e_buy >= e_H * h_sell
        # where e_H is the energy intensity (MWh_electricity per MWh_hydrogen)
        # Example: If e_H = 1.5, then 1.5 MWh of electricity is needed per 1 MWh of H2
        # This constraint ensures sufficient electricity input for the hydrogen output
        @constraint(model, physics[t=T, r=R, y=Y], e_buy[t,r,y] >= model[:e_H] * h_sell[t,r,y])

        # Constraint 3: Green Certification Balance
        # You cannot sell more Green H2 Certificates than the actual H2 you produce
        # This ensures that GC sales are physically backed by hydrogen production
        # gc_h_sell <= h_sell (at most 1 GC per unit of H2)
        @constraint(model, cert_limit[t=T, r=R, y=Y], gc_h_sell[t,r,y] <= h_sell[t,r,y])
        
        # Constraint 4: Green Backing Constraint (ANNUAL CONSTRAINT from LaTeX formulation)
        # H2 Green Certificates must be backed by Electricity Green Certificates on an annual basis
        # This ensures the green certificate chain: Elec GCs -> H2 GCs
        # Annual constraint: Sum(weighted Elec GC purchases) >= e_H * Sum(weighted H2 GC sales)
        # Interpretation: Over the year, for each unit of H2 GC sold, you need e_H units of Elec GCs
        # This accounts for the energy conversion efficiency and allows flexibility in timing
        # More realistic than hourly matching, as GC systems typically track annual compliance
        @constraint(model, green_backing_yearly[y=Y], 
            sum(W[y][r] * gc_e_buy[t,r,y] for t in T, r in R) >= 
            model[:e_H] * sum(W[y][r] * gc_h_sell[t,r,y] for t in T, r in R))

        # -- OBJECTIVE FUNCTION --
        # Maximize: Revenue (H2 + H2_GC) - Cost (Elec + Elec_GC + Operations) - ADMM Penalties
        # The objective is weighted by representative day weights (W) to account for
        # the frequency/probability of each representative day
        @expression(model, obj_green,
            sum(W[y][r] * (
                # (+) Revenue from selling Hydrogen
                # Price per MWh of hydrogen times quantity sold
                (model[:lambda_H][y][t,r] * h_sell[t,r,y]) +
                # (+) Revenue from selling Hydrogen Green Certificates
                # Price premium for green-certified hydrogen
                (model[:lambda_GC_H][y][t,r] * gc_h_sell[t,r,y]) -
                # (-) Cost of buying Electricity
                # Market price of electricity times quantity purchased
                (model[:lambda_E][y][t,r] * e_buy[t,r,y]) -
                # (-) Cost of buying Electricity Green Certificates
                # Market price of Elec GCs times quantity purchased
                (model[:lambda_GC_E][y][t,r] * gc_e_buy[t,r,y]) -
                # (-) Operational Cost (Variable O&M) - Linear cost function based on electricity input
                # Note: LaTeX specifies C_H(e) = c_{H,0} * e + (1/2) * c_{H,1} * e^2
                # For linear version: C_H(e) = c_{H,0} * e_buy
                # This represents variable operational and maintenance costs proportional to electricity input
                (model[:C_H] * e_buy[t,r,y]) - 
                # (-) ADMM Penalties for all 4 coupled variables (Standard quadratic form)
                # These penalties enforce consensus with market clearing conditions
                # Penalty = (rho/2) * (variable - reference)^2
                # For Exchange ADMM, references are zero, so penalty = (rho/2) * variable^2
                (model[:rho_H] / 2 * (h_sell[t,r,y] - model[:H_ref][y][t,r])^2) -
                (model[:rho_GC_H] / 2 * (gc_h_sell[t,r,y] - model[:GC_H_ref][y][t,r])^2) -
                (model[:rho_E] / 2 * (e_buy[t,r,y] - model[:E_ref][y][t,r])^2) -
                (model[:rho_GC_E] / 2 * (gc_e_buy[t,r,y] - model[:GC_E_ref][y][t,r])^2)
            ) for t in T, r in R, y in Y)
        )
        # Set the maximization objective for the optimization model
        @objective(model, Max, obj_green)

        # Create aliases for generic access in ADMM updates and market balancing
        # These allow other functions to access variables by name without knowing the exact variable object
        model[:h_sell] = h_sell
        model[:e_buy] = e_buy
        model[:gc_e_buy] = gc_e_buy
        model[:gc_h_sell] = gc_h_sell

    # ==========================================================================
    # CASE B: HYDROGEN CONSUMER
    # ==========================================================================
    # Identified by presence of H2 Demand Profile (:D_H_bar)
    # This agent consumes hydrogen with elastic demand (utility-based)
    # ==========================================================================
    elseif haskey(model, :D_H_bar)
        
        # -- DECISION VARIABLES --
        # d_H: Hydrogen consumption (MWh)
        # The agent chooses how much hydrogen to consume up to the demand limit
        @variable(model, 0 <= d_H[t=T, r=R, y=Y])

        # -- OPERATIONAL CONSTRAINTS --
        # Limit consumption to the demand profile
        # D_H_bar: Maximum hydrogen demand at each time step (from load profile)
        # This represents physical/contractual demand limits
        @constraint(model, [t=T, r=R, y=Y], d_H[t,r,y] <= model[:D_H_bar][y][t,r])

        # -- UTILITY PARAMETER --
        # Safe Parameter Access: Check if utility parameter exists, default to 0.0
        # Utility: Willingness to pay per unit of hydrogen consumed (EUR/MWh)
        # This represents the consumer's valuation of hydrogen
        utility_val = haskey(model, :Utility) ? model[:Utility] : 0.0

        # -- OBJECTIVE FUNCTION --
        # Maximize: Utility - Cost - ADMM Penalty
        # The consumer maximizes their surplus (utility minus cost)
        @expression(model, obj_cons,
            sum(W[y][r] * (
                # (+) Utility from consuming hydrogen
                # Linear utility function: Utility = utility_val * quantity
                (utility_val * d_H[t,r,y]) -
                # (-) Cost of buying Hydrogen
                # Market price times quantity consumed
                (model[:lambda_H][y][t,r] * d_H[t,r,y]) -
                # (-) ADMM Penalty for hydrogen consumption
                # Penalizes deviations from market consensus
                (model[:rho_H] / 2 * (d_H[t,r,y] - model[:H_ref][y][t,r])^2)
            ) for t in T, r in R, y in Y)
        )
        # Set the maximization objective
        @objective(model, Max, obj_cons)
        
        # Create alias for generic access in ADMM updates
        model[:d_H] = d_H
    end

    # Print confirmation message for debugging and progress tracking
    println("Built Hydrogen Agent: $agent_id")
end
