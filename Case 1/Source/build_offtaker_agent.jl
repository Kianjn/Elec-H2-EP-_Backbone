# ==============================================================================
# BUILD OFFTAKER AGENT MODEL (AMMONIA PRODUCERS)
# ==============================================================================
# This function constructs the JuMP optimization model for Offtaker agents.
# These agents produce End Products (e.g., Ammonia) and compete to meet fixed demand.
# It handles two types of agents:
#   1. Electrolytic Ammonia Producer (Green): Buys H2 + H2 GCs to produce EP
#   2. Grey Ammonia Producer: Produces EP via conventional means (SMR), must buy H2 GCs
#
# Both types compete in the End Product market to meet a fixed demand profile.
# Both are subject to a 42% H2 GC mandate policy requirement (applied on an annual basis).
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Offtaker_Green")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - EP_market::Dict: Dictionary containing End Product market parameters
#   - H2_market::Dict: Dictionary containing hydrogen market parameters
#   - H2_GC_market::Dict: Dictionary containing hydrogen GC market parameters
#
# Returns:
#   - Modifies the JuMP model in-place by adding variables, constraints, and objective
# ==============================================================================
function build_offtaker_agent!(agent_id::String, model::JuMP.Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict)

    # --- 1. EXTRACT SETS AND COMMON PARAMETERS ---
    # These sets define the dimensions of the optimization problem
    T = model[:T]  # Time steps (e.g., 1:24)
    R = model[:R]  # Representative days (e.g., 1:3)
    Y = model[:Y]  # Years/scenarios (e.g., [2021])
    W = model[:W]  # Representative day weights (probabilities/frequencies)

    # --- 2. DEFINE ADMM PARAMETERS ---
    # These parameters are placeholders that will be updated iteratively by the ADMM solver.

    # -- End Product (EP) Coordination Parameters --
    # These are critical for the competition between Green and Grey agents.
    # The EP market uses a fixed demand profile, and agents compete to supply it.
    # Initialize the shadow price (lambda_EP) for the End Product market
    # This represents the dual variable (Lagrange multiplier) for EP market clearing
    model[:lambda_EP] = Dict(y => EP_market["price"][y] for y in Y)
    # Initialize the penalty factor (rho) for the EP market
    model[:rho_EP] = EP_market["rho"]
    # Initialize the reference quantity for EP (ADMM tracking)
    # Exchange ADMM: references are zero
    model[:EP_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # -- Hydrogen Market Parameters --
    # Only strictly needed for Green Offtaker, but initialized for all for safety
    # Initialize H2 prices (cost for Green Offtaker when buying H2)
    model[:lambda_H] = Dict(y => H2_market["price"][y] for y in Y)
    # Initialize H2 penalty factor
    model[:rho_H] = H2_market["rho"]
    # Initialize H2 reference quantity
    model[:H_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # -- Hydrogen Green Certificate (GC) Market Parameters --
    # Both Green and Grey offtakers need H2 GCs (for policy mandate)
    # Initialize H2 GC prices (cost when buying H2 GCs)
    model[:lambda_GC_H] = Dict(y => H2_GC_market["price"][y] for y in Y)
    # Initialize H2 GC penalty factor
    model[:rho_GC_H] = H2_GC_market["rho"]
    # Initialize H2 GC reference quantity
    model[:GC_H_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # --- 3. BUILD VARIABLES AND OBJECTIVES BASED ON AGENT TYPE ---

    # ==========================================================================
    # CASE A: ELECTROLYTIC AMMONIA PRODUCER (GREEN OFFTAKER)
    # ==========================================================================
    # Identified by the presence of a conversion factor (:alpha)
    # Process: Buys H2 -> Converts to End Product (Ammonia) -> Sells EP
    # This agent uses green hydrogen to produce green ammonia
    # ==========================================================================
    if haskey(model, :alpha)
        # -- DECISION VARIABLES --
        # All variables are non-negative and indexed by time, representative day, and year
        
        # h_buy: Hydrogen purchased from the hydrogen market (MWh)
        # This is the primary feedstock for ammonia production
        @variable(model, 0 <= h_buy[t=T, r=R, y=Y])
        
        # gc_h_buy: Hydrogen Green Certificates purchased (Certificates)
        # Required to certify the ammonia as "green" and meet policy mandate
        @variable(model, 0 <= gc_h_buy[t=T, r=R, y=Y])
        
        # ep_sell: End Product (Ammonia) sold to the market (MWh)
        # This is the primary output that competes with grey ammonia
        @variable(model, 0 <= ep_sell[t=T, r=R, y=Y])

        # -- OPERATIONAL CONSTRAINTS --
        
        # Constraint 1: Input Capacity Limit
        # Limit hydrogen purchase to maximum pipeline/infrastructure capacity
        # H_buy_bar: Maximum hydrogen intake capacity (MW)
        @constraint(model, [t=T, r=R, y=Y], h_buy[t,r,y] <= model[:H_buy_bar])
        
        # Constraint 2: Output Capacity Limit
        # Limit ammonia production to factory production capacity
        # EP_sell_bar: Maximum End Product production capacity (MW)
        @constraint(model, [t=T, r=R, y=Y], ep_sell[t,r,y] <= model[:EP_sell_bar])
        
        # Constraint 3: Production Efficiency / Material Balance
        # Output (EP) is limited by Input (H2) multiplied by conversion factor alpha
        # ep_sell <= alpha * h_buy
        # alpha: Stoichiometric efficiency (yield of NH3 per unit of H2)
        # Example: If alpha = 1.0, then 1 MWh H2 produces 1 MWh NH3
        @constraint(model, [t=T, r=R, y=Y], ep_sell[t,r,y] <= model[:alpha] * h_buy[t,r,y])
        
        # Constraint 4: Policy Requirement - 42% H2 GC Mandate (YEARLY CONSTRAINT)
        # At least 42% of annual hydrogen consumed must be green-certified
        # This is a regulatory requirement applied on an annual basis (not hourly)
        # Yearly constraint: Sum(weighted GC purchases) >= 0.42 * Sum(weighted H2 purchases)
        # This allows flexibility in when GCs are purchased during the year
        # More realistic than hourly matching, as GC systems typically track annual compliance
        @constraint(model, policy_mandate_yearly[y=Y], 
            sum(W[y][r] * gc_h_buy[t,r,y] for t in T, r in R) >= 
            0.42 * sum(W[y][r] * h_buy[t,r,y] for t in T, r in R))

        # -- OBJECTIVE FUNCTION --
        # Maximize: Revenue (EP) - Costs (H2 + GCs + Processing) - ADMM Penalties
        @expression(model, obj_green,
            sum(W[y][r] * (
                # (+) Revenue from End Product (Shadow Price)
                # Market price of ammonia times quantity sold
                (model[:lambda_EP][y][t,r] * ep_sell[t,r,y]) -
                # (-) Cost of Hydrogen Input
                # Market price of hydrogen times quantity purchased
                (model[:lambda_H][y][t,r] * h_buy[t,r,y]) -
                # (-) Cost of Green Certificates
                # Price premium for green-certified hydrogen
                (model[:lambda_GC_H][y][t,r] * gc_h_buy[t,r,y]) -
                # (-) Variable Processing Cost
                # Cost of converting hydrogen to ammonia (non-fuel variable costs)
                # C_proc: Processing cost per unit of End Product (EUR/MWh_EP)
                (model[:C_proc] * ep_sell[t,r,y]) -
                # (-) ADMM Penalties for coupled variables (EP, H2, GC)
                # These penalties enforce consensus with market clearing conditions
                # Penalty = (rho/2) * variable^2 (since references are zero in Exchange ADMM)
                (model[:rho_EP] / 2 * (ep_sell[t,r,y] - model[:EP_ref][y][t,r])^2) -
                (model[:rho_H] / 2 * (h_buy[t,r,y] - model[:H_ref][y][t,r])^2) -
                (model[:rho_GC_H] / 2 * (gc_h_buy[t,r,y] - model[:GC_H_ref][y][t,r])^2)
            ) for t in T, r in R, y in Y)
        )
        # Set the maximization objective
        @objective(model, Max, obj_green)

        # Create alias for ep_sell (used in market balancing)
        model[:ep_sell] = ep_sell

    # ==========================================================================
    # CASE B: GREY AMMONIA PRODUCER
    # ==========================================================================
    # Identified by presence of EP Capacity (:EP_sell_bar) but NO alpha
    # Process: Produces EP via conventional means (e.g., SMR -> Ammonia)
    # This agent produces ammonia using natural gas (SMR process)
    # Now must buy H2 GCs to meet 42% mandate based on internal H2 consumption
    # ==========================================================================
    elseif !haskey(model, :alpha) && haskey(model, :EP_sell_bar)
        
        # -- DECISION VARIABLES --
        
        # ep_sell: End Product (Ammonia) sold to the market (MWh)
        # This competes with green ammonia to meet fixed demand
        @variable(model, 0 <= ep_sell[t=T, r=R, y=Y])
        
        # gc_h_buy_G: H2 Green Certificates bought (to meet policy mandate)
        # Grey ammonia producers must buy H2 GCs even though they don't buy physical H2
        # This is based on their internal H2 consumption equivalent
        @variable(model, 0 <= gc_h_buy_G[t=T, r=R, y=Y])

        # -- OPERATIONAL CONSTRAINTS --
        
        # Constraint 1: Output Capacity Limit
        # Limit ammonia production to factory production capacity
        # EP_sell_bar: Maximum End Product production capacity (MW)
        @constraint(model, [t=T, r=R, y=Y], ep_sell[t,r,y] <= model[:EP_sell_bar])
        
        # Constraint 2: Policy Requirement - 42% H2 GC Mandate (YEARLY CONSTRAINT)
        # Based on internal H2 consumption equivalent (gamma_NH3 * ep_sell)
        # Grey ammonia production requires H2 internally (via SMR from natural gas)
        # gamma_NH3: Specific hydrogen intensity of grey ammonia production (kg_H2/kg_NH3)
        # Typical value: ~0.18 kg H2 per kg NH3 for SMR-based production
        # Yearly constraint: Sum(weighted GC purchases) >= 0.42 * Sum(weighted internal H2 equivalent)
        # The mandate requires: Sum(GC) >= 0.42 * Sum(gamma_NH3 * EP)
        # This ensures that 42% of the annual internal H2 consumption is green-certified
        # More realistic than hourly matching, as GC systems typically track annual compliance
        @constraint(model, policy_mandate_grey_yearly[y=Y], 
            sum(W[y][r] * gc_h_buy_G[t,r,y] for t in T, r in R) >= 
            0.42 * sum(W[y][r] * (model[:gamma_NH3] * ep_sell[t,r,y]) for t in T, r in R))

        # -- OBJECTIVE FUNCTION --
        # Maximize: Revenue (EP) - Costs (Marginal + GC Purchase) - ADMM Penalties
        @expression(model, obj_grey,
            sum(W[y][r] * (
                # (+) Revenue from End Product
                # Market price of ammonia times quantity sold
                (model[:lambda_EP][y][t,r] * ep_sell[t,r,y]) -
                # (-) Marginal Cost of Production
                # Cost of producing ammonia via SMR (includes natural gas + O&M)
                # C_proc: Marginal cost per unit of End Product (EUR/MWh_EP)
                (model[:C_proc] * ep_sell[t,r,y]) -
                # (-) Cost of H2 Green Certificates
                # Price premium for green-certified hydrogen (required by policy)
                # Grey producers must buy GCs even though they don't buy physical H2
                (model[:lambda_GC_H][y][t,r] * gc_h_buy_G[t,r,y]) -
                # (-) ADMM Penalties for EP and GC purchases
                # These penalties enforce consensus with market clearing conditions
                (model[:rho_EP] / 2 * (ep_sell[t,r,y] - model[:EP_ref][y][t,r])^2) -
                (model[:rho_GC_H] / 2 * (gc_h_buy_G[t,r,y] - model[:GC_H_ref][y][t,r])^2)
            ) for t in T, r in R, y in Y)
        )
        # Set the maximization objective
        @objective(model, Max, obj_grey)

        # Create aliases for generic access in ADMM updates and market balancing
        model[:ep_sell] = ep_sell
        model[:gc_h_buy_G] = gc_h_buy_G
    end

    # Print confirmation message for debugging and progress tracking
    println("Built Offtaker Agent: $agent_id")
end
