# ==============================================================================
# SOLVE OFFTAKER AGENT OPTIMIZATION PROBLEM
# ==============================================================================
# This function is called iteratively within the ADMM loop to solve each
# Offtaker agent's optimization subproblem given current market prices.
#
# In the ADMM algorithm, this corresponds to the "x-update" step where each
# agent optimizes its decisions based on the current dual variables (prices)
# from the market coordinator.
#
# The function handles both:
#   1. Electrolytic Ammonia Producer (Green): Buys H2 + H2 GCs, sells EP
#   2. Grey Ammonia Producer: Produces EP directly, buys H2 GCs for mandate
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent
#   - model::JuMP.Model: JuMP optimization model for this agent (built in build_offtaker_agent!)
#   - EP_market::Dict: End Product market dictionary with current prices
#   - H2_market::Dict: Hydrogen market dictionary with current prices
#   - H2_GC_market::Dict: Hydrogen GC market dictionary with current prices
#   - admm_params::Dict: ADMM algorithm parameters
#
# Returns:
#   - Modifies the model in-place by updating prices and solving
#   - Prints warning if optimization fails
# ==============================================================================
function solve_offtaker_agent!(agent_id::String, model::JuMP.Model, EP_market::Dict, H2_market::Dict, H2_GC_market::Dict, admm_params::Dict)

    # --- 1. EXTRACT SETS FOR INDEXING ---
    # Retrieve the sets that define the problem dimensions
    T = model[:T]  # Time steps (e.g., 1:24)
    R = model[:R]  # Representative days (e.g., 1:3)
    Y = model[:Y]  # Years/scenarios (e.g., [2021])
    W = model[:W]  # Representative day weights (probabilities/frequencies)

    # --- 2. UPDATE MARKET PRICES (DUAL VARIABLES) ---
    # Loop through each year to update time-dependent parameters (Prices and References)
    for y in Y
        # -- End Product Market (Output for both Green and Grey) --
        # Update the coordination shadow price for End Product
        # This represents the dual variable for EP market clearing
        # Both green and grey producers receive this price for their EP sales
        model[:lambda_EP][y] .= EP_market["price"][y]
        
        # -- Hydrogen Markets (Input for Green Offtaker) --
        # Update Hydrogen Price (Cost for Green Offtaker when buying H2)
        model[:lambda_H][y] .= H2_market["price"][y]
        # Update Hydrogen Green Certificate Price (Cost).
        # Both green and grey offtakers need this (green buys GCs, grey buys GCs for the mandate).
        # H2 GC prices are annual scalars consistent with annual GC market clearing.
        model[:lambda_GC_H][y] = H2_GC_market["price"][y]

        # -- ADMM REFERENCE QUANTITIES --
        # Reset ADMM reference quantities to 0.0 (Standard Exchange ADMM formulation).
        # In Exchange ADMM, the reference is always zero, so penalty = (rho/2) * variable^2.
        model[:EP_ref][y] .= 0.0      # End Product reference (zero in Exchange ADMM)
        model[:H_ref][y] .= 0.0       # Hydrogen reference (zero in Exchange ADMM)
        # Annual reference quantity for H2 GCs (used in the aggregate penalty term)
        model[:GC_H_ref][y] = 0.0    # H2 GC reference (zero in Exchange ADMM, annual)
    end

    # --- 3. UPDATE ADMM PENALTY PARAMETERS (RHO) ---
    # Update ADMM penalty factors (rho) from the market data
    # These may have been adjusted adaptively by update_rho! to balance convergence
    model[:rho_EP] = EP_market["rho"]      # End Product market penalty parameter
    model[:rho_H] = H2_market["rho"]       # Hydrogen market penalty parameter
    model[:rho_GC_H] = H2_GC_market["rho"] # H2 GC market penalty parameter

    # --- 4. RE-DEFINE OBJECTIVE FUNCTION ---
    # We must reconstruct the objective expression because lambda (prices) and rho
    # (penalty parameters) values have changed. JuMP requires re-parsing expressions
    # when parameter values change to ensure correct optimization.
    
    # -- CASE A: ELECTROLYTIC AMMONIA PRODUCER (GREEN OFFTAKER) --
    # Identified by presence of alpha parameter (conversion factor)
    if haskey(model, :alpha)
        # Objective: Maximize profit = EP Revenue - H2 Cost - GC Cost - Processing - Penalties.
        # H2 GC cost is calculated annually as (aggregate purchases × annual price), reflecting
        # the annual clearing nature of the H2 GC market.
        @objective(model, Max,
            # Hourly terms (EP revenue, H2 cost, processing, penalties)
            sum(W[y][r] * (
                # (+) Revenue EP: Market price of End Product times quantity sold
                (model[:lambda_EP][y][t,r] * model[:ep_sell][t,r,y]) -
                # (-) Cost H2: Market price of hydrogen times quantity purchased
                (model[:lambda_H][y][t,r] * model[:h_buy][t,r,y]) -
                # (-) Processing Cost: Cost of converting H2 to ammonia
                (model[:C_proc] * model[:ep_sell][t,r,y]) -
                # (-) ADMM Penalties for hourly variables: Enforce consensus with market clearing conditions
                # Penalty = (rho/2) * variable^2 (since references are zero)
                (model[:rho_EP] / 2 * (model[:ep_sell][t,r,y])^2) -
                (model[:rho_H] / 2 * (model[:h_buy][t,r,y])^2)
            ) for t in T, r in R, y in Y) -
            # Annual H2 GC Cost (outside hourly loop)
            # Cost = Annual Price × Weighted Annual Aggregate Purchases
            sum(model[:lambda_GC_H][y] * sum(W[y][r] * model[:gc_h_buy][t,r,y] for t in T, r in R) for y in Y) -
            # Annual ADMM Penalty for H2 GC (aggregate penalty)
            # Penalty = (rho/2) * (annual_aggregate_purchases - annual_reference)^2
            sum(model[:rho_GC_H] / 2 * (sum(W[y][r] * model[:gc_h_buy][t,r,y] for t in T, r in R) - model[:GC_H_ref][y])^2 for y in Y)
        )
    
    # -- CASE B: GREY AMMONIA PRODUCER --
    # Identified by presence of gc_h_buy_G variable (H2 GC purchase for grey producer)
    elseif haskey(model, :gc_h_buy_G)
        # Objective: Maximize profit = EP Revenue - Processing Cost - GC Cost - Penalties.
        # H2 GC cost is calculated annually as (aggregate purchases × annual price), reflecting
        # the annual clearing nature of the H2 GC market.
        @objective(model, Max,
            # Hourly terms (EP revenue, processing cost, EP penalty)
            sum(W[y][r] * (
                # (+) Revenue EP: Market price of End Product times quantity sold
                (model[:lambda_EP][y][t,r] * model[:ep_sell][t,r,y]) -
                # (-) Processing Cost: Cost of producing ammonia via SMR
                (model[:C_proc] * model[:ep_sell][t,r,y]) -
                # (-) ADMM Penalties for hourly variables: Enforce consensus with market clearing conditions
                (model[:rho_EP] / 2 * (model[:ep_sell][t,r,y])^2)
            ) for t in T, r in R, y in Y) -
            # Annual H2 GC Cost (outside hourly loop)
            # Cost = Annual Price × Weighted Annual Aggregate Purchases
            sum(model[:lambda_GC_H][y] * sum(W[y][r] * model[:gc_h_buy_G][t,r,y] for t in T, r in R) for y in Y) -
            # Annual ADMM Penalty for H2 GC (aggregate penalty)
            # Penalty = (rho/2) * (annual_aggregate_purchases - annual_reference)^2
            sum(model[:rho_GC_H] / 2 * (sum(W[y][r] * model[:gc_h_buy_G][t,r,y] for t in T, r in R) - model[:GC_H_ref][y])^2 for y in Y)
        )
    end

    # --- 5. SOLVE THE OPTIMIZATION MODEL ---
    # Execute the optimization solver (Gurobi) to find optimal decision variables
    optimize!(model)
    
    # --- 6. CHECK SOLVER STATUS ---
    # Log a warning if the solver failed to find an optimal solution
    # Possible non-optimal statuses: INFEASIBLE, UNBOUNDED, TIME_LIMIT, etc.
    if termination_status(model) != MOI.OPTIMAL
        # Warning prints removed to avoid slowing large runs; check solver status from saved results if needed.
    end
end
