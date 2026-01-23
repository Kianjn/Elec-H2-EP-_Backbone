# ==============================================================================
# SOLVE ELECTRICITY GC DEMAND AGENT OPTIMIZATION PROBLEM
# ==============================================================================
# This function is called iteratively within the ADMM loop to solve each
# Electricity GC Demand Agent's optimization subproblem given current market prices.
#
# In the ADMM algorithm, this corresponds to the "x-update" step where each
# agent optimizes its decisions based on the current dual variables (prices)
# from the market coordinator.
#
# The function:
#   1. Updates market prices (dual variables) in the agent's model
#   2. Updates ADMM penalty parameters (rho)
#   3. Reconstructs the objective function with new prices
#   4. Solves the optimization problem
#   5. Checks for optimal solution
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent
#   - model::JuMP.Model: JuMP optimization model for this agent (built in build_elec_GC_demand_agent!)
#   - elec_GC_market::Dict: Electricity GC market dictionary with current prices and parameters
#   - admm_params::Dict: ADMM algorithm parameters (max_iter, epsilon, etc.)
#
# Returns:
#   - Modifies the model in-place by updating prices and solving
#   - Prints warning if optimization fails
# ==============================================================================
function solve_elec_GC_demand_agent!(agent_id::String, model::JuMP.Model, elec_GC_market::Dict, admm_params::Dict)
    
    # --- 1. EXTRACT SETS FOR INDEXING ---
    # Retrieve the sets that define the problem dimensions
    # These were stored in the model during the build phase
    T = model[:T]  # Time steps (e.g., 1:24)
    R = model[:R]  # Representative days (e.g., 1:3)
    Y = model[:Y]  # Years/scenarios (e.g., [2021])
    W = model[:W]  # Representative day weights (probabilities/frequencies)

    # --- 2. UPDATE MARKET PRICES (DUAL VARIABLES) ---
    # Loop through each year to update time-dependent price parameters
    # Prices are updated from the global market dictionary, which is modified
    # by the ADMM coordinator based on market imbalances
    for y in Y
        # Update local GC prices with the latest global market prices
        # The .= operator performs element-wise assignment to update the entire price matrix
        # elec_GC_market["price"][y] is a matrix of size (nTimesteps × nReprDays)
        # This represents the current price premium for green-certified electricity
        model[:lambda_GC_E][y] .= elec_GC_market["price"][y]
        
        # Update reference quantities for the ADMM penalty term
        # Reset ADMM reference quantities to 0.0 (Standard Exchange ADMM formulation)
        # In Exchange ADMM, the reference is always zero, so penalty = (rho/2) * variable^2
        model[:GC_E_ref][y] .= 0.0
    end
    
    # --- 3. UPDATE ADMM PENALTY PARAMETERS (RHO) ---
    # Update the ADMM penalty factor (rho) from the market data
    # This may have been adjusted adaptively by update_rho! to balance convergence
    # Higher rho = stronger penalty for market imbalances = faster constraint satisfaction
    model[:rho_GC_E] = elec_GC_market["rho"]

    # --- 4. RE-DEFINE OBJECTIVE FUNCTION ---
    # We must reconstruct the objective expression because lambda (prices) and rho
    # (penalty parameters) values have changed. JuMP requires re-parsing expressions
    # when parameter values change to ensure correct optimization.
    #
    # Objective: Maximize consumer surplus = Quadratic Utility - Cost - ADMM Penalties
    # Utility function: U(d) = A_GC * d - (1/2) * B_GC * d^2
    @objective(model, Max, 
        sum(W[y][r] * (
            # (+) Linear utility term: A_GC * d
            # Represents the base willingness to pay per certificate
            (model[:A_GC] * model[:d_GC_E][t,r,y]) -
            # (-) Quadratic utility term: (1/2) * B_GC * d^2
            # Represents diminishing marginal utility (concavity)
            (0.5 * model[:B_GC] * model[:d_GC_E][t,r,y]^2) -
            # (-) Cost of GCs
            # Market price of green certificates times quantity purchased
            (model[:lambda_GC_E][y][t,r] * model[:d_GC_E][t,r,y]) -
            # (-) ADMM Penalty term
            # Penalizes deviations from market consensus
            # Penalty = (rho/2) * variable^2 (since references are zero in Exchange ADMM)
            (model[:rho_GC_E] / 2 * (model[:d_GC_E][t,r,y])^2)
        ) for t in T, r in R, y in Y)
    )

    # --- 5. SOLVE THE OPTIMIZATION MODEL ---
    # Execute the optimization solver (Gurobi) to find optimal decision variables
    # The solver will:
    #   - Respect all constraints defined in build_elec_GC_demand_agent!
    #   - Maximize the objective function
    #   - Return optimal values for all decision variables
    optimize!(model)
    
    # --- 6. CHECK SOLVER STATUS ---
    # Verify that the solver found an optimal solution
    # If not optimal, print a warning to alert the user
    # Possible non-optimal statuses: INFEASIBLE, UNBOUNDED, TIME_LIMIT, etc.
    if termination_status(model) != MOI.OPTIMAL
        println("Warning: Agent $agent_id did not solve optimally.")
        println("  Termination status: $(termination_status(model))")
    end
end
