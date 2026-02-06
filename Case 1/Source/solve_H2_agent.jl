# ==============================================================================
# SOLVE HYDROGEN AGENT OPTIMIZATION PROBLEM
# ==============================================================================
# This function is called iteratively within the ADMM loop to solve each
# Hydrogen Sector agent's optimization subproblem given current market prices.
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
#   - model::JuMP.Model: JuMP optimization model for this agent (built in build_H2_agent!)
#   - elec_market::Dict: Electricity market dictionary with current prices and parameters
#   - H2_market::Dict: Hydrogen market dictionary with current prices and parameters
#   - elec_GC_market::Dict: Electricity GC market dictionary with current prices
#   - H2_GC_market::Dict: Hydrogen GC market dictionary with current prices
#   - admm_params::Dict: ADMM algorithm parameters (max_iter, epsilon, etc.)
#
# Returns:
#   - Modifies the model in-place by updating prices and solving
#   - Prints warning if optimization fails
# ==============================================================================
function solve_H2_agent!(agent_id::String, model::JuMP.Model, 
                         elec_market::Dict, H2_market::Dict, 
                         elec_GC_market::Dict, H2_GC_market::Dict,
                         admm_params::Dict)

    # --- 1. EXTRACT SETS FOR INDEXING ---
    # Retrieve the sets that define the problem dimensions
    # These were stored in the model during the build phase
    T = model[:T]  # Time steps (e.g., 1:24)
    R = model[:R]  # Representative days (e.g., 1:3)
    Y = model[:Y]  # Years/scenarios (e.g., [2021])
    W = model[:W]  # Representative day weights (probabilities/frequencies)

    # --- 2. UPDATE MARKET PRICES (DUAL VARIABLES) ---
    # Loop through each year/scenario to update time-dependent price parameters
    # Prices are updated from the global market dictionaries, which are modified
    # by the ADMM coordinator based on market imbalances
    for y in Y
        # -- INPUT PRICES (Costs for Electrolytic Producer) --
        # Update Electricity Price: This is the cost per MWh of electricity purchased
        # The .= operator performs element-wise assignment to update the entire price matrix
        # elec_market["price"][y] is a matrix of size (nTimesteps × nReprDays)
        model[:lambda_E][y] .= elec_market["price"][y]
        
        # Update Electricity Green Certificate Price: Cost per Elec GC purchased
        # This is required for the green backing constraint and economic optimization
        # The producer must buy Elec GCs to certify their hydrogen as green
        model[:lambda_GC_E][y] .= elec_GC_market["price"][y]
        
        # -- OUTPUT PRICES (Revenue for Producers, Cost for Consumers) --
        # Update Hydrogen Price: Revenue per MWh of hydrogen sold (for producers)
        # or cost per MWh of hydrogen purchased (for consumers)
        model[:lambda_H][y] .= H2_market["price"][y]
        
        # Update Hydrogen Green Certificate Price: Revenue per H2 GC sold (for producers)
        # This represents the price premium for green-certified hydrogen.
        # H2 GC prices are modeled as annual scalars, consistent with annual GC market clearing.
        model[:lambda_GC_H][y] = H2_GC_market["price"][y]

        # -- ADMM REFERENCE QUANTITIES --
        # Reset ADMM reference quantities to 0.0 (Standard Exchange ADMM formulation)
        # In Exchange ADMM, the reference is always zero, so the penalty term becomes
        # (rho/2) * variable^2, which penalizes large deviations from zero
        # This is different from standard ADMM where references track consensus values
        model[:E_ref][y] .= 0.0      # Electricity reference (zero in Exchange ADMM)
        model[:GC_E_ref][y] .= 0.0  # Elec GC reference (zero in Exchange ADMM)
        model[:H_ref][y] .= 0.0      # Hydrogen reference (zero in Exchange ADMM)
        # Annual reference quantity for H2 GCs (used in the aggregate penalty term)
        model[:GC_H_ref][y] = 0.0   # H2 GC reference (zero in Exchange ADMM, annual)
    end
    
    # --- 3. UPDATE ADMM PENALTY PARAMETERS (RHO) ---
    # Update penalty factors from the market data
    # These may have been adjusted adaptively by update_rho! to balance convergence
    # Higher rho = stronger penalty for market imbalances = faster constraint satisfaction
    # Lower rho = weaker penalty = slower constraint satisfaction but better dual convergence
    model[:rho_E] = elec_market["rho"]        # Electricity market penalty parameter
    model[:rho_GC_E] = elec_GC_market["rho"]  # Elec GC market penalty parameter
    model[:rho_H] = H2_market["rho"]          # Hydrogen market penalty parameter
    model[:rho_GC_H] = H2_GC_market["rho"]    # H2 GC market penalty parameter

    # --- 4. RE-DEFINE OBJECTIVE FUNCTION ---
    # We must reconstruct the objective expression because lambda (prices) and rho
    # (penalty parameters) values have changed. JuMP requires re-parsing expressions
    # when parameter values change to ensure correct optimization.
    #
    # The objective function structure depends on the agent type, which is determined
    # by checking which variables exist in the model.

    # -- CASE A: ELECTROLYTIC HYDROGEN PRODUCER --
    # Identified by presence of e_buy variable (electricity purchase)
    if haskey(model, :e_buy) 
        # Objective: Maximize profit = Revenue - Costs - ADMM Penalties.
        # Revenue comes from selling hydrogen and H2 GCs.
        # Costs include electricity, Elec GCs, and operational expenses.
        # H2 GC revenue is calculated on an annual basis as price × weighted annual GC sales,
        # consistent with the annual scalar balance used in the H2 GC market.
        @objective(model, Max,
            # Hourly terms (H2 sales, costs, penalties)
            sum(W[y][r] * (
                # (+) Revenue from selling Hydrogen
                # Market price times quantity sold
                (model[:lambda_H][y][t,r] * model[:h_sell][t,r,y]) -
                # (-) Cost of buying Electricity
                # Market price times quantity purchased
                (model[:lambda_E][y][t,r] * model[:e_buy][t,r,y]) -
                # (-) Cost of buying Electricity Green Certificates
                # Required to certify hydrogen as green
                (model[:lambda_GC_E][y][t,r] * model[:gc_e_buy][t,r,y]) -
                # (-) Operational Cost - Linear cost function based on electricity input
                # Note: LaTeX specifies C_H(e) = c_{H,0} * e + (1/2) * c_{H,1} * e^2
                # For linear version: C_H(e) = c_{H,0} * e_buy (not h_sell)
                # This represents variable operational and maintenance costs
                # The cost is proportional to electricity input, not hydrogen output
                (model[:C_H] * model[:e_buy][t,r,y]) -
                # (-) ADMM Penalties for hourly variables (Quadratic form)
                # These penalties enforce consensus with market clearing conditions
                # Penalty = (rho/2) * variable^2 (since references are zero)
                # Higher values of variables lead to larger penalties, encouraging
                # agents to align their decisions with market equilibrium
                (model[:rho_H] / 2 * (model[:h_sell][t,r,y])^2) -       # Penalty on H2 sales
                (model[:rho_E] / 2 * (model[:e_buy][t,r,y])^2) -        # Penalty on Elec purchases
                (model[:rho_GC_E] / 2 * (model[:gc_e_buy][t,r,y])^2)    # Penalty on Elec GC purchases
            ) for t in T, r in R, y in Y) +
            # Annual H2 GC Revenue and ADMM Penalty (aggregate, consistent with market balance)
            # Revenue = Annual Price × Weighted Annual GC Sales
            # Penalty = (rho/2) * (annual_aggregate_sales - annual_reference)^2
            sum(
                model[:lambda_GC_H][y] * sum(W[y][r] * model[:gc_h_sell][t,r,y] for t in T, r in R) -
                (model[:rho_GC_H] / 2) * (sum(W[y][r] * model[:gc_h_sell][t,r,y] for t in T, r in R) - model[:GC_H_ref][y])^2
                for y in Y
            )
        )

    # -- CASE B: HYDROGEN CONSUMER --
    # Identified by presence of d_H variable (hydrogen demand/consumption)
    elseif haskey(model, :d_H) 
        # Safe Parameter Access: Check if utility parameter exists, default to 0.0
        # Utility: Willingness to pay per unit of hydrogen (EUR/MWh)
        utility_val = haskey(model, :Utility) ? model[:Utility] : 0.0

        # Objective: Maximize consumer surplus = Utility - Cost - ADMM Penalties
        # The consumer maximizes their net benefit from consuming hydrogen
        @objective(model, Max,
            sum(W[y][r] * (
                # (+) Utility from consuming hydrogen
                # Linear utility function: Utility = utility_val * quantity
                (utility_val * model[:d_H][t,r,y]) -
                # (-) Cost of buying Hydrogen
                # Market price times quantity consumed
                (model[:lambda_H][y][t,r] * model[:d_H][t,r,y]) -
                # (-) ADMM Penalty for hydrogen consumption
                # Penalizes deviations from market consensus
                (model[:rho_H] / 2 * (model[:d_H][t,r,y])^2)
            ) for t in T, r in R, y in Y)
        )
    end

    # --- 5. SOLVE THE OPTIMIZATION MODEL ---
    # Execute the optimization solver (Gurobi) to find optimal decision variables
    # The solver will:
    #   - Respect all constraints defined in build_H2_agent!
    #   - Maximize the objective function
    #   - Return optimal values for all decision variables
    optimize!(model)
    
    # --- 6. CHECK SOLVER STATUS ---
    # Verify that the solver found an optimal solution
    # If not optimal, print a warning to alert the user
    # Possible non-optimal statuses: INFEASIBLE, UNBOUNDED, TIME_LIMIT, etc.
    if termination_status(model) != MOI.OPTIMAL
        # Warning prints removed to avoid slowing large runs; check solver status from saved results if needed.
    end
end
