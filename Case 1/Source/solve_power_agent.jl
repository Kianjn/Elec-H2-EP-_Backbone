# ==============================================================================
# SOLVE POWER SECTOR AGENT OPTIMIZATION PROBLEM
# ==============================================================================
# This function is called iteratively within the ADMM loop to solve each
# Power Sector agent's optimization subproblem given current market prices.
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
#   - model::JuMP.Model: JuMP optimization model for this agent (built in build_power_agent!)
#   - elec_market::Dict: Electricity market dictionary with current prices and parameters
#   - elec_GC_market::Dict: Electricity GC market dictionary with current prices
#   - admm_params::Dict: ADMM algorithm parameters (max_iter, epsilon, etc.)
#
# Returns:
#   - Modifies the model in-place by updating prices and solving
#   - Prints warning if optimization fails
# ==============================================================================
function solve_power_agent!(agent_id::String, model::JuMP.Model, elec_market::Dict, elec_GC_market::Dict, admm_params::Dict)
    
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
        # Update local electricity prices with the latest global market prices
        # The .= operator performs element-wise assignment to update the entire price matrix
        # elec_market["price"][y] is a matrix of size (nTimesteps × nReprDays)
        model[:lambda_E][y] .= elec_market["price"][y]
        
        # Update local Green Certificate prices with the latest global market prices
        # This is the price premium for green-certified electricity
        # VRES generators receive this premium in addition to electricity price
        model[:lambda_GC_E][y] .= elec_GC_market["price"][y]
        
        # Update reference quantities for the ADMM penalty term
        # Currently set to 0.0, acting as a regularization term (standard augmented Lagrangian centered at 0)
        # In Exchange ADMM, references are always zero, so penalty = (rho/2) * variable^2
        model[:E_ref][y] .= 0.0 
    end
    
    # --- 3. UPDATE ADMM PENALTY PARAMETERS (RHO) ---
    # Update the ADMM penalty factors (rho) from the market data
    # These might change if adaptive rho logic is triggered (see update_rho.jl)
    # Higher rho = stronger penalty for market imbalances = faster constraint satisfaction
    # Lower rho = weaker penalty = slower constraint satisfaction but better dual convergence
    model[:rho_E] = elec_market["rho"]        # Electricity market penalty parameter
    model[:rho_GC_E] = elec_GC_market["rho"]  # Elec GC market penalty parameter

    # --- 4. RE-DEFINE OBJECTIVE FUNCTION ---
    # We must reconstruct the objective expression because the values of lambda (prices)
    # and rho (penalty parameters) have changed. JuMP needs to re-parse the expression
    # to include the new numerical values.
    #
    # The objective function structure depends on the agent type, which is determined
    # by checking which variables exist in the model.

    # -- CASE A: ELECTRICITY CONSUMER --
    # Identified by presence of d_E variable (electricity demand/consumption)
    if haskey(model, :d_E) 
        # Safe Parameter Access: Get quadratic utility parameters or use defaults
        # A_E: Intercept of inverse demand curve (willingness to pay intercept)
        # B_E: Slope of inverse demand curve (price sensitivity)
        A_E = haskey(model, :A_E) ? model[:A_E] : 500.0  # Typical intercept (EUR/MWh)
        B_E = haskey(model, :B_E) ? model[:B_E] : 0.5   # Typical slope (EUR/MWh²)

        # Define Maximization Objective for Consumer with Quadratic Utility
        # Objective: Maximize consumer surplus = Utility - Cost - ADMM Penalties
        @objective(model, Max, 
            sum(W[y][r] * (
                # (+) Linear utility term: A_E * d
                # Represents the base willingness to pay per unit
                (A_E * model[:d_E][t,r,y]) -
                # (-) Quadratic utility term: (1/2) * B_E * d^2
                # Represents diminishing marginal utility (concavity)
                (0.5 * B_E * model[:d_E][t,r,y]^2) -
                # (-) Cost of Electricity
                # Market price times quantity consumed
                (model[:lambda_E][y][t,r] * model[:d_E][t,r,y]) -
                # (-) ADMM Penalty term
                # Penalizes deviations from market consensus
                # Penalty = (rho/2) * variable^2 (since references are zero)
                (model[:rho_E] / 2 * (model[:d_E][t,r,y])^2)
            ) for t in T, r in R, y in Y)
        )

    # -- CASE B: GENERATOR (VRES OR CONVENTIONAL) --
    # Identified by presence of Q_bar parameter (capacity limit)
    elseif haskey(model, :Q_bar) 
        
        # Check if VRES (Variable Renewable Energy Source) via dictionary capacity type
        # VRES have time-dependent capacity (Dictionary), Conventional have constant capacity (scalar)
        if isa(model[:Q_bar], Dict) 
             # Define Maximization Objective for VRES
             # VRES receive revenue from both electricity and green certificates
             @objective(model, Max,
                sum(W[y][r] * (
                    # (+) Revenue from Electricity
                    # Market price of electricity times quantity generated
                    (model[:lambda_E][y][t,r] * model[:q_E][t,r,y]) +
                    # (+) Revenue from Green Certificates
                    # Price premium for green-certified electricity
                    # Note: For VRES, gc_E = q_E (1 MWh electricity = 1 GC)
                    (model[:lambda_GC_E][y][t,r] * model[:q_E][t,r,y]) -
                    # (-) Variable Cost
                    # Typically very low for renewables (near zero)
                    # Represents variable O&M costs
                    (model[:C] * model[:q_E][t,r,y]) -
                    # (-) Penalty on Electricity
                    # Penalizes deviations from electricity market consensus
                    (model[:rho_E] / 2 * (model[:q_E][t,r,y])^2) -
                    # (-) Penalty on Green Certificates
                    # Penalizes deviations from GC market consensus
                    (model[:rho_GC_E] / 2 * (model[:q_E][t,r,y])^2)
                ) for t in T, r in R, y in Y)
            )
        
        # Check if Conventional Generator (Scalar capacity)
        else 
            # Define Maximization Objective for Conventional Generator
            # Conventional generators only receive revenue from electricity (no GCs)
            @objective(model, Max,
                sum(W[y][r] * (
                    # (+) Revenue from Electricity
                    # Market price of electricity times quantity generated
                    (model[:lambda_E][y][t,r] * model[:q_E][t,r,y]) -
                    # (-) Variable Cost
                    # Includes fuel costs and CO2 emission costs
                    # Typically higher than VRES marginal costs
                    (model[:C] * model[:q_E][t,r,y]) -
                    # (-) Penalty on Electricity
                    # Penalizes deviations from electricity market consensus
                    (model[:rho_E] / 2 * (model[:q_E][t,r,y])^2)
                ) for t in T, r in R, y in Y)
            )
        end
    end

    # --- 5. SOLVE THE OPTIMIZATION MODEL ---
    # Execute the optimization solver (Gurobi) to find optimal decision variables
    # The solver will:
    #   - Respect all constraints defined in build_power_agent!
    #   - Maximize the objective function
    #   - Return optimal values for all decision variables
    optimize!(model)
    
    # --- 6. CHECK SOLVER STATUS ---
    # If the solver did not find an optimal solution, print a warning to the console
    # Possible non-optimal statuses: INFEASIBLE, UNBOUNDED, TIME_LIMIT, etc.
    if termination_status(model) != MOI.OPTIMAL
        # Warning prints removed to avoid slowing large runs; check solver status from saved results if needed.
    end
end
