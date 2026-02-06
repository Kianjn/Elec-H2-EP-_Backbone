# ==============================================================================
# ADMM MAIN ALGORITHM LOOP
# ==============================================================================
# This function executes the Alternating Direction Method of Multipliers (ADMM)
# algorithm to find market equilibrium. ADMM is a distributed optimization algorithm
# that decomposes the centralized market clearing problem into agent subproblems.
#
# Algorithm Overview:
#   1. Each agent optimizes independently given current market prices (x-update)
#   2. Market coordinator calculates imbalances (Supply - Demand)
#   3. Market coordinator updates prices based on imbalances (dual update)
#   4. Check convergence (both primal and dual residuals below threshold)
#   5. Repeat until convergence or max iterations
#
# The algorithm coordinates multiple markets simultaneously:
#   - Electricity Market
#   - Electricity GC Market
#   - Hydrogen Market
#   - Hydrogen GC Market
#   - End Product Market
#
# Arguments:
#   - results::Dict: Dictionary to store final agent results (modified in-place)
#   - ADMM::Dict: Dictionary to store ADMM convergence metrics (modified in-place)
#   - elec_market::Dict: Electricity market dictionary (prices, balances, rho)
#   - H2_market::Dict: Hydrogen market dictionary (prices, balances, rho)
#   - elec_GC_market::Dict: Electricity GC market dictionary (prices, balances, rho)
#   - H2_GC_market::Dict: Hydrogen GC market dictionary (prices, balances, rho)
#   - EP_market::Dict: End Product market dictionary (prices, balances, demand, rho)
#   - mdict::Dict: Dictionary of JuMP models, keyed by agent ID
#   - agents::Dict: Dictionary containing lists of agent IDs by type
#   - data::Dict: Configuration dictionary containing ADMM parameters
#   - TO::TimerOutput: TimerOutput object for performance profiling
#
# Returns:
#   - Modifies results, ADMM, and market dictionaries in-place
#   - Prints convergence status to console
# ==============================================================================
function ADMM!(results, ADMM, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, mdict, agents, data, TO)
    
    # --- 1. INITIALIZATION ---
    
    # Retrieve maximum allowed iterations from the configuration dictionary
    # This prevents infinite loops if convergence is not achieved
    # Typical values: 1000-5000 iterations
    max_iter = data["ADMM"]["max_iter"]
    
    # Retrieve the convergence tolerance (epsilon) from the configuration dictionary
    # Convergence is achieved when both primal and dual residuals are below epsilon
    # Typical values: 0.001-0.0001 (smaller = stricter convergence)
    epsilon = data["ADMM"]["epsilon"]
    
    # Initialize the convergence flag to false (loop continues until true)
    # This flag will be set to true when convergence criteria are met
    converged = false
    
    # --- 2. MAIN ITERATION LOOP ---
    # Start the iteration loop. It runs until max_iter is reached or convergence is achieved.
    # The @timeit macro records the total time spent in this "ADMM Loop" section for profiling.
    # This helps identify performance bottlenecks in the optimization process.
    @timeit TO "ADMM Loop" begin
        # Use a ProgressBars.jl progress bar to track ADMM iterations.
        # This provides a single, continuously updated line instead of many printlns.
        pb = ProgressBar(1:max_iter)
        for iter in pb
        
        # ======================================================================
        # STEP A: SOLVE AGENT SUBPROBLEMS (X-UPDATE)
        # ======================================================================
        # In this step, agents optimize their own decisions based on the current market prices.
        # This corresponds to the x-update step in standard ADMM.
        # Each agent solves its optimization problem independently, treating market prices as fixed.
        # The agents' decisions will be used to calculate market imbalances in the next step.
        
        # -- Power Sector Agents --
        # Profile this specific block under "Solve Power" for performance tracking
        # Power agents include: VRES generators, Conventional generators, Electricity consumers
        @timeit TO "Solve Power" for id in agents[:power]
            # Solve the optimization model for the specific Power agent
            # Each agent optimizes its generation/consumption given current electricity and GC prices
            solve_power_agent!(id, mdict[id], elec_market, elec_GC_market, data["ADMM"])
        end
        
        # -- Hydrogen Sector Agents --
        # Profile this specific block under "Solve H2" for performance tracking
        # Hydrogen agents include: Electrolytic H2 producers, Hydrogen consumers
        @timeit TO "Solve H2" for id in agents[:H2]
            # Solve the optimization model for the specific Hydrogen agent.
            # Electrolytic producers optimize electricity purchase, H2 production, and GC trades.
            solve_H2_agent!(id, mdict[id], elec_market, H2_market, elec_GC_market, H2_GC_market, data["ADMM"])
        end
        
        # -- Offtaker Agents (Ammonia Producers) --
        # Profile this specific block under "Solve Offtaker" for performance tracking
        # Offtaker agents include: Electrolytic Ammonia Producers (green), Grey Ammonia Producers
        @timeit TO "Solve Offtaker" for id in agents[:offtaker]
            # Solve the optimization model for the specific Offtaker agent
            # Green offtakers optimize H2 purchase, GC purchase, and EP production
            # Grey offtakers optimize EP production and GC purchase (for policy mandate)
            # Note: We pass EP_market here so they see the End Product shadow price
            solve_offtaker_agent!(id, mdict[id], EP_market, H2_market, H2_GC_market, data["ADMM"])
        end

        # -- Electricity GC Demand Agents --
        # Profile this specific block under "Solve Elec GC Demand" for performance tracking
        # These agents have demand for green certificates with quadratic utility
        # Check if any Electricity GC Demand agents exist (they are optional)
        if haskey(agents, :elec_GC_demand) && length(agents[:elec_GC_demand]) > 0
            @timeit TO "Solve Elec GC Demand" for id in agents[:elec_GC_demand]
                # Solve the optimization model for the specific Electricity GC Demand agent
                # These agents optimize their GC consumption based on utility and prices
                solve_elec_GC_demand_agent!(id, mdict[id], elec_GC_market, data["ADMM"])
            end
        end

        # ======================================================================
        # STEP B: UPDATE MARKET BALANCES
        # ======================================================================
        # Calculate the net imbalance (Supply - Demand) for all markets based on agent decisions.
        # This prepares the data for the Primal Residual calculation.
        # The balance represents how far the markets are from equilibrium.
        # Positive balance = excess supply, Negative balance = excess demand
        @timeit TO "Update Balances" update_market_balances!(agents, mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, data)

        # ======================================================================
        # STEP C: CALCULATE CONVERGENCE RESIDUALS
        # ======================================================================
        # Calculate Primal Residual (Max Imbalance) and Dual Residual (Stationarity proxy).
        # These metrics determine if the system has reached equilibrium.
        # Primal Residual: Maximum absolute imbalance across all markets and time steps
        # Dual Residual: Proxy for stationarity (currently set equal to Primal for simplicity)
        # Convergence requires both residuals to be below epsilon threshold
        primal_res, dual_res, diagnostics = calculate_residuals(elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, data)
        
        # ======================================================================
        # STEP D: LOG CONVERGENCE METRICS
        # ======================================================================
            # Store the current iteration number for convergence history tracking
            # This allows plotting convergence graphs after the algorithm finishes
            push!(ADMM["iter"], iter)
        
            # Store the current Primal Residual (maximum imbalance)
            # This tracks how close the markets are to equilibrium
            push!(ADMM["primal_residual"], primal_res)
        
            # Store the current Dual Residual (stationarity measure)
            # This tracks how stable the solution is
            push!(ADMM["dual_residual"], dual_res)
        
            # Store diagnostic information about where the maximum imbalance occurs
            # This helps identify which market and time period is causing convergence issues
            push!(ADMM["diagnostics"], diagnostics)

        # ======================================================================
        # STEP E: CHECK CONVERGENCE
        # ======================================================================
            # If both residuals are below the epsilon threshold, the algorithm has converged.
            # Convergence means: markets are balanced (primal) and solution is stable (dual)
            if primal_res < epsilon && dual_res < epsilon
                # Set flag to true and break out of the loop
                converged = true
                break
            else
                # ==============================================================
                # STEP F: UPDATE DUAL VARIABLES (PRICES)
                # ==============================================================
                # If not converged, update the market prices (Lagrange multipliers).
                # This is the dual update step in ADMM.
                # Logic: Price = Price - Rho * Balance (Dual Ascent step)
                # If balance > 0 (excess supply), price decreases (encourages more demand)
                # If balance < 0 (excess demand), price increases (encourages more supply)
                # The step size is controlled by rho (penalty parameter)
                @timeit TO "Update Prices" update_prices!(elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market)

                # ==============================================================
                # STEP G: UPDATE PENALTY PARAMETER (RHO)
                # ==============================================================
                # Dynamically adjust the penalty parameter rho to balance primal and dual convergence.
                # This helps speed up convergence if one residual is much larger than the other.
                # Adaptive rho strategy:
                #   - If primal >> dual: increase rho (stronger penalty for imbalances)
                #   - If dual >> primal: decrease rho (weaker penalty, allow prices to adjust)
                # Note: EP_market is included to ensure its penalty is also updated.
                update_rho!(elec_market, H2_market, elec_GC_market, H2_GC_market, EP_market, primal_res, dual_res, data["ADMM"])
            end
        end
    end  # end @timeit "ADMM Loop"
    
    # --- 3. FINAL CHECK ---
    # If the loop finished but converged is still false, print a short summary warning.
    # This runs only once, so it has negligible impact on performance.
    if !converged
        println()  # move to next line after progress bar
        println(">> ADMM reached max iterations without convergence.")
        println("   Final Primal Residual = ", ADMM["primal_residual"][end])
        println("   Final Dual Residual   = ", ADMM["dual_residual"][end])
    else
        println()  # move to next line after progress bar
        println(">> ADMM converged successfully.")
    end
end
