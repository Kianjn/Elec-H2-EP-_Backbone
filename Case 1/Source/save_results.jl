# ==============================================================================
# SAVE SIMULATION RESULTS TO CSV FILES
# ==============================================================================
# This function processes and saves the final simulation results to CSV files.
# It creates three main output files:
#   1. ADMM_Convergence.csv: Convergence history (iterations, residuals)
#   2. Market_Prices.csv: Final market prices (electricity, hydrogen)
#   3. Agent_Summary.csv: Summary statistics for each agent (type, generation, profit)
#
# The results are saved to a folder named "Results_{nReprDays}_repr_days" to
# distinguish results from different representative day configurations.
#
# Arguments:
#   - mdict::Dict: Dictionary of solved JuMP models, keyed by agent ID
#                  Contains optimal variable values from the ADMM solution
#   - elec_market::Dict: Electricity market dictionary with final prices
#   - H2_market::Dict: Hydrogen market dictionary with final prices
#   - elec_GC_market::Dict: Electricity GC market dictionary (not saved, but kept for consistency)
#   - H2_GC_market::Dict: Hydrogen GC market dictionary (not saved, but kept for consistency)
#   - ADMM::Dict: Dictionary containing ADMM convergence metrics
#   - results::Dict: Dictionary containing agent results (not used here, kept for consistency)
#   - agents::Dict: Dictionary containing lists of agent IDs by type
#
# Returns:
#   - Creates CSV files in the results folder
#   - Prints success message to console
# ==============================================================================
function save_results(mdict, elec_market, H2_market, elec_GC_market, H2_GC_market, ADMM, results, agents)
    
    # Print a status message to the console indicating the saving process has started
    
    # --- 0. CONSTRUCT RESULTS FOLDER PATH ---
    # We must match the folder name logic used in main.jl: "Results_{nReprDays}_repr_days"
    # We can infer nReprDays from the length of the price arrays or pass it in.
    # To keep it robust, we'll recalculate it from the market data structure.
    
    # Extract year (first key from the price dictionary)
    # This assumes at least one year exists in the data
    y = collect(keys(elec_market["price"]))[1]
    
    # Get number of representative days from the dimensions of the price matrix
    # The price matrix has dimensions (nTimesteps × nReprDays)
    # We extract the second dimension (nReprDays) using size(..., 2)
    nReprDays = size(elec_market["price"][y], 2)
    
    # Construct the results folder path
    # home_dir is a global constant defined in Main.jl
    # Folder name format: "Results_3_repr_days" (if nReprDays = 3)
    results_folder = joinpath(home_dir, "Results_$(nReprDays)_repr_days")
    
    # Create the folder if it doesn't exist (safety check)
    # This ensures the folder exists before trying to write files
    if !isdir(results_folder)
        mkdir(results_folder)
    end

    # --- 1. SAVE CONVERGENCE DATA ---
    # Create a DataFrame containing the iteration history from the ADMM dictionary
    # This logs the Primal Residual (imbalance) and Dual Residual (price change) per iteration
    # This data can be used to plot convergence graphs and analyze algorithm performance
    convergence_df = DataFrame(
        Iter = ADMM["iter"],              # Iteration numbers (1, 2, 3, ...)
        PrimalRes = ADMM["primal_residual"],  # Primal residual at each iteration
        DualRes = ADMM["dual_residual"]       # Dual residual at each iteration
    )
    
    # Write the convergence DataFrame to the dynamic results folder
    # File: ADMM_Convergence.csv
    # Columns: Iter, PrimalRes, DualRes
    CSV.write(joinpath(results_folder, "ADMM_Convergence.csv"), convergence_df)

    # --- 1b. SAVE DIAGNOSTICS DATA ---
    # Create a DataFrame containing diagnostic information about where the maximum imbalance occurs
    # This helps identify which market and time period is causing convergence issues
    # Extract diagnostic information from each iteration
    diagnostics_data = []
    for (i, diag) in enumerate(ADMM["diagnostics"])
        push!(diagnostics_data, (
            Iter = ADMM["iter"][i],
            Market = diag["market"],
            Year = diag["year"],
            Time = diag["time"],
            ReprDay = diag["repr_day"],
            Imbalance = diag["imbalance"],
            AbsImbalance = diag["abs_imbalance"]
        ))
    end
    
    # Create DataFrame from the collected diagnostics data
    diagnostics_df = DataFrame(diagnostics_data)
    
    # Write the diagnostics DataFrame to the dynamic results folder
    # File: ADMM_Diagnostics.csv
    # Columns: Iter, Market, Year, Time, ReprDay, Imbalance, AbsImbalance
    CSV.write(joinpath(results_folder, "ADMM_Diagnostics.csv"), diagnostics_df)

    # --- 1c. SAVE MARKET DEBUG INFORMATION FOR ALL MARKETS ---
    # DEBUG: Save detailed market balance information for all markets
    # This helps identify issues with market clearing and convergence
    
    # 1. Electricity Market Debug (Hourly - aggregate to annual)
    if haskey(elec_market, "debug_supply") && haskey(elec_market, "debug_demand")
        elec_debug_data = []
        for y in keys(elec_market["debug_supply"])
            # Aggregate hourly supply and demand to annual totals (weighted by representative day weights)
            # We need to get weights from one of the models
            first_agent = collect(agents[:all])[1]
            if haskey(mdict[first_agent], :W)
                W = mdict[first_agent][:W]
                annual_supply = sum(W[y][r] * elec_market["debug_supply"][y][t, r] for t in 1:size(elec_market["debug_supply"][y], 1), r in 1:size(elec_market["debug_supply"][y], 2))
                annual_demand = sum(W[y][r] * elec_market["debug_demand"][y][t, r] for t in 1:size(elec_market["debug_demand"][y], 1), r in 1:size(elec_market["debug_demand"][y], 2))
                annual_balance = sum(W[y][r] * elec_market["balance"][y][t, r] for t in 1:size(elec_market["balance"][y], 1), r in 1:size(elec_market["balance"][y], 2))
                avg_price = sum(elec_market["price"][y]) / length(elec_market["price"][y])
            else
                annual_supply = sum(elec_market["debug_supply"][y])
                annual_demand = sum(elec_market["debug_demand"][y])
                annual_balance = sum(elec_market["balance"][y])
                avg_price = sum(elec_market["price"][y]) / length(elec_market["price"][y])
            end
            push!(elec_debug_data, (
                Year = y,
                Annual_Supply = annual_supply,
                Annual_Demand = annual_demand,
                Annual_Balance = annual_balance,
                Avg_Price = avg_price,
                Rho = elec_market["rho"]
            ))
        end
        if length(elec_debug_data) > 0
            elec_debug_df = DataFrame(elec_debug_data)
            CSV.write(joinpath(results_folder, "Electricity_Market_Debug.csv"), elec_debug_df)
        end
    end
    
    # 2. Hydrogen Market Debug (Hourly - aggregate to annual)
    if haskey(H2_market, "debug_supply") && haskey(H2_market, "debug_demand")
        h2_debug_data = []
        for y in keys(H2_market["debug_supply"])
            first_agent = collect(agents[:all])[1]
            if haskey(mdict[first_agent], :W)
                W = mdict[first_agent][:W]
                annual_supply = sum(W[y][r] * H2_market["debug_supply"][y][t, r] for t in 1:size(H2_market["debug_supply"][y], 1), r in 1:size(H2_market["debug_supply"][y], 2))
                annual_demand = sum(W[y][r] * H2_market["debug_demand"][y][t, r] for t in 1:size(H2_market["debug_demand"][y], 1), r in 1:size(H2_market["debug_demand"][y], 2))
                annual_balance = sum(W[y][r] * H2_market["balance"][y][t, r] for t in 1:size(H2_market["balance"][y], 1), r in 1:size(H2_market["balance"][y], 2))
                avg_price = sum(H2_market["price"][y]) / length(H2_market["price"][y])
            else
                annual_supply = sum(H2_market["debug_supply"][y])
                annual_demand = sum(H2_market["debug_demand"][y])
                annual_balance = sum(H2_market["balance"][y])
                avg_price = sum(H2_market["price"][y]) / length(H2_market["price"][y])
            end
            push!(h2_debug_data, (
                Year = y,
                Annual_Supply = annual_supply,
                Annual_Demand = annual_demand,
                Annual_Balance = annual_balance,
                Avg_Price = avg_price,
                Rho = H2_market["rho"]
            ))
        end
        if length(h2_debug_data) > 0
            h2_debug_df = DataFrame(h2_debug_data)
            CSV.write(joinpath(results_folder, "Hydrogen_Market_Debug.csv"), h2_debug_df)
        end
    end
    
    # 3. Electricity GC Market Debug (Hourly - aggregate to annual)
    if haskey(elec_GC_market, "debug_supply") && haskey(elec_GC_market, "debug_demand")
        elec_gc_debug_data = []
        for y in keys(elec_GC_market["debug_supply"])
            first_agent = collect(agents[:all])[1]
            if haskey(mdict[first_agent], :W)
                W = mdict[first_agent][:W]
                annual_supply = sum(W[y][r] * elec_GC_market["debug_supply"][y][t, r] for t in 1:size(elec_GC_market["debug_supply"][y], 1), r in 1:size(elec_GC_market["debug_supply"][y], 2))
                annual_demand = sum(W[y][r] * elec_GC_market["debug_demand"][y][t, r] for t in 1:size(elec_GC_market["debug_demand"][y], 1), r in 1:size(elec_GC_market["debug_demand"][y], 2))
                annual_balance = sum(W[y][r] * elec_GC_market["balance"][y][t, r] for t in 1:size(elec_GC_market["balance"][y], 1), r in 1:size(elec_GC_market["balance"][y], 2))
                avg_price = sum(elec_GC_market["price"][y]) / length(elec_GC_market["price"][y])
            else
                annual_supply = sum(elec_GC_market["debug_supply"][y])
                annual_demand = sum(elec_GC_market["debug_demand"][y])
                annual_balance = sum(elec_GC_market["balance"][y])
                avg_price = sum(elec_GC_market["price"][y]) / length(elec_GC_market["price"][y])
            end
            push!(elec_gc_debug_data, (
                Year = y,
                Annual_Supply = annual_supply,
                Annual_Demand = annual_demand,
                Annual_Balance = annual_balance,
                Avg_Price = avg_price,
                Rho = elec_GC_market["rho"]
            ))
        end
        if length(elec_gc_debug_data) > 0
            elec_gc_debug_df = DataFrame(elec_gc_debug_data)
            CSV.write(joinpath(results_folder, "Electricity_GC_Market_Debug.csv"), elec_gc_debug_df)
        end
    end
    
    # 4. Hydrogen GC Market Debug (Annual - already exists, keep it)
    if haskey(H2_GC_market, "debug_supply") && haskey(H2_GC_market, "debug_demand")
        h2_gc_debug_data = []
        for y in keys(H2_GC_market["debug_supply"])
            # DEBUG: Include H2 sales to verify constraint gc_h_sell <= h_sell
            annual_h2_sales = haskey(H2_GC_market, "debug_h2_sales") ? H2_GC_market["debug_h2_sales"][y] : 0.0
            push!(h2_gc_debug_data, (
                Year = y,
                Annual_Supply = H2_GC_market["debug_supply"][y],
                Annual_Demand = H2_GC_market["debug_demand"][y],
                Annual_Balance = H2_GC_market["balance"][y],
                Annual_Price = H2_GC_market["price"][y],
                Rho = H2_GC_market["rho"],
                Annual_H2_Sales = annual_h2_sales,  # DEBUG: Weighted annual H2 production (for constraint verification)
                GC_to_H2_Ratio = annual_h2_sales > 0 ? H2_GC_market["debug_supply"][y] / annual_h2_sales : Inf  # DEBUG: Should be <= 1.0
            ))
        end
        h2_gc_debug_df = DataFrame(h2_gc_debug_data)
        CSV.write(joinpath(results_folder, "H2_GC_Market_Debug.csv"), h2_gc_debug_df)
        y = collect(keys(H2_GC_market["debug_supply"]))[1]
        annual_h2_sales = haskey(H2_GC_market, "debug_h2_sales") ? H2_GC_market["debug_h2_sales"][y] : 0.0
    end
    
    # 5. End Product Market Debug (Hourly - aggregate to annual)
    if haskey(EP_market, "debug_supply") && haskey(EP_market, "debug_demand")
        ep_debug_data = []
        for y in keys(EP_market["debug_supply"])
            first_agent = collect(agents[:all])[1]
            if haskey(mdict[first_agent], :W)
                W = mdict[first_agent][:W]
                annual_supply = sum(W[y][r] * EP_market["debug_supply"][y][t, r] for t in 1:size(EP_market["debug_supply"][y], 1), r in 1:size(EP_market["debug_supply"][y], 2))
                annual_demand = sum(W[y][r] * EP_market["debug_demand"][y][t, r] for t in 1:size(EP_market["debug_demand"][y], 1), r in 1:size(EP_market["debug_demand"][y], 2))
                annual_balance = sum(W[y][r] * EP_market["balance"][y][t, r] for t in 1:size(EP_market["balance"][y], 1), r in 1:size(EP_market["balance"][y], 2))
                avg_price = sum(EP_market["price"][y]) / length(EP_market["price"][y])
            else
                annual_supply = sum(EP_market["debug_supply"][y])
                annual_demand = sum(EP_market["debug_demand"][y])
                annual_balance = sum(EP_market["balance"][y])
                avg_price = sum(EP_market["price"][y]) / length(EP_market["price"][y])
            end
            push!(ep_debug_data, (
                Year = y,
                Annual_Supply = annual_supply,
                Annual_Demand = annual_demand,
                Annual_Balance = annual_balance,
                Avg_Price = avg_price,
                Rho = EP_market["rho"]
            ))
        end
        if length(ep_debug_data) > 0
            ep_debug_df = DataFrame(ep_debug_data)
            CSV.write(joinpath(results_folder, "End_Product_Market_Debug.csv"), ep_debug_df)
        end
    end
    
    # --- 1d. SAVE PRICE HISTORY FOR ALL MARKETS ---
    # Save price and balance evolution over iterations for all markets
    
    markets_to_save = [
        ("Electricity", elec_market),
        ("Hydrogen", H2_market),
        ("Electricity_GC", elec_GC_market),
        ("Hydrogen_GC", H2_GC_market),
        ("End_Product", EP_market)
    ]
    
    for (market_name, market_dict) in markets_to_save
        if haskey(market_dict, "price_history") && haskey(market_dict, "balance_history")
            price_balance_data = []
            for y in keys(market_dict["price_history"])
                for (iter, (price, balance)) in enumerate(zip(market_dict["price_history"][y], market_dict["balance_history"][y]))
                    push!(price_balance_data, (
                        Iter = iter,
                        Year = y,
                        Price = price,
                        Balance = balance
                    ))
                end
            end
            if length(price_balance_data) > 0
                price_balance_df = DataFrame(price_balance_data)
                CSV.write(joinpath(results_folder, "$(market_name)_Price_History.csv"), price_balance_df)
            end
        end
    end

    # --- 2. SAVE MARKET PRICES ---
    # Create a DataFrame for the hourly/period prices
    # We use 'vec()' to flatten the (Timesteps × ReprDays) matrix into a single column vector
    # This converts the 2D price matrix into a 1D time series for easier analysis
    prices_df = DataFrame(
        Time = 1:length(elec_market["price"][y]),  # Create a time index from 1 to total hours
                                                      # length() gives total elements in flattened matrix
        Elec_Price = vec(elec_market["price"][y]),  # Flattened Electricity Prices (EUR/MWh)
        H2_Price = vec(H2_market["price"][y])        # Flattened Hydrogen Prices (EUR/MWh)
    )
    
    # Write the prices DataFrame
    # File: Market_Prices.csv
    # Columns: Time, Elec_Price, H2_Price
    CSV.write(joinpath(results_folder, "Market_Prices.csv"), prices_df)

    # --- 3. SAVE AGENT RESULTS (SUMMARY) ---
    # Initialize an empty DataFrame to store summary statistics for each agent
    # Columns: Agent ID, Agent Type, Total Generation/Sales, Total Objective Value (Profit/Cost)
    summary = DataFrame(
        Agent = String[],           # Agent identifier (e.g., "Gen_VRES_01")
        Type = String[],            # Agent type (e.g., "PowerGen", "H2Prod", "Offtaker")
        Total_Generation = Float64[],  # Total generation/sales volume (MWh)
        Total_Cost = Float64[]         # Total objective value (profit for producers, cost for consumers)
    )
    
    # Iterate through every agent ID in the system
    # agents[:all] contains the union of all agent types
    for id in agents[:all]
        # Retrieve the solved JuMP model for the specific agent
        # The model contains optimal variable values and objective value
        m = mdict[id]
        
        # Initialize placeholders for agent type and generation quantity
        # These will be determined by checking which variables exist in the model
        type = "Unknown"  # Default type if agent type cannot be determined
        gen = 0.0         # Default generation if agent has no generation variable
        
        # -- LOGIC TO IDENTIFY AGENT TYPE AND CALCULATE TOTAL VOLUME --
        # CRITICAL: Check for consumers FIRST (they have :q_E as alias, but :d_E is the actual variable)
        
        # Check if agent has electricity demand variable :d_E (Power Consumer)
        # Consumers have both :d_E (actual variable) and :q_E (alias), so check :d_E first
        if haskey(m, :d_E)
            type = "PowerCons"  # Power Consumer type
            # Calculate total consumption: sum of all optimal demand values
            gen = sum(value.(m[:d_E]))
            
        # Check if agent has electricity generation variable :q_E (Power Generator)
        # Note: This will NOT match consumers because we checked :d_E first
        # This includes both VRES and Conventional generators
        elseif haskey(m, :q_E)
            type = "PowerGen"  # Power Generator type
            # Calculate total generation: sum of all optimal generation values
            # value.(m[:q_E]) extracts optimal values, sum() aggregates across all indices
            gen = sum(value.(m[:q_E]))
            
        # Check if agent has hydrogen sales variable :h_sell (Hydrogen Producer)
        # This identifies Electrolytic H2 producers
        elseif haskey(m, :h_sell)
            type = "H2Prod"  # Hydrogen Producer type
            # Calculate total hydrogen production: sum of all optimal sales values
            gen = sum(value.(m[:h_sell]))
            
        # Check if agent has end-product sales variable :ep_sell (Offtaker)
        # This includes both Green and Grey Ammonia Producers
        elseif haskey(m, :ep_sell)
            type = "Offtaker"  # Offtaker (Ammonia Producer) type
            # Calculate total End Product production: sum of all optimal sales values
            gen = sum(value.(m[:ep_sell]))
            
        # Check if agent has GC demand variable :d_GC_E (Electricity GC Demand Agent)
        elseif haskey(m, :d_GC_E)
            type = "GC_Demand"  # Electricity GC Demand Agent type
            # Calculate total GC consumption: sum of all optimal GC demand values
            gen = sum(value.(m[:d_GC_E]))
        end
        
        # Add a new row to the summary DataFrame
        # objective_value(m) returns the optimal objective function value
        # For producers: this is profit (positive)
        # For consumers: this is surplus (positive) or cost (negative, depending on formulation)
        push!(summary, (id, type, gen, objective_value(m)))
    end
    
    # Write the summary DataFrame
    # File: Agent_Summary.csv
    # Columns: Agent, Type, Total_Generation, Total_Cost
    CSV.write(joinpath(results_folder, "Agent_Summary.csv"), summary)
    
    # Print a final success message with the results folder path
    # This confirms that all files were saved successfully
    # Console prints removed to improve performance; results are written directly to CSV files.
end
