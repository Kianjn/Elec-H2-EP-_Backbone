# ==============================================================================
# BUILD POWER SECTOR AGENT MODEL
# ==============================================================================
# This function constructs the JuMP optimization model for Power Sector agents.
# It handles three types of agents:
#   1. VRES (Variable Renewable Energy Source) Generators: Solar, Wind, etc.
#   2. Conventional Generators: Gas turbines, coal plants, etc.
#   3. Electricity Consumers: Elastic demand with quadratic utility function
#
# The function defines decision variables, operational constraints, and objective
# functions according to the mathematical formulation in the LaTeX document.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Gen_VRES_01")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - elec_market::Dict: Dictionary containing electricity market parameters
#   - elec_GC_market::Dict: Dictionary containing electricity GC market parameters
#
# Returns:
#   - Modifies the JuMP model in-place by adding variables, constraints, and objective
# ==============================================================================
function build_power_agent!(agent_id::String, model::JuMP.Model, elec_market::Dict, elec_GC_market::Dict)

    # --- 1. EXTRACT SETS AND COMMON PARAMETERS ---
    # These sets define the dimensions of the optimization problem
    # They were stored in the model during the parameter definition phase
    
    # T: Time steps within a representative day (e.g., 1:24 for hourly resolution)
    # Represents the hours within a single day
    T = model[:T]
    
    # R: Representative days (e.g., 1:3 for 3 representative days)
    # These are selected days that represent the full year's variability
    R = model[:R]
    
    # Y: Years/scenarios (e.g., [2021] for a single year)
    # Allows for multi-year or multi-scenario analysis
    Y = model[:Y]
    
    # W: Representative day weights (probability/frequency of each representative day)
    # These weights sum to the number of days in the year and account for how often
    # each representative day occurs in the full year
    W = model[:W] 

    # --- 2. DEFINE ADMM PARAMETERS ---
    # These parameters are initialized here but will be updated iteratively by the ADMM solver.
    # They represent market prices (dual variables) and penalty parameters for the
    # augmented Lagrangian formulation used in the ADMM algorithm.

    # -- Electricity Market Parameters --
    # Initialize the local electricity price parameter (lambda_E) with the global market price
    # lambda_E represents the dual variable (Lagrange multiplier) for electricity market clearing
    # It is a dictionary keyed by year, with each value being a matrix (nTimesteps × nReprDays)
    model[:lambda_E] = Dict(y => elec_market["price"][y] for y in Y)
    
    # Initialize the ADMM penalty factor (rho) for the electricity market
    # This parameter controls the strength of the penalty for deviating from market consensus
    # Higher rho = stronger penalty = faster constraint satisfaction
    model[:rho_E] = model[:rho]
    
    # Initialize the reference quantity (E_ref) for ADMM tracking (starts at 0)
    # In Exchange ADMM, references are always zero, so penalty = (rho/2) * variable^2
    # This is different from standard ADMM where references track consensus values
    model[:E_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # -- Electricity Green Certificate (GC) Market Parameters --
    # Initialize the local GC price parameter
    # lambda_GC_E represents the price premium for green-certified electricity
    # VRES generators receive this premium in addition to electricity price
    model[:lambda_GC_E] = Dict(y => elec_GC_market["price"][y] for y in Y)
    
    # Initialize the ADMM penalty factor for the GC market
    # Controls penalty strength for GC market consensus
    model[:rho_GC_E] = model[:rho]
    
    # Initialize the reference quantity for GC tracking (starts at 0)
    # Exchange ADMM formulation: references are zero
    model[:GC_E_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # --- 3. BUILD VARIABLES AND OBJECTIVES BASED ON AGENT TYPE ---
    # The agent type is determined by checking which parameters exist in the model
    # This allows flexible agent configuration without explicit type flags

    # ==========================================================================
    # CASE A: ELECTRICITY CONSUMER
    # ==========================================================================
    # Identified by the presence of a Demand Profile (:D_bar)
    # This agent has elastic demand with a quadratic utility function
    # Utility = A_E * d - (1/2) * B_E * d^2, where d is consumption
    # ==========================================================================
    if haskey(model, :D_bar)
        # -- DECISION VARIABLES --
        # Define the electricity consumption variable d_E (must be non-negative)
        # The consumer chooses how much electricity to consume at each time step
        # This is indexed by time (t), representative day (r), and year (y)
        @variable(model, 0 <= d_E[t=T, r=R, y=Y])
        
        # -- OPERATIONAL CONSTRAINTS --
        # Limit consumption to the defined demand profile (D_bar)
        # D_bar represents the maximum physical/contractual demand at each time step
        # If utility is high (inelastic demand), d_E will equal D_bar
        # If utility is lower (elastic demand), d_E can be less than D_bar
        # This allows for demand response behavior
        @constraint(model, demand_limit[t=T, r=R, y=Y], d_E[t,r,y] <= model[:D_bar][y][t,r])

        # -- UTILITY FUNCTION PARAMETERS --
        # Safe Parameter Access: Get quadratic utility parameters or use defaults
        # A_E: Intercept of the inverse demand curve (willingness to pay intercept)
        #      Higher A_E = higher willingness to pay = more inelastic demand
        # B_E: Slope of the inverse demand curve (price sensitivity)
        #      Higher B_E = more price-sensitive = more elastic demand
        # Default values are typical constants if not specified in configuration
        A_E = haskey(model, :A_E) ? model[:A_E] : 500.0  # Typical intercept (EUR/MWh)
        B_E = haskey(model, :B_E) ? model[:B_E] : 0.5    # Typical slope (EUR/MWh²)

        # -- OBJECTIVE FUNCTION --
        # Maximize: Quadratic Utility - Cost - ADMM Penalty
        # The consumer maximizes their surplus (utility minus cost)
        # Utility function: U(d) = A_E * d - (1/2) * B_E * d^2
        # This is a concave quadratic function representing diminishing marginal utility
        @expression(model, obj_base,
            sum(W[y][r] * (
                # (+) Linear utility term: A_E * d
                # Represents the base willingness to pay per unit
                (A_E * d_E[t,r,y]) -
                # (-) Quadratic utility term: (1/2) * B_E * d^2
                # Represents diminishing marginal utility (concavity)
                (0.5 * B_E * d_E[t,r,y]^2) -
                # (-) Cost: Price * Quantity
                # Market price of electricity times quantity consumed
                (model[:lambda_E][y][t,r] * d_E[t,r,y]) -
                # (-) ADMM Quadratic Penalty
                # Penalizes deviations from market consensus
                # Penalty = (rho/2) * (variable - reference)^2
                # Since reference is zero in Exchange ADMM: penalty = (rho/2) * variable^2
                (model[:rho_E] / 2 * (d_E[t,r,y] - model[:E_ref][y][t,r])^2)
            ) for t in T, r in R, y in Y)
        )
        # Set the maximization objective for the optimization model
        @objective(model, Max, obj_base)
        
        # Create an alias 'q_E' pointing to d_E for generic access in the solver loop
        # This allows market balancing functions to access consumption using :q_E
        # regardless of whether it's generation (q_E) or consumption (d_E)
        model[:q_E] = d_E 

    # ==========================================================================
    # CASE B: GENERATOR (VRES OR CONVENTIONAL)
    # ==========================================================================
    # Identified by the presence of a Capacity Limit (:Q_bar)
    # Sub-type is determined by whether Q_bar is a Dictionary (VRES) or scalar (Conventional)
    # ==========================================================================
    elseif haskey(model, :Q_bar)
        # -- DECISION VARIABLES --
        # Define the electricity generation variable q_E (must be non-negative)
        # The generator chooses how much electricity to produce at each time step
        # This is indexed by time (t), representative day (r), and year (y)
        @variable(model, q_E[t=T, r=R, y=Y] >= 0)

        # -- CHECK SUB-TYPE: VRES vs CONVENTIONAL --
        # If Q_bar is a Dictionary, it implies time-dependent capacity (VRES like Solar/Wind)
        # VRES have variable capacity based on weather conditions (solar irradiance, wind speed)
        # Conventional generators have constant capacity (scalar Q_bar)
        if isa(model[:Q_bar], Dict) 
            
            # ==================================================================
            # SUB-CASE B1: VRES GENERATOR (Variable Renewable Energy Source)
            # ==================================================================
            # Examples: Solar PV, Wind turbines
            # Characteristics:
            #   - Time-dependent capacity (varies with weather)
            #   - Produces both electricity and green certificates
            #   - Low/zero marginal cost
            # ==================================================================
            
            # -- VRES CONSTRAINTS --
            # Limit generation to the time-dependent profile (Capacity * Normalized Profile)
            # Q_bar[y][t,r] represents the maximum available capacity at time t, day r, year y
            # This is calculated as: Installed_Capacity × Normalized_Profile[t,r]
            # The normalized profile comes from time series data (e.g., solar irradiance)
            @constraint(model, cap_limit[t=T, r=R, y=Y], q_E[t,r,y] <= model[:Q_bar][y][t,r])
            
            # -- VRES OBJECTIVE FUNCTION --
            # VRES sells BOTH Electricity and Green Certificates (1 MWh Elec = 1 GC)
            # Revenue sources:
            #   1. Electricity market: p^E * q_E
            #   2. GC market: p^{GC,E} * q_E (since gc_E = q_E for VRES)
            # Costs:
            #   1. Variable marginal cost: c_V * q_E (typically very low for renewables)
            #   2. ADMM penalties for market consensus
            @expression(model, obj_vres,
                sum(W[y][r] * (
                    # (+) Electricity Revenue
                    # Market price of electricity times quantity generated
                    (model[:lambda_E][y][t,r] * q_E[t,r,y]) +
                    # (+) Green Certificate Revenue
                    # Price premium for green-certified electricity
                    # Note: For VRES, gc_E = q_E (1 MWh electricity = 1 GC)
                    (model[:lambda_GC_E][y][t,r] * q_E[t,r,y]) -
                    # (-) Variable Marginal Cost
                    # Typically very low for renewables (near zero)
                    # Represents variable O&M costs
                    (model[:C] * q_E[t,r,y]) -
                    # (-) Electricity ADMM Penalty
                    # Penalizes deviations from electricity market consensus
                    (model[:rho_E] / 2 * (q_E[t,r,y] - model[:E_ref][y][t,r])^2) -
                    # (-) GC ADMM Penalty
                    # Penalizes deviations from GC market consensus
                    (model[:rho_GC_E] / 2 * (q_E[t,r,y] - model[:GC_E_ref][y][t,r])^2)
                ) for t in T, r in R, y in Y)
            )
            # Set the maximization objective
            @objective(model, Max, obj_vres)
            
            # Create an alias 'gc_E' for Green Certificates (equal to generation q_E)
            # This allows market balancing to track GC production
            # For VRES: 1 MWh electricity = 1 Green Certificate
            model[:gc_E] = q_E 

        else 
            # ==================================================================
            # SUB-CASE B2: CONVENTIONAL GENERATOR
            # ==================================================================
            # Examples: Gas turbines, coal plants, nuclear
            # Characteristics:
            #   - Constant capacity (scalar Q_bar)
            #   - Produces only electricity (no green certificates)
            #   - Higher marginal cost (fuel + CO2 costs)
            # ==================================================================
            
            # -- CONVENTIONAL GENERATOR CONSTRAINTS --
            # Limit generation to the constant installed capacity (scalar Q_bar)
            # Unlike VRES, conventional generators have fixed capacity that doesn't vary with time
            # Q_bar is a scalar value representing the nameplate capacity (MW)
            @constraint(model, cap_limit[t=T, r=R, y=Y], q_E[t,r,y] <= model[:Q_bar])
            
            # -- CONVENTIONAL GENERATOR OBJECTIVE FUNCTION --
            # Maximize: Elec Revenue - Variable Cost - ADMM Penalty
            # Revenue sources:
            #   1. Electricity market: p^E * q_E
            # Costs:
            #   1. Variable marginal cost: c_C * q_E (fuel + CO2 costs)
            #   2. ADMM penalties for market consensus
            # Note: No GC revenue (conventional generators don't produce green certificates)
            @expression(model, obj_conv,
                sum(W[y][r] * (
                    # (+) Electricity Revenue
                    # Market price of electricity times quantity generated
                    (model[:lambda_E][y][t,r] * q_E[t,r,y]) -
                    # (-) Variable Marginal Cost
                    # Includes fuel costs and CO2 emission costs
                    # Typically higher than VRES marginal costs
                    (model[:C] * q_E[t,r,y]) -
                    # (-) Electricity ADMM Penalty
                    # Penalizes deviations from electricity market consensus
                    (model[:rho_E] / 2 * (q_E[t,r,y] - model[:E_ref][y][t,r])^2)
                ) for t in T, r in R, y in Y)
            )
            # Set the maximization objective
            @objective(model, Max, obj_conv)
        end
        
        # Create an alias 'q_E' for generic access
        # This allows market balancing functions to access generation using :q_E
        model[:q_E] = q_E 
    end

    # Print confirmation message for debugging and progress tracking
    println("Built Power Agent: $agent_id")
end
