# ==============================================================================
# BUILD ELECTRICITY GC DEMAND AGENT MODEL
# ==============================================================================
# This function constructs the JuMP optimization model for Electricity Green
# Certificate (GC) Demand Agents. These agents have a demand for green certificates
# with a quadratic utility function, representing environmental preferences or
# regulatory compliance requirements.
#
# The agent maximizes surplus from consuming GCs: Utility - Cost - ADMM Penalties
# Utility function: U(d) = A_GC * d - (1/2) * B_GC * d^2
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent (e.g., "Demand_GC_Elec_01")
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - elec_GC_market::Dict: Dictionary containing electricity GC market parameters
#
# Returns:
#   - Modifies the JuMP model in-place by adding variables, constraints, and objective
# ==============================================================================
function build_elec_GC_demand_agent!(agent_id::String, model::JuMP.Model, elec_GC_market::Dict)

    # --- 1. EXTRACT SETS AND COMMON PARAMETERS ---
    # These sets define the dimensions of the optimization problem
    T = model[:T]  # Time steps (e.g., 1:24)
    R = model[:R]  # Representative days (e.g., 1:3)
    Y = model[:Y]  # Years/scenarios (e.g., [2021])
    W = model[:W]  # Representative day weights (probabilities/frequencies)

    # --- 2. DEFINE ADMM PARAMETERS ---
    # These parameters are placeholders that will be updated iteratively by the ADMM solver.
    
    # Initialize Elec GC prices from the global market
    # lambda_GC_E represents the dual variable (Lagrange multiplier) for GC market clearing
    model[:lambda_GC_E] = Dict(y => elec_GC_market["price"][y] for y in Y)
    
    # Initialize the penalty factor for Elec GC market coupling
    # Controls the strength of the penalty for deviating from market consensus
    model[:rho_GC_E] = model[:rho]
    
    # Initialize the reference quantity for Elec GC ADMM tracking
    # Exchange ADMM: references are always zero
    model[:GC_E_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # --- 3. BUILD DECISION VARIABLES ---
    # d_GC_E: GC demand variable (Certificates)
    # The agent chooses how many green certificates to purchase at each time step
    # This is indexed by time (t), representative day (r), and year (y)
    @variable(model, 0 <= d_GC_E[t=T, r=R, y=Y])

    # --- 4. OPERATIONAL CONSTRAINTS ---
    # Limit demand to maximum green target
    # D_GC_E_bar: Maximum GC demand at each time step (from demand profile)
    # This represents the agent's maximum green certificate requirement/target
    @constraint(model, [t=T, r=R, y=Y], d_GC_E[t,r,y] <= model[:D_GC_E_bar][y][t,r])

    # --- 5. OBJECTIVE FUNCTION ---
    # Maximize: Quadratic Utility - Cost - ADMM Penalty
    # The agent maximizes their surplus from consuming green certificates
    # Utility function: U(d) = A_GC * d - (1/2) * B_GC * d^2
    # This is a concave quadratic function representing diminishing marginal utility
    @expression(model, obj_gc_demand,
        sum(W[y][r] * (
            # (+) Linear utility term: A_GC * d
            # Represents the base willingness to pay per certificate
            # A_GC: Intercept of inverse demand curve (EUR/Certificate)
            (model[:A_GC] * d_GC_E[t,r,y]) -
            # (-) Quadratic utility term: (1/2) * B_GC * d^2
            # Represents diminishing marginal utility (concavity)
            # B_GC: Slope of inverse demand curve (EUR/Certificate²)
            (0.5 * model[:B_GC] * d_GC_E[t,r,y]^2) -
            # (-) Cost of GCs
            # Market price of green certificates times quantity purchased
            (model[:lambda_GC_E][y][t,r] * d_GC_E[t,r,y]) -
            # (-) ADMM Penalty
            # Penalizes deviations from market consensus
            # Penalty = (rho/2) * (variable - reference)^2
            # Since reference is zero in Exchange ADMM: penalty = (rho/2) * variable^2
            (model[:rho_GC_E] / 2 * (d_GC_E[t,r,y] - model[:GC_E_ref][y][t,r])^2)
        ) for t in T, r in R, y in Y)
    )
    # Set the maximization objective for the optimization model
    @objective(model, Max, obj_gc_demand)
    
    # Create alias for generic access in ADMM updates and market balancing
    model[:d_GC_E] = d_GC_E

    # Print confirmation message for debugging and progress tracking
    println("Built Electricity GC Demand Agent: $agent_id")
end
