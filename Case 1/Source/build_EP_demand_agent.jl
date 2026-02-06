# ==============================================================================
# BUILD END PRODUCT (EP) DEMAND AGENT MODEL
# ==============================================================================
# This agent represents elastic demand for ammonia (End Product) with a quadratic
# utility function:
#   U(d) = A_EP * d - 0.5 * B_EP * d^2
# The agent maximizes surplus: Utility - Cost - ADMM Penalty.
# ==============================================================================
function build_EP_demand_agent!(agent_id::String, model::JuMP.Model, EP_market::Dict)

    # --- 1. EXTRACT SETS AND COMMON PARAMETERS ---
    T = model[:T]
    R = model[:R]
    Y = model[:Y]
    W = model[:W]

    # --- 2. DEFINE ADMM PARAMETERS ---

    # EP shadow prices
    model[:lambda_EP] = Dict(y => EP_market["price"][y] for y in Y)
    model[:rho_EP] = EP_market["rho"]
    model[:EP_ref] = Dict(y => zeros(length(T), length(R)) for y in Y)

    # --- 3. DECISION VARIABLES ---

    # d_EP: EP demand (MWh)
    @variable(model, 0 <= d_EP[t=T, r=R, y=Y])

    # --- 4. CONSTRAINTS ---

    # Limit demand to maximum profile
    @constraint(model, [t=T, r=R, y=Y], d_EP[t,r,y] <= model[:D_EP_bar][y][t,r])

    # --- 5. OBJECTIVE FUNCTION ---

    @expression(model, obj_EP_demand,
        sum(W[y][r] * (
            # (+) Utility
            (model[:A_EP] * d_EP[t,r,y]) -
            0.5 * model[:B_EP] * d_EP[t,r,y]^2 -
            # (-) Cost of EP at market shadow price
            (model[:lambda_EP][y][t,r] * d_EP[t,r,y]) -
            # (-) ADMM penalty
            (model[:rho_EP] / 2 * (d_EP[t,r,y] - model[:EP_ref][y][t,r])^2)
        ) for t in T, r in R, y in Y)
    )
    @objective(model, Max, obj_EP_demand)

    # Alias for market balancing
    model[:d_EP] = d_EP
end

