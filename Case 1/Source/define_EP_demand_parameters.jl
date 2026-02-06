# ==============================================================================
# DEFINE END PRODUCT (EP) DEMAND AGENT PARAMETERS
# ==============================================================================
# This function defines parameters specific to EP demand agents. These agents
# represent elastic demand for ammonia (End Product) with a quadratic utility
# function, similar to the electricity GC demand agents.
#
# Arguments:
#   - agent_id::String: Unique identifier for the agent
#   - model::JuMP.Model: JuMP optimization model container for this agent
#   - data::Dict: Agent-specific configuration dictionary from data.yaml
#   - ts::Dict: Time series data (for demand profile)
#   - repr_days::Dict: Representative days data (for mapping to full year)
# ==============================================================================
function define_EP_demand_parameters!(agent_id::String, model::JuMP.Model, data::Dict, ts::Dict, repr_days::Dict)

    # Extract global sets from model (already defined by define_common_parameters!)
    T = model[:T]
    R = model[:R]
    Y = model[:Y]

    # Store utility parameters (very steep to approximate inelastic baseline)
    # Utility: U(d) = A_EP * d - 0.5 * B_EP * d^2
    model[:A_EP] = data["A_EP"]
    model[:B_EP] = data["B_EP"]

    # Build maximum demand profile (per hour, per representative day, per year)
    model[:D_EP_bar] = Dict{Int, Array{Float64,2}}()

    for y in Y
        nTimesteps = length(T)
        nReprDays = length(R)

        col_name = Symbol(get(data, "Load_Column", "LOAD_EP"))
        peak_load = get(data, "PeakLoad", 10.0)

        model[:D_EP_bar][y] = [
            peak_load * ts[y][!, col_name][round(Int, nTimesteps*(repr_days[y][!,:periods][jd]-1) + jh)]
            for jh in 1:nTimesteps, jd in 1:nReprDays
        ]
    end
end

