# ==============================================================================
# SOLVE END PRODUCT (EP) DEMAND AGENT SUBPROBLEM
# ==============================================================================
function solve_EP_demand_agent!(agent_id::String, model::JuMP.Model, EP_market::Dict, admm_data::Dict)

    # Update EP prices and rho from market
    Y = model[:Y]
    for y in Y
        model[:lambda_EP][y] = EP_market["price"][y]
        model[:EP_ref][y] .= 0.0
    end
    model[:rho_EP] = EP_market["rho"]

    # Re-optimize the model
    optimize!(model)
end

