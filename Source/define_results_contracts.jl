# ==============================================================================
# define_results_contracts.jl — Initialize results and ADMM state (with contracts)
# ==============================================================================
#
# PURPOSE:
#   Extends define_results! with the bilateral contract pool. Used ONLY by
#   market_exposure_contracts.jl.
#
#   SINGLE CONTRACT POOL: g_contract cleared at λ_contract (€/MWh), pay-as-produced.
#   contract_cap is a consensus variable (both agree via penalty); no separate price.
#
#   Adds: results["contract"], results["contract_cap"], results["λ"]["contract"],
#   ADMM state for contract (price) and contract_cap (consensus only).
#
# ==============================================================================

function define_results_contracts!(admm_data::Dict, results::Dict, ADMM::Dict, agents::Dict,
                                  elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                  H2_GC_market::Dict, EP_market::Dict, contract_market::Dict;
                                  sp_prices_file::String = "")
    # Call base define_results! for the five standard markets
    define_results!(admm_data, results, ADMM, agents, elec_market, H2_market,
                    elec_GC_market, H2_GC_market, EP_market; sp_prices_file=sp_prices_file)

    n_ts = admm_data["nTimesteps"]
    n_rd = admm_data["nReprDays"]
    n_yr = admm_data["nYears"]
    shp  = (n_ts, n_rd, n_yr)

    # Add contract market to results["markets"]
    results["markets"]["contract"] = contract_market

    # Contract pool: one price λ_contract (€/MWh) for g_contract. No separate capacity price.
    init_price = contract_market["initial_price"]
    results["λ"]["contract"] = [fill(init_price, shp...)]

    # Per-agent quantity buffers for contract markets
    results["contract"]     = Dict(m => [] for m in agents[:all])
    results["contract_cap"] = Dict(m => [] for m in agents[:all])

    # ADMM state for contract markets
    ADMM["ρ"]["contract"]     = [contract_market["rho_initial"]]
    ADMM["ρ"]["contract_cap"] = [contract_market["rho_initial"]]

    ADMM["Imbalances"]["contract"]     = Any[]
    ADMM["Imbalances"]["contract_cap"] = Any[]

    ADMM["PriceHistory"]["contract"] = Float64[]

    ADMM["ImbalanceMean"]["contract"]     = Float64[]
    ADMM["ImbalanceMean"]["contract_cap"] = Float64[]

    ADMM["Residuals"]["Primal"]["contract"]     = Float64[]
    ADMM["Residuals"]["Primal"]["contract_cap"] = Float64[]
    ADMM["Residuals"]["Dual"]["contract"]       = Float64[]
    ADMM["Residuals"]["Dual"]["contract_cap"]   = Float64[]

    ADMM["BestResidual"]["Primal"]["contract"]     = Inf
    ADMM["BestResidual"]["Dual"]["contract"]       = Inf
    ADMM["BestResidual"]["Primal"]["contract_cap"] = Inf
    ADMM["BestResidual"]["Dual"]["contract_cap"]   = Inf

    ADMM["ρ_frozen"]["contract"]     = false
    ADMM["ρ_frozen"]["contract_cap"] = false

    ADMM["R_hist"]["contract"]     = Float64[]
    ADMM["R_hist"]["contract_cap"] = Float64[]

    ADMM["ResidualScale"]["Primal"]["contract"]     = 0.0
    ADMM["ResidualScale"]["Primal"]["contract_cap"] = 0.0
    ADMM["ResidualScale"]["Dual"]["contract"]       = 0.0
    ADMM["ResidualScale"]["Dual"]["contract_cap"]   = 0.0

    base_tol = get(admm_data, "epsilon", 1.0)
    ADMM["Tolerance"]["contract"]     = base_tol
    ADMM["Tolerance"]["contract_cap"] = base_tol

    return results, ADMM
end
