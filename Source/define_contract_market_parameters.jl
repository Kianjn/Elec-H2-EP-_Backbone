# ==============================================================================
# define_contract_market_parameters.jl — Bilateral contract pool parameters
# ==============================================================================
#
# PURPOSE:
#   Initializes the bilateral contract pool between VRES and the electrolyzer.
#   Used ONLY by market_exposure_contracts.jl.
#
#   SINGLE CONTRACT POOL: They contract for capacity (MW); payment is pay-as-produced
#   at λ_contract (€/MWh) for energy actually delivered. g_contract[jh,jd,jy] is
#   the delivery per timestep (≤ contract_cap). When VRES has no output (e.g.
#   night for solar), g_contract = 0, so nothing is delivered and nothing is paid.
#   No separate capacity price — only λ_contract for the delivered energy.
#
# ARGUMENTS:
#   market — Dict to be filled with initial_price, rho_initial, etc.
#   data   — Merged General + ADMM + Contracts block from data.yaml.
#
# ==============================================================================

function define_contract_market_parameters!(market::Dict, data::Dict)
    market["name"]          = "Bilateral_Contract"
    market["initial_price"] = get(data, "initial_price", 60.0)
    market["rho_initial"]   = get(data, "rho_initial", 0.5)
    market["prices"]        = [data["initial_price"]]

    # 3D shape for price array (matches other markets: hours × repr days × years)
    n_ts = data["nTimesteps"]
    n_rd = data["nReprDays"]
    n_yr = data["nYears"]
    shp  = (n_ts, n_rd, n_yr)
    market["shape"] = shp
    return market
end
