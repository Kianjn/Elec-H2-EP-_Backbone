# ==============================================================================
# define_contract_market_parameters.jl — Bilateral contract pool parameters
# ==============================================================================
#
# PURPOSE:
#   Initializes the bilateral contract pool between VRES and the electrolyzer.
#   Used ONLY by market_exposure_contracts.jl.
#
#   PER-VRES CONTRACT MARKETS: Each VRES has its own bilateral contract sub-market
#   with the electrolyzer. Payment is pay-as-produced at λ_contract[vres_id] (€/MWh).
#   Per-VRES initial prices from Contracts.{VRES_ID}.initial_price or default.
#
# ARGUMENTS:
#   market — Dict to be filled with initial_price, rho_initial, per_vres, etc.
#   data   — Full data dict (for Power and Contracts).
#   agents — Dict; agents[:power] used to identify VRES (Type from data["Power"]).
#
# ==============================================================================

function define_contract_market_parameters!(market::Dict, data::Dict, agents::Dict)
    ppa_data = haskey(data, "PPAs") ? merge(data["General"], data["ADMM"], data["PPAs"]) : merge(data["General"], data["ADMM"], Dict("initial_price" => 60.0, "rho_initial" => 0.5))

    market["name"]          = "Bilateral_Contract"
    market["initial_price"] = get(ppa_data, "initial_price", 60.0)
    market["rho_initial"]   = get(ppa_data, "rho_initial", 0.5)

    # 3D shape for price array (matches other markets: hours × repr days × years)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp  = (n_ts, n_rd, n_yr)
    market["shape"] = shp

    # Per-VRES contract sub-markets: VRES IDs from Power block
    power_data = get(data, "Power", Dict())
    contract_vres = [id for id in get(agents, :power, []) if String(get(get(power_data, id, Dict()), "Type", "")) == "VRES"]
    market["per_vres"] = Dict{String, Dict}()
    for vres_id in contract_vres
        vres_block = get(get(data, "PPAs", Dict()), vres_id, Dict())
        if isa(vres_block, Dict) && haskey(vres_block, "initial_price")
            market["per_vres"][vres_id] = Dict(
                "initial_price" => vres_block["initial_price"],
                "rho_initial"   => get(vres_block, "rho_initial", market["rho_initial"]),
            )
        else
            market["per_vres"][vres_id] = Dict(
                "initial_price" => market["initial_price"],
                "rho_initial"   => market["rho_initial"],
            )
        end
    end

    return market
end
