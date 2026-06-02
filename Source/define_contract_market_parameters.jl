# ==============================================================================
# define_contract_market_parameters.jl — Bilateral contract pool parameters
# ==============================================================================
#
# PURPOSE:
#   Initializes the bilateral contract pool between VRES and the electrolyzer.
#   Used ONLY by market_exposure_contracts.jl.
#
#   PER-VRES PPA MARKETS: Each VRES has its own bilateral PPA sub-market with
#   GreenProducer side. Payment is pay-as-produced at λ_ppa[vres_id] (€/MWh).
#   PER-H2-PRODUCER HPA MARKETS: Each GreenProducer has its own bilateral HPA
#   sub-market with GreenOfftaker side. Payment is pay-as-produced at λ_hpa[h2_id].
#
# ARGUMENTS:
#   market — Dict to be filled with initial_price, rho_initial, per_vres, etc.
#   data   — Full data dict (for Power and Contracts).
#   agents — Dict; agents[:power] used to identify VRES (Type from data["Power"]).
#
# ==============================================================================

function define_contract_market_parameters!(market::Dict, hpa_market::Dict, data::Dict, agents::Dict)
    _to_float(x, default=0.0) = begin
        if x === nothing
            return default
        end
        if x isa Number
            return Float64(x)
        end
        try
            return parse(Float64, string(x))
        catch
            return default
        end
    end
    _parse_affine_terms(cfg::Dict) = Dict(
        "constant" => _to_float(get(cfg, "constant", 0.0), 0.0),
        "elec" => _to_float(get(cfg, "elec", 0.0), 0.0),
        "elec_GC" => _to_float(get(cfg, "elec_GC", 0.0), 0.0),
        "H2" => _to_float(get(cfg, "H2", 0.0), 0.0),
        "H2_GC" => _to_float(get(cfg, "H2_GC", 0.0), 0.0),
        "EP" => _to_float(get(cfg, "EP", 0.0), 0.0)
    )
    _parse_pricing(cfg::Dict, default_price::Float64; parent::Dict=Dict()) = begin
        mode = String(get(cfg, "pricing_mode", get(parent, "pricing_mode", "endogenous_clearing")))
        parent_terms = get(parent, "index_terms", Dict())
        local_terms = get(cfg, "index_terms", Dict())
        Dict(
            "pricing_mode" => mode,
            "fixed_strike" => _to_float(get(cfg, "fixed_strike", get(parent, "fixed_strike", default_price)), default_price),
            "index_terms" => _parse_affine_terms(merge(parent_terms, local_terms)),
            "price_floor" => _to_float(get(cfg, "price_floor", get(parent, "price_floor", 0.0)), 0.0),
            "price_cap" => _to_float(get(cfg, "price_cap", get(parent, "price_cap", 1.0e9)), 1.0e9)
        )
    end

    ppa_data = haskey(data, "PPAs") ? merge(data["General"], data["ADMM"], data["PPAs"]) : merge(data["General"], data["ADMM"], Dict("initial_price" => 60.0, "rho_initial" => 0.5))

    market["name"]          = "Bilateral_Contract"
    market["initial_price"] = get(ppa_data, "initial_price", 60.0)
    market["rho_initial"]   = get(ppa_data, "rho_initial", 0.5)
    market["pricing_mode"]  = String(get(ppa_data, "pricing_mode", "endogenous_clearing"))
    market["fixed_strike"]  = _to_float(get(ppa_data, "fixed_strike", market["initial_price"]), market["initial_price"])
    market["index_terms"]   = _parse_affine_terms(get(ppa_data, "index_terms", Dict()))
    market["price_floor"]   = _to_float(get(ppa_data, "price_floor", 0.0), 0.0)
    market["price_cap"]     = _to_float(get(ppa_data, "price_cap", 1.0e9), 1.0e9)

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
        merge!(market["per_vres"][vres_id], _parse_pricing(
            market["per_vres"][vres_id],
            _to_float(market["per_vres"][vres_id]["initial_price"], market["initial_price"]);
            parent=market
        ))
    end

    hpa_data = haskey(data, "HPAs") ? merge(data["General"], data["ADMM"], data["HPAs"]) : merge(data["General"], data["ADMM"], Dict("initial_price" => 60.0, "rho_initial" => 0.5))

    hpa_market["name"]          = "Hydrogen_Bilateral_Contract"
    hpa_market["initial_price"] = get(hpa_data, "initial_price", 60.0)
    hpa_market["rho_initial"]   = get(hpa_data, "rho_initial", 0.5)
    hpa_market["pricing_mode"]  = String(get(hpa_data, "pricing_mode", "endogenous_clearing"))
    hpa_market["fixed_strike"]  = _to_float(get(hpa_data, "fixed_strike", hpa_market["initial_price"]), hpa_market["initial_price"])
    hpa_market["index_terms"]   = _parse_affine_terms(get(hpa_data, "index_terms", Dict()))
    hpa_market["price_floor"]   = _to_float(get(hpa_data, "price_floor", 0.0), 0.0)
    hpa_market["price_cap"]     = _to_float(get(hpa_data, "price_cap", 1.0e9), 1.0e9)
    hpa_market["shape"]         = shp

    # Per-H2-producer sub-markets for HPA contracts with green offtakers
    h2_data = get(data, "Hydrogen", Dict())
    contract_h2 = [id for id in get(agents, :H2, []) if String(get(get(h2_data, id, Dict()), "Type", "")) == "GreenProducer"]
    hpa_market["per_h2"] = Dict{String, Dict}()
    for h2_id in contract_h2
        h2_block = get(get(data, "HPAs", Dict()), h2_id, Dict())
        if isa(h2_block, Dict) && haskey(h2_block, "initial_price")
            hpa_market["per_h2"][h2_id] = Dict(
                "initial_price" => h2_block["initial_price"],
                "rho_initial"   => get(h2_block, "rho_initial", hpa_market["rho_initial"]),
            )
        else
            hpa_market["per_h2"][h2_id] = Dict(
                "initial_price" => hpa_market["initial_price"],
                "rho_initial"   => hpa_market["rho_initial"],
            )
        end
        merge!(hpa_market["per_h2"][h2_id], _parse_pricing(
            hpa_market["per_h2"][h2_id],
            _to_float(hpa_market["per_h2"][h2_id]["initial_price"], hpa_market["initial_price"]);
            parent=hpa_market
        ))
    end

    return market, hpa_market
end
