# ==============================================================================
# define_contract_market_parameters.jl — Bilateral contract pool parameters
# ==============================================================================
#
# PPAs (all ME contract entry points): pay-as-produced volume; scalar strike K
# (W-weighted mean of bilateral λ_ppa at convergence; uniform over hours).
#
# HPAs: volume mode set by entry point (me_pap / me_top / me_sop). Price options
# from data.yaml: price_structure (fixed | cfd) and price_benchmark.
#
# ==============================================================================

function define_contract_market_parameters!(market::Dict, hpa_market::Dict, data::Dict, agents::Dict)
    _to_float(x, default=0.0) = begin
        x === nothing && return default
        x isa Number && return Float64(x)
        try
            return parse(Float64, string(x))
        catch
            return default
        end
    end

    _parse_contract_cfg(cfg::Dict, parent::Dict, default_price::Float64) = begin
        bench = String(get(cfg, "price_benchmark",
            get(cfg, "pricing_mode", get(parent, "price_benchmark",
                get(parent, "pricing_mode", "negotiated")))))
        bench = bench == "endogenous_clearing" ? "negotiated" :
                bench == "indexed" ? get(cfg, "index_to", "electricity") :
                bench == "fixed" ? "negotiated" : bench
        Dict(
            "price_structure" => String(get(cfg, "price_structure", get(parent, "price_structure", "fixed"))),
            "price_benchmark" => lowercase(strip(bench)),
            "initial_price" => _to_float(get(cfg, "initial_price", default_price), default_price),
            "rho_initial" => _to_float(get(cfg, "rho_initial", get(parent, "rho_initial", 0.5)), 0.5),
            "price_floor" => _to_float(get(cfg, "price_floor", get(parent, "price_floor", 0.0)), 0.0),
            "price_cap" => _to_float(get(cfg, "price_cap", get(parent, "price_cap", 1.0e9)), 1.0e9),
        )
    end

    ppa_data = haskey(data, "PPAs") ? merge(data["General"], data["ADMM"], data["PPAs"]) :
        merge(data["General"], data["ADMM"], Dict("initial_price" => 60.0, "rho_initial" => 0.5))

    market["name"] = "Bilateral_PPA"
    market["volume_mode"] = "sop"
    market["price_structure"] = "fixed"
    market["price_benchmark"] = "negotiated"
    market["initial_price"] = get(ppa_data, "initial_price", 60.0)
    market["rho_initial"] = get(ppa_data, "rho_initial", 0.5)
    market["contract_warmstart_from_spot"] = false

    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    market["shape"] = shp

    power_data = get(data, "Power", Dict())
    contract_vres = [id for id in get(agents, :power, []) if
                     String(get(get(power_data, id, Dict()), "Type", "")) == "VRES"]
    market["per_vres"] = Dict{String, Dict}()
    for vres_id in contract_vres
        vres_block = get(get(data, "PPAs", Dict()), vres_id, Dict())
        init_p = haskey(vres_block, "initial_price") ? vres_block["initial_price"] : market["initial_price"]
        market["per_vres"][vres_id] = _parse_contract_cfg(
            isa(vres_block, Dict) ? vres_block : Dict(),
            merge(ppa_data, Dict("price_structure" => "fixed", "price_benchmark" => "negotiated")),
            _to_float(init_p, market["initial_price"]))
        market["per_vres"][vres_id]["initial_price"] = _to_float(init_p, market["initial_price"])
    end

    hpa_data = haskey(data, "HPAs") ? merge(data["General"], data["ADMM"], data["HPAs"]) :
        merge(data["General"], data["ADMM"], Dict("initial_price" => 80.0, "rho_initial" => 0.5))

    hpa_market["name"] = "Hydrogen_Bilateral_HPA"
    raw_mode = lowercase(String(get(agents, :hpa_volume_mode, "sop")))
    hpa_market["volume_mode"] = raw_mode == "pap" ? "sop" : raw_mode
    hpa_market["volume_mode"] in ("sop", "top") || error("Unsupported hpa_volume_mode=$(raw_mode). Use sop or top.")
    hpa_market["price_structure"] = String(get(hpa_data, "price_structure", "fixed"))
    hpa_market["price_benchmark"] = lowercase(String(get(hpa_data, "price_benchmark", "negotiated")))
    hpa_market["initial_price"] = get(hpa_data, "initial_price", 80.0)
    hpa_market["rho_initial"] = get(hpa_data, "rho_initial", 0.5)
    hpa_market["shape"] = shp

    h2_data = get(data, "Hydrogen", Dict())
    contract_h2 = [id for id in get(agents, :H2, []) if
                   String(get(get(h2_data, id, Dict()), "Type", "")) == "GreenProducer"]
    hpa_market["per_h2"] = Dict{String, Dict}()
    for h2_id in contract_h2
        h2_block = get(get(data, "HPAs", Dict()), h2_id, Dict())
        init_p = haskey(h2_block, "initial_price") ? h2_block["initial_price"] : hpa_market["initial_price"]
        hpa_market["per_h2"][h2_id] = _parse_contract_cfg(
            isa(h2_block, Dict) ? h2_block : Dict(), hpa_data,
            _to_float(init_p, hpa_market["initial_price"]))
        hpa_market["per_h2"][h2_id]["initial_price"] = _to_float(init_p, hpa_market["initial_price"])
    end

    return market, hpa_market
end
