# ==============================================================================
# define_results_contracts.jl — Initialize results and ADMM state (with contracts)
# ==============================================================================
#
# PURPOSE:
#   Extends define_results! with the bilateral contract pool. Used ONLY by
#   market_exposure_contracts.jl.
#
#   PER-VRES PPA MARKETS: each VRES has its own PPA sub-market with GreenProducer.
#   PER-H2-PRODUCER HPA MARKETS: each GreenProducer has its own HPA sub-market
#   with GreenOfftaker side.
#   Contract prices are stored as results["λ_ppa"] and results["λ_hpa"] (separate
#   from results["λ"] to avoid Dict/Vector type conflicts).
#
# ==============================================================================

function define_results_contracts!(admm_data::Dict, results::Dict, ADMM::Dict, agents::Dict,
                                  elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                                  H2_GC_market::Dict, EP_market::Dict, ppa_market::Dict, hpa_market::Dict;
                                  sp_prices_file::String = "", sp_primal_file::String = "",
                                  sp_cap_file::String = "", use_primal_warmstart::Bool = true)
    # Call base define_results! for the five standard markets (same warm-start as market_exposure)
    define_results!(admm_data, results, ADMM, agents, elec_market, H2_market,
                    elec_GC_market, H2_GC_market, EP_market;
                    sp_prices_file=sp_prices_file, sp_primal_file=sp_primal_file,
                    sp_cap_file=sp_cap_file, use_primal_warmstart=use_primal_warmstart)

    n_ts = admm_data["nTimesteps"]
    n_rd = admm_data["nReprDays"]
    n_yr = admm_data["nYears"]
    shp  = (n_ts, n_rd, n_yr)

    # Add contract market to results["markets"]
    results["markets"]["ppa"] = ppa_market
    results["markets"]["hpa"] = hpa_market

    # Per-VRES PPA sub-markets
    ppa_vres = collect(keys(get(ppa_market, "per_vres", Dict())))
    if isempty(ppa_vres)
        # Fallback: no VRES in PPA market (e.g. no VRES agents)
        ppa_vres = get(agents, :ppa_vres, String[])
    end

    # results["λ_ppa"]: Dict(vres_id => [3D arrays]) — separate key to avoid type conflict with results["λ"]
    results["λ_ppa"] = Dict{String, Vector}()
    for vres_id in ppa_vres
        pv = get(ppa_market["per_vres"], vres_id, Dict())
        init_price = get(pv, "initial_price", ppa_market["initial_price"])
        results["λ_ppa"][vres_id] = [fill(init_price, shp...)]
    end

    # Per-agent quantity buffers: VRES supply in results["ppa"], electrolyzer demand in results["ppa_from"]
    # ppa_cap: VRES stores +cap; electrolyzer stores per-VRES in results["ppa_cap_from"]
    results["ppa"]        = Dict(m => [] for m in agents[:all])
    results["ppa_cap"]    = Dict(m => [] for m in agents[:all])
    results["ppa_from"]   = Dict(m => Dict(v => [] for v in ppa_vres) for m in agents[:H2])
    results["ppa_cap_from"] = Dict(m => Dict(v => [] for v in ppa_vres) for m in agents[:H2])

    # ADMM state: per-VRES — use separate ADMM["ppa"] to avoid Dict/Vector type conflict
    base_tol = get(admm_data, "epsilon", 1.0)
    rho_init = ppa_market["rho_initial"]
    ADMM["ppa"] = Dict(
        "Imbalances"     => Dict(v => Any[] for v in ppa_vres),
        "Imbalances_cap" => Dict(v => Any[] for v in ppa_vres),
        "ρ"              => Dict(v => [get(get(ppa_market["per_vres"], v, Dict()), "rho_initial", rho_init)] for v in ppa_vres),
        "ρ_cap"          => Dict(v => [get(get(ppa_market["per_vres"], v, Dict()), "rho_initial", rho_init)] for v in ppa_vres),
        "PriceHistory"   => Dict(v => Float64[] for v in ppa_vres),
        "ImbalanceMean"  => Dict(v => Float64[] for v in ppa_vres),
        "ImbalanceMean_cap" => Dict(v => Float64[] for v in ppa_vres),
        "Primal"         => Dict(v => Float64[] for v in ppa_vres),
        "Primal_cap"     => Dict(v => Float64[] for v in ppa_vres),
        "Dual"           => Dict(v => Float64[] for v in ppa_vres),
        "Dual_cap"       => Dict(v => Float64[] for v in ppa_vres),
        "BestPrimal"     => Dict(v => Inf for v in ppa_vres),
        "BestDual"       => Dict(v => Inf for v in ppa_vres),
        "BestPrimal_cap" => Dict(v => Inf for v in ppa_vres),
        "BestDual_cap"   => Dict(v => Inf for v in ppa_vres),
        "ρ_frozen"       => Dict(v => false for v in ppa_vres),
        "ρ_frozen_cap"   => Dict(v => false for v in ppa_vres),
        "R_hist"         => Dict(v => Float64[] for v in ppa_vres),
        "R_hist_cap"     => Dict(v => Float64[] for v in ppa_vres),
        "ResidualScale_Primal"     => Dict(v => 0.0 for v in ppa_vres),
        "ResidualScale_Primal_cap" => Dict(v => 0.0 for v in ppa_vres),
        "ResidualScale_Dual"       => Dict(v => 0.0 for v in ppa_vres),
        "ResidualScale_Dual_cap"   => Dict(v => 0.0 for v in ppa_vres),
        "Tolerance"      => Dict(v => base_tol for v in ppa_vres),
        "Tolerance_cap"  => Dict(v => base_tol for v in ppa_vres),
    )

    # Store ppa_vres for use in ADMM and solve modules
    ppa_market["ppa_vres"] = ppa_vres

    # Per-H2-producer HPA sub-markets
    hpa_h2 = collect(keys(get(hpa_market, "per_h2", Dict())))
    if isempty(hpa_h2)
        hpa_h2 = get(agents, :hpa_h2, String[])
    end

    results["λ_hpa"] = Dict{String, Vector}()
    for h2_id in hpa_h2
        hv = get(hpa_market["per_h2"], h2_id, Dict())
        init_price = get(hv, "initial_price", hpa_market["initial_price"])
        results["λ_hpa"][h2_id] = [fill(init_price, shp...)]
    end

    results["hpa"]        = Dict(m => [] for m in agents[:all])
    results["hpa_cap"]    = Dict(m => [] for m in agents[:all])
    results["hpa_from"]   = Dict(m => Dict(v => [] for v in hpa_h2) for m in agents[:offtaker])
    results["hpa_cap_from"] = Dict(m => Dict(v => [] for v in hpa_h2) for m in agents[:offtaker])

    rho_init_hpa = hpa_market["rho_initial"]
    ADMM["hpa"] = Dict(
        "Imbalances"     => Dict(v => Any[] for v in hpa_h2),
        "Imbalances_cap" => Dict(v => Any[] for v in hpa_h2),
        "ρ"              => Dict(v => [get(get(hpa_market["per_h2"], v, Dict()), "rho_initial", rho_init_hpa)] for v in hpa_h2),
        "ρ_cap"          => Dict(v => [get(get(hpa_market["per_h2"], v, Dict()), "rho_initial", rho_init_hpa)] for v in hpa_h2),
        "PriceHistory"   => Dict(v => Float64[] for v in hpa_h2),
        "ImbalanceMean"  => Dict(v => Float64[] for v in hpa_h2),
        "ImbalanceMean_cap" => Dict(v => Float64[] for v in hpa_h2),
        "Primal"         => Dict(v => Float64[] for v in hpa_h2),
        "Primal_cap"     => Dict(v => Float64[] for v in hpa_h2),
        "Dual"           => Dict(v => Float64[] for v in hpa_h2),
        "Dual_cap"       => Dict(v => Float64[] for v in hpa_h2),
        "BestPrimal"     => Dict(v => Inf for v in hpa_h2),
        "BestDual"       => Dict(v => Inf for v in hpa_h2),
        "BestPrimal_cap" => Dict(v => Inf for v in hpa_h2),
        "BestDual_cap"   => Dict(v => Inf for v in hpa_h2),
        "ρ_frozen"       => Dict(v => false for v in hpa_h2),
        "ρ_frozen_cap"   => Dict(v => false for v in hpa_h2),
        "R_hist"         => Dict(v => Float64[] for v in hpa_h2),
        "R_hist_cap"     => Dict(v => Float64[] for v in hpa_h2),
        "ResidualScale_Primal"     => Dict(v => 0.0 for v in hpa_h2),
        "ResidualScale_Primal_cap" => Dict(v => 0.0 for v in hpa_h2),
        "ResidualScale_Dual"       => Dict(v => 0.0 for v in hpa_h2),
        "ResidualScale_Dual_cap"   => Dict(v => 0.0 for v in hpa_h2),
        "Tolerance"      => Dict(v => base_tol for v in hpa_h2),
        "Tolerance_cap"  => Dict(v => base_tol for v in hpa_h2),
    )
    hpa_market["hpa_h2"] = hpa_h2

    return results, ADMM
end
