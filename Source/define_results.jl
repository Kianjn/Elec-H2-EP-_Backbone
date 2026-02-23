# ==============================================================================
# define_results.jl — Initialize result and ADMM state structures
# ==============================================================================
#
# PURPOSE:
#   Allocates all dictionaries and arrays used during and after the ADMM loop:
#   (1) results: agents reference, markets reference, initial price 3D arrays
#       per market, and per-agent lists for quantities (g, h2, elec_GC, H2_GC, EP)
#       that will each hold one 3D array per iteration. (2) ADMM: initial ρ per
#       market, empty lists for full imbalance tensors, scalar PriceHistory and
#       ImbalanceMean per market (filled each iteration in ADMM.jl), Primal/Dual
#       residual lists, Tolerance per market, n_iter and walltime.
#
# ARGUMENTS:
#   admm_data — merge(General, ADMM) for nTimesteps, nReprDays, nYears, epsilon.
#   results, ADMM — Dicts to be filled (passed by reference).
#   agents — Dict of agent lists (stored in results for save_results).
#   elec_market, H2_market, ... — Market dicts (stored in results["markets"] and
#     used to read initial_price and rho_initial).
#
# ==============================================================================

function define_results!(admm_data::Dict, results::Dict, ADMM::Dict, agents::Dict,
                        elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                        H2_GC_market::Dict, EP_market::Dict;
                        sp_prices_file::String = "")
    n_ts = admm_data["nTimesteps"]
    n_rd = admm_data["nReprDays"]
    n_yr = admm_data["nYears"]
    shp = (n_ts, n_rd, n_yr)

    # Store references so save_results and post-processing can access agent/market info.
    results["agents"] = agents
    results["markets"] = Dict("elec" => elec_market, "H2" => H2_market, "elec_GC" => elec_GC_market, "H2_GC" => H2_GC_market, "EP" => EP_market)

    # results["λ"] — Per-market list of 3D price arrays, one per ADMM iteration.
    # The list grows by push! in ADMM.jl each iteration. The first element
    # seeds iteration-0 prices for the very first agent solves.
    #
    # Warm-start strategy:
    #   If a social-planner Market_Prices.csv exists, load its HOURLY prices
    #   as the initial λ. This gives each (jh,jd,jy) slot the true equilibrium
    #   price — dramatically speeding convergence for hours whose prices differ
    #   from the market-wide mean (e.g. peak hours at 483 vs. off-peak at 3).
    #   Falls back to uniform scalar fill when the file is missing or incompatible.
    sp_loaded = false
    if !isempty(sp_prices_file) && isfile(sp_prices_file)
        try
            sp_df = CSV.read(sp_prices_file, DataFrame)
            n_total = n_ts * n_rd * n_yr
            if nrow(sp_df) == n_total
                results["λ"] = Dict(
                    "elec"    => [reshape(Float64.(sp_df.Elec_Price),    shp)],
                    "H2"      => [reshape(Float64.(sp_df.H2_Price),     shp)],
                    "elec_GC" => [reshape(Float64.(sp_df.Elec_GC_Price), shp)],
                    "H2_GC"   => [max.(reshape(Float64.(sp_df.H2_GC_Price), shp), 0.0)],
                    "EP"      => [reshape(Float64.(sp_df.EP_Price),     shp)],
                )
                sp_loaded = true
                @info "Warm-started ADMM λ from social planner hourly prices ($sp_prices_file)"
            else
                @warn "Social planner prices file has $(nrow(sp_df)) rows, expected $n_total. " *
                      "Falling back to scalar warm-start."
            end
        catch e
            @warn "Could not read social planner prices ($sp_prices_file): $e. " *
                  "Falling back to scalar warm-start."
        end
    end

    if !sp_loaded
        results["λ"] = Dict(
            "elec"    => [fill(elec_market["initial_price"], shp...)],
            "H2"      => [fill(H2_market["initial_price"], shp...)],
            "elec_GC" => [fill(elec_GC_market["initial_price"], shp...)],
            "H2_GC"   => [fill(H2_GC_market["initial_price"], shp...)],
            "EP"      => [fill(EP_market["initial_price"], shp...)],
        )
    end

    # Per-agent quantity buffers: empty lists for ALL agents (even non-participants).
    # Only agents that actually participate in a given market will have 3D arrays
    # pushed (by ADMM_subroutine); non-participants keep empty lists. Initialising
    # all agents avoids key-missing errors in generic post-processing loops.
    # Key = agent ID (String), value = list of 3D arrays (one per ADMM iteration).
    results["g"]       = Dict(m => [] for m in agents[:all])   # Electricity net position (MW; + = sell, − = buy)
    results["h2"]      = Dict(m => [] for m in agents[:all])   # Hydrogen net position (MW_H2)
    results["elec_GC"] = Dict(m => [] for m in agents[:all])   # Electricity GC net position (MW_GC)
    results["H2_GC"]   = Dict(m => [] for m in agents[:all])   # Hydrogen GC net position (MW_GC)
    results["EP"]      = Dict(m => [] for m in agents[:all])   # End-product net position (MW_EP)

    # Per-agent capacity and investment history for green agents (one vector per ADMM iteration).
    # These are populated only for agents that actually own these variables (VRES, electrolyzer, green offtaker).
    results["Cap_VRES"]        = Dict(m => [] for m in agents[:all])   # VRES capacity per year (MW)
    results["Inv_VRES"]        = Dict(m => [] for m in agents[:all])   # VRES investment per year (MW)
    results["Cap_Elec_H2"]     = Dict(m => [] for m in agents[:all])   # Electrolyzer elec capacity per year (MW)
    results["Inv_Elec_H2"]     = Dict(m => [] for m in agents[:all])   # Electrolyzer elec investment per year (MW)
    results["Cap_EP_Green"]    = Dict(m => [] for m in agents[:all])   # Green offtaker EP capacity per year (MW)
    results["Inv_EP_Green"]    = Dict(m => [] for m in agents[:all])   # Green offtaker EP investment per year (MW)

    # ADMM["ρ"] — Per-market list of scalar penalty weights, one entry per ADMM
    # iteration. Updated by update_rho! (which may increase/decrease ρ based on
    # the ratio of primal to dual residuals). The first element = rho_initial.
    ADMM["ρ"] = Dict(
        "elec"    => [elec_market["rho_initial"]],
        "H2"      => [H2_market["rho_initial"]],
        "elec_GC" => [elec_GC_market["rho_initial"]],
        "H2_GC"   => [H2_GC_market["rho_initial"]],
        "EP"      => [EP_market["rho_initial"]],
    )

    # Full 3D imbalance tensor per iteration: sum of all agents' net positions in
    # each market (for EP: minus D_EP). Used in ADMM_subroutine to compute g_bar
    # (consensus centre) for the next iteration. Any[] typed for flexibility:
    # each element is a full 3D array whose exact numeric type may vary.
    ADMM["Imbalances"] = Dict(
        "elec"    => Any[],
        "H2"      => Any[],
        "elec_GC" => Any[],
        "H2_GC"   => Any[],
        "EP"      => Any[],
    )

    # Scalar summary statistics per market per iteration, written to CSV for
    # diagnostics and convergence monitoring:
    #   PriceHistory  — mean(λ) across all (jh,jd,jy) entries each iteration.
    #   ImbalanceMean — mean(|imbalance|) across all entries each iteration.
    ADMM["PriceHistory"] = Dict(
        "elec"    => Float64[],
        "H2"      => Float64[],
        "elec_GC" => Float64[],
        "H2_GC"   => Float64[],
        "EP"      => Float64[],
    )
    ADMM["ImbalanceMean"] = Dict(
        "elec"    => Float64[],
        "H2"      => Float64[],
        "elec_GC" => Float64[],
        "H2_GC"   => Float64[],
        "EP"      => Float64[],
    )

    # Primal and Dual residuals per market per iteration:
    #   Primal = L2 norm of market imbalance (how far supply ≠ demand).
    #   Dual   = L2 norm of the change in consensus variable g_bar between
    #            consecutive iterations (how much the "agreed" allocation shifted).
    # ADMM converges when both residuals fall below the market's Tolerance.
    ADMM["Residuals"] = Dict(
        "Primal" => Dict("elec" => Float64[], "H2" => Float64[], "elec_GC" => Float64[], "H2_GC" => Float64[], "EP" => Float64[]),
        "Dual"   => Dict("elec" => Float64[], "H2" => Float64[], "elec_GC" => Float64[], "H2_GC" => Float64[], "EP" => Float64[]),
    )

    # Per-market convergence tolerances.
    base_tol = get(admm_data, "epsilon", 1.0)
    ADMM["Tolerance"] = Dict(
        "elec"    => base_tol,
        "elec_GC" => base_tol,
        "EP"      => base_tol,
        "H2"      => base_tol,
        "H2_GC"   => base_tol,
    )
    ADMM["n_iter"]   = 0     # Iteration counter; incremented each ADMM loop
    ADMM["walltime"] = 0.0   # Cumulative wall-clock time (seconds); measured in ADMM.jl
    return results, ADMM
end
