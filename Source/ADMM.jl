# ==============================================================================
# ADMM.jl — Main ADMM coordination loop
# ==============================================================================
#
# PURPOSE:
#   Runs the ADMM loop until convergence or max_iter. Each iteration:
#   1. For each agent: ADMM_subroutine! updates λ, g_bar, ρ on the model, solves
#      the agent, and appends the solution quantities to results.
#   2. Compute market imbalances (sum of net positions; for EP, subtract D_EP).
#   3. Append mean imbalance per market to ImbalanceMean (for CSV).
#   4. Primal residual per market = L2 norm of imbalance.
#   5. Dual residual per market = norm of (ρ * change in consensus deviation);
#      first iteration uses Inf (no previous deviation).
#   6. Price update: λ_new = λ_old - ρ * imbalance (elementwise).
#   7. Append mean price per market to PriceHistory (for CSV).
#   8. update_rho! adapts ρ per market (Boyd rule).
#   9. If all primal and dual residuals are below tolerance, set convergence=1.
#   Progress bar is left clean (no per-iteration print).
#
# ARGUMENTS:
#   results, ADMM_state — Filled by define_results!; updated in place.
#   elec_market, ... — Market dicts (nAgents, D_EP for EP).
#   mdict — Dict of JuMP models (one per agent).
#   agents — Dict of agent lists (all, elec_market, H2_market, ...).
#   data — Contains General (nTimesteps, nReprDays, nYears) and ADMM (max_iter).
#   TO — TimerOutput for profiling each section.
#
# ==============================================================================

function ADMM!(results::Dict, ADMM_state::Dict, elec_market::Dict, H2_market::Dict,
               elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
               mdict::Dict, agents::Dict, data::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    # Shape of every 3D quantity / price tensor: (hours × representative days × years).
    # All market imbalances, prices λ, and consensus targets g_bar share this shape.
    shp = (n_ts, n_rd, n_yr)
    max_iter = data["ADMM"]["max_iter"]
    convergence = 0
    iterations = ProgressBar(1:max_iter)

    for iter in iterations
        # Early exit once convergence has been achieved (flagged at the end of
        # the previous iteration). Breaking here—rather than using a while-loop—
        # keeps the ProgressBar display frozen at the converged iteration number
        # instead of jumping ahead to max_iter.
        convergence == 1 && break

        # Solve all agents (single-threaded for deterministic order and result indexing)
        for m in agents[:all]
            ADMM_subroutine!(m, data, results, ADMM_state, elec_market, H2_market,
                            elec_GC_market, H2_GC_market, EP_market, mdict[m], agents, TO)
        end

        # ------------------------------------------------------------------
        # Imbalances (full 3D tensors, one entry per (jh, jd, jy))
        # For each market, sum the net positions of ALL participants.
        # Sign convention: generators are positive, consumers negative.
        # A positive imbalance means excess supply; negative means excess demand.
        # For the EP market, we additionally subtract D_EP (the fixed, inelastic
        # end-product demand) so that imbalance = supply − demand.
        # ------------------------------------------------------------------
        @timeit TO "Compute imbalances" begin
            imb_elec = sum(results["g"][m][end] for m in agents[:elec_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec"], imb_elec)

            imb_H2 = sum(results["h2"][m][end] for m in agents[:H2_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["H2"], imb_H2)

            imb_elec_GC = sum(results["elec_GC"][m][end] for m in agents[:elec_GC_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec_GC"], imb_elec_GC)

            imb_H2_GC = sum(results["H2_GC"][m][end] for m in agents[:H2_GC_market]; init=zeros(shp...))
            # H2_GC is now a proper hourly market (no annual aggregation).
            # Offtakers have temporal flexibility to buy GCs whenever prices
            # are low, accumulating toward their annual mandate internally.
            push!(ADMM_state["Imbalances"]["H2_GC"], imb_H2_GC)

            # EP is the only market with fixed inelastic demand D_EP;
            # subtract it so imbalance = Σ agent_supply − D_EP.
            imb_EP = sum(results["EP"][m][end] for m in agents[:EP_market]; init=zeros(shp...)) .- EP_market["D_EP"]
            push!(ADMM_state["Imbalances"]["EP"], imb_EP)
        end

        # ------------------------------------------------------------------
        # Imbalance means — scalar diagnostic (mean over all (jh,jd,jy)
        # entries of the 3D imbalance tensor). These scalars are appended
        # per iteration for CSV logging / plotting convergence curves.
        # ------------------------------------------------------------------
        @timeit TO "Imbalance means" begin
            push!(ADMM_state["ImbalanceMean"]["elec"],    mean(ADMM_state["Imbalances"]["elec"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2"],      mean(ADMM_state["Imbalances"]["H2"][end]))
            push!(ADMM_state["ImbalanceMean"]["elec_GC"], mean(ADMM_state["Imbalances"]["elec_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2_GC"],   mean(ADMM_state["Imbalances"]["H2_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["EP"],      mean(ADMM_state["Imbalances"]["EP"][end]))
        end

        # ------------------------------------------------------------------
        # Primal residuals — L2 norm of the imbalance vector: √Σ imbalance².
        # Measures how far each market is from clearing (supply = demand).
        # Smaller values ⇒ closer to feasibility. This is one half of the
        # standard ADMM stopping criterion (Boyd et al., 2011).
        # ------------------------------------------------------------------
        @timeit TO "Primal residuals" begin
            push!(ADMM_state["Residuals"]["Primal"]["elec"],    sqrt(sum(ADMM_state["Imbalances"]["elec"][end].^2)))
            push!(ADMM_state["Residuals"]["Primal"]["H2"],      sqrt(sum(ADMM_state["Imbalances"]["H2"][end].^2)))
            push!(ADMM_state["Residuals"]["Primal"]["elec_GC"], sqrt(sum(ADMM_state["Imbalances"]["elec_GC"][end].^2)))
            push!(ADMM_state["Residuals"]["Primal"]["H2_GC"],   sqrt(sum(ADMM_state["Imbalances"]["H2_GC"][end].^2)))
            push!(ADMM_state["Residuals"]["Primal"]["EP"],      sqrt(sum(ADMM_state["Imbalances"]["EP"][end].^2)))
        end

        # ------------------------------------------------------------------
        # Dual residuals — measure how much each agent's position changed
        # relative to its consensus target between successive iterations.
        #
        # For each agent m in a market with n participants, compute:
        #   diff_m = (q_m^k − q̄^k) − (q_m^{k−1} − q̄^{k−1})
        # where q̄^k = (1/(n+1)) · Σ_m q_m^k is the consensus average.
        # The (n+1) denominator comes from the sharing ADMM formulation,
        # which introduces one "market copy" alongside the n agent copies,
        # so the consensus variable is the mean of (n+1) copies.
        #
        # The dual residual = √ Σ_m (ρ · diff_m)² (L2 norm over all
        # agents and all (jh,jd,jy) timesteps). Smaller ⇒ agents are
        # settling on consistent positions. Together with the primal
        # residual, this forms the ADMM stopping criterion (Boyd et al.).
        #
        # On the first iteration there is no previous iterate to compare
        # against, so we set dual residuals to Inf (cannot be satisfied,
        # forcing at least two iterations before convergence is declared).
        # ------------------------------------------------------------------
        @timeit TO "Dual residuals" begin
            nE  = elec_market["nAgents"]
            nH  = H2_market["nAgents"]
            nEG = elec_GC_market["nAgents"]
            nHG = H2_GC_market["nAgents"]
            nEP = EP_market["nAgents"]
            if iter > 1
                dual_elec = 0.0
                for m in agents[:elec_market]
                    # Change in (own_quantity − consensus_average) from iter k−1 to k
                    diff = (results["g"][m][end] .- sum(results["g"][mstar][end] for mstar in agents[:elec_market]) ./ (nE + 1)) .-
                           (results["g"][m][end-1] .- sum(results["g"][mstar][end-1] for mstar in agents[:elec_market]) ./ (nE + 1))
                    dual_elec += sum((ADMM_state["ρ"]["elec"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["elec"], sqrt(dual_elec))
                dual_H2 = 0.0
                for m in agents[:H2_market]
                    diff = (results["h2"][m][end] .- sum(results["h2"][mstar][end] for mstar in agents[:H2_market]) ./ (nH + 1)) .-
                           (results["h2"][m][end-1] .- sum(results["h2"][mstar][end-1] for mstar in agents[:H2_market]) ./ (nH + 1))
                    dual_H2 += sum((ADMM_state["ρ"]["H2"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["H2"], sqrt(dual_H2))
                dual_elec_GC = 0.0
                for m in agents[:elec_GC_market]
                    diff = (results["elec_GC"][m][end] .- sum(results["elec_GC"][mstar][end] for mstar in agents[:elec_GC_market]) ./ (nEG + 1)) .-
                           (results["elec_GC"][m][end-1] .- sum(results["elec_GC"][mstar][end-1] for mstar in agents[:elec_GC_market]) ./ (nEG + 1))
                    dual_elec_GC += sum((ADMM_state["ρ"]["elec_GC"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["elec_GC"], sqrt(dual_elec_GC))
                dual_H2_GC = 0.0
                for m in agents[:H2_GC_market]
                    diff = (results["H2_GC"][m][end] .- sum(results["H2_GC"][mstar][end] for mstar in agents[:H2_GC_market]) ./ (nHG + 1)) .-
                           (results["H2_GC"][m][end-1] .- sum(results["H2_GC"][mstar][end-1] for mstar in agents[:H2_GC_market]) ./ (nHG + 1))
                    dual_H2_GC += sum((ADMM_state["ρ"]["H2_GC"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["H2_GC"], sqrt(dual_H2_GC))
                dual_EP = 0.0
                for m in agents[:EP_market]
                    diff = (results["EP"][m][end] .- sum(results["EP"][mstar][end] for mstar in agents[:EP_market]) ./ (nEP + 1)) .-
                           (results["EP"][m][end-1] .- sum(results["EP"][mstar][end-1] for mstar in agents[:EP_market]) ./ (nEP + 1))
                    dual_EP += sum((ADMM_state["ρ"]["EP"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["EP"], sqrt(dual_EP))
            else
                # First iteration: no previous iterate exists, so dual residuals
                # are undefined. Set to Inf to prevent premature convergence.
                push!(ADMM_state["Residuals"]["Dual"]["elec"],    Inf)
                push!(ADMM_state["Residuals"]["Dual"]["H2"],      Inf)
                push!(ADMM_state["Residuals"]["Dual"]["elec_GC"], Inf)
                push!(ADMM_state["Residuals"]["Dual"]["H2_GC"],  Inf)
                push!(ADMM_state["Residuals"]["Dual"]["EP"],      Inf)
            end
        end

        # ------------------------------------------------------------------
        # Price (dual variable) update: λ_new = λ_old − ρ · imbalance.
        # Standard ADMM dual variable update (gradient ascent on the
        # Lagrangian dual with step size ρ).
        #   • When supply > demand (positive imbalance) → price decreases.
        #   • When demand > supply (negative imbalance) → price increases.
        # NOTE: Over-relaxation (α > 1) was tested and caused divergence
        # in this tightly-coupled multi-market problem.  Standard update
        # (α = 1) is used instead.
        # ------------------------------------------------------------------
        @timeit TO "Update prices" begin
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                push!(results["λ"][mkt],
                      results["λ"][mkt][end] .- ADMM_state["ρ"][mkt][end] .* ADMM_state["Imbalances"][mkt][end])
            end
            # H2_GC price floor: the electrolyzer VOLUNTARILY issues green
            # certificates — at price < 0 no rational producer would issue,
            # so supply is identically 0 and the equilibrium price is ≥ 0.
            # Without this projection, negative prices attract unbounded
            # demand from offtakers (who profit from buying at negative
            # prices), creating a persistent limit-cycle oscillation that
            # ADMM cannot resolve.  Clamping to [0,∞) is the standard
            # "projected ADMM" technique and preserves convergence theory.
            results["λ"]["H2_GC"][end] .= max.(results["λ"]["H2_GC"][end], 0.0)
        end

        # Store scalar price diagnostics (mean price per market and iteration)
        @timeit TO "Price means" begin
            push!(ADMM_state["PriceHistory"]["elec"],    mean(results["λ"]["elec"][end]))
            push!(ADMM_state["PriceHistory"]["H2"],      mean(results["λ"]["H2"][end]))
            push!(ADMM_state["PriceHistory"]["elec_GC"], mean(results["λ"]["elec_GC"][end]))
            push!(ADMM_state["PriceHistory"]["H2_GC"],   mean(results["λ"]["H2_GC"][end]))
            push!(ADMM_state["PriceHistory"]["EP"],      mean(results["λ"]["EP"][end]))
        end

        @timeit TO "Update ρ" begin
            update_rho!(ADMM_state, iter)
        end

        # Clean progress bar: show only iteration and max; no extra printing
        set_description(iterations, "")

        # ------------------------------------------------------------------
        # Convergence check (standard ADMM stopping criterion, Boyd et al., 2011).
        # ALL five markets must have BOTH primal AND dual residuals below
        # their market-specific tolerance. If any single residual exceeds
        # its tolerance, the algorithm continues iterating. This ensures
        # that every market has simultaneously cleared (primal) and that
        # agent positions have stabilized (dual) before we declare convergence.
        #
        # IMPORTANT: skip convergence check for the first min_iter iterations.
        # In early iterations, residuals can be spuriously small (agents
        # haven't diverged from initial conditions yet), leading to premature
        # convergence with prices far from equilibrium.
        # ------------------------------------------------------------------
        min_iter = get(data["ADMM"], "min_iter", 500)
        if iter >= min_iter
            tol = ADMM_state["Tolerance"]
            if (ADMM_state["Residuals"]["Primal"]["elec"][end] <= tol["elec"] && ADMM_state["Residuals"]["Dual"]["elec"][end] <= tol["elec"] &&
                ADMM_state["Residuals"]["Primal"]["H2"][end] <= tol["H2"] && ADMM_state["Residuals"]["Dual"]["H2"][end] <= tol["H2"] &&
                ADMM_state["Residuals"]["Primal"]["elec_GC"][end] <= tol["elec_GC"] && ADMM_state["Residuals"]["Dual"]["elec_GC"][end] <= tol["elec_GC"] &&
                ADMM_state["Residuals"]["Primal"]["H2_GC"][end] <= tol["H2_GC"] && ADMM_state["Residuals"]["Dual"]["H2_GC"][end] <= tol["H2_GC"] &&
                ADMM_state["Residuals"]["Primal"]["EP"][end] <= tol["EP"] && ADMM_state["Residuals"]["Dual"]["EP"][end] <= tol["EP"])
                convergence = 1
            end
        end

        ADMM_state["n_iter"] = iter
    end

    # --------------------------------------------------------------------------
    # Summary message: convergence status, iteration count, residuals, prices
    # --------------------------------------------------------------------------
    println()  # ensure a clean line after the progress bar
    if convergence == 1
        println("ADMM convergence achieved.")
    else
        println("ADMM reached max_iter without convergence.")
    end
    n_it = ADMM_state["n_iter"]
    println("Number of iterations: ", n_it)

    market_labels = Dict(
        "elec"    => "Electricity",
        "H2"      => "Hydrogen",
        "elec_GC" => "Electricity_GC",
        "H2_GC"   => "H2_GC",
        "EP"      => "End_Product",
    )
    println("Final residuals and mean prices per market:")
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        primal = ADMM_state["Residuals"]["Primal"][key][end]
        dual   = ADMM_state["Residuals"]["Dual"][key][end]
        price  = ADMM_state["PriceHistory"][key][end]
        @printf("  %-14s  primal = %.3e,  dual = %.3e,  price_mean = %.6f\n",
                market_labels[key], primal, dual, price)
    end

    return nothing
end
