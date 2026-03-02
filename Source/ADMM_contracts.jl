# ==============================================================================
# ADMM_contracts.jl — Main ADMM loop with bilateral contract markets
# ==============================================================================
#
# PURPOSE:
#   Extends ADMM! for market_exposure_contracts. Adds the bilateral contract
#   energy market (3D) and contract capacity market (scalar). Uses
#   ADMM_subroutine_contracts! and update_rho_contracts!.
#
# ==============================================================================

function ADMM_contracts!(results::Dict, ADMM_state::Dict, elec_market::Dict, H2_market::Dict,
                         elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
                         contract_market::Dict, mdict::Dict, agents::Dict, data::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    max_iter = data["ADMM"]["max_iter"]
    convergence = 0
    iterations = ProgressBar(1:max_iter)

    for iter in iterations
        convergence == 1 && break

        for m in agents[:all]
            ADMM_subroutine_contracts!(m, data, results, ADMM_state, elec_market, H2_market,
                                       elec_GC_market, H2_GC_market, EP_market, contract_market,
                                       mdict, agents, TO)
        end

        # ------------------------------------------------------------------
        # Imbalances (standard + contract)
        # ------------------------------------------------------------------
        @timeit TO "Compute imbalances" begin
            imb_elec = sum(results["g"][m][end] for m in agents[:elec_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec"], imb_elec)

            imb_H2 = sum(results["h2"][m][end] for m in agents[:H2_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["H2"], imb_H2)

            imb_elec_GC = sum(results["elec_GC"][m][end] for m in agents[:elec_GC_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["elec_GC"], imb_elec_GC)

            imb_H2_GC = sum(results["H2_GC"][m][end] for m in agents[:H2_GC_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["H2_GC"], imb_H2_GC)

            imb_EP = sum(results["EP"][m][end] for m in agents[:EP_market]; init=zeros(shp...)) .- EP_market["D_EP"]
            push!(ADMM_state["Imbalances"]["EP"], imb_EP)

            # Contract energy (3D): VRES supplies +g_contract, electrolyzer demands -g_contract
            imb_contract = sum(results["contract"][m][end] for m in agents[:contract_market]; init=zeros(shp...))
            push!(ADMM_state["Imbalances"]["contract"], imb_contract)

            # contract_cap consensus: stored as net position (VRES +cap, electrolyzer -cap)
            imb_contract_cap = sum(isempty(results["contract_cap"][m]) ? 0.0 : results["contract_cap"][m][end]
                                   for m in agents[:contract_market])
            push!(ADMM_state["Imbalances"]["contract_cap"], imb_contract_cap)
        end

        @timeit TO "Imbalance means" begin
            push!(ADMM_state["ImbalanceMean"]["elec"],    mean(ADMM_state["Imbalances"]["elec"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2"],      mean(ADMM_state["Imbalances"]["H2"][end]))
            push!(ADMM_state["ImbalanceMean"]["elec_GC"], mean(ADMM_state["Imbalances"]["elec_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2_GC"],   mean(ADMM_state["Imbalances"]["H2_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["EP"],      mean(ADMM_state["Imbalances"]["EP"][end]))
            push!(ADMM_state["ImbalanceMean"]["contract"],     mean(ADMM_state["Imbalances"]["contract"][end]))
            push!(ADMM_state["ImbalanceMean"]["contract_cap"], abs(ADMM_state["Imbalances"]["contract_cap"][end]))
        end

        # ------------------------------------------------------------------
        # Primal residuals
        # ------------------------------------------------------------------
        @timeit TO "Primal residuals" begin
            rp_elec    = sqrt(sum(ADMM_state["Imbalances"]["elec"][end].^2))
            rp_H2      = sqrt(sum(ADMM_state["Imbalances"]["H2"][end].^2))
            rp_elec_GC = sqrt(sum(ADMM_state["Imbalances"]["elec_GC"][end].^2))
            rp_H2_GC   = sqrt(sum(ADMM_state["Imbalances"]["H2_GC"][end].^2))
            rp_EP      = sqrt(sum(ADMM_state["Imbalances"]["EP"][end].^2))
            rp_contract     = sqrt(sum(ADMM_state["Imbalances"]["contract"][end].^2))
            rp_contract_cap = abs(ADMM_state["Imbalances"]["contract_cap"][end])

            push!(ADMM_state["Residuals"]["Primal"]["elec"],    rp_elec)
            push!(ADMM_state["Residuals"]["Primal"]["H2"],      rp_H2)
            push!(ADMM_state["Residuals"]["Primal"]["elec_GC"], rp_elec_GC)
            push!(ADMM_state["Residuals"]["Primal"]["H2_GC"],   rp_H2_GC)
            push!(ADMM_state["Residuals"]["Primal"]["EP"],      rp_EP)
            push!(ADMM_state["Residuals"]["Primal"]["contract"],     rp_contract)
            push!(ADMM_state["Residuals"]["Primal"]["contract_cap"], rp_contract_cap)

            scales_pr = ADMM_state["ResidualScale"]["Primal"]
            if scales_pr["elec"] == 0.0 && rp_elec > 0.0
                scales_pr["elec"] = rp_elec
            end
            if scales_pr["H2"] == 0.0 && rp_H2 > 0.0
                scales_pr["H2"] = rp_H2
            end
            if scales_pr["elec_GC"] == 0.0 && rp_elec_GC > 0.0
                scales_pr["elec_GC"] = rp_elec_GC
            end
            if scales_pr["H2_GC"] == 0.0 && rp_H2_GC > 0.0
                scales_pr["H2_GC"] = rp_H2_GC
            end
            if scales_pr["EP"] == 0.0 && rp_EP > 0.0
                scales_pr["EP"] = rp_EP
            end
            if scales_pr["contract"] == 0.0 && rp_contract > 0.0
                scales_pr["contract"] = rp_contract
            end
            if scales_pr["contract_cap"] == 0.0 && rp_contract_cap > 0.0
                scales_pr["contract_cap"] = rp_contract_cap
            end
        end

        # ------------------------------------------------------------------
        # Dual residuals (standard + contract)
        # ------------------------------------------------------------------
        @timeit TO "Dual residuals" begin
            nE  = elec_market["nAgents"]
            nH  = H2_market["nAgents"]
            nEG = elec_GC_market["nAgents"]
            nHG = H2_GC_market["nAgents"]
            nEP = EP_market["nAgents"]
            nC  = contract_market["nAgents"]

            if iter > 1
                dual_elec = 0.0
                for m in agents[:elec_market]
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

                dual_contract = 0.0
                for m in agents[:contract_market]
                    diff = (results["contract"][m][end] .- sum(results["contract"][mstar][end] for mstar in agents[:contract_market]) ./ (nC + 1)) .-
                           (results["contract"][m][end-1] .- sum(results["contract"][mstar][end-1] for mstar in agents[:contract_market]) ./ (nC + 1))
                    dual_contract += sum((ADMM_state["ρ"]["contract"][end] .* diff).^2)
                end
                push!(ADMM_state["Residuals"]["Dual"]["contract"], sqrt(dual_contract))

                dual_contract_cap = 0.0
                _net_cap(ag) = isempty(results["contract_cap"][ag]) ? 0.0 : results["contract_cap"][ag][end]
                _net_cap_prev(ag) = length(results["contract_cap"][ag]) < 2 ? 0.0 : results["contract_cap"][ag][end-1]
                for m in agents[:contract_market]
                    diff = (_net_cap(m) - sum(_net_cap(mstar) for mstar in agents[:contract_market]) / (nC + 1)) -
                           (_net_cap_prev(m) - sum(_net_cap_prev(mstar) for mstar in agents[:contract_market]) / (nC + 1))
                    dual_contract_cap += (ADMM_state["ρ"]["contract_cap"][end] * diff)^2
                end
                push!(ADMM_state["Residuals"]["Dual"]["contract_cap"], sqrt(dual_contract_cap))

                scales_du = ADMM_state["ResidualScale"]["Dual"]
                for key in ("elec", "H2", "elec_GC", "H2_GC", "EP", "contract", "contract_cap")
                    rd = ADMM_state["Residuals"]["Dual"][key][end]
                    if scales_du[key] == 0.0 && rd < Inf
                        scales_du[key] = rd
                    end
                end
            else
                for key in ("elec", "H2", "elec_GC", "H2_GC", "EP", "contract", "contract_cap")
                    push!(ADMM_state["Residuals"]["Dual"][key], Inf)
                end
            end
        end

        # ------------------------------------------------------------------
        # Price update (standard + contract)
        # ------------------------------------------------------------------
        @timeit TO "Update prices" begin
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                rp = ADMM_state["Residuals"]["Primal"][mkt][end]
                rd = ADMM_state["Residuals"]["Dual"][mkt][end]
                tol = ADMM_state["Tolerance"][mkt]
                base = max(rp, rd)
                mid_resid_factor = 2.0
                η = base <= mid_resid_factor * tol ? 0.3 : 1.0
                push!(results["λ"][mkt],
                      results["λ"][mkt][end] .- η .* ADMM_state["ρ"][mkt][end] .* ADMM_state["Imbalances"][mkt][end])
            end
            results["λ"]["H2_GC"][end] .= max.(results["λ"]["H2_GC"][end], 0.0)

            # Contract pool: g_contract cleared at λ_contract (€/MWh). No price for contract_cap.
            rp = ADMM_state["Residuals"]["Primal"]["contract"][end]
            rd = ADMM_state["Residuals"]["Dual"]["contract"][end]
            tol = ADMM_state["Tolerance"]["contract"]
            base = max(rp, rd)
            η = base <= 2.0 * tol ? 0.3 : 1.0
            push!(results["λ"]["contract"],
                  results["λ"]["contract"][end] .- η .* ADMM_state["ρ"]["contract"][end] .* ADMM_state["Imbalances"]["contract"][end])
            results["λ"]["contract"][end] .= max.(results["λ"]["contract"][end], 0.0)
        end

        @timeit TO "Price means" begin
            push!(ADMM_state["PriceHistory"]["elec"],    mean(results["λ"]["elec"][end]))
            push!(ADMM_state["PriceHistory"]["H2"],      mean(results["λ"]["H2"][end]))
            push!(ADMM_state["PriceHistory"]["elec_GC"], mean(results["λ"]["elec_GC"][end]))
            push!(ADMM_state["PriceHistory"]["H2_GC"],   mean(results["λ"]["H2_GC"][end]))
            push!(ADMM_state["PriceHistory"]["EP"],      mean(results["λ"]["EP"][end]))
            push!(ADMM_state["PriceHistory"]["contract"], mean(results["λ"]["contract"][end]))
        end

        @timeit TO "Update ρ" begin
            update_rho_contracts!(ADMM_state, iter)
        end

        set_description(iterations, "")

        # ------------------------------------------------------------------
        # Convergence check (all 7 markets)
        # ------------------------------------------------------------------
        min_iter = get(data["ADMM"], "min_iter", 500)
        if iter >= min_iter
            eps_abs = ADMM_state["EpsilonAbs"]
            eps_rel = ADMM_state["EpsilonRel"]
            n_slots = n_ts * n_rd * n_yr
            sqrt_n = sqrt(n_slots)
            scale_pr = ADMM_state["ResidualScale"]["Primal"]
            scale_du = ADMM_state["ResidualScale"]["Dual"]

            function within_tol(key::String)
                rp = ADMM_state["Residuals"]["Primal"][key][end]
                rd = ADMM_state["Residuals"]["Dual"][key][end]
                sp = max(scale_pr[key], 1.0)
                sd = max(scale_du[key], 1.0)
                eps_pr = eps_abs * sqrt_n + eps_rel * sp
                eps_du = eps_abs * sqrt_n + eps_rel * sd
                return (rp <= eps_pr) && (rd <= eps_du)
            end

            # For contract_cap (scalar), use sqrt_n = 1 for consistency
            function within_tol_cap(key::String)
                rp = ADMM_state["Residuals"]["Primal"][key][end]
                rd = ADMM_state["Residuals"]["Dual"][key][end]
                sp = max(scale_pr[key], 1.0)
                sd = max(scale_du[key], 1.0)
                eps_pr = eps_abs * 1.0 + eps_rel * sp
                eps_du = eps_abs * 1.0 + eps_rel * sd
                return (rp <= eps_pr) && (rd <= eps_du)
            end

            if (within_tol("elec") && within_tol("H2") && within_tol("elec_GC") &&
                within_tol("H2_GC") && within_tol("EP") &&
                within_tol("contract") && within_tol_cap("contract_cap"))
                convergence = 1
            end
        end

        ADMM_state["n_iter"] = iter
    end

    println()
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
        "contract" => "Contract",
    )
    println("Final residuals and mean prices per market:")
    for key in ("elec", "H2", "elec_GC", "H2_GC", "EP", "contract")
        primal = ADMM_state["Residuals"]["Primal"][key][end]
        dual   = ADMM_state["Residuals"]["Dual"][key][end]
        price  = ADMM_state["PriceHistory"][key][end]
        @printf("  %-14s  primal = %.3e,  dual = %.3e,  price_mean = %.6f\n",
                market_labels[key], primal, dual, price)
    end
    @printf("  %-14s  primal = %.3e,  dual = %.3e  (consensus, no price)\n",
            "contract_cap", ADMM_state["Residuals"]["Primal"]["contract_cap"][end],
            ADMM_state["Residuals"]["Dual"]["contract_cap"][end])

    return nothing
end
