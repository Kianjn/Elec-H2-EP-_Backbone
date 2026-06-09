# ==============================================================================
# ADMM_contracts.jl — Main ADMM loop with bilateral contract markets
# ==============================================================================
#
# PURPOSE:
#   Extends ADMM! for market_exposure_contracts. Adds the bilateral contract
#   energy market (3D) and contract capacity market (scalar). Uses
#   ADMM_subroutine_contracts! and update_rho_contracts!.
#
# CAPACITY CONSENSUS TOLERANCE (relaxed for contracts case):
#   The capacity consensus uses a per-agent equality split (x_cap = z_cap
#   with explicit dual λ_cap and per-agent ρ_cap; see DOCUMENTATION.md §5.4)
#   that couples VRES, electrolyzer, and green offtaker investments. In the
#   contracts case, VRES splits generation between pool and contract, so
#   z_cap = f(g_bar_elec + g_bar_ppa) depends on both standard and contract
#   flow consensus. This creates stronger coupling and slower convergence
#   than in market_exposure.
#
#   We relax the capacity tolerance by CAP_CONSENSUS_TOL_RELAX so that
#   convergence can be declared when flow markets are sufficiently converged,
#   even if capacity consensus lags. The capacity residual remains monitored
#   and reported; results are still meaningful when flow markets have cleared.
#
# ==============================================================================

# Multiplier for capacity consensus tolerance. Effective tolerance for cap is
# (eps_pr, eps_du) * CAP_CONSENSUS_TOL_RELAX, i.e. we accept larger capacity
# residuals. Tune via data["ADMM"]["cap_tol_relax"] if present.
const CAP_CONSENSUS_TOL_RELAX_DEFAULT = 100.0

_cap_scalar(x) = x isa Real ? Float64(x) : (isempty(x) ? 0.0 : Float64(x[1]))

function ADMM_contracts!(results::Dict, ADMM_state::Dict, elec_market::Dict, H2_market::Dict,
                         elec_GC_market::Dict, H2_GC_market::Dict, EP_market::Dict,
                         ppa_market::Dict, hpa_market::Dict, mdict::Dict, agents::Dict, data::Dict, TO::TimerOutput)
    n_ts = data["General"]["nTimesteps"]
    n_rd = data["General"]["nReprDays"]
    n_yr = data["General"]["nYears"]
    shp = (n_ts, n_rd, n_yr)
    max_iter = data["ADMM"]["max_iter"]
    convergence = 0
    iterations = ProgressBar(1:max_iter)
    market_keys = ("elec", "H2", "elec_GC", "H2_GC", "EP", "cap")
    ppa_ids = get(ppa_market, "ppa_vres", String[])
    hpa_ids = get(hpa_market, "hpa_h2", String[])

    # Advanced controller: per-market (and per-contract-submarket) step scales.
    η_scale = Dict("elec" => 1.0, "H2" => 1.0, "elec_GC" => 1.0, "H2_GC" => 1.0, "EP" => 1.0)
    η_scale_ppa = Dict(v => 1.0 for v in ppa_ids)
    η_scale_hpa = Dict(v => 1.0 for v in hpa_ids)

    # Best-iterate checkpoint over normalized residual merit.
    best_iter = 0
    best_score = Inf
    stall_count = 0
    restart_patience = 40
    restart_factor = 1.15
    # Reform: disable checkpoint/recovery steering and rely on plain ADMM updates.
    enable_recovery_steering = false
    rollback_count = 0
    max_rollbacks = 2
    rollback_cooldown = 80
    last_rollback_iter = -rollback_cooldown
    rollback_blend = 0.35
    best_λ = Dict{String,Array{Float64,3}}()
    best_ρ = Dict{String,Float64}()
    best_λ_ppa = Dict{String,Array{Float64,3}}()
    best_ρ_ppa = Dict{String,Float64}()
    best_ρ_ppa_cap = Dict{String,Float64}()
    best_λ_hpa = Dict{String,Array{Float64,3}}()
    best_ρ_hpa = Dict{String,Float64}()
    best_ρ_hpa_cap = Dict{String,Float64}()
    # Per-agent best ρ_cap for the rollback blend (capacity equality split).
    best_ρ_cap = Dict{String,Float64}()

    # Capacity-owning agents — see ADMM.jl and DOCUMENTATION.md §5.4.
    cap_agents = get(agents, :cap_agents, String[])

    # Expose horizon sizes to update_rho_contracts! (Boyd-style abs tolerance).
    n_yr_admm = data["General"]["nYears"]
    n_ts_admm = data["General"]["nTimesteps"]
    n_rd_admm = data["General"]["nReprDays"]
    ADMM_state["n_slots"] = n_ts_admm * n_rd_admm * n_yr_admm
    ADMM_state["n_yr"]    = n_yr_admm
    ADMM_state["rho_cap_max"]        = get(get(data, "ADMM", Dict()), "rho_cap_max", 30.0)
    ADMM_state["rho_cap_inc_factor"] = get(get(data, "ADMM", Dict()), "rho_cap_inc_factor", 1.05)

    function _market_eps_std(key::String, n_slots::Int)
        eps_abs = ADMM_state["EpsilonAbs"]
        eps_rel = ADMM_state["EpsilonRel"]
        sqrt_n = sqrt(max(1, n_slots))
        sp = max(ADMM_state["ResidualScale"]["Primal"][key], 1.0)
        sd = max(ADMM_state["ResidualScale"]["Dual"][key], 1.0)
        eps_pr = eps_abs * sqrt_n + eps_rel * sp
        eps_du = eps_abs * sqrt_n + eps_rel * sd
        return eps_pr, eps_du
    end

    function _market_eps_contract(C::Dict, id::String, is_cap::Bool, n_slots::Int)
        eps_abs = ADMM_state["EpsilonAbs"]
        eps_rel = ADMM_state["EpsilonRel"]
        if is_cap
            sp = max(C["ResidualScale_Primal_cap"][id], 1.0)
            sd = max(C["ResidualScale_Dual_cap"][id], 1.0)
            # Scalar consensus (capacity): keep sqrt(1) basis for consistency with current check.
            eps_pr = eps_abs * 1.0 + eps_rel * sp
            eps_du = eps_abs * 1.0 + eps_rel * sd
        else
            sqrt_n = sqrt(max(1, n_slots))
            sp = max(C["ResidualScale_Primal"][id], 1.0)
            sd = max(C["ResidualScale_Dual"][id], 1.0)
            eps_pr = eps_abs * sqrt_n + eps_rel * sp
            eps_du = eps_abs * sqrt_n + eps_rel * sd
        end
        return eps_pr, eps_du
    end

    # Log initial mean prices (before any iteration) so diagnostics show warm-start.
    for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
        push!(ADMM_state["PriceHistory"][mkt], mean(results["λ"][mkt][end]))
    end

    for iter in iterations
        convergence == 1 && break

        for m in agents[:all]
            ADMM_subroutine_contracts!(m, data, results, ADMM_state, elec_market, H2_market,
                                       elec_GC_market, H2_GC_market, EP_market, ppa_market, hpa_market,
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

            # Contract energy (3D) per VRES: supply[v] - demand_from[v]
            ppa_vres = get(ppa_market, "ppa_vres", String[])
            h2_ids = agents[:H2]
            C = ADMM_state["ppa"]
            for vres_id in ppa_vres
                supply = isempty(results["ppa"][vres_id]) ? zeros(shp...) : results["ppa"][vres_id][end]
                demand = zeros(shp...)
                for h2_id in h2_ids
                    if !isempty(get(results["ppa_from"][h2_id], vres_id, []))
                        demand .+= results["ppa_from"][h2_id][vres_id][end]
                    end
                end
                imb_contract = supply .- demand
                push!(C["Imbalances"][vres_id], imb_contract)
            end

            # contract_cap consensus per VRES: VRES +cap, electrolyzer -cap (stored as -cap)
            for vres_id in ppa_vres
                cap_vres = isempty(results["ppa_cap"][vres_id]) ? 0.0 : results["ppa_cap"][vres_id][end]
                cap_elec_sum = 0.0
                for h2_id in h2_ids
                    if haskey(results["ppa_cap_from"][h2_id], vres_id) && !isempty(results["ppa_cap_from"][h2_id][vres_id])
                        cap_elec_sum += -results["ppa_cap_from"][h2_id][vres_id][end]  # stored as -cap
                    end
                end
                imb_contract_cap = cap_vres - cap_elec_sum
                push!(C["Imbalances_cap"][vres_id], imb_contract_cap)
            end

            # HPA energy (3D) per GreenProducer: supply[h2] - demand_from[h2]
            hpa_h2 = get(hpa_market, "hpa_h2", String[])
            offtaker_ids = agents[:offtaker]
            C_hpa = ADMM_state["hpa"]
            for h2_id in hpa_h2
                supply = isempty(results["hpa"][h2_id]) ? zeros(shp...) : results["hpa"][h2_id][end]
                demand = zeros(shp...)
                for off_id in offtaker_ids
                    if !isempty(get(results["hpa_from"][off_id], h2_id, []))
                        demand .+= results["hpa_from"][off_id][h2_id][end]
                    end
                end
                imb_contract = supply .- demand
                push!(C_hpa["Imbalances"][h2_id], imb_contract)
            end

            # hpa_cap consensus per GreenProducer: producer +cap, offtaker -cap (stored as -cap)
            for h2_id in hpa_h2
                cap_h2 = isempty(results["hpa_cap"][h2_id]) ? 0.0 : results["hpa_cap"][h2_id][end]
                cap_off_sum = 0.0
                for off_id in offtaker_ids
                    if haskey(results["hpa_cap_from"][off_id], h2_id) && !isempty(results["hpa_cap_from"][off_id][h2_id])
                        cap_off_sum += -results["hpa_cap_from"][off_id][h2_id][end]
                    end
                end
                imb_contract_cap = cap_h2 - cap_off_sum
                push!(C_hpa["Imbalances_cap"][h2_id], imb_contract_cap)
            end
        end

        @timeit TO "Imbalance means" begin
            push!(ADMM_state["ImbalanceMean"]["elec"],    mean(ADMM_state["Imbalances"]["elec"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2"],      mean(ADMM_state["Imbalances"]["H2"][end]))
            push!(ADMM_state["ImbalanceMean"]["elec_GC"], mean(ADMM_state["Imbalances"]["elec_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["H2_GC"],   mean(ADMM_state["Imbalances"]["H2_GC"][end]))
            push!(ADMM_state["ImbalanceMean"]["EP"],      mean(ADMM_state["Imbalances"]["EP"][end]))
            C = ADMM_state["ppa"]
            for vres_id in ppa_vres
                push!(C["ImbalanceMean"][vres_id],     mean(C["Imbalances"][vres_id][end]))
                push!(C["ImbalanceMean_cap"][vres_id], abs(C["Imbalances_cap"][vres_id][end]))
            end
            C_hpa = ADMM_state["hpa"]
            for h2_id in get(hpa_market, "hpa_h2", String[])
                push!(C_hpa["ImbalanceMean"][h2_id],     mean(C_hpa["Imbalances"][h2_id][end]))
                push!(C_hpa["ImbalanceMean_cap"][h2_id], abs(C_hpa["Imbalances_cap"][h2_id][end]))
            end
        end

        # ------------------------------------------------------------------
        # Primal residuals (per-VRES for contract)
        # ------------------------------------------------------------------
        @timeit TO "Primal residuals" begin
            rp_elec    = sqrt(sum(ADMM_state["Imbalances"]["elec"][end].^2))
            rp_H2      = sqrt(sum(ADMM_state["Imbalances"]["H2"][end].^2))
            rp_elec_GC = sqrt(sum(ADMM_state["Imbalances"]["elec_GC"][end].^2))
            rp_H2_GC   = sqrt(sum(ADMM_state["Imbalances"]["H2_GC"][end].^2))
            rp_EP      = sqrt(sum(ADMM_state["Imbalances"]["EP"][end].^2))

            push!(ADMM_state["Residuals"]["Primal"]["elec"],    rp_elec)
            push!(ADMM_state["Residuals"]["Primal"]["H2"],      rp_H2)
            push!(ADMM_state["Residuals"]["Primal"]["elec_GC"], rp_elec_GC)
            push!(ADMM_state["Residuals"]["Primal"]["H2_GC"],   rp_H2_GC)
            push!(ADMM_state["Residuals"]["Primal"]["EP"],      rp_EP)

            C = ADMM_state["ppa"]
            for vres_id in ppa_vres
                rp_contract     = sqrt(sum(C["Imbalances"][vres_id][end].^2))
                rp_contract_cap = abs(C["Imbalances_cap"][vres_id][end])
                push!(C["Primal"][vres_id],     rp_contract)
                push!(C["Primal_cap"][vres_id], rp_contract_cap)
                if C["ResidualScale_Primal"][vres_id] == 0.0 && rp_contract > 0.0
                    C["ResidualScale_Primal"][vres_id] = rp_contract
                end
                if C["ResidualScale_Primal_cap"][vres_id] == 0.0 && rp_contract_cap > 0.0
                    C["ResidualScale_Primal_cap"][vres_id] = rp_contract_cap
                end
            end
            C_hpa = ADMM_state["hpa"]
            for h2_id in get(hpa_market, "hpa_h2", String[])
                rp_contract = sqrt(sum(C_hpa["Imbalances"][h2_id][end].^2))
                rp_contract_cap = abs(C_hpa["Imbalances_cap"][h2_id][end])
                push!(C_hpa["Primal"][h2_id], rp_contract)
                push!(C_hpa["Primal_cap"][h2_id], rp_contract_cap)
                if C_hpa["ResidualScale_Primal"][h2_id] == 0.0 && rp_contract > 0.0
                    C_hpa["ResidualScale_Primal"][h2_id] = rp_contract
                end
                if C_hpa["ResidualScale_Primal_cap"][h2_id] == 0.0 && rp_contract_cap > 0.0
                    C_hpa["ResidualScale_Primal_cap"][h2_id] = rp_contract_cap
                end
            end

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

            # ------------------------------------------------------------
            # Capacity primal residuals — per-agent equality split
            # (mirrors ADMM.jl; see DOCUMENTATION.md §5.4 for derivation).
            # ------------------------------------------------------------
            cap_state = ADMM_state["Capacity"]
            rp_cap_sq = 0.0
            for m in cap_agents
                cap_vec = Float64[]
                if !isempty(get(results["Cap_VRES"], m, []))
                    cap_vec = results["Cap_VRES"][m][end]
                elseif !isempty(get(results["Cap_Elec_H2"], m, []))
                    cap_vec = results["Cap_Elec_H2"][m][end]
                elseif !isempty(get(results["Cap_EP_Green"], m, []))
                    cap_vec = results["Cap_EP_Green"][m][end]
                end
                z_hist = cap_state["z"][m]
                z_vec  = isempty(z_hist) ? cap_vec : z_hist[end]
                local_r = 0.0
                if !isempty(cap_vec)
                    local_r = abs(_cap_scalar(cap_vec) - _cap_scalar(z_vec))
                end
                push!(cap_state["Primal"][m], local_r)
                if cap_state["ResidualScale_Primal"][m] == 0.0 && local_r > 0.0
                    cap_state["ResidualScale_Primal"][m] = local_r
                end
                rp_cap_sq += local_r^2
            end
            rp_cap = sqrt(rp_cap_sq)
            push!(ADMM_state["Residuals"]["Primal"]["cap"], rp_cap)
            if ADMM_state["ResidualScale"]["Primal"]["cap"] == 0.0 && rp_cap > 0.0
                ADMM_state["ResidualScale"]["Primal"]["cap"] = rp_cap
            end
        end

        # ------------------------------------------------------------------
        # Capacity dual ASCENT (per-agent λ_cap update; mirrors ADMM.jl)
        # λ_m^k = λ_m^{k-1} + ρ_m^{k-1} · (x_m^k - z_m^k) per year y.
        # See DOCUMENTATION.md §5.4 for justification.
        # ------------------------------------------------------------------
        @timeit TO "Capacity dual update" begin
            cap_state = ADMM_state["Capacity"]
            for m in cap_agents
                cap_vec = Float64[]
                if !isempty(get(results["Cap_VRES"], m, []))
                    cap_vec = results["Cap_VRES"][m][end]
                elseif !isempty(get(results["Cap_Elec_H2"], m, []))
                    cap_vec = results["Cap_Elec_H2"][m][end]
                elseif !isempty(get(results["Cap_EP_Green"], m, []))
                    cap_vec = results["Cap_EP_Green"][m][end]
                end
                z_hist = cap_state["z"][m]
                z_vec  = isempty(z_hist) ? cap_vec : z_hist[end]
                ρ_m    = cap_state["ρ"][m][end]
                λ_prev = cap_state["λ"][m][end]
                if isempty(cap_vec)
                    push!(cap_state["λ"][m], copy(λ_prev))
                else
                    λ_new = [_cap_scalar(λ_prev) + ρ_m * (_cap_scalar(cap_vec) - _cap_scalar(z_vec))]
                    push!(cap_state["λ"][m], λ_new)
                end
            end
        end

        # ------------------------------------------------------------------
        # Dual residuals (standard + contract + cap)
        # ------------------------------------------------------------------
        @timeit TO "Dual residuals" begin
            nE  = elec_market["nAgents"]
            nH  = H2_market["nAgents"]
            nEG = elec_GC_market["nAgents"]
            nHG = H2_GC_market["nAgents"]
            nEP = EP_market["nAgents"]
            nC  = ppa_market["nAgents"]

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

                ppa_vres = get(ppa_market, "ppa_vres", String[])
                h2_ids = agents[:H2]
                for vres_id in ppa_vres
                    nC = 2
                    net_vres = results["ppa"][vres_id][end]
                    net_vres_prev = length(results["ppa"][vres_id]) < 2 ? zeros(shp...) : results["ppa"][vres_id][end-1]
                    net_elec = zeros(shp...)
                    net_elec_prev = zeros(shp...)
                    for h2_id in h2_ids
                        if haskey(results["ppa_from"][h2_id], vres_id) && !isempty(results["ppa_from"][h2_id][vres_id])
                            net_elec .+= .-results["ppa_from"][h2_id][vres_id][end]
                            if length(results["ppa_from"][h2_id][vres_id]) >= 2
                                net_elec_prev .+= .-results["ppa_from"][h2_id][vres_id][end-1]
                            end
                        end
                    end
                    sum_net = net_vres .+ net_elec
                    sum_net_prev = net_vres_prev .+ net_elec_prev
                    diff_vres = (net_vres .- sum_net ./ (nC + 1)) .- (net_vres_prev .- sum_net_prev ./ (nC + 1))
                    diff_elec = (net_elec .- sum_net ./ (nC + 1)) .- (net_elec_prev .- sum_net_prev ./ (nC + 1))
                    C = ADMM_state["ppa"]
                    dual_contract = sum((C["ρ"][vres_id][end] .* diff_vres).^2) + sum((C["ρ"][vres_id][end] .* diff_elec).^2)
                    push!(C["Dual"][vres_id], sqrt(dual_contract))

                    cap_vres = isempty(results["ppa_cap"][vres_id]) ? 0.0 : results["ppa_cap"][vres_id][end]
                    cap_vres_prev = length(results["ppa_cap"][vres_id]) < 2 ? 0.0 : results["ppa_cap"][vres_id][end-1]
                    cap_elec = 0.0
                    cap_elec_prev = 0.0
                    for h2_id in h2_ids
                        if haskey(results["ppa_cap_from"][h2_id], vres_id) && !isempty(results["ppa_cap_from"][h2_id][vres_id])
                            cap_elec += -results["ppa_cap_from"][h2_id][vres_id][end]
                            if length(results["ppa_cap_from"][h2_id][vres_id]) >= 2
                                cap_elec_prev += -results["ppa_cap_from"][h2_id][vres_id][end-1]
                            end
                        end
                    end
                    sum_cap = cap_vres - cap_elec
                    sum_cap_prev = cap_vres_prev - cap_elec_prev
                    diff_cap_vres = (cap_vres - sum_cap / (nC + 1)) - (cap_vres_prev - sum_cap_prev / (nC + 1))
                    diff_cap_elec = ((-cap_elec) - sum_cap / (nC + 1)) - ((-cap_elec_prev) - sum_cap_prev / (nC + 1))
                    dual_contract_cap = (C["ρ_cap"][vres_id][end] * diff_cap_vres)^2 + (C["ρ_cap"][vres_id][end] * diff_cap_elec)^2
                    push!(C["Dual_cap"][vres_id], sqrt(dual_contract_cap))
                end

                hpa_h2 = get(hpa_market, "hpa_h2", String[])
                off_ids = agents[:offtaker]
                C_hpa = ADMM_state["hpa"]
                for h2_id in hpa_h2
                    nC = 2
                    net_h2 = results["hpa"][h2_id][end]
                    net_h2_prev = length(results["hpa"][h2_id]) < 2 ? zeros(shp...) : results["hpa"][h2_id][end-1]
                    net_off = zeros(shp...)
                    net_off_prev = zeros(shp...)
                    for off_id in off_ids
                        if haskey(results["hpa_from"][off_id], h2_id) && !isempty(results["hpa_from"][off_id][h2_id])
                            net_off .+= .-results["hpa_from"][off_id][h2_id][end]
                            if length(results["hpa_from"][off_id][h2_id]) >= 2
                                net_off_prev .+= .-results["hpa_from"][off_id][h2_id][end-1]
                            end
                        end
                    end
                    sum_net = net_h2 .+ net_off
                    sum_net_prev = net_h2_prev .+ net_off_prev
                    diff_h2 = (net_h2 .- sum_net ./ (nC + 1)) .- (net_h2_prev .- sum_net_prev ./ (nC + 1))
                    diff_off = (net_off .- sum_net ./ (nC + 1)) .- (net_off_prev .- sum_net_prev ./ (nC + 1))
                    dual_contract = sum((C_hpa["ρ"][h2_id][end] .* diff_h2).^2) + sum((C_hpa["ρ"][h2_id][end] .* diff_off).^2)
                    push!(C_hpa["Dual"][h2_id], sqrt(dual_contract))

                    cap_h2 = isempty(results["hpa_cap"][h2_id]) ? 0.0 : results["hpa_cap"][h2_id][end]
                    cap_h2_prev = length(results["hpa_cap"][h2_id]) < 2 ? 0.0 : results["hpa_cap"][h2_id][end-1]
                    cap_off = 0.0
                    cap_off_prev = 0.0
                    for off_id in off_ids
                        if haskey(results["hpa_cap_from"][off_id], h2_id) && !isempty(results["hpa_cap_from"][off_id][h2_id])
                            cap_off += -results["hpa_cap_from"][off_id][h2_id][end]
                            if length(results["hpa_cap_from"][off_id][h2_id]) >= 2
                                cap_off_prev += -results["hpa_cap_from"][off_id][h2_id][end-1]
                            end
                        end
                    end
                    sum_cap = cap_h2 - cap_off
                    sum_cap_prev = cap_h2_prev - cap_off_prev
                    diff_cap_h2 = (cap_h2 - sum_cap / (nC + 1)) - (cap_h2_prev - sum_cap_prev / (nC + 1))
                    diff_cap_off = ((-cap_off) - sum_cap / (nC + 1)) - ((-cap_off_prev) - sum_cap_prev / (nC + 1))
                    dual_contract_cap = (C_hpa["ρ_cap"][h2_id][end] * diff_cap_h2)^2 + (C_hpa["ρ_cap"][h2_id][end] * diff_cap_off)^2
                    push!(C_hpa["Dual_cap"][h2_id], sqrt(dual_contract_cap))
                end

                scales_du = ADMM_state["ResidualScale"]["Dual"]
                for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                    rd = ADMM_state["Residuals"]["Dual"][key][end]
                    if scales_du[key] == 0.0 && rd < Inf
                        scales_du[key] = rd
                    end
                end
                # ----------------------------------------------------------
                # Capacity dual residuals — per-agent split (Δz-based)
                # (mirrors ADMM.jl; see DOCUMENTATION.md §5.4).
                # ----------------------------------------------------------
                cap_state = ADMM_state["Capacity"]
                dual_cap_sq = 0.0
                for m in cap_agents
                    z_hist = cap_state["z"][m]
                    ρ_m    = cap_state["ρ"][m][end]
                    if length(z_hist) >= 2
                        z_new = _cap_scalar(z_hist[end])
                        z_old = _cap_scalar(z_hist[end - 1])
                        local_s = abs(ρ_m * (z_new - z_old))
                        push!(cap_state["Dual"][m], local_s)
                        if cap_state["ResidualScale_Dual"][m] == 0.0 && local_s > 0.0 && isfinite(local_s)
                            cap_state["ResidualScale_Dual"][m] = local_s
                        end
                        dual_cap_sq += local_s^2
                    else
                        push!(cap_state["Dual"][m], Inf)
                    end
                end
                dual_cap = isempty(cap_agents) || any(isinf, [cap_state["Dual"][m][end] for m in cap_agents]) ? Inf : sqrt(dual_cap_sq)
                push!(ADMM_state["Residuals"]["Dual"]["cap"], dual_cap)
                if ADMM_state["ResidualScale"]["Dual"]["cap"] == 0.0 && isfinite(dual_cap) && dual_cap > 0.0
                    ADMM_state["ResidualScale"]["Dual"]["cap"] = dual_cap
                end
                C = ADMM_state["ppa"]
                for vres_id in ppa_vres
                    rd = C["Dual"][vres_id][end]
                    if C["ResidualScale_Dual"][vres_id] == 0.0 && rd < Inf
                        C["ResidualScale_Dual"][vres_id] = rd
                    end
                    rd = C["Dual_cap"][vres_id][end]
                    if C["ResidualScale_Dual_cap"][vres_id] == 0.0 && rd < Inf
                        C["ResidualScale_Dual_cap"][vres_id] = rd
                    end
                end
                C_hpa = ADMM_state["hpa"]
                for h2_id in hpa_h2
                    rd = C_hpa["Dual"][h2_id][end]
                    if C_hpa["ResidualScale_Dual"][h2_id] == 0.0 && rd < Inf
                        C_hpa["ResidualScale_Dual"][h2_id] = rd
                    end
                    rd = C_hpa["Dual_cap"][h2_id][end]
                    if C_hpa["ResidualScale_Dual_cap"][h2_id] == 0.0 && rd < Inf
                        C_hpa["ResidualScale_Dual_cap"][h2_id] = rd
                    end
                end
            else
                for key in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                    push!(ADMM_state["Residuals"]["Dual"][key], Inf)
                end
                push!(ADMM_state["Residuals"]["Dual"]["cap"], Inf)
                # Keep per-agent capacity dual history aligned with iterations
                # (iter 1 has undefined dual because z^{k-1} does not exist).
                cap_state = ADMM_state["Capacity"]
                for m in cap_agents
                    push!(cap_state["Dual"][m], Inf)
                end
                C = ADMM_state["ppa"]
                for vres_id in get(ppa_market, "ppa_vres", String[])
                    push!(C["Dual"][vres_id], Inf)
                    push!(C["Dual_cap"][vres_id], Inf)
                end
                C_hpa = ADMM_state["hpa"]
                for h2_id in get(hpa_market, "hpa_h2", String[])
                    push!(C_hpa["Dual"][h2_id], Inf)
                    push!(C_hpa["Dual_cap"][h2_id], Inf)
                end
            end
        end

        # ------------------------------------------------------------------
        # Merit tracking (normalized residuals) and controller adaptation
        # ------------------------------------------------------------------
        n_slots = n_ts * n_rd * n_yr
        merit = Dict{String,Float64}()
        for key in market_keys
            rp = ADMM_state["Residuals"]["Primal"][key][end]
            rd = ADMM_state["Residuals"]["Dual"][key][end]
            eps_pr, eps_du = _market_eps_std(key, n_slots)
            m = max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))
            merit[key] = isfinite(m) ? m : 1e12
        end
        C = ADMM_state["ppa"]
        for vres_id in ppa_ids
            rp = C["Primal"][vres_id][end]
            rd = C["Dual"][vres_id][end]
            eps_pr, eps_du = _market_eps_contract(C, vres_id, false, n_slots)
            merit["ppa_" * vres_id] = isfinite(max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))) ?
                                      max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9)) : 1e12
            rp_cap = C["Primal_cap"][vres_id][end]
            rd_cap = C["Dual_cap"][vres_id][end]
            eps_pr_cap, eps_du_cap = _market_eps_contract(C, vres_id, true, n_slots)
            merit["ppa_cap_" * vres_id] = isfinite(max(rp_cap / max(eps_pr_cap, 1e-9), rd_cap / max(eps_du_cap, 1e-9))) ?
                                          max(rp_cap / max(eps_pr_cap, 1e-9), rd_cap / max(eps_du_cap, 1e-9)) : 1e12
        end
        C_hpa = ADMM_state["hpa"]
        for h2_id in hpa_ids
            rp = C_hpa["Primal"][h2_id][end]
            rd = C_hpa["Dual"][h2_id][end]
            eps_pr, eps_du = _market_eps_contract(C_hpa, h2_id, false, n_slots)
            merit["hpa_" * h2_id] = isfinite(max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9))) ?
                                    max(rp / max(eps_pr, 1e-9), rd / max(eps_du, 1e-9)) : 1e12
            rp_cap = C_hpa["Primal_cap"][h2_id][end]
            rd_cap = C_hpa["Dual_cap"][h2_id][end]
            eps_pr_cap, eps_du_cap = _market_eps_contract(C_hpa, h2_id, true, n_slots)
            merit["hpa_cap_" * h2_id] = isfinite(max(rp_cap / max(eps_pr_cap, 1e-9), rd_cap / max(eps_du_cap, 1e-9))) ?
                                        max(rp_cap / max(eps_pr_cap, 1e-9), rd_cap / max(eps_du_cap, 1e-9)) : 1e12
        end

        score = maximum(values(merit))
        if score < best_score
            best_score = score
            best_iter = iter
            stall_count = 0
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                best_λ[mkt] = copy(results["λ"][mkt][end])
                best_ρ[mkt] = ADMM_state["ρ"][mkt][end]
            end
            # Capacity is per-agent: snapshot every agent's current ρ_m.
            for m in cap_agents
                best_ρ_cap[m] = ADMM_state["Capacity"]["ρ"][m][end]
            end
            for vres_id in ppa_ids
                best_λ_ppa[vres_id] = copy(results["λ_ppa"][vres_id][end])
                best_ρ_ppa[vres_id] = C["ρ"][vres_id][end]
                best_ρ_ppa_cap[vres_id] = C["ρ_cap"][vres_id][end]
            end
            for h2_id in hpa_ids
                best_λ_hpa[h2_id] = copy(results["λ_hpa"][h2_id][end])
                best_ρ_hpa[h2_id] = C_hpa["ρ"][h2_id][end]
                best_ρ_hpa_cap[h2_id] = C_hpa["ρ_cap"][h2_id][end]
            end
        else
            stall_count += 1
        end

        if iter > 1
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                rp_prev = ADMM_state["Residuals"]["Primal"][mkt][end-1]
                rd_prev = ADMM_state["Residuals"]["Dual"][mkt][end-1]
                eps_pr_prev, eps_du_prev = _market_eps_std(mkt, n_slots)
                merit_prev = max(rp_prev / max(eps_pr_prev, 1e-9), rd_prev / max(eps_du_prev, 1e-9))
                merit_now = merit[mkt]
                if merit_now > 1.02 * merit_prev
                    η_scale[mkt] = max(0.15, 0.85 * η_scale[mkt])
                elseif merit_now < 0.98 * merit_prev
                    η_scale[mkt] = min(1.0, 1.03 * η_scale[mkt])
                end
            end
            for vres_id in ppa_ids
                rp_prev = C["Primal"][vres_id][end-1]
                rd_prev = C["Dual"][vres_id][end-1]
                eps_pr_prev, eps_du_prev = _market_eps_contract(C, vres_id, false, n_slots)
                merit_prev = max(rp_prev / max(eps_pr_prev, 1e-9), rd_prev / max(eps_du_prev, 1e-9))
                merit_now = merit["ppa_" * vres_id]
                if merit_now > 1.02 * merit_prev
                    η_scale_ppa[vres_id] = max(0.15, 0.85 * η_scale_ppa[vres_id])
                elseif merit_now < 0.98 * merit_prev
                    η_scale_ppa[vres_id] = min(1.0, 1.03 * η_scale_ppa[vres_id])
                end
            end
            for h2_id in hpa_ids
                rp_prev = C_hpa["Primal"][h2_id][end-1]
                rd_prev = C_hpa["Dual"][h2_id][end-1]
                eps_pr_prev, eps_du_prev = _market_eps_contract(C_hpa, h2_id, false, n_slots)
                merit_prev = max(rp_prev / max(eps_pr_prev, 1e-9), rd_prev / max(eps_du_prev, 1e-9))
                merit_now = merit["hpa_" * h2_id]
                if merit_now > 1.02 * merit_prev
                    η_scale_hpa[h2_id] = max(0.15, 0.85 * η_scale_hpa[h2_id])
                elseif merit_now < 0.98 * merit_prev
                    η_scale_hpa[h2_id] = min(1.0, 1.03 * η_scale_hpa[h2_id])
                end
            end
        end

        # ------------------------------------------------------------------
        # Price update (standard + contract)
        # ------------------------------------------------------------------
        @timeit TO "Update prices" begin
            # Scale-aware damping of dual updates, consistent with base ADMM.
            η_min = 0.25
            eps_abs = ADMM_state["EpsilonAbs"]
            eps_rel = ADMM_state["EpsilonRel"]
            n_slots_upd = max(1, get(ADMM_state, "n_slots", n_slots))
            sqrt_n = sqrt(n_slots_upd)
            scale_pr = ADMM_state["ResidualScale"]["Primal"]
            scale_du = ADMM_state["ResidualScale"]["Dual"]
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                rp = ADMM_state["Residuals"]["Primal"][mkt][end]
                rd = ADMM_state["Residuals"]["Dual"][mkt][end]
                base = max(rp, rd)
                sp = max(scale_pr[mkt], 1.0)
                sd = max(scale_du[mkt], 1.0)
                eps_pr = eps_abs * sqrt_n + eps_rel * sp
                eps_du = eps_abs * sqrt_n + eps_rel * sd
                eps_m = max(eps_pr, eps_du)
                η_raw = base >= 1.5 * eps_m ? 1.0 : max(η_min, base / max(1.5 * eps_m, 1e-9))
                η = η_scale[mkt] * η_raw
                push!(results["λ"][mkt],
                      results["λ"][mkt][end] .- η .* ADMM_state["ρ"][mkt][end] .* ADMM_state["Imbalances"][mkt][end])
            end
            results["λ"]["H2_GC"][end] .= max.(results["λ"]["H2_GC"][end], 0.0)

            # PPA pool: per-VRES λ_ppa (€/MWh). No price for ppa_cap.
            ppa_vres = get(ppa_market, "ppa_vres", String[])
            C = ADMM_state["ppa"]
            for vres_id in ppa_vres
                rp = C["Primal"][vres_id][end]
                rd = C["Dual"][vres_id][end]
                base = max(rp, rd)
                eps_pr, eps_du = _market_eps_contract(C, vres_id, false, n_slots_upd)
                eps_m = max(eps_pr, eps_du)
                η_raw = base >= 1.5 * eps_m ? 1.0 : max(η_min, base / max(1.5 * eps_m, 1e-9))
                η = η_scale_ppa[vres_id] * η_raw
                push!(results["λ_ppa"][vres_id],
                      results["λ_ppa"][vres_id][end] .- η .* C["ρ"][vres_id][end] .* C["Imbalances"][vres_id][end])
                results["λ_ppa"][vres_id][end] .= max.(results["λ_ppa"][vres_id][end], 0.0)
            end

            hpa_h2 = get(hpa_market, "hpa_h2", String[])
            C_hpa = ADMM_state["hpa"]
            for h2_id in hpa_h2
                rp = C_hpa["Primal"][h2_id][end]
                rd = C_hpa["Dual"][h2_id][end]
                base = max(rp, rd)
                eps_pr, eps_du = _market_eps_contract(C_hpa, h2_id, false, n_slots_upd)
                eps_m = max(eps_pr, eps_du)
                η_raw = base >= 1.5 * eps_m ? 1.0 : max(η_min, base / max(1.5 * eps_m, 1e-9))
                η = η_scale_hpa[h2_id] * η_raw
                push!(results["λ_hpa"][h2_id],
                      results["λ_hpa"][h2_id][end] .- η .* C_hpa["ρ"][h2_id][end] .* C_hpa["Imbalances"][h2_id][end])
                results["λ_hpa"][h2_id][end] .= max.(results["λ_hpa"][h2_id][end], 0.0)
            end
        end

        @timeit TO "Price means" begin
            push!(ADMM_state["PriceHistory"]["elec"],    mean(results["λ"]["elec"][end]))
            push!(ADMM_state["PriceHistory"]["H2"],      mean(results["λ"]["H2"][end]))
            push!(ADMM_state["PriceHistory"]["elec_GC"], mean(results["λ"]["elec_GC"][end]))
            push!(ADMM_state["PriceHistory"]["H2_GC"],   mean(results["λ"]["H2_GC"][end]))
            push!(ADMM_state["PriceHistory"]["EP"],      mean(results["λ"]["EP"][end]))
            C = ADMM_state["ppa"]
            for vres_id in get(ppa_market, "ppa_vres", String[])
                push!(C["PriceHistory"][vres_id], mean(results["λ_ppa"][vres_id][end]))
            end
            C_hpa = ADMM_state["hpa"]
            for h2_id in get(hpa_market, "hpa_h2", String[])
                push!(C_hpa["PriceHistory"][h2_id], mean(results["λ_hpa"][h2_id][end]))
            end
        end

        @timeit TO "Update ρ" begin
            update_rho_contracts!(ADMM_state, iter)
        end

        # Anti-stall recovery: if the run has moved far away from the best
        # checkpoint for a long window, damp steps immediately and only
        # occasionally blend toward checkpoint λ/ρ. This preserves anti-drift
        # benefits while avoiding repeating hard-reset cycles.
        if enable_recovery_steering && stall_count >= restart_patience &&
           score > restart_factor * best_score && !isempty(best_λ)
            for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                η_scale[mkt] = max(0.15, 0.9 * η_scale[mkt])
            end
            for vres_id in ppa_ids
                η_scale_ppa[vres_id] = max(0.15, 0.9 * η_scale_ppa[vres_id])
            end
            for h2_id in hpa_ids
                η_scale_hpa[h2_id] = max(0.15, 0.9 * η_scale_hpa[h2_id])
            end
            can_rollback = rollback_count < max_rollbacks && (iter - last_rollback_iter) >= rollback_cooldown
            if can_rollback
                α = rollback_blend
                for mkt in ("elec", "H2", "elec_GC", "H2_GC", "EP")
                    results["λ"][mkt][end] .= (1.0 - α) .* results["λ"][mkt][end] .+ α .* best_λ[mkt]
                    ADMM_state["ρ"][mkt][end] = (1.0 - α) * ADMM_state["ρ"][mkt][end] + α * best_ρ[mkt]
                    η_scale[mkt] = max(0.15, 0.85 * η_scale[mkt])
                end
                # Capacity rollback: blend each cap agent's ρ_m toward its best.
                for m in cap_agents
                    if haskey(best_ρ_cap, m)
                        cur = ADMM_state["Capacity"]["ρ"][m][end]
                        ADMM_state["Capacity"]["ρ"][m][end] = (1.0 - α) * cur + α * best_ρ_cap[m]
                    end
                end
                C = ADMM_state["ppa"]
                for vres_id in ppa_ids
                    if haskey(best_λ_ppa, vres_id)
                        results["λ_ppa"][vres_id][end] .= (1.0 - α) .* results["λ_ppa"][vres_id][end] .+ α .* best_λ_ppa[vres_id]
                        C["ρ"][vres_id][end] = (1.0 - α) * C["ρ"][vres_id][end] + α * best_ρ_ppa[vres_id]
                        C["ρ_cap"][vres_id][end] = (1.0 - α) * C["ρ_cap"][vres_id][end] + α * best_ρ_ppa_cap[vres_id]
                        η_scale_ppa[vres_id] = max(0.15, 0.85 * η_scale_ppa[vres_id])
                    end
                end
                C_hpa = ADMM_state["hpa"]
                for h2_id in hpa_ids
                    if haskey(best_λ_hpa, h2_id)
                        results["λ_hpa"][h2_id][end] .= (1.0 - α) .* results["λ_hpa"][h2_id][end] .+ α .* best_λ_hpa[h2_id]
                        C_hpa["ρ"][h2_id][end] = (1.0 - α) * C_hpa["ρ"][h2_id][end] + α * best_ρ_hpa[h2_id]
                        C_hpa["ρ_cap"][h2_id][end] = (1.0 - α) * C_hpa["ρ_cap"][h2_id][end] + α * best_ρ_hpa_cap[h2_id]
                        η_scale_hpa[h2_id] = max(0.15, 0.85 * η_scale_hpa[h2_id])
                    end
                end
                rollback_count += 1
                last_rollback_iter = iter
            end
            stall_count = 0
        end

        set_description(iterations, "")

        # ------------------------------------------------------------------
        # Convergence check (all markets)
        # ------------------------------------------------------------------
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

        C = ADMM_state["ppa"]
        contract_ok = true
        for vres_id in get(ppa_market, "ppa_vres", String[])
            rp = C["Primal"][vres_id][end]
            rd = C["Dual"][vres_id][end]
            sp = max(C["ResidualScale_Primal"][vres_id], 1.0)
            sd = max(C["ResidualScale_Dual"][vres_id], 1.0)
            eps_pr = eps_abs * sqrt_n + eps_rel * sp
            eps_du = eps_abs * sqrt_n + eps_rel * sd
            contract_ok = contract_ok && (rp <= eps_pr) && (rd <= eps_du)
        end
        cap_ok = true
        for vres_id in get(ppa_market, "ppa_vres", String[])
            rp = C["Primal_cap"][vres_id][end]
            rd = C["Dual_cap"][vres_id][end]
            sp = max(C["ResidualScale_Primal_cap"][vres_id], 1.0)
            sd = max(C["ResidualScale_Dual_cap"][vres_id], 1.0)
            eps_pr = eps_abs * 1.0 + eps_rel * sp
            eps_du = eps_abs * 1.0 + eps_rel * sd
            cap_ok = cap_ok && (rp <= eps_pr) && (rd <= eps_du)
        end
        C_hpa = ADMM_state["hpa"]
        hpa_ok = true
        for h2_id in get(hpa_market, "hpa_h2", String[])
            rp = C_hpa["Primal"][h2_id][end]
            rd = C_hpa["Dual"][h2_id][end]
            sp = max(C_hpa["ResidualScale_Primal"][h2_id], 1.0)
            sd = max(C_hpa["ResidualScale_Dual"][h2_id], 1.0)
            eps_pr = eps_abs * sqrt_n + eps_rel * sp
            eps_du = eps_abs * sqrt_n + eps_rel * sd
            hpa_ok = hpa_ok && (rp <= eps_pr) && (rd <= eps_du)
        end
        hpa_cap_ok = true
        for h2_id in get(hpa_market, "hpa_h2", String[])
            rp = C_hpa["Primal_cap"][h2_id][end]
            rd = C_hpa["Dual_cap"][h2_id][end]
            sp = max(C_hpa["ResidualScale_Primal_cap"][h2_id], 1.0)
            sd = max(C_hpa["ResidualScale_Dual_cap"][h2_id], 1.0)
            eps_pr = eps_abs * 1.0 + eps_rel * sp
            eps_du = eps_abs * 1.0 + eps_rel * sd
            hpa_cap_ok = hpa_cap_ok && (rp <= eps_pr) && (rd <= eps_du)
        end

        # Capacity consensus: use relaxed tolerance (see file header).
        # Effective eps = (eps_pr, eps_du) * cap_tol_relax so we accept larger residuals.
        # ----------------------------------------------------------------
        # Per-agent capacity convergence (new equality-split formulation):
        # every cap-owning agent must satisfy its own Boyd test on r_m, s_m.
        # The optional cap_tol_relax knob still applies (multiplies the
        # right-hand side) to keep backwards compatibility with the
        # data.yaml configuration. See DOCUMENTATION.md §5.4.
        # ----------------------------------------------------------------
        cap_tol_relax = get(get(data, "ADMM", Dict()), "cap_tol_relax", CAP_CONSENSUS_TOL_RELAX_DEFAULT)
        sqrt_y = 1.0
        cap_state = ADMM_state["Capacity"]
        cap_consensus_ok = true
        for m in cap_agents
            rp_m = cap_state["Primal"][m][end]
            rd_m = cap_state["Dual"][m][end]
            sp_m = max(cap_state["ResidualScale_Primal"][m], 1.0)
            sd_m = max(cap_state["ResidualScale_Dual"][m], 1.0)
            eps_pr_m = cap_tol_relax * (eps_abs * sqrt_y + eps_rel * sp_m)
            eps_du_m = cap_tol_relax * (eps_abs * sqrt_y + eps_rel * sd_m)
            if !(isfinite(rp_m) && isfinite(rd_m) &&
                 rp_m <= eps_pr_m && rd_m <= eps_du_m)
                cap_consensus_ok = false
                break
            end
        end
        if (within_tol("elec") && within_tol("H2") && within_tol("elec_GC") &&
            within_tol("H2_GC") && within_tol("EP") &&
            contract_ok && cap_ok && hpa_ok && hpa_cap_ok && cap_consensus_ok)
            convergence = 1
        end

        ADMM_state["n_iter"] = iter
    end

    println()
    ADMM_state["converged"] = (convergence == 1)
    if !ADMM_state["converged"]
        @printf("ADMM reached max_iter without convergence (best score %.4f at iter %d).\n",
                best_score, best_iter)
    end

    return nothing
end
