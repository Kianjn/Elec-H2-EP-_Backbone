# ==============================================================================
# save_results.jl — Write all result and diagnostic CSVs
# ==============================================================================
#
# PURPOSE:
#   Writes into market_exposure_results/: (1) ADMM_Convergence.csv — per-iteration
#   primal and dual residuals for every market. (2) ADMM_Diagnostics.csv — per-
#   iteration rho, mean price, and mean imbalance for every market. (3) One
#   *_Market_History.csv per market (Electricity, Hydrogen, Electricity_GC,
#   H2_GC, End_Product) with iter, rho, price_mean, imb_mean, primal_res, dual_res.
#   (4) Agent_Summary.csv — one row per agent with AgentID and Group (power, H2,
#   offtaker, elec_GC_demand).
#
# ARGUMENTS:
#   mdict — Dict of JuMP models (not used here; could be used to dump variables).
#   elec_market, H2_market, ... — Market dicts (for reference; most data comes
#     from ADMM_state and results).
#   ADMM_state — Contains ρ, PriceHistory, ImbalanceMean, Imbalances, Residuals.
#   results — Contains λ and agents/markets refs.
#   agents — Dict of agent lists for the summary table.
#
# ==============================================================================

function save_results(mdict::Dict, elec_market::Dict, H2_market::Dict, elec_GC_market::Dict,
                     H2_GC_market::Dict, ADMM_state::Dict, results::Dict, agents::Dict)
    results_dir = joinpath(@__DIR__, "..", "market_exposure_results")
    isdir(results_dir) || mkdir(results_dir)

    # Determine the number of ADMM iterations actually performed.
    # WHY length(Imbalances): each iteration appends exactly one imbalance value,
    # so the list length equals the iteration count.
    # NOTE: ρ has length n_it+1 because it includes the initial value set *before*
    # the first iteration plus one update per iteration. When building per-iteration
    # DataFrames below we therefore slice ρ as [1:n_it] to align with the other
    # per-iteration vectors (prices, imbalances, residuals) that have length n_it.
    n_it = length(ADMM_state["Imbalances"]["elec"])

    # ── ADMM_Convergence.csv ─────────────────────────────────────────────
    # One row per ADMM iteration with primal and dual residuals for every
    # market. Used to generate convergence plots that show whether the ADMM
    # algorithm is approaching feasibility (primal) and stationarity (dual).
    conv_df = DataFrame(
        iter          = 1:n_it,
        elec_primal   = ADMM_state["Residuals"]["Primal"]["elec"],
        elec_dual     = ADMM_state["Residuals"]["Dual"]["elec"],
        H2_primal     = ADMM_state["Residuals"]["Primal"]["H2"],
        H2_dual       = ADMM_state["Residuals"]["Dual"]["H2"],
        elec_GC_primal = ADMM_state["Residuals"]["Primal"]["elec_GC"],
        elec_GC_dual   = ADMM_state["Residuals"]["Dual"]["elec_GC"],
        H2_GC_primal   = ADMM_state["Residuals"]["Primal"]["H2_GC"],
        H2_GC_dual     = ADMM_state["Residuals"]["Dual"]["H2_GC"],
        EP_primal      = ADMM_state["Residuals"]["Primal"]["EP"],
        EP_dual        = ADMM_state["Residuals"]["Dual"]["EP"],
    )
    CSV.write(joinpath(results_dir, "ADMM_Convergence.csv"), conv_df)

    # ── ADMM_Diagnostics.csv ──────────────────────────────────────────────
    # One row per ADMM iteration with ρ (penalty parameter), mean price, and
    # mean imbalance for each market. Used to understand how prices and
    # imbalances evolve across iterations and to diagnose oscillation or
    # divergence. ρ is sliced [1:n_it] because its vector is one element
    # longer than the other per-iteration vectors (see n_it note above).
    diag_df = DataFrame(
        iter             = 1:n_it,
        elec_rho         = ADMM_state["ρ"]["elec"][1:n_it],
        elec_price_mean  = ADMM_state["PriceHistory"]["elec"],
        elec_imb_mean    = ADMM_state["ImbalanceMean"]["elec"],
        H2_rho           = ADMM_state["ρ"]["H2"][1:n_it],
        H2_price_mean    = ADMM_state["PriceHistory"]["H2"],
        H2_imb_mean      = ADMM_state["ImbalanceMean"]["H2"],
        elec_GC_rho        = ADMM_state["ρ"]["elec_GC"][1:n_it],
        elec_GC_price_mean = ADMM_state["PriceHistory"]["elec_GC"],
        elec_GC_imb_mean   = ADMM_state["ImbalanceMean"]["elec_GC"],
        H2_GC_rho          = ADMM_state["ρ"]["H2_GC"][1:n_it],
        H2_GC_price_mean   = ADMM_state["PriceHistory"]["H2_GC"],
        H2_GC_imb_mean     = ADMM_state["ImbalanceMean"]["H2_GC"],
        EP_rho             = ADMM_state["ρ"]["EP"][1:n_it],
        EP_price_mean      = ADMM_state["PriceHistory"]["EP"],
        EP_imb_mean        = ADMM_state["ImbalanceMean"]["EP"],
    )
    CSV.write(joinpath(results_dir, "ADMM_Diagnostics.csv"), diag_df)

    # ── Per-market history CSVs ──────────────────────────────────────────
    # Same convergence + diagnostic data reorganized into one CSV per market
    # (e.g. Electricity_Market_History.csv). This makes it easier to plot or
    # analyse a single market without filtering the combined tables above.
    markets = Dict(
        "elec"    => "Electricity",
        "H2"      => "Hydrogen",
        "elec_GC" => "Electricity_GC",
        "H2_GC"   => "H2_GC",
        "EP"      => "End_Product",
    )
    for (key, name) in markets
        df = DataFrame(
            iter       = 1:n_it,
            rho        = ADMM_state["ρ"][key][1:n_it],
            price_mean = ADMM_state["PriceHistory"][key],
            imb_mean   = ADMM_state["ImbalanceMean"][key],
            primal_res = ADMM_state["Residuals"]["Primal"][key],
            dual_res   = ADMM_state["Residuals"]["Dual"][key],
        )
        CSV.write(joinpath(results_dir, string(name, "_Market_History.csv")), df)
    end

    # --------------------------------------------------------------------------
    # Agent_Summary.csv — GROUP MEMBERSHIP
    # --------------------------------------------------------------------------
    # One row per agent recording which sector / group it belongs to (power,
    # H2, offtaker, elec_GC_demand). This is a look-up table so that
    # downstream analysis can map AgentIDs to their roles without re-parsing
    # data.yaml.
    agent_rows = String[]
    agent_types = String[]
    for k in (:power, :H2, :offtaker, :elec_GC_demand)
        haskey(agents, k) || continue
        for id in agents[k]
            push!(agent_rows, String(id))
            push!(agent_types, String(k))
        end
    end
    agents_df = DataFrame(AgentID = agent_rows, Group = agent_types)
    CSV.write(joinpath(results_dir, "Agent_Summary.csv"), agents_df)

    # --------------------------------------------------------------------------
    # Agent_Quantities_Final.csv — PER-AGENT NET QUANTITIES AT FINAL ITERATION
    # --------------------------------------------------------------------------
    # Builds a compact summary showing each agent's total energy traded in
    # every market at the last ADMM iteration.
    #
    # Net position sign convention (consistent with the rest of the model):
    #   +  = supply (selling into the market)
    #   −  = demand (buying from the market)
    #
    # _total_last helper: takes the history dict for a given market, selects
    # the 3D array from the *last* ADMM iteration (arr_dict[id][end]), and
    # sums over all (jh, jd, jy) entries to collapse it into a single scalar.
    # This scalar is the total energy traded by that agent across the full
    # modeled year (all hours × representative days × scenario years).

    function _total_last(arr_dict::Dict, id)
        lst = arr_dict[id]
        return isempty(lst) ? 0.0 : sum(lst[end])
    end

    agent_ids_q = String[]
    group_q     = String[]
    elec_q      = Float64[]
    h2_q        = Float64[]
    elec_gc_q   = Float64[]
    h2_gc_q     = Float64[]
    ep_q        = Float64[]

    for k in (:power, :H2, :offtaker, :elec_GC_demand)
        haskey(agents, k) || continue
        for id in agents[k]
            push!(agent_ids_q, String(id))
            push!(group_q, String(k))
            push!(elec_q,    _total_last(results["g"],       id))
            push!(h2_q,      _total_last(results["h2"],      id))
            push!(elec_gc_q, _total_last(results["elec_GC"], id))
            push!(h2_gc_q,   _total_last(results["H2_GC"],   id))
            push!(ep_q,      _total_last(results["EP"],      id))
        end
    end

    agent_q_df = DataFrame(
        AgentID        = agent_ids_q,
        Group          = group_q,
        elec_net_sum   = elec_q,
        H2_net_sum     = h2_q,
        elec_GC_net_sum = elec_gc_q,
        H2_GC_net_sum  = h2_gc_q,
        EP_net_sum     = ep_q,
    )
    CSV.write(joinpath(results_dir, "Agent_Quantities_Final.csv"), agent_q_df)

    # --------------------------------------------------------------------------
    # Offtaker_GC_Diagnostics.csv — GREEN-CERTIFICATE COMPLIANCE PER OFFTAKER
    # --------------------------------------------------------------------------
    # For each offtaker agent, exports total end-product (EP) output, total H₂
    # consumed, total H₂ green certificates (GCs) consumed, the resulting GC
    # share, the regulatory mandate (γ_GC), and the slack (share − mandate).
    #
    # Sign convention: offtakers *buy* H₂ and H₂ GCs, so their net positions
    # in those markets are negative. We negate h2_net and h2gc_net below to
    # obtain the positive quantity consumed, which is more intuitive for
    # compliance reporting.
    #
    # gc_share = H₂ GCs consumed / EP produced — fraction of output backed by
    #            green certificates.
    # gc_slack = gc_share − γ_GC mandate. Positive → compliant; negative → short.

    if haskey(agents, :offtaker)
        off_ids    = agents[:offtaker]
        off_agent  = String[]
        off_type   = String[]
        ep_total   = Float64[]
        h2_in_tot  = Float64[]
        h2_gc_tot  = Float64[]
        gc_share   = Float64[]
        gc_mandate = Float64[]
        gc_slack   = Float64[]

        for id in off_ids
            m = mdict[id]
            t = String(get(m.ext[:parameters], :Type, ""))
            γ = get(m.ext[:parameters], :gamma_GC, 0.42)

            ep_list   = results["EP"][id]
            h2_list   = results["h2"][id]
            h2gc_list = results["H2_GC"][id]

            ep_sum   = isempty(ep_list)   ? 0.0 : sum(ep_list[end])
            h2_net   = isempty(h2_list)   ? 0.0 : sum(h2_list[end])      # < 0 for offtakers (they buy H₂)
            h2gc_net = isempty(h2gc_list) ? 0.0 : sum(h2gc_list[end])    # < 0 for offtakers (they buy H₂ GCs)

            # Negate to convert negative net positions into positive consumed
            # quantities, which are easier to interpret in a compliance context.
            h2_in_sum  = -h2_net       # total H₂ consumed (positive)
            h2_gc_sum  = -h2gc_net     # total H₂ GCs consumed (positive)
            # gc_share: fraction of EP output backed by green H₂ certificates
            share      = (ep_sum > 0 && h2_gc_sum > 0) ? h2_gc_sum / ep_sum : 0.0
            # slack > 0 means the offtaker exceeds its green mandate
            slack      = share - γ

            push!(off_agent,  String(id))
            push!(off_type,   t)
            push!(ep_total,   ep_sum)
            push!(h2_in_tot,  h2_in_sum)
            push!(h2_gc_tot,  h2_gc_sum)
            push!(gc_share,   share)
            push!(gc_mandate, γ)
            push!(gc_slack,   slack)
        end

        off_df = DataFrame(
            AgentID      = off_agent,
            Type         = off_type,
            EP_total     = ep_total,
            H2_in_total  = h2_in_tot,
            H2_GC_total  = h2_gc_tot,
            GC_share     = gc_share,
            GC_mandate   = gc_mandate,
            GC_slack     = gc_slack,
        )
        CSV.write(joinpath(results_dir, "Offtaker_GC_Diagnostics.csv"), off_df)
    end

    # --------------------------------------------------------------------------
    # H2_Producer_Diagnostics.csv — ELECTROLYZER GREEN-FRACTION SUMMARY
    # --------------------------------------------------------------------------
    # For each hydrogen-producing agent, summarizes total H₂ production,
    # total H₂ green certificates (GCs) issued, and the ratio GC/H₂.
    # The ratio indicates what fraction of this producer's hydrogen output
    # is certified green (i.e. backed by renewable electricity GCs).

    if haskey(agents, :H2)
        h2_ids        = agents[:H2]
        el_agent      = String[]
        h2_prod_total = Float64[]
        h2_gc_total   = Float64[]
        gc_to_h2      = Float64[]

        for id in h2_ids
            h2_list   = results["h2"][id]
            h2gc_list = results["H2_GC"][id]

            h2_sum   = isempty(h2_list)   ? 0.0 : sum(h2_list[end])      # > 0 for producers (they sell H₂)
            h2gc_sum = isempty(h2gc_list) ? 0.0 : sum(h2gc_list[end])    # > 0 when issuing H₂ GCs
            # Ratio of H₂ GCs issued to total H₂ produced: what fraction of
            # this producer's output is certified as green hydrogen.
            ratio    = h2_sum != 0.0 ? h2gc_sum / h2_sum : 0.0

            push!(el_agent,      String(id))
            push!(h2_prod_total, h2_sum)
            push!(h2_gc_total,   h2gc_sum)
            push!(gc_to_h2,      ratio)
        end

        el_df = DataFrame(
            AgentID       = el_agent,
            H2_total      = h2_prod_total,
            H2_GC_total   = h2_gc_total,
            GC_per_H2     = gc_to_h2,
        )
        CSV.write(joinpath(results_dir, "H2_Producer_Diagnostics.csv"), el_df)
    end

    return nothing
end
