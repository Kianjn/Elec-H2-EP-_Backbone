# ==============================================================================
# compute_social_risk_metrics.jl — Social CVaR reporting (SP and ADMM)
# ==============================================================================

using JuMP
using DataFrames
using CSV
import Printf: @sprintf, @printf
using Statistics

"""
    empirical_cvar(loss, P, beta) -> (cvar, alpha, u)

Rockafellar–Uryasev CVaR on discrete scenarios: min_alpha alpha + (1/(1-beta)) E[(L-alpha)_+].
"""
function empirical_cvar(loss::AbstractVector{<:Real}, P::AbstractVector{<:Real}, beta::Real)
    loss = Float64.(loss)
    P = Float64.(P)
    n = length(loss)
    length(P) == n || error("loss and P length mismatch")
    sP = sum(P)
    abs(sP - 1.0) > 1e-8 && (P = P ./ sP)
    one_minus_beta = max(1e-6, 1.0 - Float64(beta))

    best_cvar = Inf
    best_alpha = 0.0
    best_u = zeros(n)
    candidates = unique(vcat(loss, [minimum(loss) - 1.0, maximum(loss) + 1.0]))
    for a in candidates
        u = max.(loss .- a, 0.0)
        c = a + sum(P[i] * u[i] for i in eachindex(P)) / one_minus_beta
        if c < best_cvar
            best_cvar, best_alpha, best_u = c, a, u
        end
    end
    return best_cvar, best_alpha, best_u
end

"""Per-year planner welfare for one agent (no market transfers)."""
function _agent_welfare_per_year(m::Model, agents::Dict; JH, JD, JY, W)
    p = m.ext[:parameters]
    ts = m.ext[:timeseries]
    vars = m.ext[:variables]
    atype = String(get(p, :Type, ""))
    Wd = Dict(jy => Dict(jd => W[jd, jy] for jd in JD) for jy in JY)
    wy = Dict{Int, Float64}()

    if atype == "Consumer"
        A_E = get(p, :A_E, 500.0)
        B_E = get(p, :B_E, 0.5)
        d = vars[:d]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * (A_E * value(d[jh, jd, jy]) -
                        0.5 * B_E * value(d[jh, jd, jy])^2) for jh in JH, jd in JD)
        end

    elseif atype == "VRES"
        C = get(p, :MarginalCost, 0.0)
        F_cap = get(p, :FixedCost_per_MW, 0.0)
        cap = vars[:cap_VRES]
        gen_at = if haskey(vars, :g_EOM) && haskey(vars, :g_ppa)
            (jh, jd, jy) -> value(vars[:g_EOM][jh, jd, jy]) + value(vars[:g_ppa][jh, jd, jy])
        else
            g = vars[:g]
            (jh, jd, jy) -> value(g[jh, jd, jy])
        end
        for jy in JY
            op = sum(Wd[jy][jd] * C * gen_at(jh, jd, jy) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif atype == "Conventional"
        stage_cap = get(p, :ConvStageCap, [0.0, 0.0, 0.0])
        stage_base = get(p, :ConvStageBaseCost, zeros(3, length(JY)))
        stage_slope = get(p, :ConvStageSlope, zeros(3, length(JY)))
        if haskey(vars, :g_stage)
            for jy in JY
                cost = 0.0
                for jh in JH, jd in JD, s in 1:3
                    gs = value(vars[:g_stage][s, jh, jd, jy])
                    cost += Wd[jy][jd] * (stage_base[s, jy] * gs + 0.5 * stage_slope[s, jy] * gs^2)
                end
                wy[jy] = -cost
            end
        else
            C = get(p, :MarginalCost, 0.0)
            g = vars[:g]
            for jy in JY
                wy[jy] = -sum(Wd[jy][jd] * C * value(g[jh, jd, jy]) for jh in JH, jd in JD)
            end
        end

    elseif atype == "GreenProducer"
        C_H = get(p, :MarginalCost, get(p, :OperationalCost, 0.0))
        F_cap = get(p, :FixedCost_per_MW_Electrolyzer, 0.0)
        cap = vars[:cap_H2_y]
        h = vars[:h2_out]
        for jy in JY
            op = sum(Wd[jy][jd] * C_H * value(h[jh, jd, jy]) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif haskey(vars, :d_H)
        utility_val = get(p, :Utility, 0.0)
        d_H = vars[:d_H]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * utility_val * value(d_H[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif atype == "GreenOfftaker"
        C_proc = get(p, :MarginalCost, get(p, :ProcessingCost, 0.0))
        F_cap = get(p, :FixedCost_per_MW_EP_Out, 0.0)
        cap = vars[:cap_EP_y]
        ep = vars[:ep]
        for jy in JY
            op = sum(Wd[jy][jd] * C_proc * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
            wy[jy] = -(op + F_cap * value(cap))
        end

    elseif atype == "GreyOfftaker"
        C_proc = get(p, :MarginalCostByYear, fill(get(p, :MarginalCost, 0.0), length(JY)))
        ep = vars[:ep]
        for jy in JY
            wy[jy] = -sum(Wd[jy][jd] * C_proc[jy] * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif atype == "EPImporter"
        C_proc = get(p, :ImportCost, 0.0)
        ep = vars[:ep]
        for jy in JY
            wy[jy] = -sum(Wd[jy][jd] * C_proc * value(ep[jh, jd, jy]) for jh in JH, jd in JD)
        end

    elseif haskey(vars, :d_gc)
        A_GC = get(p, :A_GC, 0.0)
        B_GC = get(p, :B_GC, 0.5)
        d_gc = vars[:d_gc]
        for jy in JY
            wy[jy] = sum(Wd[jy][jd] * (A_GC * value(d_gc[jh, jd, jy]) -
                        0.5 * B_GC * value(d_gc[jh, jd, jy])^2) for jh in JH, jd in JD)
        end
    end

    return wy
end

function aggregate_social_welfare_per_year(mdict::Dict, agents::Dict)
    ref = mdict[agents[:all][1]]
    JH = collect(ref.ext[:sets][:JH])
    JD = collect(ref.ext[:sets][:JD])
    JY = collect(ref.ext[:sets][:JY])
    W = ref.ext[:parameters][:W]
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    social = Dict{Int, Float64}(jy => 0.0 for jy in JY)
    for id in agents[:all]
        haskey(mdict, id) || continue
        wy = _agent_welfare_per_year(mdict[id], agents; JH=JH, JD=JD, JY=JY, W=W)
        for (jy, w) in wy
            social[jy] += w
        end
    end
    return JY, P, [social[jy] for jy in JY]
end

function extract_private_cvar_by_agent(mdict::Dict, agents::Dict)
    rows = NamedTuple[]
    for id in agents[:all]
        m = mdict[id]
        t = String(get(m.ext[:parameters], :Type, ""))
        vars = m.ext[:variables]
        cvar_val, alpha_val = NaN, NaN
        if t == "VRES" && haskey(vars, :CVaR_VRES)
            cvar_val = value(vars[:CVaR_VRES])
            alpha_val = haskey(vars, :alpha_VRES) ? value(vars[:alpha_VRES]) : NaN
        elseif t == "GreenProducer" && haskey(vars, :CVaR_H2)
            cvar_val = value(vars[:CVaR_H2])
            alpha_val = haskey(vars, :alpha_H2) ? value(vars[:alpha_H2]) : NaN
        elseif t == "GreenOfftaker" && haskey(vars, :CVaR_GreenOfftaker)
            cvar_val = value(vars[:CVaR_GreenOfftaker])
            alpha_val = haskey(vars, :alpha_G) ? value(vars[:alpha_G]) : NaN
        elseif t in ("GreenH2Coalition", "GreenCoalition") && haskey(vars, :CVaR_coalition)
            cvar_val = value(vars[:CVaR_coalition])
            alpha_val = haskey(vars, :alpha_coalition) ? value(vars[:alpha_coalition]) : NaN
        end
        isfinite(cvar_val) || continue
        push!(rows, (agent=id, type=t, cvar_private=cvar_val, alpha_private=alpha_val))
    end
    return rows
end

"""Parse a numeric metric cell from Risk_Metrics.csv (may be String or Real)."""
function _parse_metric_value(v)
    v isa Real && return Float64(v)
    v isa Missing && return nothing
    s = strip(string(v))
    (isempty(s) || s == "-" || lowercase(s) == "nan") && return nothing
    return tryparse(Float64, s)
end

function load_sp_social_cvar_benchmark(project_root::String)
    path = joinpath(project_root, "social_planner_results", "Risk_Metrics.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    # Prefer the ex-post recomputed CVaR. The stored `social_CVaR` is the planner's
    # cvar_social variable, which carries zero objective weight at gamma = 1 and is
    # therefore left arbitrarily loose; comparing an ADMM ex-post CVaR against it
    # would report a large spurious gap. Fall back to it only if the recomputed row
    # is missing (older result files).
    for key in ("social_CVaR_recomputed", "social_CVaR")
        for i in 1:nrow(df)
            String(df.Metric[i]) == key || continue
            parsed = _parse_metric_value(df.Value[i])
            parsed === nothing || return parsed
        end
    end
    return nothing
end

function extract_sp_risk_metrics(planner::Model, planner_state::Dict, mdict::Dict, agents::Dict)
    JY = collect(planner_state[:JY])
    ref = mdict[agents[:all][1]]
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    gamma = Float64(planner_state[:gamma])
    beta = Float64(planner_state[:beta])

    sw_aux = planner_state[:sw_aux]
    social_welfare = planner_state[:social_welfare]
    sw_y = Float64[value(sw_aux[jy]) for jy in JY]
    swelfare_y = Float64[value(social_welfare[jy]) for jy in JY]
    loss_y = -sw_y

    cvar_model = value(planner_state[:cvar_social])
    alpha_model = value(planner_state[:alpha_social])
    cvar_check, alpha_check, _ = empirical_cvar(loss_y, P, beta)

    expected_welfare = sum(P[jy] * sw_y[jy] for jy in JY)
    # Must match build_social_planner! objective: γ·Σ P[y]·sw_aux[y] − (1−γ)·CVaR.
    # Read from the solved model (not sum(sw_y), which ignores P and inflates by nYears).
    risk_adj_obj = objective_value(planner)
    if !isfinite(risk_adj_obj)
        risk_adj_obj = gamma * expected_welfare - (1 - gamma) * cvar_model
    end

    return (
        case = "social_planner",
        gamma = gamma,
        beta = beta,
        expected_social_welfare = expected_welfare,
        social_welfare_per_year = swelfare_y,
        social_loss_per_year = loss_y,
        social_CVaR = cvar_model,
        alpha_social = alpha_model,
        social_CVaR_recomputed = cvar_check,
        sum_private_CVaR = NaN,
        risk_adjusted_objective = risk_adj_obj,
        social_CVaR_gap_vs_SP = 0.0,
        private_cvar_rows = NamedTuple[],
    )
end

"""Read γ and β from agent parameters (stored as :γ/:β in define_common_parameters!)."""
function _read_risk_params(p::Dict)
    gamma = Float64(get(p, :γ, get(p, :gamma, 1.0)))
    beta  = Float64(get(p, :β, get(p, :beta, 0.95)))
    return gamma, beta
end

function extract_admm_risk_metrics(mdict::Dict, agents::Dict, case_label::String; project_root::String)
    ref = mdict[agents[:all][1]]
    gamma, beta = _read_risk_params(ref.ext[:parameters])
    JY, P, swelfare_y = aggregate_social_welfare_per_year(mdict, agents)
    loss_y = -swelfare_y
    cvar_expost, alpha_expost, _ = empirical_cvar(loss_y, P, beta)
    expected_welfare = sum(P[jy] * swelfare_y[jy] for jy in JY)
    priv = extract_private_cvar_by_agent(mdict, agents)
    sum_priv = isempty(priv) ? 0.0 : sum(r.cvar_private for r in priv)
    sp_cvar = load_sp_social_cvar_benchmark(project_root)
    gap = sp_cvar === nothing ? NaN : (cvar_expost - sp_cvar)

    return (
        case = case_label,
        gamma = gamma,
        beta = beta,
        expected_social_welfare = expected_welfare,
        social_welfare_per_year = swelfare_y,
        social_loss_per_year = loss_y,
        social_CVaR = cvar_expost,
        alpha_social = alpha_expost,
        social_CVaR_recomputed = cvar_expost,
        sum_private_CVaR = sum_priv,
        risk_adjusted_objective = NaN,
        social_CVaR_gap_vs_SP = gap,
        private_cvar_rows = priv,
    )
end

function _risk_metrics_table(metrics::NamedTuple)
    sp_bench = metrics.case == "social_planner" ? metrics.social_CVaR :
        (isfinite(metrics.social_CVaR_gap_vs_SP) ? metrics.social_CVaR - metrics.social_CVaR_gap_vs_SP : NaN)
    DataFrame(
        Metric = String[
            "case", "gamma", "beta", "expected_social_welfare", "social_CVaR", "alpha_social",
            "social_CVaR_recomputed", "sum_private_CVaR", "risk_adjusted_objective",
            "social_CVaR_gap_vs_SP", "SP_social_CVaR_benchmark",
        ],
        Value = [
            metrics.case, metrics.gamma, metrics.beta, metrics.expected_social_welfare,
            metrics.social_CVaR, metrics.alpha_social, metrics.social_CVaR_recomputed,
            metrics.sum_private_CVaR, metrics.risk_adjusted_objective,
            metrics.social_CVaR_gap_vs_SP, sp_bench,
        ],
        Unit = String["-", "-", "-", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR", "EUR"],
    )
end

function write_sp_risk_outputs!(planner::Model, planner_state::Dict, mdict::Dict,
                                agents::Dict, results_folder::String)
    metrics = extract_sp_risk_metrics(planner, planner_state, mdict, agents)
    CSV.write(joinpath(results_folder, "Risk_Metrics.csv"), _risk_metrics_table(metrics))
    JY = collect(planner_state[:JY])
    P = [mdict[agents[:all][1]].ext[:parameters][:P][jy] for jy in JY]
    CSV.write(joinpath(results_folder, "Social_Welfare_Per_Year.csv"),
        DataFrame(case=fill(metrics.case, length(JY)), scenario_year=JY, probability=P,
                  social_welfare=metrics.social_welfare_per_year,
                  social_loss=metrics.social_loss_per_year))
    return metrics
end

function write_admm_risk_outputs!(mdict::Dict, agents::Dict, results_dir::String;
                                  case_label::String = "market_exposure")
    project_root = dirname(results_dir)
    metrics = extract_admm_risk_metrics(mdict, agents, case_label; project_root=project_root)
    CSV.write(joinpath(results_dir, "Risk_Metrics.csv"), _risk_metrics_table(metrics))
    ref = mdict[agents[:all][1]]
    JY = collect(ref.ext[:sets][:JY])
    P = Float64[ref.ext[:parameters][:P][jy] for jy in JY]
    CSV.write(joinpath(results_dir, "Social_Welfare_Per_Year.csv"),
        DataFrame(case=fill(metrics.case, length(JY)), scenario_year=JY, probability=P,
                  social_welfare=metrics.social_welfare_per_year,
                  social_loss=metrics.social_loss_per_year))
    if !isempty(metrics.private_cvar_rows)
        CSV.write(joinpath(results_dir, "Private_CVaR_By_Agent.csv"),
            DataFrame(
                Agent=[r.agent for r in metrics.private_cvar_rows],
                Type=[r.type for r in metrics.private_cvar_rows],
                CVaR_private=[r.cvar_private for r in metrics.private_cvar_rows],
                alpha_private=[r.alpha_private for r in metrics.private_cvar_rows],
            ))
    end
    return metrics
end

function print_risk_metrics_summary!(metrics::NamedTuple; title::String = "Risk metrics")
    # Internal CVaR is on loss L = -welfare; report tail welfare = -CVaR(L) (higher is better).
    # Always report the EX-POST empirical CVaR of the realised welfare vector rather
    # than the planner's cvar_social variable. At gamma = 1 the objective puts zero
    # weight on that variable, so the solver leaves it anywhere above its true value
    # and the printed tail would otherwise come out below the worst scenario, which
    # is impossible. At gamma < 1 the two agree because the objective drives it down.
    tail_welfare = -metrics.social_CVaR_recomputed
    min_sw = minimum(metrics.social_welfare_per_year)
    spread = maximum(metrics.social_welfare_per_year) - min_sw
    tail_pct = round(100 * (1 - metrics.beta); digits=0)
    risk_neutral = metrics.gamma >= 1.0 - 1e-12
    n_scen = length(metrics.social_welfare_per_year)

    println()
    println("-" ^ 72)
    println("  ", title)
    println("-" ^ 72)
    @printf("  Case:                    %s\n", metrics.case)
    @printf("  gamma:                   %.4f\n", metrics.gamma)
    @printf("  beta:                    %.4f  (tail = worst %.0f%% of scenario years)\n",
            metrics.beta, tail_pct)
    @printf("  E[social welfare]:            %10.3f bn EUR\n", metrics.expected_social_welfare / 1e9)
    @printf("  Tail welfare (CVaR):          %10.3f bn EUR  (higher = safer tail)\n", tail_welfare / 1e9)
    @printf("  Min scenario welfare:         %10.3f bn EUR\n", min_sw / 1e9)
    @printf("  Welfare spread (max−min):     %10.3f bn EUR\n", spread / 1e9)

    if metrics.case == "social_planner"
        @printf("  Risk-adjusted objective:        %10.3f bn EUR\n", metrics.risk_adjusted_objective / 1e9)
    elseif isfinite(metrics.social_CVaR_gap_vs_SP)
        sp_tail = tail_welfare + metrics.social_CVaR_gap_vs_SP  # gap is on loss: ME − SP
        tail_gap = -metrics.social_CVaR_gap_vs_SP               # ME tail − SP tail
        @printf("  SP tail welfare (benchmark):    %10.3f bn EUR\n", sp_tail / 1e9)
        @printf("  Tail welfare gap vs SP:        %+10.1f M EUR  (negative ⇒ ME worse)\n", tail_gap / 1e6)
    end

    if risk_neutral
        println("  (gamma = 1: risk-neutral; tail lines are ex-post at the stated beta.)")
    end
    # With n equiprobable scenarios a tail of (1-beta) narrower than 1/n collapses
    # onto the single worst scenario, so beta loses all resolution. Flag it, because
    # a risk-aversion sweep over such betas would show no variation at all.
    if (1 - metrics.beta) < 1 / n_scen - 1e-9
        @printf("  (NOTE: beta = %.2f implies a %.1f%% tail, narrower than one of the %d\n",
                metrics.beta, 100 * (1 - metrics.beta), n_scen)
        @printf("   equiprobable scenarios (%.1f%%), so CVaR collapses to the worst scenario.\n",
                100 / n_scen)
        @printf("   For a tail spanning k scenarios use beta = 1 - k/%d, e.g. %.3f for k=3.)\n",
                n_scen, 1 - 3 / n_scen)
    end
    return nothing
end
