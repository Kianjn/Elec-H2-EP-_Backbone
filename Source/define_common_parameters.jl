# ==============================================================================
# define_common_parameters.jl — Common setup for every agent model
# ==============================================================================
#
# PURPOSE:
#   Called once per agent during initialization. It (1) creates the standard
#   index sets (years, representative days, hours) and stores them on the
#   JuMP model; (2) builds the weight matrix W and scenario probabilities P;
#   (3) sets risk-related parameters γ and β if present in data; (4) determines
#   which markets this agent participates in from its Type and pushes the agent
#   ID into the corresponding lists (agents[:elec_market], etc.); (5) allocates
#   ADMM placeholder arrays (λ, g_bar) and scalar ρ for each market so that
#   ADMM_subroutine can overwrite them each iteration without reallocating.
#
# ARGUMENTS:
#   m         — Agent ID (string).
#   mod       — The JuMP model for this agent (we write into mod.ext).
#   data      — Merged dict: General + ADMM + agent-specific block from data.yaml.
#   ts        — Dict keyed by year; values are DataFrames of hourly time series.
#   repr_days — Dict keyed by year; values are DataFrames with :periods, :weights.
#   agents    — Dict of agent lists; we push! m into the relevant market lists.
#
# ==============================================================================

function define_common_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict, agents::Dict)
    # --- Storage on the JuMP model ---
    # Six separate dictionaries keep the model's data cleanly separated by role:
    #   :sets        — Index ranges (JY, JD, JH) that define the model dimensions.
    #   :parameters  — Scalar/array constants (capacities, costs, ADMM prices, etc.).
    #   :timeseries  — 3D hourly profiles built from CSV data (AF, LOAD_E, …).
    #   :variables   — JuMP decision variables (filled later by build_*_agent!).
    #   :constraints — JuMP constraints (filled later by build_*_agent!).
    #   :expressions — JuMP expressions like objective terms (filled by build_*_agent!).
    # Separating them avoids name collisions and lets downstream code (build_*,
    # solve_*, ADMM_subroutine) access exactly the namespace it needs.
    mod.ext[:sets]        = Dict{Symbol,Any}()
    mod.ext[:parameters]  = Dict{Symbol,Any}()
    mod.ext[:timeseries]  = Dict{Symbol,Any}()
    mod.ext[:variables]   = Dict{Symbol,Any}()
    mod.ext[:constraints] = Dict{Symbol,Any}()
    mod.ext[:expressions] = Dict{Symbol,Any}()

    # --- Dimensions from config ---
    n_years     = data["nYears"]      # Number of scenarios (e.g. 1).
    n_repr_days = data["nReprDays"]   # Number of representative days (e.g. 3 or 8).
    n_timesteps = data["nTimesteps"]  # Hours per representative day (e.g. 24).

    # --- Index sets ---
    JY = 1:n_years
    JD = 1:n_repr_days
    JH = 1:n_timesteps

    mod.ext[:sets][:JY] = JY
    mod.ext[:sets][:JD] = JD
    mod.ext[:sets][:JH] = JH

    # --- Representative-day weights ---
    # W[jd, jy] = how many real calendar days representative day jd stands for in
    # year jy. This weight is used to scale per-representative-day objective values
    # up to a full-year total (e.g. sum over jh,jd,jy of W[jd,jy] * cost[jh,jd,jy]).
    #
    # We need the calendar year to index into repr_days, which is keyed by calendar
    # year (e.g. 2021). The global `years` dict (defined in the main script) maps
    # scenario index → calendar year. We access it via Main.years because this
    # function runs inside a module/function scope that doesn't see the global directly.
    base_year = get(data, "base_year", 2021)
    _years = isdefined(Main, :years) ? Main.years : Dict(1 => base_year)
    # Build the 2D weight matrix: for each (jd, jy), look up the calendar year
    # from _years, then pull the weight of that representative day from repr_days.
    W = [repr_days[_years[jy]][!, :weights][jd] for jd in JD, jy in JY]
    mod.ext[:parameters][:W] = W

    # --- Scenario probabilities and risk parameters ---
    # P[jy] = probability of year-scenario jy occurring. With nYears=1, P=[1.0].
    # For multi-scenario runs (nYears>1), we assume a uniform prior unless
    # overridden in data.yaml and use P in CVaR risk terms for green agents.
    P = ones(n_years) ./ n_years
    mod.ext[:parameters][:P] = P
    # γ (gamma): per-agent risk weight in the objective. γ=1 gives a strictly
    #   risk-neutral agent; γ<1 increases the weight on the CVaR(loss) term for
    #   agents that implement risk aversion (currently only VRES, electrolyzer,
    #   and GreenOfftaker read and use γ explicitly).
    # β (beta): CVaR confidence level τ (e.g. 0.95 = worst 5% of scenarios).
    # All agents receive γ, β here for convenience; build_* files decide
    # whether to actually use them in the objective.
    mod.ext[:parameters][:γ] = get(data, "gamma", 1.0)
    mod.ext[:parameters][:β] = get(data, "beta", 0.95)

    # --- Agent type and market participation ---
    # Type comes from the agent block in data.yaml (e.g. "VRES", "Consumer").
    agent_type = String(get(data, "Type", ""))
    # Store on the model so build_*_agent! functions can branch on it.
    mod.ext[:parameters][:Type] = agent_type

    # Which markets this agent trades in. Used both to push m into the right
    # agent-list and to set boolean flags on the model so ADMM_subroutine and
    # solve_* know which price/quantity arrays to read/write.
    #
    # Economic rationale for each mapping:
    #   elec:    Generators (VRES, Conv) sell electricity; Consumer buys; GreenProducer
    #            buys electricity to run the electrolyzer → all participate in elec.
    #   H2:      GreenProducer sells H2; GreenOfftaker buys H2 → bilateral H2 market.
    #   elec_GC: VRES produces electricity GCs; GreenProducer can trade GCs; GC_Demand
    #            buys GCs to satisfy renewable obligations.
    #   H2_GC:   GreenProducer creates H2 GCs; GreenOfftaker & GreyOfftaker need GCs
    #            to certify their end product as (partially) green.
    #   EP:      GreenOfftaker, GreyOfftaker, and EPImporter supply end product to
    #            meet the fixed EP demand; they compete on cost in the EP market.
    in_elec    = agent_type in ("VRES", "Conventional", "Consumer", "GreenProducer")
    in_H2      = agent_type in ("GreenProducer", "GreenOfftaker")
    in_elec_GC = agent_type in ("VRES", "GreenProducer", "GC_Demand")
    in_H2_GC   = agent_type in ("GreenProducer", "GreenOfftaker", "GreyOfftaker")
    in_EP      = agent_type in ("GreenOfftaker", "GreyOfftaker", "EPImporter")

    if in_elec    push!(agents[:elec_market], m) end
    if in_H2      push!(agents[:H2_market], m) end
    if in_elec_GC push!(agents[:elec_GC_market], m) end
    if in_H2_GC   push!(agents[:H2_GC_market], m) end
    if in_EP      push!(agents[:EP_market], m) end

    mod.ext[:parameters][:in_elec_market]    = in_elec
    mod.ext[:parameters][:in_H2_market]      = in_H2
    mod.ext[:parameters][:in_elec_GC_market] = in_elec_GC
    mod.ext[:parameters][:in_H2_GC_market]   = in_H2_GC
    mod.ext[:parameters][:in_EP_market]      = in_EP

    # --- ADMM placeholders (overwritten each iteration in ADMM_subroutine) ---
    # Pre-allocated so ADMM_subroutine can overwrite each iteration without
    # reallocating memory. Zeros are the initial values before the first ADMM
    # iteration; they will be replaced by actual prices / consensus terms.
    #   λ_*     = Lagrange multiplier (price) array [jh, jd, jy] for market *.
    #   g_bar_* = Consensus / penalty centre for market * (mean of all agents'
    #             net positions; used in the augmented-Lagrangian penalty term).
    #   ρ_*     = Scalar penalty weight for market * (updated by update_rho!).
    # All markets get the same 3D shape so we can broadcast element-wise in
    # price and consensus updates inside ADMM.jl.
    shp = (n_timesteps, n_repr_days, n_years)

    # Electricity market ADMM arrays
    mod.ext[:parameters][:λ_elec]     = zeros(shp)
    mod.ext[:parameters][:g_bar_elec] = zeros(shp)
    mod.ext[:parameters][:ρ_elec]     = 1.0

    # Hydrogen market ADMM arrays
    mod.ext[:parameters][:λ_H2]      = zeros(shp)
    mod.ext[:parameters][:g_bar_H2]   = zeros(shp)
    mod.ext[:parameters][:ρ_H2]       = 1.0

    # Electricity GC market ADMM arrays
    mod.ext[:parameters][:λ_elec_GC]     = zeros(shp)
    mod.ext[:parameters][:g_bar_elec_GC]  = zeros(shp)
    mod.ext[:parameters][:ρ_elec_GC]     = 1.0

    # Hydrogen GC market ADMM arrays
    mod.ext[:parameters][:λ_H2_GC]     = zeros(shp)
    mod.ext[:parameters][:g_bar_H2_GC]  = zeros(shp)
    mod.ext[:parameters][:ρ_H2_GC]      = 1.0

    # End-product market ADMM arrays
    mod.ext[:parameters][:λ_EP]     = zeros(shp)
    mod.ext[:parameters][:g_bar_EP]  = zeros(shp)
    mod.ext[:parameters][:ρ_EP]      = 1.0

    return mod, agents
end
