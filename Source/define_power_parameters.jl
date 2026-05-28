# ==============================================================================
# define_power_parameters.jl — Power-sector agent parameters and timeseries
# ==============================================================================
#
# PURPOSE:
#   Called after define_common_parameters! for each power-sector agent. Fills
#   mod.ext[:parameters] and mod.ext[:timeseries] with type-specific data from
#   data.yaml and with 3D arrays built from the full-year time series (using
#   representative day indices). Supports: VRES (capacity + profile), Conventional
#   (capacity + constant availability + 3-stage thermal stack), Consumer (peak load + load profile +
#   quadratic utility parameters A_E, B_E).
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — Same convention as define_common_parameters!;
#     data here is merged General + Power[agent] so it contains Type, Capacity,
#     Profile_Column or Load_Column, etc.
#
# ==============================================================================

function define_power_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    # Short-hand references to the model's parameter and timeseries dicts.
    params = mod.ext[:parameters]
    times  = mod.ext[:timeseries]

    n_ts = data["nTimesteps"]
    n_rd = data["nReprDays"]
    n_yr = data["nYears"]
    JY   = mod.ext[:sets][:JY]
    JD   = mod.ext[:sets][:JD]
    JH   = mod.ext[:sets][:JH]

    base_year = get(data, "base_year", 2021)
    _years    = isdefined(Main, :years) ? Main.years : Dict(1 => base_year)

    agent_type = String(get(data, "Type", ""))

    if agent_type == "VRES"
        # --- Variable Renewable Energy Source (e.g. solar, wind) ---
        params[:Capacity]      = data["Capacity"]       # Installed capacity (MW); physical upper bound on output
        params[:Profile_Column] = String(data["Profile_Column"])  # Timeseries column name (e.g. "SOLAR") with hourly capacity factors
        params[:MarginalCost]  = data["MarginalCost"]   # €/MWh; typically 0 for renewables (no fuel cost)
        # Annualised fixed investment cost per MW of installed VRES capacity (€/MW-year).
        # Read from data.yaml if present; default 0.0 keeps previous behaviour.
        params[:FixedCost_per_MW] = get(data, "FixedCost_per_MW", 0.0)
        col = params[:Profile_Column]

        # Build 3D availability factor AF[jh, jd, jy] (capacity factor profile, values in 0–1).
        # AF tells build_power_agent! the fraction of Capacity available at each hour.
        # The timeseries CSV stores representative-day data sequentially in its first
        # (nReprDays * nTimesteps) rows: day 1 occupies rows 1–24, day 2 rows 25–48,
        # etc.  We index directly: row = (jd-1)*n_ts + jh.
        times[:AF] = Array{Float64}(undef, n_ts, n_rd, n_yr)
        for jy in JY
            yr = _years[jy]
            for jd in JD, jh in JH
                row = (jd - 1) * n_ts + jh
                times[:AF][jh, jd, jy] = ts[yr][!, Symbol(col)][row]
            end
        end

    elseif agent_type == "Conventional"
        # --- Dispatchable thermal generator (coal + biomass + NG stack) ---
        params[:Capacity] = data["Capacity"]  # Installed capacity (MW)
        # Keep MarginalCost only as optional legacy fallback if stage inputs are absent.
        params[:MarginalCost] = get(data, "MarginalCost", 60.0)

        # Three-stage convex marginal-cost curve (single aggregated conventional agent):
        #   Stage s has capacity cap_s and marginal cost MC_s(x) = base_s + slope_s * x
        #   for x in [0, cap_s], with continuity between stage endpoints.
        #
        # Primary inputs (recommended, absolute values):
        #   - StageCapacityShares : length-3 capacity split (normalized internally)
        #   - StageBaseCosts      : length-3 starting MC values (€/MWh), e.g. coal/biomass/NG
        #   - FinalMarginalCost   : MC at end of stage 3 (€/MWh)
        #
        # Slopes are derived internally to enforce continuous stage transitions:
        #   end(MC_1) = base_2, end(MC_2) = base_3, end(MC_3) = FinalMarginalCost.
        #
        # Backward compatibility:
        #   - If StageBaseCosts are missing, old multiplier keys are accepted.
        #   - If FinalMarginalCost is missing, it is inferred from legacy slope keys when available.
        shares_raw = Float64.(get(data, "StageCapacityShares", [1 / 3, 1 / 3, 1 / 3]))
        if length(shares_raw) != 3
            error("Conventional generator requires 3 entries for StageCapacityShares")
        end
        shares = max.(shares_raw, 1e-9)
        shares ./= sum(shares)
        caps = params[:Capacity] .* shares

        base_costs = if haskey(data, "StageBaseCosts")
            Float64.(data["StageBaseCosts"])
        elseif haskey(data, "StageBaseCostMultipliers")
            Float64.(data["StageBaseCostMultipliers"]) .* params[:MarginalCost]
        else
            # Defaults: coal, biomass, natural-gas-like starting marginal costs.
            [35.0, 55.0, 85.0]
        end

        final_mc = if haskey(data, "FinalMarginalCost")
            Float64(data["FinalMarginalCost"])
        elseif haskey(data, "StageSlopeMultipliers")
            # Infer an endpoint from legacy slope multipliers.
            slope_mult = Float64.(data["StageSlopeMultipliers"])
            if length(slope_mult) != 3
                error("Conventional generator requires 3 entries for StageSlopeMultipliers")
            end
            slope_legacy = slope_mult .* (params[:MarginalCost] / max(params[:Capacity], 1e-9))
            base_costs[3] + slope_legacy[3] * caps[3]
        else
            # Default endpoint (NG-like high-load marginal cost).
            140.0
        end

        if length(base_costs) != 3
            error("Conventional generator requires 3 entries for StageBaseCosts")
        end
        if !(base_costs[1] <= base_costs[2] <= base_costs[3] <= final_mc + 1e-9)
            error("Conventional stage MC endpoints must be nondecreasing: base1 <= base2 <= base3 <= FinalMarginalCost")
        end

        slopes = [
            (base_costs[2] - base_costs[1]) / max(caps[1], 1e-9),
            (base_costs[3] - base_costs[2]) / max(caps[2], 1e-9),
            (final_mc      - base_costs[3]) / max(caps[3], 1e-9),
        ]

        params[:ConvStageCap] = caps
        params[:ConvStageBaseCost] = base_costs
        params[:ConvStageSlope] = max.(slopes, 0.0)
        params[:ConvFinalMarginalCost] = final_mc

        # Constant AF = 1.0 at every hour: dispatchable thermal generation.
        # No timeseries profile is needed; the optimizer decides dispatch level.
        times[:AF] = ones(n_ts, n_rd, n_yr)

    elseif agent_type == "Consumer"
        # --- Electricity consumer with quadratic utility ---
        params[:PeakLoad]    = data["PeakLoad"]         # Peak demand (MW); scales the normalized load profile to absolute MW
        params[:Load_Column] = String(data["Load_Column"])  # Timeseries column with normalized load shape (0–1)
        # Quadratic utility function: U(d) = A_E·d − ½·B_E·d²
        # The inverse demand (willingness to pay) is: p(d) = A_E − B_E·d
        # A_E = intercept (€/MWh): maximum willingness to pay for the first MW
        # B_E = slope (€/MWh²): rate at which willingness to pay decreases with consumption
        params[:A_E]         = data["A_E"]
        params[:B_E]         = data["B_E"]

        # Build 3D load profile LOAD_E[jh, jd, jy] (normalized, 0–1).
        # PeakLoad * LOAD_E gives absolute MW demand at each hour.
        # Row mapping: same direct indexing as VRES AF (see comment above).
        col = params[:Load_Column]
        times[:LOAD_E] = Array{Float64}(undef, n_ts, n_rd, n_yr)
        for jy in JY
            yr = _years[jy]
            for jd in JD, jh in JH
                row = (jd - 1) * n_ts + jh
                times[:LOAD_E][jh, jd, jy] = ts[yr][!, Symbol(col)][row]
            end
        end
    end

    return mod
end
