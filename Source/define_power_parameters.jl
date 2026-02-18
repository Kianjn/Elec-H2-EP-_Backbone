# ==============================================================================
# define_power_parameters.jl — Power-sector agent parameters and timeseries
# ==============================================================================
#
# PURPOSE:
#   Called after define_common_parameters! for each power-sector agent. Fills
#   mod.ext[:parameters] and mod.ext[:timeseries] with type-specific data from
#   data.yaml and with 3D arrays built from the full-year time series (using
#   representative day indices). Supports: VRES (capacity + profile), Conventional
#   (capacity + constant availability), Consumer (peak load + load profile +
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
        # --- Dispatchable thermal generator ---
        params[:Capacity]     = data["Capacity"]        # Installed capacity (MW)
        params[:MarginalCost] = data["MarginalCost"]    # €/MWh; fuel + O&M cost of generation
        # Constant AF = 1.0 at every hour: conventional generators are always
        # available up to their full capacity (dispatchable thermal generation).
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
