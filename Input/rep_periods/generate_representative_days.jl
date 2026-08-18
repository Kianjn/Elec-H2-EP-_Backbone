# =============================================================================
# generate_representative_days.jl
# -----------------------------------------------------------------------------
# Builds the model's weather-scenario time-series inputs using
# RepresentativePeriodsFinder.jl (CLUSTERING method, hierarchical / medoid-based).
#
# There are 5 weather years (labels 1..5). The model's full scenario set is the
# cross product of these with 3 natural-gas price levels, giving 15 scenarios;
# the gas dimension carries no time series, so it lives in Data/data.yaml
# (Scenarios block) rather than here. See DOCUMENTATION.md §9.7-9.8.
#
# Pipeline:
#   1. Fetch a full calendar year of hourly weather (Open-Meteo ERA5) for each of
#      the 5 source years and convert it to solar/wind capacity factors and an
#      NL electricity-load shape -> 8760-row full-year CSV.
#   2. Run RPF hierarchical clustering (via config_template.yaml) to pick
#      `representative_periods` medoid days out of 365.
#   3. Translate RPF's native outputs into the files the model reads:
#        Input/timeseries_<Y>.csv                      (192 rows: Time,SOLAR,LOAD_E,LOAD_H,LOAD_EP,WIND)
#        Input/output_<Y>/decision_variables_short.csv (periods, weights, selected_periods)
#        Input/output_<Y>/decision_variables.csv       (all 365 days)
#        Input/output_<Y>/ordering_variable.csv        (365 x nRepr assignment matrix)
#
# TWO CROSS-YEAR CALIBRATIONS (both essential, see the constants below):
#   * Capacity factors are scaled by ONE pair of multipliers derived from the
#     reference year, applied identically to all five years.
#   * Load is normalised by ONE constant, the peak across all five years.
#   Doing either per-year would erase the very differences the scenarios exist to
#   represent: every year would look equally windy and peak at exactly 1.0.
#
# Existing Input/timeseries_*.csv and Input/output_* are backed up once to
# Input/_legacy_inputs_backup/ before the first overwrite.
#
# Run (from repo root):
#   julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl
# Optionally pass specific labels (calibration is still computed from all years):
#   julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl 1 5
# =============================================================================

using Dates
using HTTP
using JSON
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using RepresentativePeriodsFinder

const HERE          = @__DIR__
const INPUT_DIR     = normpath(joinpath(HERE, ".."))
const CACHE_DIR     = joinpath(HERE, "weather_cache")
const WEATHER_FULL  = joinpath(HERE, "weather_full")
const RESULTS_DIR   = joinpath(HERE, "results")
const TEMPLATE      = joinpath(HERE, "config_template.yaml")
const BACKUP_DIR    = joinpath(INPUT_DIR, "_legacy_inputs_backup")

# Central Netherlands (Utrecht region) — representative for national VRES.
const LAT, LON = 52.09, 5.12
const N_HOURS  = 24

# Hierarchical-clustering dissimilarity metric (squared Euclidean distance).
# Defined at top level and injected into the config as a real Function so RPF
# uses it directly instead of `eval`-ing a string at runtime (which triggers a
# Julia world-age error when the whole run is wrapped in a function like main()).
ssd(x, y) = sum((x .- y) .^ 2)

# ----------------------------------------------------------------- scenarios ---
#
# The five weather years were chosen from ten candidate ERA5 years (2010-2019) by
# an exhaustive max-min diversity search: of all 252 possible 5-subsets, this one
# maximises the MINIMUM pairwise distance between the years it contains, over
# z-scored features covering monthly solar and wind capacity factors, annual CFs,
# wind/solar variability, dunkelflaute frequency, and heating/cooling degree
# hours. The search was run on consistently calibrated data (see below), so the
# spread reflects weather, not scaling.
const SCENARIOS = [
    (label = 1, source_year = 2015, role = "Benign high-wind reference year: highest wind CF (0.28), fewest dunkelflaute hours (3.0%)"),
    (label = 2, source_year = 2010, role = "Cold low-wind stress year: coldest winter, lowest wind CF (0.19), most dunkelflaute hours (8.2%)"),
    (label = 3, source_year = 2016, role = "Mid-range year: moderate wind (0.22) and solar, mild winter"),
    (label = 4, source_year = 2017, role = "High-wind / low-solar year: wind 0.25, lowest solar CF (0.173)"),
    (label = 5, source_year = 2018, role = "Hot high-solar year: highest solar CF (0.193) and by far the hottest summer"),
]

# Label whose ERA5 year defines the common NL capacity-factor calibration.
const REFERENCE_LABEL = 1
# [NL] CBS 2024 national annual normalised capacity factors for the installed fleet.
const SOLAR_CF_TARGET = 0.182  # [CBS-RE] — 18.24% (table 82610, 2024)
const WIND_CF_TARGET  = 0.280  # [CBS-RE], on+offshore weighted — 28.01% (2024)

# ------------------------------------------------------------ demand model ---
#
# Electricity demand responds to the weather year through heating and cooling
# degree hours, the standard way of linking temperature to load (Bessec &
# Fouquau 2008; ENTSO-E / JRC PECD demand modelling):
#
#   load_raw[h] = hod[hour] * dow[weekday] * light[doy] * (1 + a_H*HDD + a_C*CDD)
#     HDD[h] = max(T_ref_H - T[h], 0)      T_ref_H = 15.5 C  (EU/UK standard base)
#     CDD[h] = max(T[h] - T_ref_C, 0)      T_ref_C = 22.0 C
#
# a_H = 1.0%/C is deliberately modest: NL space heating is gas-dominated, so the
# electricity system's temperature gradient is much weaker than in electrically
# heated countries. a_C = 0.8%/C covers air conditioning, which bites only in the
# hottest summers - this is what makes 2018 a genuinely distinct scenario.
# `light` is a small residual seasonal term for lighting and activity that is not
# temperature driven.
const HDD_BASE      = 15.5   # C
const CDD_BASE      = 22.0   # C
const HEAT_SENS     = 0.010  # per C  [BF-2008]
const COOL_SENS     = 0.008  # per C  [BF-2008]
const LIGHT_AMPL    = 0.05   # +/-5% non-thermal seasonal (lighting/activity)
const WEEKEND_FACTOR = 0.87  # [NL] weekend load ~13% below weekday [ENTSOE-LOAD]

# ------------------------------------------------------------------ weather ---

function fetch_year(source_year::Int)
    isdir(CACHE_DIR) || mkpath(CACHE_DIR)
    cache_path = joinpath(CACHE_DIR, "open_meteo_$(source_year).json")
    if isfile(cache_path)
        return JSON.parse(read(cache_path, String))
    end
    url = string(
        "https://archive-api.open-meteo.com/v1/archive",
        "?latitude=", LAT, "&longitude=", LON,
        "&start_date=", source_year, "-01-01&end_date=", source_year, "-12-31",
        "&hourly=wind_speed_100m,shortwave_radiation,temperature_2m",
        "&timezone=Europe%2FAmsterdam",
    )
    @info "Fetching ERA5 weather for $source_year ..."
    body = String(HTTP.get(url; readtimeout = 120, retries = 3).body)
    write(cache_path, body)
    return JSON.parse(body)
end

# Onshore/offshore mix power curve; output in [0, 1]. Input wind speed in km/h.
function wind_speed_to_cf(speed_kmh::Vector{Float64})
    v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
    cf = zeros(length(speed_kmh))
    @inbounds for i in eachindex(speed_kmh)
        v = max(speed_kmh[i] / 3.6, 0.0)  # m/s
        if v >= v_cut_in && v < v_rated
            cf[i] = ((v - v_cut_in) / (v_rated - v_cut_in))^3
        elseif v >= v_rated && v < v_cut_out
            cf[i] = 1.0
        end
    end
    return cf
end

# GHI -> PV capacity factor (STC reference 1000 W/m², capped at 1).
radiation_to_cf(rad::Vector{Float64}) = clamp.(rad ./ 1000.0, 0.0, 1.0)

# Bisect the single multiplier that makes the REFERENCE year's annual mean hit
# `target` after clipping to [0,1]. The multiplier is then reused unchanged for
# every other year, so inter-year CF differences are pure weather.
function calibration_multiplier(cf::Vector{Float64}, target::Float64)
    lo, hi = 0.0, 50.0
    for _ in 1:60
        mid = (lo + hi) / 2
        if mean(clamp.(cf .* mid, 0.0, 1.0)) < target
            lo = mid
        else
            hi = mid
        end
    end
    return hi
end

# Un-normalised NL electricity demand for one year. The caller divides every year
# by a single shared constant so that cold years genuinely peak higher and consume
# more than mild years.
function nl_load_raw(source_year::Int, temperature_c::Vector{Float64})
    n = length(temperature_c)
    jan1 = Date(source_year, 1, 1)
    load = zeros(n)
    for h in 0:(n - 1)
        day  = h ÷ N_HOURS
        hour = h % N_HOURS
        doy  = day + 1

        # Hour-of-day activity shape: overnight base + morning ramp + evening peak.
        # The three amplitudes are set so the resulting profile reproduces the NL
        # 2024 system load factor of ~0.636 (108.5 TWh net consumption against a
        # 19.48 GW peak [CBS-EP; ENTSOE-PEAK-2024]); a peakier shape would understate annual energy.
        hod = 0.58 +
              0.18 * exp(-0.5 * ((hour - 8) / 3.5)^2) +
              0.30 * exp(-0.5 * ((hour - 19) / 2.8)^2)

        # Real calendar weekday/weekend split for this source year.
        dow = Dates.dayofweek(jan1 + Day(day)) >= 6 ? WEEKEND_FACTOR : 1.0

        # Non-thermal seasonal term (lighting, activity), peaking mid-winter.
        light = 1.0 + LIGHT_AMPL * cos(2π * (doy - 15) / 365.0)

        # Temperature response via heating/cooling degree hours.
        t = temperature_c[h + 1]
        thermal = 1.0 + HEAT_SENS * max(HDD_BASE - t, 0.0) +
                        COOL_SENS * max(t - CDD_BASE, 0.0)

        load[h + 1] = hod * dow * light * thermal
    end
    return load
end

# Raw (uncalibrated) hourly series for one source year, trimmed to 8760 hours.
function raw_year(source_year::Int)
    hourly = fetch_year(source_year)["hourly"]
    wind_kmh = Float64.(hourly["wind_speed_100m"])
    rad      = Float64.(hourly["shortwave_radiation"])
    temp     = Float64.(hourly["temperature_2m"])
    n = min(length(wind_kmh), 8760)          # drop Dec 31 in leap years
    return (
        solar = radiation_to_cf(rad[1:n]),
        wind  = wind_speed_to_cf(wind_kmh[1:n]),
        temp  = temp[1:n],
    )
end

# ------------------------------------------------------------------- driver ---

function backup_existing_inputs()
    isdir(BACKUP_DIR) && return  # only back up once (preserve true originals)
    made = false
    for f in readdir(INPUT_DIR)
        full = joinpath(INPUT_DIR, f)
        is_ts  = startswith(f, "timeseries_") && endswith(f, ".csv")
        is_out = startswith(f, "output_")
        if is_ts || is_out
            made || (mkpath(BACKUP_DIR); made = true)
            cp(full, joinpath(BACKUP_DIR, f); force = true)
        end
    end
    made && @info "Backed up previous inputs to $BACKUP_DIR"
end

# Write the 8760-row CSV that RPF will cluster.
function write_full_year_csv(label, solar, wind, load)
    isdir(WEATHER_FULL) || mkpath(WEATHER_FULL)
    path = joinpath(WEATHER_FULL, "timeseries_full_$(label).csv")
    CSV.write(path, DataFrame(
        SOLAR  = round.(solar, digits = 6),
        WIND   = round.(wind,  digits = 6),
        LOAD_E = round.(load,  digits = 6),
    ))
    return path
end

# Run RPF hierarchical clustering for one scenario; returns its result dir.
function run_rpf(label, full_year_csv)
    result_dir = joinpath(RESULTS_DIR, string(label))
    isdir(result_dir) || mkpath(result_dir)

    pf = PeriodsFinder(TEMPLATE; populate_entries = false)
    pf.config["time_series"]["default"]["source"] = abspath(full_year_csv)
    pf.config["results"]["result_dir"] = abspath(result_dir)
    # Replace each ordering-error function string with the top-level `ssd`
    # Function (avoids the runtime-eval world-age problem; see note above).
    for (_, entry) in pf.config["method"]["options"]["ordering_error"]
        entry["function"] = ssd
    end
    populate_entries!(pf)
    find_representative_periods(pf; reset = true)   # clustering (no optimizer)
    return result_dir
end

# --------------------------------------------------------- weight rebalance ---
#
# Medoid clustering picks REAL days near each cluster centre, which deliberately
# under-samples the tails. The side effect is that the weighted representative
# year does not reproduce the full year's annual means: raw RPF output missed the
# annual solar CF by up to 12% and the annual wind CF by up to 12%, and by
# DIFFERENT amounts in different years. That last part is what makes it
# unacceptable here, because it distorts the very inter-year spread these five
# scenarios were selected to represent.
#
# Fix: keep the medoid day profiles exactly as they are and adjust only their
# weights, by the smallest change that reproduces the annual means. With n_j the
# cluster sizes RPF produced, solve
#
#   min ||w - n||^2   s.t.   sum_j w_j = 365
#                            sum_j w_j * daymean_X(j) = 365 * annual_mean_X
#                                                       for X in SOLAR, WIND, LOAD_E
#
# plus a floor w_j >= WEIGHT_FLOOR so no medoid is dropped or given a negative
# weight. It preserves total days, every hourly shape, and the chronological
# ordering, and only redistributes how often each medoid stands in for the rest
# of the year.
#
# Without the floor the problem has the closed form w = n + A'(A A')^-1 (b - A n).
# With it, the feasible set is the intersection of an affine subspace {Aw = b} and
# a box {w >= floor}, both convex, so Dykstra's alternating projection algorithm
# converges to the exact projection of n onto the intersection. That is a proper
# solve rather than a heuristic, which matters because a naive active-set pass
# fails on the 2018 scenario.
const WEIGHT_FLOOR = 1.0   # days; never let a representative day vanish

function rebalance_weights(weights::Vector{Float64}, rp::DataFrame,
                           full::NamedTuple, label)
    nd = length(weights)
    daymean(col) = [mean(col[(j - 1) * N_HOURS + 1:j * N_HOURS]) for j in 1:nd]

    A = vcat(ones(1, nd),
             daymean(Float64.(rp.SOLAR))',
             daymean(Float64.(rp.WIND))',
             daymean(Float64.(rp.LOAD_E))')
    total_days = sum(weights)
    b = [total_days,
         total_days * mean(full.solar),
         total_days * mean(full.wind),
         total_days * mean(full.load)]

    AAt = factorize(A * A')
    project_affine(v) = v + A' * (AAt \ (b - A * v))
    project_floor(v)  = max.(v, WEIGHT_FLOOR)

    x = copy(weights)
    p = zeros(nd)
    q = zeros(nd)
    for _ in 1:20_000
        y = project_affine(x + p); p = x + p - y
        x = project_floor(y + q);  q = y + q - x
        if maximum(abs, A * x - b) < 1e-9
            break
        end
    end

    if maximum(abs, A * x - b) > 1e-6 || minimum(x) < WEIGHT_FLOOR - 1e-9
        @warn "Scenario $label: no weighting of the selected medoids reproduces all " *
              "annual means with every weight >= $WEIGHT_FLOOR day; keeping RPF's " *
              "original weights (residual $(round(maximum(abs, A * x - b), digits = 4)))."
        return weights, false
    end
    return x, true
end

# Translate RPF outputs -> the CSVs the model reads.
function export_model_files(label, result_dir, full::NamedTuple)
    out_dir = joinpath(INPUT_DIR, "output_$(label)")
    isdir(out_dir) || mkpath(out_dir)

    for f in ("decision_variables_short.csv", "decision_variables.csv", "ordering_variable.csv")
        src = joinpath(result_dir, f)
        isfile(src) && cp(src, joinpath(out_dir, f); force = true)
    end

    # Representative-day hourly profiles (192 rows, ascending rep-day order).
    rp = CSV.read(joinpath(result_dir, "resulting_profiles.csv"), DataFrame)
    ts_out = DataFrame(
        Time    = 1:nrow(rp),
        SOLAR   = round.(Float64.(rp.SOLAR),  digits = 6),
        LOAD_E  = round.(Float64.(rp.LOAD_E), digits = 6),
        LOAD_H  = fill(0.8, nrow(rp)),   # constant H2 demand shape (as in legacy inputs)
        LOAD_EP = fill(0.9, nrow(rp)),   # constant end-product demand shape
        WIND    = round.(Float64.(rp.WIND),   digits = 6),
    )
    CSV.write(joinpath(INPUT_DIR, "timeseries_$(label).csv"), ts_out)

    # Rebalance the weights so the weighted representative year matches the
    # full year's annual solar CF, wind CF and mean load.
    short = CSV.read(joinpath(out_dir, "decision_variables_short.csv"), DataFrame)
    w_new, adjusted = rebalance_weights(Float64.(short.weights), rp, full, label)
    short.weights = w_new
    CSV.write(joinpath(out_dir, "decision_variables_short.csv"), short)

    # Mirror the adjustment into the all-365-day file so the two stay consistent.
    long_path = joinpath(out_dir, "decision_variables.csv")
    if adjusted && isfile(long_path)
        long = CSV.read(long_path, DataFrame)
        for (j, p) in enumerate(short.periods)
            row = findfirst(==(p), long.periods)
            row === nothing || (long.weights[row] = w_new[j])
        end
        CSV.write(long_path, long)
    end

    return ts_out, w_new, adjusted
end

function main()
    requested = isempty(ARGS) ? [sc.label for sc in SCENARIOS] : parse.(Int, ARGS)
    backup_existing_inputs()

    # --- Pass 1: load every year and derive the two shared calibrations --------
    raw = Dict(sc.label => raw_year(sc.source_year) for sc in SCENARIOS)

    ref = raw[REFERENCE_LABEL]
    m_solar = calibration_multiplier(ref.solar, SOLAR_CF_TARGET)
    m_wind  = calibration_multiplier(ref.wind,  WIND_CF_TARGET)
    @info "Common NL capacity-factor calibration from label $REFERENCE_LABEL: " *
          "solar x$(round(m_solar, digits = 4)), wind x$(round(m_wind, digits = 4))"

    calibrated = Dict{Int,Any}()
    for sc in SCENARIOS
        r = raw[sc.label]
        calibrated[sc.label] = (
            solar    = clamp.(r.solar .* m_solar, 0.0, 1.0),
            wind     = clamp.(r.wind  .* m_wind,  0.0, 1.0),
            load_raw = nl_load_raw(sc.source_year, r.temp),
            temp     = r.temp,
        )
    end

    # One load normaliser for all years: the highest hour anywhere in the
    # scenario set becomes 1.0, so PeakLoad in data.yaml is the system peak
    # across scenarios and milder years sit genuinely below it.
    load_norm = maximum(maximum(calibrated[sc.label].load_raw) for sc in SCENARIOS)
    @info "Common load normaliser (peak across all scenarios): $(round(load_norm, digits = 4))"

    # --- Pass 2: cluster and export -------------------------------------------
    summaries = Any[]
    for sc in SCENARIOS
        sc.label in requested || continue
        @info "=== Weather scenario $(sc.label) (ERA5 source $(sc.source_year)) ==="
        c = calibrated[sc.label]
        load = c.load_raw ./ load_norm

        full_csv = write_full_year_csv(sc.label, c.solar, c.wind, load)
        result_dir = run_rpf(sc.label, full_csv)
        full = (solar = c.solar, wind = c.wind, load = load)
        ts_out, w_new, adjusted = export_model_files(sc.label, result_dir, full)

        short = CSV.read(joinpath(INPUT_DIR, "output_$(sc.label)", "decision_variables_short.csv"), DataFrame)
        dwind  = [mean(c.wind[(d - 1) * 24 + 1:d * 24])  for d in 1:(length(c.wind) ÷ 24)]
        dsolar = [mean(c.solar[(d - 1) * 24 + 1:d * 24]) for d in 1:(length(c.solar) ÷ 24)]

        # Reproduce the model's own annual aggregation as a self-check.
        wavg(col) = sum(w_new[j] * sum(col[(j - 1) * 24 + 1:j * 24]) for j in 1:length(w_new)) / 8760

        push!(summaries, Dict(
            "label"            => sc.label,
            "source_year"      => sc.source_year,
            "role"             => sc.role,
            "annual_solar_cf"  => round(mean(c.solar), digits = 4),
            "annual_wind_cf"   => round(mean(c.wind),  digits = 4),
            "peak_load_pu"     => round(maximum(load), digits = 4),
            "mean_load_pu"     => round(mean(load), digits = 4),
            "dunkelflaute_pct" => round(100 * mean((dwind .< 0.10) .& (dsolar .< 0.05)), digits = 2),
            "mean_hdd"         => round(mean(max.(HDD_BASE .- c.temp, 0.0)), digits = 3),
            "mean_cdd"         => round(mean(max.(c.temp .- CDD_BASE, 0.0)), digits = 4),
            "weights_rebalanced" => adjusted,
            "repr_solar_cf"    => round(wavg(ts_out.SOLAR),  digits = 4),
            "repr_wind_cf"     => round(wavg(ts_out.WIND),   digits = 4),
            "repr_mean_load"   => round(wavg(ts_out.LOAD_E), digits = 4),
            "medoid_days"      => Int.(short.periods),
            "weights"          => round.(w_new, digits = 4),
            "weights_sum"      => round(sum(w_new), digits = 2),
        ))
    end

    open(joinpath(INPUT_DIR, "weather_scenario_summary.json"), "w") do io
        JSON.print(io, summaries, 2)
    end

    println("\nWeather scenario summary (load in p.u. of the cross-scenario peak).")
    println("`repr` columns are what the model actually sees: the weighted")
    println("representative days. They should match the full-year columns.")
    println("  label  src   solarCF  reprSol  windCF   reprWind  peak    mean    reprMean  dunkel%  HDD    CDD     Wsum")
    for s in summaries
        println("  ", rpad(s["label"], 7), rpad(s["source_year"], 6),
                rpad(s["annual_solar_cf"], 9), rpad(s["repr_solar_cf"], 9),
                rpad(s["annual_wind_cf"], 9), rpad(s["repr_wind_cf"], 10),
                rpad(s["peak_load_pu"], 8), rpad(s["mean_load_pu"], 8),
                rpad(s["repr_mean_load"], 10),
                rpad(s["dunkelflaute_pct"], 9), rpad(s["mean_hdd"], 7),
                rpad(s["mean_cdd"], 8), s["weights_sum"])
    end
    println("\nDONE")
end

main()
