# =============================================================================
# generate_representative_days.jl
# -----------------------------------------------------------------------------
# Builds the model's scenario-year time-series inputs using
# RepresentativePeriodsFinder.jl (CLUSTERING method, hierarchical / medoid-based).
#
# Pipeline per scenario label Y (2021..2030):
#   1. Fetch a full calendar year of hourly weather (Open-Meteo ERA5) for the
#      source year mapped to Y, convert to solar/wind capacity factors and a
#      synthetic NL electricity-load shape -> 8760-row full-year CSV.
#   2. Run RPF hierarchical clustering (via config_template.yaml) to pick
#      `representative_periods` medoid days out of 365.
#   3. Translate RPF's native outputs into the files the model reads:
#        Input/timeseries_<Y>.csv                      (192 rows: Time,SOLAR,LOAD_E,LOAD_H,LOAD_EP,WIND)
#        Input/output_<Y>/decision_variables_short.csv (periods, weights, selected_periods)
#        Input/output_<Y>/decision_variables.csv       (all 365 days)
#        Input/output_<Y>/ordering_variable.csv        (365 x nRepr assignment matrix)
#
# Existing Input/timeseries_*.csv and Input/output_* are backed up once to
# Input/_legacy_inputs_backup/ before the first overwrite.
#
# Run (from repo root):
#   julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl
# Optionally pass specific labels:
#   julia --project=Input/rep_periods Input/rep_periods/generate_representative_days.jl 2021 2025
# =============================================================================

using Dates
using HTTP
using JSON
using CSV
using DataFrames
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

# Scenario label -> source ERA5 calendar year + metadata. Labels are names only;
# they do NOT have to equal the calendar year of the underlying weather.
const SCENARIOS = [
    (label = 2021, source_year = 2015, role = "Baseline NL reference weather (CBS-calibrated annual CF targets)", solar_cf_target = 0.18, wind_cf_target = 0.28),
    (label = 2022, source_year = 2010, role = "Low-wind / moderate-solar year (dunkelflaute-prone)"),
    (label = 2023, source_year = 2012, role = "Average mixed VRES year"),
    (label = 2024, source_year = 2013, role = "High-solar / moderate-wind year"),
    (label = 2025, source_year = 2014, role = "High-wind year"),
    (label = 2026, source_year = 2016, role = "Windy winter, moderate solar"),
    (label = 2027, source_year = 2017, role = "Calm summer, lower wind CF"),
    (label = 2028, source_year = 2018, role = "Strong solar summer"),
    (label = 2029, source_year = 2019, role = "Cold winter, elevated load + moderate VRES"),
    (label = 2030, source_year = 2011, role = "Alternative tail scenario (low solar, variable wind)"),
]

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

# Scale hourly CF (via bisection) so the annual mean hits `target` after clipping.
function scale_to_target(cf::Vector{Float64}, target::Float64)
    m = sum(cf) / length(cf)
    m <= 1e-9 && return cf
    lo, hi = 0.0, max(10.0, target / max(m, 1e-9) * 2)
    hi_final = hi
    for _ in 1:48
        mid = (lo + hi) / 2
        scaled = clamp.(cf .* mid, 0.0, 1.0)
        if sum(scaled) / length(scaled) < target
            lo = mid
        else
            hi = mid
        end
        hi_final = hi
    end
    return clamp.(cf .* hi_final, 0.0, 1.0)
end

# Normalized NL electricity demand shape (peak = 1.0): typical hour-of-day and
# seasonal (winter-peaking) pattern with mild temperature sensitivity.
function nl_load_profile(n_days::Int, temperature_c::Vector{Float64})
    n = n_days * N_HOURS
    load = zeros(n)
    t_ref = 10.0
    for h in 0:(n - 1)
        day  = h ÷ N_HOURS
        hour = h % N_HOURS
        doy  = day + 1
        hod = 0.55 +
              0.18 * exp(-0.5 * ((hour - 8) / 3.5)^2) +
              0.35 * exp(-0.5 * ((hour - 19) / 2.8)^2)
        season = 0.88 + 0.22 * cos(2π * (doy - 15) / 365.0)
        temp = h + 1 <= length(temperature_c) ? temperature_c[h + 1] : t_ref
        temp_factor = 1.0 + 0.012 * (t_ref - temp)
        load[h + 1] = hod * season * temp_factor
    end
    return load ./ maximum(load)
end

# Build full-year (8760h) solar/wind/load CF vectors for one scenario.
function build_full_year(sc)
    data = fetch_year(sc.source_year)
    hourly = data["hourly"]
    wind_kmh = Float64.(hourly["wind_speed_100m"])
    rad      = Float64.(hourly["shortwave_radiation"])
    temp     = Float64.(hourly["temperature_2m"])

    n = min(length(wind_kmh), 8760)          # drop Dec 31 in leap years
    wind_kmh, rad, temp = wind_kmh[1:n], rad[1:n], temp[1:n]
    n_days = n ÷ N_HOURS

    wind  = wind_speed_to_cf(wind_kmh)
    solar = radiation_to_cf(rad)
    haskey(sc, :solar_cf_target) && (solar = scale_to_target(solar, sc.solar_cf_target))
    haskey(sc, :wind_cf_target)  && (wind  = scale_to_target(wind,  sc.wind_cf_target))
    load = nl_load_profile(n_days, temp)

    return solar, wind, load
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
    df = DataFrame(
        SOLAR = round.(solar, digits = 6),
        WIND  = round.(wind,  digits = 6),
        LOAD_E = round.(load, digits = 6),
    )
    CSV.write(path, df)
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

# Translate RPF outputs -> the CSVs the model reads.
function export_model_files(label, result_dir)
    out_dir = joinpath(INPUT_DIR, "output_$(label)")
    isdir(out_dir) || mkpath(out_dir)

    # decision_variables + ordering matrix are already in the model's format.
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
    return ts_out
end

function process_scenario(sc)
    label = sc.label
    @info "=== Scenario $label (ERA5 source $(sc.source_year)) ==="
    solar, wind, load = build_full_year(sc)
    full_csv = write_full_year_csv(label, solar, wind, load)
    result_dir = run_rpf(label, full_csv)
    export_model_files(label, result_dir)

    short = CSV.read(joinpath(result_dir, "decision_variables_short.csv"), DataFrame)
    return Dict(
        "label"           => label,
        "source_year"     => sc.source_year,
        "role"            => sc.role,
        "annual_solar_cf" => round(sum(solar) / length(solar), digits = 4),
        "annual_wind_cf"  => round(sum(wind)  / length(wind),  digits = 4),
        "medoid_days"     => Int.(short.periods),
        "weights"         => Float64.(short.weights),
        "weights_sum"     => round(sum(short.weights), digits = 2),
    )
end

function main()
    requested = isempty(ARGS) ? [sc.label for sc in SCENARIOS] : parse.(Int, ARGS)
    backup_existing_inputs()
    summaries = Any[]
    for sc in SCENARIOS
        sc.label in requested || continue
        push!(summaries, process_scenario(sc))
    end
    open(joinpath(INPUT_DIR, "weather_scenario_summary.json"), "w") do io
        JSON.print(io, summaries, 2)
    end
    println("\nScenario summary:")
    for s in summaries
        println("  ", s["label"], ": source=", s["source_year"],
                 ", solar_CF=", s["annual_solar_cf"], ", wind_CF=", s["annual_wind_cf"],
                 ", weights_sum=", s["weights_sum"])
    end
    println("\nDONE")
end

main()
