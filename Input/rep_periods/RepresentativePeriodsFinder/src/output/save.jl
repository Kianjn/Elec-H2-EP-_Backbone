"""
    save(pf::PeriodsFinder, result_dir::String = get_abspath_to_result_dir(pf);)

Writes out the representative periods selected by `pf` to `.csv` files in `result_dir`.
"""
FileIO.save(pf::PeriodsFinder, args...; kwargs...) = write_out_results(pf, args...; kwargs...)

"""
    write_out_results

Alias for `save`.
"""
function write_out_results(
        pf::PeriodsFinder,
        result_dir::String = get_abspath_to_result_dir(pf),
    )
    mkrootdirs(result_dir)

    # Selection and weights
    df_dv = DataFrame(
        periods = get_set_of_periods(pf),
        weights = pf.w,
        selected_periods = pf.u
    )
    CSV.write(joinpath(result_dir, "decision_variables.csv"), df_dv, delim=',')
    df_dv_s = deepcopy(sort(df_dv[df_dv[!,:selected_periods] .> 0, :]))
    CSV.write(joinpath(result_dir, "decision_variables_short.csv"), df_dv_s, delim=',')

    # Resulting time series
    df_ts = DataFrame()
    nppt = get_number_of_time_steps_per_period(pf)
    nrp = get_number_of_representative_periods(pf)
    rep_periods = get_set_of_representative_periods(pf)
    periods = get_set_of_periods(pf)
    time_steps = get_set_of_time_steps(pf)
    df_ts[!,:period] = repeat(rep_periods, inner=nppt)
    df_ts[!,:time_step] = repeat(time_steps, outer=length(rep_periods))
    vals = get_time_series_values(pf)
    for (ts_name, ts) in get_set_of_time_series(pf)
        df_ts[!,Symbol(ts_name)] = vals[ts_name][rep_periods,:]'[:]
    end
    CSV.write(joinpath(result_dir, "resulting_profiles.csv"), df_ts, delim=',')

    # Save the ordering variable
    if any(values(pf.v) .> 0)
        v = [
            pf.v[i,j] for i in periods, j in 1:length(rep_periods)
        ]
        permIdx = sortperm(rep_periods)
        v = v[:,permIdx]
        df = DataFrame(v, Symbol.(rep_periods[permIdx]))
        CSV.write(joinpath(result_dir, "ordering_variable.csv"), df, delim=',')
    end

    # Copy of config-file
    YAML.write_file(joinpath(result_dir, "config_file.yaml"), pf.config)

    # Objective value and bound to yaml file
    if isdefined(pf, :m)
        obj_val = objective_value(pf.m)
        obj_bound = try; objective_bound(pf.m); catch; NaN; end;
        optStatus = Dict(
            "objective_value" => obj_val,
            "objective_bound" => obj_bound,
            "optimality_gap_percentage" => (obj_val - obj_bound)/obj_val * 100
        )
        YAML.write_file(joinpath(result_dir, "optimization_results.yaml"), optStatus)
    end

    return pf
end

"""
    write_out_results(pf::PeriodsFinder, 
        result_dir::String = get_abspath_to_result_dir(pf);
        timestamps
    )

Writes out the synthetic time series as determined by `pf` for the ranges in `timestamp` to a `.csv` file in `result_dir`. A synthetic time series is the same length as the original time series but composed only of representative periods.

# Example
```julia
timestamps = Dict("Load" => DateTime(1970,1,1):Hour(1):DateTime(1970,1,2))
# result_dir not specified, since it is specified in pf.config 
create_synthetic_time_series_plots(pf; timestamps=timestamps)
```
"""
function write_out_synthetic_timeseries(pf::PeriodsFinder,
        result_dir::String = get_abspath_to_result_dir(pf);
        timestamps = Dict{String,Array{DateTime,1}}()
    )
    mkrootdirs(result_dir)
    for (ts_name, ts) in get_set_of_time_series(pf)
        tstamp = haskey(timestamps, ts_name) ? timestamps[ts_name] : timestamp(ts)
        ts_synth = get_synthetic_time_series(pf, ts)
        CSV.write(
            joinpath(result_dir, "$(ts_name)_synthetic_time_series.csv"),
            ts_synth
        )
    end
    return nothing
end