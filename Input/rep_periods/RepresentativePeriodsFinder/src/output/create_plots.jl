"""
    create_plots(pf::PeriodsFinder, result_dir::String)

Creates plots of original and aggregated duration curves and a heatmap of the ordering variable `v` if applicable and saves them in `result_dir`.
"""
function create_plots(
    pf::PeriodsFinder, result_dir::String=get_abspath_to_result_dir(pf)
)
    mkrootdirs(result_dir)
    filter_ts_names = recursive_get(
        pf.config, "results", "duration_curve_plot", "filter", String[]
    )
    marker = recursive_get(
        pf.config, "results", "duration_curve_plot", "marker", :cross
    )
    line = recursive_get(
        pf.config, "results", "duration_curve_plot", "line", :auto
    )
    for (ts_name, ts) in get_set_of_time_series(pf)
        ts_name ∈ filter_ts_names && continue
        p = create_duration_curve(
            pf, ts_name; line=Symbol(line), marker=Symbol(marker)
        )
        file_pdf = joinpath(result_dir, "$(ts_name)_duration_curve.pdf")
        savefig(p, file_pdf)
    end

    # Heatmap of ordering variable v, if v isn't too big
    if isdefined(pf, :v) && any(values(pf.v) .> 0) && all(size(pf.v) .< 1000)
        p = create_ordering_heatmap(pf)
        savefig(p, joinpath(result_dir, "ordering_heatmap.pdf"))
    end
    return nothing
end

"""
    create_duration_curve(pf::PeriodsFinder, ts_name::String; kwargs...)

Create the duration curve for `ts_name`.

# Example
```julia
pf = find_representative_periods(config_file)
p = create_duration_curve(pf, "Load")
display(p)

# Keyword Arguments

* `line=:auto`: line type.
* `marker=:cross`: marker type.
* `original_discretised=false`: if `true`, include original discretised duration curve.
* `aggregated_discretised=false`: if `true`, include aggregated discretised duration curve.
```
"""
function create_duration_curve(
    pf::PeriodsFinder,
    ts_name::String;
    line=:auto,
    marker=:cross,
    original_discretised=false,
    aggregated_discretised=false,
)
    rep_periods = get_set_of_representative_periods(pf)
    weights = pf.w[rep_periods]
    ntp = get_number_of_time_steps_per_period(pf)
    norm_val = get_normalised_time_series_values(pf, ts_name)
    nt = get_total_number_of_time_steps(pf)

    # Original
    x = [x for x in range(0; stop=nt) / nt * 100.0]
    # x = [x for x in range(0; stop=(nt-1)) / (nt-1) * 100.0]
    y = sort(norm_val[:]; rev=true)
    prepend!(y, y[1]) # first value has duration 0! Think about it.
    p = Plots.plot(
        x,
        y;
        xlim=(0, 100),
        dpi=300,
        size=(800, 800 / 28 * 21),
        label="Original",
        line=line,
        marker=marker,
    )

    # Reduced
    y = norm_val[rep_periods, :]'[:]
    x = transpose(weights * ones(1, ntp))[:] / nt * 100.0
    df = sort(DataFrame(; x=x, y=y), :y; rev=true)
    df = vcat(DataFrame(; x=0.0, y=df[1, :y]), df)
    df[!, :x] = cumsum(df[!, :x])
    # df = vcat(DataFrame(; x=0.0, y=df[1, :y]), df)

    Plots.plot!(
        p,
        df.x,
        df.y;
        label="Aggregated",
        title=ts_name,
        line=line,
        marker=marker,
    )

    if original_discretised || aggregated_discretised
        bins = get_set_of_bins(pf)
        y = get_bin_interval_values(pf, ts_name)
        y = midpoint(y)
        if original_discretised
            L = get_discretised_duration_curve(pf)
            x = 100 .- [L[ts_name][b] for b in bins] .* 100
            Plots.plot!(
                p,
                x,
                y;
                label="Original (discretised)",
                line=line,
                marker=marker,
            )
        end
        if aggregated_discretised
            N_total = get_number_of_periods(pf)
            A = get_duration_curve_parameter(pf)
            x =
                100 .-
                [
                    sum(
                        pf.w[j] / N_total * A[ts_name][j, b] for
                        j in rep_periods
                    ) for b in bins
                ] .* 100
            Plots.plot!(
                p,
                x,
                y;
                label="Aggregated (discretised)",
                line=line,
                marker=marker,
            )
        end
    end

    xaxis!("Duration [%]", 0:10:100)
    yaxis!("Normalised value [-]")
    return p
end

"""
    create_ordering_heatmap(pf)
    
Returns a heatmap showing the mapping of representative periods to non-representative periods. 
"""
function create_ordering_heatmap(pf::PeriodsFinder)
    rep_periods = get_set_of_representative_periods(pf)
    periods = get_set_of_periods(pf)
    v = zeros(length.([periods, periods])...)
    v[:, rep_periods] = pf.v
    return Plots.heatmap(
        v;
        ylabel="Sum = Ordering of periods",
        xlabel="Sum = Weighting of rep. period",
        title="Diagonal = Selection of rep. period",
        aspect_ratio=:equal,
        xlim=[minimum(periods), maximum(periods)],
        ylim=[minimum(periods), maximum(periods)],
    )
end

"""
    create_synthetic_time_series_plots(pf::PeriodsFinder,
        result_dir::String = get_abspath_to_result_dir(pf);
        timestamps
    )

Creates plots of original and synthetic time series for the ranges in `timestamp` and saves them in `result_dir`.

# Example
```julia
timestamps = Dict("Load" => DateTime(1970,1,1):Hour(1):DateTime(1970,1,2))
# result_dir not specified, since it is specified in pf.config 
create_synthetic_time_series_plots(pf; timestamps=timestamps)
```
"""
function create_synthetic_time_series_plots(
    pf::PeriodsFinder,
    result_dir::String=get_abspath_to_result_dir(pf);
    timestamps=Dict{String,Vector{DateTime}}(),
)
    mkrootdirs(result_dir)
    for (ts_name, ts) in get_set_of_time_series(pf)
        p = create_synthetic_time_series_plot(
            pf, ts_name; timestamps=timestamps
        )
        savefig(p, joinpath(result_dir, "$(ts_name)_synthetic_time_series.pdf"))
    end
    return nothing
end

"""
    create_synthetic_time_series_plot(pf::PeriodsFinder, ts_name; timestamps)
    
# Example
```julia
timestamps = Dict("Load" => DateTime(1970,1,1):Hour(1):DateTime(1970,1,2))
p = create_synthetic_time_series_plot(pf, "Load"; timestamps=timestamps)
display(p)
```
"""
function create_synthetic_time_series_plot(
    pf::PeriodsFinder,
    ts_name::String;
    timestamps=Dict{String,Vector{DateTime}}(),
)
    ts = pf.time_series[ts_name]
    tstamp = haskey(timestamps, ts_name) ? timestamps[ts_name] : timestamp(ts)
    return Plots.plot(get_synthetic_time_series(pf, ts)[tstamp]; title=ts_name)
end
