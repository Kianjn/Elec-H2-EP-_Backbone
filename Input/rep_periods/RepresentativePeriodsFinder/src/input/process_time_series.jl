"""
    interpolate_missing_values!(pf::PeriodsFinder, ts_name::String)

Interpolates any missing values in the time series. Currently, this is only done for rows with a time stamp:

```
1970-01-01 00:00 10.0
1970-01-01 00:01 NA     # This value will be interpolated
#                       # But this missing row, for hour 2, will not
1970-01-01 00:03 10.5
```

"""
function interpolate_missing_values!(pf::PeriodsFinder, ta::TimeArray)
    ts_name = meta(ta)["name"]
    ts_dict = pf.config["time_series"]

    interpolation_type = get(meta(ta), "interpolation_type", "linear")
    itp = make_interpolator(ta, interpolation_type)
    
    A = values(ta)
    n_missing = sum(ismissing.(A))
    if n_missing > 0
        @debug "Interpolating for $n_missing values"
        for i in 1:length(ta)
            val = values(ta[i])[1]
            x = time_to_grid_val(timestamp(ta[1])[1], timestamp(ta[i])[1])
            if ismissing(val)
                A[i,:] .= itp(x)
            end
        end
    end

    return ta
end

"""
    resample!(pf::PeriodsFinder, ts_name::String)

Resamples the time series to the global sampling time such that all time series have the same resolution.

If the time series' sampling time is lower than the global's, linear interpolation is used to resample the time series. In the case that it is greater, the mean value is used. This is done on a per timestep basis.

# Examples
Converting a variable duration time series to an hourly time series.
```
1970-01-01T00:00:00 -> Value for hour 1
1970-01-01T01:00:00 -> Linear interpolation between hours 1 and 4 to get hour 2
                    -> Linear interpolation between hours 1 and 4 to get hour 3
1970-01-01T04:00:00 -> 15 minute jump, weight of 1/4 of hour 4's value
1970-01-01T04:15:00 -> 45 minute jump, weight of 3/4 of hour 4's value
                    -> 30 minute jump, weight of 1/2 of hour 5's value
1970-01-01T05:30:00 -> 
1970-01-01T06:00:00 -> And so on
...
1970-31-12T023:00:00 -> Last time step is assumed to be of the same length as the time series sampling time
```
"""
function resample!(pf::PeriodsFinder, ta::TimeArray)
    ts_name = meta(ta)["name"]
    ts_dict = pf.config["time_series"]
    timestamps = timestamp(ta)

    interpolation_type = get(meta(ta), "interpolation_type", "linear")
    itp = make_interpolator(ta, interpolation_type)

    sampling_time = get_sampling_time(pf)
    new_timestamps = timestamps[1]:sampling_time:timestamps[end]
    n_total = length(new_timestamps)
    start_time = timestamps[1]
    end_time = start_time + n_total * sampling_time

    @assert typeof(Hour((start_time - meta(ta)["start"])).value) <: Int
    
    data = Float64[]
    index_closest_t_before = 1
    for i in 1:n_total-1
        t_start = start_time + (i-1) * sampling_time
        t_end = t_start + sampling_time
        index_closest_t_after = findfirst(
            x -> x >= t_end, timestamps[index_closest_t_before:end]
        ) + length(timestamps[1:index_closest_t_before]) - 1
        index_closest_t_after
        index_closest_t_before
        timestamps[index_closest_t_after]
        (t, w) = get_contributing_time_slices(
            index_closest_t_before, index_closest_t_after,
            t_start, t_end, timestamps
        )
        if length(t) > 1
            push!(data, mean(
                values(ta[t]), weights(Dates.value.(w))
            ))
        else
            x = time_to_grid_val(start_time, t_start)
            push!(data, itp[x])
        end

        index_closest_t_before = findfirst(
            x -> x >= t_end, timestamps[index_closest_t_before:end]
        ) + length(timestamps[1:index_closest_t_before]) - 1
    end
    push!(data, values(ta[end])[1])

    return ta = TimeArray(new_timestamps, data, colnames(ta), meta(ta))
end

"""
    make_interpolator(ta::TimeArray, interpolation_type::String)

Makes an interpolation object from a TimeArray. Assumes point values in time, as opposed to constant values over a block or slice in time.
"""
function make_interpolator(ta::TimeArray, interpolation_type::String)
    y = eltype(values(ta))[]
    grid = Int64[]
    for i in 1:length(ta)
        val = values(ta[i])[1]
        x = time_to_grid_val(timestamp(ta[1])[1], timestamp(ta[i])[1])
        if ismissing(val) == false
            push!(y, val)
            push!(grid, x)
        end
    end

    if interpolation_type == "linear"
        itp = interpolate((grid,), y, Gridded(Linear()))
        itp = extrapolate(itp, Flat())
        return itp
    elseif interpolation_type == "constant"
        itp = interpolate((grid,), y, Gridded(Constant()))
        itp = extrapolate(itp, Flat())
        return itp
    end

end

function time_to_grid_val(t_init::DateTime, t::DateTime)
    return (t - t_init).value
end

"""
    get_contributing_time_slices(
        idx_start::Integer, idx_end::Integer,
        t_start::DateTime, t_end::DateTime,
        timestamps::Array{DateTime,1}
    )
Get the time slices contributing to the current time slice as well as their contribution (i.e. weights).

# Example

```julia
timestamps = DateTime(1970,1,1) .+ [
    Hour(0),
    Hour(1),
    Hour(4),
    Hour(4) + Minute(15),
    Hour(5) + Minute(30),
    Hour(6)
]
idx_start = 2
idx_end = 5
t_start = DateTime(1970,1,1) + Hour(3)
t_end = DateTime(1970,1,1) + Hour(5)
(t, weights) = get_contributing_time_slices(idx_start, idx_end, t_start, t_end, timestamps)

julia> t
3-element Array{DateTime,1}:
 1970-01-01T01:00:00
 1970-01-01T04:00:00
 1970-01-01T04:15:00

julia> weights
3-element Array{Millisecond,1}:
 3600000 milliseconds
 900000 milliseconds
 2700000 milliseconds
```

"""
function get_contributing_time_slices(
        idx_start::Integer, idx_end::Integer,
        t_start::DateTime, t_end::DateTime,
        timestamps::Array{DateTime,1}
    )
    tslices = timestamps[idx_start:idx_end]
    weights = Millisecond[]
    t = DateTime[]
    for i in 1:length(tslices) - 1
        push!(t, tslices[i])
        if tslices[i] < t_start
            push!(weights, tslices[i+1] - t_start)
        elseif tslices[i+1] > t_end
            push!(weights, t_end - tslices[i])
        else
            push!(weights, tslices[i+1] - tslices[i])
        end
    end
    return (t, weights)
end
