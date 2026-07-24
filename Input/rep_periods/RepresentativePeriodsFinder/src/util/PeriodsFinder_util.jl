"""
    populate_entries!(pf::PeriodsFinder; reset_inputs::Bool=true)

Re-reads time series from the `.csv` files. If `reset_inputs=true`, also calls `reset_inputs!(pf)`.
"""
function populate_entries!(pf::PeriodsFinder)
    S = get_set_of_time_series_names(pf)
    
    for ts_name in S
        @info "Adding $ts_name..."
        ta = read_time_series(pf, ts_name)
        if haskey(meta(ta), "interpolation_type")
            @info "Interpolating missing values..."
            interpolate_missing_values!(pf, ta)
        end
        if haskey(meta(ta), "resample") && meta(ta)["resample"] == true
            @info "Resampling..."
            resample!(pf, ta)
        end
        pf.time_series[ts_name] = TimeArray(
            DateTime.(timestamp(ta)), Float64.(values(ta)), 
            colnames(ta), meta(ta)
        )
    end

    reset_inputs!(pf)

    return pf
end

"""
    reset_inputs!(pf::PeriodsFinder)

Resets the inputs of `pf`, specifically `pf.x` and `pf.inputs`. Should be called after making changes to `pf.config` to ensure that these are applied when selecting representative periods.

# Example

```@jldoctest; setup = :(using RepresentativePeriodsFinder)
RPF = RepresentativePeriodsFinder
pf = PeriodsFinder(RPF.datadir("only_load.yaml"); populate_entries=true);
println("Length of L: ", length(RPF.get_discretised_duration_curve(pf)["Load"]))

pf.config["method"]["optimization"]["duration_curve_error"]["number_bins"] = 80;
println("Length of L: ", length(RPF.get_discretised_duration_curve(pf)["Load"]))

reset_inputs!(pf);
println("Length of L: ", length(RPF.get_discretised_duration_curve(pf)["Load"]))

# output

[ Info: Adding Load...
[ Info: Interpolating missing values...
Length of L: 40
Length of L: 40
Length of L: 80

```
"""
function reset_inputs!(pf::PeriodsFinder)
    pf.x = Dict{String,Array{Float64,2}}()

    # Instantiate inputs, load error functions
    ord_err_names = get_set_of_ordering_errors(pf)
    pf.inputs = Dict(
        :ordering_error_functions => Dict{String,Function}(
            ord_err => get_ordering_error_function(pf, ord_err)
            for ord_err in ord_err_names
            # Do this now to avoid "new world" issues
        )
    )

    # To avoid weird bugs, also update the meta dictionary in the time series:
    S = get_set_of_time_series_names(pf)
    for (ts_name, ts) in get_set_of_time_series(pf)
        ts_name ∉ S && continue
        ts_dict = pf.config["time_series"]
        new_dict = Dict(
            k => v for (k, v) in merge(ts_dict["default"], ts_dict[ts_name])
        )
        md = meta(ts)
        md = merge(md, new_dict)
    end

    return pf
end
