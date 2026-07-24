############################## Copyright (C) 2019  #############################
#       The content of this file is VITO (Vlaamse Instelling voor              #
#       Technologisch Onderzoek  N.V.) proprietary.                            #
################################################################################
module RepresentativePeriodsFinder

    using CSV                               # -> read csv files
    using DataFrames                        # -> csv read into DataFrames
    using Dates                             # -> date time parsing
    using TimeSeries                        # -> time series type
    using YAML                              # -> read config-file
    using StatsBase                         # -> function to fit into Histogram bins
    using JuMP                              # -> optimization suite
    using Interpolations                    # -> interpolating time series
    using Plots; gr()                       # -> output plots
    using LinearAlgebra                     # -> for e.g. transpose()
    using NamedTupleTools                   # -> for easy creation of NamedTuples
    using MathOptInterface                  # -> for solver queries
    using FileIO                            # -> to redefine load and save
    using UnPack                            # -> to unpack dicts
    using ProgressMeter                     # -> showing progress for clustering

    # Utility
    include("util/types.jl")
    include("util/PeriodsFinder.jl")
    include("util/util.jl")
    include("util/get.jl")
    include("util/PeriodsFinder_util.jl") # Has to come after "get.jl" for doctest reasons. I think....

    # Input and data processing
    include("input/read_time_series.jl")
    include("input/process_time_series.jl")

    # Period finding methods
    include("methods/optimization.jl")
    include("methods/clustering.jl")

    # Outputs and results
    include("output/create_plots.jl")
    include("output/save.jl")
    include("output/load.jl")

    # Exported methods
    export PeriodsFinder, populate_entries!, reset_inputs!,
    find_representative_periods, find_representative_periods!,
    optimize_periods_finder_model!, make_periods_finder_model!,
    cluster_periods!,
    create_plots, create_duration_curve, create_ordering_heatmap,
    create_synthetic_time_series_plots, create_synthetic_time_series_plot,
    write_out_results, write_out_synthetic_timeseries,
    save, load

    # Some useful numbers
    max_periods = 365 # Number of periods before warning
    clust_options = (
        "chronological",
        "hierarchical"
    )

    """
        find_representative_periods

    Finds representative periods for a given set of time series, optionally ordering these as well.

    # Arguments
    One of the following must be specified:
    * `config_file::String` contains location of the `.yaml` configuration file.
    * `pf::PeriodsFinder` can be created as `PeriodsFinder(config_file)`.
    
    # Keyword Arguments
    * `optimizer::MOI.OptimizerWithAttributes` e.g. `Cbc.Optimizer`.
    * `reset::Bool=true` if `true` forces new representative periods to be found, else if `false` representative periods are ordered throughout the year. Only applicable if the selected method is `optimization`.

    # Examples
    ```julia
    config_file = "config.yaml"
    find_representative_periods(config_file)

    pf = PeriodsFinder(config_file)
    pf.u = rand(Bool, N_total) # Randomly select days
    find_representative_periods(pf, optimizer=Gurobi.Optimizer, reset=false)

    find_representative_periods(config_file, pf) # throws an error!
    ```
    """
    find_representative_periods

    function find_representative_periods(pf::PeriodsFinder;
            optimizer=nothing,
            reset::Bool=true
        )
        reset && (pf.u = Bool[])
        method = get_selection_method(pf)

        # Safety measure against having too many periods
        if get_number_of_periods(pf) > max_periods && method == "optimization"
            @warn """
            Number of periods is greater than $max_periods and selection algorithm is $method. This could crash Julia if you are also ordering periods.
            """
        end

        # Make and solve model
        if method == "optimization"
            m = make_periods_finder_model!(pf, optimizer)
            stat = optimize_periods_finder_model!(pf, m)
        elseif method == "clustering"
            time_period_clustering(pf)
            stat = MOI.OPTIMAL
        else
            error("""Selection method is $method, must be either "optimization" or "clustering".""")
        end

        # Write out results
        save_results = recursive_get(pf.config, "results", "save_results", true)
        create_plots_bool = recursive_get(
            pf.config, "results", "create_plots", true
        )
        found_solution = (method == "clustering" || has_values(pf.m))
        if stat in [MOI.OPTIMAL, MOI.TIME_LIMIT] && found_solution
            save_results && write_out_results(pf)
            create_plots_bool && create_plots(pf)
        end
        return pf
    end

    function find_representative_periods(config_file::String; kwargs...)
        pf = PeriodsFinder(config_file, populate_entries=true)
        find_representative_periods(pf; kwargs...)
    end

    """
        find_representative_periods!(pf; kwargs...)
    
    Alias for `find_representative_periods(pf; kwargs...)`.
    """
    function find_representative_periods!(pf::PeriodsFinder; 
        optimizer=nothing, reset::Bool=true
    )
        return find_representative_periods(pf; optimizer=optimizer, reset=reset)
    end

end


