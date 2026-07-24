# [Getting started] (@id getting_started_page)

[`RepresentativePeriodsFinder.jl`](https://ucm.pages.gitlab\.kuleuven\.be/representativeperiodsfinder.jl/) works principally by specifying a `.yaml` configuration file, which specifies which time series to load, how to select them, and where to save the results. In the process, a `PeriodsFinder` type is defined which holds all this data.

It is perhaps easiest to check out an [example of a configuration file](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl/-/blob/master/data/default.yaml), though the rest of this page will describe what one of these should look like. For illustrative purposes, suppose that this configuration file is located at `/home/user/Desktop/selecting_periods/config_file.yaml`, and that it contains the text shown below:

```yaml
# Specify csv files for time series and other options
time_series:
  default: # These values will be used unless specified in the entries below
    source: "time_series.csv"
    csv_options: # These options are passed to CSV.read
        delim: ";"
    timestamp: "Timestamp" # optional
    weight: 1 # weight of time series in objective / clustering
    interpolation_type: "linear" # or "constant" or don't specify
    sampling_time: "Hour(1)" # Only necessary if a timestamp is not given
    start: "2017-01-01 00:00:00" # Can also specify an integer which corresponds to rows in the .csv file

  Load:
    value_column: "Load" # Column name to be read in csv file

  Solar:
    value_column: "load_factor_PV"

method:
  options: # Method agnostic options
    mandatory_periods: [200] # Leave empty to specify none
    total_periods: 365
    representative_periods: 8
    time_steps_per_period: 24
    sampling_time: "Hour(1)" # This is a DateTime format - for quarter hours, Hour(0.25)
  
  optimization:
    integral_weights: false
    binary_ordering: true
    equal_weights: false
    duration_curve_error:
      weight: 0.5
      number_bins: 40
      type: "absolute" # Or "absolute"

results:
  save_results: true
  # Options for exactly what to save here
  result_dir: 'results' # Relative to this file
  create_plots: true
```

The three principal entries of `config_file.yaml`, `time_series`, `method` and `results` are explained in the sections on [loading in a time series](@ref ts_gs), [selecting representative periods](@ref select_gs) and [Saving and inspecting results](@ref save_gs).

For more advanced use cases it may be helpful to check out the examples (see left hand side of this page) as well as the tests in the [`test`](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl/-/tree/dev/test) directory of this repository.

## [Loading in time series] (@id ts_gs)
`config_file.yaml` specifies two time series to be read, `Load` and `Solar`, from `time_series.csv`, which has 3 columns: `Timestamp`, `Load` and `load_factor_PV`.

Load the time series by typing:
```julia
using RepresentativePeriodsFinder
config_file = "/home/user/Desktop/selecting_periods/config_file.yaml"
pf = PeriodsFinder(config_file)
```

## [Selecting representative periods] (@id select_gs)
The configuration specified by `config_file.yaml` will select 8 representative days of 24 hours by solving an optimisation problem which minimises the difference between the original and aggregated load duration curves. Additionally, the 200th day of the year is forced to be selected.

To select the representative periods:
```julia
using Pkg
Pkg.add("JuMP") # add optimisation package
Pkg.add("Cbc") # add package for open source solver
using JuMP, Cbc
optimizer = optimizer_with_attributes(Cbc.Optimizer, "seconds" => 300) # seconds specifies time out limit
pf = find_representative_periods(pf, optimizer=optimizer)
```

!!! note
  We're able to use the `Cbc` solver here because the resulting optimisation is a mixed integer linear problem. If we had chosen to the duration curve error type to be `"squared"` then we would need a solver capable of handling mixed integer quadratic problems, such as Gurobi.

## [Saving and inspecting the results] (@id save_gs)
The configuration specified by `config_file.yaml` will save the selected representative periods to `.csv` files in `/home/user/Desktop/selecting_periods/results/` along with plots of the original and aggregated duration curves.
