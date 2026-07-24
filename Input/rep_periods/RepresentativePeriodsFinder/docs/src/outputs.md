# Outputs

## Saving and inspecting results

### Overview

If the option `save_results` or `create_plots` in `results` of the configuration file is true, then running [`find_representative_periods`](@ref) will save the selection results to `.csv` and create duration curve plots in the specified directory. 

Otherwise the following functions can be called:
```julia
dir = "/home/user/Desktop/selecting_periods/results"
save(pf, dir)
create_plots(pf, dir)
```

If no directory is specified then the directory specified in `results: result_dir` will be used.

### Options description

The results / output entry of the `.yaml` file may look something like this:

```yaml
results:
  save_results: true
  # Options for exactly what to save here
  result_dir: 'results' # Relative to this file
  create_plots: true # Creates plots
  duration_curve_plots:
    marker: cross # Marker type, e.g. "none"
    line: scatter # Line type, e.g. "steppre"
    filter: [] # Time series to avoid
  ordering_variable_heatmap_plot: false
```

* `save_results=true`: If `true`, saves results i.e. the periods selected, their weights, their ordering, ... as `.csv` files in `result_dir`.
* `result_dir="results"`: Result path relative to the location of the `.yaml` file.
* `create_plots=true`: If `true`, also saves duration curve plots in `result_dir`.
* `duration_curve_plots`:
  * `marker="cross"`: Marker type (see [`Plots.jl`](https://docs.juliaplots.org/latest/generated/attributes_series/))
  * `line="auto"`: Line type (see [`Plots.jl`](https://docs.juliaplots.org/latest/generated/attributes_series/))
  * `filter=[]`: Vector of strings of the time series whose duration curve error plots should not be plotted.
* `ordering_variable_heatmap_plot=false`: If true, a heatmap of the ordering variable will be saved to `result_dir`. Not recommended for > 365 periods for computational issues.

### Additional utility functions.

* [`create_synthetic_time_series_plot`](@ref)

## Loading results

Results saved using [`save(pf, dir)`](@ref) can also be loaded using [`load(pf, dir)`](@ref).
