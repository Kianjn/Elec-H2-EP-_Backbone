# Loading time series

## Overview

In the absence of additional documentation, refer to the examples in the [`test`](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl/-/tree/dev/test) directory of this repository to understand the options available for loading in time series.

For the purposes of illustration, consider [`default.yaml`](https://gitlab.kuleuven.be/UCM/representativedaysfinder.jl/-/blob/dev/test/input_data/default.yaml) which is used for testing this package.

```yaml
time_series:
  default: # These values will be used unless specified in the entries below
    source: "time_series.csv"
    csv_options: # These are passed to CSV.read
        delim: ";"
        silence_warnings: false # If true, won't throw error msgs when reading csv
    timestamp: "Timestamp" # optional
    weight: 1 # weight of time series in objective / clustering
    interpolation_type: "linear" # or "constant" or don't specify
    sampling_time: "Hour(1)" # Only necessary if a timestamp is not given
  
  Residual Load:
    source: "residual_load.csv"
    csv_options: 
      dateformat: "yyyy-mm-dd HH:MM:SS"
      # delim: ";"
      missingstring: "#N/A"
      # missingstrings: ["NA", "N/A", "na"] # Careful with the plural!
      silencewarnings: true
    value_column: "Residual Load [MW]"
    timestamp_column: "Timestamp"
    
    start: "2017-01-01 00:00:00"
    resample: true # or false or don't specify

  Load:
    value_column: "Load"
    start: 1 # Can also specify an index in the CSV file (not including header)

  LFW:
    value_column: "LFW"
    csv_options:
      delim: ';'
    weight: 1

  LFS:
    value_column: "LFS"
    weight: 1
```

To load the above time series (without selecting representative periods) run [`pf = PeriodsFinder(config_file)`](@ref PeriodsFinder) where `config_file` is the path to your chosen configuration file.

The `default` entry specifies options to be used if these are not specified for the time series entries. So `Load` is read in from `time_series.csv` since there is no `source` option specified for it, while `Residual Load` is read from `residual_load.csv`.

To "deactivate" a time series (i.e. only output it in the results without having it affect the aggregation algorithm) simply set the `weight` entry to `0`.

## Options description

Default is indicated by an `=` sign.

* `source`: Path of the `.csv` file to be read relative to the `.yaml` file.
* `csv_options`: Key word arguments that are passed to [`CSV.read`](https://csv.juliadata.org/stable/reading.html).
* `value_column`: Name of the value column in the `.csv` file.
* `timestamp`: Name of timestamp column in the `.csv` file (optional, but necessary for evventual interpolation).
* `weight=0.0`: Weight assigned to this time series when selecting representative periods.
* `interpolation_type="linear"`: Can be `"linear"`, `"constant"` or unspecified if no interpolation should be done. Using this feature for anything other than marginal cases is not recommended, particularly since the package works with time stamps and not time slices which makes interpolation ambiguous at times. See [`interpolate_missing_values!`](@ref RepresentativePeriodsFinder.interpolate_missing_values!), [`resample!`](@ref RepresentativePeriodsFinder.resample!) and [`get_contributing_time_slices`](@ref RepresentativePeriodsFinder.get_contributing_time_slices).
* `resample=false`: If `true`, the time series will be resampled so as to have a constant time step length. Again, using this option is to be avoided if possible and see the above references.
* `sampling_time=Hour(1)`: Sampling time or length of a timestep in a `DateTime` format.
* `start=1`: Starting point of the time series as an integer corresponding to the row or a time stamp. Useful if you have e.g. 5 years worth of data for one time series but only 1 for another and you want to specify to use year 3 of the first. The end of the time series is specified by the total number of time steps specified (see [Selecting representative periods](@ref)).
