using RepresentativePeriodsFinder
RPF = RepresentativePeriodsFinder

config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file)

@info "Populating entries.."
populate_entries!(pf)

print("\n\n")
@info "Adding different time series..."
pf.config["time_series"]["Missing Values"] = Dict(
    "source" => "missing_values.csv",
    "csv_options" => Dict(
        "dateformat" => "yyyy-mm-dd HH:MM:SS",
    ),
    "value_column"=> "Missing values",
    "timestamp_column"=> "Timestamp",
    "interpolate" => "linear"
)

pf.config["time_series"]["Linear Increase to Resample"] = Dict(
    "source" => "linear_increasing_24_hours.csv",
    "csv_options" => Dict(
        "delim" => ",",
    ),
    "value_column"=> "Value",
    "timestamp_column"=> "Timestamp",
    "resample" => true,
    "interpolate" => "linear"
)

populate_entries!(pf)