using RepresentativePeriodsFinder
RPF = RepresentativePeriodsFinder
using Dates

# Create PeriodsFinder
config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file, populate_entries=true)

# Delete entries for optimization, add intermediate periods
# and use only 1 ordering error
delete!(pf.config["method"], "optimization")
pf.config["method"]["clustering"]["intermediate_periods"] = [200,100]
delete!(pf.config["method"]["options"]["ordering_error"], "ord_err_1") # remove from set
RPF.get_set_of_ordering_errors(pf) # Only ord_err_2 remains
pf.inputs[:ordering_error_functions]["ord_err_2"] = (
   (x,y) -> (sum((x[i] - y[i])^2 for i in eachindex(x)))
)

# Create and solve optimization problem
find_representative_periods(pf, reset=true)

# Solve hierarchical clustering
pf.config["method"]["clustering"]["type"] = "chronological"
find_representative_periods(pf, reset=true)

# Can check out the synthetic time series
create_synthetic_time_series_plots(pf, timestamps=Dict(
        "Load" => DateTime(1970):Hour(1):DateTime(1970,1,7),
    )
)