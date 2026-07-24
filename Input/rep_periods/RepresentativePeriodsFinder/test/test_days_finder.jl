using RepresentativePeriodsFinder, JuMP, Cbc
RPF = RepresentativePeriodsFinder

# Create PeriodsFinder
config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file, populate_entries=true)

# Delete entries for time series and duration curve error
delete!(pf.config["method"]["optimization"], "time_series_error")
delete!(pf.config["method"]["options"], "ordering_error")
delete!(pf.config["method"], "clustering")

# Make life a bit easier for the optimiser, reduce the number of days
N_total = 20
N_rep = 2
pf.config["method"]["options"]["total_periods"] = N_total
pf.config["method"]["options"]["representative_periods"] = N_rep

# Setup optimizer
opt = optimizer_with_attributes(Cbc.Optimizer, "seconds" => 60)
# using Gurobi
# opt = optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 60)

# Create and solve optimization problem
# find_representative_periods(pf, optimizer=opt, reset=true)

# Try absolute error (this is the only possibility with Cbc)
pf.config["method"]["optimization"]["duration_curve_error"]["type"] = "absolute"
find_representative_periods(pf, optimizer=opt, reset=true)

