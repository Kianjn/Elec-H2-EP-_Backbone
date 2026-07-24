# # Re-ordering days in a year
# This script illustrates how to re-order days in a year, for example in order to go from 10 representative days to a continuous year of 365 (representative) days.

# First we create an instance of a PeriodsFinder. Here we are using the default `.yaml` configuration file, but keep in mind that a lot of the following lines of code could be avoided by creating your own custom file.
using RepresentativePeriodsFinder;
RPF = RepresentativePeriodsFinder;
config_file = RPF.datadir("default.yaml");
pf = PeriodsFinder(config_file, populate_entries=true);

# Change the result directory name.
script_name = splitext(splitdir(@__FILE__)[2])[1];
result_dir = joinpath(@__DIR__, "results", script_name);
pf.config["results"]["result_dir"] = result_dir;

# Turn off automatic plot creation.
pf.config["results"]["create_plots"] = false;

# We will only concern ourselves with mapping load, just to make things simple.
delete!(pf.config["time_series"], "LFW");
delete!(pf.config["time_series"], "LFS",);
delete!(pf.config["time_series"], "Residual Load");

# Next we define the total number of periods and the number of representative periods. This script was run without access to a professional solver such as CPLEX or Gurobi, so we will limit ourselves to creating a "year" of 20 days. Further on we will use an alternative formulation which will be able to create a full year of 365 days.
pf.config["method"]["options"]["total_periods"] = 20;
pf.config["method"]["options"]["representative_periods"] = 10;

# We define our representative periods, where each element in the vector is a day in the year.
rep_periods = [1, 3, 4, 5, 7, 12, 15, 16, 18, 19];

# We populate the results of the selection variable, `pf.u`, with our representative days. This is the only step that (currently) cannot be done within the `.yaml` file.
pf.u = zeros(Bool, 20);
pf.u[rep_periods] .= 1 ;

# The following lines define the selection and ordering method, specifically we will use an optimisation method which minimises only the absolute (as opposed to squared) time series error. 
delete!(pf.config["method"], "clustering");
delete!(pf.config["method"]["optimization"], "duration_curve_error");
pf.config["method"]["optimization"]["time_series_error"]["type"] = "absolute";

# We set binary ordering to false, which means that a non-representative day can be represented by a linear combination of representative days.
pf.config["method"]["optimization"]["binary_ordering"] = false;

# We set the mandatory periods to be selected to our representative periods.
pf.config["method"]["options"]["mandatory_periods"] = rep_periods;

# Setup the optimizer and solve to re-order the days. **Importantly don't forget to pass the keyword argument** `reset=false`!
using Ipopt, JuMP;
opt = optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 100);
find_representative_periods(pf, optimizer=opt, reset=false);

# We can plot the resulting synthetic time series. Note how days 1 and 3, which are representative, match up exactly while other days must be approximated. 
using Dates;
timestamps = Dict("Load" => DateTime(1970,1,1):Hour(1):DateTime(1970,1,7));
create_synthetic_time_series_plot(pf, "Load"; timestamps=timestamps)

# We can also check out the heatmap, which shows how representative days are mapped to non-representative days throughout the year:
create_ordering_heatmap(pf)

# We can also write out the this synthetic time series to `pf.config["results"]["result_dir"]`.
# ```julia
# write_out_synthetic_timeseries(pf, timestamps=timestamps)
# ```

# This last optimisation was done by solving a linear problem. We can also do this by solving a binary problem, where each representative day is mapped to a non-representative day as opposed to using a linear combination of days. The advantage of this is that we can define fancier error functions.

# Change the total number of days and re-define our representative periods:
pf.config["method"]["options"]["total_periods"] = 365;
rep_periods = [i for i in 1:35:350];
pf.config["method"]["options"]["mandatory_periods"] = rep_periods;
pf.u = zeros(Bool, 365);
pf.u[rep_periods] .= 1;

# For reasons not worth getting into, we need to re-populate `pf`:
populate_entries!(pf);

# We get rid of the time series error entry:
delete!(pf.config["method"]["optimization"], "time_series_error");

# Now we define our ordering error function, which is the error associated with representing a day `x` with another day `y`.
pf.config["method"]["options"]["ordering_error"] = Dict(
    "ord_err_1" => Dict(
        "function" => (x,y) -> sum((x .- y).^2),
        "weight" => 1.0
    )
);

# Here we've used the 2-norm error, but we could have specified anything we wanted.

# We set binary ordering to `true`:
pf.config["method"]["optimization"]["binary_ordering"] = true;

# Setup the optimizer and solve to re-order the days.
using Cbc;
opt = optimizer_with_attributes(Cbc.Optimizer, "seconds" => 60);
find_representative_periods(pf, optimizer=opt, reset=false);

# Let's inspect the heatmap again
result_dir = joinpath(@__DIR__, "results", script_name * "_binary");
pf.config["results"]["result_dir"] = result_dir;
create_ordering_heatmap(pf)

# Note how we have solid colors now, since we don't represent days by linear combinations of representative days.
