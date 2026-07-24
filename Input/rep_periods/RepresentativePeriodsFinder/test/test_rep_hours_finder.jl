# # Selecting representative hours using optimisation

# In this example we'll see how to select representative hours from a single day using the optimisation based approach. Obviously this approach could be extended to hours in a year, as is done at the end.

# First we create an instance of a PeriodsFinder. We'll use the default `.yaml` configuration file and modify it, but you could just edit it directly.
using RepresentativePeriodsFinder, PrettyPrint, JuMP, HiGHS, Plots;
RPF = RepresentativePeriodsFinder;
config_file = RPF.datadir("default.yaml");
pf = PeriodsFinder(config_file; populate_entries=false);
pprintln(pf.config)

# Delete entries for clustering, time series and ordering error. This way we'll select hours.
delete!(pf.config["method"]["optimization"], "time_series_error");
delete!(pf.config["method"]["options"], "ordering_error");
delete!(pf.config["method"], "clustering");

# For debugging and illustrative purposes, let's focus just on solar availailability time series
ts_orig = copy(pf.config["time_series"]);
for ts_name in ["LFW", "Load", "Residual Load"]
    delete!(pf.config["time_series"], ts_name)
end

# Use absolute error (this is the only possibility with MILP solvers)
pf.config["method"]["optimization"]["duration_curve_error"]["type"] = "absolute";

# Increase number of bins so as to get exact solution
pf.config["method"]["optimization"]["duration_curve_error"]["number_bins"] =
    1_000;

# Make life a bit easier for the optimiser, let's reduce the number of hours from which we're selecting, i.e. we're shortening the year to the first day. This will also help with understanding what's happening.
pf.config["method"]["options"]["total_periods"] = 24;

# There are 8 non-zero values in the first day and 16 zero values. With 9 representative hours we chould be able to capture the entire spectrum.
n_rep = 9;
pf.config["method"]["options"]["representative_periods"] = n_rep;

# Change period length to 1 (24 = day selection)
pf.config["method"]["options"]["time_steps_per_period"] = 1;

# Turn off automatic plot creation
pf.config["results"]["create_plots"] = false;

# Change the result directory name.
script_name = splitext(splitdir(@__FILE__)[2])[1];
result_dir = joinpath(@__DIR__, "results", script_name);
pf.config["results"]["result_dir"] = result_dir;

# This is what the resulting configuration file looks like now:
pprintln(pf.config);

# Populate the entries now (could have also done this initially)
populate_entries!(pf)

# Setup optimizer. Let's use HiGHS for a change.
opt = optimizer_with_attributes(
    HiGHS.Optimizer, "time_limit" => 60.0, "output_flag" => true
);

# Find the representative hours
find_representative_periods(pf; optimizer=opt, reset=true);

# Plot the duration curve of solar
plt_vec = Plots.Plot[]
for (ts_name, ts) in RPF.get_set_of_time_series(pf)
    push!(
        plt_vec,
        create_duration_curve(
            pf,
            ts_name;
            line=:solid,
            marker=:none,
            original_discretised=true,
            aggregated_discretised=true,
        ),
    )
end
Plots.plot(plt_vec...)

# We can see that there's some discretisation issues at the lower end of the duration curve. This is more of a plotting issue than an optimisation issue - the solver was able to reduce the error to 0 by assigning the non-zero hours to be representative and have a weight 1 and one hour with a zero value to have a weight of 16:

objective_value(pf.m)

pf.w[RPF.get_set_of_representative_periods(pf)]

# We can repeat this exercise for all time series and 7 days (though this time with less bins, since this drastically increases computation time).

pf.config["time_series"] = ts_orig;
pf.config["method"]["options"]["total_periods"] = 7 * 24;
pf.config["method"]["options"]["representative_periods"] = 20;
pf.config["method"]["optimization"]["duration_curve_error"]["number_bins"] = 40;
populate_entries!(pf)
find_representative_periods(pf; optimizer=opt, reset=true);

# Plot the duration curves of all the time series to see how we got on:
plt_vec = Plots.Plot[]
for (ts_name, ts) in RPF.get_set_of_time_series(pf)
    push!(plt_vec, create_duration_curve(pf, ts_name;))
end
Plots.plot(plt_vec...)

# Here we only selected 10 hours from 168 using 40 bins, and as you can see above it did not find a solution within 60 seconds. Selecting hours from the year using an optimisation method is therefore not recommended as it can be computationally expensive. A clustering based method is preferable.

#jlAssert that the ordering variable is not made
#jl@testset "Optimisation based hour selection" begin
#jl    @test typeof(pf.m.ext[:variables][:v]) <: RPF.SingleValuedContainer
#jl    @test haskey(pf.m.ext[:constraints], :restrict_ordering) == false
#jl    @test haskey(pf.m.ext[:constraints], :linear_combination_representative_periods) == false
#jl    @test length(RPF.get_set_of_representative_periods(pf)) == n_rep
#jlend
