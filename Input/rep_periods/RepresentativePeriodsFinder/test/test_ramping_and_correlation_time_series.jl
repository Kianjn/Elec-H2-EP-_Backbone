# # Ramping and correlation time series

# In the original paper by [Poncelet et al.](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf) there was the possibility of using a correlation time series, e.g. for electrical load and solar power (which are highly correlated over the course of a day). Equally, it may be interesting to select representative periods based on a ramping or mileage time series (e.g. to select days with a high rate of change in the load for example).

# Eventually, this could be added as a core feature within `RepresentativePeriodsFinder.jl`, and indeed there are still two open issues regarding this (see [here](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl/-/issues/16) and [here](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl/-/issues/17)). This is not a priority however and will not be done unless someone (that could be you!) opens a pull request. Anyway, enough faffing about - let's get cracking!

# ## Ramping time series

# Let's create a `PeriodsFinder` instance:
using RepresentativePeriodsFinder, Plots, TimeSeries, HiGHS, JuMP;
RPF = RepresentativePeriodsFinder;
config_file = RPF.datadir("default.yaml");
pf = PeriodsFinder(config_file; populate_entries=false);

# Helper functions for later:
function make_optimization!(pf::PeriodsFinder)
    pf.config["results"]["save_results"] = false
    pf.config["method"]["options"]["representative_periods"] = 8
    delete!(pf.config["method"]["optimization"], "time_series_error")
    pf.config["method"]["optimization"]["duration_curve_error"]["type"] = "absolute"
    delete!(pf.config["method"]["options"], "ordering_error")
    delete!(pf.config["method"], "clustering")
    reset_inputs!(pf)
    return pf
end

function make_clustering!(pf::PeriodsFinder)
    pf.config["results"]["save_results"] = false
    pf.config["method"]["options"]["representative_periods"] = 8
    delete!(pf.config["method"], "optimization")
    delete!(pf.config["method"]["options"]["ordering_error"], "ord_err_2")
    pf.config["method"]["options"]["ordering_error"]["ord_err_1"]["weight"] =
        1.0
    reset_inputs!(pf)
    pf.inputs[:ordering_error_functions]["ord_err_1"] = (
        (x, y) -> (sum((x[i] - y[i])^2 for i in eachindex(x)))
    )
    return pf
end;

function plot_duration_curves(pf::PeriodsFinder)
    plt_vec = [
        create_duration_curve(
            pf,
            ts_name;
            line=:solid,
            marker=:none,
        ) for (ts_name, ts) in RPF.get_set_of_time_series(pf)
    ]
    plt = Plots.plot(plt_vec...)
    display(plt)
    return plt
end;

# Delete some time series to speed up the selection process
for ts_name in ["LFW", "Load", "LFS"]
    delete!(pf.config["time_series"], ts_name)
end
populate_entries!(pf);

# Let's now create a ramping time series based on the residual load time series using `diff`. Using `deepcopy` instead of `copy` is extremely important if you want to avoid weird bugs! Otherwise any changes made to the `meta` dictionary in one time series will be made to that of the other time series to.
pf.time_series["RL Ramp"] = diff(deepcopy(pf.time_series["Residual Load"]));

# The above time series is 8783 entries long, but that's not as an issue - the configuration file will only select the first 8,760 entries. However, it starts an hour earlier than "Residual Load" - this is an issue! Let's fix it.
timestamp(pf.time_series["RL Ramp"]) .-= Hour(1)

# We also need to change some settings in the "RL Ramp" meta dictionary:
meta(pf.time_series["RL Ramp"])["name"] = "RL Ramp";
meta(pf.time_series["RL Ramp"])["weight"] = 0.1;

# And also add an entry for it in the config dictionary. This avoids some unfortunate bugs.
pf.config["time_series"]["RL Ramp"] = Dict()

# And now let's find us some representative days:
make_optimization!(pf)
opt = optimizer_with_attributes(HiGHS.Optimizer, "time_limit" => 15.0)
find_representative_periods(pf; optimizer=opt, reset=true)

# And what do these duration curves look like then?
plot_duration_curves(pf)

# It's worth pointing out how we got a pretty good approximation of the duration curves despite an MIP gap of 100%. This happens quite often, though it's of course always good to check.

# ## Correlation time series

# What you saw above was the hard way of doing things. It's much easier to simply add the new timeseries to the .csv file and then load it, which is what we'll do here.

# The original paper of [Poncelet et al.](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wp-2015-10b.pdf) suggests to use an additional time series which explicitly captures the correlation between two time series. It's easier to understand this with an example.

using Statistics, CSV, DataFrames, YAML;
config_file = RPF.datadir("default.yaml");
pf = PeriodsFinder(config_file; populate_entries=true);
x_1 = permutedims(RPF.get_normalised_time_series_values(pf)["LFS"], [2, 1])[:];
x_2 = permutedims(RPF.get_normalised_time_series_values(pf)["Load"], [2, 1])[:];
x_1_m, x_2_m = mean.((x_1, x_2));
corr_12 = [(x_1[i] - x_1_m) * (x_2[i] - x_2_m) for i in eachindex(x_1)];
T = 1:48
p = Plots.plot(
    T .- 1,
    getindex(hcat(x_1, x_2, corr_12), T, :);
    lab=["Solar" "Load" "Correlation"],
    xlab="Time [h]",
    ylab="Normalised time series value [-]",
    legend=:topleft,
    color=[:red :blue :orange],
    lw=2,
    ylims=(-1, 1),
)
Plots.plot!(
    p,
    T[[1, end]] .- 1,
    hcat([x_1_m, x_1_m], [x_2_m, x_2_m]);
    lab=["Mean Solar" "Mean Load"],
    color=[:red :blue],
    lw=2,
    linestyle=:dash,
)

# Clearly the correlation time series is positive when the solar and load time series are both positive or both negative, and negative otherwise. In this way, it's capturing the correlation between the time series.

# Let's write this time series to a new file:
corr_ts_file = RPF.datadir("corr_time_series.csv");
CSV.write(corr_ts_file, DataFrame(; corr12=corr_12));

# Now let's make a new config file with the time series in it:
pf.config["time_series"]["Corr Load-LFS"] = Dict(
    "source" => "corr_time_series.csv",
    "csv_options" => Dict("validate" => false),
    "value_column" => "corr12",
    "weight" => 10,
    "start" => 1,
);
# for ts_name in ["Residual Load", "Corr Load-LFS", "Load", "LFS"]
for ts_name in ["Residual Load", "Load", "LFS"]
    delete!(pf.config["time_series"], ts_name)
end
corr_config_file = RPF.datadir("corr.yaml")
YAML.write_file(corr_config_file, pf.config);

# Make a new `PeriodsFinder` instance:
pf = PeriodsFinder(corr_config_file; populate_entries=true);

# Find them days:
make_clustering!(pf);
find_representative_periods(pf);

# Plot the duration curves:
plot_duration_curves(pf)

# The approximation of the duration curve of the wind time series is unsurprisingly not great, since we put such a high weight on the correlation time series. 

#jl # Delete the files we created.
#jl rm(corr_ts_file; force=true)
#jl rm(corr_config_file; force=true)

