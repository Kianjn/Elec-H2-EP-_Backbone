using RepresentativePeriodsFinder, JuMP, Cbc
RPF = RepresentativePeriodsFinder

# Create PeriodsFinder
config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file, populate_entries=true)

# Delete entries for time series error and ordering error
delete!(pf.config["method"]["optimization"], "time_series_error")
delete!(pf.config["method"]["optimization"], "duration_curve_error")

# Can edit the number of representative days to be chosen here
n_rep = 2
n_total = 10
pf.config["method"]["options"]["representative_periods"] = n_rep
pf.config["method"]["options"]["total_periods"] = n_total
pf.config["method"]["optimization"]["integral_weights"] = true # for speed up

# Make optimisation model
m = RPF.make_periods_finder_model!(pf, optimizer_with_attributes(Cbc.Optimizer))

# Optimise it
RPF.optimize_periods_finder_model!(pf, m)

# Create the plots
RPF.create_plots(pf)

@testset "Ordering days" begin
    @test sum(pf.v) ≈ n_total
    @test all(sum(pf.v, dims=2) .≈ 1) # At least one day selected
    @test all(sum(pf.v, dims=1)[:] .≈ pf.w[pf.w .> 0])
end