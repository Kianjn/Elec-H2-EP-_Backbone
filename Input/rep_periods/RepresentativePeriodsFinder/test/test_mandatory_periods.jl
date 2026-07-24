using RepresentativePeriodsFinder, JuMP, Cbc, StatsBase
RPF = RepresentativePeriodsFinder
# using Gurobi

# Create PeriodsFinder
config_file = RPF.datadir("default.yaml")
pf = PeriodsFinder(config_file, populate_entries=true)

# Delete clustering entries and ensure problem is an IP
delete!(pf.config["method"], "clustering")
delete!(pf.config["method"]["optimization"], "time_series_error")
delete!(pf.config["method"]["optimization"], "duration_curve_error")
pf.config["method"]["optimization"]["binary_ordering"] = true

# Reduce number of days
N_total = 20
N_rep = 5
pf.config["method"]["options"]["total_periods"] = N_total
pf.config["method"]["options"]["total_periods"] = N_rep

# Setup optimizer
opt = optimizer_with_attributes(Cbc.Optimizer)

@testset "Mandatory periods" begin
    # Special case of all periods set to mandatory
    pf.config["method"]["options"]["mandatory_periods"] = [i for i in 1:N_rep]
    find_representative_periods(pf, optimizer=opt) 
    @test all(pf.u[1:N_rep] .== 1)

    # Only on the first day of the year
    pf.config["method"]["options"]["mandatory_periods"] = [1]
    find_representative_periods(pf, optimizer=opt)
    @test pf.u[1] == 1

    # Same for clustering
    delete!(pf.config["method"], "optimization")
    pf.config["method"]["clustering"] = Dict(
        "type" => "hierarchical"
    )
    find_representative_periods(pf)
    @test pf.u[1] == 1
end