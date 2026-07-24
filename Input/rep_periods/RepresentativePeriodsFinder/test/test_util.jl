using RepresentativePeriodsFinder, TimeSeries, LinearAlgebra
RPF = RepresentativePeriodsFinder

@testset "Utilities" begin
    pf = PeriodsFinder()
    @test typeof(pf) <: RPF.PeriodsFinder

    # Test whether has_ordering_error makes sense
    pf.config["method"] = Dict(
        "options" => Dict(),
        "optimization" => Dict(
            "time_series_error" => Dict(
                "weight" => 0.5,
                "type" => "squared"
            )
        )
    )
    @test RPF.has_ordering_error(pf) == true
    
    pf.config["method"]["options"]["ordering_error"] = Dict(
        "ord_err_1" => Dict(
            "function" => x -> 1,
            "weight" => 0.0
        )
    )
    @test RPF.has_ordering_error(pf) == true
    @test_logs (:warn, ) RPF.get_error_term_weights(pf)
    pf.config["method"]["options"]["ordering_error"]["ord_err_1"]["weight"] = -1.0
    @test_logs (:error, ) RPF.get_error_term_weights(pf)
    
    delete!(pf.config["method"]["optimization"], "time_series_error")
    @test RPF.has_ordering_error(pf) == true
    
    # Test whether non-unique date / times in time series throws an error
    ta = TimeArray(
        [
            DateTime(2018, 11, 21, 12, 0),
            DateTime(2018, 11, 21, 12, 0)
        ],
        [10.2, 11.2],
        [:col1],
        Dict("name" => "example", "start" => DateTime(2018, 11, 21, 12, 0))
    )
    pf.time_series["t"] = ta
    pf.x = Dict{String, Matrix{Float64}}()
    pf.config = Dict(
        "method" => Dict(
            "options" => Dict(
                "sampling_time" => Hour(1),
                "total_periods" => 2,
                "time_steps_per_period" => 1
            )
        )
    )
    @test_throws AssertionError RPF.get_normalised_time_series_values(pf, "t")
    
    # Test whether bin discretisation does not lead to numerical errors
    ta = TimeArray(
        [
            DateTime(2018, 11, 21, 12, 0),
            DateTime(2018, 11, 21, 13, 0),
            DateTime(2018, 11, 21, 14, 0)
        ],
        [-1.0, 0.0, 1.0],
        [:col1],
        Dict("name" => "example", "start" => DateTime(2018, 11, 21, 12, 0))
    )
    pf.time_series["t"] = ta
    pf.x = Dict{String, Matrix{Float64}}()
    pf.config = Dict(
        "method" => Dict(
            "options" => Dict(
                "sampling_time" => Hour(1),
                "total_periods" => 3,
                "time_steps_per_period" => 1,
            ),
            "optimization" => Dict(
                "duration_curve_error" => Dict(
                    "number_bins" => 3
                )
            )
        )
    )
    pf.inputs = Dict()
    hpp = RPF.get_histogram_per_period(pf, "t")
    @assert hpp == I # I is from LinearAlgebra
end