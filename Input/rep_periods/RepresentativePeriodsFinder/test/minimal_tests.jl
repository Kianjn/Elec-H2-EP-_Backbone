# Only test essentials, used for testing different julia versions
using Test

include(joinpath(@__DIR__, "test_util.jl"))
include(joinpath(@__DIR__, "test_days_finder.jl"))
include(joinpath(@__DIR__, "test_saving_and_loading.jl"))
