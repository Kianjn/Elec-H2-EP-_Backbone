using Pkg, Documenter, Literate, RepresentativePeriodsFinder

# GR issues
ENV["GRDIR"] = ""
ENV["GKSwstype"] = "100"
Pkg.build("GR")

# This is heavily copied from COSMO.jl: https://github.com/oxfordcontrol/COSMO.jl/blob/master/docs/make.jl#L1

@info "Building examples..."
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")
fix_suffix(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
    return """
           The source files for all examples can be found in [/test](https://gitlab.kuleuven.be/UCM/representativeperiodsfinder.jl/-/tree/master/test/).
           """ * cont
end

# Copy some of the tests and input data to the example folder
data_path = RepresentativePeriodsFinder.datadir()
test_path = joinpath(@__DIR__, "..", "test")
example_file_names = [
    "test_days_re_ordering.jl",
    "test_rep_hours_finder.jl",
    "test_interfacing_with_TimeSeriesClustering.jl",
    "test_ramping_and_correlation_time_series.jl",
]
examples_path = joinpath(@__DIR__, "..", "examples")
isdir(examples_path) == false && mkdir(examples_path)
cp(data_path, joinpath(examples_path, "input_data"); force=true)
for ex in example_file_names
    cp(joinpath(test_path, ex), joinpath(examples_path, ex); force=true)
end

# Rename the examples to get rid of the test
for file in readdir(examples_path)
    endswith(file, ".jl") == false && continue
    isnothing(match(r"test_", file)) && continue
    new_file = string(split(file, "test_")[2])
    mv(
        joinpath(examples_path, file),
        joinpath(examples_path, new_file);
        force=true,
    )
end

# Build the example documentation
build_path = joinpath(@__DIR__, "src", "examples")
for file in readdir(examples_path)
    endswith(file, ".jl") == false && continue
    Literate.markdown(
        joinpath(examples_path, file),
        build_path;
        preprocess=fix_math_md,
        postprocess=postprocess,
        documenter=true,
        credit=true,
    )
end

@info "Building documentation..."
makedocs(;
    sitename="RepresentativePeriodsFinder",
    format=Documenter.HTML(),
    modules=[RepresentativePeriodsFinder],
    pages=[
        "Home" => "index.md",
        "User Guide" => [
            "Getting started" => "getting_started.md",
            "Loading time series" => "loading_time_series.md",
            "Selecting periods" => "selecting_periods.md",
            "Outputs" => "outputs.md",
            "Troubleshooting" => "troubleshooting.md",
        ],
        "Examples" => [
            "Re ordering representative days" => "examples/days_re_ordering.md",
            "Selecting representative hours" => "examples/rep_hours_finder.md",
            "Interfacing with TimeSeriesClustering.jl" => "examples/interfacing_with_TimeSeriesClustering.md",
            "Ramping and correlation time series" => "examples/ramping_and_correlation_time_series.md",
        ],
        "Methods" => "methods.md",
        "API Reference" => "api.md",
    ],
)
