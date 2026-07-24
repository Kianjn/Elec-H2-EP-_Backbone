using Pkg

here = @__DIR__
Pkg.activate(here)
Pkg.develop(path=joinpath(here, "RepresentativePeriodsFinder"))
Pkg.resolve()
Pkg.instantiate()
println("SETUP_OK")
