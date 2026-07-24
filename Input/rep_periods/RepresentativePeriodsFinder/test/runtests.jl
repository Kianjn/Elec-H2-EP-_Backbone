using Test, Pkg

tests = [
    f for f in readdir(@__DIR__) if !(isnothing(match(r"test_", f))) && !(isnothing(match(r".jl", f)))
]

println("Starting tests")
for t in tests
  println("\n","#"^80)
  println("Running ", t,)
  println("#"^80, "\n")
  include(t)
end

# Remove the files
rm(joinpath(@__DIR__, "results"), recursive=true)
