# ==============================================================================
# define_offtaker_parameters.jl — Offtaker agent parameters
# ==============================================================================
#
# PURPOSE:
#   Copies all key-value pairs from the agent's data block (Hydrogen_Offtaker in
#   data.yaml) into mod.ext[:parameters], so that build_offtaker_agent! and
#   solve_offtaker_agent! can read Type, Capacity, Capacity_H2_In, Capacity_EP_Out,
#   Alpha, ProcessingCost, MarginalCost, gamma_NH3, ImportCost, etc. Also sets
#   gamma_GC (default 0.42) for the 42% H₂ GC mandate shared by green and grey
#   offtakers.
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — data = merge(General, Hydrogen_Offtaker[agent]).
#
# ==============================================================================

function define_offtaker_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]

    # Copy every key-value pair from the agent's data.yaml block into params.
    # Why copy all keys generically instead of listing them explicitly?
    # → Flexible: any new parameter added to data.yaml (e.g. a ramp rate or
    #   storage capacity) is automatically available in build_offtaker_agent!
    #   and solve_offtaker_agent! without modifying this function.
    # Symbol(k) converts the YAML string key to a Julia Symbol so we can
    # access values as params[:Capacity], params[:Alpha], etc.
    for (k, v) in data
        params[Symbol(k)] = v
    end

    # Regulatory green-certificate mandate: at least 42% of end-product output
    # must be backed by hydrogen Guarantees of Origin (H₂ GCs). This reflects
    # EU renewable-energy targets for hard-to-abate sectors. Both green and grey
    # offtakers are subject to this constraint. Defaults to 0.42 if not overridden.
    params[:gamma_GC] = get(data, "gamma_GC", 0.42)

    return mod
end
