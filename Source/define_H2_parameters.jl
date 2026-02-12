# ==============================================================================
# define_H2_parameters.jl — Electrolytic H₂ producer parameters
# ==============================================================================
#
# PURPOSE:
#   Fills mod.ext[:parameters] for the hydrogen-sector agent (electrolyzer).
#   Reads capacity (electrolyzer and H₂ output), specific consumption (MWh per
#   unit H₂), and operational cost. Computes efficiency η_elec_H2 = 1 /
#   SpecificConsumption (units of H₂ per MWh of electricity) used in the build
#   step to link electricity consumption to H₂ production.
#
# ARGUMENTS:
#   m, mod, data, ts, repr_days — Same as elsewhere; ts/repr_days not used for
#     this agent but kept for a uniform interface.
#
# ==============================================================================

function define_H2_parameters!(m::String, mod::Model, data::Dict, ts::Dict, repr_days::Dict)
    params = mod.ext[:parameters]

    params[:Type] = String(get(data, "Type", ""))

    # Maximum electricity input to the electrolyzer (MW_elec).
    # This is the electrical equipment rating; the optimizer's electricity
    # consumption variable is bounded by this value.
    params[:Capacity_Electrolyzer] = data["Capacity_Electrolyzer"]

    # Maximum hydrogen output (MW_H2). May differ from Capacity_Electrolyzer / SpecificConsumption
    # if downstream piping or compression is the bottleneck (models a separate physical limit).
    params[:Capacity_H2_Output]     = data["Capacity_H2_Output"]

    # Specific electricity consumption: MWh of electricity per MWh of H2 produced.
    # E.g. 1.5 means the electrolyzer consumes 1.5 MWh_elec to produce 1 MWh_H2
    # (≈ 67% efficiency), a typical value for PEM electrolysis.
    params[:SpecificConsumption] = data["SpecificConsumption"]

    # Variable operational cost (€/MWh_H2): water treatment, stack degradation,
    # auxiliaries. Defaults to 0 if not specified.
    params[:OperationalCost]      = get(data, "OperationalCost", 0.0)

    # Conversion efficiency η_elec_H2 = 1 / SpecificConsumption (MWh_H2 per MWh_elec).
    # Used in build_H2_agent! as: h2_out = η_elec_H2 * e_in.
    # Computed here (rather than in build_H2_agent!) to keep parameter definitions
    # separate from model-building logic, making it easier to audit and override.
    params[:η_elec_H2] = 1.0 / params[:SpecificConsumption]

    return mod
end
