"""
    load(PeriodsFinder, dir; populate_entries=true)

Loads a `PeriodsFinder` type from `dir`.
"""
function FileIO.load(::Type{PeriodsFinder}, dir::String; 
        populate_entries::Bool=true
    )
    pf = PeriodsFinder(
        joinpath(dir, "config_file.yaml"); 
        populate_entries=populate_entries
    )
    load(pf, dir; populate_entries=false)
    return pf
end

"""
    load(pf::PeriodsFinder, dir=get_abspath_to_result_dir(pf);         
        populate_entries=true
    )

Loads selected periods, weights and orderings to `pf` from `dir`.
"""
function FileIO.load(pf::PeriodsFinder, 
        dir::String=get_abspath_to_result_dir(pf);
        populate_entries::Bool=true
    )
    df = CSV.read(joinpath(dir, "decision_variables.csv"), DataFrame)
    pf.u = Vector(df[:, :selected_periods])
    pf.w = Vector(df[:, :weights])
    orderingFile = joinpath(
        dir, "ordering_variable.csv"
    )
    if isfile(orderingFile)
        df = CSV.read(orderingFile, DataFrame)
        pf.v = Array(df)
    end
    populate_entries && populate_entries!(pf)
    return pf
end