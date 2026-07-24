"""
    PeriodsFinder(config_file::String; populate_entries::Bool=false)

Type used for finding representative periods in a time series (see `find_representative_periods`). If `populate_entries=true` then time series are loaded into `pf`.

# Entries

* `config_file`: Path of configuration file used to create the `PeriodsFinder`.
* `config`: Dictionary of input parameters.
* `time_series`: Dictionary of time series.
* `x`: Normalise time series in matrix format (rows = timesteps, columns = periods).
* `inputs`: Used internally to save variables which are computationally intensive to calculate.
* `m`: JuMP model used for representative period selection.
* `u`: Selection variable, a vector whose entries are 1 if a period is selected and 0 otherwise.
* `w`: Weighting variable, a vector whose entries greater than 0 if a period is selected and 0 otherwise.
* `v`: Ordering variable, a matrix where the sum over the columns is equal to `w` and the diagonal is equal to `u`.
"""
mutable struct PeriodsFinder
    ###########################################################################
    # Configuration
    ###########################################################################
    config_file::AbstractString
    config::Dict{Any,Any}

    ###########################################################################
    # Inputs
    ###########################################################################
    time_series::Dict{String,FloatTimeArray}
    x::Dict{String,Array{Float64,2}} # Normalised time series in matrix format
    inputs::Dict{Symbol,Any} # Any saved inputs (sets, parameters, etc)
    # TODO: Change this to Dict{Union{Symbol,String},Any} - much nicer.

    ###########################################################################
    # optimization model
    ###########################################################################
    m::JuMP.Model

    ###########################################################################
    # Results
    ###########################################################################
    u::Array{Bool,1}    # Selection variable
    w::Array{Float64,1} # Weight variable (can also be integer)
    v::Array{T,2} where T <: Union{Bool,Float64} # Ordering variable (not always defined, can be float)

    function PeriodsFinder()
        pf = new()
        pf.config = Dict{Any,Any}()
        pf.time_series = Dict{String,TimeArray}()
        return pf
    end

    function PeriodsFinder(config_file::String;     
            populate_entries::Bool=false
        )
        pf = new()
        pf.config_file = config_file
        @assert isfile(config_file)
        pf.config = YAML.load(open(config_file))
        !(haskey(pf.config, "base_dir")) && (pf.config["base_dir"] = dirname(config_file))
        pf.time_series = Dict{String,TimeArray}()

        if populate_entries == true
            pf = populate_entries!(pf)
        end

        return pf
    end

    # Pretty prints of long term planning model
    function Base.print(io::IO, pf::PeriodsFinder)
        println(io, "Representative Periods Finder")
        isdefined(pf, :config_file) && println(io, "Configuration file: \n\t$(pf.config_file)")
    end

    function Base.show(io::IO, pf::PeriodsFinder)
        print(io, pf)
    end
end