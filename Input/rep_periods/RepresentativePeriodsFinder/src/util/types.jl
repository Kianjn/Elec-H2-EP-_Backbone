"""
    FloatTimeArray = TimeArray{Float64,1,DateTime,Array{Float64,1}}

The concrete TimeArray type used in PeriodsFinder.
"""
FloatTimeArray = TimeArray{Float64,1,DateTime,Array{Float64,1}}

"""
    MissingFloatTimeArray = TimeArray{T,1,DateTime,A} where T <: Union{Missing, Float64}, A <: Array{Union{Missing, Float64},1}

A concrete TimeArray type.
"""
MissingFloatTimeArray = TimeArray{T,1,DateTime,A} where T <: Union{Missing, Float64} where A <: Array{Union{Missing, Float64},1}

"""
    mutable struct SingleValuedContainer{T}
        x::T

Returns a specified value `x` no matter the indexing.
"""
mutable struct SingleValuedContainer{T}

    x::T

    SingleValuedContainer{T}(x) where T <: Any = new(x)

    function SingleValuedContainer()
        SingleValuedContainer(nothing)
    end

    SingleValuedContainer(x::T) where T <: Any = SingleValuedContainer{T}(x)

    function Base.getindex(svc::SingleValuedContainer, args...)
        return svc.x
    end

    function Base.print(io::IO, svc::SingleValuedContainer)
        print(io, "An SingleValuedContainer with field $(svc.x)")
    end

    function Base.show(io::IO, svc::SingleValuedContainer)
        print(io, svc)
    end

    function Base.length(svc::SingleValuedContainer)
        return length(svc.x)
    end
end

SVC = SingleValuedContainer