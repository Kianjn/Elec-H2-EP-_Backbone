function config_get(d1::AbstractDict, d2::AbstractDict, k::String, default::Any)
    if haskey(d1, k)
        return d1[k]
    elseif haskey(d2, k)
        return d2[k]
    else
        return default
    end
end

function mkrootdirs(dir::String)
    dirVec = splitpath(dir)
    dd = dirVec[1]
    for d in dirVec[2:end]
        dd = joinpath(dd, d)
        if isdir(dd) == false
            mkdir(dd)
        end
    end
end

function recursive_get(d, args...)
    if length(args) > 1
        if haskey(d, args[1])
            recursive_get(d[args[1]], args[2:end]...)
        else
            return args[end]
        end
    else
        return d # if length(args) > 1 then d is the value you want
    end
end

function recursive_set(d, args...; collection_type=typeof(d))
    if length(args) > 1
        if haskey(d, args[1]) == false
            d[args[1]] = collection_type()
        end
        recursive_set(
            d[args[1]], args[2:end]...; collection_type=collection_type
        )
    end
    return last(args)
end

function normalize_values(
    A, lb=-1.0, ub=1.0; min_val=minimum(A), max_val=maximum(A)
)
    A = (A .- min_val) ./ (max_val - min_val) .* (ub - lb) .+ lb
    return replace(A, NaN => 0.0)
end

squared(x) = x^2

var_value(x::JuMP.Containers.DenseAxisArray) = value.(x).data
var_value(x::JuMP.Containers.SparseAxisArray) = value.(x).data
var_value(x::Array{VariableRef,1}) = value.(x)
var_value(x::Array{VariableRef,2}) = value.(x)
var_value(x::T) where {T<:Any} = x # Catch all

import Base.length
Base.length(expr::JuMP.GenericAffExpr) = 1

function get_number_of_clusters_of_adjacent_values(x::AbstractVector)
    y = sort(x)
    numAdj = 1
    for j in 2:length(y)
        if y[j - 1] + 1 == y[j]
            numAdj += 0
        else
            numAdj += 1
        end
    end
    return numAdj
end

midpoint(x::AbstractVector) = [(x[i] + x[i+1])/2 for i in eachindex(x)[1:end-1]]

datadir(args...) = normpath(joinpath(@__DIR__, "..", "..", "data", args...))