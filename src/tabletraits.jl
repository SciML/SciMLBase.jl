Tables.isrowtable(::Type{<:AbstractTimeseriesSolution}) = true
Tables.columns(x::AbstractTimeseriesSolution) = Tables.columntable(Tables.rows(x))

struct AbstractTimeseriesSolutionRows{T, U}
    names::Vector{Symbol}
    types::Vector{Type}
    lookup::Dict{Symbol, Int}
    t::T
    u::U
end

function AbstractTimeseriesSolutionRows(names, types, t, u)
    AbstractTimeseriesSolutionRows(names, types,
        Dict(nm => i for (i, nm) in enumerate(names)), t, u)
end

Base.length(x::AbstractTimeseriesSolutionRows) = length(x.u)
function Base.eltype(::Type{AbstractTimeseriesSolutionRows{T, U}}) where {T, U}
    AbstractTimeseriesSolutionRow{eltype(T), eltype(U)}
end
function Base.iterate(x::AbstractTimeseriesSolutionRows, st = 1)
    st > length(x) ? nothing :
    (AbstractTimeseriesSolutionRow(x.names, x.lookup, x.t[st], x.u[st]), st + 1)
end

function Tables.rows(sol::AbstractTimeseriesSolution)
    VT = eltype(sol.u)
    syms = variable_symbols(sol)
    if VT <: AbstractArray
        N = length(sol.u[1])
        names = [
            :timestamp,
            (isempty(syms) ? (Symbol("value", i) for i in 1:N) :
             (getname(syms[i]) for i in 1:N))...
        ]
        types = Type[eltype(sol.t), (eltype(sol.u[1]) for i in 1:N)...]
    else
        names = [:timestamp, isempty(syms) ? :value : syms[1]]
        types = Type[eltype(sol.t), VT]
    end
    return AbstractTimeseriesSolutionRows(names, types, sol.t, sol.u)
end

Tables.schema(x::AbstractTimeseriesSolutionRows) = Tables.Schema(x.names, x.types)

struct AbstractTimeseriesSolutionRow{T, U} <: Tables.AbstractRow
    names::Vector{Symbol}
    lookup::Dict{Symbol, Int}
    t::T
    u::U
end

Tables.columnnames(x::AbstractTimeseriesSolutionRow) = getfield(x, :names)
function Tables.getcolumn(x::AbstractTimeseriesSolutionRow, i::Int)
    i == 1 ? getfield(x, :t) : getfield(x, :u)[i - 1]
end
function Tables.getcolumn(x::AbstractTimeseriesSolutionRow, nm::Symbol)
    nm === :timestamp ? getfield(x, :t) : getfield(x, :u)[getfield(x, :lookup)[nm] - 1]
end

IteratorInterfaceExtensions.isiterable(sol::AbstractTimeseriesSolution) = true
function IteratorInterfaceExtensions.getiterator(sol::AbstractTimeseriesSolution)
    Tables.datavaluerows(Tables.rows(sol))
end
#TableTraits.isiterabletable(sol::AbstractTimeseriesSolution) = true
Tables.istable(::EnsembleSolution) = false
