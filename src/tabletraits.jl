Tables.isrowtable(::Type{<:AbstractTimeseriesSolution}) = true
Tables.columns(x::AbstractTimeseriesSolution) = Tables.columntable(Tables.rows(x))

struct AbstractTimeseriesSolutionRows{T, U}
    names::Vector{Symbol}
    types::Vector{Type}
    lookup::Dict{Symbol, Int}
    t::T
    u::U
end

AbstractTimeseriesSolutionRows(names, types, t, u) = AbstractTimeseriesSolutionRows(names, types, Dict(nm => i for (i, nm) in enumerate(names)), t, u)

Base.length(x::AbstractTimeseriesSolutionRows) = length(x.u)
Base.eltype(::Type{AbstractTimeseriesSolutionRows{T, U}}) where {T, U} = AbstractTimeseriesSolutionRow{eltype(T), eltype(U)}
Base.iterate(x::AbstractTimeseriesSolutionRows, st=1) = st > length(x) ? nothing : (AbstractTimeseriesSolutionRow(x.names, x.lookup, x.t[st], x.u[st]), st + 1)

function Tables.rows(sol::AbstractTimeseriesSolution)
    VT = eltype(sol.u)
    if VT <: AbstractArray
        N = length(sol.u[1])
        names = [:timestamp, (has_syms(sol.prob.f) ? (sol.prob.f.syms[i] for i = 1:N) : (Symbol("value", i) for i = 1:N))...]
        types = Type[eltype(sol.t), (eltype(sol.u[1]) for i = 1:N)...]
    else
        names = [:timestamp, has_syms(sol.prob.f) ? sol.prob.f.syms[1] : :value]
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
Tables.getcolumn(x::AbstractTimeseriesSolutionRow, i::Int) = i == 1 ? getfield(x, :t) : getfield(x, :u)[i - 1]
Tables.getcolumn(x::AbstractTimeseriesSolutionRow, nm::Symbol) = nm === :timestamp ? getfield(x, :t) : getfield(x, :u)[getfield(x, :lookup)[nm] - 1]

IteratorInterfaceExtensions.isiterable(sol::AbstractTimeseriesSolution) = true
IteratorInterfaceExtensions.getiterator(sol::AbstractTimeseriesSolution) =
    Tables.datavaluerows(Tables.rows(sol))
#TableTraits.isiterabletable(sol::AbstractTimeseriesSolution) = true
Tables.istable(::EnsembleSolution) = false
