#=
(Symbolic) save_idxs interface:

Allows symbolically indexing solutions where a subset of the variables are saved.

To implement this interface, the solution must store a `SavedSubsystem` if it contains a
subset of all timeseries variables. The `get_saved_subsystem` function must be implemented
to return the `SavedSubsystem` if present and `nothing` otherwise.

The solution must forward `is_timeseries_parameter`, `timeseries_parameter_index`,
`with_updated_parameter_timeseries_values` and `get_saveable_values` to
`SavedSubsystemWithFallback(sol.saved_subsystem, symbolic_container(sol))`.

Additionally, it must implement `state_values` to always return the full state vector, using
the `SciMLProblem`'s `u0` as a reference, and updating the saved values in it.

See the implementation for `ODESolution` as a reference.
=#

"""
    get_saved_subsystem(sol) -> Union{SavedSubsystem, Nothing}

Return the saved-subsystem metadata carried by a solution with symbolic or
partial `save_idxs`.

Concrete time-series solution types that store only a subset of symbolic states
or time-series parameters should provide a `saved_subsystem` field or overload
this function to return the corresponding [`SavedSubsystem`](@ref). Return
`nothing` when the solution saved the full symbolic state, when the problem has
no symbolic index provider, or when no symbolic subset metadata is required.

Solution indexing code uses this hook to decide whether symbolic queries should
be forwarded directly to `symbolic_container(sol)` or through
[`SavedSubsystemWithFallback`](@ref).
"""
get_saved_subsystem(_) = nothing

struct VectorTemplate
    type::DataType
    size::Int
end

struct TupleOfArraysWrapper{T}
    x::T
end

function TupleOfArraysWrapper(vt::Vector{VectorTemplate})
    return TupleOfArraysWrapper(Tuple(map(t -> Vector{t.type}(undef, t.size), vt)))
end

function Base.getindex(t::TupleOfArraysWrapper, i::Tuple{Int, Int})
    return t.x[i[1]][i[2]]
end

function Base.setindex!(t::TupleOfArraysWrapper, val, i::Tuple{Int, Int})
    return t.x[i[1]][i[2]] = val
end

function as_diffeq_array(vt::Vector{VectorTemplate}, t)
    return DiffEqArray(typeof(TupleOfArraysWrapper(vt))[], t, (1, 1))
end

function get_root_indp(prob::AbstractSciMLProblem)
    return get_root_indp(prob.f)
end

function get_root_indp(f::T) where {T <: AbstractSciMLFunction}
    if hasfield(T, :sys)
        return f.sys
    elseif hasfield(T, :f) && f.f isa AbstractSciMLFunction
        return get_root_indp(f.f)
    else
        return nothing
    end
end

function get_root_indp(prob::LinearProblem)
    return get_root_indp(prob.f)
end

get_root_indp(prob::AbstractJumpProblem) = get_root_indp(prob.prob)

get_root_indp(x) = x

function get_root_indp(f::SymbolicLinearInterface)
    return get_root_indp(f.sys)
end

# Everything from this point on is public API

"""
    $(TYPEDSIGNATURES)

A representation of the symbolic subsystem saved in a solution.

`SavedSubsystem(indp, pobj, saved_idxs)` records how the saved values relate to
the original symbolic system described by the index provider `indp` and
parameter object `pobj`. `saved_idxs` may contain integer state indexes,
symbolic state variables, symbolic arrays of state variables, or symbolic
time-series parameters. Every symbolic entry must resolve to either a state
variable or a time-series parameter of `indp`; other symbolic entries are
rejected. Observed variables (quantities computed from the full state via
`observed`, not stored in `u`) are **not** supported in `save_idxs` yet and
raise a dedicated `ArgumentError` pointing at workarounds and
DifferentialEquations.jl#1036.

The object is stored on solution types when `save_idxs` omits part of the
symbolic state or time-series parameter set. It lets `sol[x]`, `state_values`,
and time-series parameter queries keep using the original symbols even though
the solution arrays contain only the saved subset.

The constructor returns `nothing` when no metadata is needed, including
`saved_idxs === nothing`, unavailable symbolic metadata, or requests that save
all state variables and all time-series parameters.

Solver-author contract:

  - Pass the returned object to `build_solution(...; saved_subsystem = ss)` when
    constructing a solution from symbolic `save_idxs`.
  - Store the object on concrete time-series solution types as
    `saved_subsystem`, or overload [`get_saved_subsystem`](@ref).
  - Forward symbolic time-series parameter operations through
    [`SavedSubsystemWithFallback`](@ref) when `saved_subsystem !== nothing`.
"""
struct SavedSubsystem{V, T, M, I, P, Q, C}
    """
    `Dict` mapping indexes of saved variables in the parent system to corresponding
    indexes in the saved continuous timeseries.
    """
    state_map::V
    """
    `Dict` mapping indexes of saved timeseries parameters in the parent system to
    corresponding `ParameterTimeseriesIndex`es in the save parameter timeseries.
    """
    timeseries_params_map::T
    """
    `Dict` mapping `ParameterTimeseriesIndex`es to indexes of parameters in the
    system. (`timeseries_parameter_index => parameter_index`)
    """
    timeseries_idx_to_param_idx::M
    """
    `Set` of all timeseries_idxs that are saved as-is.
    """
    identity_partitions::I
    """
    `Dict` mapping timeseries indexes to a vector of `VectorTemplate`s to use for storing
    that subsetted timeseries partition
    """
    timeseries_partition_templates::P
    """
    `Dict` mapping timeseries indexes to a vector of `ParameterTimeseriesIndex` in that
    partition. Only for saved partitions not in `identity_partitions`.
    """
    indexes_in_partition::Q
    """
    Map of timeseries indexes to the number of saved timeseries parameters in that
    partition.
    """
    partition_count::C
end

SavedSubsystem(indp, pobj, ::Nothing) = nothing

function SavedSubsystem(indp, pobj, idx::Int)
    _indp = get_root_indp(indp)
    if _indp === EMPTY_SYMBOLCACHE || _indp === nothing
        return nothing
    end
    state_map = Dict(1 => idx)
    return SavedSubsystem(state_map, nothing, nothing, nothing, nothing, nothing, nothing)
end

function SavedSubsystem(indp, pobj, saved_idxs::Union{AbstractArray, Tuple})
    _indp = get_root_indp(indp)
    if _indp === EMPTY_SYMBOLCACHE || _indp === nothing
        return nothing
    end
    if eltype(saved_idxs) == Int
        state_map = Dict{Int, Int}(v => k for (k, v) in enumerate(saved_idxs))
        return SavedSubsystem(
            state_map, nothing, nothing, nothing, nothing, nothing, nothing
        )
    end

    # array state symbolics must be scalarized
    saved_idxs = collect(
        Iterators.flatten(
            map(saved_idxs) do sym
                if symbolic_type(sym) == NotSymbolic()
                    (sym,)
                elseif sym isa AbstractArray && is_variable(indp, sym)
                    collect(sym)
                else
                    (sym,)
                end
            end
        )
    )

    saved_state_idxs = Int[]
    ts_idx_to_type_to_param_idx = Dict()
    ts_idx_to_count = Dict()
    num_ts_params = 0
    TParammapKeys = Union{}
    TParamIdx = Union{}
    timeseries_idx_to_param_idx = Dict()
    for var in saved_idxs
        if (idx = variable_index(indp, var)) !== nothing
            push!(saved_state_idxs, idx)
        elseif (idx = timeseries_parameter_index(indp, var)) !== nothing
            TParammapKeys = Base.promote_typejoin(TParammapKeys, typeof(idx))
            # increment total number of ts params
            num_ts_params += 1
            # get dict mapping type to idxs for this timeseries_idx
            buf = get!(() -> Dict(), ts_idx_to_type_to_param_idx, idx.timeseries_idx)
            # get type of parameter
            pidx = parameter_index(indp, var)
            timeseries_idx_to_param_idx[idx] = pidx
            TParamIdx = Base.promote_typejoin(TParamIdx, typeof(pidx))
            val = parameter_values(pobj, pidx)
            T = typeof(val)
            # get vector of idxs for this type
            buf = get!(() -> [], buf, T)
            # push to it
            push!(buf, idx)
            # update count of variables in this partition
            cnt = get(ts_idx_to_count, idx.timeseries_idx, 0)
            ts_idx_to_count[idx.timeseries_idx] = cnt + 1
        else
            throw(_invalid_save_idxs_symbol_error(indp, var))
        end
    end

    # type of timeseries_idxs
    Ttsidx = Union{}
    for k in keys(ts_idx_to_type_to_param_idx)
        Ttsidx = Base.promote_typejoin(Ttsidx, typeof(k))
    end

    # timeseries_idx to timeseries_parameter_index for all params
    all_ts_params = Dict()
    num_all_ts_params = 0
    for var in parameter_symbols(indp)
        if (idx = timeseries_parameter_index(indp, var)) !== nothing
            num_all_ts_params += 1
            buf = get!(() -> [], all_ts_params, idx.timeseries_idx)
            push!(buf, idx)
        end
    end

    save_all_states = length(saved_state_idxs) == length(variable_symbols(indp))
    save_all_tsparams = num_ts_params == num_all_ts_params
    if save_all_states && save_all_tsparams
        # if we're saving everything
        return nothing
    end
    if save_all_states
        sort!(saved_state_idxs)
        state_map = saved_state_idxs
    else
        state_map = Dict(saved_state_idxs .=> collect(eachindex(saved_state_idxs)))
    end

    if save_all_tsparams
        if isempty(ts_idx_to_type_to_param_idx)
            identity_partitions = ()
        else
            identity_partitions = Set{Ttsidx}(keys(ts_idx_to_type_to_param_idx))
        end
        return SavedSubsystem(
            state_map, nothing, nothing, identity_partitions, nothing, nothing, nothing
        )
    end

    if num_ts_params == 0
        return SavedSubsystem(state_map, nothing, nothing, (), nothing, nothing, nothing)
    end

    identitypartitions = Set{Ttsidx}()
    parammap = Dict()
    timeseries_partition_templates = Dict()
    TsavedParamIdx = ParameterTimeseriesIndex{Ttsidx, NTuple{2, Int}}
    indexes_in_partition = Dict{Ttsidx, Vector{TParammapKeys}}()
    for (tsidx, type_to_idxs) in ts_idx_to_type_to_param_idx
        if ts_idx_to_count[tsidx] == length(all_ts_params[tsidx])
            push!(identitypartitions, tsidx)
            continue
        end
        templates = VectorTemplate[]
        for (type, idxs) in type_to_idxs
            template = VectorTemplate(type, length(idxs))
            push!(templates, template)
            for (i, idx) in enumerate(idxs)
                pti = ParameterTimeseriesIndex(tsidx, (length(templates), i))
                parammap[idx] = pti

                buf = get!(() -> TsavedParamIdx[], indexes_in_partition, tsidx)
                push!(buf, idx)
            end
        end
        timeseries_partition_templates[tsidx] = templates
    end
    parammap = Dict{TParammapKeys, TsavedParamIdx}(parammap)
    timeseries_partition_templates = Dict{Ttsidx, Vector{VectorTemplate}}(timeseries_partition_templates)
    ts_idx_to_count = Dict{Ttsidx, Int}(ts_idx_to_count)
    timeseries_idx_to_param_idx = Dict{TParammapKeys, TParamIdx}(timeseries_idx_to_param_idx)
    return SavedSubsystem(
        state_map, parammap, timeseries_idx_to_param_idx, identitypartitions,
        timeseries_partition_templates, indexes_in_partition, ts_idx_to_count
    )
end

"""
    $(TYPEDSIGNATURES)

Return the original-system state indexes saved by a `SavedSubsystem`.

The returned vector is ordered like the saved state portion of the solution. It
does not include saved time-series parameters; when `save_idxs` selected only
time-series parameters, this returns an empty vector. Solver setup uses this
helper to translate symbolic `save_idxs` into the integer state indexes passed
to low-level save machinery.
"""
function get_saved_state_idxs(ss::SavedSubsystem)
    idxs = Vector{valtype(ss.state_map)}(undef, length(ss.state_map))
    for (k, v) in ss.state_map
        idxs[v] = k
    end
    return idxs
end

"""
    $(TYPEDEF)

A symbolic indexing adapter for subsetted solutions.

`SavedSubsystemWithFallback(saved_subsystem, fallback)` combines the subset map
with the original symbolic index provider. The `fallback` is returned from
`symbolic_container`, while time-series parameter queries are filtered and
renumbered through `saved_subsystem`.

Use this wrapper whenever a solution saved only part of the symbolic state or
time-series parameter set. It preserves the original symbolic names while
ensuring:

  - unsaved time-series parameters are reported as unavailable in the saved
    solution,
  - fully saved time-series partitions reuse the fallback representation, and
  - partially saved partitions allocate compact saved buffers and map updates
    back to the original parameter object.

The wrapper implements the time-series parameter hooks used by the
`SymbolicIndexingInterface`: `is_timeseries_parameter`,
`timeseries_parameter_index`, `create_parameter_timeseries_collection`,
`get_saveable_values`, and `with_updated_parameter_timeseries_values`.
"""
struct SavedSubsystemWithFallback{S <: SavedSubsystem, T}
    saved_subsystem::S
    fallback::T
end

function SymbolicIndexingInterface.symbolic_container(sswf::SavedSubsystemWithFallback)
    return sswf.fallback
end

function SymbolicIndexingInterface.is_timeseries_parameter(
        sswf::SavedSubsystemWithFallback, sym
    )
    return timeseries_parameter_index(sswf, sym) !== nothing
end

function SymbolicIndexingInterface.timeseries_parameter_index(
        sswf::SavedSubsystemWithFallback, sym
    )
    ss = sswf.saved_subsystem
    ss.timeseries_params_map === nothing && return nothing
    if symbolic_type(sym) == NotSymbolic()
        sym isa ParameterTimeseriesIndex || return nothing
        sym.timeseries_idx in ss.identity_partitions && return sym
        return get(ss.timeseries_params_map, sym, nothing)
    end
    v = timeseries_parameter_index(sswf.fallback, sym)
    return timeseries_parameter_index(sswf, v)
end

function create_parameter_timeseries_collection(sswf::SavedSubsystemWithFallback, ps, tspan)
    original = create_parameter_timeseries_collection(sswf.fallback, ps, tspan)
    ss = sswf.saved_subsystem
    if original === nothing
        return nothing
    end
    newcollection = map(enumerate(original)) do (i, buffer)
        i in ss.identity_partitions && return buffer
        ss.partition_count === nothing && return buffer
        cnt = get(ss.partition_count, i, 0)
        cnt == 0 && return buffer

        return as_diffeq_array(ss.timeseries_partition_templates[i], buffer.t)
    end

    return ParameterTimeseriesCollection(newcollection, parameter_values(original))
end

function get_saveable_values(sswf::SavedSubsystemWithFallback, ps, tsidx)
    ss = sswf.saved_subsystem
    original = get_saveable_values(sswf.fallback, ps, tsidx)
    tsidx in ss.identity_partitions && return original
    ss.partition_count === nothing && return nothing
    cnt = get(ss.partition_count, tsidx, 0)
    cnt == 0 && return nothing

    toaw = TupleOfArraysWrapper(ss.timeseries_partition_templates[tsidx])
    for idx in ss.indexes_in_partition[tsidx]
        toaw[ss.timeseries_params_map[idx].parameter_idx] = original[idx.parameter_idx]
    end
    return toaw
end

function SymbolicIndexingInterface.with_updated_parameter_timeseries_values(
        sswf::SavedSubsystemWithFallback, ps, args...
    )
    ss = sswf.saved_subsystem
    for (tsidx, val) in args
        if tsidx in ss.identity_partitions
            ps = with_updated_parameter_timeseries_values(sswf.fallback, ps, tsidx => val)
            continue
        end
        ss.partition_count === nothing && continue
        cnt = get(ss.partition_count, tsidx, 0)
        cnt == 0 && continue

        # now we know val isa TupleOfArraysWrapper
        for idx in ss.indexes_in_partition[tsidx]
            set_parameter!(
                ps, val[ss.timeseries_params_map[idx].parameter_idx],
                ss.timeseries_idx_to_param_idx[idx]
            )
        end
    end

    return ps
end

"""
    $(TYPEDSIGNATURES)

Translate user-facing `save_idxs` into solver state indexes and subsystem
metadata.

Given a SciML problem `prob` and a possibly symbolic `save_idxs`, return
`(state_save_idxs, saved_subsystem)`. `state_save_idxs` is the integer-only
state selection that should be passed to solver save machinery, while
`saved_subsystem` is the [`SavedSubsystem`](@ref) to pass to `build_solution`.

The helper preserves the scalar/vector shape of the user's request where it
matters: scalar state selections remain scalar, vector state selections remain
vectors, and selections containing only time-series parameters return
`Int[]` for the state portion. Either return value may be `nothing` when no
subset handling is needed.
"""
get_save_idxs_and_saved_subsystem(prob, ::Nothing) = nothing, nothing
function get_save_idxs_and_saved_subsystem(prob, save_idxs::Vector{Int})
    return save_idxs, SavedSubsystem(prob, parameter_values(prob), save_idxs)
end
function get_save_idxs_and_saved_subsystem(prob, save_idx::Int)
    return save_idx, SavedSubsystem(prob, parameter_values(prob), save_idx)
end
function get_save_idxs_and_saved_subsystem(prob, save_idxs)
    if !(save_idxs isa AbstractArray) || symbolic_type(save_idxs) != NotSymbolic()
        _save_idxs = (save_idxs,)
    else
        _save_idxs = save_idxs
    end
    saved_subsystem = SavedSubsystem(prob, parameter_values(prob), _save_idxs)
    if saved_subsystem !== nothing
        _save_idxs = get_saved_state_idxs(saved_subsystem)
        if isempty(_save_idxs)
            # no states to save
            save_idxs = Int[]
        elseif !(save_idxs isa AbstractArray) ||
                symbolic_type(save_idxs) != NotSymbolic()
            # only a single state to save, and save it as a scalar timeseries instead of
            # single-element array
            save_idxs = only(_save_idxs)
        else
            save_idxs = _save_idxs
        end
    else
        # `SavedSubsystem` also returns `nothing` when the selection saves every
        # state variable (and every time-series parameter), in which case
        # symbolic `save_idxs` must still be translated to integer state indexes,
        # preserving the requested order, so raw symbols never leak into integer
        # indexing. Non-symbolic `save_idxs` (e.g. `Colon`) pass through untouched.
        save_idxs = translate_symbolic_save_idxs(prob, save_idxs)
    end

    return save_idxs, saved_subsystem
end

"""
    $(TYPEDSIGNATURES)

Translate symbolic `save_idxs` into integer state indexes, preserving the
requested order and scalarizing symbolic arrays. Non-symbolic `save_idxs` (an
integer, `Colon`, etc.) are returned unchanged. A single symbolic entry that
resolves to one state is returned as a scalar so it is saved as a scalar
timeseries rather than a single-element array.
"""
function translate_symbolic_save_idxs(indp, save_idxs)
    if save_idxs isa AbstractArray && symbolic_type(save_idxs) == NotSymbolic()
        translated = Int[]
        for sym in save_idxs
            _append_state_indices!(translated, indp, sym)
        end
        return translated
    elseif symbolic_type(save_idxs) == NotSymbolic()
        return save_idxs
    else
        translated = Int[]
        _append_state_indices!(translated, indp, save_idxs)
        return length(translated) == 1 ? only(translated) : translated
    end
end

function _append_state_indices!(translated, indp, sym)
    if symbolic_type(sym) == NotSymbolic()
        push!(translated, sym)
    elseif sym isa AbstractArray && is_variable(indp, sym)
        for s in collect(sym)
            push!(translated, variable_index(indp, s))
        end
    else
        idx = variable_index(indp, sym)
        if idx === nothing
            throw(_invalid_save_idxs_symbol_error(indp, sym; allow_timeseries_params = false))
        end
        push!(translated, idx)
    end
    return translated
end

"""
    _invalid_save_idxs_symbol_error(indp, var; allow_timeseries_params=true)

Build the `ArgumentError` for a symbolic entry that cannot be used in
`save_idxs`. Observed quantities get a dedicated message (DifferentialEquations.jl#1036);
other non-state / non-timeseries-parameter symbols keep a generic rejection.
"""
function _invalid_save_idxs_symbol_error(indp, var; allow_timeseries_params::Bool = true)
    if is_observed(indp, var)
        return ArgumentError(
            string(
                "Saving observed variables via `save_idxs` is not yet supported (got $var). ",
                "Observed quantities are computed from the full state and are not stored in `u`, ",
                "so the solver cannot select them with integer state indices. ",
                "Workarounds: (1) omit `save_idxs` and index the full solution with `sol[$var]`; ",
                "(2) use `DiffEqCallbacks.SavingCallback` to record observed values at save times; ",
                "(3) pass the state variables the observed quantity depends on in `save_idxs`. ",
                "See SciML/DifferentialEquations.jl#1036."
            )
        )
    elseif allow_timeseries_params
        return ArgumentError(
            "Can only save variables and timeseries parameters. Got $var."
        )
    else
        return ArgumentError("Can only save variables. Got $var.")
    end
end
