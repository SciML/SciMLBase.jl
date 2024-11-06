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
    t.x[i[1]][i[2]]
end

function Base.setindex!(t::TupleOfArraysWrapper, val, i::Tuple{Int, Int})
    t.x[i[1]][i[2]] = val
end

function as_diffeq_array(vt::Vector{VectorTemplate}, t)
    return DiffEqArray(typeof(TupleOfArraysWrapper(vt))[], t, (1, 1))
end

function is_empty_indp(indp)
    isempty(variable_symbols(indp)) && isempty(parameter_symbols(indp)) &&
        isempty(independent_variable_symbols(indp))
end

# Everything from this point on is public API

"""
    $(TYPEDSIGNATURES)

A representation of the subsystem of a given system which is saved in a solution. Created
by providing an index provider and the indexes of saved variables in the system. The indexes
can also be symbolic variables. All indexes must refer to state variables, or timeseries
parameters.

The arguments to the constructor are an index provider, the parameter object and the indexes
of variables to save.

This object is stored in the solution object and used for symbolic indexing of the subsetted
solution.

In case the provided `saved_idxs` is `nothing` or `isempty`, or if the provided
`saved_idxs` includes all of the variables and timeseries parameters, returns `nothing`.
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

function SavedSubsystem(indp, pobj, saved_idxs)
    # nothing saved
    if saved_idxs === nothing || isempty(saved_idxs)
        return nothing
    end

    # this is required because problems with no system have an empty `SymbolCache`
    # as their symbolic container.
    if is_empty_indp(indp)
        return nothing
    end

    # array state symbolics must be scalarized
    saved_idxs = collect(Iterators.flatten(map(saved_idxs) do sym
        if symbolic_type(sym) == NotSymbolic()
            (sym,)
        elseif sym isa AbstractArray && is_variable(indp, sym)
            collect(sym)
        else
            (sym,)
        end
    end))

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
            throw(ArgumentError("Can only save variables and timeseries parameters. Got $var."))
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
            state_map, nothing, nothing, identity_partitions, nothing, nothing, nothing)
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
        timeseries_partition_templates, indexes_in_partition, ts_idx_to_count)
end

"""
    $(TYPEDSIGNATURES)

Given a `SavedSubsystem`, return the subset of state indexes of the original system that are
saved, in the order they are saved.
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

A combination of a `SavedSubsystem` and a fallback index provider. The provided fallback
is used as the `symbolic_container` for the `SavedSubsystemWithFallback`. Manually
implements `is_timeseries_parameter` and `timeseries_parameter_index` using the
`SavedSubsystem` to return the appropriate indexes for the subset of saved variables,
and `nothing`/`false` otherwise.

Also implements `create_parameter_timeseries_collection`, `get_saveable_values` and
`with_updated_parameter_timeseries_values` to appropriately handled subsetted timeseries
parameters.
"""
struct SavedSubsystemWithFallback{S <: SavedSubsystem, T}
    saved_subsystem::S
    fallback::T
end

function SymbolicIndexingInterface.symbolic_container(sswf::SavedSubsystemWithFallback)
    sswf.fallback
end

function SymbolicIndexingInterface.is_timeseries_parameter(
        sswf::SavedSubsystemWithFallback, sym)
    timeseries_parameter_index(sswf, sym) !== nothing
end

function SymbolicIndexingInterface.timeseries_parameter_index(
        sswf::SavedSubsystemWithFallback, sym)
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
        sswf::SavedSubsystemWithFallback, ps, args...)
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
            set_parameter!(ps, val[ss.timeseries_params_map[idx].parameter_idx],
                ss.timeseries_idx_to_param_idx[idx])
        end
    end

    return ps
end

"""
    $(TYPEDSIGNATURES)

Given a SciMLProblem `prob` and (possibly symbolic) `save_idxs`, return the `save_idxs`
corresponding to the state variables and a `SavedSubsystem` to pass to `build_solution`.

The second return value (corresponding to the `SavedSubsystem`) may be `nothing` in case
one is not required. `save_idxs` may be a scalar or `nothing`.
"""
function get_save_idxs_and_saved_subsystem(prob, save_idxs)
    if save_idxs === nothing
        saved_subsystem = nothing
    else
        if !(save_idxs isa AbstractArray) || symbolic_type(save_idxs) != NotSymbolic()
            _save_idxs = [save_idxs]
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
        end
    end

    return save_idxs, saved_subsystem
end
