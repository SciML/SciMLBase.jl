@generated function struct_as_namedtuple(st)
    A = (Expr(:(=), n, :(st.$n)) for n in setdiff(fieldnames(st), (:kwargs,)))
    return Expr(:tuple, A...)
end

Base.@pure function remaker_of(prob::T) where {T <: AbstractSciMLProblem}
    parameterless_type(T){isinplace(prob)}
end
Base.@pure remaker_of(alg::T) where {T} = parameterless_type(T)

# Define `remaker_of` for the types that does not (make sense to)
# implement `isinplace` trait:
for T in [
        NoiseProblem,
        SplitFunction,  # TODO: use isinplace path for type-stability
        TwoPointBVPFunction,    # EnsembleProblem,
    ]
    @eval remaker_of(::$T) = $T
end

"""
    remake(thing; <keyword arguments>)

Re-construct `thing` with new field values specified by the keyword
arguments.
"""
function remake(thing; kwargs...)
    return _remake_internal(thing; kwargs...)
end

function _remake_internal(thing; kwargs...)
    T = remaker_of(thing)
    named_thing = struct_as_namedtuple(thing)
    return if :kwargs ∈ fieldnames(typeof(thing))
        if :args ∈ fieldnames(typeof(thing))
            named_thing = Base.structdiff(named_thing, (; args = ()))
            if :args ∉ keys(kwargs)
                k = Base.structdiff(named_thing, (; args = ()))
                if :kwargs ∉ keys(kwargs)
                    T(; named_thing..., thing.kwargs..., kwargs...)
                else
                    T(; named_thing..., kwargs[:kwargs]...)
                end
            else
                kwargs2 = Base.structdiff((; kwargs...), (; args = ()))
                if :kwargs ∉ keys(kwargs)
                    T(kwargs[:args]...; named_thing..., thing.kwargs..., kwargs2...)
                else
                    T(kwargs[:args]...; named_thing..., kwargs2[:kwargs]...)
                end
            end
        else
            if :kwargs ∉ keys(kwargs)
                T(; named_thing..., thing.kwargs..., kwargs...)
            else
                T(; named_thing..., kwargs[:kwargs]...)
            end
        end
    else
        T(; named_thing..., kwargs...)
    end
end

function isrecompile(prob::ODEProblem{iip}) where {iip}
    return (prob.f isa ODEFunction) ? !isfunctionwrapper(prob.f.f) : true
end

"""
    remake(prob::AbstractSciMLProblem; u0 = missing, p = missing, interpret_symbolicmap = true, use_defaults = false)

Remake the given problem `prob`. If `u0` or `p` are given, they will be used instead
of the unknowns/parameters of the problem. Either of them can be a symbolic map if
the problem has an associated system. If `interpret_symbolicmap == false`, `p` will never
be interpreted as a symbolic map and used as-is for parameters. `use_defaults` allows
controlling whether the default values from the system will be used to calculate missing
values in the symbolic map passed to `u0` or `p`. It is only valid when either `u0` or
`p` have been explicitly provided as a symbolic map and the problem has an associated
system.
"""
function remake(
        prob::AbstractSciMLProblem; u0 = missing,
        p = missing, interpret_symbolicmap = true, use_defaults = false, kwargs...
    )
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    return _remake_internal(prob; kwargs..., u0, p)
end

function remake(
        prob::AbstractIntervalNonlinearProblem; p = missing,
        interpret_symbolicmap = true, use_defaults = false, kwargs...
    )
    _, p = updated_u0_p(prob, [], p; interpret_symbolicmap, use_defaults)
    return _remake_internal(prob; kwargs..., p)
end

function remake(prob::AbstractNoiseProblem; kwargs...)
    return _remake_internal(prob; kwargs...)
end

function remake(
        prob::AbstractIntegralProblem; p = missing, interpret_symbolicmap = true, use_defaults = false, kwargs...
    )
    _, p = updated_u0_p(prob, nothing, p; interpret_symbolicmap, use_defaults)
    return _remake_internal(prob; kwargs..., p)
end

"""
    $(TYPEDSIGNATURES)

Check if the type `T` of an `AbstractSciMLFunction` has any type-erased (abstract) type
parameters beyond `iip` and `specialize`. Returns `true` if any field type parameter (index
3 and beyond) is not a concrete type (`isconcretetype` returns false), indicating that type
erasure was applied (e.g. by `promote_f` for AutoSpecialize compilation caching).
"""
@generated function _has_type_erased_params(::Type{T}) where {T <: AbstractSciMLFunction}
    params = T.parameters
    for i in 3:length(params)
        p = params[i]
        if !isconcretetype(p)
            return true
        end
    end
    return false
end

"""
    $(TYPEDSIGNATURES)

Reconstruct `source` preserving the non-concrete (erased) type parameters from
`TargetType` while using `source`'s actual concrete types for all other parameters.

Used to preserve type erasure from `promote_f`/`unwrapped_f` for AutoSpecialize:
the keyword constructor in `remake` narrows abstract type parameters back to concrete
types, and this function restores the erased type parameters (e.g. `Union{Nothing,
OverrideInitData}` for initialization_data) while allowing concrete field types
(like the function `f`) to change freely.
"""
@generated function _reconstruct_as_type(
        ::Type{TargetType}, source::SourceType
) where {TargetType <: AbstractSciMLFunction, SourceType <: AbstractSciMLFunction}
    target_params = collect(TargetType.parameters)
    source_params = collect(SourceType.parameters)

    # For erased (non-concrete) type parameters at index >= 3, keep the target's
    # abstract type. For all other parameters, use the source's actual type.
    mixed_params = similar(target_params, Any)
    for i in eachindex(target_params)
        if i >= 3 && !isconcretetype(target_params[i])
            mixed_params[i] = target_params[i]
        else
            mixed_params[i] = source_params[i]
        end
    end

    MixedType = TargetType.name.wrapper{mixed_params...}
    nf = fieldcount(MixedType)
    field_exprs = [:(getfield(source, $i)) for i in 1:nf]
    return :($(MixedType)($(field_exprs...)))
end

"""
    $(TYPEDSIGNATURES)

Widen all bounded type parameters of an `AbstractSciMLFunction` to their upper bounds.

For example, an `ODEFunction` has `ID <: Union{Nothing, OverrideInitData}` and
`NLP <: Union{Nothing, ODENLStepData}`. This function replaces the concrete types of
those parameters with `Union{Nothing, OverrideInitData}` and `Union{Nothing, ODENLStepData}`
respectively, while leaving all unbounded (`<: Any`) type parameters concrete.

This ensures that all AutoSpecialize instances of a function type share the same type
regardless of model-specific details (e.g. initialization functions), preventing
recompilation of `promote_f` and solver code for each model.
"""
@generated function widen_bounded_type_params(f::F) where {F <: AbstractSciMLFunction}
    # Walk the UnionAll chain to collect TypeVars and their upper bounds
    wrapper = F.name.wrapper
    typevars = TypeVar[]
    body = wrapper
    while body isa UnionAll
        push!(typevars, body.var)
        body = body.body
    end

    params = collect(F.parameters)
    new_params = similar(params, Any)
    for i in eachindex(params)
        if i <= length(typevars) && typevars[i].ub !== Any
            new_params[i] = typevars[i].ub
        else
            new_params[i] = params[i]
        end
    end

    NewType = wrapper{new_params...}
    nf = fieldcount(NewType)
    field_exprs = [:(getfield(f, $i)) for i in 1:nf]
    return :($(NewType)($(field_exprs...)))
end

"""
    $(TYPEDSIGNATURES)

A utility function which merges two `NamedTuple`s `a` and `b`, assuming that the
keys of `a` are a subset of those of `b`. Values in `b` take priority over those
in `a`, except if they are `nothing`. Keys not present in `a` are assumed to have
a value of `nothing`.
"""
function _similar_namedtuple_merge_ignore_nothing(a::NamedTuple, b::NamedTuple)
    ks = fieldnames(typeof(b))
    return NamedTuple{ks}(
        ntuple(Val(length(ks))) do i
            something(get(b, ks[i], nothing), get(a, ks[i], nothing), Some(nothing))
        end
    )
end

"""
    remake(func::AbstractSciMLFunction; f = missing, g = missing, f2 = missing, kwargs...)

`remake` the given `func`. Return an `AbstractSciMLFunction` of the same kind, `isinplace` and
`specialization` as `func`. Retain the properties of `func`, except those that are overridden
by keyword arguments. For stochastic functions (e.g. `SDEFunction`) the `g` keyword argument
is used to override `func.g`. For split functions (e.g. `SplitFunction`) the `f2` keyword
argument is used to override `func.f2`, and `f` is used for `func.f1`. If
`f isa AbstractSciMLFunction` and `func` is not a split function, properties of `f` will
override those of `func` (but not ones provided via keyword arguments). Properties of `f` that
are `nothing` will fall back to those in `func` (unless provided via keyword arguments). If
`f` is a different type of `AbstractSciMLFunction` from `func`, the returned function will be
of the kind of `f` unless `func` is a split function. If `func` is a split function, `f` and
`f2` will be wrapped in the appropriate `AbstractSciMLFunction` type with the same `isinplace`
and `specialization` as `func`.
"""
function remake(
        func::AbstractSciMLFunction; f = missing, g = missing, f2 = missing, kwargs...
    )
    # retain iip and spec of original function
    iip = isinplace(func)
    spec = specialization(func)
    # retain properties of original function
    props = getproperties(func)
    forig = f

    if f === missing || is_split_function(func)
        # if no `f` is provided, create the same type of SciMLFunction
        T = parameterless_type(func)
        f = isdefined(func, :f) ? func.f : func.f1
    elseif f isa AbstractSciMLFunction
        iip = isinplace(f)
        spec = specialization(f)
        # if `f` is a SciMLFunction, create that type
        T = parameterless_type(f)
        # properties of `f` take priority over those in the existing `func`
        # ignore properties of `f` which are `nothing` but present in `func`
        props = _similar_namedtuple_merge_ignore_nothing(props, getproperties(f))
        f = isdefined(f, :f) ? f.f : f.f1
    else
        # if `f` is provided but not a SciMLFunction, create the same type
        T = parameterless_type(func)
    end

    # minor hack to avoid breaking MTK, since prior to ~9.57 in `remake_initialization_data`
    # it creates a `NonlinearFunction` inside a `NonlinearFunction`. Just recursively unwrap
    # in this case and forget about properties.
    while !is_split_function(T) && f isa AbstractSciMLFunction
        f = isdefined(f, :f) ? f.f : f.f1
    end

    props = @delete props.f
    props = @delete props.f1

    args = (f,)
    if is_split_function(T)
        # `f1` and `f2` are wrapped in another SciMLFunction, unless they're
        # already wrapped in the appropriate type or are an `AbstractSciMLOperator`
        if !(f isa Union{AbstractSciMLOperator, split_function_f_wrapper(T)})
            f = split_function_f_wrapper(T){iip, spec}(f)
        end
        if hasproperty(func, :f2)
            # For SplitFunction
            # we don't do the same thing as `g`, because for SDEs `g` is
            # stored in the problem as well, whereas for Split ODEs etc
            # f2 is a part of the function. Thus, if the user provides
            # a SciMLFunction for `f` which contains `f2` we use that.
            f2 = coalesce(f2, get(props, :f2, missing), func.f2)
            if !(f2 isa Union{AbstractSciMLOperator, split_function_f_wrapper(T)})
                f2 = split_function_f_wrapper(T){iip, spec}(f2)
            end

            props = @delete props.f2

            if !ismissing(forig) && hasproperty(forig, :_func_cache)
                props = @delete props._func_cache
                props = @insert props._func_cache = forig._func_cache
            end

            args = (args..., f2)
        end
    end
    if isdefined(func, :g)
        # For SDEs/SDDEs where `g` is not a keyword
        g = coalesce(g, func.g)
        props = @delete props.g
        args = (args..., g)
    end
    result = T{iip, spec}(args...; props..., kwargs...)
    # Preserve type erasure from AutoSpecialize's promote_f. The keyword constructor
    # above uses typeof(field) for each type parameter, which restores concrete types
    # and undoes the intentional type erasure. Re-apply the original abstract type
    # parameters to maintain compilation caching benefits.
    # The _has_type_erased_params check is @generated and resolves at compile time,
    # so this branch is eliminated entirely for the common non-erased case.
    #
    # Check both `func` (the original function being remade) and `forig` (the incoming
    # `f` keyword argument, if it was an AbstractSciMLFunction). When `get_concrete_problem`
    # calls `remake(prob; f=promoted_f)`, the promoted_f from `unwrapped_f` has type-erased
    # params but the original `prob.f` does not — so we must check `forig` too.
    if _has_type_erased_params(typeof(func))
        return _reconstruct_as_type(typeof(func), result)
    elseif forig isa AbstractSciMLFunction && _has_type_erased_params(typeof(forig))
        return _reconstruct_as_type(typeof(forig), result)
    end
    return result
end

"""
    remake(prob::ODEProblem; f = missing, u0 = missing, tspan = missing,
           p = missing, kwargs = missing, _kwargs...)

Remake the given `ODEProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(
        prob::ODEProblem; f = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        build_initializeprob = Val{true},
        use_defaults = false,
        lazy_initialization = nothing,
        _kwargs...
    )
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    iip = isinplace(prob)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, tspan[1], p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    if specialization(f) === FunctionWrapperSpecialize
        ptspan = promote_tspan(tspan)
        if iip
            f = remake(
                f; f = wrapfun_iip(unwrapped_f(f.f), (newu0, newu0, newp, ptspan[1]))
            )
        else
            f = remake(
                f; f = wrapfun_oop(unwrapped_f(f.f), (newu0, newu0, newp, ptspan[1]))
            )
        end
    end

    prob = if kwargs === missing
        ODEProblem{iip}(
            f, newu0, tspan, newp, prob.problem_type; prob.kwargs...,
            _kwargs...
        )
    else
        ODEProblem{iip}(f, newu0, tspan, newp, prob.problem_type; kwargs...)
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

function SciMLBase.remake(
        prob::AbstractDynamicOptProblem; f = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        wrapped_model = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        lazy_initialization = nothing,
        _kwargs...
    )

    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    f = coalesce(f, prob.f)
    wrapped_model = coalesce(wrapped_model, prob.wrapped_model)

    T = parameterless_type(typeof(prob))

    prob = if kwargs === missing
        T(f, newu0, tspan, newp, wrapped_model; prob.kwargs..., _kwargs...)
    else
        T(f, newu0, tspan, newp, wrapped_model; kwargs...)
    end

    u0, p = maybe_eager_initialize_problem(prob, nothing, lazy_initialization)

    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

"""
    remake_initializeprob(sys, scimlfn, u0, t0, p)

!!! warning
    This method is deprecated. Please see `remake_initialization_data`

Re-create the initialization problem present in the function `scimlfn`, using the
associated system `sys`, and the user-provided new values of `u0`, initial time `t0` and
`p`. By default, returns `nothing, nothing, nothing, nothing` if `scimlfn` does not have an
initialization problem, and
`scimlfn.initializeprob, scimlfn.update_initializeprob!, scimlfn.initializeprobmap, scimlfn.initializeprobpmap`
if it does.

Note that `u0` or `p` may be `missing` if the user does not provide a value for them.
"""
function remake_initializeprob(sys, scimlfn, u0, t0, p)
    if !has_initialization_data(scimlfn)
        return nothing, nothing, nothing, nothing
    end
    initdata = scimlfn.initialization_data
    return initdata.initializeprob, initdata.update_initializeprob!,
        initdata.initializeprobmap, initdata.initializeprobpmap
end

"""
    $(TYPEDSIGNATURES)

Wrapper around `remake_initialization_data` for backward compatibility when `newu0` and
`newp` were not arguments.
"""
function remake_initialization_data_compat_wrapper(sys, scimlfn, u0, t0, p, newu0, newp)
    return if hasmethod(
            remake_initialization_data,
            Tuple{typeof(sys), typeof(scimlfn), typeof(u0), typeof(t0), typeof(p)}
        )
        remake_initialization_data(sys, scimlfn, u0, t0, p)
    else
        remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)
    end
end

"""
    remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)

Re-create the initialization data present in the function `scimlfn`, using the
associated system `sys`, the user provided new values of `u0`, initial time `t0`,
user-provided `p`, new u0 vector `newu0` and new parameter object `newp`. By default,
this calls `remake_initializeprob` for backward compatibility and attempts to construct
an `OverrideInitData` from the result.

Note that `u0` or `p` may be `missing` if the user does not provide a value for them.
"""
function remake_initialization_data(sys, scimlfn, u0, t0, p, newu0, newp)
    return reconstruct_initialization_data(
        nothing, remake_initializeprob(sys, scimlfn, u0, t0, p)...
    )
end

"""
    remake(prob::BVProblem; f = missing, u0 = missing, tspan = missing,
           p = missing, kwargs = missing, problem_type = missing, _kwargs...)

Remake the given `BVProblem`.
"""
function remake(
        prob::BVProblem{uType, tType, iip, nlls}; f = missing, bc = missing,
        u0 = missing, tspan = missing, p = missing, kwargs = missing, problem_type = missing,
        interpret_symbolicmap = true, use_defaults = false, _kwargs...
    ) where {
        uType, tType, iip, nlls,
    }
    if tspan === missing
        tspan = prob.tspan
    end

    u0, p = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if problem_type === missing
        problem_type = prob.problem_type
    end

    twopoint = problem_type isa TwoPointBVProblem

    if bc === missing
        bc = prob.f.bc
    end

    if f === missing
        _f = prob.f
    elseif f isa BVPFunction
        _f = f
        bc = f.bc
    elseif specialization(prob.f) === FunctionWrapperSpecialize
        ptspan = promote_tspan(tspan)
        if iip
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(
                wrapfun_iip(
                    f,
                    (u0, u0, p, ptspan[1])
                ), bc; prob.f.bcresid_prototype
            )
        else
            _f = BVPFunction{iip, FunctionWrapperSpecialize, twopoint}(
                wrapfun_oop(
                    f,
                    (u0, p, ptspan[1])
                ), bc; prob.f.bcresid_prototype
            )
        end
    else
        _f = BVPFunction{isinplace(prob), specialization(prob.f), twopoint}(
            f, bc;
            prob.f.bcresid_prototype
        )
    end

    return if kwargs === missing
        BVProblem{iip}(
            _f, bc, u0, tspan, p; problem_type, nlls = Val(nlls), prob.kwargs...,
            _kwargs...
        )
    else
        BVProblem{iip}(_f, bc, u0, tspan, p; problem_type, nlls = Val(nlls), kwargs...)
    end
end

"""
    remake(prob::SDEProblem; f = missing, g = missing, u0 = missing, tspan = missing,
           p = missing, noise = missing, noise_rate_prototype = missing,
           seed = missing, kwargs = missing, _kwargs...)

Remake the given `SDEProblem`.
"""
function remake(
        prob::SDEProblem;
        f = missing,
        g = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        noise = missing,
        noise_rate_prototype = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        seed = missing,
        kwargs = missing,
        lazy_initialization = nothing,
        build_initializeprob = Val{true},
        _kwargs...
    )
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, tspan[1], p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    if noise === missing
        noise = prob.noise
    end

    if noise_rate_prototype === missing
        noise_rate_prototype = prob.noise_rate_prototype
    end

    if seed === missing
        seed = prob.seed
    end
    f = coalesce(f, prob.f)
    g = coalesce(g, prob.g)
    f = remake(prob.f; f, g, initialization_data)
    iip = isinplace(prob)

    prob = if kwargs === missing
        SDEProblem{iip}(
            f,
            newu0,
            tspan,
            newp;
            noise,
            noise_rate_prototype,
            seed,
            prob.kwargs...,
            _kwargs...
        )
    else
        SDEProblem{iip}(f, newu0, tspan, newp; noise, noise_rate_prototype, seed, kwargs...)
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

function remake(
        prob::DDEProblem; f = missing, h = missing, u0 = missing,
        tspan = missing, p = missing, constant_lags = missing,
        dependent_lags = missing, order_discontinuity_t0 = missing,
        neutral = missing, kwargs = missing, interpret_symbolicmap = true,
        use_defaults = false, lazy_initialization = nothing, build_initializeprob = Val{true},
        _kwargs...
    )
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, tspan[1], p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    h = coalesce(h, prob.h)
    constant_lags = coalesce(constant_lags, prob.constant_lags)
    dependent_lags = coalesce(dependent_lags, prob.dependent_lags)
    order_discontinuity_t0 = coalesce(order_discontinuity_t0, prob.order_discontinuity_t0)
    neutral = coalesce(neutral, prob.neutral)

    iip = isinplace(prob)

    prob = if kwargs === missing
        DDEProblem{iip}(
            f,
            newu0,
            h,
            tspan,
            newp;
            constant_lags,
            dependent_lags,
            order_discontinuity_t0,
            neutral,
            prob.kwargs...,
            _kwargs...
        )
    else
        DDEProblem{iip}(
            f, newu0, h, tspan, newp; constant_lags, dependent_lags,
            order_discontinuity_t0, neutral, kwargs...
        )
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

function remake(
        prob::SDDEProblem;
        f = missing,
        g = missing,
        h = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        constant_lags = missing,
        dependent_lags = missing,
        order_discontinuity_t0 = missing,
        neutral = missing,
        noise = missing,
        noise_rate_prototype = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        seed = missing,
        kwargs = missing,
        lazy_initialization = nothing,
        build_initializeprob = Val{true},
        _kwargs...
    )
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, tspan[1], p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    if noise === missing
        noise = prob.noise
    end

    if noise_rate_prototype === missing
        noise_rate_prototype = prob.noise_rate_prototype
    end

    if seed === missing
        seed = prob.seed
    end

    f = coalesce(f, prob.f)
    g = coalesce(g, prob.g)
    f = remake(prob.f; f, g, initialization_data)
    iip = isinplace(prob)

    h = coalesce(h, prob.h)
    constant_lags = coalesce(constant_lags, prob.constant_lags)
    dependent_lags = coalesce(dependent_lags, prob.dependent_lags)
    order_discontinuity_t0 = coalesce(order_discontinuity_t0, prob.order_discontinuity_t0)
    neutral = coalesce(neutral, prob.neutral)

    prob = if kwargs === missing
        SDDEProblem{iip}(
            f,
            g,
            newu0,
            h,
            tspan,
            newp;
            noise,
            noise_rate_prototype,
            seed,
            constant_lags,
            dependent_lags,
            order_discontinuity_t0,
            neutral,
            prob.kwargs...,
            _kwargs...
        )
    else
        SDDEProblem{iip}(
            f, g, newu0, tspan, newp; noise, noise_rate_prototype, seed, constant_lags,
            dependent_lags, order_discontinuity_t0, neutral, kwargs...
        )
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

"""
    remake(prob::DAEProblem; f = missing, du0 = missing, u0 = missing, tspan = missing,
           p = missing, differential_vars = missing, kwargs = missing, _kwargs...)

Remake the given `DAEProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(
        prob::DAEProblem; f = missing,
        du0 = missing,
        u0 = missing,
        tspan = missing,
        p = missing,
        differential_vars = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        lazy_initialization = nothing,
        build_initializeprob = Val{true},
        _kwargs...
    )
    if tspan === missing
        tspan = prob.tspan
    end

    newu0, newp = updated_u0_p(prob, u0, p, tspan[1]; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, tspan[1], p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, tspan[1], p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    du0 = coalesce(du0, prob.du0)
    differential_vars = coalesce(differential_vars, prob.differential_vars)

    iip = isinplace(prob)

    prob = if kwargs === missing
        DAEProblem{iip}(f, du0, newu0, tspan, newp; differential_vars, prob.kwargs..., _kwargs...)
    else
        DAEProblem{iip}(f, du0, newu0, tspan, newp; differential_vars, kwargs...)
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

"""
    remake(prob::OptimizationProblem; f = missing, u0 = missing, p = missing,
        lb = missing, ub = missing, int = missing, lcons = missing, ucons = missing,
        sense = missing, kwargs = missing, _kwargs...)

Remake the given `OptimizationProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(
        prob::OptimizationProblem;
        f = missing,
        u0 = missing,
        p = missing,
        lb = missing,
        ub = missing,
        int = missing,
        lcons = missing,
        ucons = missing,
        sense = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        _kwargs...
    )
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    if f === missing
        f = prob.f
    end
    if lb === missing
        lb = prob.lb
    end
    if ub === missing
        ub = prob.ub
    end
    if int === missing
        int = prob.int
    end
    if lcons === missing
        lcons = prob.lcons
    end
    if ucons === missing
        ucons = prob.ucons
    end
    if sense === missing
        sense = prob.sense
    end

    return if kwargs === missing
        OptimizationProblem{isinplace(prob)}(
            f = f, u0 = u0, p = p, lb = lb,
            ub = ub, int = int,
            lcons = lcons, ucons = ucons,
            sense = sense; prob.kwargs..., _kwargs...
        )
    else
        OptimizationProblem{isinplace(prob)}(
            f = f, u0 = u0, p = p, lb = lb,
            ub = ub, int = int,
            lcons = lcons, ucons = ucons,
            sense = sense; kwargs...
        )
    end
end

"""
    remake(prob::NonlinearProblem; f = missing, u0 = missing, p = missing,
        problem_type = missing, kwargs = missing, _kwargs...)

Remake the given `NonlinearProblem`.
If `u0` or `p` are given as symbolic maps `ModelingToolkit.jl` has to be loaded.
"""
function remake(
        prob::NonlinearProblem;
        f = missing,
        u0 = missing,
        p = missing,
        problem_type = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        lazy_initialization = nothing,
        build_initializeprob = Val{true},
        _kwargs...
    )
    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, nothing, p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, nothing, p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    if problem_type === missing
        problem_type = prob.problem_type
    end

    prob = if kwargs === missing
        NonlinearProblem{isinplace(prob)}(
            f = f, u0 = newu0, p = newp,
            problem_type = problem_type; prob.kwargs...,
            _kwargs...
        )
    else
        NonlinearProblem{isinplace(prob)}(
            f = f, u0 = newu0, p = newp,
            problem_type = problem_type; kwargs...
        )
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

function remake(
        prob::SteadyStateProblem;
        f = missing,
        u0 = missing,
        p = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false,
        lazy_initialization = nothing,
        build_initializeprob = Val{true},
        _kwargs...
    )
    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, Inf, p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, Inf, p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    prob = if kwargs === missing
        SteadyStateProblem{isinplace(prob)}(
            f = f, u0 = newu0, p = newp; prob.kwargs...,
            _kwargs...
        )
    else
        SteadyStateProblem{isinplace(prob)}(f = f, u0 = newu0, p = newp; kwargs...)
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

"""
    remake(prob::NonlinearLeastSquaresProblem; f = missing, u0 = missing, p = missing,
        kwargs = missing, _kwargs...)

Remake the given `NonlinearLeastSquaresProblem`.
"""
function remake(
        prob::NonlinearLeastSquaresProblem; f = missing, u0 = missing, p = missing,
        interpret_symbolicmap = true, use_defaults = false, kwargs = missing,
        lazy_initialization = nothing, build_initializeprob = Val{true}, _kwargs...
    )
    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)

    if build_initializeprob == Val{true} || build_initializeprob == true
        if f !== missing && has_initialization_data(f)
            initialization_data = remake_initialization_data(
                prob.f.sys, f, u0, nothing, p, newu0, newp
            )
        else
            initialization_data = remake_initialization_data(
                prob.f.sys, prob.f, u0, nothing, p, newu0, newp
            )
        end
    else
        initialization_data = nothing
    end

    f = coalesce(f, prob.f)
    f = remake(prob.f; f, initialization_data)

    prob = if kwargs === missing
        prob = NonlinearLeastSquaresProblem{isinplace(prob)}(;
            f, u0 = newu0, p = newp, prob.kwargs...,
            _kwargs...
        )
    else
        prob = NonlinearLeastSquaresProblem{isinplace(prob)}(;
            f, u0 = newu0, p = newp, kwargs...
        )
    end

    u0, p = maybe_eager_initialize_problem(prob, initialization_data, lazy_initialization)
    @reset prob.u0 = u0
    @reset prob.p = p

    return prob
end

function scc_update_subproblems(probs::Vector, newu0, newp, parameters_alias)
    offset = Ref(0)
    return map(probs) do subprob
        # N should be inferred if `prob` is type-stable and `subprob.u0 isa StaticArray`
        N = length(state_values(subprob))
        if ArrayInterface.ismutable(newu0)
            _u0 = newu0[(offset[] + 1):(offset[] + N)]
        else
            _u0 = StaticArraysCore.similar_type(
                newu0, StaticArraysCore.Size(N)
            )(newu0[(offset[] + 1):(offset[] + N)])
        end
        subprob = if parameters_alias === Val(true)
            remake(subprob; u0 = _u0, p = newp)
        else
            remake(subprob; u0 = _u0)
        end
        offset[] += length(state_values(subprob))
        return subprob
    end
end

@inline _scc_update_subproblems(newu0, newp, ::Val{P}, offset::Int) where {P} = ()
@inline function _scc_update_subproblems(
        newu0, newp, ::Val{parameters_alias}, offset::Int,
        subprob, probs...
    ) where {parameters_alias}
    u0 = state_values(subprob)
    if u0 !== nothing
        N = length(state_values(subprob))
        if ArrayInterface.ismutable(newu0)
            _u0 = newu0[(offset + 1):(offset + N)]
        else
            _u0 = StaticArraysCore.similar_type(
                newu0, StaticArraysCore.Size(N)
            )(newu0[(offset + 1):(offset + N)])
        end
        if parameters_alias
            subprob = remake(subprob; u0 = _u0, p = newp)
        else
            subprob = remake(subprob; u0 = _u0)
        end
        offset += N
    end

    return (
        subprob,
        _scc_update_subproblems(newu0, newp, Val{parameters_alias}(), offset, probs...)...,
    )
end

function scc_update_subproblems(probs::Tuple, newu0, newp, ::Val{P}) where {P}
    return _scc_update_subproblems(newu0, newp, Val{P}(), 0, probs...)
end

"""
    remake(prob::SCCNonlinearProblem; u0 = missing, p = missing, probs = missing,
        parameters_alias = prob.parameters_alias, sys = missing, explicitfuns! = missing)

Remake the given `SCCNonlinearProblem`. `u0` is the state vector for the entire problem,
which will be chunked appropriately and used to `remake` the individual subproblems. `p`
is the parameter object for `prob`. If `parameters_alias`, the same parameter object will be
used to `remake` the individual subproblems. Otherwise if `p !== missing`, this function will
error and require that `probs` be specified. `probs` is the collection of subproblems. Even if
`probs` is explicitly specified, the value of `u0` provided to `remake` will be used to
override the values in `probs`. `sys` is the index provider for the full system.
"""
function remake(
        prob::SCCNonlinearProblem; u0 = missing, p = missing, probs = missing,
        parameters_alias = prob.parameters_alias, f = missing, sys = missing,
        interpret_symbolicmap = true, use_defaults = false, explicitfuns! = missing
    )
    if parameters_alias isa Bool
        parameters_alias = Val(parameters_alias)
    end
    if p !== missing && parameters_alias === Val(false) && probs === missing
        throw(ArgumentError("`parameters_alias` is `false` for the given `SCCNonlinearProblem`. Please provide the subproblems using the keyword `probs` with the parameters updated appropriately in each."))
    end
    newu0, newp = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    if probs === missing
        probs = prob.probs
    end
    if explicitfuns! === missing
        explicitfuns! = prob.explictfuns!
    end
    if sys === missing
        sys = prob.f.sys
    end
    if u0 !== missing || p !== missing && parameters_alias === Val(true)
        probs = scc_update_subproblems(probs, newu0, newp, parameters_alias)
    end
    f = coalesce(f, prob.f)
    f = remake(f; sys)

    return SCCNonlinearProblem{
        typeof(probs), typeof(explicitfuns!), typeof(f), typeof(newp),
    }(
        probs, explicitfuns!, f, newp, parameters_alias
    )
end

function remake(
        prob::LinearProblem; u0 = missing, p = missing, A = missing, b = missing,
        f = missing, interpret_symbolicmap = true, use_defaults = false, kwargs = missing,
        _kwargs...
    )
    u0, p = updated_u0_p(prob, u0, p; interpret_symbolicmap, use_defaults)
    f = coalesce(f, prob.f)
    # We want to copy to avoid aliasing, but don't want to unnecessarily copy
    A = @coalesce(A, copy(prob.A))
    b = @coalesce(b, copy(prob.b))

    A, b = _get_new_A_b(f, p, A, b)

    if kwargs === missing
        return LinearProblem{isinplace(prob)}(A, b, p; u0, f, prob.kwargs..., _kwargs...)
    else
        return LinearProblem{isinplace(prob)}(A, b, p; u0, f, kwargs...)
    end
end

"""
    $(TYPEDSIGNATURES)

A helper function to call `get_new_A_b` if `f isa SymbolicLinearInterface`.
"""
_get_new_A_b(f, p, A, b; kw...) = A, b

function _get_new_A_b(f::SymbolicLinearInterface, p, A, b; kw...)
    return get_new_A_b(f.sys, f, p, A, b; kw...)
end

# public API
"""
    $(TYPEDSIGNATURES)

A function to return the updated `A` and `b` matrices for a `LinearProblem` after `remake`.
`root_indp` is the innermost index provider found by recursively, calling
`SymbolicIndexingInterface.symbolic_container`, provided for dispatch. Returns the new `A`
`b` matrices. Mutation of `A` and `b` is permitted.

All implementations must accept arbitrary keyword arguments in case they are added in the
future.
"""
get_new_A_b(root_indp, f, p, A, b; kw...) = A, b

function varmap_has_var(varmap, var)
    return haskey(varmap, var) || hasname(var) && haskey(varmap, getname(var))
end

function varmap_get(varmap, var, default = nothing)
    if haskey(varmap, var)
        return varmap[var]
    end
    if hasname(var)
        name = getname(var)
        if haskey(varmap, name)
            return varmap[name]
        end
    end
    return default
end

"""
    $(TYPEDSIGNATURES)

Check if `varmap::Dict{Any, Any}` contains cyclic values for any symbolic variables in
`syms`. Falls back on the basis of `symbolic_container(indp)`. Returns `false` by default.
"""
function detect_cycles(indp, varmap, syms)
    if hasmethod(symbolic_container, Tuple{typeof(indp)}) &&
            (sc = symbolic_container(indp)) != indp
        return detect_cycles(sc, varmap, syms)
    else
        return false
    end
end

anydict(d::Dict{Any, Any}) = d
anydict(d) = Dict{Any, Any}(d)
anydict() = Dict{Any, Any}()

function _updated_u0_p_internal(
        prob, ::Missing, ::Missing, t0; interpret_symbolicmap = true, use_defaults = false
    )
    return state_values(prob), parameter_values(prob)
end
function _updated_u0_p_internal(
        prob, ::Missing, p, t0; interpret_symbolicmap = true, use_defaults = false
    )
    u0 = state_values(prob)

    if p isa AbstractArray && isempty(p)
        return _updated_u0_p_internal(
            prob, u0, parameter_values(prob), t0; interpret_symbolicmap
        )
    end
    eltype(p) <: Pair && interpret_symbolicmap || return u0, p
    defs = default_values(prob)
    p = fill_p(prob, anydict(p); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(false), p, Val(true), t0)
end

function _updated_u0_p_internal(
        prob, u0, ::Missing, t0; interpret_symbolicmap = true, use_defaults = false
    )
    p = parameter_values(prob)

    eltype(u0) <: Pair || return u0, p
    defs = default_values(prob)
    u0 = fill_u0(prob, anydict(u0); defs, use_defaults)
    return _updated_u0_p_symmap(prob, u0, Val(true), p, Val(false), t0)
end

function _updated_u0_p_internal(
        prob, u0, p, t0; interpret_symbolicmap = true, use_defaults = false
    )
    isu0symbolic = eltype(u0) <: Pair
    ispsymbolic = eltype(p) <: Pair && interpret_symbolicmap

    if !isu0symbolic && !ispsymbolic
        return u0, p
    end
    defs = default_values(prob)
    if isu0symbolic
        u0 = fill_u0(prob, anydict(u0); defs, use_defaults)
    end
    if ispsymbolic
        p = fill_p(prob, anydict(p); defs, use_defaults)
    end
    return _updated_u0_p_symmap(prob, u0, Val(isu0symbolic), p, Val(ispsymbolic), t0)
end

function fill_u0(prob, u0; defs = nothing, use_defaults = false)
    return fill_vars(
        prob, u0; defs, use_defaults, allsyms = variable_symbols(prob),
        index_function = variable_index
    )
end

function fill_p(prob, p; defs = nothing, use_defaults = false)
    return fill_vars(
        prob, p; defs, use_defaults, allsyms = parameter_symbols(prob),
        index_function = parameter_index
    )
end

function fill_vars(
        prob, varmap; defs = nothing, use_defaults = false, allsyms, index_function
    )
    idx_to_vsym = anydict(index_function(prob, sym) => sym for sym in allsyms)
    sym_to_idx = anydict()
    idx_to_sym = anydict()
    idx_to_val = anydict()
    for (k, v) in varmap
        v === nothing && continue
        idx = index_function(prob, k)
        idx === nothing && continue
        # If `k` is an array symbolic, and `[:k => [1.0, 2.0]]` is provided
        if idx isa AbstractArray && symbolic_type(k) == ScalarSymbolic()
            k = [idx_to_vsym[i] for i in idx]
        elseif !(idx isa AbstractArray) || symbolic_type(k) != ArraySymbolic()
            idx = (idx,)
            k = (k,)
            v = (v,)
        end
        for (kk, vv, ii) in zip(k, v, idx)
            sym_to_idx[kk] = ii
            kk = idx_to_vsym[ii]
            sym_to_idx[kk] = ii
            idx_to_sym[ii] = kk
            idx_to_val[ii] = vv
        end
    end
    for sym in allsyms
        haskey(sym_to_idx, sym) && continue
        idx = index_function(prob, sym)
        haskey(idx_to_val, idx) && continue
        sym_to_idx[sym] = idx
        idx_to_sym[idx] = sym
        idx_to_val[idx] = if defs !== nothing &&
                (defval = varmap_get(defs, sym)) !== nothing &&
                (symbolic_type(defval) != NotSymbolic() || use_defaults)
            defval
        else
            getsym(prob, sym)(prob)
        end
    end
    newvals = anydict()
    for (idx, val) in idx_to_val
        newvals[idx_to_sym[idx]] = val
    end
    for (k, v) in varmap
        haskey(sym_to_idx, k) && continue
        v === nothing && continue
        newvals[k] = v
    end
    return newvals
end

struct CyclicDependencyError <: Exception
    varmap::Dict{Any, Any}
    vars::Any
end

function Base.showerror(io::IO, err::CyclicDependencyError)
    println(io, "Detected cyclic dependency in initial values:")
    for (k, v) in err.varmap
        println(io, k, " => ", v)
    end
    return println(io, "While trying to solve for variables: ", err.vars)
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{false}, t0)
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p

    if detect_cycles(prob, u0, variable_symbols(prob))
        throw(CyclicDependencyError(u0, variable_symbols(prob)))
    end
    for (k, v) in u0
        u0[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, u0)
    end

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    isdep || return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p

    # FIXME: need to provide `u` since the observed function expects it.
    # This is sort of an implicit dependency on MTK. The values of `u` won't actually be
    # used, since any state symbols in the expression were substituted out earlier.
    temp_state = ProblemState(;
        u = state_values(prob), p = p, t = t0,
        h = is_markovian(prob) ? nothing : get_history_function(prob)
    )
    for (k, v) in u0
        u0[k] = symbolic_type(v) === NotSymbolic() ? v : getsym(prob, v)(temp_state)
    end
    return remake_buffer(prob, state_values(prob), keys(u0), values(u0)), p
end

function _updated_u0_p_symmap(prob, u0, ::Val{false}, p, ::Val{true}, t0)
    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))

    if detect_cycles(prob, p, parameter_symbols(prob))
        throw(CyclicDependencyError(p, parameter_symbols(prob)))
    end
    for (k, v) in p
        p[k] = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, p)
    end

    isdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)
    isdep || return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))

    # FIXME: need to provide `p` since the observed function expects an `MTKParameters`
    # this is sort of an implicit dependency on MTK. The values of `p` won't actually be
    # used, since any parameter symbols in the expression were substituted out earlier.
    temp_state = ProblemState(;
        u = u0, p = parameter_values(prob), t = t0,
        h = is_markovian(prob) ? nothing : get_history_function(prob)
    )
    for (k, v) in p
        p[k] = symbolic_type(v) === NotSymbolic() ? v : getsym(prob, v)(temp_state)
    end
    return u0, remake_buffer(prob, parameter_values(prob), keys(p), values(p))
end

function _updated_u0_p_symmap(prob, u0, ::Val{true}, p, ::Val{true}, t0)
    isu0dep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in u0)
    ispdep = any(symbolic_type(v) !== NotSymbolic() for (_, v) in p)

    if !isu0dep && !ispdep
        return remake_buffer(prob, state_values(prob), keys(u0), values(u0)),
            remake_buffer(prob, parameter_values(prob), keys(p), values(p))
    end

    varmap = merge(u0, p)
    allsyms = [variable_symbols(prob); parameter_symbols(prob)]
    if detect_cycles(prob, varmap, allsyms)
        throw(CyclicDependencyError(varmap, allsyms))
    end
    if is_time_dependent(prob)
        varmap[only(independent_variable_symbols(prob))] = t0
    end
    for (k, v) in u0
        v = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
        # if `symbolic_evaluate` can't get us a concrete value,
        # use the old one from `prob`.
        if symbolic_type(v) != NotSymbolic()
            v = getu(prob, k)(prob)
        end
        u0[k] = v
    end
    for (k, v) in p
        v = symbolic_type(v) === NotSymbolic() ? v : symbolic_evaluate(v, varmap)
        if symbolic_type(v) != NotSymbolic()
            v = getu(prob, k)(prob)
        end
        p[k] = v
    end
    return remake_buffer(prob, state_values(prob), keys(u0), values(u0)),
        remake_buffer(prob, parameter_values(prob), keys(p), values(p))
end

function updated_u0_p(
        prob, u0, p, t0 = nothing; interpret_symbolicmap = true,
        use_defaults = false
    )
    if u0 === missing && p === missing
        return state_values(prob), parameter_values(prob)
    end
    if prob.f !== nothing && has_sys(prob.f) && prob.f.sys === nothing
        if interpret_symbolicmap && eltype(p) !== Union{} && eltype(p) <: Pair
            throw(
                ArgumentError(
                    "This problem does not support symbolic maps with " *
                        "`remake`, i.e. it does not have a symbolic origin. Please use `remake`" *
                        "with the `p` keyword argument as a vector of values (paying attention to" *
                        "parameter order) or pass `interpret_symbolicmap = false` as a keyword argument"
                )
            )
        end
        if eltype(u0) !== Union{} && eltype(u0) <: Pair
            throw(
                ArgumentError(
                    "This problem does not support symbolic maps with" *
                        " remake, i.e. it does not have a symbolic origin. Please use `remake`" *
                        "with the `u0` keyword argument as a vector of values, paying attention to the order."
                )
            )
        end
        return (u0 === missing ? state_values(prob) : u0),
            (p === missing ? parameter_values(prob) : p)
    end
    newu0,
        newp = _updated_u0_p_internal(
        prob, u0, p, t0; interpret_symbolicmap, use_defaults
    )
    return late_binding_update_u0_p(prob, u0, p, t0, newu0, newp)
end

"""
    $(TYPEDSIGNATURES)

A function to perform custom modifications to `newu0` and/or `newp` after they have been
constructed in `remake`. `root_indp` is the innermost index provider found by recursively
calling `SymbolicIndexingInterface.symbolic_container`, provided for dispatch. Returns
the updated `newu0` and `newp`.
"""
function late_binding_update_u0_p(prob, root_indp, u0, p, t0, newu0, newp)
    return newu0, newp
end

"""
    $(TYPEDSIGNATURES)

Calls `late_binding_update_u0_p(prob, root_indp, u0, p, t0, newu0, newp)` after finding
`root_indp`.
"""
function late_binding_update_u0_p(prob, u0, p, t0, newu0, newp)
    root_indp = get_root_indp(prob)
    return late_binding_update_u0_p(prob, root_indp, u0, p, t0, newu0, newp)
end

# overloaded in MTK to intercept symbolic remake
function process_p_u0_symbolic(prob, p, u0)
    return if prob isa Union{AbstractDEProblem, OptimizationProblem, NonlinearProblem}
        throw(ArgumentError("Please load `ModelingToolkit.jl` in order to support symbolic remake."))
    else
        throw(ArgumentError("Symbolic remake for $(typeof(prob)) is currently not supported, consider opening an issue."))
    end
end

function maybe_eager_initialize_problem(
        prob::AbstractSciMLProblem, initialization_data,
        lazy_initialization::Union{Nothing, Bool}
    )
    if lazy_initialization === nothing
        lazy_initialization = !is_trivial_initialization(initialization_data)
    end
    if initialization_data !== nothing && !lazy_initialization &&
            (!is_time_dependent(prob) || current_time(prob) !== nothing)
        u0, p,
            _ = get_initial_values(
            prob, prob, prob.f, OverrideInit(), Val(isinplace(prob))
        )
        if u0 !== nothing && eltype(u0) == Any && isempty(u0)
            u0 = nothing
        end
    else
        u0 = state_values(prob)
        p = parameter_values(prob)
    end
    return u0, p
end

function remake(thing::AbstractJumpProblem; kwargs...)
    return parameterless_type(thing)(remake(thing.prob; kwargs...))
end

function remake(thing::AbstractEnsembleProblem; kwargs...)
    T = parameterless_type(thing)
    en_kwargs = [k for k in kwargs if first(k) ∈ fieldnames(T)]
    return T(remake(thing.prob; setdiff(kwargs, en_kwargs)...); en_kwargs...)
end
