"""
    step!(integ::DEIntegrator [, dt [, stop_at_tdt]])

Perform one (successful) step on the integrator.

Alternative, if a `dt` is given, then `step!` the integrator until
there is a temporal difference `≥ dt` in `integ.t`.  When `true` is
passed to the optional third argument, the integrator advances exactly
`dt`.
"""
function step!(d::DEIntegrator)
    error("Integrator stepping is not implemented")
end

"""
    resize!(integrator::DEIntegrator,k::Int)

Resizes the DE to a size `k`. This chops off the end of the array, or adds blank values at the end, depending on whether
`k > length(integrator.u)`.
"""
function Base.resize!(i::DEIntegrator, ii::Int)
    error("resize!: method has not been implemented for the integrator")
end

"""
    deleteat!(integrator::DEIntegrator,idxs)

Shrinks the ODE by deleting the `idxs` components.
"""
function Base.deleteat!(i::DEIntegrator, ii)
    error("deleteat!: method has not been implemented for the integrator")
end

"""
    addat!(integrator::DEIntegrator,idxs,val)

Grows the ODE by adding the `idxs` components. Must be contiguous indices.
"""
function addat!(i::DEIntegrator, idxs, val = zeros(length(idxs)))
    error("addat!: method has not been implemented for the integrator")
end

"""
    get_tmp_cache(i::DEIntegrator)

Returns a tuple of internal cache vectors which are safe to use as temporary arrays. This should be used
for integrator interface and callbacks which need arrays to write into in order to be non-allocating.
The length of the tuple is dependent on the method.
"""
function get_tmp_cache(i::DEIntegrator)
    error("get_tmp_cache!: method has not been implemented for the integrator")
end
function user_cache(i::DEIntegrator)
    error("user_cache: method has not been implemented for the integrator")
end
function u_cache(i::DEIntegrator)
    error("u_cache: method has not been implemented for the integrator")
end
function du_cache(i::DEIntegrator)
    error("du_cache: method has not been implemented for the integrator")
end
ratenoise_cache(i::DEIntegrator) = ()
rand_cache(i::DEIntegrator) = ()

"""
    full_cache(i::DEIntegrator)

Returns an iterator over the cache arrays of the method. This can be used to change internal values as needed.
"""
function full_cache(i::DEIntegrator)
    error("full_cache: method has not been implemented for the integrator")
end

"""
    resize_non_user_cache!(integrator::DEIntegrator,k::Int)

Resizes the non-user facing caches to be compatible with a DE of size `k`. This includes resizing Jacobian caches.

!!! note

    In many cases, [`resize!`](@ref) simply resizes [`full_cache`](@ref) variables and then
    calls this function. This finer control is required for some `AbstractArray`
    operations.
"""
function resize_non_user_cache!(i::DEIntegrator, ii::Int)
    error("resize_non_user_cache!: method has not been implemented for the integrator")
end

"""
    deleteat_non_user_cache!(integrator::DEIntegrator,idxs)

[`deleteat!`](@ref)s the non-user facing caches at indices `idxs`. This includes resizing Jacobian caches.

!!! note

    In many cases, `deleteat!` simply `deleteat!`s [`full_cache`](@ref) variables and then
    calls this function. This finer control is required for some `AbstractArray`
    operations.
"""
function deleteat_non_user_cache!(i::DEIntegrator, idxs)
    error("deleteat_non_user_cache!: method has not been implemented for the integrator")
end

"""
    addat_non_user_cache!(i::DEIntegrator,idxs)

[`addat!`](@ref)s the non-user facing caches at indices `idxs`. This includes resizing Jacobian caches.

!!! note

    In many cases, `addat!` simply `addat!`s [`full_cache`](@ref) variables and then
    calls this function. This finer control is required for some `AbstractArray`
    operations.
"""
function addat_non_user_cache!(i::DEIntegrator, idxs)
    error("addat_non_user_cache!: method has not been implemented for the integrator")
end

"""
    terminate!(i::DEIntegrator[, retcode = :Terminated])

Terminates the integrator by emptying `tstops`. This can be used in events and callbacks to immediately
end the solution process.  Optionally, `retcode` may be specified (see: [Return Codes (RetCodes)](@ref retcodes)).
"""
function terminate!(i::DEIntegrator)
    error("terminate!: method has not been implemented for the integrator")
end

"""
    get_du(i::DEIntegrator)

Returns the derivative at `t`.
"""
function get_du(i::DEIntegrator)
    error("get_du: method has not been implemented for the integrator")
end

"""
    get_du!(out,i::DEIntegrator)

Write the current derivative at `t` into `out`.
"""
function get_du!(out, i::DEIntegrator)
    error("get_du: method has not been implemented for the integrator")
end
function get_dt(i::DEIntegrator)
    error("get_dt: method has not been implemented for the integrator")
end

"""
    get_proposed_dt(i::DEIntegrator)

Gets the proposed `dt` for the next timestep.
"""
function get_proposed_dt(i::DEIntegrator)
    error("get_proposed_dt: method has not been implemented for the integrator")
end

"""
    set_proposed_dt(i::DEIntegrator,dt)
    set_proposed_dt(i::DEIntegrator,i2::DEIntegrator)

Sets the proposed `dt` for the next timestep. If the second argument isa `DEIntegrator`, then it sets the timestepping of
the first argument to match that of the second one. Note that due to PI control and step acceleration, this is more than matching
the factors in most cases.
"""
function set_proposed_dt!(i::DEIntegrator, dt)
    error("set_proposed_dt!: method has not been implemented for the integrator")
end

"""
    savevalues!(integrator::DEIntegrator,
      force_save=false) -> Tuple{Bool, Bool}

Try to save the state and time variables at the current time point, or the
`saveat` point by using interpolation when appropriate. It returns a tuple that
is `(saved, savedexactly)`. If `savevalues!` saved value, then `saved` is true,
and if `savevalues!` saved at the current time point, then `savedexactly` is
true.

The saving priority/order is as follows:

  - `save_on`

      + `saveat`
      + `force_save`
      + `save_everystep`
"""
function savevalues!(i::DEIntegrator)
    error("savevalues!: method has not been implemented for the integrator")
end

"""
    u_modified!(i::DEIntegrator,bool)

Sets `bool` which states whether a change to `u` occurred, allowing the solver to handle the discontinuity. By default,
this is assumed to be true if a callback is used. This will result in the re-calculation of the derivative at
`t+dt`, which is not necessary if the algorithm is FSAL and `u` does not experience a discontinuous change at the
end of the interval. Thus, if `u` is unmodified in a callback, a single call to the derivative calculation can be
eliminated by `u_modified!(integrator,false)`.
"""
function u_modified!(i::DEIntegrator, bool)
    error("u_modified!: method has not been implemented for the integrator")
end

"""
    add_tstop!(i::DEIntegrator,t)

Adds a `tstop` at time `t`.
"""
function add_tstop!(i::DEIntegrator, t)
    error("add_tstop!: method has not been implemented for the integrator")
end

"""
    has_tstop(i::DEIntegrator)

Checks if integrator has any stopping times defined.
"""
function has_tstop(i::DEIntegrator)
    error("has_tstop: method has not been implemented for the integrator")
end

"""
    first_tstop(i::DEIntegrator)

Gets the first stopping time of the integrator.
"""
function first_tstop(i::DEIntegrator)
    error("first_tstop: method has not been implemented for the integrator")
end

"""
    pop_tstop!(i::DEIntegrator)

Pops the last stopping time from the integrator.
"""
function pop_tstop!(i::DEIntegrator)
    error("pop_tstop!: method has not been implemented for the integrator")
end

"""
    add_saveat!(i::DEIntegrator,t)

Adds a `saveat` time point at `t`.
"""
function add_saveat!(i::DEIntegrator, t)
    error("add_saveat!: method has not been implemented for the integrator")
end

function set_abstol!(i::DEIntegrator, t)
    error("set_abstol!: method has not been implemented for the integrator")
end
function set_reltol!(i::DEIntegrator, t)
    error("set_reltol!: method has not been implemented for the integrator")
end

"""
    reinit!(integrator::DEIntegrator,args...; kwargs...)

The reinit function lets you restart the integration at a new value.

# Arguments

  - `u0`: Value of `u` to start at. Default value is `integrator.sol.prob.u0`

# Keyword Arguments

  - `t0`: Starting timepoint. Default value is `integrator.sol.prob.tspan[1]`
  - `tf`: Ending timepoint. Default value is `integrator.sol.prob.tspan[2]`
  - `erase_sol=true`: Whether to start with no other values in the solution, or keep the previous solution.
  - `tstops`, `d_discontinuities`, & `saveat`: Cache where these are stored. Default is the original cache.
  - `reset_dt`: Set whether to reset the current value of `dt` using the automatic `dt` determination algorithm. Default is
    `(integrator.dtcache == zero(integrator.dt)) && integrator.opts.adaptive`
  - `reinit_callbacks`: Set whether to run the callback initializations again (and `initialize_save` is for that). Default is `true`.
  - `reinit_cache`: Set whether to re-run the cache initialization function (i.e. resetting FSAL, not allocating vectors)
    which should usually be true for correctness. Default is `true`.

Additionally, once can access [`auto_dt_reset!`](@ref) which will run the auto `dt` initialization algorithm.
"""
function reinit!(integrator::DEIntegrator, args...; kwargs...)
    error("reinit!: method has not been implemented for the integrator")
end

"""
    initialize_dae!(integrator::DEIntegrator,initializealg = integrator.initializealg)

Runs the DAE initialization to find a consistent state vector. The optional
argument `initializealg` can be used to specify a different initialization
algorithm to use.
"""
function initialize_dae!(integrator::DEIntegrator)
    error("initialize_dae!: method has not been implemented for the integrator")
end
function initialize_dae!(integrator::DEIntegrator, initializealg)
    if !(initializealg isa NoInit)
        error("initialize_dae!: $(typeof(initializealg)) method has not been implemented for the integrator")
    end
end

"""
    auto_dt_reset!(integrator::DEIntegrator)

Run the auto `dt` initialization algorithm.
"""
function auto_dt_reset!(integrator::DEIntegrator)
    error("auto_dt_reset!: method has not been implemented for the integrator")
end

"""
    change_t_via_interpolation!(integrator::DEIntegrator,t,modify_save_endpoint=Val{false})

Modifies the current `t` and changes all of the corresponding values using the local interpolation. If the current solution
has already been saved, one can provide the optional value `modify_save_endpoint` to also modify the endpoint of `sol` in the
same manner.
"""
function change_t_via_interpolation!(i::DEIntegrator, args...)
    error("change_t_via_interpolation!: method has not been implemented for the integrator")
end

addsteps!(i::DEIntegrator, args...) = nothing

"""
    reeval_internals_due_to_modification!(integrator::DEIntegrator, continuous_modification::Bool=true;
                                          callback_initializealg = nothing)

Update DE integrator after changes by callbacks.
For DAEs (either implicit or semi-explicit), this requires re-solving alebraic variables.
If continuous_modification is true (or unspecified), this should also recalculate interpolation data.
Otherwise the integrator is allowed to skip recalculating the interpolation.

# Arguments

  - `continuous_modification`: determines whether the modification is due to a continuous change (continuous callback)
    or a discrete callback. For a continuous change, this can include a change to time which requires a re-evaluation
    of the interpolations.
  - `callback_initializealg`: the initialization algorithm provided by the callback. For DAEs, this is the choice for the
    initialization that is done post callback. The default value of `nothing` means that the initialization choice
    used for the DAE should be performed post-callback.
"""
function reeval_internals_due_to_modification!(
        integrator::DEIntegrator, continuous_modification;
        callback_initializealg = nothing)
    reeval_internals_due_to_modification!(integrator::DEIntegrator)
end
function reeval_internals_due_to_modification!(
        integrator::DEIntegrator; callback_initializealg = nothing)
    nothing
end

"""
    set_t!(integrator::DEIntegrator, t)

Set current time point of the `integrator` to `t`.
"""
function set_t!(integrator::DEIntegrator, t)
    error("set_t!: method has not been implemented for the integrator")
end

"""
    set_u!(integrator::DEIntegrator, u)
    set_u!(integrator::DEIntegrator, sym, val)

Set current state of the `integrator` to `u`. Alternatively, set the state of variable
`sym` to value `val`.
"""
function set_u! end

function set_u!(integrator::DEIntegrator, u)
    error("set_u!: method has not been implemented for the integrator")
end

function set_u!(integrator::DEIntegrator, sym, val)
    # So any error checking happens to ensure we actually _can_ set state
    set_u!(integrator, integrator.u)

    if symbolic_type(sym) == NotSymbolic()
        error("sym must be a symbol")
    end
    i = variable_index(integrator, sym)

    if isnothing(i)
        error("$sym is not a state variable")
    end

    integrator.u[i] = val
    u_modified!(integrator, true)
end

"""
    set_ut!(integrator::DEIntegrator, u, t)

Set current state of the `integrator` to `u` and `t`
"""
function set_ut!(integrator::DEIntegrator, u, t)
    set_u!(integrator, u)
    set_t!(integrator, t)
end

"""
    get_sol(integrator::DEIntegrator)

Get the solution object contained in the integrator.
"""
function get_sol(integrator::DEIntegrator)
    return integrator.sol
end

### Addat isn't a real thing. Let's make it a real thing Gretchen

function addat!(a::AbstractArray, idxs, val = nothing)
    if val === nothing
        resize!(a, length(a) + length(idxs))
    else
        error("real addat! on arrays isn't supported yet")
        #=
        flip_range = last(idxs):-1:idxs.start
        @show idxs,flip_range
        splice!(a,flip_range,val)
        =#
    end
end

### Indexing
function getsyms(integrator::DEIntegrator)
    syms = variable_symbols(integrator)
    if isempty(syms)
        syms = keys(integrator.u)
    end
    return syms
end

function getindepsym(integrator::DEIntegrator)
    syms = independent_variable_symbols(integrator)
    if isempty(syms)
        return nothing
    end
    return syms
end

function getparamsyms(integrator::DEIntegrator)
    psyms = parameter_symbols(integrator)
    if isempty(psyms)
        return nothing
    end
    return psyms
end

function getobserved(integrator::DEIntegrator)
    if has_observed(integrator.f)
        return integrator.f.observed
    else
        return DEFAULT_OBSERVED
    end
end

function sym_to_index(sym, integrator::DEIntegrator)
    idx = variable_index(integrator, sym)
    if idx === nothing
        idx = findfirst(isequal(sym), keys(integrator.u))
    end
    return idx
end

# SymbolicIndexingInterface
SymbolicIndexingInterface.symbolic_container(A::DEIntegrator) = A.f
SymbolicIndexingInterface.parameter_values(A::DEIntegrator) = A.p
SymbolicIndexingInterface.state_values(A::DEIntegrator) = A.u
SymbolicIndexingInterface.current_time(A::DEIntegrator) = A.t
function SymbolicIndexingInterface.set_state!(A::DEIntegrator, val, idx)
    A.u[idx] = val
    u_modified!(A, true)
end

SymbolicIndexingInterface.is_time_dependent(::DEIntegrator) = true

# TODO make this nontrivial once dynamic state selection works
SymbolicIndexingInterface.constant_structure(::DEIntegrator) = true

function Base.getproperty(A::DEIntegrator, sym::Symbol)
    if sym === :destats && hasfield(typeof(A), :stats)
        @warn "destats has been deprecated for stats"
        getfield(A, :stats)
    elseif sym === :ps
        return ParameterIndexingProxy(A)
    else
        return getfield(A, sym)
    end
end

Base.@propagate_inbounds function Base.getindex(A::DEIntegrator, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym]` for parameter indexing.")
    end
    return getu(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym]` for parameter indexing.")
    end
    return getu(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, ::SymbolicIndexingInterface.SolvedVariables)
    return getindex(A, variable_symbols(A))
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, ::SymbolicIndexingInterface.AllVariables)
    return getindex(A, all_variable_symbols(A))
end

function observed(A::DEIntegrator, sym)
    getobserved(A)(sym, A.u, A.p, A.t)
end

function Base.setindex!(A::DEIntegrator, val, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym] = $val` for parameter indexing.")
    end
    setu(A, sym)(A, val)
end

function Base.setindex!(A::DEIntegrator, val, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
       is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym] = $val` for parameter indexing.")
    end
    setu(A, sym)(A, val)
end

### Integrator traits

has_reinit(i::DEIntegrator) = false

### Display

function Base.summary(io::IO, I::DEIntegrator)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(I)),
        no_color, " with uType ",
        type_color, typeof(I.u),
        no_color, " and tType ",
        type_color, typeof(I.t),
        no_color)
end
function Base.show(io::IO, A::DEIntegrator)
    println(io, string("t: ", A.t))
    print(io, "u: ")
    show(io, A.u)
end
function Base.show(io::IO, m::MIME"text/plain", A::DEIntegrator)
    println(io, string("t: ", A.t))
    print(io, "u: ")
    show(io, m, A.u)
end

### Error check (retcode)

last_step_failed(integrator::DEIntegrator) = false

"""
    check_error(integrator)

Check state of `integrator` and return one of the
[Return Codes](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#retcodes)
"""
function check_error(integrator::DEIntegrator)
    if integrator.sol.retcode ∉ (ReturnCode.Success, ReturnCode.Default)
        return integrator.sol.retcode
    end
    opts = integrator.opts
    verbose = opts.verbose
    # This implementation is intended to be used for ODEIntegrator and
    # SDEIntegrator.
    if isnan(integrator.dt)
        if verbose
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        end
        return ReturnCode.DtNaN
    end
    if integrator.iter > opts.maxiters
        if verbose
            @warn("Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).")
        end
        return ReturnCode.MaxIters
    end

    # The last part:
    # Bail out if we take a step with dt less than the minimum value (which may be time dependent)
    # except if we are successfully taking such a small timestep is to hit a tstop exactly
    # We also exit if the ODE is unstable according to a user chosen callback
    # but only if we accepted the step to prevent from bailing out as unstable
    # when we just took way too big a step)
    step_accepted = !hasproperty(integrator, :accept_step) || integrator.accept_step
    if !opts.force_dtmin && opts.adaptive
        if abs(integrator.dt) <= abs(opts.dtmin) &&
           (!step_accepted || (hasproperty(opts, :tstops) ?
             integrator.t + integrator.dt < integrator.tdir * first(opts.tstops) :
             true))
            if verbose
                if isdefined(integrator, :EEst)
                    EEst = ", and step error estimate = $(integrator.EEst)"
                else
                    EEst = ""
                end
                @warn("dt($(integrator.dt)) <= dtmin($(opts.dtmin)) at t=$(integrator.t)$EEst. Aborting. There is either an error in your model specification or the true solution is unstable.")
            end
            return ReturnCode.DtLessThanMin
        elseif !step_accepted && integrator.t isa AbstractFloat &&
               abs(integrator.dt) <= abs(eps(integrator.t))
            if verbose
                if isdefined(integrator, :EEst)
                    EEst = ", and step error estimate = $(integrator.EEst)"
                else
                    EEst = ""
                end
                @warn("At t=$(integrator.t), dt was forced below floating point epsilon $(integrator.dt)$EEst. Aborting. There is either an error in your model specification or the true solution is unstable (or the true solution can not be represented in the precision of $(eltype(integrator.u))).")
            end
            return ReturnCode.Unstable
        end
    end
    if step_accepted &&
       opts.unstable_check(integrator.dt, integrator.u, integrator.p, integrator.t)
        if verbose
            @warn("Instability detected. Aborting")
        end
        return ReturnCode.Unstable
    end
    if last_step_failed(integrator)
        if verbose
            @warn("Newton steps could not converge and algorithm is not adaptive. Use a lower dt.")
        end
        return ReturnCode.ConvergenceFailure
    end
    return ReturnCode.Success
end

function postamble! end

"""
    check_error!(integrator)

Same as `check_error` but also set solution's return code
(`integrator.sol.retcode`) and run `postamble!`.
"""
function check_error!(integrator::DEIntegrator)
    code = check_error(integrator)
    integrator.sol = solution_new_retcode(integrator.sol, code)
    if code != ReturnCode.Success
        postamble!(integrator)
    end
    return code
end

### Default Iterator Interface
function done(integrator::DEIntegrator)
    if !(integrator.sol.retcode in (ReturnCode.Default, ReturnCode.Success))
        return true
    elseif isempty(integrator.opts.tstops)
        postamble!(integrator)
        return true
    elseif integrator.just_hit_tstop
        integrator.just_hit_tstop = false
        if integrator.opts.stop_at_next_tstop
            postamble!(integrator)
            return true
        end
    end
    false
end
function Base.iterate(integrator::DEIntegrator, state = 0)
    done(integrator) && return nothing
    state += 1
    step!(integrator) # Iter updated in the step! header
    # Next is callbacks -> iterator  -> top
    return integrator, state
end

Base.eltype(::Type{T}) where {T <: DEIntegrator} = T
Base.IteratorSize(::Type{<:DEIntegrator}) = Base.SizeUnknown()

### Other Iterators

struct IntegratorTuples{I}
    integrator::I
end

function Base.iterate(tup::IntegratorTuples, state = 0)
    done(tup.integrator) && return nothing
    step!(tup.integrator) # Iter updated in the step! header
    state += 1
    # Next is callbacks -> iterator  -> top
    return (tup.integrator.u, tup.integrator.t), state
end

function Base.eltype(::Type{
        IntegratorTuples{I},
}) where {U, T,
        I <:
        DEIntegrator{<:Any, <:Any, U, T}}
    Tuple{U, T}
end
Base.IteratorSize(::Type{<:IntegratorTuples}) = Base.SizeUnknown()

RecursiveArrayTools.tuples(integrator::DEIntegrator) = IntegratorTuples(integrator)

"""
$(TYPEDEF)
"""
struct IntegratorIntervals{I}
    integrator::I
end

function Base.iterate(tup::IntegratorIntervals, state = 0)
    done(tup.integrator) && return nothing
    state += 1
    step!(tup.integrator) # Iter updated in the step! header
    # Next is callbacks -> iterator  -> top
    return (tup.integrator.uprev, tup.integrator.tprev, tup.integrator.u, tup.integrator.t),
    state
end

function Base.eltype(::Type{
        IntegratorIntervals{I},
}) where {U, T,
        I <:
        DEIntegrator{<:Any, <:Any, U, T
        }}
    Tuple{U, T, U, T}
end
Base.IteratorSize(::Type{<:IntegratorIntervals}) = Base.SizeUnknown()

intervals(integrator::DEIntegrator) = IntegratorIntervals(integrator)

struct TimeChoiceIterator{T, T2}
    integrator::T
    ts::T2
end

function Base.iterate(iter::TimeChoiceIterator, state = 1)
    state > length(iter.ts) && return nothing
    t = iter.ts[state]
    integrator = iter.integrator
    if isinplace(integrator.sol.prob)
        tmp = first(get_tmp_cache(integrator))
        if t == integrator.t
            tmp .= integrator.u
        elseif t < integrator.t
            integrator(tmp, t)
        else
            step!(integrator, t - integrator.t)
            integrator(tmp, t)
        end
        return (tmp, t), state + 1
    else
        if t == integrator.t
            tmp = integrator.u
        elseif t < integrator.t
            tmp = integrator(t)
        else
            step!(integrator, t - integrator.t)
            tmp = integrator(t)
        end
        return (tmp, t), state + 1
    end
end

Base.length(iter::TimeChoiceIterator) = length(iter.ts)

@recipe function f(integrator::DEIntegrator;
        denseplot = (integrator.opts.calck ||
                     integrator isa AbstractSDEIntegrator) &&
                    integrator.iter > 0,
        plotdensity = 10,
        plot_analytic = false, vars = nothing, idxs = nothing)
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true)
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    int_vars = interpret_vars(idxs, integrator.sol)

    if denseplot
        # Generate the points from the plot from dense function
        plott = collect(range(integrator.tprev, integrator.t; length = plotdensity))
        if plot_analytic
            plot_analytic_timeseries = [integrator.sol.prob.f.analytic(
                                            integrator.sol.prob.u0,
                                            integrator.sol.prob.p,
                                            t) for t in plott]
        end
    else
        plott = nothing
    end

    dims = length(int_vars[1])
    for var in int_vars
        @assert length(var) == dims
    end
    # Should check that all have the same dims!

    plot_vecs = []
    for i in 2:dims
        push!(plot_vecs, [])
    end

    labels = String[]# Array{String, 2}(1, length(int_vars)*(1+plot_analytic))
    strs = String[]
    varsyms = variable_symbols(integrator)
    @show plott

    for x in int_vars
        for j in 2:dims
            if denseplot
                if (x[j] isa Integer && x[j] == 0) ||
                   isequal(x[j], getindepsym_defaultt(integrator))
                    push!(plot_vecs[j - 1], plott)
                else
                    push!(plot_vecs[j - 1], Vector(integrator(plott; idxs = x[j])))
                end
            else # just get values
                if x[j] == 0
                    push!(plot_vecs[j - 1], integrator.t)
                elseif x[j] == 1 && !(integrator.u isa AbstractArray)
                    push!(plot_vecs[j - 1], integrator.u)
                else
                    push!(plot_vecs[j - 1], integrator.u[x[j]])
                end
            end

            if !isempty(varsyms) && x[j] isa Integer
                push!(strs, String(getname(varsyms[x[j]])))
            elseif hasname(x[j])
                push!(strs, String(getname(x[j])))
            else
                push!(strs, "u[$(x[j])]")
            end
        end
        add_labels!(labels, x, dims, integrator.sol, strs)
    end

    if plot_analytic
        for x in int_vars
            for j in 1:dims
                if denseplot
                    push!(plot_vecs[j],
                        u_n(plot_timeseries, x[j], sol, plott, plot_timeseries))
                else # Just get values
                    if x[j] == 0
                        push!(plot_vecs[j], integrator.t)
                    elseif x[j] == 1 && !(integrator.u isa AbstractArray)
                        push!(plot_vecs[j],
                            integrator.sol.prob.f(Val{:analytic}, integrator.t,
                                integrator.sol[1]))
                    else
                        push!(plot_vecs[j],
                            integrator.sol.prob.f(Val{:analytic}, integrator.t,
                                integrator.sol[1])[x[j]])
                    end
                end
            end
            add_labels!(labels, x, dims, integrator.sol, strs)
        end
    end

    xflip --> integrator.tdir < 0

    if denseplot
        seriestype --> :path
    else
        seriestype --> :scatter
    end

    # Special case labels when idxs = (:x,:y,:z) or (:x) or [:x,:y] ...
    if idxs isa Tuple && (typeof(idxs[1]) == Symbol && typeof(idxs[2]) == Symbol)
        xlabel --> idxs[1]
        ylabel --> idxs[2]
        if length(idxs) > 2
            zlabel --> idxs[3]
        end
    end
    if getindex.(int_vars, 1) == zeros(length(int_vars)) ||
       getindex.(int_vars, 2) == zeros(length(int_vars))
        xlabel --> "t"
    end

    linewidth --> 3
    #xtickfont --> font(11)
    #ytickfont --> font(11)
    #legendfont --> font(11)
    #guidefont  --> font(11)
    label --> reshape(labels, 1, length(labels))
    (plot_vecs...,)
end

function step!(integ::DEIntegrator, dt, stop_at_tdt = false)
    (dt * integ.tdir) < 0 * oneunit(dt) && error("Cannot step backward.")
    t = integ.t
    next_t = t + dt
    stop_at_tdt && add_tstop!(integ, next_t)
    while integ.t * integ.tdir < next_t * integ.tdir
        step!(integ)
        integ.sol.retcode in (ReturnCode.Default, ReturnCode.Success) || break
    end
end

has_stats(i::DEIntegrator) = false

"""
    isadaptive(i::DEIntegrator)

Checks if the integrator is adaptive
"""
function isadaptive(integrator::DEIntegrator)
    isdefined(integrator.opts, :adaptive) ? integrator.opts.adaptive : false
end

function SymbolicIndexingInterface.get_history_function(integ::AbstractDDEIntegrator)
    DDESolutionHistoryWrapper(get_sol(integ))
end
