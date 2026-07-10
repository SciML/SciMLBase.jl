"""
    step!(integ::DEIntegrator [, dt [, stop_at_tdt]])

Advance a differential equation integrator.

With one argument, perform one accepted solver step according to the concrete
algorithm. With `dt`, repeatedly step until the signed time displacement from
the starting time is at least `dt`. When `stop_at_tdt` is true, the generic
fallback adds a temporary `tstop` so the integrator lands exactly at `t + dt`.
Negative stepping relative to `integ.tdir` is rejected by the fallback.
"""
function step!(d::DEIntegrator)
    error("Integrator stepping is not implemented")
end

"""
    resize!(integrator::DEIntegrator,k::Int)

Resize the state dimension of an integrator to length `k`.

Concrete integrators that support dynamic state sizes should resize `u`, saved
state caches, user-facing caches, and any algorithm-specific non-user caches so
future steps see a consistent state layout. Shrinking removes trailing state
entries; growing appends solver-defined blank/default values.
"""
function Base.resize!(i::DEIntegrator, ii::Int)
    error("resize!: method has not been implemented for the integrator")
end

"""
    deleteat!(integrator::DEIntegrator,idxs)

Delete state components from a dynamic-size integrator.

Implementations should remove the selected entries from `integrator.u`, saved
state caches, and any dependent non-user caches. Symbolic indexing metadata is
assumed to remain valid only when the concrete solver documents support for
dynamic state selection.
"""
function Base.deleteat!(i::DEIntegrator, ii)
    error("deleteat!: method has not been implemented for the integrator")
end

"""
    addat!(integrator::DEIntegrator,idxs,val)

Insert state components into a dynamic-size integrator.

`idxs` must describe contiguous positions. Implementations should insert `val`
or solver-defined defaults into `integrator.u`, saved state caches, and any
dependent non-user caches so subsequent stepping uses the new state dimension.
"""
function addat!(i::DEIntegrator, idxs, val = zeros(length(idxs)))
    error("addat!: method has not been implemented for the integrator")
end

"""
    get_tmp_cache(i::DEIntegrator)

Return temporary work arrays owned by the integrator.

The returned tuple is intended for callbacks and integrator-interface code that
needs non-allocating scratch storage. Callers may mutate these arrays during the
current operation, but must not store them for later use or assume a fixed tuple
length across algorithms.
"""
function get_tmp_cache(i::DEIntegrator)
    error("get_tmp_cache!: method has not been implemented for the integrator")
end
"""
    user_cache(integrator::DEIntegrator)

Return user-accessible cache components from the integrator.

These arrays are documented by the concrete solver as safe for user or callback
mutation. They are distinct from temporary caches whose contents may be
overwritten by the next integrator operation.
"""
function user_cache(i::DEIntegrator)
    error("user_cache: method has not been implemented for the integrator")
end

"""
    u_cache(integrator::DEIntegrator)

Return state-like cache arrays used by the integrator.

Concrete solvers use these arrays for stage values, interpolation data, or other
intermediate state storage. Generic resizing and callback code may use this
interface when it needs to keep state-shaped caches consistent with `u`.
"""
function u_cache(i::DEIntegrator)
    error("u_cache: method has not been implemented for the integrator")
end

"""
    du_cache(integrator::DEIntegrator)

Return derivative-like cache arrays used by the integrator.

These arrays store intermediate derivatives, residuals, or rate values whose
shape follows the state. Concrete solvers should document whether users may
mutate them directly or should treat them as internal storage.
"""
function du_cache(i::DEIntegrator)
    error("du_cache: method has not been implemented for the integrator")
end

"""
    ratenoise_cache(integrator::DEIntegrator)

Return an iterable of state-shaped rate-noise caches owned by a stochastic
integrator.

Generic resizing operations use this collection to keep noise-rate workspaces
aligned with `integrator.u`. The returned arrays are solver-owned mutable
scratch storage; users should not retain or modify them independently of the
integrator. Deterministic integrators and stochastic methods without such caches
use the default empty tuple.
"""
ratenoise_cache(i::DEIntegrator) = ()

"""
    rand_cache(integrator::DEIntegrator)

Return an iterable of state-shaped random-increment caches owned by a stochastic
integrator.

Generic resizing operations use this collection when random workspaces follow
the state shape, notably for diagonal-noise methods. The returned arrays are
solver-owned mutable scratch storage; they are not random-number generators and
should not be retained or modified independently of the integrator. Integrators
without such caches use the default empty tuple.
"""
rand_cache(i::DEIntegrator) = ()

"""
    full_cache(i::DEIntegrator)

Return an iterator over all state-sized cache arrays managed by the method.

`full_cache` is the broad cache interface used by generic resizing, adaptation,
and callback utilities that need to keep every state-shaped cache synchronized.
Concrete solvers should include user and non-user caches whose leading state
dimension must track `integrator.u`.
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

Return the derivative represented by the integrator at its current `(u, p, t)`.

An implementation may return an internal derivative cache or evaluate the
problem function when no valid cache exists. Treat the returned value as
read-only because mutating an aliased cache can corrupt later steps. Use
[`get_du!`](@ref) when caller-owned output storage is required.

This operation is optional when a derivative is not meaningful or available.
For example, discrete steppers have no continuous derivative, and some DAE
integrators cannot provide one before their first initialized step. Direct
changes to `u`, `p`, or `t` must be reported through the integrator mutation
interface so a cached derivative is refreshed before it is queried.
"""
function get_du(i::DEIntegrator)
    error("get_du: method has not been implemented for the integrator")
end

"""
    get_du!(out, i::DEIntegrator)

Write the derivative represented by the integrator at its current `(u, p, t)`
into caller-owned `out`.

`out` must have a shape and element type compatible with the derivative. An
implementation may copy a valid internal cache or evaluate the problem function
directly. Use the contents of `out` after the call; concrete methods are not
required to return `out`. The same derivative-availability restrictions as
[`get_du`](@ref) apply.
"""
function get_du!(out, i::DEIntegrator)
    error("get_du: method has not been implemented for the integrator")
end

"""
    get_dt(i::DEIntegrator)

Return the integrator's active step-size increment.

This is the signed increment associated with the current or most recently
attempted step, according to the concrete solver. It can differ from
[`get_proposed_dt`](@ref), which reports the controller's proposal for the next
step. Concrete integrators that do not expose an active step size may leave this
optional hook unimplemented.
"""
function get_dt(i::DEIntegrator)
    error("get_dt: method has not been implemented for the integrator")
end

"""
    get_proposed_dt(i::DEIntegrator)

Return the signed step-size increment currently proposed for the next step.

For adaptive methods this is the controller proposal. For fixed-step methods it
is normally the configured step size. The actual next step may be shortened to
land on a `tstop`, rejected and retried, or otherwise adjusted by the solver, so
this value is a proposal rather than a promise about the next accepted time.
"""
function get_proposed_dt(i::DEIntegrator)
    error("get_proposed_dt: method has not been implemented for the integrator")
end

"""
    set_proposed_dt!(i::DEIntegrator, dt)
    set_proposed_dt!(i::DEIntegrator, i2::DEIntegrator)

Set the signed step-size proposal used for the next step.

The scalar form updates every step-size field that the concrete solver requires
to honor a new proposal. It does not bypass error control, rejection, or
`tstop` handling, and therefore does not guarantee that the next accepted step
has exactly that size.

The two-integrator form synchronizes the first integrator's time-stepping state
with the second. Adaptive implementations should copy the controller history or
other state needed to reproduce the proposal, rather than only copying one `dt`
field. This form is optional for integrators that cannot share compatible
controller state.
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
    derivative_discontinuity!(i::DEIntegrator, bool)

Record whether a callback or direct integrator mutation introduced a derivative
discontinuity.

The flag describes whether `f(u, p, t)` may have changed discontinuously because
`u`, `p`, `t`, or the definition of `f` changed. Solvers use this to decide
whether to recompute derivatives, interpolation data, FSAL caches, or Jacobians
before the next step. Callback code should leave the default discontinuity
behavior in place after state-changing effects, and may call
`derivative_discontinuity!(integrator, false)` only when it did not change the
state, parameters, time, or dynamics.
"""
function derivative_discontinuity!(i::DEIntegrator, bool)
    error("derivative_discontinuity!: method has not been implemented for the integrator")
end

"""
    u_modified!(i::DEIntegrator, bool)

Deprecated alias for `derivative_discontinuity!`. Use
`derivative_discontinuity!(i, bool)` to tell the integrator whether the right-hand
side data changed discontinuously.
"""
function u_modified!(i::DEIntegrator, bool)
    Base.depwarn(
        "`u_modified!(i::DEIntegrator, bool)` is deprecated, use " *
            "`derivative_discontinuity!(i, bool)` instead.",
        :u_modified!
    )
    return derivative_discontinuity!(i, bool)
end

"""
    add_tstop!(i::DEIntegrator, t)

Schedule a future stopping time at the physical time `t`.

An integrator must not accept a stop behind its current time in the direction of
integration. A `tstop` constrains stepping so the integrator reaches `t`
exactly when the method supports step-size changes or interpolation. It does not
by itself request that the solution be saved there; use [`add_saveat!`](@ref) or
the solver's saving options for output.

Implementations commonly store `tstops` as direction-normalized priority keys
`integrator.tdir * t`. The companion queue accessors expose those keys so generic
stepping code can compare them with `integrator.tdir * integrator.t` in both
forward and reverse integration.
"""
function add_tstop!(i::DEIntegrator, t)
    error("add_tstop!: method has not been implemented for the integrator")
end

"""
    has_tstop(i::DEIntegrator)

Return whether the integrator has any pending stopping times.

This query must be consistent with [`first_tstop`](@ref) and
[`pop_tstop!`](@ref): when it returns `false`, neither queue accessor may be
called until another stop is added.
"""
function has_tstop(i::DEIntegrator)
    error("has_tstop: method has not been implemented for the integrator")
end

"""
    first_tstop(i::DEIntegrator)

Return the next pending stopping-time key without removing it.

Stopping times are ordered in the direction of integration. The returned value
is direction-normalized as `integrator.tdir * tstop`, matching the queue key used
by generic solver and callback code. Recover the physical time as
`integrator.tdir * first_tstop(integrator)` when `integrator.tdir` is `1` or
`-1`. Calling this on an empty queue is invalid; check [`has_tstop`](@ref)
first.
"""
function first_tstop(i::DEIntegrator)
    error("first_tstop: method has not been implemented for the integrator")
end

"""
    pop_tstop!(i::DEIntegrator)

Remove and return the next pending stopping-time key.

The value and ordering follow [`first_tstop`](@ref): this removes the earliest
stop in the direction of integration, not the most recently inserted stop, and
returns its direction-normalized queue key. Calling this on an empty queue is
invalid; check [`has_tstop`](@ref) first.
"""
function pop_tstop!(i::DEIntegrator)
    error("pop_tstop!: method has not been implemented for the integrator")
end

"""
    add_saveat!(i::DEIntegrator, t)

Schedule solution output at the future physical time `t`.

An integrator must not accept a save point behind its current time in the
direction of integration. `saveat` normally uses interpolation when `t` lies
inside a step and therefore does not force the integrator to step exactly to
`t`. Add a matching [`add_tstop!`](@ref) when an exact step endpoint is also
required. Saving still follows the solver's `save_on`, `save_idxs`, and related
output options.
"""
function add_saveat!(i::DEIntegrator, t)
    error("add_saveat!: method has not been implemented for the integrator")
end

"""
    set_abstol!(i::DEIntegrator, abstol)

Update the absolute error tolerance used by subsequent adaptive steps.

Concrete implementations must refresh any controller or scaling state derived
from the old tolerance. The accepted scalar or array tolerance shapes follow the
solver's `abstol` option. Integrators that do not support changing tolerances at
runtime may leave this optional hook unimplemented.
"""
function set_abstol!(i::DEIntegrator, t)
    error("set_abstol!: method has not been implemented for the integrator")
end

"""
    set_reltol!(i::DEIntegrator, reltol)

Update the relative error tolerance used by subsequent adaptive steps.

Concrete implementations must refresh any controller or scaling state derived
from the old tolerance. The accepted scalar or array tolerance shapes follow the
solver's `reltol` option. Integrators that do not support changing tolerances at
runtime may leave this optional hook unimplemented.
"""
function set_reltol!(i::DEIntegrator, t)
    error("set_reltol!: method has not been implemented for the integrator")
end

"""
    has_rng(integrator::DEIntegrator) -> Bool

Return whether `integrator` supports the live RNG interface formed by
[`get_rng`](@ref) and [`set_rng!`](@ref).

An integrator that returns `true` must carry a valid `AbstractRNG` for its whole
lifetime, using `Random.default_rng()` when the solver supports the interface but
the caller supplied no RNG. Generic code must query this trait before accessing
or replacing the RNG. The default is `false`.
"""
has_rng(::DEIntegrator) = false

"""
    get_rng(integrator::DEIntegrator) -> AbstractRNG

Return the live random number generator used for future stochastic work by the
integrator.

The returned object is not a copy: advancing or reseeding it changes the random
stream used by subsequent steps. Call [`has_rng`](@ref) first in generic code.
The fallback throws when the concrete integrator does not support RNG access.
"""
function get_rng(integrator::DEIntegrator)
    error(
        "Integrator of type $(typeof(integrator)) does not carry an RNG. " *
            "Ensure the solver package version supports the SciMLBase RNG interface " *
            "(has_rng / get_rng / set_rng!)."
    )
end

"""
    set_rng!(integrator::DEIntegrator, rng) -> nothing

Replace the random number generator used for future stochastic work by the
integrator.

Concrete integrators commonly require `rng` to have the same concrete type as
the existing generator because that type is part of the integrator or noise
process representation. Implementations must update every live reference used
by the integrator and its noise process. Call [`has_rng`](@ref) first in generic
code; the fallback throws for unsupported integrators.

This is needed for RNG types that don't support `Random.seed!`, such as
counter-based RNGs (Random123.jl's Philox, Threefry) which are configured via
`(key, counter)` pairs rather than a single seed. For these types, reseeding
requires constructing a new instance and swapping it in.

For RNGs that support `Random.seed!`, reseeding the object returned by
[`get_rng`](@ref) is usually sufficient. `set_rng!` is needed when reseeding
requires constructing a replacement instance.
"""
function set_rng!(integrator::DEIntegrator, rng)
    error("Integrator of type $(typeof(integrator)) does not support set_rng!.")
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
    return if !(initializealg isa NoInit)
        error("initialize_dae!: $(typeof(initializealg)) method has not been implemented for the integrator")
    end
end

"""
    auto_dt_reset!(integrator::DEIntegrator)

Recompute the integrator's initial step size from its current state.

Concrete solvers should apply the same automatic step-size selection used during
`init`, including the current state, time, parameters, tolerances, integration
direction, and method-specific limits. They must update the active step size and
any proposal state needed by the next step. This operation may evaluate the
problem function and increment solver statistics. Its return value is not part
of the interface.
"""
function auto_dt_reset!(integrator::DEIntegrator)
    error("auto_dt_reset!: method has not been implemented for the integrator")
end

"""
    change_t_via_interpolation!(integrator::DEIntegrator,t,modify_save_endpoint=Val{false},reinitialize_alg=nothing)

Move the integrator to time `t` using the method's local interpolation.

Concrete solvers should update `integrator.t`, `integrator.u`, interpolation
state, and any dependent caches consistently. If the current endpoint has
already been saved, `modify_save_endpoint` controls whether the saved endpoint
in `integrator.sol` is rewritten as well. `reinitialize_alg` is available for
methods that must rerun initialization after the time/state change.
"""
function change_t_via_interpolation!(i::DEIntegrator, args...)
    error("change_t_via_interpolation!: method has not been implemented for the integrator")
end

"""
    addsteps!(integrator::DEIntegrator, args...)

Materialize any lazy stage or derivative data required to interpolate the
integrator's current step.

Interpolation and callback code calls this hook before requesting off-grid
values. Concrete solvers with lazy dense output should populate their
interpolation caches idempotently; solvers whose interpolation needs no extra
data use the default no-op. The optional arguments are solver-specific controls
for cache construction and are not a portable user interface.
"""
addsteps!(i::DEIntegrator, args...) = nothing

"""
    reeval_internals_due_to_modification!(integrator::DEIntegrator, continuous_modification::Bool=true;
                                          callback_initializealg = nothing)

Update an integrator after callback-driven mutation.

For DAEs, callback effects may require re-solving algebraic variables to restore
consistency. If `continuous_modification` is true, solvers should also refresh
interpolation data because the mutation can affect the current continuous
segment. For discrete-only modifications, solvers may skip interpolation
recalculation when their method permits it.

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
        callback_initializealg = nothing
    )
    return reeval_internals_due_to_modification!(integrator::DEIntegrator)
end
function reeval_internals_due_to_modification!(
        integrator::DEIntegrator; callback_initializealg = nothing
    )
    return nothing
end

"""
    set_t!(integrator::DEIntegrator, t)

Set the current time of `integrator` to `t`.

`set_t!` is the direct time-mutation hook used by callbacks and generic
integrator utilities. It changes the independent variable without implying that
the state should be interpolated to the new time.

# Interface rules

  - Implementations must keep `integrator.t`, method-specific time caches, and
    any time-dependent controller state consistent with the new time.
  - `set_t!` should not change `integrator.u` except for solver-specific
    bookkeeping required to keep an already-mutated state valid.
  - Use [`change_t_via_interpolation!`](@ref) when moving to `t` should also
    recompute `u` from the method's interpolation.
  - If changing time invalidates interpolation, error estimates, or dense output
    caches, the implementation must refresh them or require callers to follow
    with [`reeval_internals_due_to_modification!`](@ref).
"""
function set_t!(integrator::DEIntegrator, t)
    error("set_t!: method has not been implemented for the integrator")
end

"""
    set_u!(integrator::DEIntegrator, u)
    set_u!(integrator::DEIntegrator, sym, val)

Set the current state of `integrator`.

The two-argument form replaces the full state and must be implemented by
concrete integrators that support direct state mutation. The three-argument form
is the generic symbolic-state update path: it verifies that `sym` is a state
variable, writes `val` into `integrator.u`, and marks a derivative discontinuity.
Parameter updates should use `integrator.ps[sym]` or SymbolicIndexingInterface
parameter setters instead of `set_u!`.

# Interface rules

  - Full-state updates must keep `integrator.u` and any solver-owned state caches
    that mirror `u` consistent.
  - Symbolic updates are only for state variables. They must reject parameters
    and unknown symbols rather than silently adding new state.
  - State mutation is treated as a derivative discontinuity because cached
    derivatives, interpolation data, and step controllers may no longer describe
    the current state.
  - Solvers that need additional work after a state change should implement
    [`reeval_internals_due_to_modification!`](@ref) and document when callbacks
    or generic code must call it.
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
    return derivative_discontinuity!(integrator, true)
end

"""
    set_ut!(integrator::DEIntegrator, u, t)

Set the current state and time of `integrator`.

The fallback calls [`set_u!`](@ref) and then [`set_t!`](@ref), so concrete
integrators can specialize either lower-level mutation hook or overload
`set_ut!` directly when state/time changes must be applied atomically.

# Interface rules

  - `set_ut!` is the preferred hook when a callback or initialization routine
    changes state and time together.
  - The default ordering is state first, then time. Integrators whose caches
    require a different ordering must overload `set_ut!`.
  - After returning, `state_values(integrator)` and `current_time(integrator)`
    should observe the updated `u` and `t`.
"""
function set_ut!(integrator::DEIntegrator, u, t)
    set_u!(integrator, u)
    return set_t!(integrator, t)
end

"""
    get_sol(integrator::DEIntegrator)

Return the solution object contained in `integrator`.

This is the public accessor for solver and generic-interface code that needs the
live solution accumulator during integration. For example, delayed symbolic
states may evaluate the current history through `get_sol(integrator)` instead of
reaching into `integrator.sol` directly.

# Interface rules

  - The returned object is the integrator's current solution storage, not a
    defensive copy.
  - Solver implementations may update this object as stepping, saving, and
    callback handling proceed.
  - Code that only needs the final solve result should use `solve`/`solve!`
    rather than relying on `get_sol` during integration.
"""
function get_sol(integrator::DEIntegrator)
    return integrator.sol
end

### Addat isn't a real thing. Let's make it a real thing Gretchen

function addat!(a::AbstractArray, idxs, val = nothing)
    return if val === nothing
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
    return derivative_discontinuity!(A, true)
end

SymbolicIndexingInterface.is_time_dependent(::DEIntegrator) = true

# TODO make this nontrivial once dynamic state selection works
SymbolicIndexingInterface.constant_structure(::DEIntegrator) = true

function Base.getproperty(A::DEIntegrator, sym::Symbol)
    if sym === :ps
        return ParameterIndexingProxy(A)
    else
        return getfield(A, sym)
    end
end

Base.@propagate_inbounds function Base.getindex(A::DEIntegrator, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, sym::Union{AbstractArray, Tuple}
    )
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
            is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, ::SymbolicIndexingInterface.SolvedVariables
    )
    return getindex(A, variable_symbols(A))
end

Base.@propagate_inbounds function Base.getindex(
        A::DEIntegrator, ::SymbolicIndexingInterface.AllVariables
    )
    return getindex(A, all_variable_symbols(A))
end

function observed(A::DEIntegrator, sym)
    return getobserved(A)(sym, A.u, A.p, A.t)
end

function Base.setindex!(A::DEIntegrator, val, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym] = $val` for parameter indexing.")
    end
    return setsym(A, sym)(A, val)
end

function Base.setindex!(A::DEIntegrator, val, sym::Union{AbstractArray, Tuple})
    if symbolic_type(sym) == NotSymbolic() && any(x -> is_parameter(A, x), sym) ||
            is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `integrator.ps[$sym] = $val` for parameter indexing.")
    end
    return setsym(A, sym)(A, val)
end

### Integrator traits

"""
    has_reinit(i::DEIntegrator)

Return whether `i` supports reinitialization through [`reinit!`](@ref).

Generic code should query this trait before attempting to reuse an initialized
solver object. A `true` result guarantees support for restarting from a new
initial state and integration interval through the concrete integrator's
documented `reinit!` method. Supported optional keywords can still vary by
problem family. The default is `false`.
"""
has_reinit(i::DEIntegrator) = false
@doc """
    has_reinit(i::DEIntegrator)

Return whether `i` supports reinitialization through [`reinit!`](@ref).

Generic code should query this trait before attempting to reuse an initialized
solver object. A `true` result guarantees support for restarting from a new
initial state and integration interval through the concrete integrator's
documented `reinit!` method. Supported optional keywords can still vary by
problem family. The default is `false`.
""" has_reinit

log_instability(integrator) = ""

### Display

function Base.summary(io::IO, I::DEIntegrator)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(I)),
        no_color, " with uType ",
        type_color, typeof(I.u),
        no_color, " and tType ",
        type_color, typeof(I.t),
        no_color
    )
end
function Base.show(io::IO, A::DEIntegrator)
    println(io, string("t: ", A.t))
    print(io, "u: ")
    return show(io, A.u)
end
function Base.show(io::IO, m::MIME"text/plain", A::DEIntegrator)
    println(io, string("t: ", A.t))
    print(io, "u: ")
    return show(io, m, A.u)
end

### Error check (retcode)

last_step_failed(integrator::DEIntegrator) = false

"""
    check_error(integrator)

Inspect `integrator` and return the [`ReturnCode`](@ref) that describes whether
integration may continue.

The common implementation preserves an existing terminal return code and checks
for a NaN step size, iteration limits, a step size at or below `dtmin`, a
user-supplied instability predicate, and failed nonlinear steps. It does not
mutate `integrator.sol.retcode`; use [`check_error!`](@ref) when the solution
must be updated. Concrete integrators may specialize the checks while preserving
the return-code contract.
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
        @SciMLMessage("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.", verbose, :dt_NaN)
        return ReturnCode.DtNaN
    end
    if integrator.iter > opts.maxiters
        @SciMLMessage(
            "Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).",
            verbose,
            :max_iters
        )
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
                (
                !step_accepted || (
                    hasproperty(opts, :tstops) ?
                        integrator.t + integrator.dt < integrator.tdir * first(opts.tstops) :
                        true
                )
            )
            diagnostic = verbosity_to_bool(verbose.dt_min_unstable) ? log_instability(integrator) : ""
            EEst = isdefined(integrator, :EEst) ? lazy", step error estimate = $(integrator.EEst)" : ""
            @SciMLMessage(lazy"dt($(integrator.dt)) <= dtmin($(opts.dtmin)) at t=$(integrator.t)$EEst. Aborting. There is either an error in your model specification or the true solution is unstable.$diagnostic", verbose, :dt_min_unstable)
            return ReturnCode.DtLessThanMin
        elseif !step_accepted && integrator.t isa AbstractFloat && abs(integrator.dt) <= abs(eps(integrator.t))
            diagnostic = verbosity_to_bool(verbose.dt_epsilon) ? log_instability(integrator) : ""
            EEst = isdefined(integrator, :EEst) ? lazy", step error estimate = $(integrator.EEst)" : ""
            @SciMLMessage(lazy"At t=$(integrator.t), dt was forced below floating point epsilon $(integrator.dt)$EEst. Aborting. There is either an error in your model specification or the true solution is unstable (or it cannot be represented in $(eltype(integrator.u)) precision).$diagnostic", verbose, :dt_epsilon)
            return ReturnCode.Unstable
        end
    end
    if step_accepted &&
            opts.unstable_check(integrator.dt, integrator.u, integrator.p, integrator.t)
        diagnostic = verbosity_to_bool(verbose.instability) ? log_instability(integrator) : ""
        @SciMLMessage("Instability detected. Aborting.$diagnostic", verbose, :instability)
        return ReturnCode.Unstable
    end
    if last_step_failed(integrator)
        @SciMLMessage("Newton steps could not converge and algorithm is not adaptive. Use a lower dt.", verbose, :newton_convergence)
        return ReturnCode.ConvergenceFailure
    end
    return ReturnCode.Success
end

function postamble! end

"""
    check_error!(integrator)

Run [`check_error`](@ref), store the resulting code in
`integrator.sol.retcode`, and return that code.

When the code is not `ReturnCode.Success`, the common implementation also calls
the solver's `postamble!` hook so pending bookkeeping and finalization are
performed before the solve exits. A successful check updates the return code but
does not finalize the integrator.
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
    return false
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


@recipe function f(
        integrator::DEIntegrator;
        denseplot = (
            integrator.opts.calck ||
                integrator isa AbstractSDEIntegrator
        ) &&
            integrator.iter > 0,
        plotdensity = 10,
        plot_analytic = false, vars = nothing, idxs = nothing
    )
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument vars will be removed in a future version. Please use keyword argument idxs instead.",
            :f; force = true
        )
        (idxs !== nothing) &&
            error("Simultaneously using keywords vars and idxs is not supported. Please only use idxs.")
        idxs = vars
    end

    int_vars = interpret_vars(idxs, integrator.sol)

    if denseplot
        # Generate the points from the plot from dense function
        plott = collect(range(integrator.tprev, integrator.t; length = plotdensity))
        if plot_analytic
            plot_analytic_timeseries = [
                integrator.sol.prob.f.analytic(
                        integrator.sol.prob.u0,
                        integrator.sol.prob.p,
                        t
                    ) for t in plott
            ]
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

    labels = String[] # Array{String, 2}(1, length(int_vars)*(1+plot_analytic))
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
                    push!(
                        plot_vecs[j],
                        u_n(plot_timeseries, x[j], sol, plott, plot_timeseries)
                    )
                else # Just get values
                    if x[j] == 0
                        push!(plot_vecs[j], integrator.t)
                    elseif x[j] == 1 && !(integrator.u isa AbstractArray)
                        push!(
                            plot_vecs[j],
                            integrator.sol.prob.f(
                                Val{:analytic}, integrator.t,
                                integrator.sol[1]
                            )
                        )
                    else
                        push!(
                            plot_vecs[j],
                            integrator.sol.prob.f(
                                Val{:analytic}, integrator.t,
                                integrator.sol[1]
                            )[x[j]]
                        )
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
    return
end

"""
    has_stats(i::DEIntegrator)

Return whether `i` exposes mutable solve statistics through its integrator
interface.

Solver integrators that maintain counters such as function evaluations, rejected
steps, nonlinear iterations, or linear solves should overload this trait to
return `true` and provide the corresponding statistics through their documented
integrator fields. The default is `false`, which tells generic code not to assume
that a `stats` field or stats update path exists.
"""
has_stats(i::DEIntegrator) = false
@doc """
    has_stats(i::DEIntegrator)

Return whether `i` exposes mutable solve statistics through its integrator
interface.

Solver integrators that maintain counters such as function evaluations, rejected
steps, nonlinear iterations, or linear solves should overload this trait to
return `true` and provide the corresponding statistics through their documented
integrator fields. The default is `false`, which tells generic code not to assume
that a `stats` field or stats update path exists.
""" has_stats

"""
    isadaptive(i::DEIntegrator)

Checks if the integrator is adaptive
"""
function isadaptive(integrator::DEIntegrator)
    return isdefined(integrator.opts, :adaptive) ? integrator.opts.adaptive : false
end

function SymbolicIndexingInterface.get_history_function(
        integ::Union{
            AbstractDDEIntegrator, AbstractSDDEIntegrator,
        }
    )
    return DDESolutionHistoryWrapper(get_sol(integ))
end
