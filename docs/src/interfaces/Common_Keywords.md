# [Common `solve` and `init` Keyword Arguments](@id common_solver_keywords)

SciML problem families share a vocabulary of keyword arguments for `solve` and
`init`. This vocabulary is an interoperability contract, not a promise that
every backend implements every option. Concrete solver packages must document
unsupported keywords and any defaults that differ from the common behavior.

Problem constructors may store solve keywords in `prob.kwargs`. High-level solve
implementations merge those stored keywords before dispatching to the backend:
keywords supplied directly to `solve` or `init` take precedence. When both the
problem and the call provide callbacks, they are combined into a `CallbackSet`
by default; `merge_callbacks = false` lets a backend disable that combination.

Method-specific configuration belongs in the algorithm constructor. For
example, an automatic differentiation backend used only by one algorithm should
be selected by `MyAlgorithm(; autodiff = ...)`, while tolerances, saving, and
callback controls that apply across a problem family belong in `solve` or
`init`.

## Default Algorithm Hints

When no algorithm is supplied, `alg_hints` gives the package that owns default
selection high-level facts about the problem. It is a collection of symbols;
the default selector decides which hints it supports.

The common deterministic hints are:

  - `:auto`: let the selector choose whether stiffness detection or switching is
    appropriate. This is the usual default.
  - `:nonstiff`: prefer an explicit nonstiff method.
  - `:stiff`: prefer a method intended for stiff equations.

Stochastic selectors additionally use:

  - `:additive`: the diffusion is independent of the state.
  - `:commutative`: the noise vector fields satisfy the commutativity assumptions
    required by methods that avoid general iterated stochastic integrals.
  - `:stratonovich`: select a solver with the Stratonovich interpretation rather
    than the default Ito interpretation.

`:interpolant` and `:memorybound` are reserved hints for interpolation quality
and memory-bound workloads. Standard selectors may currently ignore them.
Passing an explicit algorithm bypasses default selection, so `alg_hints` should
not be used to configure an already selected method.

## Output Control

Saving defaults favor interactive use. The OrdinaryDiffEq-style defaults below
are the reference behavior; wrapped or non-time-stepping solvers may support a
smaller subset.

  - `dense`: Save method-specific data needed for continuous interpolation.
    For algorithms with dense output, the default is
    `save_everystep && isempty(saveat)` unless the algorithm uses linear
    interpolation by default. With `dense = false`, callable time-series
    solutions use interpolation supported by the stored points, commonly linear
    interpolation.
  - `saveat`: Save at specified independent-variable values. A scalar expands to
    a regular range over `tspan`; a collection supplies the values directly.
    Providing `saveat` alone changes the usual defaults of `save_everystep` and
    `dense` to `false`. The default is an empty collection.
  - `save_idxs`: Save only selected state components. It may be an integer,
    collection of indices, or supported symbolic state/time-series-parameter
    selection. Observed variables are not supported (see
    [symbolic save_idxs](@ref symbolic_save_idxs) and
    SciML/DifferentialEquations.jl#1036). Symbolically subsetted solutions must
    preserve the [saved-subsystem interface](@ref symbolic_save_idxs).
  - `tstops`: Require the integrator to stop at additional values. This is used
    for discontinuities, singularities, and externally scheduled events. A
    scalar or collection is known at initialization; a callable such as
    `tstops(p, tspan)` is late-bound and requires an algorithm for which
    [`allows_late_binding_tstops`](@ref SciMLBase.allows_late_binding_tstops) is
    true. Fixed-step methods either take
    a shorter step, interpolate, or reject incompatible stops according to their
    documented capabilities.
  - `d_discontinuities`: Mark discontinuities in low-order derivatives of the
    vector field. Each value is also a stop. OrdinaryDiffEq advances one ULP in
    the integration direction and refreshes derivative caches on the
    post-discontinuity side. Its convention is right-continuous: `f` at the
    marked value is the old regime and `f` just after it is the new regime.
  - `save_everystep`: Save every accepted step. The usual default is
    `isempty(saveat)`.
  - `save_on`: Master switch for intermediate saving. When false it overrides
    `dense`, `saveat`, and `save_everystep`. The default is `true`.
  - `save_start`: Include the initial value. The OrdinaryDiffEq default is true
    when every step is saved, `saveat` is empty or scalar, or the initial value
    occurs in `saveat`. Explicit `false` suppresses the initial value even if it
    appears in `saveat`.
  - `save_end`: Force inclusion of the final value. The derived default follows
    the same conditions as `save_start`, evaluated at `tspan[end]`.
  - `initialize_save`: Save after callback initialization when initialization
    modifies the state. The default is `true`.
  - `save_discretes`: Save supported discrete/time-varying parameter partitions
    alongside the state. The default for OrdinaryDiffEq integrators is `true`.
  - `save_noise`: Preserve the stochastic noise path when the solver supports
    it. The default is backend-specific and is `false` in the common
    OrdinaryDiffEq initialization path.

Do not combine `dense = true` with a nonempty `saveat` in OrdinaryDiffEq-style
integrators. Dense output requires the data retained at every accepted step;
use a saving callback when additional sampled output is needed at the same time.

## Step-Size Control

Adaptive methods compare a normalized local error against one. The common
componentwise scaling is equivalent to

```math
\frac{\mathrm{error}}
     {\mathrm{abstol} +
      \max(\mathrm{internalnorm}(u_{prev}),
           \mathrm{internalnorm}(u))\,\mathrm{reltol}}.
```

`abstol` controls error near zero; `reltol` controls error relative to the
state magnitude. Either tolerance may be scalar or, when supported, shaped like
the state for componentwise control.

  - `adaptive`: Enable adaptive stepping for a method that supports it. The
    usual default is true for adaptive algorithms.
  - `abstol`, `reltol`: Absolute and relative local-error tolerances. Current
    OrdinaryDiffEq defaults are `1e-6` and `1e-3` for deterministic equations;
    stochastic solver families commonly use `1e-2` for both. Backends may choose
    different defaults.
  - `dt`: Initial step size for adaptive methods and nominal step size for
    fixed-step methods. Adaptive methods choose it automatically when omitted.
  - `dtmax`, `dtmin`: Bounds on adaptive step size. Defaults depend on the
    problem time span and backend.
  - `force_dtmin`: Continue at `dtmin` even when the local error test rejects
    that step size. The default is `false`; setting it to true permits tolerance
    violations and is unsupported by many wrapped solvers.
  - `internalnorm`: Callable `internalnorm(u, t)` used to reduce state and error
    quantities. Solver code may also call it on scalar state elements.

For a nonadaptive method:

  - With `dt`, ordinary steps use that size and may shorten a step to hit a
    compatible `tstop`.
  - With `tstops` but no `dt`, the stops define the step endpoints.
  - With neither `dt` nor `tstops`, a solver that cannot infer a fixed step must
    throw an error.

### Advanced Adaptive Controls

The following controls are meaningful only for algorithms/controllers that use
them. Their defaults are algorithm-specific.

  - `controller`: Step-size controller object.
  - `gamma`: Safety factor used by the controller.
  - `beta1`, `beta2`: Stabilization parameters for PI/PID-like controllers.
  - `qmax`, `qmin`: Bounds on the proposed step-size ratio.
  - `qsteady_min`, `qsteady_max`: Ratio interval in which the current step size
    is retained.
  - `qoldinit`: Initial history value for stabilized controllers.
  - `failfactor`: Factor used to reduce a step after an implicit solve failure.

## Memory and Ownership

  - `calck`: Retain intermediate interpolation data needed during integration.
    This is distinct from post-solve `dense` output. OrdinaryDiffEq enables it
    for callbacks, dense output, or nonempty `saveat`; disabling it can reduce
    memory only when no requested operation needs interpolation.
  - `alias`: An [`AbstractAliasSpecifier`](@ref SciMLBase.AbstractAliasSpecifier)
    or boolean convenience value
    controlling whether solver caches may retain references to problem inputs.
    See the [alias specifier interface](@ref alias_specifier_interface) for the
    tri-state ownership rules.

Reusable solver caches are problem- and backend-specific. They are not a common
`cache` keyword contract; use the cache/init interface documented by the owning
solver package.

## Termination, Callbacks, and Overrides

  - `maxiters`: Maximum solver iterations. OrdinaryDiffEq defaults to
    `1_000_000` for adaptive algorithms and `typemax(Int)` for fixed-step
    algorithms; other solver families choose their own limits.
  - `maxtime`: Optional wall-clock limit for backends that support timed
    termination.
  - `callback`: Callback or `CallbackSet` executed by the solver. See the
    [callback interface](@ref callback_interface) for condition/effect,
    ordering, saving, and initialization rules.
  - `initializealg`: Initialization algorithm for DAEs and constrained/mass-
    matrix problems. The common marker interface includes
    [`CheckInit`](@ref SciMLBase.CheckInit), [`NoInit`](@ref SciMLBase.NoInit),
    and [`OverrideInit`](@ref SciMLBase.OverrideInit); solver packages may add
    concrete initialization methods.
  - `isoutofdomain`: Predicate `isoutofdomain(u, p, t)`. Returning true rejects a
    proposed step. The default accepts every state.
  - `unstable_check`: Predicate `unstable_check(dt, u, p, t)` used for early
    termination after detected numerical instability. The default is
    backend-specific.
  - `termination_condition`: Solver-family-specific convergence or termination
    condition.
  - `verbose`: Boolean or verbosity policy controlling solver diagnostics.
  - `u0`, `p`: Call-site replacements for the problem's initial state and
    parameters. `nothing` means to use the values stored in the problem.
  - `wrap`: Control whether a structured `problem_type(prob)` marker receives its preferred
    solution wrapper. Ordinary differential equation solves use `Val(true)` by
    default and accept `Val(false)` to keep the underlying solution.
  - `rng`: Explicit random number generator for stochastic solver operations and
    callbacks. When supported it takes precedence over `seed`; use
    [`has_rng`](@ref), [`get_rng`](@ref), and [`set_rng!`](@ref) for an initialized
    integrator.
  - `seed`: Seed used by solver families that construct their own RNG/noise
    process instead of accepting `rng` directly.
  - `userdata`: Backend-owned object stored on an integrator for application or
    callback use.

## Progress Monitoring

Progress uses the Julia logging interface and `ProgressLogging.jl`-compatible
consumers; it is not tied to a particular IDE.

  - `progress`: Enable progress events. Default is `false`.
  - `progress_steps`: Accepted steps between events. Default is `1000` in
    OrdinaryDiffEq.
  - `progress_name`: Display name for the progress operation.
  - `progress_message`: Callable used to build the message. The common
    OrdinaryDiffEq callable reports `dt`, `t`, and a largest-magnitude state
    component.
  - `progress_id`: Logging identifier used to distinguish simultaneous solves.

## Error Calculations

When a problem function provides an analytical solution, solution construction
can compute diagnostic errors:

  - `timeseries_errors`: Compute errors at saved solution points. The common
    differential-equation default is `true`.
  - `dense_errors`: Compute interpolation errors on a denser reference grid.
    The common default is `false` and the option requires analytical and
    interpolation support.

These diagnostics populate solution error fields; they do not change adaptive
step acceptance.

## Automatic Differentiation

`sensealg` selects the sensitivity/automatic-differentiation strategy for a
solve. It is passed through the high-level solve interface so differentiation
rules can dispatch on it. See the
[automatic differentiation and sensitivity interface](@ref sensealg).
