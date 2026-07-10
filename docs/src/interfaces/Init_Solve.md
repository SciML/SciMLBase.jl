# The SciML init and solve Functions

`solve` function has the default definition

```julia
solve(args...; kwargs...) = solve!(init(args...; kwargs...))
```

The interface for the three functions is as follows:

```julia
init(::ProblemType, args...; kwargs...)::IteratorType
solve!(::IteratorType)::SolutionType
```

where `ProblemType`, `IteratorType`, and `SolutionType` are the types defined in
your package.

To avoid method ambiguity, the first argument of `solve`, `solve!`, and `init`
_must_ be dispatched on the type defined in your package.  For example, do
_not_ define a method such as

```julia
init(::AbstractVector, ::AlgorithmType)
```

## `init` and the Iterator Interface

`init`'s return gives an `IteratorType` which is designed to allow the user to
have more direct handling over the internal solving process. Because of this
internal nature, the `IteratorType` has a less unified interface across problem
types than other portions like `ProblemType` and `SolutionType`. For example,
for differential equations this is the
[Integrator Interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/)
designed for mutating solutions in a manner for callback implementation, which
is distinctly different from the
[LinearSolve init interface](https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface)
which is designed for caching efficiency with reusing factorizations.

## `__solve` and High-Level Handling

While `init` and `solve` are the common entry point for users, solver packages will
mostly define dispatches on `SciMLBase.__init` and `SciMLBase.__solve`. The reason is
because this allows for `SciMLBase.init` and `SciMLBase.solve` to have common
implementations across all solvers for doing things such as checking for common
errors and throwing high level messages. Solvers can opt-out of the high level
error handling by directly defining `SciMLBase.init` and `SciMLBase.solve` instead,
though this is not recommended in order to allow for uniformity of the error messages.

```@docs
SciMLBase.__init
SciMLBase.__solve
```

## Low-Level Integrator Interface

Differential equation `init` methods return mutable `DEIntegrator` objects.
Solver packages should expose state changes, callback effects, cache access, and
manual stepping through this interface instead of requiring users or callbacks to
reach into solver-specific internals.

### Integrator Rules

- `step!(integrator)` advances one accepted solver step. `step!(integrator, dt)`
  advances by a signed time interval in the direction of `integrator.tdir`.
- Callback code should mutate integrators through public hooks such as `set_u!`,
  `set_t!`, `set_ut!`, `terminate!`, `add_tstop!`, and
  `derivative_discontinuity!`.
- `get_tmp_cache` returns scratch arrays whose contents may be reused by the
  next integrator operation. Do not store them beyond the current callback or
  method call.
- `user_cache` exposes caches documented by the solver as safe for user
  mutation. `full_cache`, `u_cache`, `du_cache`, and the non-user cache hooks are
  for solver and generic-interface code that must keep internal caches aligned
  with `integrator.u`.
- Dynamic-size integrators that support `resize!`, `deleteat!`, or `addat!`
  must update state, saved state, and non-user caches consistently.
- Symbolic state access follows `SymbolicIndexingInterface`: `integrator[sym]`
  reads state variables, `set_u!(integrator, sym, val)` writes state variables,
  and parameter access should go through `integrator.ps[sym]`.

```@docs
SciMLBase.DEIntegrator
SciMLBase.AbstractSteadyStateIntegrator
SciMLBase.AbstractODEIntegrator
SciMLBase.AbstractSecondOrderODEIntegrator
SciMLBase.AbstractSDEIntegrator
SciMLBase.AbstractRODEIntegrator
SciMLBase.AbstractDDEIntegrator
SciMLBase.AbstractDAEIntegrator
SciMLBase.AbstractSDDEIntegrator
SciMLBase.DECache
SciMLBase.step!
Base.resize!(::SciMLBase.DEIntegrator, ::Int)
Base.deleteat!(::SciMLBase.DEIntegrator, ::Any)
SciMLBase.addat!
SciMLBase.get_tmp_cache
SciMLBase.user_cache
SciMLBase.u_cache
SciMLBase.du_cache
SciMLBase.ratenoise_cache
SciMLBase.rand_cache
SciMLBase.full_cache
SciMLBase.resize_non_user_cache!
SciMLBase.deleteat_non_user_cache!
SciMLBase.addat_non_user_cache!
SciMLBase.terminate!
SciMLBase.add_tstop!
SciMLBase.has_tstop
SciMLBase.first_tstop
SciMLBase.pop_tstop!
SciMLBase.add_saveat!
SciMLBase.get_du
SciMLBase.get_du!
SciMLBase.get_dt
SciMLBase.get_proposed_dt
SciMLBase.set_proposed_dt!
SciMLBase.set_abstol!
SciMLBase.set_reltol!
SciMLBase.derivative_discontinuity!
SciMLBase.u_modified!
SciMLBase.savevalues!
SciMLBase.reinit!
SciMLBase.auto_dt_reset!
SciMLBase.change_t_via_interpolation!
SciMLBase.addsteps!
SciMLBase.reeval_internals_due_to_modification!
SciMLBase.set_t!
SciMLBase.set_u!
SciMLBase.set_ut!
SciMLBase.get_sol
SciMLBase.check_error
SciMLBase.check_error!
SciMLBase.initialize_dae!
SciMLBase.has_reinit
SciMLBase.has_rng
SciMLBase.get_rng
SciMLBase.set_rng!
```

### Mutable Integrator Controls

```@docs
SciMLBase.get_dt
SciMLBase.set_abstol!
SciMLBase.set_reltol!
SciMLBase.u_modified!
SciMLBase.addsteps!
```

## Initialization Interface

```@docs
SciMLBase.OverrideInitData
SciMLBase.get_initial_values
```

## Argument Validation

```@docs
SciMLBase.numargs
SciMLBase.FunctionArgumentsError
SciMLBase.TooFewArgumentsError
SciMLBase.TooManyArgumentsError
```
