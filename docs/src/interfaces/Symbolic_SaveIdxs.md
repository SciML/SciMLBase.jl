# [Symbolic `save_idxs` and Saved Subsystems](@id symbolic_save_idxs)

```@docs
SciMLBase.AllObserved
```

When a symbolic problem is solved with `save_idxs`, a solution may contain only a
subset of the original state variables or time-series parameters. The solution
interface still needs symbolic indexing to behave as if the original system were
available.

The saved-subsystem interface records how the saved arrays map back to the
original symbolic system. It is used by solution indexing, `state_values`, and
time-series parameter storage so solver packages can save less data without
breaking symbolic queries.

## Required Behavior

- `sol[x]` should use the saved value for a saved symbolic state `x`.
- `state_values(sol, i)` should reconstruct a full state-like object by starting
  from the problem's original state layout and replacing the saved entries.
- Time-series parameters saved from discrete callbacks should be recognized as
  time-series parameters only when that parameter's time series was saved.
- Saving only time-series parameters is valid; the state `save_idxs` passed to a
  low-level solver is then `Int[]`.
- **Observed variables are not supported in `save_idxs`.** Selecting an observed
  quantity raises an `ArgumentError` that names the limitation and lists
  workarounds (full-state solve + `sol[obs]`, `DiffEqCallbacks.SavingCallback`,
  or saving the dependent states). Supporting observed `save_idxs` requires
  evaluating observed functions at save points and extending `SavedSubsystem`
  with an observed-column map; see SciML/DifferentialEquations.jl#1036.

## Solver-Author Flow

Solver packages that construct solutions from symbolic `save_idxs` should call
`get_save_idxs_and_saved_subsystem(prob, save_idxs)` before solving. The first
return value is the integer state selection passed to the solver. The second
return value is passed to `build_solution(...; saved_subsystem = ss)` and then
stored on the concrete solution or exposed through `get_saved_subsystem`.

Concrete time-series solution types with a saved subsystem must route symbolic
time-series parameter operations through `SavedSubsystemWithFallback`; the
fallback implementations for `AbstractTimeseriesSolution` do this when
`get_saved_subsystem(sol) !== nothing`.

## API

```@docs
SciMLBase.get_saved_subsystem
SciMLBase.SavedSubsystem
SciMLBase.get_saved_state_idxs
SciMLBase.SavedSubsystemWithFallback
SciMLBase.get_save_idxs_and_saved_subsystem
SciMLBase.create_parameter_timeseries_collection
SciMLBase.get_saveable_values
SciMLBase.save_discretes!
```
