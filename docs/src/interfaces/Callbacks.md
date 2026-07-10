# [Callback Interface](@id callback_interface)

Callbacks describe event handling and solver-side actions that run during a
differential equation solve. They are passed through the common `callback`
keyword, usually as a single callback or as a `CallbackSet`.

Continuous callbacks locate events inside a solver step by monitoring zero
crossings. Discrete callbacks are evaluated at solver-controlled points and
trigger when their condition is true. Vector continuous callbacks combine many
continuous events into one condition function.

## Interface Rules

- Callback `condition` functions inspect the proposed state, time, and
  integrator without mutating the integrator.
- Callback `affect!` functions may mutate the integrator. State changes that
  introduce discontinuities should normally use `save_positions = (true, true)`.
- Continuous callbacks and vector continuous callbacks are processed before
  discrete callbacks. Discrete callbacks are then applied in order.
- For DAEs, callback effects may require consistent reinitialization. The
  callback `initializealg` keyword controls this per callback when it is not
  inherited from `solve`.
- ModelingToolkit-generated callbacks may carry `saved_clock_partitions`.
  SciMLBase's discrete-save hooks use this metadata to keep time-series
  parameters synchronized with callback effects.

The root-finding mode for continuous events is selected by
`SciMLBase.LeftRootFind`, `SciMLBase.RightRootFind`, or
`SciMLBase.NoRootFind`.

Callback-specific discrete-save hooks extend [`SciMLBase.save_discretes!`](@ref),
which is documented with the [symbolic save-index interface](@ref symbolic_save_idxs).

## API

```@docs
SciMLBase.DECallback
SciMLBase.AbstractContinuousCallback
SciMLBase.AbstractDiscreteCallback
SciMLBase.NoRootFind
SciMLBase.LeftRootFind
SciMLBase.RightRootFind
SciMLBase.ContinuousCallback
SciMLBase.VectorContinuousCallback
SciMLBase.DiscreteCallback
SciMLBase.CallbackSet
SciMLBase.split_callbacks
SciMLBase.save_final_discretes!
SciMLBase.save_discretes_if_enabled!
```
