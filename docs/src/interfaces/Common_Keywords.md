# Common Keyword Arguments

The following defines the keyword arguments which are meant to be preserved
throughout all of the SciMLProblem cases (where applicable).

## Default Algorithm Hinting

To help choose the default algorithm, the keyword argument `alg_hints` is
provided to `solve`. `alg_hints` is a `Vector{Symbol}` which describe the
problem at a high level to the solver. The options are:

This functionality is derived via the benchmarks in
[SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl)

Currently this is only implemented for the differential equation solvers.

## Output Control

These arguments control the output behavior of the solvers. It defaults to maximum
output to give the best interactive user experience, but can be reduced all the
way to only saving the solution at the final timepoint.

The following options are all related to output control. See the "Examples"
section at the end of this page for some example usage.

* `dense`: Denotes whether to save the extra pieces required for dense (continuous)
  output. Default is `save_everystep && !isempty(saveat)` for algorithms which have
  the ability to produce dense output, i.e. by default it's `true` unless the user
  has turned off saving on steps or has chosen a `saveat` value. If `dense=false`,
  the solution still acts like a function, and `sol(t)` is a linear interpolation
  between the saved time points.
* `saveat`: Denotes specific times to save the solution at, during the solving
  phase. The solver will save at each of the timepoints in this array in the
  most efficient manner available to the solver. If only `saveat` is given, then
  the arguments `save_everystep` and `dense` are `false` by default.
  If `saveat` is given a number, then it will automatically expand to
  `tspan[1]:saveat:tspan[2]`. For methods where interpolation is not possible,
  `saveat` may be equivalent to `tstops`. The default value is `[]`.
* `save_idxs`: Denotes the indices for the components of the equation to save.
  Defaults to saving all indices. For example, if you are solving a 3-dimensional ODE,
  and given `save_idxs = [1, 3]`, only the first and third components of the
  solution will be outputted.
  Notice that of course in this case the outputed solution will be two-dimensional.
* `tstops`: Denotes *extra* times that the timestepping algorithm must step to.
  This should be used to help the solver deal with discontinuities and
  singularities, since stepping exactly at the time of the discontinuity will
  improve accuracy. If a method cannot change timesteps (fixed timestep
  multistep methods), then `tstops` will use an interpolation,
  matching the behavior of `saveat`. If a method cannot change timesteps and
  also cannot interpolate, then `tstops` must be a multiple of `dt` or else an
  error will be thrown. Default is `[]`.
* `d_discontinuities:` Denotes locations of discontinuities in low order derivatives.
  This will force FSAL algorithms which assume derivative continuity to re-evaluate
  the derivatives at the point of discontinuity. The default is `[]`.
* `save_everystep`: Saves the result at every step.
  Default is true if `isempty(saveat)`.
* `save_on`: Denotes whether intermediate solutions are saved. This overrides the
  settings of `dense`, `saveat` and `save_everystep` and is used by some applicatioins
  to manually turn off saving temporarily. Everyday use of the solvers should leave
  this unchanged. Defaults to `true`.
* `save_start`: Denotes whether the initial condition should be included in
  the solution type as the first timepoint. Defaults to `true`.
* `save_end`: Denotes whether the final timepoint is forced to be saved,
  regardless of the other saving settings. Defaults to `true`.
* `initialize_save`: Denotes whether to save after the callback initialization
  phase (when `u_modified=true`). Defaults to `true`.

Note that `dense` requires `save_everystep=true` and `saveat=false`.

## Stepsize Control

These arguments control the timestepping routines.

#### Basic Stepsize Control

* `adaptive`: Turns on adaptive timestepping for appropriate methods. Default
  is true.
* `abstol`: Absolute tolerance in adaptive timestepping. This is the tolerance
  on local error estimates, not necessarily the global error (though these quantities
  are related).
* `reltol`: Relative tolerance in adaptive timestepping.  This is the tolerance
  on local error estimates, not necessarily the global error (though these quantities
  are related).
* `dt`: Sets the initial stepsize. This is also the stepsize for fixed
  timestep methods. Defaults to an automatic choice if the method is adaptive.
* `dtmax`: Maximum dt for adaptive timestepping. Defaults are
  package-dependent.
* `dtmin`: Minimum dt for adaptive timestepping. Defaults are
  package-dependent.

#### Fixed Stepsize Usage

Note that if a method does not have adaptivity, the following rules apply:

* If `dt` is set, then the algorithm will step with size `dt` each iteration.
* If `tstops` and `dt` are both set, then the algorithm will step with either a
  size `dt`, or use a smaller step to hit the `tstops` point.
* If `tstops` is set without `dt`, then the algorithm will step directly to
  each value in `tstops`
* If neither `dt` nor `tstops` are set, the solver will throw an error.

## Memory Optimizations

* `alias_u0`: allows the solver to alias the initial condition array that is contained
  in the problem struct. Defaults to false.
* `cache`: pass a solver cache to decrease the construction time. This is not implemented
  for any of the problem interfaces at this moment.

## Miscellaneous

* `maxiters`: Maximum number of iterations before stopping.
* `callback`: Specifies a callback function that is called between iterations.
* `verbose`: Toggles whether warnings are thrown when the solver exits early.
  Defaults to true.

## Progress Monitoring

These arguments control the usage of the progressbar in the logger.

* `progress`: Turns on/off the Juno progressbar. Default is false.
* `progress_steps`: Numbers of steps between updates of the progress bar.
  Default is 1000.
* `progress_name`: Controls the name of the progressbar. Default is the name
  of the problem type.
* `progress_message`: Controls the message with the progressbar. Defaults to
  showing `dt`, `t`, the maximum of `u`.

The progress bars all use the Julia Logging interface in order to be generic
to the IDE or programming tool that is used. For more information on how this
is all put together, see [this discussion](https://github.com/FedeClaudi/Term.jl/discussions/67).

## Error Calculations

If you are using the test problems (i.e. `SciMLFunction`s where `f.analytic` is
defined), then options control the errors which are calculated. By default,
any cheap error estimates are always calculated. Extra keyword arguments include:

* `timeseries_errors`
* `dense_errors`

for specifying more expensive errors.

## Automatic Differentiation Control

See the [Automatic Differentiation page for a full description of `sensealg`](@ref sensealg)