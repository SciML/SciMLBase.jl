# SciMLBase

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/SciMLBase/stable)

[![codecov](https://codecov.io/gh/SciML/SciMLBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/SciMLBase.jl)
[![Build Status](https://github.com/SciML/SciMLBase.jl/workflows/CI/badge.svg)](https://github.com/SciML/SciMLBase.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

SciMLBase.jl is the core interface definition of the SciML ecosystem. It is a
low dependency library made to be depended on by the downstream libraries to
supply the common interface and allow for interexchange of mathematical problems.

## v3.0 Breaking Changes

#### RecursiveArrayTools v4: Solution types are now AbstractArrays (#1297)

**Most impactful change.** `AbstractVectorOfArray` (and thus `ODESolution`, `DDESolution`, `RODESolution`, `DAESolution`) now subtypes `AbstractArray`:

- **`sol[i]` returns the `i`th scalar element** (column-major), not the `i`th timestep. Use `sol.u[i]` or `sol[:, i]` for timesteps.
- **`length(sol)` returns total elements** (`prod(size(sol))`). Use `length(sol.u)` for number of timesteps.
- **`iterate(sol)` iterates scalar elements**. Use `sol.u` for timestep iteration.
- **`map(f, sol)` maps over elements**. Use `map(f, sol.u)` for timesteps.

#### Ensemble RNG redesign (#1252)

- `prob_func(prob, i, repeat)` → `prob_func(prob, ctx)` where `ctx::EnsembleContext`
- `output_func(sol, i)` → `output_func(sol, ctx)`
- `EnsembleContext` includes `sim_id`, `repeat`, `rng`, `sim_seed`, `worker_id`, `master_rng`
- New `seed`/`rng`/`rng_func` kwargs on `solve()` for deterministic, thread-count-independent ensemble solves

#### Removed deprecated APIs
- `u_modified!` renamed to `derivative_discontinuity!` (#1289)
- Removed `deprecated.jl`: old type aliases (`DEAlgorithm`, `DEProblem`, `DESolution`, etc.), constructors, deprecated accessors (#1291)
- Removed backward compat shims in `remake.jl` and MLStyle extension (#1292)
- Removed old iterators: `tuples`, `intervals`, `TimeChoiceIterator` (#1290)

#### Simplified getproperty
- Removed redundant `getproperty` overloads on solution abstract types (#1293)
- Removed deprecated `getproperty` aliases (`.destats`, `.x`, `.lb`/`.ub`, `.minimizer`, `.minimum`) (#1294)

#### Other breaking changes
- Replaced Moshi with plain Julia structs for Clocks — 23% precompilation improvement (#1295)
- `ODEFunction` uses `DEFAULT_SPECIALIZATION` (AutoSpecialize) for convenience constructors (#1300)
- Propagate `interp`/`dense` to DiffEqArrays from solution callables (#1297)
- `is_discrete_time_domain(nothing)` now returns `false` (#1306)

#### Migration Guide

| Old (v2) | New (v3) |
|----------|----------|
| `sol[i]` (timestep) | `sol.u[i]` or `sol[:, i]` |
| `length(sol)` (timesteps) | `length(sol.u)` |
| `for u in sol` | `for u in sol.u` |
| `u_modified!(integrator, true)` | `derivative_discontinuity!(integrator, true)` |
| `prob_func(prob, i, repeat)` | `prob_func(prob, ctx)` — use `ctx.sim_id`, `ctx.repeat` |
| `output_func(sol, i)` | `output_func(sol, ctx)` |
| `sol.destats` | `sol.stats` |
| `ODEFunction{true}(f)` (FullSpecialize) | Now uses AutoSpecialize by default |

## v2.0 Breaking Changes

The breaking changes in v2.0 are:

  - `IntegralProblem` has moved to an interface with `IntegralFunction` and `BatchedIntegralFunction` which requires specifying `prototype`s for the values to be modified
    instead of `nout` and `batch`. https://github.com/SciML/SciMLBase.jl/pull/497
  - `ODEProblem` was made temporarily into a `mutable struct` to allow for EnzymeRules support. Using the mutation throws a warning that this is only experimental and should not be relied on.
    https://github.com/SciML/SciMLBase.jl/pull/501
  - `BVProblem` now has a new interface for `TwoPointBVProblem` which splits the bc terms for the two sides, forcing a true two-point BVProblem to allow for further specializations and to allow
    for wrapping Fortran solvers in the interface. https://github.com/SciML/SciMLBase.jl/pull/477
  - `SDEProblem` constructor was changed to remove an anti-pattern which required passing the diffusion function `g` twice, i.e. `SDEProblem(SDEFunction(f,g),g, ...)`.
    Now this is simply `SDEProblem(SDEFunction(f,g),...)`. https://github.com/SciML/SciMLBase.jl/pull/489
