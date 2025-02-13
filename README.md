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
