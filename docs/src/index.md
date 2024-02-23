# The SciML Common Interface for Julia Equation Solvers

The SciML common interface ties together the numerical solvers of the Julia
package ecosystem into a single unified interface. It is designed for maximal
efficiency and parallelism, while incorporating essential features for large-scale
scientific machine learning such as differentiability, composability, and sparsity.

This documentation is made to pool together the docs of the various SciML libraries
to paint the overarching picture, establish development norms, and document the
shared/common functionality.

## Domains of SciML

The SciML common interface covers the following domains:

  - Linear systems (`LinearProblem`)
    
      + Direct methods for dense and sparse
      + Iterative solvers with preconditioning

  - Nonlinear Systems (`NonlinearProblem`)
    
      + Rootfinding for systems of nonlinear equations
  - Interval Nonlinear Systems
    
      + Bracketing rootfinders for nonlinear equations with interval bounds
  - Integrals (quadrature) (`IntegralProblem`)
  - Differential Equations
    
      + Discrete equations (function maps, discrete stochastic (Gillespie/Markov)
        simulations) (`DiscreteProblem`)
      + Ordinary differential equations (ODEs) (`ODEProblem`)
      + Split and Partitioned ODEs (Symplectic integrators, IMEX Methods) (`SplitODEProblem`)
      + Stochastic ordinary differential equations (SODEs or SDEs) (`SDEProblem`)
      + Stochastic differential-algebraic equations (SDAEs) (`SDEProblem` with mass matrices)
      + Random differential equations (RODEs or RDEs) (`RODEProblem`)
      + Differential algebraic equations (DAEs) (`DAEProblem` and `ODEProblem` with mass matrices)
      + Delay differential equations (DDEs) (`DDEProblem`)
      + Neutral, retarded, and algebraic delay differential equations (NDDEs, RDDEs, and DDAEs)
      + Stochastic delay differential equations (SDDEs) (`SDDEProblem`)
      + Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)
      + Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions) (`AbstractDEProblem`s with callbacks)
  - Optimization (`OptimizationProblem`)
    
      + Nonlinear (constrained) optimization
  - (Stochastic/Delay/Differential-Algebraic) Partial Differential Equations (`PDESystem`)
    
      + Finite difference and finite volume methods
      + Interfaces to finite element methods
      + Physics-Informed Neural Networks (PINNs)
      + Integro-Differential Equations
      + Fractional Differential Equations

The SciML common interface also includes
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
for defining such systems symbolically, allowing for optimizations like automated
generation of parallel code, symbolic simplification, and generation of sparsity
patterns.

## Extended SciML Domain

In addition to the purely numerical representations of mathematical objects, there are also
sets of problem types associated with common mathematical algorithms. These are:

  - Data-driven modeling
    
      + Discrete-time data-driven dynamical systems (`DiscreteDataDrivenProblem`)
      + Continuous-time data-driven dynamical systems (`ContinuousDataDrivenProblem`)
      + Symbolic regression (`DirectDataDrivenProblem`)

  - Uncertainty quantification and expected values (`ExpectationProblem`)

## Inverse Problems, Parameter Estimation, and Structural Identification

We note that parameter estimation and inverse problems are solved directly on their
constituent problem types using tools like [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/).
Thus for example, there is no `ODEInverseProblem`, and instead `ODEProblem` is used to
find the parameters `p` that solve the inverse problem.

## Common Interface High Level

The SciML interface is common as the usage of arguments is standardized across
all of the problem domains. Underlying high level ideas include:

  - All domains use the same interface of defining a `AbstractSciMLProblem` which is then
    solved via `solve(prob,alg;kwargs)`, where `alg` is a `AbstractSciMLAlgorithm`. The
    keyword argument namings are standardized across the organization.
  - `AbstractSciMLProblem`s are generally defined by a `SciMLFunction` which can define
    extra details about a model function, such as its analytical Jacobian, its
    sparsity patterns and so on.
  - There is an organization-wide method for defining linear and nonlinear solvers
    used within other solvers, giving maximum control of performance to the user.
  - Types used within the packages are defined by the input types. For example,
    packages attempt to internally use the type of the initial condition as the
    type for the state within differential equation solvers.
  - `solve` calls should be thread-safe and parallel-safe.
  - `init(prob,alg;kwargs)` returns an iterator which allows for directly iterating
    over the solution process
  - High performance is key. Any performance that is not at the top level is considered
    a bug and should be reported as such.
  - All functions have an in-place and out-of-place form, where the in-place form
    is made to utilize mutation for high performance on large-scale problems and
    the out-of-place form is for compatibility with tooling like static arrays and
    some reverse-mode automatic differentiation systems.

## User-Facing Solver Libraries

  - [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
    
      + Multi-package interface of high performance numerical solvers of
        differential equations

  - [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
    
      + The symbolic modeling package which implements the SciML symbolic common
        interface.
  - [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/)
    
      + Multi-package interface for specifying linear solvers (direct, sparse,
        and iterative), along with tools for caching and preconditioners
        for use in large-scale modeling.
  - [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/)
    
      + High performance numerical solving of nonlinear systems.
  - [Integrals.jl](https://docs.sciml.ai/Integrals/stable/)
    
      + Multi-package interface for high performance, batched, and parallelized
        numerical quadrature.
  - [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)
    
      + Multi-package interface for numerical solving of optimization problems.
  - [NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/)
    
      + Physics-Informed Neural Network (PINN) package for transforming partial
        differential equations into optimization problems.
  - [DiffEqOperators.jl](https://docs.sciml.ai/DiffEqOperators/stable/)
    
      + Automated finite difference method (FDM) package for transforming partial
        differential equations into nonlinear problems and ordinary differential
        equations.
  - [DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/stable/)
    
      + High level package for scientific machine learning applications, such as
        neural and universal differential equations, solving of inverse problems,
        parameter estimation, nonlinear optimal control, and more.
  - [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/)
    
      + Multi-package interface for data-driven modeling, Koopman dynamic mode
        decomposition, symbolic regression/sparsification, and automated model
        discovery.
  - [SciMLExpectations.jl](https://docs.sciml.ai/SciMLExpectations/stable/)
    
      + Extension to the dynamical modeling tools for calculating expectations.

## Interface Implementation Libraries

  - [SciMLBase.jl](https://docs.sciml.ai/SciMLBase/stable/)
    
      + The core package defining the interface which is consumed by the modeling
        and solver packages.

  - [DiffEqBase.jl](https://github.com/SciML/DiffEqBase.jl)
    
      + The core package defining the extended interface which is consumed by the
        differential equation solver packages.
  - [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/)
    
      + A package which pools together the definition of derivative overloads to
        define the common `sensealg` automatic differentiation interface.
  - [DiffEqNoiseProcess.jl](https://docs.sciml.ai/DiffEqNoiseProcess/stable/)
    
      + A package which defines the stochastic `AbstractNoiseProcess` interface
        for the SciML ecosystem.
  - [RecursiveArrayTools.jl](https://docs.sciml.ai/RecursiveArrayTools/stable/)
    
      + A package which defines the underlying `AbstractVectorOfArray` structure
        used as the output for all time series results.
  - [ArrayInterface.jl](https://docs.sciml.ai/ArrayInterface/stable/)
    
      + The package which defines the extended `AbstractArray` interface employed
        throughout the SciML ecosystem.

## Using-Facing Modeling Libraries

There are too many to name here and this will be populated when there is time!

## Flowchart Example for PDE-Constrained Optimal Control

The following example showcases how the pieces of the common interface connect to solve a problem
that mixes inference, symbolics, and numerics.

![](https://user-images.githubusercontent.com/1814174/126318252-1e4152df-e6e2-42a3-8669-f8608f81a095.png)

## External Binding Libraries

  - [diffeqr](https://github.com/SciML/diffeqr)
    
      + Solving differential equations in R using DifferentialEquations.jl with ModelingToolkit for JIT compilation and GPU-acceleration

  - [diffeqpy](https://github.com/SciML/diffeqpy)
    
      + Solving differential equations in Python using DifferentialEquations.jl

## Solver Libraries

There are too many to name here. Check out the
[SciML Organization Github Page](https://github.com/SciML) for details.

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
