# SciMLAlgorithms

## Definition of the AbstractSciMLAlgorithm Interface

`SciMLAlgorithms` are defined as types which have dispatches to the function signature:

```julia
CommonSolve.solve(prob::AbstractSciMLProblem, alg::AbstractSciMLAlgorithm; kwargs...)
```

### Algorithm-Specific Arguments

Note that because the keyword arguments of `solve` are designed to be common across the whole
problem type, algorithms should have the algorithm-specific keyword arguments defined as part
of the algorithm constructor. For example, `Rodas5` has a choice of `autodiff::Bool` which is
not common across all ODE solvers, and thus `autodiff` is an algorithm-specific keyword argument
handled via `Rodas5(autodiff=true)`.

### Remake

`remake` is applicable to `AbstractSciMLAlgorithm` values and lets solver
packages replace constructor fields while preserving the concrete algorithm
type. This is useful for internal transformations such as changing an automatic
differentiation chunk size. User code should normally construct the desired
algorithm directly because supported replacement fields are defined by each
concrete algorithm.

## Common Algorithm Keyword Arguments

An algorithm constructor stores choices that are specific to that numerical
method, such as an automatic differentiation backend, linear solver,
preconditioner, stage limiter, or method variant. Options shared across methods
for a problem family belong to `solve` and `init`: saving, tolerances, step-size
control, callbacks, progress, initialization, and RNG handling use the
[common keyword interface](@ref common_solver_keywords).

Concrete algorithms must document constructor fields, supported common
keywords, and any keyword whose meaning or default differs from the family
contract. Solver code should reject or diagnose unsupported common keywords
rather than silently treating an allow-listed name as proof of capability.

### Compatibility Diagnostics

```@docs
SciMLBase.check_keywords
SciMLBase.warn_compat
```

## Traits

```@docs
SciMLBase.isautodifferentiable
SciMLBase.allows_arbitrary_number_types
SciMLBase.allowscomplex
SciMLBase.isadaptive
SciMLBase.isdiscrete
SciMLBase.forwarddiffs_model
SciMLBase.forwarddiffs_model_time
SciMLBase.forwarddiff_chunksize
SciMLBase.has_lazy_interpolation
SciMLBase.allows_late_binding_tstops
SciMLBase.supports_opt_cache_interface
SciMLBase.has_init
SciMLBase.has_step
SciMLBase.supports_solve_rng
SciMLBase.alg_order
SciMLBase.allowsbounds
SciMLBase.requiresbounds
SciMLBase.allowsconstraints
SciMLBase.requiresconstraints
SciMLBase.requiresgradient
SciMLBase.allowsfg
SciMLBase.requireshessian
SciMLBase.allowsfgh
SciMLBase.requiresconsjac
SciMLBase.allowsconsjvp
SciMLBase.allowsconsvjp
SciMLBase.requiresconshess
SciMLBase.requireslagh
SciMLBase.allowscallback
SciMLBase.allows_non_wiener_noise
SciMLBase.requires_additive_noise
SciMLBase.AlgorithmInterpretation
SciMLBase.AlgorithmInterpretation.Ito
SciMLBase.AlgorithmInterpretation.Stratonovich
SciMLBase.alg_interpretation
```

### Abstract SciML Algorithms

```@docs
SciMLBase.AbstractSciMLAlgorithm
SciMLBase.AbstractDEAlgorithm
SciMLBase.AbstractLinearAlgorithm
SciMLBase.AbstractNonlinearAlgorithm
SciMLBase.AbstractIntervalNonlinearAlgorithm
SciMLBase.AbstractIntegralAlgorithm
SciMLBase.AbstractOptimizationAlgorithm
SciMLBase.AbstractSteadyStateAlgorithm
SciMLBase.AbstractBVPAlgorithm
SciMLBase.AbstractODEAlgorithm
SciMLBase.AbstractSecondOrderODEAlgorithm
SciMLBase.AbstractRODEAlgorithm
SciMLBase.AbstractSDEAlgorithm
SciMLBase.AbstractDAEAlgorithm
SciMLBase.AbstractDDEAlgorithm
SciMLBase.AbstractSDDEAlgorithm
SciMLBase.EnsembleAlgorithm
SciMLBase.DAEInitializationAlgorithm
SciMLBase.AbstractDiscretization
SciMLBase.AbstractDiscretizationMetadata
```

### DAE Initialization Algorithms

```@docs
SciMLBase.NoInit
SciMLBase.CheckInit
SciMLBase.OverrideInit
```
