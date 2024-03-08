"""
isautodifferentiable(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
isautodifferentiable(alg::AbstractSciMLAlgorithm) = false

"""
forwarddiffs_model(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses ForwardDiff.jl
on the model function is called with ForwardDiff.jl

Defaults to false as only pure-Julia algorithms can have this be true.
"""
forwarddiffs_model(alg::AbstractSciMLAlgorithm) = false

"""
forwarddiffs_model_time(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses ForwardDiff.jl
on the model `f(u,p,t)` function is called with ForwardDiff.jl on the `t` argument.

Defaults to false as only a few pure-Julia algorithms (Rosenbrock methods)
have this as true
"""
forwarddiffs_model_time(alg::AbstractSciMLAlgorithm) = false

"""
allows_arbitrary_number_types(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
direct automatic differentiation, i.e. can have algorithms like
ForwardDiff or ReverseDiff attempt to differentiate directly
through the solver.

Defaults to false as only pure-Julia algorithms can have this be true.
"""
allows_arbitrary_number_types(alg::AbstractSciMLAlgorithm) = false

"""
allowscomplex(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm is compatible with
having complex numbers as the state variables.

Defaults to false.
"""
allowscomplex(alg::AbstractSciMLAlgorithm) = false

"""
isadaptive(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm uses adaptivity,
i.e. has a non-quasi-static compute graph.

Defaults to true.
"""
isadaptive(alg::AbstractDEAlgorithm) = true
# Default to assuming adaptive, safer error("Adaptivity algorithm trait not set.")

"""
isdiscrete(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm allows for
discrete state values, such as integers.

Defaults to false.
"""
isdiscrete(alg::AbstractDEAlgorithm) = false

"""
allowsbounds(opt)

Trait declaration for whether an optimizer allows for
box constraints passed with `lb` and `ub` in
`OptimizationProblem`.

Defaults to false.
"""
allowsbounds(opt) = false

"""
requiresbounds(opt)

Trait declaration for whether an optimizer requires
box constraints passed with `lb` and `ub` in
`OptimizationProblem`.

Defaults to false.
"""
requiresbounds(opt) = false

"""
allowsconstraints(opt)

Trait declaration for whether an optimizer allows
non-linear constraints specified in `cons` in
`OptimizationFunction`.

Defaults to false.
"""
allowsconstraints(opt) = false

"""
requiresconstraints(opt)

Trait declaration for whether an optimizer
requires non-linear constraints specified in
`cons` in `OptimizationFunction`.

Defaults to false.
"""
requiresconstraints(opt) = false

"""
requiresderivative(opt)

Trait declaration for whether an optimizer
requires derivatives specified in
`cons` in `OptimizationProblem`.

Defaults to false.
"""
requiresderivative(opt) = false

"""
requiresgradient(opt)

Trait declaration for whether an optimizer
requires gradient in `instantiate_function`.

Defaults to false.
"""
requiresgradient(opt) = false

"""
requireshessian(opt)

Trait declaration for whether an optimizer
requires hessian in `instantiate_function`.

Defaults to false.
"""
requireshessian(opt) = false

"""
requiresconsjac(opt)

Trait declaration for whether an optimizer
requires cons_j in `instantiate_function`, that is, does the optimizer require a constant Jacobian.

Defaults to false.
"""
requiresconsjac(opt) = false

"""
requiresconshess(opt)

Trait declaration for whether an optimizer
requires cons_h in `instantiate_function`, that is, does the optimizer require a constant hessian.

Defaults to false.
"""
requiresconshess(opt) = false

"""
allowscallback(opt)

Trait declaration for whether an optimizer
supports passing a `callback` to `solve`
for an `OptimizationProblem`.

Defaults to true.
"""
allowscallback(opt) = true

"""
alg_order(alg)

The theoretic convergence order of the algorithm.
"""
function alg_order(alg::AbstractODEAlgorithm)
    error("Order is not defined for this algorithm")
end
