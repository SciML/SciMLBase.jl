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
    has_lazy_interpolation(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm computes the solution interpolation lazily.

Defaults to false.
"""
has_lazy_interpolation(alg::AbstractDEAlgorithm) = false
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
    requiresgradient(opt)

Trait declaration for whether an optimizer
requires gradient in `instantiate_function`.

Defaults to false.
"""
requiresgradient(opt) = false

"""
    allowsfg(opt)

Trait declaration for whether an optimizer
allows combined function and gradient evaluation
in `instantiate_function`.

Defaults to false.
"""
allowsfg(opt) = false

"""
    requireshessian(opt)

Trait declaration for whether an optimizer
requires hessian in `instantiate_function`.

Defaults to false.
"""
requireshessian(opt) = false

"""
    allowsfgh(opt)

Trait declaration for whether an optimizer
allows combined function, gradient, and hessian
evaluation in `instantiate_function`.

Defaults to false.
"""
allowsfgh(opt) = false

"""
    requiresconsjac(opt)

Trait declaration for whether an optimizer
requires `cons_j` in `instantiate_function`, that is,
does the optimizer require a constraints' Jacobian.

Defaults to false.
"""
requiresconsjac(opt) = false

"""
    allowsconsjvp(opt)

Trait declaration for whether an optimizer
allows  constraint's jacobian vector product
in `instantiate_function`.

Defaults to false.
"""
allowsconsjvp(opt) = false

"""
    allowsconsvjp(opt)

Trait declaration for whether an optimizer
allows  constraint's vector jacobian product
in `instantiate_function`.

Defaults to false.
"""
allowsconsvjp(opt) = false

"""
    requiresconshess(opt)

Trait declaration for whether an optimizer
requires cons_h in `instantiate_function`, that is,
does the optimizer require constraints' hessian.

Defaults to false.
"""
requiresconshess(opt) = false

"""
    requireslagh(opt)

Trait declaration for whether an optimizer
requires lag_h in `instantiate_function`, that is,
does the optimizer require lagrangian hessian.

Defaults to false.
"""
requireslagh(opt) = false

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

The theoretic convergence order of the algorithm. If the method is adaptive order, this is treated
as the maximum order of the algorithm.
"""
function alg_order(alg::AbstractODEAlgorithm)
    error("Order is not defined for this algorithm")
end

"""
    allows_non_wiener_noise(alg::AbstractSDEAlgorithm)

Trait declaration for whether an algorithm allows for non-wiener noise.
In general, this is false for any high order (that uses levy areas) or adaptive method.

Defaults to false.
"""
allows_non_wiener_noise(alg::AbstractSDEAlgorithm) = false

"""
    requires_additive_noise(alg::AbstractSDEAlgorithm)

Trait declaration for whether an algorithm requires additive noise, i.e. the noise
function is not a function of `u`.

Defaults to false
"""
requires_additive_noise(alg::AbstractSDEAlgorithm) = false

EnumX.@enumx AlgorithmInterpretation Ito Stratonovich

"""
    alg_interpretation(alg)

Integral interpolation for the SDE solver algorithm. SDEs solutions depend on the chosen definition of the stochastic integral. In the Ito calculus,
the left-hand rule is taken, while Stratonovich calculus uses the right-hand rule. Unlike in standard Riemannian integration, these integral rules do
not converge to the same answer. In the context of a stochastic differential equation, the underlying solution (and its mean, variance, etc.) is dependent
on the integral rule that is chosen. This trait describes which interpretation the solver algorithm subscribes to, and thus whether the solution should
be interpreted as the solution to the SDE under the Ito or Stratonovich interpretation.

For more information, see https://oatml.cs.ox.ac.uk/blog/2022/03/22/ito-strat.html as a good high-level explanation.

!!! note

    The expected solution statistics are dependent on this output. Solutions from solvers with different
    interpretations are expected to have different answers on almost all SDEs without additive noise.
"""
function alg_interpretation(alg::AbstractSciMLAlgorithm)
    error("Algorithm interpretation is not defined for this algorithm. It can be either `AlgorithmInterpretation.Ito` or `AlgorithmInterpretation.Stratonovich`")
end
