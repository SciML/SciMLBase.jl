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
    forwarddiff_chunksize(alg::AbstractSciMLAlgorithm)

Trait declaration for the ForwardDiff chunk size used by the algorithm
when calling the model function with ForwardDiff.Dual numbers.

Returns a `Val{N}()`: `Val(0)` means unspecified (the framework will choose
a default, typically 1 for FunctionWrapper compatibility). `Val(N)` for any
positive integer `N` means the algorithm will use chunk size `N`.

This is used by DiffEqBase to compile FunctionWrapper variants with matching
Dual number chunk sizes, avoiding `NoFunctionWrapperFoundError`.

Defaults to `Val(0)` (unspecified).
"""
forwarddiff_chunksize(alg::AbstractSciMLAlgorithm) = Val(0)

"""
    allows_arbitrary_number_types(alg::AbstractDEAlgorithm)

Trait declaration for whether an algorithm supports state, parameter, and time
number types beyond the standard floating-point and complex floating-point
types.

Algorithms that return `true` should be implemented generically enough to work
with SciML-compatible number and container types, subject to the additional rules
in the [SciML container and number interface](@ref arrayandnumber). Wrapped
C/Fortran solvers usually cannot support this and should keep the default.

Defaults to `false`.
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

Trait declaring whether an optimization algorithm supports finite box bounds.

Return `true` when the solver can use the `lb` and `ub` fields of an
`OptimizationProblem`. Return `false` when the solver does not accept bound
constraints and callers should reject or ignore nontrivial bounds before
dispatching to the solver. This trait describes support for variable bounds,
not nonlinear constraints from `OptimizationFunction.cons`.

The default is `false`.
"""
allowsbounds(opt) = false

"""
    requiresbounds(opt)

Trait declaring whether an optimization algorithm requires box bounds.

Return `true` when the solver interface is only valid for problems that provide
`lb` and `ub` bounds. Algorithms that merely support optional bounds should
override [`allowsbounds`](@ref) and keep this trait `false`.

The default is `false`.
"""
requiresbounds(opt) = false

"""
    allowsconstraints(opt)

Trait declaring whether an optimization algorithm supports nonlinear constraints.

Return `true` when the solver can use the `cons` callback stored in an
`OptimizationFunction` together with `lcons` and `ucons` from the
`OptimizationProblem`. This trait does not describe box-bound support; use
[`allowsbounds`](@ref) for `lb` and `ub`.

The default is `false`.
"""
allowsconstraints(opt) = false

"""
    requiresconstraints(opt)

Trait declaring whether an optimization algorithm requires nonlinear constraints.

Return `true` when the solver interface is meaningful only for constrained
problems with an `OptimizationFunction.cons` callback and matching constraint
bounds. Algorithms that support both constrained and unconstrained problems
should return `false`.

The default is `false`.
"""
requiresconstraints(opt) = false

"""
    requiresgradient(opt)

Trait declaring whether an optimization algorithm requires objective gradients.

Return `true` when solver setup must obtain a gradient callback for the
objective, either supplied manually on `OptimizationFunction` or generated by
the selected AD backend during function instantiation. Algorithms that can run
with objective values only should keep the default.

The default is `false`.
"""
requiresgradient(opt) = false

"""
    allowsfg(opt)

Trait declaring whether an optimization algorithm can use combined objective and
gradient evaluation.

Return `true` when the solver can consume a callback that computes the objective
value and gradient in one evaluation. Function-instantiation code can use this
to preserve or generate an efficient `fg`-style callback instead of separate
objective and gradient calls.

The default is `false`.
"""
allowsfg(opt) = false

"""
    requireshessian(opt)

Trait declaring whether an optimization algorithm requires an objective Hessian.

Return `true` when solver setup must obtain a Hessian callback for the
objective, either supplied manually on `OptimizationFunction` or generated by
the selected AD backend. Algorithms that can run without second derivatives
should keep the default.

The default is `false`.
"""
requireshessian(opt) = false

"""
    allowsfgh(opt)

Trait declaring whether an optimization algorithm can use combined objective,
gradient, and Hessian evaluation.

Return `true` when the solver can consume a callback that computes the objective
value, gradient, and Hessian in one evaluation. Function-instantiation code can
use this to preserve or generate an efficient `fgh`-style callback.

The default is `false`.
"""
allowsfgh(opt) = false

"""
    requiresconsjac(opt)

Trait declaring whether an optimization algorithm requires a constraint
Jacobian.

Return `true` when constrained solver setup must obtain `cons_j`, the Jacobian
of `OptimizationFunction.cons` with respect to the optimization state. The
callback may be supplied manually or generated by an AD backend during function
instantiation.

The default is `false`.
"""
requiresconsjac(opt) = false

"""
    allowsconsjvp(opt)

Trait declaring whether an optimization algorithm can use constraint
Jacobian-vector products.

Return `true` when the solver can consume `cons_jvp`, a callback applying the
constraint Jacobian to a vector without materializing the full Jacobian.
Algorithms that require the full constraint Jacobian should use
[`requiresconsjac`](@ref) instead.

The default is `false`.
"""
allowsconsjvp(opt) = false

"""
    allowsconsvjp(opt)

Trait declaring whether an optimization algorithm can use constraint
vector-Jacobian products.

Return `true` when the solver can consume `cons_vjp`, a callback applying the
adjoint action of the constraint Jacobian without materializing the full
Jacobian.

The default is `false`.
"""
allowsconsvjp(opt) = false

"""
    requiresconshess(opt)

Trait declaring whether an optimization algorithm requires constraint Hessians.

Return `true` when constrained solver setup must obtain `cons_h`, the Hessian
information for the nonlinear constraints in `OptimizationFunction.cons`.
Algorithms that instead use the Hessian of the Lagrangian should override
[`requireslagh`](@ref).

The default is `false`.
"""
requiresconshess(opt) = false

"""
    requireslagh(opt)

Trait declaring whether an optimization algorithm requires a Lagrangian Hessian.

Return `true` when solver setup must obtain `lag_h`, the Hessian of the
Lagrangian combining the objective, constraint multipliers, and any solver
scaling arguments expected by the backend. This is distinct from requiring
separate objective and constraint Hessians.

The default is `false`.
"""
requireslagh(opt) = false

"""
    allowscallback(opt)

Trait declaring whether an optimization algorithm supports solve callbacks.

Return `true` when the solver can accept the `callback` keyword for an
`OptimizationProblem` solve. Return `false` for solver backends where callbacks
cannot be represented or would be silently ignored.

The default is `true`.
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

"""
    AlgorithmInterpretation

Enum of stochastic integral interpretations used by SDE algorithms. The values are
`AlgorithmInterpretation.Ito` and `AlgorithmInterpretation.Stratonovich`.
"""
EnumX.@enumx AlgorithmInterpretation Ito Stratonovich

"""
    alg_interpretation(alg)

Integral interpolation for the SDE solver algorithm. SDEs solutions depend on the chosen definition of the stochastic integral. In the Ito calculus,
the left-hand rule is taken, while Stratonovich calculus uses the right-hand rule. Unlike in standard Riemannian integration, these integral rules do
not converge to the same answer. In the context of a stochastic differential equation, the underlying solution (and its mean, variance, etc.) is dependent
on the integral rule that is chosen. This trait describes which interpretation the solver algorithm subscribes to, and thus whether the solution should
be interpreted as the solution to the SDE under the Ito or Stratonovich interpretation.

For more information, see <https://oatml.cs.ox.ac.uk/blog/2022/03/22/ito-strat.html> as a good high-level explanation.

!!! note

    The expected solution statistics are dependent on this output. Solutions from solvers with different
    interpretations are expected to have different answers on almost all SDEs without additive noise.
"""
function alg_interpretation(alg::AbstractSciMLAlgorithm)
    error("Algorithm interpretation is not defined for this algorithm. It can be either `AlgorithmInterpretation.Ito` or `AlgorithmInterpretation.Stratonovich`")
end

"""
    $(TYPEDSIGNATURES)

Trait declaration for whether an algorithm supports specifying `tstops` as a function `tstops(p, tspan)` to be called after
initialization.

Defaults to false.
"""
allows_late_binding_tstops(alg::AbstractODEAlgorithm) = false

"""
    $(TYPEDSIGNATURES)

Trait declaration for whether the optimization algorithm supports the `init` interface.

Deprecated as this is not an optimization-specific idea and should use the traits for general caching.
"""
supports_opt_cache_interface(alg) = false

"""
    $(TYPEDSIGNATURES)

Trait declaring whether `alg` supports the caching/iterator interface through
`init(prob, alg; kwargs...)`.

Algorithms that return `true` should provide an `init`/`__init` path that
constructs an object which can later be advanced or finished with `solve!`.
Returning `false` means users should call `solve` directly, or that the package
has not exposed a reusable cache for this algorithm. The default is `false`.
"""
has_init(a) = false

"""
    $(TYPEDSIGNATURES)

Trait declaring whether an initialized object for `alg` supports direct
advancement through `step!`.

Algorithms that return `true` should have an `init` path whose returned iterator
or integrator can be advanced by `step!`. This is stronger than supporting
`solve!`: a cached solver may be finishable with `solve!` without exposing
manual stepping. See the init/solve interface documentation for the solver-side
contract. The default is `false`.
"""
has_step(a) = false

"""
    supports_solve_rng(prob, alg) -> Bool

Whether `solve(prob, alg; rng=...)` is supported for this problem/algorithm path.

Pass `alg = nothing` to query support for the default solver-selection path
(i.e., `solve(prob; rng=...)`).
"""
supports_solve_rng(::AbstractSciMLProblem, alg) = false
supports_solve_rng(prob, alg) = false  # fallback for non-problem types (e.g., Vector of problems)
