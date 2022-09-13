"""
$(TYPEDEF)

Representation of the solution to an linear system Ax=b defined by a LinearProblem

## Fields

- `u`: the representation of the optimization's solution.
- `resid`: the residual of the solver, if the method is an iterative method.
- `alg`: the algorithm type used by the solver.
- `iters`: the number of iterations used to solve the equation, if the method is an iterative
  method.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === :Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
- `cache`: the `LinearCache` object containing the solver's internal cached variables. This
  is given to allow continuation of solver usage, for example, solving `Ax=b` with the same
  `A` and a new `b` without refactorizing `A`. See the caching interface tutorial for details
  on how to use the `cache` effectively: http://linearsolve.sciml.ai/dev/tutorials/caching_interface/
"""
struct LinearSolution{T, N, uType, R, A, C} <: AbstractLinearSolution{T, N}
    u::uType
    resid::R
    alg::A
    retcode::Symbol
    iters::Int
    cache::C
end

function build_linear_solution(alg, u, resid, cache;
                               retcode = :Default,
                               iters = 0)
    T = eltype(eltype(u))
    N = length((size(u)...,))
    LinearSolution{T, N, typeof(u), typeof(resid), typeof(alg), typeof(cache)}(u, resid,
                                                                               alg, retcode,
                                                                               iters, cache)
end

"""
$(TYPEDEF)

Representation of the solution to an quadrature integral_lb^ub f(x) dx defined by a IntegralProblem

## Fields

- `u`: the representation of the optimization's solution.
- `resid`: the residual of the solver.
- `alg`: the algorithm type used by the solver.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully (`sol.retcode === :Success`), whether it terminated due to a user-defined
  callback (`sol.retcode === :Terminated`), or whether it exited due to an error. For more
  details, see the return code section of the DifferentialEquations.jl documentation.
- `chi`: the variance estimate of the estimator from Monte Carlo quadrature methods.
"""
struct IntegralSolution{T, N, uType, R, P, A, C} <: AbstractIntegralSolution{T, N}
    u::uType
    resid::R
    prob::P
    alg::A
    retcode::Symbol
    chi::C
end

struct QuadratureSolution end
@deprecate QuadratureSolution(args...; kwargs...) IntegralSolution(args...; kwargs...)

function build_solution(prob::AbstractIntegralProblem,
                        alg, u, resid; calculate_error = true,
                        chi = nothing,
                        retcode = :Default, kwargs...)
    T = eltype(eltype(u))
    N = length((size(u)...,))

    IntegralSolution{T, N, typeof(u), typeof(resid),
                     typeof(prob), typeof(alg), typeof(chi)}(u, resid, prob, alg, retcode,
                                                             chi)
end

# Test comment please ignore
