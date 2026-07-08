"""
$(TYPEDEF)

Representation of the solution to an linear system `Ax=b` defined by a `LinearProblem`

## Fields

  - `u`: the representation of the linear solve's solution.
  - `resid`: the residual of the solver, if the method is an iterative method. Returns a mutable
    type, thus if scalar it is wrapped in a `Ref`.
  - `alg`: the algorithm type used by the solver.
  - `iters`: the number of iterations used to solve the equation, if the method is an iterative
    method.
  - `retcode`: the return code from the solver. Used to determine whether the solver solved
    successfully or whether it exited due to an error. For more details, see
    [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
  - `cache`: the `LinearCache` object containing the solver's internal cached variables. This
    is given to allow continuation of solver usage, for example, solving `Ax=b` with the same
    `A` and a new `b` without refactorizing `A`. See the caching interface tutorial for details
    on how to use the `cache` effectively: <https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface>
  - `stats`: statistics of the solver, such as the number of function evaluations required.
"""
struct LinearSolution{T, N, uType, R, A, C, S} <: AbstractLinearSolution{T, N}
    u::uType
    resid::R
    alg::A
    retcode::ReturnCode.T
    iters::Int
    cache::C
    stats::S
end

function build_linear_solution(
        alg, u, resid, cache;
        retcode = ReturnCode.Default,
        iters = 0, stats = nothing
    )
    T = eltype(eltype(u))
    N = length((size(u)...,))
    return LinearSolution{
        T, N, typeof(u), typeof(resid), typeof(alg), typeof(cache),
        typeof(stats),
    }(
        u,
        resid,
        alg,
        retcode,
        iters,
        cache,
        stats
    )
end

"""
$(TYPEDEF)

Representation of the solution to an eigenvalue problem defined by an `EigenvalueProblem`.

## Fields

  - `u`: the computed eigenvalues.
  - `vectors`: the corresponding eigenvectors, stored as the columns of a matrix.
  - `prob`: the `EigenvalueProblem` that was solved.
  - `alg`: the algorithm type used by the solver.
  - `retcode`: the return code from the solver. Used to determine whether the solver solved
    successfully or whether it exited due to an error. For more details, see
    [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
  - `resid`: the residual(s) of the computed eigenpairs, if provided by the solver.
  - `stats`: statistics of the solver, if provided.
"""
struct EigenvalueSolution{T, N, U, V, P, A, R, S} <: AbstractEigenvalueSolution{T, N}
    u::U
    vectors::V
    prob::P
    alg::A
    retcode::ReturnCode.T
    resid::R
    stats::S
end

function build_eigenvalue_solution(
        prob, alg, values, vectors;
        retcode = ReturnCode.Success, resid = nothing, stats = nothing
    )
    T = eltype(eltype(values))
    N = length((size(values)...,))
    return EigenvalueSolution{
        T, N, typeof(values), typeof(vectors), typeof(prob), typeof(alg),
        typeof(resid), typeof(stats),
    }(values, vectors, prob, alg, retcode, resid, stats)
end

"""
$(TYPEDEF)

Representation of the solution to an quadrature integral_lb^ub f(x) dx defined by a `IntegralProblem`

## Fields

  - `u`: the representation of the optimization's solution.
  - `resid`: the residual of the solver.
  - `alg`: the algorithm type used by the solver.
  - `retcode`: the return code from the solver. Used to determine whether the solver solved
    successfully or whether it exited due to an error. For more details, see
    [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
  - `chi`: the variance estimate of the estimator from Monte Carlo quadrature methods.
  - `stats`: statistics of the solver, such as the number of function evaluations required.
"""
struct IntegralSolution{T, N, uType, R, P, A, C, S} <: AbstractIntegralSolution{T, N}
    u::uType
    resid::R
    prob::P
    alg::A
    retcode::ReturnCode.T
    chi::C
    stats::S
end

struct QuadratureSolution end
@deprecate QuadratureSolution(args...; kwargs...) IntegralSolution(args...; kwargs...)

"""
    build_solution(prob, alg, args...; kwargs...)

Construct the solution object returned by a SciML solver for `prob` solved with `alg`.
Solver implementations use this helper to attach standard fields such as the original
problem, algorithm, return code, statistics, dense interpolation, and saved values.
"""
function build_solution(
        prob::AbstractIntegralProblem,
        alg, u, resid; chi = nothing,
        retcode = ReturnCode.Default, stats = nothing, kwargs...
    )
    T = eltype(eltype(u))
    N = length((size(u)...,))

    return IntegralSolution{
        T, N, typeof(u), typeof(resid), typeof(prob), typeof(alg), typeof(chi),
        typeof(stats),
    }(
        u,
        resid,
        prob,
        alg,
        retcode,
        chi,
        stats
    )
end
@doc """
    build_solution(prob, alg, args...; kwargs...)

Construct the solution object returned by a SciML solver for `prob` solved with `alg`.
Solver implementations use this helper to attach standard fields such as the original
problem, algorithm, return code, statistics, dense interpolation, and saved values.
""" build_solution

function wrap_sol(sol)
    return if hasproperty(sol, :prob) && hasproperty(sol.prob, :problem_type)
        wrap_sol(sol, sol.prob.problem_type)
    else
        sol
    end
end

# Define a default `wrap_sol` that does nothing
wrap_sol(sol, _) = sol
