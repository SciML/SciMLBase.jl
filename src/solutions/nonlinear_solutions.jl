"""
$(TYPEDEF)

Representation of the solution to a nonlinear equation defined by a NonlinearProblem,
or the steady state solution to a differential equation defined by a SteadyStateProblem.

## Fields

- `u`: the representation of the nonlinear equation's solution.
- `resid`: the residual of the solution.
- `prob`: the original NonlinearProblem/SteadyStateProblem that was solved.
- `alg`: the algorithm type used by the solver.
- `original`: if the solver is wrapped from an alternative solver ecosystem, such as
  NLsolve.jl, then this is the original return from said solver library.
- `retcode`: the return code from the solver. Used to determine whether the solver solved
  successfully or whether it exited due to an error. For more details, see
  [the return code documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes).
- `left`: if the solver is bracketing method, this is the final left bracket value.
- `right`: if the solver is bracketing method, this is the final right bracket value.
- `stats`: statistics of the solver, such as the number of function evaluations required.
"""
struct NonlinearSolution{T, N, uType, R, P, A, O, uType2, S} <:
       AbstractNonlinearSolution{T, N}
    u::uType
    resid::R
    prob::P
    alg::A
    retcode::ReturnCode.T
    original::O
    left::uType2
    right::uType2
    stats::S
end

function Base.show(io::IO,
                   t::Type{<:NonlinearSolution{T, N}}) where {T, N}
    if TruncatedStacktraces.VERBOSE[]
      invoke(show, Tuple{IO, Type}, io, t)
    else
        print(io, "NonlinearSolution{$T,$N,…}")
    end
end

const SteadyStateSolution = NonlinearSolution

function build_solution(prob::AbstractNonlinearProblem,
                        alg, u, resid; calculate_error = true,
                        retcode = ReturnCode.Default,
                        original = nothing,
                        left = nothing,
                        right = nothing,
                        stats = nothing,
                        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    NonlinearSolution{T, N, typeof(u), typeof(resid), typeof(prob), typeof(alg),
                      typeof(original), typeof(left), typeof(stats)}(u, resid, prob, alg,
                                                                     retcode, original,
                                                                     left, right, stats)
end

function sensitivity_solution(sol::AbstractNonlinearSolution, u)
    T = eltype(eltype(u))
    N = ndims(u)

    NonlinearSolution{T, N, typeof(u), typeof(sol.resid), typeof(sol.prob),
                      typeof(sol.alg), typeof(sol.original), typeof(sol.left),
                      typeof(sol.stats)}(u, sol.resid, sol.prob, sol.alg, sol.retcode,
                                         sol.original, sol.left, sol.right, sol.stats)
end
