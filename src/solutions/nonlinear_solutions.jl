
"""
$(TYPEDEF)
Statistics from the nonlinear equation solver about the solution process.
## Fields
- nf: Number of function evaluations.
- njacs: Number of Jacobians created during the solve.
- nfactors: Number of factorzations of the jacobian required for the solve.
- nsolve: Number of linear solves `W\b` required for the solve.
- nsteps: Total number of iterations for the nonlinear solver.
"""
mutable struct NLStats
    nf::Int
    njacs::Int
    nfactors::Int
    nsolve::Int
    nsteps::Int
end

function Base.show(io::IO, ::MIME"text/plain", s::NLStats)
    println(io, summary(s))
    @printf io "%-50s %-d\n" "Number of function evaluations:" s.nf
    @printf io "%-50s %-d\n" "Number of Jacobians created:" s.njacs
    @printf io "%-50s %-d\n" "Number of factorizations:" s.nfactors
    @printf io "%-50s %-d\n" "Number of linear solves:" s.nsolve
    @printf io "%-50s %-d" "Number of nonlinear solver iterations:" s.nsteps
end

function Base.merge(s1::NLStats, s2::NLStats)
    NLStats(s1.nf + s2.nf, s1.njacs + s2.njacs, s1.nfactors + s2.nfactors,
        s1.nsolve + s2.nsolve, s1.nsteps + s2.nsteps)
end

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
struct NonlinearSolution{T, N, uType, R, P, A, O, uType2, S, Tr} <:
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
    trace::Tr
end

TruncatedStacktraces.@truncate_stacktrace NonlinearSolution 1 2

const SteadyStateSolution = NonlinearSolution

get_p(p::AbstractNonlinearSolution) = p.prob.p

function build_solution(prob::AbstractNonlinearProblem,
        alg, u, resid; calculate_error = true,
        retcode = ReturnCode.Default,
        original = nothing,
        left = nothing,
        right = nothing,
        stats = nothing,
        trace = nothing,
        kwargs...)
    T = eltype(eltype(u))
    N = ndims(u)

    NonlinearSolution{T, N, typeof(u), typeof(resid), typeof(prob), typeof(alg),
        typeof(original), typeof(left), typeof(stats), typeof(trace)}(u, resid, prob, alg,
        retcode, original, left, right, stats, trace)
end

function sensitivity_solution(sol::AbstractNonlinearSolution, u)
    T = eltype(eltype(u))
    N = ndims(u)

    # Some of the subtypes might not have a trace field
    trace = hasfield(typeof(sol), :trace) ? sol.trace : nothing

    NonlinearSolution{T, N, typeof(u), typeof(sol.resid), typeof(sol.prob),
        typeof(sol.alg), typeof(sol.original), typeof(sol.left),
        typeof(sol.stats), typeof(trace)}(u, sol.resid, sol.prob, sol.alg, sol.retcode,
        sol.original, sol.left, sol.right, sol.stats, trace)
end
