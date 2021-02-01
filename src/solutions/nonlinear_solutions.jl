"""
$(TYPEDEF)
"""
struct NonlinearSolution{T,N,uType,R,P,A,O} <: AbstractNonlinearSolution{T,N}
  u::uType
  resid::R
  prob::P
  alg::A
  retcode::Symbol
  original::O
  left::uType
  right::uType
end

const SteadyStateSolution = NonlinearSolution

function build_solution(prob::AbstractNonlinearProblem,
                        alg,u,resid;calculate_error = true,
                        retcode = :Default,
                        original = nothing,
                        left = nothing, 
                        right = nothing,
                        kwargs...)

  T = eltype(eltype(u))
  N = length((size(prob.u0)...,))

  NonlinearSolution{T,N,typeof(u),typeof(resid),
                    typeof(prob),typeof(alg),typeof(original)}(
                    u,resid,prob,alg,retcode,original,left,right)
end

function sensitivity_solution(sol::AbstractNonlinearProblem,u)
  T = eltype(eltype(u))
  N = length((size(sol.prob.u0)...,))

  NonlinearSolution{T,N,typeof(u),typeof(sol.resid),
                    typeof(sol.prob),typeof(sol.alg),
                    typeof(sol.original)}(
                    u,sol.resid,sol.prob,sol.alg,sol.retcode,sol.original)
end
