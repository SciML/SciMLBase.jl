"""
$(TYPEDEF)
"""
struct NonlinearSolution{T,N,uType,R,P,A,O,uType2} <: AbstractNonlinearSolution{T,N}
  u::uType
  resid::R
  prob::P
  alg::A
  retcode::Symbol
  original::O
  left::uType2
  right::uType2
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
                    typeof(prob),typeof(alg),typeof(original),typeof(left)}(
                    u,resid,prob,alg,retcode,original,left,right)
end

function sensitivity_solution(sol::AbstractNonlinearSolution,u)
  T = eltype(eltype(u))
  N = length((size(sol.prob.u0)...,))

  NonlinearSolution{T,N,typeof(u),typeof(sol.resid),
                    typeof(sol.prob),typeof(sol.alg),
                    typeof(sol.original),typeof(sol.left)}(
                    u,sol.resid,sol.prob,sol.alg,sol.retcode,sol.original,sol.left,sol.right)
end
