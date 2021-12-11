"""
$(TYPEDEF)
"""
struct LinearSolution{T,N,uType,R,A} <: AbstractLinearSolution{T,N}
  u::uType
  resid::R
  alg::A
  retcode::Symbol
end

function build_linear_solution(alg,u,resid;
                               retcode = :Default)
  T = eltype(eltype(u))
  N = length((size(u)...,))
  LinearSolution{T,N,typeof(u),typeof(resid),typeof(alg)}(u,resid,alg,retcode)
end

struct QuadratureSolution{T,N,uType,R,P,A,C} <: AbstractQuadratureSolution{T,N}
  u::uType
  resid::R
  prob::P
  alg::A
  retcode::Symbol
  chi::C
end

function build_solution(prob::AbstractQuadratureProblem,
                        alg,u,resid;calculate_error = true,
                        chi = nothing,
                        retcode = :Default, kwargs...)

  T = eltype(eltype(u))
  N = length((size(u)...,))

  QuadratureSolution{T,N,typeof(u),typeof(resid),
                     typeof(prob),typeof(alg),typeof(chi)}(
                     u,resid,prob,alg,retcode,chi)
end
