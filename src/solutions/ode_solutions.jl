"""
$(TYPEDEF)
"""
struct ODESolution{T,N,uType,uType2,DType,tType,rateType,P,A,IType,DE} <: AbstractODESolution{T,N,uType}
  u::uType
  u_analytic::uType2
  errors::DType
  t::tType
  k::rateType
  prob::P
  alg::A
  interp::IType
  dense::Bool
  tslocation::Int
  destats::DE
  retcode::Symbol
end
(sol::ODESolution)(t,deriv::Type=Val{0};idxs=nothing,continuity=:left) = sol(t,deriv,idxs,continuity)
(sol::ODESolution)(v,t,deriv::Type=Val{0};idxs=nothing,continuity=:left) = sol.interp(v,t,idxs,deriv,sol.prob.p,continuity)

function (sol::ODESolution)(t::Real,deriv,idxs::Nothing,continuity)
  sol.interp(t,idxs,deriv,sol.prob.p,continuity)
end

function (sol::ODESolution)(t::AbstractVector{<:Real},deriv,idxs::Nothing,continuity)
  augment(sol.interp(t,idxs,deriv,sol.prob.p,continuity), sol)
end

function (sol::ODESolution)(t::Real,deriv,idxs::Integer,continuity)
  sol.interp(t,idxs,deriv,sol.prob.p,continuity)
end
function (sol::ODESolution)(t::Real,deriv,idxs::AbstractVector{<:Integer},continuity)
  sol.interp(t,idxs,deriv,sol.prob.p,continuity)
end
function (sol::ODESolution)(t::AbstractVector{<:Real},deriv,idxs::Integer,continuity)
  A = sol.interp(t,idxs,deriv,sol.prob.p,continuity)
  syms = hasproperty(sol.prob.f, :syms) && sol.prob.f.syms !== nothing ? [sol.prob.f.syms[idxs]] : nothing
  observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  DiffEqArray(A.u, A.t, syms, getindepsym(sol),observed,p)
end
function (sol::ODESolution)(t::AbstractVector{<:Real},deriv,idxs::AbstractVector{<:Integer},continuity)
  A = sol.interp(t,idxs,deriv,sol.prob.p,continuity)
  syms = hasproperty(sol.prob.f, :syms) && sol.prob.f.syms !== nothing ? sol.prob.f.syms[idxs] : nothing
  observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  DiffEqArray(A.u, A.t, syms, getindepsym(sol),observed,p)
end

function (sol::ODESolution)(t::Real,deriv,idxs,continuity)
  issymbollike(idxs) || error("Incorrect specification of `idxs`")
  augment(sol.interp([t],nothing,deriv,sol.prob.p,continuity), sol)[idxs][1]
end

function (sol::ODESolution)(t::Real,deriv,idxs::AbstractVector,continuity)
  all(issymbollike.(idxs)) || error("Incorrect specification of `idxs`")
  interp_sol = augment(sol.interp([t],nothing,deriv,sol.prob.p,continuity), sol)
  [first(interp_sol[idx]) for idx in idxs]
end

function (sol::ODESolution)(t::AbstractVector{<:Real},deriv,idxs,continuity)
  issymbollike(idxs) || error("Incorrect specification of `idxs`")
  interp_sol = augment(sol.interp(t,nothing,deriv,sol.prob.p,continuity), sol)
  observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  DiffEqArray(interp_sol[idxs], t, [idxs], getindepsym(sol), observed, p)
end

function (sol::ODESolution)(t::AbstractVector{<:Real},deriv,idxs::AbstractVector,continuity)
  all(issymbollike.(idxs)) || error("Incorrect specification of `idxs`")
  interp_sol = augment(sol.interp(t,nothing,deriv,sol.prob.p,continuity), sol)
  observed = has_observed(sol.prob.f) ? sol.prob.f.observed : DEFAULT_OBSERVED
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  DiffEqArray([[interp_sol[idx][i] for idx in idxs] for i in 1:length(t)], t, idxs, getindepsym(sol), observed, p)
end

function residual(sol::AbstractODESolution, t)
  f = sol.prob.f
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  if isinplace(f)
    if f.cache === nothing
      # Actually I feel like this should be 'similar(sol.prob.f(du,u,p,t))', but
      # to call it, I need the type of du!
      cache = similar(sol.u)
      f(cache, sol(t), p, t)
      return sol(t,Val{1}) - cache
    else
      f(f.cache, sol(t), p, t)
      return sol(t,Val{1}) - cache
    end
  end
  u̇ = sol(t,Val{1})
  u = sol(t)
  u̇ - f(u, p, t)
end

function max_residual_estimate(sol::AbstractODESolution)
  f = sol.prob.f
  p = hasproperty(sol.prob, :p) ? sol.prob.p : nothing
  max_res = 0
  for i in 1:length(sol.t)
    tᵢ = sol.t[i]
    uᵢ = sol.u[i]
    u̇ᵢ = sol(tᵢ,Val{1})
    # For vector valued ODEs, does abs = sup-norm?
    # If not, we need something like abs_ = norm
    abs_res = abs(u̇ᵢ - f(uᵢ, p, tᵢ))
    if abs_res > max_res
      max_res = abs_res
    end
  end

  for i in 1:length(sol.t)-1
    tᵢ = sol.t[i]
    tᵢ₊₁ = sol.t[i+1]
    Δᵢ = tᵢ₊₁ - tᵢ
    t = tᵢ + Δᵢ/4
    u = sol(t)
    u̇ = sol(t,Val{1})
    abs_res = abs(u̇ - f(u, p, t))
    if abs_res > max_res
      max_res = abs_res
    end
    t = tᵢ + Δᵢ/2
    u = sol(t)
    u̇ = sol(t,Val{1})
    abs_res = abs(u̇ - f(u, p, t))
    if abs_res > max_res
      max_res = abs_res
    end
    t = tᵢ + 3Δᵢ/4
    u = sol(t)
    u̇ = sol(t,Val{1})
    abs_res = abs(u̇ - f(u, p, t))
    if abs_res > max_res
      max_res = abs_res
    end
  end
  max_res
end

function build_solution(
        prob::Union{AbstractODEProblem,AbstractDDEProblem},
        alg,t,u;timeseries_errors=length(u)>2,
        dense=false,dense_errors=dense,
        calculate_error = true,
        k=nothing,
        interp = LinearInterpolation(t,u),
        retcode = :Default, destats=nothing, kwargs...)

  T = eltype(eltype(u))
  N = length((size(prob.u0)..., length(u)))

  if typeof(prob.f) <: Tuple
    f = prob.f[1]
  else
    f = prob.f
  end

  if has_analytic(f)
    u_analytic = Vector{typeof(prob.u0)}()
    errors = Dict{Symbol,real(eltype(prob.u0))}()
    sol = ODESolution{T,N,typeof(u),typeof(u_analytic),typeof(errors),typeof(t),typeof(k),
                      typeof(prob),typeof(alg),typeof(interp),typeof(destats)}(u,u_analytic,
                       errors,t,k,prob,alg,interp,dense,0,destats,retcode)
    if calculate_error
      calculate_solution_errors!(sol;timeseries_errors=timeseries_errors,dense_errors=dense_errors)
    end
    return sol
  else
    return ODESolution{T,N,typeof(u),Nothing,Nothing,typeof(t),typeof(k),
                       typeof(prob),typeof(alg),typeof(interp),typeof(destats)}(u,nothing,nothing,
                       t,k,prob,alg,interp,dense,0,destats,retcode)
  end
end

function calculate_solution_errors!(sol::AbstractODESolution;fill_uanalytic=true,timeseries_errors=true,dense_errors=true)

  f = sol.prob.f

  if fill_uanalytic
    for i in 1:size(sol.u,1)
      if sol.prob isa AbstractDDEProblem
        push!(sol.u_analytic, f.analytic(sol.prob.u0, sol.prob.h, sol.prob.p, sol.t[i]))
      else
        push!(sol.u_analytic, f.analytic(sol.prob.u0, sol.prob.p, sol.t[i]))
      end
    end
  end

  save_everystep = length(sol.u) > 2
  if !isempty(sol.u_analytic)
    sol.errors[:final] = norm(recursive_mean(abs.(sol.u[end].-sol.u_analytic[end])))

    if save_everystep && timeseries_errors
      sol.errors[:l∞] = norm(maximum(vecvecapply((x)->abs.(x),sol.u-sol.u_analytic)))
      sol.errors[:l2] = norm(sqrt(recursive_mean(vecvecapply((x)->float.(x).^2,sol.u-sol.u_analytic))))
      if sol.dense && dense_errors
        densetimes = collect(range(sol.t[1], stop=sol.t[end], length=100))
        interp_u = sol(densetimes)
        interp_analytic = VectorOfArray([f.analytic(sol.prob.u0,sol.prob.p,t) for t in densetimes])
        sol.errors[:L∞] = norm(maximum(vecvecapply((x)->abs.(x),interp_u-interp_analytic)))
        sol.errors[:L2] = norm(sqrt(recursive_mean(vecvecapply((x)->float.(x).^2,interp_u-interp_analytic))))
      end
    end
  end
end

function build_solution(sol::AbstractODESolution{T,N},u_analytic,errors) where {T,N}
  ODESolution{T,N,typeof(sol.u),typeof(u_analytic),typeof(errors),typeof(sol.t),typeof(sol.k),
                     typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.destats)}(
                     sol.u,u_analytic,errors,sol.t,sol.k,sol.prob,
                     sol.alg,sol.interp,sol.dense,sol.tslocation,sol.destats,sol.retcode)
end

function solution_new_retcode(sol::AbstractODESolution{T,N},retcode) where {T,N}
  ODESolution{T,N,typeof(sol.u),typeof(sol.u_analytic),typeof(sol.errors),
                     typeof(sol.t),typeof(sol.k),
                     typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.destats)}(
                     sol.u,sol.u_analytic,sol.errors,sol.t,sol.k,sol.prob,
                     sol.alg,sol.interp,sol.dense,sol.tslocation,sol.destats,retcode)
 end

function solution_new_tslocation(sol::AbstractODESolution{T,N},tslocation) where {T,N}
  ODESolution{T,N,typeof(sol.u),typeof(sol.u_analytic),typeof(sol.errors),
                    typeof(sol.t),typeof(sol.k),
                    typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.destats)}(
                    sol.u,sol.u_analytic,sol.errors,sol.t,sol.k,sol.prob,
                    sol.alg,sol.interp,sol.dense,tslocation,sol.destats,sol.retcode)
end

function solution_slice(sol::AbstractODESolution{T,N},I) where {T,N}
  ODESolution{T,N,typeof(sol.u),typeof(sol.u_analytic),typeof(sol.errors),
                     typeof(sol.t),typeof(sol.k),
                     typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.destats)}(
                     sol.u[I],
                     sol.u_analytic === nothing ? nothing : sol.u_analytic[I],
                     sol.errors,sol.t[I],
                     sol.dense ? sol.k[I] : sol.k,
                     sol.prob,
                     sol.alg,sol.interp,false,sol.tslocation,sol.destats,sol.retcode)
 end

 function sensitivity_solution(sol::AbstractODESolution,u,t)
   T = eltype(eltype(u))
   N = length((size(sol.prob.u0)..., length(u)))
   interp = if typeof(sol.interp) <: LinearInterpolation
     LinearInterpolation(t,u)
   elseif typeof(sol.interp) <: ConstantInterpolation
     ConstantInterpolation(t,u)
   else
     SensitivityInterpolation(t,u)
   end

   ODESolution{T,N,typeof(u),typeof(sol.u_analytic),typeof(sol.errors),
               typeof(t),Nothing,typeof(sol.prob),typeof(sol.alg),
               typeof(interp),typeof(sol.destats)}(
               u,sol.u_analytic,sol.errors,t,nothing,sol.prob,
               sol.alg,interp,
               sol.dense,sol.tslocation,
               sol.destats,sol.retcode)
 end
