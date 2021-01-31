"""
$(TYPEDEF)
"""
struct StandardDDEProblem end

"""
$(TYPEDEF)
"""
struct DDEProblem{uType,tType,lType,lType2,isinplace,P,F,H,K,PT} <:
                          AbstractDDEProblem{uType,tType,lType,isinplace}
  f::F
  u0::uType
  h::H
  tspan::tType
  p::P
  constant_lags::lType
  dependent_lags::lType2
  kwargs::K
  neutral::Bool
  order_discontinuity_t0::Int
  problem_type::PT

  @add_kwonly function DDEProblem{iip}(f::AbstractDDEFunction{iip}, u0, h, tspan, p=NullParameters();
                                       constant_lags = (),
                                       dependent_lags = (),
                                       neutral = f.mass_matrix !== I && det(f.mass_matrix) != 1,
                                       order_discontinuity_t0 = 0,
                                       problem_type = StandardDDEProblem(),
                                       kwargs...) where {iip}
    _tspan = promote_tspan(tspan)
    new{typeof(u0),typeof(_tspan),typeof(constant_lags),typeof(dependent_lags),isinplace(f),
        typeof(p),typeof(f),typeof(h),typeof(kwargs),typeof(problem_type)}(
          f, u0, h, _tspan, p, constant_lags, dependent_lags, kwargs, neutral,
          order_discontinuity_t0, problem_type)
  end

  function DDEProblem{iip}(f::AbstractDDEFunction{iip}, h, tspan::Tuple, p=NullParameters();
                           order_discontinuity_t0 = 1, kwargs...) where iip
    DDEProblem{iip}(f, h(p, first(tspan)), h, tspan, p;
                    order_discontinuity_t0 = max(1, order_discontinuity_t0), kwargs...)
  end

  function DDEProblem{iip}(f, args...; kwargs...) where iip
    DDEProblem{iip}(convert(DDEFunction{iip}, f), args...; kwargs...)
  end
end

DDEProblem(f, args...; kwargs...) =
  DDEProblem(convert(DDEFunction, f), args...; kwargs...)

DDEProblem(f::AbstractDDEFunction, args...; kwargs...) =
  DDEProblem{isinplace(f)}(f, args...; kwargs...)

"""
$(TYPEDEF)
"""
abstract type AbstractDynamicalDDEProblem end

"""
$(TYPEDEF)
"""
struct DynamicalDDEProblem{iip} <: AbstractDynamicalDDEProblem end

# u' = f1(v,h)
# v' = f2(t,u,h)
"""
    DynamicalDDEProblem(f::DynamicalDDEFunction,v0,u0,tspan,p=NullParameters(),callback=CallbackSet())

Define a dynamical DDE problem from a [`DynamicalDDEFunction`](@ref).
"""
function DynamicalDDEProblem(f::DynamicalDDEFunction,v0,u0,h,tspan,p=NullParameters();dependent_lags=(),kwargs...)
  DDEProblem(f,ArrayPartition(v0,u0),h,tspan,p;
             problem_type=DynamicalDDEProblem{isinplace(f)}(),
             dependent_lags=ntuple(i->(u,p,t)->dependent_lags[i](u[1],u[2],p,t),length(dependent_lags)),
             kwargs...)
end
function DynamicalDDEProblem(f::DynamicalDDEFunction,h,tspan,p=NullParameters();kwargs...)
  DynamicalDDEProblem(f,h(p,first(tspan))...,h,tspan,p;kwargs...)
end
function DynamicalDDEProblem(f1::Function,f2::Function,args...;kwargs...)
  DynamicalDDEProblem(DynamicalDDEFunction(f1,f2),args...;kwargs...)
end

"""
    DynamicalDDEProblem{isinplace}(f1,f2,v0,u0,h,tspan,p=NullParameters(),callback=CallbackSet())

Define a dynamical DDE problem from the two functions `f1` and `f2`.

# Arguments
* `f1` and `f2`: The functions in the DDE.
* `v0` and `u0`: The initial conditions.
* `h`: The initial history function.
* `tspan`: The timespan for the problem.
* `p`: Parameter values for `f1` and `f2`.
* `callback`: A callback to be applied to every solver which uses the problem. Defaults to nothing.

`isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.
"""
function DynamicalDDEProblem{iip}(f1::Function,f2::Function,args...;kwargs...) where iip
  DynamicalDDEProblem(DynamicalDDEFunction{iip}(f1,f2),args...;kwargs...)
end

# u'' = f(du,u,h,p,t)
"""
$(TYPEDEF)
"""
struct SecondOrderDDEProblem{iip} <: AbstractDynamicalDDEProblem end
function SecondOrderDDEProblem(f,args...;kwargs...)
  iip = isinplace(f,6)
  SecondOrderDDEProblem{iip}(f,args...;kwargs...)
end

"""
    SecondOrderDDEProblem{isinplace}(f,du0,u0,h,tspan,p=NullParameters(),callback=CallbackSet())

Define a second order DDE problem with the specified function.

# Arguments
* `f`: The function for the second derivative.
* `du0`: The initial derivative.
* `u0`: The initial condition.
* `h`: The initial history function.
* `tspan`: The timespan for the problem.
* `p`: Parameter values for `f`.
* `callback`: A callback to be applied to every solver which uses the problem. Defaults to nothing.

`isinplace` optionally sets whether the function is inplace or not.
This is determined automatically, but not inferred.
"""
function SecondOrderDDEProblem{iip}(f,args...;kwargs...) where iip
  if iip
    f2 = function (du,v,u,h,p,t)
      du .= v
    end
  else
    f2 = function (v,u,h,p,t)
      v
    end
  end
  DynamicalDDEProblem{iip}(f,f2,args...;problem_type=SecondOrderDDEProblem{iip}(),kwargs...)
end
function SecondOrderDDEProblem(f::DynamicalDDEFunction,args...;kwargs...)
  iip = isinplace(f.f1, 6)
  if f.f2.f === nothing
    if iip
      f2 = function (du,v,u,h,p,t)
        du .= v
      end
    else
      f2 = function (v,u,h,p,t)
        v
      end
    end
    return DynamicalDDEProblem(DynamicalDDEFunction{iip}(f.f1,f2;mass_matrix=f.mass_matrix,analytic=f.analytic),
                               args...;problem_type=SecondOrderDDEProblem{iip}(),kwargs...)
  else
    return DynamicalDDEProblem(DynamicalDDEFunction{iip}(f.f1,f.f2;mass_matrix=f.mass_matrix,analytic=f.analytic),
                               args...;problem_type=SecondOrderDDEProblem{iip}(),kwargs...)
  end
end
