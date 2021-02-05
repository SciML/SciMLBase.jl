"""
$(TYPEDEF)
"""
struct StandardSDEProblem end

"""
$(TYPEDEF)
"""
struct SDEProblem{uType,tType,isinplace,P,NP,F,G,K,ND} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  f::F
  g::G
  u0::uType
  tspan::tType
  p::P
  noise::NP
  kwargs::K
  noise_rate_prototype::ND
  seed::UInt64
  @add_kwonly function SDEProblem{iip}(f::AbstractSDEFunction{iip},g,u0,
          tspan,p=NullParameters();
          noise_rate_prototype = nothing,
          noise= nothing, seed = UInt64(0),
          kwargs...) where {iip}
    _tspan = promote_tspan(tspan)

    new{typeof(u0),typeof(_tspan),
        isinplace(f),typeof(p),
        typeof(noise),typeof(f),typeof(f.g),
        typeof(kwargs),
        typeof(noise_rate_prototype)}(
        f,f.g,u0,_tspan,p,
        noise,kwargs,
        noise_rate_prototype,seed)
  end

  function SDEProblem{iip}(f,g,u0,tspan,p=NullParameters();kwargs...) where {iip}
    SDEProblem(convert(SDEFunction{iip},f,g),g,u0,tspan,p;kwargs...)
  end
end

#=
function SDEProblem(f::AbstractSDEFunction,u0,tspan,p=NullParameters();kwargs...)
  SDEProblem(f,f.g,u0,tspan,p;kwargs...)
end
=#

function SDEProblem(f::AbstractSDEFunction,g,u0,tspan,p=NullParameters();kwargs...)
  SDEProblem{isinplace(f)}(f,g,u0,tspan,p;kwargs...)
end

function SDEProblem(f,g,u0,tspan,p=NullParameters();kwargs...)
  SDEProblem(convert(SDEFunction,f,g),g,u0,tspan,p;kwargs...)
end

"""
$(TYPEDEF)
"""
abstract type AbstractSplitSDEProblem end

"""
$(TYPEDEF)
"""
struct SplitSDEProblem{iip} <: AbstractSplitSDEProblem end
# u' = Au + f
function SplitSDEProblem(f1,f2,g,u0,tspan,p=NullParameters();kwargs...)
  SplitSDEProblem(SplitSDEFunction(f1,f2,g),g,u0,tspan,p;kwargs...)
end

SplitSDEProblem(f::SplitSDEFunction,g,u0,tspan,p=NullParameters();kwargs...) =
  SplitSDEProblem{isinplace(f)}(f,g,u0,tspan,p;kwargs...)

function SplitSDEProblem{iip}(f1,f2,g,u0,tspan,p=NullParameters();kwargs...) where iip
  SplitSDEProblem(SplitSDEFunction(f1,f2,g),g,u0,tspan,p;kwargs...)
end
function SplitSDEProblem{iip}(f::SplitSDEFunction,g,u0,tspan,p=NullParameters();
                                     func_cache=nothing,kwargs...) where iip
  if f.cache === nothing && iip
    cache = similar(u0)
    _f = SplitSDEFunction{iip}(f.f1, f.f2, f.g; mass_matrix=f.mass_matrix,
                              _func_cache=cache, analytic=f.analytic)
  else
    _f = f
  end
  SDEProblem(_f,g,u0,tspan,p;kwargs...)
end

"""
$(TYPEDEF)
"""
abstract type AbstractDynamicalSDEProblem end

"""
$(TYPEDEF)
"""
struct DynamicalSDEProblem{iip} <: AbstractDynamicalSDEProblem end

function DynamicalSDEProblem(f1,f2,g,v0,u0,tspan,p=NullParameters();kwargs...)
  DynamicalSDEProblem(DynamicalSDEFunction(f1,f2,g),g,v0,u0,tspan,p;kwargs...)
end

DynamicalSDEProblem(f::DynamicalSDEFunction,g,v0,u0,tspan,p=NullParameters();kwargs...) =
  DynamicalSDEProblem{isinplace(f)}(f,g,u0,v0,tspan,p;kwargs...)

function DynamicalSDEProblem{iip}(f1,f2,g,v0,u0,tspan,p=NullParameters();kwargs...) where iip
  DynamicalSDEProblem(DynamicalSDEFunction(f1,f2,g),g,v0,u0,tspan,p;kwargs...)
end
function DynamicalSDEProblem{iip}(f::DynamicalSDEFunction,g,v0,u0,tspan,p=NullParameters();
                                     func_cache=nothing,kwargs...) where iip
  if f.cache === nothing && iip
    cache = similar(u0)
    _f = DynamicalSDEFunction{iip}(f.f1, f.f2, f.g; mass_matrix=f.mass_matrix,
                              _func_cache=cache, analytic=f.analytic)
  else
    _f = f
  end
  SDEProblem(_f,g,ArrayPartition(v0,u0),tspan,p;kwargs...)
end