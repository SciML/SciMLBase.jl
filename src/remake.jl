@generated function struct_as_namedtuple(st)
  A = (Expr(:(=), n, :(st.$n)) for n in setdiff(fieldnames(st),(:kwargs,)))
  Expr(:tuple, A...)
end

Base.@pure remaker_of(prob::T) where {T<:SciMLProblem} = parameterless_type(T){isinplace(prob)}
Base.@pure remaker_of(alg::T) where {T} = parameterless_type(T)

# Define `remaker_of` for the types that does not (make sense to)
# implement `isinplace` trait:
for T in [
    NoiseProblem,
    SplitFunction,  # TODO: use isinplace path for type-stability
    TwoPointBVPFunction,
    ]
  @eval remaker_of(::$T) = $T
end

"""
    remake(thing; <keyword arguments>)

Re-construct `thing` with new field values specified by the keyword
arguments.
"""
function remake(thing; kwargs...)
  T = remaker_of(thing)
  if :kwargs ∈ fieldnames(typeof(thing))
    T(; struct_as_namedtuple(thing)...,thing.kwargs...,kwargs...)
  else
    T(; struct_as_namedtuple(thing)...,kwargs...)
  end
end

isrecompile(prob::ODEProblem{iip}) where {iip} = (prob.f isa ODEFunction) ? !isfunctionwrapper(prob.f.f) : true

function remake(prob::ODEProblem; f=missing,
                                  u0=missing,
                                  tspan=missing,
                                  p=missing,
                                  kwargs...)
  if f === missing
    f = prob.f
  elseif !isrecompile(thing)
    if isinplace(thing)
      f = wrapfun_iip(unwrap_fw(f),(u0,u0,p,tspan[1]))
    else
      f = wrapfun_oop(unwrap_fw(f),(u0,p,tspan[1]))
    end
    f = convert(ODEFunction{isinplace(prob)},f)
  else
    f = convert(ODEFunction{isinplace(prob)},f)
  end

  if u0 === missing
    u0 = prob.u0
  end

  if tspan === missing
    tspan = prob.tspan
  end

  if p === missing
    p = prob.p
  end

  ODEProblem{isinplace(prob)}(f,u0,tspan,p,prob.problem_type;prob.kwargs..., kwargs...)
end

function remake(thing::AbstractJumpProblem; kwargs...)
  parameterless_type(thing)(remake(thing.prob;kwargs...))
end
