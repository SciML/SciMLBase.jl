module SciMLBaseFunctionPropertiesExt

using SciMLBase: AbstractSciMLFunction
using FunctionProperties: FunctionProperties

# Analyze the wrapped right-hand side, not the `AbstractSciMLFunction` functor itself. The functor
# is dispatch plumbing: operator/`isa` selection (`if f.f isa AbstractSciMLOperator`) and, for an
# MTK-compiled RHS, splat forwarding into a generated function. Its branches are value-independent,
# and one of them (the dead operator path) is live IR that the scan would otherwise follow into
# arity-mismatch handling. Looking through to `f.f` is what makes `hasbranching` report the user's
# right-hand side, which is what callers (e.g. SciMLSensitivity, deciding whether a ReverseDiff
# tape can be compiled) actually care about.
FunctionProperties.hasbranching(f::AbstractSciMLFunction, args...) =
    FunctionProperties.hasbranching(f.f, args...)

if isdefined(FunctionProperties, :islinear)
    FunctionProperties.islinear(f::AbstractSciMLFunction, args...; kwargs...) =
        FunctionProperties.islinear(f.f, args...; kwargs...)
end

if isdefined(FunctionProperties, :isautonomous)
    FunctionProperties.isautonomous(f::AbstractSciMLFunction, args...; kwargs...) =
        FunctionProperties.isautonomous(f.f, args...; kwargs...)
end

end
