"""
The SciMLBaseMLStyleExt module provides backwards compatibility for those using MLStyle
to pattern match the Clocks algebraic data type (ADT).

Before this ADT was made using Expronomicon, which directly supported MLStyle.
Now that the ADT is made using Moshi, we need to define MLStyle custom patterns.
Using Moshi pattern matching is recommended over relying on this package extension.
"""
module SciMLBaseMLStyleExt

using SciMLBase: TimeDomain, ContinuousClock, SolverStepClock, PeriodicClock
using MLStyle: MLStyle
using MLStyle.AbstractPatterns: literal, wildcard, PComp, BasicPatterns, decons
using Moshi.Data: isa_variant

# This makes Singletons also work without parentheses in matches
MLStyle.is_enum(::Type{ContinuousClock}) = true
MLStyle.is_enum(::Type{SolverStepClock}) = true
function MLStyle.pattern_uncall(::Type{ContinuousClock}, self::Function, _, _, _)
    literal(ContinuousClock())
end
MLStyle.pattern_uncall(T::TimeDomain, self::Function, _, _, _) = literal(T())

function MLStyle.pattern_uncall(::Type{SolverStepClock}, self::Function, _, _, _)
    literal(SolverStepClock())
end

function periodic_clock_pattern(c)
    if c isa TimeDomain && isa_variant(c, PeriodicClock)
        (c.dt, c.phase)
    else
        # These values are used in match results, but they shouldn't.
        # This means that any wildcard pattern will be `nothing`, see broken test.
        (nothing, nothing)
    end
end

function MLStyle.pattern_uncall(
        ::Type{PeriodicClock}, self::Function, type_params, type_args, args)
    @assert isempty(type_params)
    @assert isempty(type_args)
    n_args = length(args)

    trans(expr) = Expr(:call, periodic_clock_pattern, expr)
    type_infer(_...) = Any

    extract = if n_args <= 1
        (expr::Any, i::Int, ::Any, ::Any) -> expr
    else
        (expr::Any, i::Int, ::Any, ::Any) -> Core._expr(:ref, expr, i)
    end

    comp = PComp(
        "PeriodicClock(c)", type_infer; view = BasicPatterns.SimpleCachablePre(trans))

    ps = if n_args === 0
        []
    elseif n_args === 1
        [self(Expr(:call, Some, args[1]))]
    elseif n_args === 2
        if args[2] == :(_...)
            [self(args[1]), wildcard]
        else
            [self(args[1]), self(args[2])]
        end
    else
        error("too many arguments")
    end

    decons(comp, extract, ps)
end

end
