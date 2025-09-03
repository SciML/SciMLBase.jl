module SciMLBaseForwardDiffExt

using SciMLBase, ForwardDiff
using ArrayInterface

import SciMLBase: 
    wrapfun_oop, wrapfun_iip, isdualtype, value, DualEltypeChecker,
    AbstractTimeseriesSolution, NonlinearProblem, NonlinearLeastSquaresProblem,
    ODEProblem, SDEProblem, RODEProblem, DDEProblem, PDEProblem, DAEProblem,
    RecursiveArrayTools, totallength, sse, anyeltypedual, reduce_tup

eltypedual(x) = eltype(x) <: ForwardDiff.Dual
isdualtype(::Type{<:ForwardDiff.Dual}) = true

promote_dual(::Type{T}, ::Type{T2}) where {T <: ForwardDiff.Dual, T2} = T
function promote_dual(::Type{T},
        ::Type{T2}) where {T <: ForwardDiff.Dual, T2 <: ForwardDiff.Dual}
    T
end
promote_dual(::Type{T}, ::Type{T2}) where {T, T2 <: ForwardDiff.Dual} = T2

function promote_dual(::Type{T},
        ::Type{T2}) where {T3, T4, V, V2 <: ForwardDiff.Dual, N, N2,
        T <: ForwardDiff.Dual{T3, V, N},
        T2 <: ForwardDiff.Dual{T4, V2, N2}}
    T2
end
function promote_dual(::Type{T},
        ::Type{T2}) where {T3, T4, V <: ForwardDiff.Dual, V2, N, N2,
        T <: ForwardDiff.Dual{T3, V, N},
        T2 <: ForwardDiff.Dual{T4, V2, N2}}
    T
end
function promote_dual(::Type{T},
        ::Type{T2}) where {
        T3, V <: ForwardDiff.Dual, V2 <: ForwardDiff.Dual,
        N,
        T <: ForwardDiff.Dual{T3, V, N},
        T2 <: ForwardDiff.Dual{T3, V2, N}}
    ForwardDiff.Dual{T3, promote_dual(V, V2), N}
end

"""
    promote_dual(::Type{T},::Type{T2})

Is like the number promotion system, but always prefers a dual number type above
anything else. For higher order differentiation, it returns the most dualiest of
them all. This is then used to promote `u0` into the suspected highest differentiation
space for solving the equation.
"""
promote_dual(::Type{T}, ::Type{T2}) where {T, T2} = T

# `reduce` and `map` are specialized on tuples to be unrolled (via recursion)
# Therefore, they can be type stable even with heterogeneous input types.
# We also don't care about allocating any temporaries with them, as it should
# all be unrolled and optimized away.
# Being unrolled also means const prop can work for things like
# `mapreduce(f, op, propertynames(x))`
# where `f` may call `getproperty` and thus have return type dependent
# on the particular symbol.
# `mapreduce` hasn't received any such specialization.
@inline diffeqmapreduce(f::F, op::OP, x::Tuple) where {F, OP} = reduce_tup(op, map(f, x))
@inline function diffeqmapreduce(f::F, op::OP, x::NamedTuple) where {F, OP}
    reduce_tup(op, map(f, x))
end
# For other container types, we probably just want to call `mapreduce`
@inline diffeqmapreduce(f::F, op::OP, x) where {F, OP} = mapreduce(f, op, x, init = Any)

getval(::Val{I}) where {I} = I
getval(::Type{Val{I}}) where {I} = I
getval(I::Int) = I

const DUALCHECK_RECURSION_MAX = 10

function (dec::DualEltypeChecker)(::Val{Y}) where {Y}
    isdefined(dec.x, Y) || return Any
    getval(dec.counter) >= DUALCHECK_RECURSION_MAX && return Any
    anyeltypedual(getfield(dec.x, Y), Val{getval(dec.counter)})
end

# Untyped dispatch: catch composite types, check all of their fields
"""
    anyeltypedual(x)

Searches through a type to see if any of its values are parameters. This is used to
then promote other values to match the dual type. For example, if a user passes a parameter

which is a `Dual` and a `u0` which is a `Float64`, after the first time step, `f(u,p,t) = p*u`
will change `u0` from `Float64` to `Dual`. Thus the state variable always needs to be converted
to a dual number before the solve. Worse still, this needs to be done in the case of
`f(du,u,p,t) = du[1] = p*u[1]`, and thus running `f` and taking the return value is not a valid
way to calculate the required state type.

But given the properties of automatic differentiation requiring that differentiation of parameters
implies differentiation of state, we assume any dual parameters implies differentiation of state
and then attempt to upconvert `u0` to match that dual-ness. Because this changes types, this needs
to be specified at compiled time and thus cannot have a Bool-based opt out, so in the future this
may be extended to use a preference system to opt-out with a `UPCONVERT_DUALS`. In the case where
upconversion is not done automatically, the user is required to upconvert all initial conditions
themselves, for an example of how this can be confusing to a user see
<https://discourse.julialang.org/t/typeerror-in-julia-turing-when-sampling-for-a-forced-differential-equation/82937>
"""
@generated function anyeltypedual(x, ::Type{Val{counter}}) where {counter}
    x = x.name === Core.Compiler.typename(Type) ? x.parameters[1] : x
    if isdualtype(x)
        :($x)
    elseif fieldnames(x) === ()
        :(Any)
    elseif counter < DUALCHECK_RECURSION_MAX
        T = diffeqmapreduce(x -> anyeltypedual(x, Val{counter + 1}), promote_dual,
            x.parameters)
        if T === Any || isconcretetype(T)
            :($T)
        else
            :(diffeqmapreduce(DualEltypeChecker($x, $counter + 1), promote_dual,
                map(Val, fieldnames($(typeof(x))))))
        end
    else
        :(Any)
    end
end

const FORWARDDIFF_AUTODETECTION_FAILURE_MESSAGE = """
                                Failed to automatically detect ForwardDiff compatability of
                                the parameter object. In order for ForwardDiff.jl automatic
                                differentiation to work on a solution object, the state of
                                the differential equation or nonlinear solve (`u0`) needs to
                                be converted to a Dual type which matches the values being
                                differentiated. For example, for a loss function loss(p)
                                where `p`` is a `Vector{Float64}`, this conversion is
                                equivalent to:

                                ```julia
                                # Convert u0 to match the new Dual element type of `p`
                                _prob = remake(prob, u0 = eltype(p).(prob.u0))
                                ```

                                In most cases, SciML tools are able to do this conversion
                                automatically. However, it seems you have provided a
                                parameter type for which this automatic conversion has failed.

                                To fix this, you can do the conversion yourself. For example,
                                if you have a parameter vector being optimized `p` which is
                                then put into an odd struct, you can manually convert `u0`
                                to match `p`:

                                ```julia
                                function loss(p)
                                    _prob = remake(prob, u0 = eltype(p).(prob.u0), p = MyStruct(p))
                                    sol = solve(_prob, ...)
                                    # do stuff on sol
                                end
                                ```

                                Or you can define a dispatch on `SciMLBase.anyeltypedual`
                                which tells the system what fields to interpret as the
                                differentiable parts. For example, to support ODESolutions
                                as parameters we tell it the data is `sol.u` and `sol.t` via:

                                ```julia
                                function SciMLBase.anyeltypedual(sol::ODESolution, counter = 0)
                                    SciMLBase.anyeltypedual((sol.u, sol.t))
                                end
                                ```

                                To opt a type out of the dual checking, define an overload
                                that returns Any. For example:

                                ```julia
                                function SciMLBase.anyeltypedual(::YourType, ::Type{Val{counter}}) where {counter}
                                    Any
                                end
                                ```

                                If you have defined this on a common type which should
                                be more generally supported, please open a pull request
                                adding this dispatch. If you need help defining this dispatch,
                                feel free to open an issue.
                                """

struct ForwardDiffAutomaticDetectionFailure <: Exception end

function Base.showerror(io::IO, e::ForwardDiffAutomaticDetectionFailure)
    print(io, FORWARDDIFF_AUTODETECTION_FAILURE_MESSAGE)
end

function anyeltypedual(::Type{Union{}})
    throw(ForwardDiffAutomaticDetectionFailure())
end

function anyeltypedual(::Type{<:AbstractTimeseriesSolution{T, N}},
        ::Type{Val{counter}} = Val{0}) where {T, N, counter}
    anyeltypedual(T)
end

function anyeltypedual(
        ::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <:
                                                              NonlinearProblem{
        uType, iip, pType}} where {uType, iip, pType}
    return anyeltypedual((uType, pType), Val{counter})
end

function anyeltypedual(
        ::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <:
                                                              NonlinearLeastSquaresProblem{
        uType, iip, pType}} where {uType, iip, pType}
    return anyeltypedual((uType, pType), Val{counter})
end

function anyeltypedual(x::SciMLBase.RecipesBase.AbstractPlot,
        ::Type{Val{counter}} = Val{0}) where {counter}
    Any
end
function anyeltypedual(x::Returns, ::Type{Val{counter}} = Val{0}) where {counter}
    anyeltypedual(x.value, Val{counter})
end

Base.@assume_effects :foldable function __anyeltypedual(::Type{T}) where {T}
    if T isa Union
        promote_dual(anyeltypedual(T.a), anyeltypedual(T.b))
    elseif hasproperty(T, :parameters)
        mapreduce(anyeltypedual, promote_dual, T.parameters; init = Any)
    else
        T
    end
end
function anyeltypedual(::Type{T}, ::Type{Val{counter}} = Val{0}) where {counter} where {T}
    __anyeltypedual(T)
end

function anyeltypedual(::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <:
                                                              Union{AbstractArray, Set}}
    anyeltypedual(eltype(T))
end
Base.@pure function __anyeltypedual_ntuple(::Type{T}) where {T <: NTuple}
    if isconcretetype(eltype(T))
        return eltype(T)
    end
    if isempty(T.parameters)
        Any
    else
        mapreduce(anyeltypedual, promote_dual, T.parameters; init = Any)
    end
end
function anyeltypedual(
        ::Type{T}, ::Type{Val{counter}} = Val{0}) where {counter} where {T <: NTuple}
    __anyeltypedual_ntuple(T)
end

# Any in this context just means not Dual
function anyeltypedual(
        x::SciMLBase.NullParameters, ::Type{Val{counter}} = Val{0}) where {counter}
    Any
end

function anyeltypedual(sol::RecursiveArrayTools.AbstractDiffEqArray, counter = 0)
    diffeqmapreduce(anyeltypedual, promote_dual, (sol.u, sol.t))
end

function anyeltypedual(prob::Union{ODEProblem, SDEProblem, RODEProblem, DDEProblem},
        ::Type{Val{counter}} = Val{0}) where {counter}
    anyeltypedual((prob.u0, prob.p, prob.tspan))
end

function anyeltypedual(
        prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem, OptimizationProblem},
        ::Type{Val{counter}} = Val{0}) where {counter}
    anyeltypedual((prob.u0, prob.p))
end

function anyeltypedual(x::Number, ::Type{Val{counter}} = Val{0}) where {counter}
    anyeltypedual(typeof(x))
end
function anyeltypedual(
        x::Union{String, Symbol}, ::Type{Val{counter}} = Val{0}) where {counter}
    typeof(x)
end
function anyeltypedual(x::Union{AbstractArray{T}, Set{T}},
        ::Type{Val{counter}} = Val{0}) where {counter} where {
        T <:
        Union{Number,
        Symbol,
        String}}
    anyeltypedual(T)
end
function anyeltypedual(x::Union{AbstractArray{T}, Set{T}},
        ::Type{Val{counter}} = Val{0}) where {counter} where {
        T <: Union{
        AbstractArray{
            <:Number,
        },
        Set{
            <:Number,
        }}}
    anyeltypedual(eltype(x))
end
function anyeltypedual(x::Union{AbstractArray{T}, Set{T}},
        ::Type{Val{counter}} = Val{0}) where {counter} where {N, T <: NTuple{N, <:Number}}
    anyeltypedual(eltype(x))
end

# Try to avoid this dispatch because it can lead to type inference issues when !isconcrete(eltype(x))
function anyeltypedual(x::AbstractArray, ::Type{Val{counter}} = Val{0}) where {counter}
    if isconcretetype(eltype(x))
        anyeltypedual(eltype(x))
    elseif !isempty(x) && all(i -> isassigned(x, i), 1:length(x)) &&
           counter < DUALCHECK_RECURSION_MAX
        _counter = Val{counter + 1}
        mapreduce(y -> anyeltypedual(y, _counter), promote_dual, x)
    else
        # This fallback to Any is required since otherwise we cannot handle `undef` in all cases
        #  misses cases of
        Any
    end
end

function anyeltypedual(x::Set, ::Type{Val{counter}} = Val{0}) where {counter}
    if isconcretetype(eltype(x))
        anyeltypedual(eltype(x))
    else
        # This fallback to Any is required since otherwise we cannot handle `undef` in all cases
        Any
    end
end

function anyeltypedual(x::Tuple, ::Type{Val{counter}} = Val{0}) where {counter}
    # Handle the empty tuple case separately for inference and to avoid mapreduce error
    if x === ()
        Any
    else
        diffeqmapreduce(anyeltypedual, promote_dual, x)
    end
end
function anyeltypedual(x::AbstractDict, ::Type{Val{counter}} = Val{0}) where {counter}
    isempty(x) ? eltype(values(x)) : mapreduce(anyeltypedual, promote_dual, values(x))
end
function anyeltypedual(x::NamedTuple, ::Type{Val{counter}} = Val{0}) where {counter}
    anyeltypedual(values(x))
end

function anyeltypedual(
        f::SciMLBase.AbstractSciMLFunction, ::Type{Val{counter}}) where {counter}
    Any
end

anyeltypedual(::@Kwargs{}, ::Type{Val{counter}} = Val{0}) where {counter} = Any
anyeltypedual(::Type{@Kwargs{}}, ::Type{Val{counter}} = Val{0}) where {counter} = Any

# Opt out since these are using for preallocation, not differentiation
function anyeltypedual(x::Union{ForwardDiff.AbstractConfig, Module},
        ::Type{Val{counter}} = Val{0}) where {counter}
    Any
end
function anyeltypedual(x::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <:
                                                              ForwardDiff.AbstractConfig}
    Any
end

function anyeltypedual(x::ForwardDiff.DiffResults.DiffResult,
        ::Type{Val{counter}} = Val{0}) where {counter}
    Any
end
function anyeltypedual(x::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <:
                                                              ForwardDiff.DiffResults.DiffResult}
    Any
end

function anyeltypedual(::Type{T},
        ::Type{Val{counter}} = Val{0}) where {counter} where {T <: ForwardDiff.Dual}
    T
end

value(x::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = V
value(x::ForwardDiff.Dual) = value(ForwardDiff.value(x))

unitfulvalue(x::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = V
unitfulvalue(x::ForwardDiff.Dual) = unitfulvalue(ForwardDiff.value(x))

sse(x::ForwardDiff.Dual) = sse(ForwardDiff.value(x)) + sum(sse, ForwardDiff.partials(x))

function SciMLBase.totallength(x::ForwardDiff.Dual)
    return SciMLBase.totallength(ForwardDiff.value(x)) +
           sum(SciMLBase.totallength, ForwardDiff.partials(x))
end

end
