"""
    promote_tspan(tspan)

Normalize a differential equation time span into the representation stored on a
problem.

For a two-element tuple, `promote_tspan` returns `promote(t1, t2)` so both
endpoints have a common type. A scalar `tspan` is interpreted as `(zero(tspan),
tspan)`. A two-element array is accepted and converted through the same tuple
path, while arrays of any other length throw an error because saved output times
belong in `saveat`, not in `tspan`. `nothing` and function-valued time spans are
returned unchanged.

This function normalizes the stored value; it does not decide whether a solver
supports the resulting time type. Adaptive solvers generally require a floating
point independent variable, while non-adaptive solvers may support exact or
custom time types when the chosen algorithm also supports them.
"""
promote_tspan((t1, t2)::Tuple{T, S}) where {T, S} = promote(t1, t2)
promote_tspan(tspan::Number) = (zero(tspan), tspan)
promote_tspan(tspan::Nothing) = (nothing, nothing)
promote_tspan(tspan::Function) = tspan
function promote_tspan(tspan::AbstractArray)
    return length(tspan) == 2 ? promote_tspan((first(tspan), last(tspan))) :
        throw(error("The length of tspan must be two (and preferably, tspan should be a tuple, i.e. (0.0,1.0)). If you are trying to include other values for saving reasons, see the [common solver arguments page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/) for information on the saving command saveat."))
end

### Displays

function Base.summary(io::IO, prob::AbstractDEProblem)
    type_color, no_color = get_colorizers(io)
    print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, " with uType ",
        type_color, typeof(prob.u0),
        no_color, " and tType ",
        type_color,
        prob.tspan isa Function ?
            "Unknown" : (
                prob.tspan === nothing ?
                "Nothing" : typeof(prob.tspan[1])
            ),
        no_color,
        ". In-place: ", type_color, isinplace(prob), no_color
    )
    init = initialization_status(prob)
    !isnothing(init) && begin
        println(io)
        print(
            io, "Initialization status: ", type_color,
            initialization_status(prob), no_color
        )
    end

    return hasproperty(prob.f, :mass_matrix) && begin
        println(io)
        print(
            io, "Non-trivial mass matrix: ", type_color,
            !(prob.f.mass_matrix isa LinearAlgebra.UniformScaling{Bool}), no_color
        )
    end
end

function Base.summary(io::IO, prob::AbstractLinearProblem)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color
    )
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractLinearProblem)
    summary(io, A)
    println(io)
    print(io, "b: ")
    return show(io, mime, A.b)
end

function Base.summary(io::IO, prob::AbstractNonlinearProblem{uType, iip}) where {uType, iip}
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, " with uType ",
        type_color, uType,
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color
    )
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractNonlinearProblem)
    summary(io, A)
    println(io)
    print(io, "u0: ")
    return show(io, mime, state_values(A))
end

function Base.show(io::IO, mime::MIME"text/plain", A::IntervalNonlinearProblem)
    summary(io, A)
    println(io)
    print(io, "Interval: ")
    return show(io, mime, A.tspan)
end

function Base.summary(io::IO, prob::AbstractOptimizationProblem)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color
    )
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractOptimizationProblem)
    summary(io, A)
    println(io)
    print(io, "u0: ")
    return show(io, mime, A.u0)
end

function Base.summary(io::IO, prob::AbstractIntegralProblem)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color
    )
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractIntegralProblem)
    summary(io, A)
    return println(io)
end

function Base.summary(io::IO, prob::AbstractNoiseProblem)
    return print(
        io,
        nameof(typeof(prob)), " with WType ", typeof(prob.noise.curW), " and tType ",
        typeof(prob.tspan[1]), ". In-place: ", isinplace(prob)
    )
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractDEProblem)
    summary(io, A)
    println(io)
    print(io, "timespan: ")
    show(io, mime, A.tspan)
    println(io)
    print(io, "u0: ")
    return show(io, mime, A.u0)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractNoiseProblem)
    summary(io, A)
    println(io)
    print(io, "timespan: ")
    show(io, mime, A.tspan)
    return println(io)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractDAEProblem)
    summary(io, A)
    println(io)
    print(io, "timespan: ")
    show(io, mime, A.tspan)
    println(io)
    print(io, "u0: ")
    show(io, mime, A.u0)
    println(io)
    print(io, "du0: ")
    return show(io, mime, A.du0)
end

function Base.summary(io::IO, prob::AbstractEnsembleProblem)
    type_color, no_color = get_colorizers(io)
    return print(
        io,
        nameof(typeof(prob)),
        " with problem ",
        nameof(typeof(prob.prob))
    )
end
Base.show(io::IO, mime::MIME"text/plain", A::AbstractEnsembleProblem) = summary(io, A)

"""
$(TYPEDEF)

A singleton type used as the default value of the parameter argument `p` in
`AbstractSciMLProblem` constructors (for example `ODEProblem(f, u0, tspan)` with no
`p` supplied). It marks the absence of parameters.

`NullParameters` is deliberately not indexable or broadcastable: any attempt to index
into it (such as `p[1]` or `x .+ p` inside a problem's function) throws an error with a
message explaining that no parameters were passed to the problem. This turns the common
mistake of forgetting to supply `p` (or using the wrong function signature) into an
informative error rather than a confusing downstream failure.

## Example

```julia
prob = ODEProblem(f, u0, tspan)      # prob.p === NullParameters()
prob = ODEProblem(f, u0, tspan, p)   # prob.p === p
```
"""
struct NullParameters end

const NO_PARAMETERS_INDEX_ERROR_MESSAGE = """
An indexing operation was performed on a NullParameters object. This means no parameters were passed
into the AbstractSciMLProblem (e.x.: ODEProblem) but the parameters object `p` was used in an indexing
expression (e.x. `p[i]`, or `x .+ p`). Two common reasons for this issue are:

1. Forgetting to pass parameters into the problem constructor. For example, `ODEProblem(f,u0,tspan)` should
   be `ODEProblem(f,u0,tspan,p)` in order to use parameters.

2. Using the wrong function signature. For example, with `ODEProblem`s the function signature is always
   `f(du,u,p,t)` for the in-place form or `f(u,p,t)` for the out-of-place form. Note that the `p` argument
   will always be in the function signature regardless of if the problem is defined with parameters!
"""

struct NullParameterIndexError <: Exception end

function Base.showerror(io::IO, e::NullParameterIndexError)
    return println(io, NO_PARAMETERS_INDEX_ERROR_MESSAGE)
end

function Base.getindex(::NullParameters, i...)
    throw(NullParameterIndexError())
end
function Base.iterate(::NullParameters)
    throw(NullParameterIndexError())
end

function Base.show(io::IO, mime::MIME"text/plain", A::AbstractPDEProblem)
    summary(io, A.prob)
    return println(io)
end
function Base.summary(io::IO, prob::AbstractPDEProblem)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color
    )
end

Base.copy(p::SciMLBase.NullParameters) = p

SymbolicIndexingInterface.is_time_dependent(::AbstractDEProblem) = true
SymbolicIndexingInterface.is_time_dependent(::AbstractNonlinearProblem) = false
