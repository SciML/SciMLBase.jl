"""
    promote_tspan(tspan)

Convert the `tspan` field of a `AbstractDEProblem` to a `(tmin, tmax)` tuple, where both
elements are of the same type. If `tspan` is a function, returns it as-is.
"""
promote_tspan((t1, t2)::Tuple{T, S}) where {T, S} = promote(t1, t2)
promote_tspan(tspan::Number) = (zero(tspan), tspan)
promote_tspan(tspan::Nothing) = (nothing, nothing)
promote_tspan(tspan::Function) = tspan
function promote_tspan(tspan::AbstractArray)
    length(tspan) == 2 ? promote_tspan((first(tspan), last(tspan))) :
    throw(error("The length of tspan must be two (and preferably, tspan should be a tuple, i.e. (0.0,1.0)). If you are trying to include other values for saving reasons, see the [common solver arguments page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/) for information on the saving command saveat."))
end

### Displays

function Base.summary(io::IO, prob::AbstractDEProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, " with uType ",
        type_color, typeof(prob.u0),
        no_color, " and tType ",
        type_color,
        prob.tspan isa Function ?
        "Unknown" : (prob.tspan === nothing ?
         "Nothing" : typeof(prob.tspan[1])),
        no_color, 
        ". In-place: ", type_color, isinplace(prob), no_color) 
    init = initialization_status(prob)
    !isnothing(init) && begin 
        println(io)
        print(io, "Initialization status: ", type_color, initialization_status(prob), no_color)
    end

    hasproperty(prob.f, :mass_matrix) && begin
        println(io)
        print(io, "Non-trivial mass matrix: ", type_color, !(prob.f.mass_matrix isa LinearAlgebra.UniformScaling{Bool}), no_color)
    end
end

function Base.summary(io::IO, prob::AbstractLinearProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractLinearProblem)
    summary(io, A)
    println(io)
    print(io, "b: ")
    show(io, mime, A.b)
end

function Base.summary(io::IO, prob::AbstractNonlinearProblem{uType, iip}) where {uType, iip}
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, " with uType ",
        type_color, uType,
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractNonlinearProblem)
    summary(io, A)
    println(io)
    print(io, "u0: ")
    show(io, mime, state_values(A))
end

function Base.show(io::IO, mime::MIME"text/plain", A::IntervalNonlinearProblem)
    summary(io, A)
    println(io)
    print(io, "Interval: ")
    show(io, mime, A.tspan)
end

function Base.summary(io::IO, prob::AbstractOptimizationProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractOptimizationProblem)
    summary(io, A)
    println(io)
    print(io, "u0: ")
    show(io, mime, A.u0)
end

function Base.summary(io::IO, prob::AbstractIntegralProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, ". In-place: ",
        type_color, isinplace(prob),
        no_color)
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractIntegralProblem)
    summary(io, A)
    println(io)
end

function Base.summary(io::IO, prob::AbstractNoiseProblem)
    print(io,
        nameof(typeof(prob)), " with WType ", typeof(prob.noise.curW), " and tType ",
        typeof(prob.tspan[1]), ". In-place: ", isinplace(prob))
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractDEProblem)
    summary(io, A)
    println(io)
    print(io, "timespan: ")
    show(io, mime, A.tspan)
    println(io)
    print(io, "u0: ")
    show(io, mime, A.u0)
     
end
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractNoiseProblem)
    summary(io, A)
    println(io)
    print(io, "timespan: ")
    show(io, mime, A.tspan)
    println(io)
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
    show(io, mime, A.du0)
end

function Base.summary(io::IO, prob::AbstractEnsembleProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
        nameof(typeof(prob)),
        " with problem ",
        nameof(typeof(prob.prob)))
end
Base.show(io::IO, mime::MIME"text/plain", A::AbstractEnsembleProblem) = summary(io, A)

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
    println(io, NO_PARAMETERS_INDEX_ERROR_MESSAGE)
end

function Base.getindex(::NullParameters, i...)
    throw(NullParameterIndexError())
end
function Base.iterate(::NullParameters)
    throw(NullParameterIndexError())
end

function Base.show(io::IO, mime::MIME"text/plain", A::AbstractPDEProblem)
    summary(io, A.prob)
    println(io)
end
function Base.summary(io::IO, prob::AbstractPDEProblem)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color)
end

Base.copy(p::SciMLBase.NullParameters) = p

SymbolicIndexingInterface.is_time_dependent(::AbstractDEProblem) = true
SymbolicIndexingInterface.is_time_dependent(::AbstractNonlinearProblem) = false
