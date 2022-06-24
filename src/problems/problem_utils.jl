"""
    promote_tspan(tspan)

Convert the `tspan` field of a `DEProblem` to a `(tmin, tmax)` tuple, where both
elements are of the same type. If `tspan` is a function, returns it as-is.
"""
promote_tspan((t1, t2)::Tuple{T, S}) where {T, S} = promote(t1, t2)
promote_tspan(tspan::Number) = (zero(tspan), tspan)
promote_tspan(tspan::Nothing) = (nothing, nothing)
promote_tspan(tspan::Function) = tspan
function promote_tspan(tspan::AbstractArray)
    length(tspan) == 2 ? promote_tspan((first(tspan), last(tspan))) :
    throw(error("The length of tspan must be two (and preferably, tspan should be a tuple, i.e. (0.0,1.0)). If you are trying to include other values for saving reasons, note see the [common solver arguments page](https://docs.juliadiffeq.org/latest/basics/common_solver_opts/) for information on the saving command saveat."))
end

### Displays

function Base.summary(io::IO, prob::DEProblem)
    type_color, no_color = get_colorizers(io)
    print(io,
          type_color, nameof(typeof(prob)),
          no_color, " with uType ",
          type_color, typeof(prob.u0),
          no_color, " and tType ",
          type_color,
          typeof(prob.tspan) <: Function ?
          "Unknown" : (prob.tspan === nothing ?
           "Nothing" : typeof(prob.tspan[1])),
          no_color, ". In-place: ",
          type_color, isinplace(prob),
          no_color)
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
    show(io, mime, A.u0)
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
          nameof(typeof(prob)), " with WType ", typeof(prob.noise.W[1]), " and tType ",
          typeof(prob.tspan[1]), ". In-place: ", isinplace(prob))
end
function Base.show(io::IO, mime::MIME"text/plain", A::DEProblem)
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
TreeViews.hastreeview(x::DEProblem) = true
function TreeViews.treelabel(io::IO, x::DEProblem,
                             mime::MIME"text/plain" = MIME"text/plain"())
    summary(io, x)
end

struct NullParameters end
function Base.getindex(::NullParameters, i...)
    error("Parameters were indexed but the parameters are `nothing`. You likely forgot to pass in parameters to the DEProblem!")
end
function Base.iterate(::NullParameters)
    error("Parameters were indexed but the parameters are `nothing`. You likely forgot to pass in parameters to the DEProblem!")
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
