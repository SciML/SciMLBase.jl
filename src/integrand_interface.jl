"""
    InplaceIntegrand(f!, result::AbstractArray)

Constructor for a `InplaceIntegrand` accepting an integrand of the form `f!(y,x,p)`. The
caller also provides an output array needed to store the result of the quadrature.
Intermediate `y` arrays are allocated during the calculation, and the final result is is
written to `result`, so use the IntegralSolution immediately after the calculation to read
the result, and don't expect it to persist if the same integrand is used for another
calculation.
"""
struct InplaceIntegrand{F,T<:AbstractArray}
    # in-place function f!(y, x, p) that takes one x value and outputs an array of results in-place
    f!::F
    I::T
end


"""
    BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch=typemax(Int))

Constructor for a `BatchIntegrand` accepting an integrand of the form `f!(y,x,p) = y .= f!.(x, Ref(p))`
that can evaluate the integrand at multiple quadrature nodes using, for example, threads,
the GPU, or distributed-memory. The `max_batch` keyword is a soft limit on the number of
nodes passed to the integrand. The buffers `y,x` must both be `resize!`-able since the
number of evaluation points may vary between calls to `f!`.
"""
struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x, p) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end


"""
    BatchIntegrand(f!, y, x; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` with pre-allocated buffers.
"""
BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

"""
    BatchIntegrand(f!, y::Type, x::Type=Nothing; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` whose range type is known. The domain type is optional.
Array buffers for those types are allocated internally.
"""
BatchIntegrand(f!, Y::Type, X::Type=Nothing; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, Y[], X[], max_batch)
