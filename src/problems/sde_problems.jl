"""
$(TYPEDEF)
"""
struct StandardSDEProblem end

@doc doc"""

Defines an stochastic differential equation (SDE) problem.
Documentation Page: https://docs.sciml.ai/DiffEqDocs/stable/types/sde_types/

## Mathematical Specification of a SDE Problem

To define an SDE Problem, you simply need to give the forcing function `f`,
the noise function `g`, and the initial condition `u₀` which define an SDE:

```math
du = f(u,p,t)dt + Σgᵢ(u,p,t)dWⁱ
```

`f` and `g` should be specified as `f(u,p,t)` and  `g(u,p,t)` respectively, and `u₀`
should be an AbstractArray whose geometry matches the desired geometry of `u`.
Note that we are not limited to numbers or vectors for `u₀`; one is allowed to
provide `u₀` as arbitrary matrices / higher dimension tensors as well. A vector
of `g`s can also be defined to determine an SDE of higher Ito dimension.

## Problem Type

Wraps the data which defines an SDE problem

```math
u = f(u,p,t)dt + Σgᵢ(u,p,t)dWⁱ
```

with initial condition `u0`.

### Constructors

- `SDEProblem(f::SDEFunction,g,u0,tspan,p=NullParameters();noise=WHITE_NOISE,noise_rate_prototype=nothing)`
- `SDEProblem{isinplace,specialize}(f,g,u0,tspan,p=NullParameters();noise=WHITE_NOISE,noise_rate_prototype=nothing)` :
  Defines the SDE with the specified functions. The default noise is `WHITE_NOISE`.
  `isinplace` optionally sets whether the function is inplace or not. This is
  determined automatically, but not inferred. `specialize` optionally controls
  the specialization level. See the [specialization levels section of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Levels)
  for more details. The default is `AutoSpecialize.

Parameters are optional, and if not given then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the
[DiffEqFunctions](@ref performance_overloads)
page.

### Fields

* `f`: The drift function in the SDE.
* `g`: The noise function in the SDE.
* `u0`: The initial condition.
* `tspan`: The timespan for the problem.
* `p`: The optional parameters for the problem. Defaults to `NullParameters`.
* `noise`: The noise process applied to the noise upon generation. Defaults to
  Gaussian white noise. For information on defining different noise processes,
  see [the noise process documentation page](@ref noise_process).
* `noise_rate_prototype`: A prototype type instance for the noise rates, that
  is the output `g`. It can be any type which overloads `A_mul_B!` with itself
  being the middle argument. Commonly, this is a matrix or sparse matrix. If
  this is not given, it defaults to `nothing`, which means the problem should
  be interpreted as having diagonal noise.
* `kwargs`: The keyword arguments passed onto the solves.

## Example Problems

Examples problems can be found in [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl).

To use a sample problem, such as `prob_sde_linear`, you can do something like:

```julia
#] add SDEProblemLibrary
using SDEProblemLibrary
prob = SDEProblemLibrary.prob_sde_linear
sol = solve(prob)
```
"""
struct SDEProblem{uType, tType, isinplace, P, NP, F, G, K, ND} <:
       AbstractSDEProblem{uType, tType, isinplace, ND}
    f::F
    g::G
    u0::uType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    noise_rate_prototype::ND
    seed::UInt64
    @add_kwonly function SDEProblem{iip}(f::AbstractSDEFunction{iip}, u0,
        tspan, p = NullParameters();
        noise_rate_prototype = nothing,
        noise = nothing, seed = UInt64(0),
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(p),
            typeof(noise), typeof(f), typeof(f.g),
            typeof(kwargs),
            typeof(noise_rate_prototype)}(f, f.g, _u0, _tspan, p,
            noise, kwargs,
            noise_rate_prototype, seed)
    end

    function SDEProblem{iip}(f, g, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        SDEProblem(SDEFunction{iip}(f, g), u0, tspan, p; kwargs...)
    end
end

TruncatedStacktraces.@truncate_stacktrace SDEProblem 3 1 2

function SDEProblem(f::AbstractSDEFunction, u0, tspan, p = NullParameters(); kwargs...)
    SDEProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

function SDEProblem(f, g, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    SDEProblem{iip}(SDEFunction{iip}(f, g), u0, tspan, p; kwargs...)
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
#=
function SplitSDEProblem(f1, f2, g, u0, tspan, p = NullParameters(); kwargs...)
    SplitSDEProblem(SplitSDEFunction(f1, f2, g), g, u0, tspan, p; kwargs...)
end
=#
function SplitSDEProblem{iip}(f1, f2, g, u0, tspan, p = NullParameters(); kwargs...) where {iip}
    SplitSDEProblem{iip}(SplitSDEFunction(f1, f2, g), u0, tspan, p; kwargs...)
end

function SplitSDEProblem(f1, f2, g, u0, tspan, p = NullParameters(); kwargs...)
    ff = SplitSDEFunction(f1, f2, g)
    SplitSDEProblem{isinplace(ff)}(ff, u0, tspan, p; kwargs...)
end

function SplitSDEProblem(f::SplitSDEFunction, u0, tspan, p = NullParameters(); kwargs...)
    SplitSDEProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

function SplitSDEProblem{iip}(f::SplitSDEFunction, u0, tspan, p = NullParameters();
    func_cache = nothing, kwargs...) where {iip}
    if f.cache === nothing && iip
        cache = similar(u0)
        _f = SplitSDEFunction{iip}(f.f1, f.f2, f.g; mass_matrix = f.mass_matrix,
            _func_cache = cache, analytic = f.analytic)
    else
        _f = f
    end
    SDEProblem(_f, u0, tspan, p; kwargs...)
end

"""
$(TYPEDEF)
"""
abstract type AbstractDynamicalSDEProblem end

"""
$(TYPEDEF)
"""
struct DynamicalSDEProblem{iip} <: AbstractDynamicalSDEProblem end

function DynamicalSDEProblem(f1, f2, g, v0, u0, tspan, p = NullParameters(); kwargs...)
    ff = DynamicalSDEFunction(f1, f2, g)
    DynamicalSDEProblem{isinplace(ff)}(ff, v0, u0, tspan, p; kwargs...)
end

function DynamicalSDEProblem{iip}(f1, f2, g, v0, u0, tspan, p = NullParameters(); kwargs...) where {iip}
    ff = DynamicalSDEFunction(f1, f2, g)
    DynamicalSDEProblem{iip}(ff, v0, u0, tspan, p; kwargs...)
end

function DynamicalSDEProblem(f::DynamicalSDEFunction, v0, u0, tspan,
    p = NullParameters(); kwargs...)
    DynamicalSDEProblem{isinplace(f)}(f, v0, u0, tspan, p; kwargs...)
end

function DynamicalSDEProblem{iip}(f::DynamicalSDEFunction, v0, u0, tspan,
    p = NullParameters();
    func_cache = nothing, kwargs...) where {iip}
    if f.cache === nothing && iip
        cache = similar(u0)
        _f = DynamicalSDEFunction{iip}(f.f1, f.f2, f.g; mass_matrix = f.mass_matrix,
            _func_cache = cache, analytic = f.analytic)
    else
        _f = f
    end
    SDEProblem(_f, ArrayPartition(v0, u0), tspan, p; kwargs...)
end
