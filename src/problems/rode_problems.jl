@doc doc"""

Defines a random ordinary differential equation (RODE) problem.
Documentation Page: [https://docs.sciml.ai/DiffEqDocs/stable/types/rode_types/](https://docs.sciml.ai/DiffEqDocs/stable/types/rode_types/)

## Mathematical Specification of a RODE Problem

To define a RODE Problem, you simply need to give the function ``f`` and the initial
condition ``u_0`` which define an ODE:

```math
\frac{du}{dt} = f(u,p,t,W(t))
```

where `W(t)` is a random process. `f` should be specified as `f(u,p,t,W)`
(or in-place as `f(du,u,p,t,W)`), and `u₀` should be an AbstractArray (or number)
whose geometry matches the desired geometry of `u`. Note that we are not limited
to numbers or vectors for `u₀`; one is allowed to provide `u₀` as arbitrary matrices
/ higher dimension tensors as well.

### Constructors

- `RODEProblem(f::RODEFunction,u0,tspan,p=NullParameters();noise=WHITE_NOISE,rand_prototype=nothing,callback=nothing)`
- `RODEProblem{isinplace,specialize}(f,u0,tspan,p=NullParameters();noise=WHITE_NOISE,rand_prototype=nothing,callback=nothing,mass_matrix=I)` :
  Defines the RODE with the specified functions. The default noise is `WHITE_NOISE`.
  `isinplace` optionally sets whether the function is inplace or not. This is
  determined automatically, but not inferred. `specialize` optionally controls
  the specialization level. See the [specialization levels section of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Levels)
  for more details. The default is `AutoSpecialize.

For more details on the in-place and specialization controls, see the ODEFunction documentation.

Parameters are optional, and if not given, then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters. Any extra keyword arguments are passed on to the solvers. For example,
if you set a `callback` in the problem, then that `callback` will be added in
every solve call.

For specifying Jacobians and mass matrices, see the
[DiffEqFunctions](@ref performance_overloads)
page.

### Fields

* `f`: The drift function in the SDE.
* `u0`: The initial condition.
* `tspan`: The timespan for the problem.
* `p`: The optional parameters for the problem. Defaults to `NullParameters`.
* `noise`: The noise process applied to the noise upon generation. Defaults to
  Gaussian white noise. For information on defining different noise processes,
  see [the noise process documentation page](@ref noise_process)
* `rand_prototype`: A prototype type instance for the noise vector. It defaults
  to `nothing`, which means the problem should be interpreted as having a noise
  vector whose size matches `u0`.
* `kwargs`: The keyword arguments passed onto the solves.
"""
mutable struct RODEProblem{uType, tType, isinplace, P, NP, F, K, ND} <:
               AbstractRODEProblem{uType, tType, isinplace, ND}
    f::F
    u0::uType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    rand_prototype::ND
    seed::UInt64
    @add_kwonly function RODEProblem{iip}(f::RODEFunction{iip}, u0, tspan,
            p = NullParameters();
            rand_prototype = nothing,
            noise = nothing, seed = UInt64(0),
            kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(p),
            typeof(noise), typeof(f), typeof(kwargs),
            typeof(rand_prototype)}(f, _u0, _tspan, p, noise, kwargs,
            rand_prototype, seed)
    end
    function RODEProblem{iip}(f, u0, tspan, p = NullParameters(); kwargs...) where {iip}
        RODEProblem(RODEFunction{iip}(f), u0, tspan, p; kwargs...)
    end
end

function RODEProblem(f::RODEFunction, u0, tspan, p = NullParameters(); kwargs...)
    RODEProblem{isinplace(f)}(f, u0, tspan, p; kwargs...)
end

function RODEProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    RODEProblem(RODEFunction(f), u0, tspan, p; kwargs...)
end

@doc doc"""

Holds information on what variables to alias
when solving an RODEProblem. Conforms to the AbstractAliasSpecifier interface. 
    `RODEAliasSpecifier(;alias_p = nothing, alias_f = nothing, alias_u0 = false, alias_du0 = false, alias_tstops = false, alias = nothing)`

When a keyword argument is `nothing`, the default behaviour of the solver is used.

### Keywords 
* `alias_p::Union{Bool, Nothing}`
* `alias_f::Union{Bool, Nothing}`
* `alias_u0::Union{Bool, Nothing}`: alias the u0 array. Defaults to false .
* `alias_du0::Union{Bool, Nothing}`: alias the du0 array for DAEs. Defaults to false.
* `alias_tstops::Union{Bool, Nothing}`: alias the tstops array
* `alias_noise::Union{Bool,Nothing}`: alias the noise process
* `alias_jumps::Union{Bool, Nothing}`: alias jump process if wrapped in a JumpProcess
* `alias::Union{Bool, Nothing}`: sets all fields of the `RODEAliasSpecifier` to `alias`

"""
struct RODEAliasSpecifier <: AbstractAliasSpecifier
    alias_p::Union{Bool, Nothing}
    alias_f::Union{Bool, Nothing}
    alias_u0::Union{Bool, Nothing}
    alias_du0::Union{Bool, Nothing}
    alias_tstops::Union{Bool, Nothing}
    alias_noise::Union{Bool, Nothing}
    alias_jumps::Union{Bool, Nothing}

    function RODEAliasSpecifier(; alias_p = nothing, alias_f = nothing, alias_u0 = nothing,
            alias_du0 = nothing, alias_tstops = nothing, alias_noise = nothing,
            alias_jumps = nothing, alias = nothing)
        if alias == true
            new(true, true, true, true, true, true, true)
        elseif alias == false
            new(false, false, false, false, false, false, false)
        elseif isnothing(alias)
            new(alias_p, alias_f, alias_u0, alias_du0,
                alias_tstops, alias_noise, alias_jumps)
        end
    end
end
