const RECOMPILE_BY_DEFAULT = true

abstract type AbstractSpecialization end
struct NoSpecialize <: AbstractSpecialization end
struct FunctionWrapperSpecialize <: AbstractSpecialization end
struct FullSpecialize <: AbstractSpecialization end

function DEFAULT_OBSERVED(sym, u, p, t)
    error("Indexing symbol $sym is unknown.")
end

function DEFAULT_OBSERVED_NO_TIME(sym, u, p)
    error("Indexing symbol $sym is unknown.")
end

function Base.summary(io::IO, prob::AbstractSciMLFunction)
    type_color, no_color = get_colorizers(io)
    print(io,
          type_color, nameof(typeof(prob)),
          no_color, ". In-place: ",
          type_color, isinplace(prob),
          no_color)
end

const NONCONFORMING_FUNCTIONS_ERROR_MESSAGE = """
                                              Nonconforming functions detected. If a model function `f` is defined
                                              as in-place, then all constituant functions like `jac` and `paramjac`
                                              must be in-place (and vice versa with out-of-place). Detected that
                                              some overloads did not conform to the same convention as `f`.
                                              """

struct NonconformingFunctionsError <: Exception
    nonconforming::Vector{String}
end

function Base.showerror(io::IO, e::NonconformingFunctionsError)
    println(io, NONCONFORMING_FUNCTIONS_ERROR_MESSAGE)
    print(io, "Nonconforming functions: ")
    printstyled(io, e.nonconforming; bold = true, color = :red)
end

"""
$(TYPEDEF)
"""
abstract type AbstractODEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    ODEFunction{iip,F,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,S2,O,TCV} <: AbstractODEFunction{iip,recompile}

A representation of an ODE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
ODEFunction{iip,recompile}(f;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           syms = __has_syms(f) ? f.syms : nothing,
                           indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

`iip` is the the optional boolean for determining whether a given function is written to
be used in-place or out-of-place. In-place functions are `f!(du,u,p,t)` where the return
is ignored and the result is expected to be mutated into the value of `du`. Out-of-place
functions are `du=f(u,p,t)`.

Normally this is determined automatically by looking at the method table for `f` and seeing
the maximum number of arguments in available dispatches. For this reason, the constructor
`ODEFunction(f)` generally works (but is type-unstable). However, for type-stability or
to enforce correctness, this option is passed via `ODEFunction{true}(f)`.

## recompile: Controlling Compilation and Specialization

The `recompile` parameter controls whether the ODEFunction will fully specialize on the
`typeof(f)`. This causes recompilation of the solver for each new `f` function, but
gives the maximum compiler information and runtime speed. By default `recompile = true`.
If `recompile = false`, the `ODEFunction` uses `Any` type parameters for each of the
functions, allowing for the reuse of compilation caches but adding a dynamic dispatch
at the `f` call sites, potentially leading to runtime regressions.

Overriding the `true` default is done by passing a second type parameter after `iip`,
for example `ODEFunction{true,false}(f)` is an in-place function with no recompilation
specialization.

## Fields

The fields of the ODEFunction type directly match the names of the inputs.

## More Details on Jacobians

The following example creates an inplace `ODEFunction` whose jacobian is a `Diagonal`:

```julia
using LinearAlgebra
f = (du,u,p,t) -> du .= t .* u
jac = (J,u,p,t) -> (J[1,1] = t; J[2,2] = t; J)
jp = Diagonal(zeros(2))
fun = ODEFunction(f; jac=jac, jac_prototype=jp)
```

Note that the integrators will always make a deep copy of `fun.jac_prototype`, so
there's no worry of aliasing.

In general the jacobian prototype can be anything that has `mul!` defined, in
particular sparse matrices or custom lazy types that support `mul!`. A special case
is when the `jac_prototype` is a `AbstractDiffEqLinearOperator`, in which case you
do not need to supply `jac` as it is automatically set to `update_coefficients!`.
Refer to the AbstractSciMLOperators documentation for more information
on setting up time/parameter dependent operators.

## Examples

### Declaring Explicit Jacobians for ODEs

The most standard case, declaring a function for a Jacobian is done by overloading
the function `f(du,u,p,t)` with an in-place updating function for the Jacobian:
`f_jac(J,u,p,t)` where the value type is used for dispatch. For example,
take the LotkaVolterra model:

```julia
function f(du,u,p,t)
  du[1] = 2.0 * u[1] - 1.2 * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end
```

To declare the Jacobian we simply add the dispatch:

```julia
function f_jac(J,u,p,t)
  J[1,1] = 2.0 - 1.2 * u[2]
  J[1,2] = -1.2 * u[1]
  J[2,1] = 1 * u[2]
  J[2,2] = -3 + u[1]
  nothing
end
```

Then we can supply the Jacobian with our ODE as:

```julia
ff = ODEFunction(f;jac=f_jac)
```

and use this in an `ODEProblem`:

```julia
prob = ODEProblem(ff,ones(2),(0.0,10.0))
```

## Symbolically Generating the Functions

See the `modelingtoolkitize` function from
[ModelingToolkit.jl](https://github.com/JuliaDiffEq/ModelingToolkit.jl) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions.
"""
struct ODEFunction{iip, recompile, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, S,
                   S2, O, TCV,
                   SYS} <: AbstractODEFunction{iip}
    f::F
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    indepsym::S2
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
    SplitFunction{iip,F1,F2,TMM,C,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractODEFunction{iip,recompile}

A representation of a split ODE function `f`, defined by:

```math
M \frac{du}{dt} = f_1(u,p,t) + f_2(u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

Generally, for ODE integrators the `f_1` portion should be considered the
"stiff portion of the model" with larger time scale separation, while the
`f_2` portion should be considered the "non-stiff portion". This interpretation
is directly used in integrators like IMEX (implicit-explicit integrators)
and exponential integrators.

## Constructor

```julia
SplitFunction{iip,recompile}(f1,f2;
                             mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                             analytic = __has_analytic(f) ? f.analytic : nothing,
                             tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                             jac = __has_jac(f) ? f.jac : nothing,
                             jvp = __has_jvp(f) ? f.jvp : nothing,
                             vjp = __has_vjp(f) ? f.vjp : nothing,
                             jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                             sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                             paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                             syms = __has_syms(f) ? f.syms : nothing,
                             indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                             colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                             sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,p,t)` or `du = f_i(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f_1(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df_1}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df_1}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df_1}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df_1}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## Note on the Derivative Definition

The derivatives, such as the Jacobian, are only defined on the `f1` portion of the split ODE.
This is used to treat the `f1` implicit while keeping the `f2` portion explicit.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the SplitFunction type directly match the names of the inputs.

## Symbolically Generating the Functions

See the `modelingtoolkitize` function from
[ModelingToolkit.jl](https://github.com/JuliaDiffEq/ModelingToolkit.jl) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions. See `ModelingToolkit.SplitODEProblem` for
information on generating the SplitFunction from this symbolic engine.
"""
struct SplitFunction{iip, recompile, F1, F2, TMM, C, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt,
                     TPJ, S, O,
                     TCV, SYS} <: AbstractODEFunction{iip}
    f1::F1
    f2::F2
    mass_matrix::TMM
    cache::C
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
    DynamicalODEFunction{iip,F1,F2,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractODEFunction{iip,recompile}

A representation of an ODE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,p,t)
```

as a partitioned ODE:

```math
M_1 \frac{du}{dt} = f_1(u,p,t)
M_2 \frac{du}{dt} = f_2(u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DynamicalODEFunction{iip,recompile}(f1,f2;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                    syms = __has_syms(f) ? f.syms : nothing,
                                    indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                                    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                    sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,p,t)` or `du = f_i(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M_i` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator. Should be given as a tuple
  of mass matrices, i.e. `(M_1, M_2)` for the mass matrices of equations 1 and 2
  respectively.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalODEFunction type directly match the names of the inputs.
"""
struct DynamicalODEFunction{iip, recompile, F1, F2, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW,
                            TWt, TPJ, S,
                            O, TCV, SYS} <: AbstractODEFunction{iip}
    f1::F1
    f2::F2
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractDDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    DDEFunction{iip,F,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractDDEFunction{iip,recompile}

A representation of a DDE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,h,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DDEFunction{iip,recompile}(f;
                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                 analytic = __has_analytic(f) ? f.analytic : nothing,
                 tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                 jac = __has_jac(f) ? f.jac : nothing,
                 jvp = __has_jvp(f) ? f.jvp : nothing,
                 vjp = __has_vjp(f) ? f.vjp : nothing,
                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                 syms = __has_syms(f) ? f.syms : nothing,
                 indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,h,p,t)` or `du = f(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The histroy function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,h,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,h,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,h,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,h,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,h,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DDEFunction type directly match the names of the inputs.
"""
struct DDEFunction{iip, recompile, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, S, O,
                   TCV, SYS
                   } <:
       AbstractDDEFunction{iip}
    f::F
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
    DynamicalDDEFunction{iip,F1,F2,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractDDEFunction{iip,recompile}

A representation of a DDE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,h,p,t)
```

as a partitioned ODE:

```math
M_1 \frac{du}{dt} = f_1(u,h,p,t)
M_2 \frac{du}{dt} = f_2(u,h,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DynamicalDDEFunction{iip,recompile}(f1,f2;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                    syms = __has_syms(f) ? f.syms : nothing,
                                    indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                                    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                    sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,h,p,t)` or `du = f_i(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The histroy function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M_i` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator. Should be given as a tuple
  of mass matrices, i.e. `(M_1, M_2)` for the mass matrices of equations 1 and 2
  respectively.
- `analytic(u0,h,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,h,p,t)` or dT=tgrad(u,h,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,h,p,t)` or `J=jac(u,h,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,h,p,t)` or `Jv=jvp(v,u,h,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,h,p,t)` or `Jv=vjp(v,u,h,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,h,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalDDEFunction type directly match the names of the inputs.
"""
struct DynamicalDDEFunction{iip, recompile, F1, F2, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW,
                            TWt, TPJ, S,
                            O, TCV, SYS} <: AbstractDDEFunction{iip}
    f1::F1
    f2::F2
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractDiscreteFunction{iip} <:
              AbstractDiffEqFunction{iip} end

@doc doc"""
    DiscreteFunction{iip,F,Ta,S,O} <: AbstractDiscreteFunction{iip,recompile}

A representation of an discrete dynamical system `f`, defined by:

```math
u_{n+1} = f(u,p,t_{n+1})
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DiscreteFunction{iip,recompile}(f;
                                analytic = __has_analytic(f) ? f.analytic : nothing,
                                syms = __has_syms(f) ? f.syms : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DiscreteFunction type directly match the names of the inputs.
"""
struct DiscreteFunction{iip, recompile, F, Ta, S, O, SYS} <:
       AbstractDiscreteFunction{iip}
    f::F
    analytic::Ta
    syms::S
    observed::O
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractSDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    SDEFunction{iip,F,G,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,GG,S,O,TCV} <: AbstractSDEFunction{iip,recompile}

A representation of an SDE function `f`, defined by:

```math
M du = f(u,p,t)dt + g(u,p,t) dW
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
SDEFunction{iip,recompile}(f,g;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           ggprime = nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           syms = __has_syms(f) ? f.syms : nothing,
                           indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `ggprime(J,u,p,t)` or `J = ggprime(u,p,t)`: returns the Milstein derivative
  ``\frac{dg(u,p,t)}{du} g(u,p,t)``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the ODEFunction type directly match the names of the inputs.
"""
struct SDEFunction{iip, recompile, F, G, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ,
                   GG, S, O,
                   TCV, SYS
                   } <: AbstractSDEFunction{iip}
    f::F
    g::G
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    ggprime::GG
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
    SplitSDEFunction{iip,F1,F2,G,TMM,C,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractSDEFunction{iip,recompile}

A representation of a split SDE function `f`, defined by:

```math
M \frac{du}{dt} = f_1(u,p,t) + f_2(u,p,t) + g(u,p,t) dW
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

Generally, for SDE integrators the `f_1` portion should be considered the
"stiff portion of the model" with larger time scale separation, while the
`f_2` portion should be considered the "non-stiff portion". This interpretation
is directly used in integrators like IMEX (implicit-explicit integrators)
and exponential integrators.

## Constructor

```julia
SplitSDEFunction{iip,recompile}(f1,f2,g;
                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                 analytic = __has_analytic(f) ? f.analytic : nothing,
                 tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                 jac = __has_jac(f) ? f.jac : nothing,
                 jvp = __has_jvp(f) ? f.jvp : nothing,
                 vjp = __has_vjp(f) ? f.vjp : nothing,
                 ggprime = nothing,
                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                 syms = __has_syms(f) ? f.syms : nothing,
                 indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. All of the remaining functions
are optional for improving or accelerating the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the SDE function. Can be used
  to determine that the equation is actually a stochastic differential-algebraic equation (SDAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/sdae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f_1(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df_1}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df_1}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df_1}{du}^\ast v``
- `ggprime(J,u,p,t)` or `J = ggprime(u,p,t)`: returns the Milstein derivative
  ``\frac{dg(u,p,t)}{du} g(u,p,t)``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df_1}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## Note on the Derivative Definition

The derivatives, such as the Jacobian, are only defined on the `f1` portion of the split ODE.
This is used to treat the `f1` implicit while keeping the `f2` portion explicit.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the SplitSDEFunction type directly match the names of the inputs.
"""
struct SplitSDEFunction{iip, recompile, F1, F2, G, TMM, C, Ta, Tt, TJ, JVP, VJP, JP, SP, TW,
                        TWt, TPJ,
                        S, O, TCV, SYS} <: AbstractSDEFunction{iip}
    f1::F1
    f2::F2
    g::G
    mass_matrix::TMM
    cache::C
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
    DynamicalSDEFunction{iip,F1,F2,G,TMM,C,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractSDEFunction{iip,recompile}

A representation of an SDE function `f` and `g`, defined by:

```math
M du = f(u,p,t) dt + g(u,p,t) dW_t
```

as a partitioned ODE:

```math
M_1 du = f_1(u,p,t) dt + g(u,p,t) dW_t
M_2 du = f_2(u,p,t) dt + g(u,p,t) dW_t
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DynamicalSDEFunction{iip,recompile}(f1,f2;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    ggprime=nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                    syms = __has_syms(f) ? f.syms : nothing,
                                    indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                                    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                    sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,p,t)` or `du = f_i(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M_i` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator. Should be given as a tuple
  of mass matrices, i.e. `(M_1, M_2)` for the mass matrices of equations 1 and 2
  respectively.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `ggprime(J,u,p,t)` or `J = ggprime(u,p,t)`: returns the Milstein derivative
  ``\frac{dg(u,p,t)}{du} g(u,p,t)``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalSDEFunction type directly match the names of the inputs.
"""
struct DynamicalSDEFunction{iip, recompile, F1, F2, G, TMM, C, Ta, Tt, TJ, JVP, VJP, JP, SP,
                            TW, TWt,
                            TPJ, S, O, TCV, SYS} <: AbstractSDEFunction{iip}
    # This is a direct copy of the SplitSDEFunction, maybe it's not necessary and the above can be used instead.
    f1::F1
    f2::F2
    g::G
    mass_matrix::TMM
    cache::C
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractRODEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    RODEFunction{iip,F,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractRODEFunction{iip,recompile}

A representation of an RODE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,p,t,W)dt
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
RODEFunction{iip,recompile}(f;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           syms = __has_syms(f) ? f.syms : nothing,
                           indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the RODEFunction type directly match the names of the inputs.
"""
struct RODEFunction{iip, recompile, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, S,
                    O, TCV, SYS
                    } <:
       AbstractRODEFunction{iip}
    f::F
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractDAEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    DAEFunction{iip,F,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractDAEFunction{iip,recompile}

A representation of an implicit DAE function `f`, defined by:

```math
0 = f(\frac{du}{dt},u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DAEFunction{iip,recompile}(f;
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           syms = __has_syms(f) ? f.syms : nothing,
                           indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(out,du,u,p,t)` or `out = f(du,u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `jac(J,du,u,p,gamma,t)` or `J=jac(du,u,p,gamma,t)`: returns the implicit DAE Jacobian
  defined as ``gamma \frac{dG}{d(du)} + \frac{dG}{du}``
- `jvp(Jv,v,du,u,p,gamma,t)` or `Jv=jvp(v,du,u,p,gamma,t)`: returns the directional
  derivative``\frac{df}{du} v``
- `vjp(Jv,v,du,u,p,gamma,t)` or `Jv=vjp(v,du,u,p,gamma,t)`: returns the adjoint
  derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DAEFunction type directly match the names of the inputs.

## Examples


### Declaring Explicit Jacobians for DAEs

For fully implicit ODEs (`DAEProblem`s), a slightly different Jacobian function
is necessary. For the DAE

```math
G(du,u,p,t) = res
```

The Jacobian should be given in the form `gamma*dG/d(du) + dG/du ` where `gamma`
is given by the solver. This means that the signature is:

```julia
f(J,du,u,p,gamma,t)
```

For example, for the equation

```julia
function testjac(res,du,u,p,t)
  res[1] = du[1] - 2.0 * u[1] + 1.2 * u[1]*u[2]
  res[2] = du[2] -3 * u[2] - u[1]*u[2]
end
```

we would define the Jacobian as:

```julia
function testjac(J,du,u,p,gamma,t)
  J[1,1] = gamma - 2.0 + 1.2 * u[2]
  J[1,2] = 1.2 * u[1]
  J[2,1] = - 1 * u[2]
  J[2,2] = gamma - 3 - u[1]
  nothing
end
```

## Symbolically Generating the Functions

See the `modelingtoolkitize` function from
[ModelingToolkit.jl](https://github.com/JuliaDiffEq/ModelingToolkit.jl) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions.
"""
struct DAEFunction{iip, recompile, F, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, S, O, TCV,
                   SYS} <:
       AbstractDAEFunction{iip}
    f::F
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractSDDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
    SDDEFunction{iip,F,G,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,GG,S,O,TCV} <: AbstractSDDEFunction{iip,recompile}

A representation of a SDDE function `f`, defined by:

```math
M du = f(u,h,p,t) dt + g(u,h,p,t) dW_t
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
SDDEFunction{iip,recompile}(f,g;
                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                 analytic = __has_analytic(f) ? f.analytic : nothing,
                 tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                 jac = __has_jac(f) ? f.jac : nothing,
                 jvp = __has_jvp(f) ? f.jvp : nothing,
                 vjp = __has_vjp(f) ? f.vjp : nothing,
                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                 syms = __has_syms(f) ? f.syms : nothing,
                 indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,h,p,t)` or `du = f(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The histroy function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://diffeq.sciml.ai/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,h,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,u,h,p,t)` or `J=jac(u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,h,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,h,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,h,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DDEFunction type directly match the names of the inputs.
"""
struct SDDEFunction{iip, recompile, F, G, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ,
                    GG, S, O,
                    TCV, SYS} <: AbstractSDDEFunction{iip}
    f::F
    g::G
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    ggprime::GG
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractNonlinearFunction{iip, recompile} <:
              AbstractSciMLFunction{iip, recompile} end

@doc doc"""
    NonlinearFunction{iip,F,TMM,Ta,Tt,TJ,JVP,VJP,JP,SP,TW,TWt,TPJ,S,O,TCV} <: AbstractNonlinearFunction{iip,recompile}

A representation of an nonlinear system of equations `f`, defined by:

```math
0 = f(u,p)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
NonlinearFunction{iip, recompile}(f;
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           syms = __has_syms(f) ? f.syms : nothing,
                           indepsym= __has_indepsym(f) ? f.indepsym : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p)` or `du = f(u,p)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(u0,p)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `jac(J,u,p)` or `J=jac(u,p)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p)` or `Jv=jvp(v,u,p)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p)` or `Jv=vjp(v,u,p)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u0 = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `indepsym`: the canonical naming for the independent variable. Defaults to nothing, which
  internally uses `t` as the representation in any plots.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the NonlinearFunction type directly match the names of the inputs.
"""
struct NonlinearFunction{iip, recompile, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ,
                         S, O, TCV,
                         SYS
                         } <: AbstractNonlinearFunction{iip}
    f::F
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    syms::S
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
    OptimizationFunction{iip,AD,F,G,H,HV,C,CJ,CH,HP,CJP,CHP,S,HCV,CJCV,CHCV} <: AbstractOptimizationFunction{iip,recompile}

A representation of an optimization of an objective function `f`, defined by:

```math
\\min_{u} f(u,p)
```

and all of its related functions, such as the gradient of `f`, its Hessian,
and more. For all cases, `u` is the state and `p` are the parameters.

## Constructor

```julia
OptimizationFunction{iip}(f, adtype::AbstractADType = NoAD();
                          grad = nothing, hess = nothing, hv = nothing,
                          cons = nothing, cons_j = nothing, cons_h = nothing,
                          hess_prototype = nothing, cons_jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                          cons_hess_prototype = nothing,
                          syms = __has_syms(f) ? f.syms : nothing, hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          cons_jac_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          cons_hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          sys = __has_sys(f) ? f.sys : nothing)
```

- `adtype`: see the section "Defining Optimization Functions via AD"
- `grad(G,u,p)` or `G=grad(u,p)`: the gradient of `f` with respect to `u`
- `hess(H,u,p)` or `H=hess(u,p)`: the Hessian of `f` with respect to `u`
- `hv(Hv,u,v,p)` or `Hv=hv(u,v,p)`: the Hessian-vector product ``(d^2 f / du^2) v``.
- `cons(res,x,p)` or `cons(x,p)` : the constraints function, should mutate or return a vector
    with value of the `i`th constraint, evaluated at the current values of variables
    inside the optimization routine. This takes just the function evaluations
    and the equality or inequality assertion is applied by the solver based on the constraint
    bounds passed as `lcons` and `ucons` to [`OptimizationProblem`](@ref), in case of equality
    constraints `lcons` and `ucons` should be passed equal values.
- `cons_j(res,x,p)` or `res=cons_j(x,p)`: the Jacobian of the constraints.
- `cons_h(res,x,p)` or `res=cons_h(x,p)`: the Hessian of the constraints, provided as
   an array of Hessians with `res[i]` being the Hessian with respect to the `i`th output on `cons`.
- `paramjac(pJ,u,p)`: returns the parameter Jacobian ``df/dp``.
- `hess_prototype`: a prototype matrix matching the type that matches the Hessian. For example,
  if the Hessian is tridiagonal, then an appropriately sized `Hessian` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Hessian.
  The default is `nothing`, which means a dense Hessian.
- `cons_jac_prototype`: a prototype matrix matching the type that matches the constraint Jacobian.
  The default is `nothing`, which means a dense constraint Jacobian.
- `cons_hess_prototype`: a prototype matrix matching the type that matches the constraint Hessian.
  This is defined as an array of matrices, where `hess[i]` is the Hessian w.r.t. the `i`th output.
  For example, if the Hessian is sparse, then `hess` is a `Vector{SparseMatrixCSC}`.
  The default is `nothing`, which means a dense constraint Hessian.
- `syms`: the symbol names for the elements of the equation. This should match `u0` in size. For
  example, if `u = [0.0,1.0]` and `syms = [:x, :y]`, this will apply a canonical naming to the
  values, allowing `sol[:x]` in the solution and automatically naming values in plots.
- `hess_colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `hess_prototype`. This specializes the Hessian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.
- `cons_jac_colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `cons_jac_prototype`.
- `cons_hess_colorvec`: an array of color vector according to the SparseDiffTools.jl definition for
  the sparsity pattern of the `cons_hess_prototype`.

## Defining Optimization Functions Via AD

While using the keyword arguments gives the user control over defining
all of the possible functions, the simplest way to handle the generation
of an `OptimizationFunction` is by specifying an AD type. By doing so,
this will automatically fill in all of the extra functions. For example,

```julia
OptimizationFunction(f,AutoZygote())
```

will use [Zygote.jl](https://github.com/FluxML/Zygote.jl) to define
all of the necessary functions. Note that if any functions are defined
directly, the auto-AD definition does not overwrite the user's choice.

Each of the AD-based constructors are documented separately via their
own dispatches.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## recompile: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the OptimizationFunction type directly match the names of the inputs.
"""
struct OptimizationFunction{iip, AD, F, G, H, HV, C, CJ, CH, HP, CJP, CHP, S, HCV, CJCV,
                            CHCV, EX, CEX, SYS} <: AbstractOptimizationFunction{iip}
    f::F
    adtype::AD
    grad::G
    hess::H
    hv::HV
    cons::C
    cons_j::CJ
    cons_h::CH
    hess_prototype::HP
    cons_jac_prototype::CJP
    cons_hess_prototype::CHP
    syms::S
    hess_colorvec::HCV
    cons_jac_colorvec::CJCV
    cons_hess_colorvec::CHCV
    expr::EX
    cons_expr::CEX
    sys::SYS
end

######### Backwards Compatibility Overloads

(f::ODEFunction)(args...) = f.f(args...)
(f::NonlinearFunction)(args...) = f.f(args...)

function (f::DynamicalODEFunction)(u, p, t)
    ArrayPartition(f.f1(u.x[1], u.x[2], p, t), f.f2(u.x[1], u.x[2], p, t))
end
function (f::DynamicalODEFunction)(du, u, p, t)
    f.f1(du.x[1], u.x[1], u.x[2], p, t)
    f.f2(du.x[2], u.x[1], u.x[2], p, t)
end

(f::SplitFunction)(u, p, t) = f.f1(u, p, t) + f.f2(u, p, t)
function (f::SplitFunction)(du, u, p, t)
    f.f1(f.cache, u, p, t)
    f.f2(du, u, p, t)
    du .+= f.cache
end

(f::DiscreteFunction)(args...) = f.f(args...)
(f::DAEFunction)(args...) = f.f(args...)
(f::DDEFunction)(args...) = f.f(args...)

function (f::DynamicalDDEFunction)(u, h, p, t)
    ArrayPartition(f.f1(u.x[1], u.x[2], h, p, t), f.f2(u.x[1], u.x[2], h, p, t))
end
function (f::DynamicalDDEFunction)(du, u, h, p, t)
    f.f1(du.x[1], u.x[1], u.x[2], h, p, t)
    f.f2(du.x[2], u.x[1], u.x[2], h, p, t)
end
function Base.getproperty(f::DynamicalDDEFunction, name::Symbol)
    if name === :f
        # Use the f property as an alias for calling the function itself, so DynamicalDDEFunction fits the same interface as DDEFunction as expected by the ODEFunctionWrapper in DelayDiffEq.jl.
        return f
    end
    return getfield(f, name)
end

(f::SDEFunction)(args...) = f.f(args...)
(f::SDDEFunction)(args...) = f.f(args...)
(f::SplitSDEFunction)(u, p, t) = f.f1(u, p, t) + f.f2(u, p, t)

function (f::SplitSDEFunction)(du, u, p, t)
    f.f1(f.cache, u, p, t)
    f.f2(du, u, p, t)
    du .+= f.cache
end

(f::RODEFunction)(args...) = f.f(args...)

######### Basic Constructor

function ODEFunction{iip}(f;
                          mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                          analytic = __has_analytic(f) ? f.analytic : nothing,
                          tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                          jac = __has_jac(f) ? f.jac : nothing,
                          jvp = __has_jvp(f) ? f.jvp : nothing,
                          vjp = __has_vjp(f) ? f.vjp : nothing,
                          jac_prototype = __has_jac_prototype(f) ?
                                          f.jac_prototype :
                                          nothing,
                          sparsity = __has_sparsity(f) ? f.sparsity :
                                     jac_prototype,
                          Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                          Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                          paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                          syms = __has_syms(f) ? f.syms : nothing,
                          indepsym = __has_indepsym(f) ? f.indepsym : nothing,
                          observed = __has_observed(f) ? f.observed :
                                     DEFAULT_OBSERVED,
                          colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                       recompile}
    if mass_matrix === I && typeof(f) <: Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if recompile === FunctionWrapperSpecialize &&
       !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
        if iip
            f = wrapfun_iip(f)
        else
            f = wrapfun_oop(f)
        end
    end

    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    jaciip = jac !== nothing ? isinplace(jac, 4, "jac", iip) : iip
    tgradiip = tgrad !== nothing ? isinplace(tgrad, 4, "tgrad", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 5, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 5, "vjp", iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact, 5, "Wfact", iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t, 5, "Wfact_t", iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac, 4, "paramjac", iip) : iip

    nonconforming = (jaciip, tgradiip, jvpiip, vjpiip, Wfactiip, Wfact_tiip,
                     paramjaciip) .!= iip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["jac", "tgrad", "jvp", "vjp", "Wfact", "Wfact_t", "paramjac"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if recompile === NoSpecialize
        ODEFunction{iip, recompile,
                    Any, Any, Any, Any,
                    Any, Any, Any, typeof(jac_prototype),
                    typeof(sparsity), Any, Any, Any,
                    typeof(syms), typeof(indepsym), Any, typeof(_colorvec),
                    typeof(sys)}(f, mass_matrix, analytic, tgrad, jac,
                                 jvp, vjp, jac_prototype, sparsity, Wfact,
                                 Wfact_t, paramjac, syms, indepsym,
                                 observed, _colorvec, sys)
    else
        ODEFunction{iip, recompile,
                    typeof(f), typeof(mass_matrix), typeof(analytic), typeof(tgrad),
                    typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                    typeof(sparsity), typeof(Wfact), typeof(Wfact_t), typeof(paramjac),
                    typeof(syms), typeof(indepsym), typeof(observed), typeof(_colorvec),
                    typeof(sys)}(f, mass_matrix, analytic, tgrad, jac,
                                 jvp, vjp, jac_prototype, sparsity, Wfact,
                                 Wfact_t, paramjac, syms, indepsym,
                                 observed, _colorvec, sys)
    end
end

function ODEFunction{iip}(f; kwargs...) where {iip}
    ODEFunction{iip, FullSpecialize}(f; kwargs...)
end
ODEFunction{iip}(f::ODEFunction; kwargs...) where {iip} = f
ODEFunction(f; kwargs...) = ODEFunction{isinplace(f, 4), FullSpecialize}(f; kwargs...)
ODEFunction(f::ODEFunction; kwargs...) = f

function unwrapped_f(f::ODEFunction)
    ff = unwrapped_f(f.f)
    if specialization(f) === NoSpecialize
        ODEFunction{isinplace(f), specialization(f), Any, Any, Any,
                    Any, Any, Any, Any, typeof(f.jac_prototype),
                    typeof(f.sparsity), Any, Any, Any,
                    typeof(f.syms), Any, Any, typeof(f.colorvec),
                    typeof(f.sys)}(ff, f.mass_matrix, f.analytic, f.tgrad, f.jac,
                                   f.jvp, f.vjp, f.jac_prototype, f.sparsity, f.Wfact,
                                   f.Wfact_t, f.paramjac, f.syms, f.indepsym,
                                   f.observed, f.colorvec, f.sys)
    else
        ODEFunction{isinplace(f), specialization(f), typeof(ff), typeof(f.mass_matrix),
                    typeof(f.analytic), typeof(f.tgrad),
                    typeof(f.jac), typeof(f.jvp), typeof(f.vjp), typeof(f.jac_prototype),
                    typeof(f.sparsity), typeof(f.Wfact), typeof(f.Wfact_t),
                    typeof(f.paramjac),
                    typeof(f.syms), typeof(f.indepsym), typeof(f.observed),
                    typeof(f.colorvec),
                    typeof(f.sys)}(ff, f.mass_matrix, f.analytic, f.tgrad, f.jac,
                                   f.jvp, f.vjp, f.jac_prototype, f.sparsity, f.Wfact,
                                   f.Wfact_t, f.paramjac, f.syms, f.indepsym,
                                   f.observed, f.colorvec, f.sys)
    end
end

"""
$(SIGNATURES)

Converts a NonlinearFunction into a ODEFunction.
"""
function ODEFunction(f::NonlinearFunction)
    iip = isinplace(f)
    _f = iip ? (du, u, p, t) -> (f.f(du, u, p); nothing) : (u, p, t) -> f.f(u, p)
    if f.analytic !== nothing
        _analytic = (u0, p, t) -> f.analytic(u0, p)
    else
        _analytic = nothing
    end
    if f.tgrad !== nothing
        _tgrad = iip ? (dT, u, p, t) -> (f.tgrad(dT, u, p); nothing) :
                 (u, p, t) -> f.tgrad(u, p)
    else
        _tgrad = nothing
    end
    if f.jac !== nothing
        _jac = iip ? (J, u, p, t) -> (f.jac(J, u, p); nothing) : (u, p, t) -> f.jac(u, p)
    else
        _jac = nothing
    end
    if f.jvp !== nothing
        _jvp = iip ? (Jv, u, p, t) -> (f.jvp(Jv, u, p); nothing) : (u, p, t) -> f.jvp(u, p)
    else
        _jvp = nothing
    end
    if f.vjp !== nothing
        _vjp = iip ? (vJ, u, p, t) -> (f.vjp(vJ, u, p); nothing) : (u, p, t) -> f.vjp(u, p)
    else
        _vjp = nothing
    end

    ODEFunction{iip, specialization(f)}(_f;
                                        mass_matrix = f.mass_matrix,
                                        analytic = _analytic,
                                        tgrad = _tgrad,
                                        jac = _jac,
                                        jvp = _jvp,
                                        vjp = _vjp,
                                        jac_prototype = f.jac_prototype,
                                        sparsity = f.sparsity,
                                        paramjac = f.paramjac,
                                        syms = f.syms,
                                        indepsym = nothing,
                                        observed = f.observed,
                                        colorvec = f.colorvec)
end

@add_kwonly function SplitFunction(f1, f2, mass_matrix, cache, analytic, tgrad, jac, jvp,
                                   vjp, jac_prototype, sparsity, Wfact, Wfact_t, paramjac,
                                   syms, observed, colorvec, sys)
    f1 = ODEFunction(f1)
    f2 = ODEFunction(f2)

    if !(typeof(f1) <: AbstractDiffEqOperator || typeof(f1.f) <: AbstractDiffEqOperator) &&
       isinplace(f1) != isinplace(f2)
        throw(NonconformingFunctionsError(["f2"]))
    end

    SplitFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
                  typeof(mass_matrix),
                  typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
                  typeof(vjp), typeof(jac_prototype), typeof(sparsity),
                  typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                  typeof(observed), typeof(colorvec), typeof(sys)
                  }(f1, f2, mass_matrix, cache, analytic, tgrad, jac, jvp, vjp,
                    jac_prototype, sparsity, Wfact, Wfact_t, paramjac, syms,
                    observed, colorvec, sys)
end
function SplitFunction{iip, recompile}(f1, f2;
                                       mass_matrix = __has_mass_matrix(f1) ?
                                                     f1.mass_matrix : I,
                                       _func_cache = nothing,
                                       analytic = __has_analytic(f1) ? f1.analytic :
                                                  nothing,
                                       tgrad = __has_tgrad(f1) ? f1.tgrad : nothing,
                                       jac = __has_jac(f1) ? f1.jac : nothing,
                                       jvp = __has_jvp(f1) ? f1.jvp : nothing,
                                       vjp = __has_vjp(f1) ? f1.vjp : nothing,
                                       jac_prototype = __has_jac_prototype(f1) ?
                                                       f1.jac_prototype :
                                                       nothing,
                                       sparsity = __has_sparsity(f1) ? f1.sparsity :
                                                  jac_prototype,
                                       Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
                                       Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t : nothing,
                                       paramjac = __has_paramjac(f1) ? f1.paramjac :
                                                  nothing,
                                       syms = __has_syms(f1) ? f1.syms : nothing,
                                       observed = __has_observed(f1) ? f1.observed :
                                                  DEFAULT_OBSERVED,
                                       colorvec = __has_colorvec(f1) ? f1.colorvec :
                                                  nothing,
                                       sys = __has_sys(f1) ? f1.sys : nothing) where {iip,
                                                                                      recompile
                                                                                      }
    if recompile === NoSpecialize
        SplitFunction{iip, recompile, Any, Any, Any, Any, Any, Any, Any, Any, Any,
                      Any, Any, Any, Any, Any, Any}(f1, f2, mass_matrix, _func_cache,
                                                    analytic,
                                                    tgrad, jac, jvp, vjp, jac_prototype,
                                                    sparsity, Wfact, Wfact_t, paramjac,
                                                    syms,
                                                    observed, colorvec, sys)
    else
        SplitFunction{iip, recompile, typeof(f1), typeof(f2), typeof(mass_matrix),
                      typeof(_func_cache), typeof(analytic),
                      typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                      typeof(jac_prototype), typeof(sparsity),
                      typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                      typeof(observed),
                      typeof(colorvec),
                      typeof(sys)}(f1, f2, mass_matrix, _func_cache, analytic, tgrad, jac,
                                   jvp, vjp, jac_prototype,
                                   sparsity, Wfact, Wfact_t, paramjac, syms, observed,
                                   colorvec, sys)
    end
end

SplitFunction(f1, f2; kwargs...) = SplitFunction{isinplace(f2, 4)}(f1, f2; kwargs...)
function SplitFunction{iip}(f1, f2; kwargs...) where {iip}
    SplitFunction{iip, FullSpecialize}(ODEFunction(f1), ODEFunction{iip}(f2);
                                       kwargs...)
end
SplitFunction(f::SplitFunction; kwargs...) = f

@add_kwonly function DynamicalODEFunction{iip}(f1, f2, mass_matrix, analytic, tgrad, jac,
                                               jvp, vjp, jac_prototype, sparsity, Wfact,
                                               Wfact_t, paramjac, syms, observed,
                                               colorvec, sys) where {iip}
    f1 = typeof(f1) <: AbstractDiffEqOperator ? f1 : ODEFunction(f1)
    f2 = ODEFunction(f2)

    if isinplace(f1) != isinplace(f2)
        throw(NonconformingFunctionsError(["f2"]))
    end

    DynamicalODEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
                         typeof(mass_matrix),
                         typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
                         typeof(vjp),
                         typeof(jac_prototype),
                         typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                         typeof(observed), typeof(colorvec),
                         typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
                                      vjp, jac_prototype, sparsity, Wfact, Wfact_t,
                                      paramjac, syms, observed, colorvec, sys)
end

function DynamicalODEFunction{iip}(f1, f2;
                                   mass_matrix = __has_mass_matrix(f1) ?
                                                 f1.mass_matrix : I,
                                   analytic = __has_analytic(f1) ? f1.analytic :
                                              nothing,
                                   tgrad = __has_tgrad(f1) ? f1.tgrad : nothing,
                                   jac = __has_jac(f1) ? f1.jac : nothing,
                                   jvp = __has_jvp(f1) ? f1.jvp : nothing,
                                   vjp = __has_vjp(f1) ? f1.vjp : nothing,
                                   jac_prototype = __has_jac_prototype(f1) ?
                                                   f1.jac_prototype : nothing,
                                   sparsity = __has_sparsity(f1) ? f1.sparsity :
                                              jac_prototype,
                                   Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
                                   Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t :
                                             nothing,
                                   paramjac = __has_paramjac(f1) ? f1.paramjac :
                                              nothing,
                                   syms = __has_syms(f1) ? f1.syms : nothing,
                                   observed = __has_observed(f1) ? f1.observed :
                                              DEFAULT_OBSERVED,
                                   colorvec = __has_colorvec(f1) ? f1.colorvec :
                                              nothing,
                                   sys = __has_sys(f1) ? f1.sys : nothing) where {
                                                                                  iip,
                                                                                  recompile
                                                                                  }
    if recompile === NoSpecialize
        DynamicalODEFunction{iip, recompile, Any, Any, Any, Any, Any, Any, Any,
                             Any, Any, Any, Any, Any, Any, Any}(f1, f2, mass_matrix,
                                                                analytic,
                                                                tgrad,
                                                                jac, jvp, vjp,
                                                                jac_prototype,
                                                                sparsity,
                                                                Wfact, Wfact_t, paramjac,
                                                                syms,
                                                                observed, colorvec, sys)
    else
        DynamicalODEFunction{iip, recompile, typeof(f1), typeof(f2), typeof(mass_matrix),
                             typeof(analytic),
                             typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                             typeof(jac_prototype), typeof(sparsity),
                             typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                             typeof(observed),
                             typeof(colorvec),
                             typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
                                          vjp, jac_prototype, sparsity,
                                          Wfact, Wfact_t, paramjac, syms, observed,
                                          colorvec, sys)
    end
end

function DynamicalODEFunction(f1, f2 = nothing; kwargs...)
    DynamicalODEFunction{isinplace(f1, 5)}(f1, f2; kwargs...)
end
function DynamicalODEFunction{iip}(f1, f2; kwargs...) where {iip}
    DynamicalODEFunction{iip, FullSpecialize}(ODEFunction{iip}(f1),
                                              ODEFunction{iip}(f2); kwargs...)
end
DynamicalODEFunction(f::DynamicalODEFunction; kwargs...) = f

function DiscretEFunction{iip}(f;
                               analytic = __has_analytic(f) ? f.analytic :
                                          nothing,
                               syms = __has_syms(f) ? f.syms : nothing,
                               observed = __has_observed(f) ? f.observed :
                                          DEFAULT_OBSERVED,
                               sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                            recompile
                                                                            }
    if recompile === NoSpecialize
        DiscreteFunction{iip, recompile, Any, Any, Any, Any, Any}(f, analytic, syms,
                                                                  observed, sys)
    else
        DiscreteFunction{iip, recompile, typeof(f), typeof(analytic),
                         typeof(syms), typeof(observed), typeof(sys)}(f, analytic, syms,
                                                                      observed, sys)
    end
end

function DiscreteFunction{iip}(f; kwargs...) where {iip}
    DiscreteFunction{iip, FullSpecialize}(f; kwargs...)
end
DiscreteFunction{iip}(f::DiscreteFunction; kwargs...) where {iip} = f
function DiscreteFunction(f; kwargs...)
    DiscreteFunction{isinplace(f, 4), FullSpecialize}(f; kwargs...)
end
DiscreteFunction(f::DiscreteFunction; kwargs...) = f

function unwrapped_f(f::DiscreteFunction)
    ff = unwrapped_f(f)
    recompile = specialization(f)

    if recompile === NoSpecialize
        DiscreteFunction{isinplace(f), typeof(ff), typeof(f.analytic), typeof(f.syms),
                         typeof(f.observed), typeof(f.sys)}(ff, f.analytic, f.syms,
                                                            f.observed,
                                                            f.sys)
    else
        DiscreteFunction{isinplace(f), typeof(ff), typeof(f.analytic), typeof(f.syms),
                         typeof(f.observed), typeof(f.sys)}(ff, f.analytic, f.syms,
                                                            f.observed,
                                                            f.sys)
    end
end

function SDEFunction{iip}(f, g;
                          mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                          analytic = __has_analytic(f) ? f.analytic : nothing,
                          tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                          jac = __has_jac(f) ? f.jac : nothing,
                          jvp = __has_jvp(f) ? f.jvp : nothing,
                          vjp = __has_vjp(f) ? f.vjp : nothing,
                          jac_prototype = __has_jac_prototype(f) ?
                                          f.jac_prototype :
                                          nothing,
                          sparsity = __has_sparsity(f) ? f.sparsity :
                                     jac_prototype,
                          Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                          Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                          paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                          ggprime = nothing,
                          syms = __has_syms(f) ? f.syms : nothing,
                          observed = __has_observed(f) ? f.observed :
                                     DEFAULT_OBSERVED,
                          colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                       recompile}
    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    giip = isinplace(g, 4, "g", iip)
    jaciip = jac !== nothing ? isinplace(jac, 4, "jac", iip) : iip
    tgradiip = tgrad !== nothing ? isinplace(tgrad, 4, "tgrad", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 5, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 5, "vjp", iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact, 5, "Wfact", iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t, 5, "Wfact_t", iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac, 4, "paramjac", iip) : iip

    nonconforming = (giip, jaciip, tgradiip, jvpiip, vjpiip, Wfactiip, Wfact_tiip,
                     paramjaciip) .!= iip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["g", "jac", "tgrad", "jvp", "vjp", "Wfact", "Wfact_t", "paramjac"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if recompile === NoSpecialize
        SDEFunction{iip, Any, Any, Any, Any, Any, Any, Any, Any,
                    Any, Any, Any, Any, typeof(syms), Any,
                    typeof(_colorvec), Any, Any}(f, g, mass_matrix, analytic,
                                                 tgrad, jac, jvp, vjp,
                                                 jac_prototype, sparsity,
                                                 Wfact, Wfact_t, paramjac, ggprime, syms,
                                                 observed, _colorvec, sys)
    else
        SDEFunction{iip, typeof(f), typeof(g),
                    typeof(mass_matrix), typeof(analytic), typeof(tgrad),
                    typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                    typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
                    typeof(paramjac), typeof(ggprime), typeof(syms),
                    typeof(observed), typeof(_colorvec), typeof(sys)}(f, g, mass_matrix,
                                                                      analytic, tgrad, jac,
                                                                      jvp, vjp,
                                                                      jac_prototype,
                                                                      sparsity, Wfact,
                                                                      Wfact_t,
                                                                      paramjac, ggprime,
                                                                      syms,
                                                                      observed, _colorvec,
                                                                      sys)
    end
end

function unwrapped_f(f::SDEFunction)
    ff, g = unwrapped_f(f.f), unwrapped(f.g)
    recompile = specialization(f)

    if recompile === NoSpecialize
        SDEFunction{isinplace(f), Any, Any,
                    typeoff(f.mass_matrix), Any, Any,
                    Any, Any, Any, typeof(f.jac_prototype),
                    typeof(f.sparsity), Any, Any,
                    Any, Any, typeof(f.syms),
                    typeof(f.observed), typeof(f.colorvec), typeof(f.sys)}(ff, g,
                                                                           f.mass_matrix,
                                                                           f.analytic,
                                                                           f.tgrad, f.jac,
                                                                           f.jvp, f.vjp,
                                                                           f.jac_prototype,
                                                                           f.sparsity,
                                                                           f.Wfact,
                                                                           f.Wfact_t,
                                                                           f.paramjac,
                                                                           f.ggprime,
                                                                           f.syms,
                                                                           f.observed,
                                                                           f.colorvec,
                                                                           f.sys)
    else
        SDEFunction{isinplace(f), typeof(ff), typeof(g),
                    typeoff(f.mass_matrix), typeof(f.analytic), typeof(f.tgrad),
                    typeof(f.jac), typeof(f.jvp), typeof(f.vjp), typeof(f.jac_prototype),
                    typeof(f.sparsity), typeof(f.Wfact), typeof(f.Wfact_t),
                    typeof(f.paramjac), typeof(f.ggprime), typeof(f.syms),
                    typeof(f.observed), typeof(f.colorvec), typeof(f.sys)}(ff, g,
                                                                           f.mass_matrix,
                                                                           f.analytic,
                                                                           f.tgrad, f.jac,
                                                                           f.jvp, f.vjp,
                                                                           f.jac_prototype,
                                                                           f.sparsity,
                                                                           f.Wfact,
                                                                           f.Wfact_t,
                                                                           f.paramjac,
                                                                           f.ggprime,
                                                                           f.syms,
                                                                           f.observed,
                                                                           f.colorvec,
                                                                           f.sys)
    end
end

function SDEFunction{iip}(f, g; kwargs...) where {iip}
    SDEFunction{iip, FullSpecialize}(f, g; kwargs...)
end
SDEFunction{iip}(f::SDEFunction, g; kwargs...) where {iip} = f
function SDEFunction(f, g; kwargs...)
    SDEFunction{isinplace(f, 4), FullSpecialize}(f, g; kwargs...)
end
SDEFunction(f::SDEFunction; kwargs...) = f

@add_kwonly function SplitSDEFunction(f1, f2, g, mass_matrix, cache, analytic, tgrad, jac,
                                      jvp, vjp,
                                      jac_prototype, Wfact, Wfact_t, paramjac, observed,
                                      syms, colorvec, sys)
    f1 = typeof(f1) <: AbstractDiffEqOperator ? f1 : SDEFunction(f1)
    f2 = SDEFunction(f2)
    SplitFunction{isinplace(f2), typeof(f1), typeof(f2), typeof(g), typeof(mass_matrix),
                  typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
                  typeof(vjp),
                  typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                  typeof(observed),
                  typeof(colorvec),
                  typeof(sys)}(f1, f2, mass_matrix, cache, analytic, tgrad, jac,
                               jac_prototype, Wfact, Wfact_t, paramjac, syms, observed,
                               colorvec, sys)
end

function SplitSDEFunction{iip}(f1, f2, g;
                               mass_matrix = __has_mass_matrix(f1) ?
                                             f1.mass_matrix :
                                             I,
                               _func_cache = nothing,
                               analytic = __has_analytic(f1) ? f1.analytic :
                                          nothing,
                               tgrad = __has_tgrad(f1) ? f1.tgrad : nothing,
                               jac = __has_jac(f1) ? f1.jac : nothing,
                               jac_prototype = __has_jac_prototype(f1) ?
                                               f1.jac_prototype : nothing,
                               sparsity = __has_sparsity(f1) ? f1.sparsity :
                                          jac_prototype,
                               jvp = __has_jvp(f1) ? f1.jvp : nothing,
                               vjp = __has_vjp(f1) ? f1.vjp : nothing,
                               Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
                               Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t :
                                         nothing,
                               paramjac = __has_paramjac(f1) ? f1.paramjac :
                                          nothing,
                               syms = __has_syms(f1) ? f1.syms : nothing,
                               observed = __has_observed(f1) ? f1.observed :
                                          DEFAULT_OBSERVED,
                               colorvec = __has_colorvec(f1) ? f1.colorvec :
                                          nothing,
                               sys = __has_sys(f1) ? f1.sys : nothing) where {
                                                                              iip,
                                                                              recompile
                                                                              }
    if recompile === NoSpecialize
        SplitSDEFunction{iip, recompile, Any, Any, Any, Any, Any,
                         Any, Any, Any, Any, Any,
                         Any, Any, Any, Any, Any, Any}(f1, f2, g, mass_matrix, _func_cache,
                                                       analytic,
                                                       tgrad, jac, jvp, vjp, jac_prototype,
                                                       sparsity,
                                                       Wfact, Wfact_t, paramjac, syms,
                                                       observed,
                                                       colorvec, sys)
    else
        SplitSDEFunction{iip, recompile, typeof(f1), typeof(f2), typeof(g),
                         typeof(mass_matrix), typeof(_func_cache),
                         typeof(analytic),
                         typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                         typeof(jac_prototype), typeof(sparsity),
                         typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                         typeof(observed),
                         typeof(colorvec),
                         typeof(sys)}(f1, f2, g, mass_matrix, _func_cache, analytic,
                                      tgrad, jac, jvp, vjp, jac_prototype, sparsity,
                                      Wfact, Wfact_t, paramjac, syms, observed, colorvec,
                                      sys)
    end
end

function SplitSDEFunction(f1, f2, g; kwargs...)
    SplitSDEFunction{isinplace(f2, 4)}(f1, f2, g; kwargs...)
end
function SplitSDEFunction{iip}(f1, f2, g; kwargs...) where {iip}
    SplitSDEFunction{iip, FullSpecialize}(SDEFunction(f1, g), SDEFunction{iip}(f2, g),
                                          g; kwargs...)
end
SplitSDEFunction(f::SplitSDEFunction; kwargs...) = f

@add_kwonly function DynamicalSDEFunction(f1, f2, g, mass_matrix, cache, analytic, tgrad,
                                          jac, jvp, vjp,
                                          jac_prototype, Wfact, Wfact_t, paramjac,
                                          syms, observed, colorvec, sys)
    f1 = typeof(f1) <: AbstractDiffEqOperator ? f1 : SDEFunction(f1)
    f2 = SDEFunction(f2)
    DynamicalSDEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2), typeof(g),
                         typeof(mass_matrix),
                         typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac),
                         typeof(jvp), typeof(vjp),
                         typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                         typeof(observed), typeof(colorvec),
                         typeof(sys)}(f1, f2, g, mass_matrix, cache, analytic, tgrad,
                                      jac, jac_prototype, Wfact, Wfact_t, paramjac, syms,
                                      observed, colorvec, sys)
end

function DynamicalSDEFunction{iip}(f1, f2, g;
                                   mass_matrix = __has_mass_matrix(f1) ?
                                                 f1.mass_matrix : I,
                                   _func_cache = nothing,
                                   analytic = __has_analytic(f1) ? f1.analytic :
                                              nothing,
                                   tgrad = __has_tgrad(f1) ? f1.tgrad : nothing,
                                   jac = __has_jac(f1) ? f1.jac : nothing,
                                   jac_prototype = __has_jac_prototype(f1) ?
                                                   f1.jac_prototype : nothing,
                                   sparsity = __has_sparsity(f1) ? f1.sparsity :
                                              jac_prototype,
                                   jvp = __has_jvp(f1) ? f1.jvp : nothing,
                                   vjp = __has_vjp(f1) ? f1.vjp : nothing,
                                   Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
                                   Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t :
                                             nothing,
                                   paramjac = __has_paramjac(f1) ? f1.paramjac :
                                              nothing,
                                   syms = __has_syms(f1) ? f1.syms : nothing,
                                   observed = __has_observed(f1) ? f1.observed :
                                              DEFAULT_OBSERVED,
                                   colorvec = __has_colorvec(f1) ? f1.colorvec :
                                              nothing,
                                   sys = __has_sys(f1) ? f1.sys : nothing) where {
                                                                                  iip,
                                                                                  recompile
                                                                                  }
    if recompile === NoSpecialize
        DynamicalSDEFunction{iip, recompile, Any, Any, Any, Any, Any,
                             Any, Any, Any, Any, Any,
                             Any, Any, Any, Any, Any, Any}(f1, f2, g, mass_matrix,
                                                           _func_cache,
                                                           analytic, tgrad, jac, jvp, vjp,
                                                           jac_prototype, sparsity,
                                                           Wfact, Wfact_t, paramjac, syms,
                                                           observed, colorvec, sys)
    else
        DynamicalSDEFunction{iip, recompile, typeof(f1), typeof(f2), typeof(g),
                             typeof(mass_matrix), typeof(_func_cache),
                             typeof(analytic),
                             typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                             typeof(jac_prototype), typeof(sparsity),
                             typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                             typeof(observed), typeof(colorvec),
                             typeof(sys)}(f1, f2, g, mass_matrix, _func_cache, analytic,
                                          tgrad, jac, jvp, vjp, jac_prototype, sparsity,
                                          Wfact, Wfact_t, paramjac, syms, observed,
                                          colorvec, sys)
    end
end

function DynamicalSDEFunction(f1, f2, g; kwargs...)
    DynamicalSDEFunction{isinplace(f2, 5)}(f1, f2, g; kwargs...)
end
function DynamicalSDEFunction{iip}(f1, f2, g; kwargs...) where {iip}
    DynamicalSDEFunction{iip, FullSpecialize}(SDEFunction{iip}(f1, g),
                                              SDEFunction{iip}(f2, g), g; kwargs...)
end
DynamicalSDEFunction(f::DynamicalSDEFunction; kwargs...) = f

function RODEFunction{iip, true}(f;
                                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                 analytic = __has_analytic(f) ? f.analytic : nothing,
                                 tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                                 jac = __has_jac(f) ? f.jac : nothing,
                                 jvp = __has_jvp(f) ? f.jvp : nothing,
                                 vjp = __has_vjp(f) ? f.vjp : nothing,
                                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype :
                                                 nothing,
                                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                 Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                                 Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                 syms = __has_syms(f) ? f.syms : nothing,
                                 observed = __has_observed(f) ? f.observed :
                                            DEFAULT_OBSERVED,
                                 colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                 sys = __has_sys(f) ? f.sys : nothing) where {iip}
    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    # Setup when the design is finalized by useful integrators

    #=
    jaciip = jac !== nothing ? isinplace(jac,4,"jac",iip) : iip
    tgradiip = tgrad !== nothing ? isinplace(tgrad,4,"tgrad",iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp,5,"jvp",iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp,5,"vjp",iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact,4,"Wfact",iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t,4,"Wfact_t",iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac,4,"paramjac",iip) : iip

    nonconforming = (jaciip,tgradiip,jvpiip,vjpiip,Wfactiip,Wfact_tiip,paramjaciip) .!= iip
    if any(nonconforming)
       nonconforming = findall(nonconforming)
       functions = ["jac","tgrad","jvp","vjp","Wfact","Wfact_t","paramjac"][nonconforming]
       throw(NonconformingFunctionsError(functions))
    end
    =#

    if recompile === NoSpecialize
        RODEFunction{iip, recompile, Any, Any, Any, Any, Any,
                     Any, Any, Any, Any, Any,
                     typeof(syms), Any, typeof(_colorvec), Any}(f, mass_matrix, analytic,
                                                                tgrad,
                                                                jac, jvp, vjp,
                                                                jac_prototype,
                                                                sparsity, Wfact, Wfact_t,
                                                                paramjac, syms, observed,
                                                                _colorvec, sys)
    else
        RODEFunction{iip, recompile, typeof(f), typeof(mass_matrix),
                     typeof(analytic), typeof(tgrad),
                     typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                     typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
                     typeof(paramjac), typeof(syms), typeof(observed), typeof(_colorvec),
                     typeof(sys)}(f, mass_matrix, analytic, tgrad,
                                  jac, jvp, vjp, jac_prototype, sparsity,
                                  Wfact, Wfact_t, paramjac, syms, observed,
                                  _colorvec, sys)
    end
end

function RODEFunction{iip}(f; kwargs...) where {iip}
    RODEFunction{iip, FullSpecialize}(f; kwargs...)
end
RODEFunction{iip}(f::RODEFunction; kwargs...) where {iip} = f
function RODEFunction(f; kwargs...)
    RODEFunction{isinplace(f, 5), FullSpecialize}(f; kwargs...)
end
RODEFunction(f::RODEFunction; kwargs...) = f

function DAEFunction{iip}(f;
                          analytic = __has_analytic(f) ? f.analytic : nothing,
                          tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                          jac = __has_jac(f) ? f.jac : nothing,
                          jvp = __has_jvp(f) ? f.jvp : nothing,
                          vjp = __has_vjp(f) ? f.vjp : nothing,
                          jac_prototype = __has_jac_prototype(f) ?
                                          f.jac_prototype :
                                          nothing,
                          sparsity = __has_sparsity(f) ? f.sparsity :
                                     jac_prototype,
                          Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                          Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                          paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                          syms = __has_syms(f) ? f.syms : nothing,
                          observed = __has_observed(f) ? f.observed :
                                     DEFAULT_OBSERVED,
                          colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                       recompile}
    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    jaciip = jac !== nothing ? isinplace(jac, 6, "jac", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 7, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 7, "vjp", iip) : iip

    nonconforming = (jaciip, jvpiip, vjpiip) .!= iip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["jac", "jvp", "vjp"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if recompile === NoSpecialize
        DAEFunction{iip, recompile, Any, Any, Any,
                    Any, Any, Any, Any, Any,
                    Any, Any, Any, typeof(syms),
                    Any, typeof(_colorvec), Any}(f, analytic, tgrad, jac, jvp,
                                                 vjp, jac_prototype, sparsity,
                                                 Wfact, Wfact_t,
                                                 paramjac, observed, syms,
                                                 _colorvec, sys)
    else
        DAEFunction{iip, recompile, typeof(f), typeof(analytic), typeof(tgrad),
                    typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                    typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
                    typeof(paramjac), typeof(syms), typeof(observed), typeof(_colorvec),
                    typeof(sys)}(f, analytic, tgrad, jac, jvp, vjp,
                                 jac_prototype, sparsity, Wfact, Wfact_t,
                                 paramjac, syms, observed, _colorvec, sys)
    end
end

function DAEFunction{iip}(f; kwargs...) where {iip}
    DAEFunction{iip, FullSpecialize}(f; kwargs...)
end
DAEFunction{iip}(f::DAEFunction; kwargs...) where {iip} = f
DAEFunction(f; kwargs...) = DAEFunction{isinplace(f, 5), FullSpecialize}(f; kwargs...)
DAEFunction(f::DAEFunction; kwargs...) = f

function DDEFunction{iip}(f;
                          mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                          analytic = __has_analytic(f) ? f.analytic : nothing,
                          tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                          jac = __has_jac(f) ? f.jac : nothing,
                          jvp = __has_jvp(f) ? f.jvp : nothing,
                          vjp = __has_vjp(f) ? f.vjp : nothing,
                          jac_prototype = __has_jac_prototype(f) ?
                                          f.jac_prototype :
                                          nothing,
                          sparsity = __has_sparsity(f) ? f.sparsity :
                                     jac_prototype,
                          Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                          Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                          paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                          syms = __has_syms(f) ? f.syms : nothing,
                          observed = __has_observed(f) ? f.observed :
                                     DEFAULT_OBSERVED,
                          colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                       recompile}
    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    if recompile === NoSpecialize
        DDEFunction{iip, recompile, Any, Any, Any, Any,
                    Any, Any, Any, Any, Any,
                    Any, typeof(syms), Any, typeof(_colorvec), Any}(f, mass_matrix,
                                                                    analytic,
                                                                    tgrad,
                                                                    jac, jvp, vjp,
                                                                    jac_prototype,
                                                                    sparsity, Wfact,
                                                                    Wfact_t,
                                                                    paramjac, syms,
                                                                    observed,
                                                                    _colorvec, sys)
    else
        DDEFunction{iip, recompile, typeof(f), typeof(mass_matrix), typeof(analytic),
                    typeof(tgrad),
                    typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                    typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
                    typeof(paramjac), typeof(syms), typeof(observed),
                    typeof(_colorvec), typeof(sys)}(f, mass_matrix, analytic,
                                                    tgrad, jac, jvp, vjp,
                                                    jac_prototype, sparsity,
                                                    Wfact, Wfact_t, paramjac,
                                                    syms, observed, _colorvec, sys)
    end
end

function DDEFunction{iip}(f; kwargs...) where {iip}
    DDEFunction{iip, FullSpecialize}(f; kwargs...)
end
DDEFunction{iip}(f::DDEFunction; kwargs...) where {iip} = f
DDEFunction(f; kwargs...) = DDEFunction{isinplace(f, 5), FullSpecialize}(f; kwargs...)
DDEFunction(f::DDEFunction; kwargs...) = f

@add_kwonly function DynamicalDDEFunction{iip}(f1, f2, mass_matrix, analytic, tgrad, jac,
                                               jvp, vjp,
                                               jac_prototype, sparsity, Wfact, Wfact_t,
                                               paramjac,
                                               syms, observed, colorvec) where {iip}
    f1 = typeof(f1) <: AbstractDiffEqOperator ? f1 : DDEFunction(f1)
    f2 = DDEFunction(f2)
    DynamicalDDEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
                         typeof(mass_matrix),
                         typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
                         typeof(vjp),
                         typeof(jac_prototype),
                         typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                         typeof(observed), typeof(colorvec),
                         typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
                                      vjp, jac_prototype, sparsity, Wfact, Wfact_t,
                                      paramjac, syms, observed, colorvec, sys)
end
function DynamicalDDEFunction{iip}(f1, f2;
                                   mass_matrix = __has_mass_matrix(f1) ?
                                                 f1.mass_matrix : I,
                                   analytic = __has_analytic(f1) ? f1.analytic :
                                              nothing,
                                   tgrad = __has_tgrad(f1) ? f1.tgrad : nothing,
                                   jac = __has_jac(f1) ? f1.jac : nothing,
                                   jvp = __has_jvp(f1) ? f1.jvp : nothing,
                                   vjp = __has_vjp(f1) ? f1.vjp : nothing,
                                   jac_prototype = __has_jac_prototype(f1) ?
                                                   f1.jac_prototype : nothing,
                                   sparsity = __has_sparsity(f1) ? f1.sparsity :
                                              jac_prototype,
                                   Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
                                   Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t :
                                             nothing,
                                   paramjac = __has_paramjac(f1) ? f1.paramjac :
                                              nothing,
                                   syms = __has_syms(f1) ? f1.syms : nothing,
                                   observed = __has_observed(f1) ? f1.observed :
                                              DEFAULT_OBSERVED,
                                   colorvec = __has_colorvec(f1) ? f1.colorvec :
                                              nothing,
                                   sys = __has_sys(f1) ? f1.sys : nothing) where {
                                                                                  iip,
                                                                                  recompile
                                                                                  }
    if recompile === NoSpecialize
        DynamicalDDEFunction{iip, recompile, Any, Any, Any, Any, Any, Any, Any,
                             Any, Any, Any, Any, Any, Any, Any}(f1, f2, mass_matrix,
                                                                analytic,
                                                                tgrad,
                                                                jac, jvp, vjp,
                                                                jac_prototype,
                                                                sparsity,
                                                                Wfact, Wfact_t, paramjac,
                                                                syms,
                                                                observed, colorvec, sys)
    else
        DynamicalDDEFunction{iip, typeof(f1), typeof(f2), typeof(mass_matrix),
                             typeof(analytic),
                             typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
                             typeof(jac_prototype), typeof(sparsity),
                             typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(syms),
                             typeof(observed),
                             typeof(colorvec),
                             typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
                                          vjp, jac_prototype, sparsity,
                                          Wfact, Wfact_t, paramjac, syms, observed,
                                          colorvec, sys)
    end
end

function DynamicalDDEFunction(f1, f2 = nothing; kwargs...)
    DynamicalDDEFunction{isinplace(f1, 6)}(f1, f2; kwargs...)
end
function DynamicalDDEFunction{iip}(f1, f2; kwargs...) where {iip}
    DynamicalDDEFunction{iip, FullSpecialize}(DDEFunction{iip}(f1),
                                              DDEFunction{iip}(f2); kwargs...)
end
DynamicalDDEFunction(f::DynamicalDDEFunction; kwargs...) = f

function SDDEFunction{iip}(f, g;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix :
                                         I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ?
                                           f.jac_prototype :
                                           nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity :
                                      jac_prototype,
                           Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                           Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           ggprime = nothing,
                           syms = __has_syms(f) ? f.syms : nothing,
                           observed = __has_observed(f) ? f.observed :
                                      DEFAULT_OBSERVED,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                        recompile
                                                                        }
    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    if recompile === NoSpecialize
        SDDEFunction{iip, recompile, Any, Any, Any, Any, Any,
                     Any, Any, Any, Any, Any,
                     Any, Any, typeof(syms), Any, typeof(_colorvec), Any}(f, g, mass_matrix,
                                                                          analytic, tgrad,
                                                                          jac,
                                                                          jvp,
                                                                          vjp,
                                                                          jac_prototype,
                                                                          sparsity, Wfact,
                                                                          Wfact_t,
                                                                          paramjac, ggprime,
                                                                          syms,
                                                                          observed,
                                                                          _colorvec,
                                                                          sys)
    else
        SDDEFunction{iip, recompile, typeof(f), typeof(g),
                     typeof(mass_matrix), typeof(analytic), typeof(tgrad),
                     typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                     typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
                     typeof(paramjac), typeof(ggprime), typeof(syms), typeof(observed),
                     typeof(_colorvec), typeof(sys)}(f, g, mass_matrix,
                                                     analytic, tgrad, jac,
                                                     jvp, vjp, jac_prototype,
                                                     sparsity, Wfact,
                                                     Wfact_t,
                                                     paramjac, ggprime, syms,
                                                     observed, _colorvec, sys)
    end
end

function SDDEFunction{iip}(f, g; kwargs...) where {iip}
    SDDEFunction{iip, FullSpecialize}(f, g; kwargs...)
end
SDDEFunction{iip}(f::SDDEFunction, g; kwargs...) where {iip} = f
function SDDEFunction(f, g; kwargs...)
    SDDEFunction{isinplace(f, 5), FullSpecialize}(f, g; kwargs...)
end
SDDEFunction(f::SDDEFunction; kwargs...) = f

function NonlinearFunction{iip, recompile}(f;
                                           mass_matrix = __has_mass_matrix(f) ?
                                                         f.mass_matrix :
                                                         I,
                                           analytic = __has_analytic(f) ? f.analytic :
                                                      nothing,
                                           tgrad = __has_tgrad(f) ? f.tgrad : nothing,
                                           jac = __has_jac(f) ? f.jac : nothing,
                                           jvp = __has_jvp(f) ? f.jvp : nothing,
                                           vjp = __has_vjp(f) ? f.vjp : nothing,
                                           jac_prototype = __has_jac_prototype(f) ?
                                                           f.jac_prototype : nothing,
                                           sparsity = __has_sparsity(f) ? f.sparsity :
                                                      jac_prototype,
                                           Wfact = __has_Wfact(f) ? f.Wfact : nothing,
                                           Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
                                           paramjac = __has_paramjac(f) ? f.paramjac :
                                                      nothing,
                                           syms = __has_syms(f) ? f.syms : nothing,
                                           observed = __has_observed(f) ? f.observed :
                                                      DEFAULT_OBSERVED_NO_TIME,
                                           colorvec = __has_colorvec(f) ? f.colorvec :
                                                      nothing,
                                           sys = __has_sys(f) ? f.sys : nothing) where {iip,
                                                                                        recompile
                                                                                        }
    if mass_matrix === I && typeof(f) <: Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    jaciip = jac !== nothing ? isinplace(jac, 3, "jac", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 4, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 4, "vjp", iip) : iip

    nonconforming = (jaciip, jvpiip, vjpiip) .!= iip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["jac", "jvp", "vjp"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if recompile === NoSpecialize
        NonlinearFunction{iip, recompile,
                          Any, Any, Any, Any, Any,
                          Any, Any, Any, Any, Any,
                          Any, Any, typeof(syms), Any,
                          typeof(_colorvec), Any}(f, mass_matrix,
                                                  analytic, tgrad, jac,
                                                  jvp, vjp,
                                                  jac_prototype,
                                                  sparsity, Wfact,
                                                  Wfact_t, paramjac,
                                                  syms, observed,
                                                  _colorvec, sys)
    else
        NonlinearFunction{iip, recompile,
                          typeof(f), typeof(mass_matrix), typeof(analytic), typeof(tgrad),
                          typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
                          typeof(sparsity), typeof(Wfact),
                          typeof(Wfact_t), typeof(paramjac), typeof(syms), typeof(observed),
                          typeof(_colorvec), typeof(sys)}(f, mass_matrix, analytic, tgrad,
                                                          jac,
                                                          jvp, vjp, jac_prototype, sparsity,
                                                          Wfact,
                                                          Wfact_t, paramjac, syms, observed,
                                                          _colorvec, sys)
    end
end

function NonlinearFunction{iip}(f; kwargs...) where {iip}
    NonlinearFunction{iip, FullSpecialize}(f; kwargs...)
end
NonlinearFunction{iip}(f::NonlinearFunction; kwargs...) where {iip} = f
function NonlinearFunction(f; kwargs...)
    NonlinearFunction{isinplace(f, 3), FullSpecialize}(f; kwargs...)
end
NonlinearFunction(f::NonlinearFunction; kwargs...) = f

struct NoAD <: AbstractADType end

(f::OptimizationFunction)(args...) = f.f(args...)
OptimizationFunction(args...; kwargs...) = OptimizationFunction{true}(args...; kwargs...)

function OptimizationFunction{iip}(f, adtype::AbstractADType = NoAD();
                                   grad = nothing, hess = nothing, hv = nothing,
                                   cons = nothing, cons_j = nothing, cons_h = nothing,
                                   hess_prototype = nothing,
                                   cons_jac_prototype = __has_jac_prototype(f) ?
                                                        f.jac_prototype : nothing,
                                   cons_hess_prototype = nothing,
                                   syms = __has_syms(f) ? f.syms : nothing,
                                   hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                   cons_jac_colorvec = __has_colorvec(f) ? f.colorvec :
                                                       nothing,
                                   cons_hess_colorvec = __has_colorvec(f) ? f.colorvec :
                                                        nothing,
                                   expr = nothing, cons_expr = nothing,
                                   sys = __has_sys(f) ? f.sys : nothing) where {iip}
    OptimizationFunction{iip, typeof(adtype), typeof(f), typeof(grad), typeof(hess),
                         typeof(hv),
                         typeof(cons), typeof(cons_j), typeof(cons_h),
                         typeof(hess_prototype),
                         typeof(cons_jac_prototype), typeof(cons_hess_prototype),
                         typeof(syms), typeof(hess_colorvec), typeof(cons_jac_colorvec),
                         typeof(cons_hess_colorvec), typeof(expr), typeof(cons_expr),
                         typeof(sys)}(f, adtype, grad, hess,
                                      hv, cons, cons_j, cons_h,
                                      hess_prototype, cons_jac_prototype,
                                      cons_hess_prototype, syms, hess_colorvec,
                                      cons_jac_colorvec, cons_hess_colorvec, expr,
                                      cons_expr, sys)
end

########## Existance Functions

# Check that field/property exists (may be nothing)
__has_jac(f) = isdefined(f, :jac)
__has_jvp(f) = isdefined(f, :jvp)
__has_vjp(f) = isdefined(f, :vjp)
__has_tgrad(f) = isdefined(f, :tgrad)
__has_Wfact(f) = isdefined(f, :Wfact)
__has_Wfact_t(f) = isdefined(f, :Wfact_t)
__has_paramjac(f) = isdefined(f, :paramjac)
__has_jac_prototype(f) = isdefined(f, :jac_prototype)
__has_sparsity(f) = isdefined(f, :sparsity)
__has_mass_matrix(f) = isdefined(f, :mass_matrix)
__has_syms(f) = isdefined(f, :syms)
__has_indepsym(f) = isdefined(f, :indepsym)
__has_observed(f) = isdefined(f, :observed)
__has_analytic(f) = isdefined(f, :analytic)
__has_colorvec(f) = isdefined(f, :colorvec)
__has_sys(f) = isdefined(f, :sys)

# compatibility
has_invW(f::AbstractSciMLFunction) = false
has_analytic(f::AbstractSciMLFunction) = __has_analytic(f) && f.analytic !== nothing
has_jac(f::AbstractSciMLFunction) = __has_jac(f) && f.jac !== nothing
has_jvp(f::AbstractSciMLFunction) = __has_jvp(f) && f.jvp !== nothing
has_vjp(f::AbstractSciMLFunction) = __has_vjp(f) && f.vjp !== nothing
has_tgrad(f::AbstractSciMLFunction) = __has_tgrad(f) && f.tgrad !== nothing
has_Wfact(f::AbstractSciMLFunction) = __has_Wfact(f) && f.Wfact !== nothing
has_Wfact_t(f::AbstractSciMLFunction) = __has_Wfact_t(f) && f.Wfact_t !== nothing
has_paramjac(f::AbstractSciMLFunction) = __has_paramjac(f) && f.paramjac !== nothing
has_syms(f::AbstractSciMLFunction) = __has_syms(f) && f.syms !== nothing
has_indepsym(f::AbstractSciMLFunction) = __has_indepsym(f) && f.indepsym !== nothing
function has_observed(f::AbstractSciMLFunction)
    __has_observed(f) && f.observed !== DEFAULT_OBSERVED && f.observed !== nothing
end
has_colorvec(f::AbstractSciMLFunction) = __has_colorvec(f) && f.colorvec !== nothing

# TODO: find an appropriate way to check `has_*`
has_jac(f::Union{SplitFunction, SplitSDEFunction}) = has_jac(f.f1)
has_jvp(f::Union{SplitFunction, SplitSDEFunction}) = has_jvp(f.f1)
has_vjp(f::Union{SplitFunction, SplitSDEFunction}) = has_vjp(f.f1)
has_tgrad(f::Union{SplitFunction, SplitSDEFunction}) = has_tgrad(f.f1)
has_Wfact(f::Union{SplitFunction, SplitSDEFunction}) = has_Wfact(f.f1)
has_Wfact_t(f::Union{SplitFunction, SplitSDEFunction}) = has_Wfact_t(f.f1)
has_paramjac(f::Union{SplitFunction, SplitSDEFunction}) = has_paramjac(f.f1)
has_colorvec(f::Union{SplitFunction, SplitSDEFunction}) = has_colorvec(f.f1)

has_jac(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_jac(f.f1)
has_jvp(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_jvp(f.f1)
has_vjp(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_vjp(f.f1)
has_tgrad(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_tgrad(f.f1)
has_Wfact(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_Wfact(f.f1)
has_Wfact_t(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_Wfact_t(f.f1)
has_paramjac(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_paramjac(f.f1)
has_colorvec(f::Union{DynamicalODEFunction, DynamicalDDEFunction}) = has_colorvec(f.f1)

has_jac(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_jac(f.f)
has_jvp(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_jvp(f.f)
has_vjp(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_vjp(f.f)
has_tgrad(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_tgrad(f.f)
has_Wfact(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_Wfact(f.f)
has_Wfact_t(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_Wfact_t(f.f)
has_paramjac(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_paramjac(f.f)
has_colorvec(f::Union{UDerivativeWrapper, UJacobianWrapper}) = has_colorvec(f.f)

has_jac(f::JacobianWrapper) = has_jac(f.f)
has_jvp(f::JacobianWrapper) = has_jvp(f.f)
has_vjp(f::JacobianWrapper) = has_vjp(f.f)
has_tgrad(f::JacobianWrapper) = has_tgrad(f.f)
has_Wfact(f::JacobianWrapper) = has_Wfact(f.f)
has_Wfact_t(f::JacobianWrapper) = has_Wfact_t(f.f)
has_paramjac(f::JacobianWrapper) = has_paramjac(f.f)
has_colorvec(f::JacobianWrapper) = has_colorvec(f.f)

######### Additional traits

islinear(f) = false # fallback
islinear(::AbstractDiffEqFunction) = false
islinear(f::ODEFunction) = islinear(f.f)
islinear(f::SplitFunction) = islinear(f.f1)

struct IncrementingODEFunction{iip, recompile, F} <: AbstractODEFunction{iip}
    f::F
end

function IncrementingODEFunction{iip}(f) where {iip}
    IncrementingODEFunction{iip, FullSpecialize, typeof(f)}(f)
end
function IncrementingODEFunction(f)
    IncrementingODEFunction{isinplace(f, 7), FullSpecialize, typeof(f)}(f)
end

(f::IncrementingODEFunction)(args...; kwargs...) = f.f(args...; kwargs...)

for S in [:ODEFunction
          :DiscreteFunction
          :DAEFunction
          :DDEFunction
          :SDEFunction
          :RODEFunction
          :SDDEFunction
          :NonlinearFunction
          :IncrementingODEFunction]
    @eval begin function ConstructionBase.constructorof(::Type{<:$S{iip}}) where {
                                                                                  iip
                                                                                  }
        (args...) -> $S{iip, map(typeof, args)...}(args...)
    end end
end
