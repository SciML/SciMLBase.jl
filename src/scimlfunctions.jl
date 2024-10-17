const RECOMPILE_BY_DEFAULT = true

"""
$(TYPEDEF)

Supertype for the specialization types. Controls the compilation and
function specialization behavior of SciMLFunctions, ultimately controlling
the runtime vs compile-time trade-off.
"""
abstract type AbstractSpecialization end

"""
$(TYPEDEF)

The default specialization level for problem functions. `AutoSpecialize`
works by applying a function wrap just-in-time before the solve process
to disable just-in-time re-specialization of the solver to the specific
choice of model `f` and thus allow for using a cached solver compilation
from a different `f`. This wrapping process can lead to a small decreased
runtime performance with a benefit of a greatly decreased compile-time.

## Note About Benchmarking and Runtime Optimality

It is recommended that `AutoSpecialize` is not used in any benchmarking
due to the potential effect of function wrapping on runtimes. `AutoSpecialize`'s
use case is targeted at decreased latency for REPL performance and
not for cases where where top runtime performance is required (such as in
optimization loops). Generally, for non-stiff equations the cost will be minimal
and potentially not even measurable. For stiff equations, function wrapping
has the limitation that only chunk sized 1 Dual numbers are allowed, which
can decrease Jacobian construction performance.

## Limitations of `AutoSpecialize`

The following limitations are not fundamental to the implementation of `AutoSpecialize`,
but are instead chosen as a compromise between default precompilation times and
ease of maintenance. Please open an issue to discuss lifting any potential
limitations.

* `AutoSpecialize` is only setup to wrap the functions from in-place ODEs. Other
  cases are excluded for the time being due to time limitations.
* `AutoSpecialize` will only lead to compilation reuse if the ODEFunction's other
  functions (such as jac and tgrad) are the default `nothing`. These could be
  JIT wrapped as well in a future version.
* `AutoSpecialize`'d functions are only compatible with Jacobian calculations
  performed with chunk size 1, and only with tag `DiffEqBase.OrdinaryDiffEqTag()`.
  Thus ODE solvers written on the common interface must be careful to detect
  the `AutoSpecialize` case and perform differentiation under these constraints,
  use finite differencing, or manually unwrap before solving. This will lead
  to decreased runtime performance for sufficiently large Jacobians.
* `AutoSpecialize` only wraps on Julia v1.8 and higher.
* `AutoSpecialize` does not handle cases with units. If unitful values are detected,
  wrapping is automatically disabled.
* `AutoSpecialize` only wraps cases for which `promote_rule` is defined between `u0`
  and dual numbers, `u0` and `t`, and for which `ArrayInterface.promote_eltype`
  is defined on `u0` to dual numbers.
* `AutoSpecialize` only wraps cases for which `f.mass_matrix isa UniformScaling`, the
  default.
* `AutoSpecialize` does not wrap cases where `f isa AbstractSciMLOperator`
* By default, only the `u0 isa Vector{Float64}`, `eltype(tspan) isa Float64`, and
  `typeof(p) isa Union{Vector{Float64},SciMLBase.NullParameters}` are specialized
  by the solver libraries. Other forms can be specialized with
  `AutoSpecialize`, but must be done in the precompilation of downstream libraries.
* `AutoSpecialize`d functions are manually unwrapped in adjoint methods in
  SciMLSensitivity.jl in order to allow compiler support for automatic differentiation.
  Improved versions of adjoints which decrease the recompilation surface will come
  in non-breaking updates.

Cases where automatic wrapping is disabled are equivalent to `FullSpecialize`.

## Example

```julia
f(du,u,p,t) = (du .= u)

# Note this is the same as ODEProblem(f, [1.0], (0.0,1.0))
# If no preferences are set
ODEProblem{true, SciMLBase.AutoSpecialize}(f, [1.0], (0.0,1.0))
```
"""
struct AutoSpecialize <: AbstractSpecialization end

"""
$(TYPEDEF)

`NoSpecialize` forces SciMLFunctions to not specialize on the types
of functions wrapped within it. This ultimately contributes to a
form such that every `prob.f` type is the same, meaning compilation
caches are fully reused, with the downside of losing runtime performance.
`NoSpecialize` is the form that most fully trades off runtime for compile
time. Unlike `AutoSpecialize`, `NoSpecialize` can be used with any
`SciMLFunction`.

## Example

```julia
f(du,u,p,t) = (du .= u)
ODEProblem{true, SciMLBase.NoSpecialize}(f, [1.0], (0.0,1.0))
```
"""
struct NoSpecialize <: AbstractSpecialization end

"""
$(TYPEDEF)

`FunctionWrapperSpecialize` is an eager wrapping choice which
performs a function wrapping during the `ODEProblem` construction.
This performs the function wrapping at the earliest possible point,
giving the best compile-time vs runtime performance, but with the
difficulty that any usage of `prob.f` needs to account for the
function wrapper's presence. While optimal in a performance sense,
this method has many usability issues with nonstandard solvers
and analyses as it requires unwrapping before re-wrapping for any
type changes. Thus this method is not used by default. Given that
the compile-time different is almost undetectable from AutoSpecialize,
this method is mostly used as a benchmarking reference for speed
of light for `AutoSpecialize`.

## Limitations of `FunctionWrapperSpecialize`

`FunctionWrapperSpecialize` has all of the limitations of `AutoSpecialize`,
but also includes the limitations:

* `prob.f` is directly specialized to the types of `(u,p,t)`, and any usage
  of `prob.f` on other types first requires using
  `SciMLBase.unwrapped_f(prob.f)` to remove the function wrapper.
* `FunctionWrapperSpecialize` can only be used by the `ODEProblem` constructor.
  If an `ODEFunction` is being constructed, the user must manually use
  `DiffEqBase.wrap_iip` on `f` before calling
  `ODEFunction{true,FunctionWrapperSpecialize}(f)`. This is a fundamental
  limitation of the approach as the types of `(u,p,t)` are required in the
  construction process and not accessible in the `AbstractSciMLFunction` constructors.

## Example

```julia
f(du,u,p,t) = (du .= u)
ODEProblem{true, SciMLBase.FunctionWrapperSpecialize}(f, [1.0], (0.0,1.0))
```
"""
struct FunctionWrapperSpecialize <: AbstractSpecialization end

"""
$(TYPEDEF)

`FullSpecialize` is an eager specialization choice which
directly types the `AbstractSciMLFunction` struct to match the type
of the model `f`. This forces recompilation of the solver on each
new function type `f`, leading to the most compile times with the
benefit of having the best runtime performance.

`FullSpecialize` should be used in all cases where top runtime performance
is required, such as in long-running simulations and benchmarking.

## Example

```julia
f(du,u,p,t) = (du .= u)
ODEProblem{true, SciMLBase.FullSpecialize}(f, [1.0], (0.0,1.0))
```
"""
struct FullSpecialize <: AbstractSpecialization end

specstring = Preferences.@load_preference("SpecializationLevel", "AutoSpecialize")
if specstring ∉
   ("NoSpecialize", "FullSpecialize", "AutoSpecialize", "FunctionWrapperSpecialize")
    error("SpecializationLevel preference $specstring is not in the allowed set of choices (NoSpecialize, FullSpecialize, AutoSpecialize, FunctionWrapperSpecialize).")
end

const DEFAULT_SPECIALIZATION = getproperty(SciMLBase, Symbol(specstring))

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
                                              as in-place, then all constituent functions like `jac` and `paramjac`
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

const INTEGRAND_MISMATCH_FUNCTIONS_ERROR_MESSAGE = """
                                              Nonconforming functions detected. If an integrand function `f` is defined
                                              as out-of-place (`f(u,p)`), then no integrand_prototype can be passed into the
                                              function constructor. Likewise if `f` is defined as in-place (`f(out,u,p)`), then
                                              an integrand_prototype is required. Either change the use of the function
                                              constructor or define the appropriate dispatch for `f`.
                                              """

struct IntegrandMismatchFunctionError <: Exception
    iip::Bool
    integrand_passed::Bool
end

function Base.showerror(io::IO, e::IntegrandMismatchFunctionError)
    println(io, INTEGRAND_MISMATCH_FUNCTIONS_ERROR_MESSAGE)
    print(io, "Mismatch: IIP=")
    printstyled(io, e.iip; bold = true, color = :red)
    print(io, ", Integrand passed=")
    printstyled(io, e.integrand_passed; bold = true, color = :red)
end

"""
$(TYPEDEF)
"""
abstract type AbstractODEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of an ODE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
ODEFunction{iip,specialize}(f;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
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
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

`iip` is the optional boolean for determining whether a given function is written to
be used in-place or out-of-place. In-place functions are `f!(du,u,p,t)` where the return
is ignored, and the result is expected to be mutated into the value of `du`. Out-of-place
functions are `du=f(u,p,t)`.

Normally, this is determined automatically by looking at the method table for `f` and seeing
the maximum number of arguments in available dispatches. For this reason, the constructor
`ODEFunction(f)` generally works (but is type-unstable). However, for type-stability or
to enforce correctness, this option is passed via `ODEFunction{true}(f)`.

## specialize: Controlling Compilation and Specialization

The `specialize` parameter controls the specialization level of the ODEFunction
on the function `f`. This allows for a trade-off between compile and run time performance.
The available specialization levels are:

* `SciMLBase.AutoSpecialize`: this form performs a lazy function wrapping on the
  functions of the ODE in order to stop recompilation of the ODE solver, but allow
  for the `prob.f` to stay unwrapped for normal usage. This is the default specialization
  level and strikes a balance in compile time vs runtime performance.
* `SciMLBase.FullSpecialize`: this form fully specializes the `ODEFunction` on the
  constituent functions that make its fields. As such, each `ODEFunction` in this
  form is uniquely typed, requiring re-specialization and compilation for each new
  ODE definition. This form has the highest compile-time at the cost of being the
  most optimal in runtime. This form should be preferred for long-running calculations
  (such as within optimization loops) and for benchmarking.
* `SciMLBase.NoSpecialize`: this form fully unspecializes the function types in the ODEFunction
  definition by using an `Any` type declaration. As a result, it can result in reduced runtime
  performance, but is the form that induces the least compile-time.
* `SciMLBase.FunctionWrapperSpecialize`: this is an eager function wrapping form. It is
  unsafe with many solvers, and thus is mostly used for development testing.

For more details, see the
[specialization levels section of the SciMLBase documentation](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/#Specialization-Levels).

## Fields

The fields of the ODEFunction type directly match the names of the inputs.

## More Details on Jacobians

The following example creates an inplace `ODEFunction` whose Jacobian is a `Diagonal`:

```julia
using LinearAlgebra
f = (du,u,p,t) -> du .= t .* u
jac = (J,u,p,t) -> (J[1,1] = t; J[2,2] = t; J)
jp = Diagonal(zeros(2))
fun = ODEFunction(f; jac=jac, jac_prototype=jp)
```

Note that the integrators will always make a deep copy of `fun.jac_prototype`, so
there's no worry of aliasing.

In general, the Jacobian prototype can be anything that has `mul!` defined, in
particular sparse matrices or custom lazy types that support `mul!`. A special case
is when the `jac_prototype` is a `AbstractSciMLOperator`, in which case you
do not need to supply `jac` as it is automatically set to `update_coefficients!`.
Refer to the AbstractSciMLOperators documentation for more information
on setting up time/parameter dependent operators.

## Examples

### Declaring Explicit Jacobians for ODEs

The most standard case, declaring a function for a Jacobian is done by overloading
the function `f(du,u,p,t)` with an in-place updating function for the Jacobian:
`f_jac(J,u,p,t)` where the value type is used for dispatch. For example,
take the Lotka-Volterra model:

```julia
function f(du,u,p,t)
  du[1] = 2.0 * u[1] - 1.2 * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end
```

To declare the Jacobian, we simply add the dispatch:

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
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions.
"""
struct ODEFunction{iip, specialize, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, WP, TPJ,
    O, TCV,
    SYS, IProb, UIProb, IProbMap, IProbPmap} <: AbstractODEFunction{iip}
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
    W_prototype::WP
    paramjac::TPJ
    observed::O
    colorvec::TCV
    sys::SYS
    initializeprob::IProb
    update_initializeprob!::UIProb
    initializeprobmap::IProbMap
    initializeprobpmap::IProbPmap
end

@doc doc"""
$(TYPEDEF)

A representation of a split ODE function `f`, defined by:

```math
M \frac{du}{dt} = f_1(u,p,t) + f_2(u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

Generally, for ODE integrators the `f_1` portion should be considered the
"stiff portion of the model" with larger timescale separation, while the
`f_2` portion should be considered the "non-stiff portion". This interpretation
is directly used in integrators like IMEX (implicit-explicit integrators)
and exponential integrators.

## Constructor

```julia
SplitFunction{iip,specialize}(f1,f2;
                             mass_matrix = __has_mass_matrix(f1) ? f1.mass_matrix : I,
                             analytic = __has_analytic(f1) ? f1.analytic : nothing,
                             tgrad= __has_tgrad(f1) ? f1.tgrad : nothing,
                             jac = __has_jac(f1) ? f1.jac : nothing,
                             jvp = __has_jvp(f1) ? f1.jvp : nothing,
                             vjp = __has_vjp(f1) ? f1.vjp : nothing,
                             jac_prototype = __has_jac_prototype(f1) ? f1.jac_prototype : nothing,
                             sparsity = __has_sparsity(f1) ? f1.sparsity : jac_prototype,
                             paramjac = __has_paramjac(f1) ? f1.paramjac : nothing,
                             colorvec = __has_colorvec(f1) ? f1.colorvec : nothing,
                             sys = __has_sys(f1) ? f1.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,p,t)` or `du = f_i(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of the `SplitFunction`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the SplitFunction type directly match the names of the inputs.

## Symbolically Generating the Functions

See the `modelingtoolkitize` function from
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions. See `ModelingToolkit.SplitODEProblem` for
information on generating the SplitFunction from this symbolic engine.
"""
struct SplitFunction{
    iip, specialize, F1, F2, TMM, C, Ta, Tt, TJ, JVP, VJP, JP, WP, SP, TW, TWt,
    TPJ, O,
    TCV, SYS, IProb, UIProb, IProbMap, IProbPmap} <: AbstractODEFunction{iip}
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
    W_prototype::WP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    observed::O
    colorvec::TCV
    sys::SYS
    initializeprob::IProb
    update_initializeprob!::UIProb
    initializeprobmap::IProbMap
    initializeprobpmap::IProbPmap
end

@doc doc"""
$(TYPEDEF)

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
DynamicalODEFunction{iip,specialize}(f1,f2;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
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
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalODEFunction type directly match the names of the inputs.
"""
struct DynamicalODEFunction{iip, specialize, F1, F2, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW,
    TWt, TPJ,
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
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractDDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of a DDE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,h,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DDEFunction{iip,specialize}(f;
                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                 analytic = __has_analytic(f) ? f.analytic : nothing,
                 tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                 jac = __has_jac(f) ? f.jac : nothing,
                 jvp = __has_jvp(f) ? f.jvp : nothing,
                 vjp = __has_vjp(f) ? f.vjp : nothing,
                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,h,p,t)` or `du = f(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The history function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DDEFunction type directly match the names of the inputs.
"""
struct DDEFunction{
    iip, specialize, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, O, TCV, SYS
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
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
$(TYPEDEF)

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
DynamicalDDEFunction{iip,specialize}(f1,f2;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                    sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,u,h,p,t)` or `du = f_i(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The history function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M_i` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalDDEFunction type directly match the names of the inputs.
"""
struct DynamicalDDEFunction{iip, specialize, F1, F2, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW,
    TWt, TPJ,
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
$(TYPEDEF)

A representation of a discrete dynamical system `f`, defined by:

```math
u_{n+1} = f(u,p,t_{n+1})
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DiscreteFunction{iip,specialize}(f;
                                analytic = __has_analytic(f) ? f.analytic : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DiscreteFunction type directly match the names of the inputs.
"""
struct DiscreteFunction{iip, specialize, F, Ta, O, SYS} <:
       AbstractDiscreteFunction{iip}
    f::F
    analytic::Ta
    observed::O
    sys::SYS
end

@doc doc"""
$(TYPEDEF)

A representation of an discrete dynamical system `f`, defined by:

```math
0 = f(u_{n+1}, u_{n}, p, t_{n+1}, integ)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.
`integ` contains the fields:
```julia
dt: the time step
```

## Constructor

```julia
ImplicitDiscreteFunction{iip,specialize}(f;
                                analytic = __has_analytic(f) ? f.analytic : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(residual, u_next, u, p, t)` or `residual = f(u_next, u, p, t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the ImplicitDiscreteFunction type directly match the names of the inputs.
"""
struct ImplicitDiscreteFunction{iip, specialize, F, Ta, O, SYS} <:
       AbstractDiscreteFunction{iip}
    f::F
    analytic::Ta
    observed::O
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractSDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of an SDE function `f`, defined by:

```math
M du = f(u,p,t)dt + g(u,p,t) dW
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
SDEFunction{iip,specialize}(f,g;
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
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that both the function `f` and `g` are required. This function should
be given as `f!(du,u,p,t)` or `du = f(u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the ODEFunction type directly match the names of the inputs.
"""
struct SDEFunction{iip, specialize, F, G, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ,
    GG, O,
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
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
$(TYPEDEF)

A representation of a split SDE function `f`, defined by:

```math
M \frac{du}{dt} = f_1(u,p,t) + f_2(u,p,t) + g(u,p,t) dW
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

Generally, for SDE integrators the `f_1` portion should be considered the
"stiff portion of the model" with larger timescale separation, while the
`f_2` portion should be considered the "non-stiff portion". This interpretation
is directly used in integrators like IMEX (implicit-explicit integrators)
and exponential integrators.

## Constructor

```julia
SplitSDEFunction{iip,specialize}(f1,f2,g;
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
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. All of the remaining functions
are optional for improving or accelerating the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the SDE function. Can be used
  to determine that the equation is actually a stochastic differential-algebraic equation (SDAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/sdae_solve/.
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

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the SplitSDEFunction type directly match the names of the inputs.
"""
struct SplitSDEFunction{iip, specialize, F1, F2, G, TMM, C, Ta, Tt, TJ, JVP, VJP, JP, SP,
    TW,
    TWt, TPJ,
    O, TCV, SYS} <: AbstractSDEFunction{iip}
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
    observed::O
    colorvec::TCV
    sys::SYS
end

@doc doc"""
$(TYPEDEF)

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
DynamicalSDEFunction{iip,specialize}(f1,f2;
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
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalSDEFunction type directly match the names of the inputs.
"""
struct DynamicalSDEFunction{iip, specialize, F1, F2, G, TMM, C, Ta, Tt, TJ, JVP, VJP, JP,
    SP,
    TW, TWt,
    TPJ, O, TCV, SYS} <: AbstractSDEFunction{iip}
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
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractRODEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of a RODE function `f`, defined by:

```math
M \frac{du}{dt} = f(u,p,t,W)dt
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
RODEFunction{iip,specialize}(f;
                           mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                           colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                           sys = __has_sys(f) ? f.sys : nothing,
                           analytic_full = __has_analytic_full(f) ? f.analytic_full : false)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,p,t,W)` or `du = f(u,p,t,W)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the RODE function. Can be used
  to determine that the equation is actually a random differential-algebraic equation (RDAE)
  if `M` is singular.
- `analytic`: (u0,p,t,W)` or `analytic(sol)`: used to pass an analytical solution function for the analytical
  solution of the RODE. Generally only used for testing and development of the solvers.
  The exact form depends on the field `analytic_full`.
- `analytic_full`: a boolean to indicate whether to use the form `analytic(u0,p,t,W)` (if `false`)
  or the form `analytic!(sol)` (if `true`). The former is expected to return the solution `u(t)` of
  the equation, given the initial condition `u0`, the parameter `p`, the current time `t` and the
  value `W=W(t)` of the noise at the given time `t`. The latter case is useful when the solution
  of the RODE depends on the whole history of the noise, which is available in `sol.W.W`, at
  times `sol.W.t`. In this case, `analytic(sol)` must mutate explicitly the field `sol.u_analytic`
  with the corresponding expected solution at `sol.W.t` or `sol.t`.
- `tgrad(dT,u,p,t,W)` or dT=tgrad(u,p,t,W): returns ``\frac{\partial f(u,p,t,W)}{\partial t}``
- `jac(J,u,p,t,W)` or `J=jac(u,p,t,W)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t,W)` or `Jv=jvp(v,u,p,t,W)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t,W)` or `Jv=vjp(v,u,p,t,W)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t,W)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the RODEFunction type directly match the names of the inputs.
"""
struct RODEFunction{
    iip, specialize, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, O, TCV, SYS
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
    observed::O
    colorvec::TCV
    sys::SYS
    analytic_full::Bool
end

"""
$(TYPEDEF)
"""
abstract type AbstractDAEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of an implicit DAE function `f`, defined by:

```math
0 = f(\frac{du}{dt},u,p,t)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DAEFunction{iip,specialize}(f;
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

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
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) for
automatically symbolically generating the Jacobian and more from the
numerically-defined functions.
"""
struct DAEFunction{iip, specialize, F, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ, O, TCV,
    SYS, IProb, UIProb, IProbMap, IProbPmap} <:
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
    observed::O
    colorvec::TCV
    sys::SYS
    initializeprob::IProb
    update_initializeprob!::UIProb
    initializeprobmap::IProbMap
    initializeprobpmap::IProbPmap
end

"""
$(TYPEDEF)
"""
abstract type AbstractSDDEFunction{iip} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of a SDDE function `f`, defined by:

```math
M du = f(u,h,p,t) dt + g(u,h,p,t) dW_t
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
SDDEFunction{iip,specialize}(f,g;
                 mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                 analytic = __has_analytic(f) ? f.analytic : nothing,
                 tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                 jac = __has_jac(f) ? f.jac : nothing,
                 jvp = __has_jvp(f) ? f.jvp : nothing,
                 vjp = __has_vjp(f) ? f.vjp : nothing,
                 jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                 sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                 paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                 colorvec = __has_colorvec(f) ? f.colorvec : nothing
                 sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(du,u,h,p,t)` or `du = f(u,h,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling. The history function
`h` acts as an interpolator over time, i.e. `h(t)` with options matching
the solution interface, i.e. `h(t; save_idxs = 2)`.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DDEFunction type directly match the names of the inputs.
"""
struct SDDEFunction{iip, specialize, F, G, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt, TPJ,
    GG, O,
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
    observed::O
    colorvec::TCV
    sys::SYS
end

"""
$(TYPEDEF)
"""
abstract type AbstractNonlinearFunction{iip} <: AbstractSciMLFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of a nonlinear system of equations `f`, defined by:

```math
0 = f(u,p)
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
NonlinearFunction{iip, specialize}(f;
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           jac = __has_jac(f) ? f.jac : nothing,
                           jvp = __has_jvp(f) ? f.jvp : nothing,
                           vjp = __has_vjp(f) ? f.vjp : nothing,
                           jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                           sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                           paramjac = __has_paramjac(f) ? f.paramjac : nothing,
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
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the NonlinearFunction type directly match the names of the inputs.
"""
struct NonlinearFunction{iip, specialize, F, TMM, Ta, Tt, TJ, JVP, VJP, JP, SP, TW, TWt,
    TPJ, O, TCV, SYS, RP} <: AbstractNonlinearFunction{iip}
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
    observed::O
    colorvec::TCV
    sys::SYS
    resid_prototype::RP
end

"""
$(TYPEDEF)
"""
abstract type AbstractIntervalNonlinearFunction{iip} <: AbstractSciMLFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of an interval nonlinear system of equations `f`, defined by:

```math
f(t,p) = u = 0
```

and all of its related functions. For all cases, `p` are the parameters and `t` is the
interval variable.

## Constructor

```julia
IntervalNonlinearFunction{iip, specialize}(f;
                           analytic = __has_analytic(f) ? f.analytic : nothing,
                           sys = __has_sys(f) ? f.sys : nothing)
```

Note that only the function `f` itself is required. This function should
be given as `f!(u,t,p)` or `u = f(t,p)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `analytic(p)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the IntervalNonlinearFunction type directly match the names of the inputs.
"""
struct IntervalNonlinearFunction{iip, specialize, F, Ta,
    O, SYS
} <: AbstractIntervalNonlinearFunction{iip}
    f::F
    analytic::Ta
    observed::O
    sys::SYS
end

"""
$(TYPEDEF)

A representation of an objective function `f`, defined by:

```math
\\min_{u} f(u,p)
```

and all of its related functions, such as the gradient of `f`, its Hessian,
and more. For all cases, `u` is the state which in this case are the optimization variables and `p` are the fixed parameters or data.

## Constructor

```julia
OptimizationFunction{iip}(f, adtype::AbstractADType = NoAD();
                          grad = nothing, hess = nothing, hv = nothing,
                          cons = nothing, cons_j = nothing, cons_jvp = nothing,
                          cons_vjp = nothing, cons_h = nothing,
                          hess_prototype = nothing,
                          cons_jac_prototype = nothing,
                          cons_hess_prototype = nothing,
                          observed = __has_observed(f) ? f.observed : DEFAULT_OBSERVED_NO_TIME,
                          lag_h = nothing,
                          hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          cons_jac_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          cons_hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                          lag_hess_colorvec = nothing,
                          sys = __has_sys(f) ? f.sys : nothing)
```

## Positional Arguments

- `f(u,p)`: the function to optimize. `u` are the optimization variables and `p` are fixed parameters or data used in the objective,
even if no such parameters are used in the objective it should be an argument in the function. For minibatching `p` can be used to pass in
a minibatch, take a look at the tutorial [here](https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/) to see how to do it.
This should return a scalar, the loss value, as the return output.
- `adtype`: see the Defining Optimization Functions via AD section below.

## Keyword Arguments

- `grad(G,u,p)` or `G=grad(u,p)`: the gradient of `f` with respect to `u`.
- `hess(H,u,p)` or `H=hess(u,p)`: the Hessian of `f` with respect to `u`.
- `hv(Hv,u,v,p)` or `Hv=hv(u,v,p)`: the Hessian-vector product ``(d^2 f / du^2) v``.
- `cons(res,u,p)` or `res=cons(u,p)` : the constraints function, should mutate the passed `res` array
    with value of the `i`th constraint, evaluated at the current values of variables
    inside the optimization routine. This takes just the function evaluations
    and the equality or inequality assertion is applied by the solver based on the constraint
    bounds passed as `lcons` and `ucons` to [`OptimizationProblem`](@ref), in case of equality
    constraints `lcons` and `ucons` should be passed equal values.
- `cons_j(J,u,p)` or `J=cons_j(u,p)`: the Jacobian of the constraints.
- `cons_jvp(Jv,u,v,p)` or `Jv=cons_jvp(u,v,p)`: the Jacobian-vector product of the constraints.
- `cons_vjp(Jv,u,v,p)` or `Jv=cons_vjp(u,v,p)`: the Jacobian-vector product of the constraints.
- `cons_h(H,u,p)` or `H=cons_h(u,p)`: the Hessian of the constraints, provided as
   an array of Hessians with `res[i]` being the Hessian with respect to the `i`th output on `cons`.
- `hess_prototype`: a prototype matrix matching the type that matches the Hessian. For example,
  if the Hessian is tridiagonal, then an appropriately sized `Hessian` matrix can be used
  as the prototype and optimization solvers will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Hessian.
  The default is `nothing`, which means a dense Hessian.
- `cons_jac_prototype`: a prototype matrix matching the type that matches the constraint Jacobian.
  The default is `nothing`, which means a dense constraint Jacobian.
- `cons_hess_prototype`: a prototype matrix matching the type that matches the constraint Hessian.
  This is defined as an array of matrices, where `hess[i]` is the Hessian w.r.t. the `i`th output.
  For example, if the Hessian is sparse, then `hess` is a `Vector{SparseMatrixCSC}`.
  The default is `nothing`, which means a dense constraint Hessian.
- `lag_h(res,u,sigma,mu,p)` or `res=lag_h(u,sigma,mu,p)`: the Hessian of the Lagrangian,
  where `sigma` is a multiplier of the cost function and `mu` are the Lagrange multipliers
  multiplying the constraints. This can be provided instead of `hess` and `cons_h`
  to solvers that directly use the Hessian of the Lagrangian.
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

When [Symbolic Problem Building with ModelingToolkit](https://docs.sciml.ai/Optimization/stable/tutorials/symbolic/) interface is used the following arguments are also relevant:

- `observed`: an algebraic combination of optimization variables that is of interest to the user
    which will be available in the solution. This can be single or multiple expressions.
- `sys`: field that stores the `OptimizationSystem`.

## Defining Optimization Functions via AD

While using the keyword arguments gives the user control over defining
all of the possible functions, the simplest way to handle the generation
of an `OptimizationFunction` is by specifying an option from ADTypes.jl
which lets the user choose the Automatic Differentiation backend to use
for automatically filling in all of the extra functions. For example,

```julia
OptimizationFunction(f,AutoForwardDiff())
```

will use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) to define
all of the necessary functions. Note that if any functions are defined
directly, the auto-AD definition does not overwrite the user's choice.

Each of the AD-based constructors are documented separately via their
own dispatches below in the [Automatic Differentiation Construction Choice Recommendations](@ref ad) section.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the OptimizationFunction type directly match the names of the inputs.
"""
struct OptimizationFunction{
    iip, AD, F, G, FG, H, FGH, HV, C, CJ, CJV, CVJ, CH, HP, CJP, CHP, O,
    EX, CEX, SYS, LH, LHP, HCV, CJCV, CHCV, LHCV} <:
       AbstractOptimizationFunction{iip}
    f::F
    adtype::AD
    grad::G
    fg::FG
    hess::H
    fgh::FGH
    hv::HV
    cons::C
    cons_j::CJ
    cons_jvp::CJV
    cons_vjp::CVJ
    cons_h::CH
    hess_prototype::HP
    cons_jac_prototype::CJP
    cons_hess_prototype::CHP
    observed::O
    expr::EX
    cons_expr::CEX
    sys::SYS
    lag_h::LH
    lag_hess_prototype::LHP
    hess_colorvec::HCV
    cons_jac_colorvec::CJCV
    cons_hess_colorvec::CHCV
    lag_hess_colorvec::LHCV
end

"""
$(TYPEDEF)
"""

struct MultiObjectiveOptimizationFunction{
    iip, AD, F, J, H, HV, C, CJ, CJV, CVJ, CH, HP, CJP, CHP, O,
    EX, CEX, SYS, LH, LHP, HCV, CJCV, CHCV, LHCV} <:
       AbstractOptimizationFunction{iip}
    f::F
    adtype::AD
    jac::J
    hess::H
    hv::HV
    cons::C
    cons_j::CJ
    cons_jvp::CJV
    cons_vjp::CVJ
    cons_h::CH
    hess_prototype::HP
    cons_jac_prototype::CJP
    cons_hess_prototype::CHP
    observed::O
    expr::EX
    cons_expr::CEX
    sys::SYS
    lag_h::LH
    lag_hess_prototype::LHP
    hess_colorvec::HCV
    cons_jac_colorvec::CJCV
    cons_hess_colorvec::CHCV
    lag_hess_colorvec::LHCV
end

"""
$(TYPEDEF)
"""
abstract type AbstractBVPFunction{iip, twopoint} <: AbstractDiffEqFunction{iip} end

@doc doc"""
$(TYPEDEF)

A representation of a BVP function `f`, defined by:

```math
\frac{du}{dt} = f(u, p, t)
```

and the constraints:

```math
g(u, p, t) = 0
```

If the size of `g(u, p, t)` is different from the size of `u`, then the constraints are
interpreted as a least squares problem, i.e. the objective function is:

```math
\min_{u} \| g_i(u, p, t) \|^2
```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

```julia
BVPFunction{iip, specialize}(f, bc;
    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
    analytic = __has_analytic(f) ? f.analytic : nothing,
    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
    jac = __has_jac(f) ? f.jac : nothing,
    bcjac = __has_jac(bc) ? bc.jac : nothing,
    jvp = __has_jvp(f) ? f.jvp : nothing,
    vjp = __has_vjp(f) ? f.vjp : nothing,
    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
    bcjac_prototype = __has_jac_prototype(bc) ? bc.jac_prototype : nothing,
    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
    syms = nothing,
    indepsym= nothing,
    paramsyms = nothing,
    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
    bccolorvec = __has_colorvec(f) ? bc.colorvec : nothing,
    sys = __has_sys(f) ? f.sys : nothing,
    twopoint::Union{Val, Bool} = Val(false))
```

Note that both the function `f` and boundary condition `bc` are required. `f` should
be given as `f(du,u,p,t)` or `out = f(u,p,t)`. `bc` should be given as `bc(res, u, p, t)`.
See the section on `iip` for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f` and `bc`. These include:

- `mass_matrix`: the mass matrix `M` represented in the BVP function. Can be used
  to determine that the equation is actually a BVP for differential algebraic equation (DAE)
  if `M` is singular.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the BVP. Generally only used for testing and development of the solvers.
- `tgrad(dT,u,h,p,t)` or dT=tgrad(u,p,t): returns ``\frac{\partial f(u,p,t)}{\partial t}``
- `jac(J,du,u,p,gamma,t)` or `J=jac(du,u,p,gamma,t)`: returns ``\frac{df}{du}``
- `bcjac(J,du,u,p,gamma,t)` or `J=jac(du,u,p,gamma,t)`: erturns ``\frac{dbc}{du}``
- `jvp(Jv,v,du,u,p,gamma,t)` or `Jv=jvp(v,du,u,p,gamma,t)`: returns the directional
  derivative``\frac{df}{du} v``
- `vjp(Jv,v,du,u,p,gamma,t)` or `Jv=vjp(v,du,u,p,gamma,t)`: returns the adjoint
  derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `bcjac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.
- `bccolorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `bcjac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

Additional Options:

- `twopoint`: Specify that the BVP is a two-point boundary value problem. Use `Val(true)` or
  `Val(false)` for type stability.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the BVPFunction type directly match the names of the inputs.
"""
struct BVPFunction{iip, specialize, twopoint, F, BF, TMM, Ta, Tt, TJ, BCTJ, JVP, VJP,
    JP, BCJP, BCRP, SP, TW, TWt, TPJ, O, TCV, BCTCV,
    SYS} <: AbstractBVPFunction{iip, twopoint}
    f::F
    bc::BF
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    bcjac::BCTJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    bcjac_prototype::BCJP
    bcresid_prototype::BCRP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    observed::O
    colorvec::TCV
    bccolorvec::BCTCV
    sys::SYS
end

@doc doc"""
$(TYPEDEF)

A representation of a dynamical BVP function `f`, defined by:

```math
M \frac{ddu}{dt} = f(du,u,p,t)
```

along with its boundary condition:

```math

```

and all of its related functions, such as the Jacobian of `f`, its gradient
with respect to time, and more. For all cases, `u0` is the initial condition,
`p` are the parameters, and `t` is the independent variable.

## Constructor

```julia
DynamicalBVPFunction{iip,specialize}(f, bc;
                                    mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
                                    analytic = __has_analytic(f) ? f.analytic : nothing,
                                    tgrad= __has_tgrad(f) ? f.tgrad : nothing,
                                    jac = __has_jac(f) ? f.jac : nothing,
                                    jvp = __has_jvp(f) ? f.jvp : nothing,
                                    vjp = __has_vjp(f) ? f.vjp : nothing,
                                    jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
                                    sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
                                    paramjac = __has_paramjac(f) ? f.paramjac : nothing,
                                    colorvec = __has_colorvec(f) ? f.colorvec : nothing,
                                    sys = __has_sys(f) ? f.sys : nothing
                                    twopoint::Union{Val, Bool} = Val(false))
```

Note that only the functions `f_i` themselves are required. These functions should
be given as `f_i!(du,du,u,p,t)` or `ddu = f_i(du,u,p,t)`. See the section on `iip`
for more details on in-place vs out-of-place handling.

All of the remaining functions are optional for improving or accelerating
the usage of `f`. These include:

- `mass_matrix`: the mass matrix `M_i` represented in the ODE function. Can be used
  to determine that the equation is actually a differential-algebraic equation (DAE)
  if `M` is singular. Note that in this case special solvers are required, see the
  DAE solver page for more details: https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/.
  Must be an AbstractArray or an AbstractSciMLOperator. Should be given as a tuple
  of mass matrices, i.e. `(M_1, M_2)` for the mass matrices of equations 1 and 2
  respectively.
- `analytic(u0,p,t)`: used to pass an analytical solution function for the analytical
  solution of the ODE. Generally only used for testing and development of the solvers.
- `tgrad(dT,du,u,p,t)` or dT=tgrad(du,u,p,t): returns ``\frac{\partial f(du,u,p,t)}{\partial t}``
- `jac(J,du,u,p,t)` or `J=jac(du,u,p,t)`: returns ``\frac{df}{du}``
- `jvp(Jv,v,u,p,t)` or `Jv=jvp(v,u,p,t)`: returns the directional derivative``\frac{df}{du} v``
- `vjp(Jv,v,u,p,t)` or `Jv=vjp(v,u,p,t)`: returns the adjoint derivative``\frac{df}{du}^\ast v``
- `jac_prototype`: a prototype matrix matching the type that matches the Jacobian. For example,
  if the Jacobian is tridiagonal, then an appropriately sized `Tridiagonal` matrix can be used
  as the prototype and integrators will specialize on this structure where possible. Non-structured
  sparsity patterns should use a `SparseMatrixCSC` with a correct sparsity pattern for the Jacobian.
  The default is `nothing`, which means a dense Jacobian.
- `paramjac(pJ,du,u,p,t)`: returns the parameter Jacobian ``\frac{df}{dp}``.
- `colorvec`: a color vector according to the SparseDiffTools.jl definition for the sparsity
  pattern of the `jac_prototype`. This specializes the Jacobian construction when using
  finite differences and automatic differentiation to be computed in an accelerated manner
  based on the sparsity pattern. Defaults to `nothing`, which means a color vector will be
  internally computed on demand when required. The cost of this operation is highly dependent
  on the sparsity pattern.

## iip: In-Place vs Out-Of-Place

For more details on this argument, see the ODEFunction documentation.

## specialize: Controlling Compilation and Specialization

For more details on this argument, see the ODEFunction documentation.

## Fields

The fields of the DynamicalBVPFunction type directly match the names of the inputs.
"""
struct DynamicalBVPFunction{
    iip, specialize, twopoint, F, BF, TMM, Ta, Tt, TJ, BCTJ, JVP, VJP,
    JP, BCJP, BCRP, SP, TW, TWt, TPJ, O, TCV, BCTCV,
    SYS} <: AbstractBVPFunction{iip, twopoint}
    f::F
    bc::BF
    mass_matrix::TMM
    analytic::Ta
    tgrad::Tt
    jac::TJ
    bcjac::BCTJ
    jvp::JVP
    vjp::VJP
    jac_prototype::JP
    bcjac_prototype::BCJP
    bcresid_prototype::BCRP
    sparsity::SP
    Wfact::TW
    Wfact_t::TWt
    paramjac::TPJ
    observed::O
    colorvec::TCV
    bccolorvec::BCTCV
    sys::SYS
end

@doc doc"""
    IntegralFunction{iip,specialize,F,T} <: AbstractIntegralFunction{iip}

A representation of an integrand `f` defined by:

```math
f(u, p)
```

For an in-place form of `f` see the `iip` section below for details on in-place or
out-of-place handling.

```julia
IntegralFunction{iip,specialize}(f, [integrand_prototype])
```

Note that only `f` is required, and in the case of inplace integrands a mutable array
`integrand_prototype` to store the result of the integrand. If `integrand_prototype` is
present for either in-place or out-of-place integrands it is used to infer the return type
of the integrand.

## iip: In-Place vs Out-Of-Place

Out-of-place functions must be of the form ``y = f(u, p)`` and in-place functions of the form
``f(y, u, p)``, where `y` is a number or array containing the output. Since `f` is allowed to return any type (e.g. real or complex numbers or
arrays), in-place functions must provide a container `integrand_prototype` that is of the
right type and size for the variable ``y``, and the result is written to this container in-place.
When in-place forms are used, in-place array operations, i.e. broadcasting, may be used by
algorithms to reduce allocations. If `integrand_prototype` is not provided, `f` is assumed
to be out-of-place.

## specialize

This field is currently unused

## Fields

The fields of the IntegralFunction type directly match the names of the inputs.
"""
struct IntegralFunction{iip, specialize, F, T} <:
       AbstractIntegralFunction{iip}
    f::F
    integrand_prototype::T
end

@doc doc"""
    BatchIntegralFunction{iip,specialize,F,T} <: AbstractIntegralFunction{iip}

A batched representation of an (non-batched) integrand `f(u, p)` that can be
evaluated at multiple points simultaneously using threads, the gpu, or
distributed memory defined by:

```math
by = bf(bu, p)
```

Here we prefix variables with `b` to indicate they are batched variables, which
implies that they are arrays whose **last** dimension is reserved for batching
different evaluation points, or function values, and may be of a variable
length. ``bu`` is an array whose elements correspond to distinct evaluation
points to `f`,  and `bf` is a function to evaluate `f` 'point-wise' so that
`f(bu[..., i], p) == bf(bu, p)[..., i]`. For example, a simple batching implementation
of a scalar, univariate function is via broadcasting: `bf(bu, p) = f.(bu, Ref(p))`,
although this interface exists in order to allow user parallelization.
In general, the integration algorithm is allowed to vary the number of
evaluation points between subsequent calls to `bf`.

For an in-place form of `bf` see the `iip` section below for details on in-place
or out-of-place handling.

```julia
BatchIntegralFunction{iip,specialize}(bf, [integrand_prototype];
                                     max_batch=typemax(Int))
```
Note that only `bf` is required, and in the case of inplace integrands a mutable
array `integrand_prototype` to store a batch of integrand evaluations, with
a last "batching" dimension.

The keyword `max_batch` is used to set a soft limit on the number of points to
batch at the same time so that memory usage is controlled.

If `integrand_prototype` is present for either in-place or out-of-place integrands it is
used to infer the return type of the integrand.

## iip: In-Place vs Out-Of-Place

Out-of-place functions must be of the form `by = bf(bu, p)` and in-place
functions of the form `bf(by, bu, p)` where `by` is a batch array containing the
output. Since the algorithm may vary the number of points to batch, the batching
dimension can be of any length, including zero, and since `bf` is allowed to
return arrays of any type (e.g. real or complex) or size, in-place functions
must provide a container `integrand_prototype` of the desired type and size for
`by`. If `integrand_prototype` is not provided, `bf` is assumed to be
out-of-place.

In the out-of-place case, we require `f(bu[..., i], p) == bf(bu, p)[..., i]`,
and certain algorithms, such as those implemented in C, may infer the type or
shape of `by` by calling `bf` with an empty array of input points, i.e. `bu`
with `size(bu)[end] == 0`. Then it is expected for the resulting `by` to have
the same type and `size(by)[begin:end-1]` for all subsequent calls.

When the in-place form is used, we require `f(by[..., i], bu[..., i], p) ==
bf(by, bu, p)[..., i]` and `size(by)[begin:end-1] ==
size(integrand_prototype)[begin:end-1]`. The algorithm should always pass the
integrand `by` arrays that are `similar` to `integrand_prototype`, and may use
views and in-place array operations to reduce allocations.

## specialize

This field is currently unused

## Fields

The fields of the BatchIntegralFunction type are `f`, corresponding to `bf`
above, and `integrand_prototype`.
"""
struct BatchIntegralFunction{iip, specialize, F, T} <:
       AbstractIntegralFunction{iip}
    f::F
    integrand_prototype::T
    max_batch::Int
end

######### Backwards Compatibility Overloads

(f::ODEFunction)(args...) = f.f(args...)
(f::NonlinearFunction)(args...) = f.f(args...)
(f::IntervalNonlinearFunction)(args...) = f.f(args...)
(f::IntegralFunction)(args...) = f.f(args...)
(f::BatchIntegralFunction)(args...) = f.f(args...)

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
(f::ImplicitDiscreteFunction)(args...) = f.f(args...)
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

(f::BVPFunction)(args...) = f.f(args...)
(f::DynamicalBVPFunction)(args...) = f.f(args...)

######### Basic Constructor

function ODEFunction{iip, specialize}(f;
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
        W_prototype = __has_W_prototype(f) ? f.W_prototype : nothing,
        paramjac = __has_paramjac(f) ? f.paramjac : nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        initializeprob = __has_initializeprob(f) ? f.initializeprob : nothing,
        update_initializeprob! = __has_update_initializeprob!(f) ?
                                 f.update_initializeprob! : nothing,
        initializeprobmap = __has_initializeprobmap(f) ? f.initializeprobmap : nothing,
        initializeprobpmap = __has_initializeprobpmap(f) ? f.initializeprobpmap : nothing
) where {iip,
        specialize
}
    if mass_matrix === I && f isa Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if (specialize === FunctionWrapperSpecialize) &&
       !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
        error("FunctionWrapperSpecialize must be used on the problem constructor for access to u0, p, and t types!")
    end

    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
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

    _f = prepare_function(f)

    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    @assert typeof(initializeprob) <:
            Union{Nothing, NonlinearProblem, NonlinearLeastSquaresProblem}

    if specialize === NoSpecialize
        ODEFunction{iip, specialize,
            Any, Any, Any, Any,
            Any, Any, Any, typeof(jac_prototype),
            typeof(sparsity), Any, Any, typeof(W_prototype), Any,
            Any,
            typeof(_colorvec),
            typeof(sys), Any, Any, Any, Any}(_f, mass_matrix, analytic, tgrad, jac,
            jvp, vjp, jac_prototype, sparsity, Wfact,
            Wfact_t, W_prototype, paramjac,
            observed, _colorvec, sys, initializeprob, update_initializeprob!, initializeprobmap,
            initializeprobpmap)
    elseif specialize === false
        ODEFunction{iip, FunctionWrapperSpecialize,
            typeof(_f), typeof(mass_matrix), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t), typeof(W_prototype),
            typeof(paramjac),
            typeof(observed),
            typeof(_colorvec),
            typeof(sys), typeof(initializeprob), typeof(update_initializeprob!),
            typeof(initializeprobmap), typeof(initializeprobpmap)}(_f, mass_matrix,
            analytic, tgrad, jac,
            jvp, vjp, jac_prototype, sparsity, Wfact,
            Wfact_t, W_prototype, paramjac,
            observed, _colorvec, sys, initializeprob, update_initializeprob!, initializeprobmap,
            initializeprobpmap)
    else
        ODEFunction{iip, specialize,
            typeof(_f), typeof(mass_matrix), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t), typeof(W_prototype),
            typeof(paramjac),
            typeof(observed),
            typeof(_colorvec),
            typeof(sys), typeof(initializeprob), typeof(update_initializeprob!),
            typeof(initializeprobmap),
            typeof(initializeprobpmap)}(_f, mass_matrix, analytic, tgrad, jac,
            jvp, vjp, jac_prototype, sparsity, Wfact,
            Wfact_t, W_prototype, paramjac,
            observed, _colorvec, sys, initializeprob, update_initializeprob!, initializeprobmap,
            initializeprobpmap)
    end
end

function ODEFunction{iip}(f; kwargs...) where {iip}
    ODEFunction{iip, FullSpecialize}(f; kwargs...)
end
ODEFunction{iip}(f::ODEFunction; kwargs...) where {iip} = f
ODEFunction(f; kwargs...) = ODEFunction{isinplace(f, 4), FullSpecialize}(f; kwargs...)
ODEFunction(f::ODEFunction; kwargs...) = f

function unwrapped_f(f::ODEFunction, newf = unwrapped_f(f.f))
    if specialization(f) === NoSpecialize
        ODEFunction{isinplace(f), specialization(f), Any, Any, Any,
            Any, Any, Any, Any, typeof(f.jac_prototype),
            typeof(f.sparsity), Any, Any, Any,
            Any, typeof(f.colorvec),
            typeof(f.sys), Any, Any, Any, Any}(
            newf, f.mass_matrix, f.analytic, f.tgrad, f.jac,
            f.jvp, f.vjp, f.jac_prototype, f.sparsity, f.Wfact,
            f.Wfact_t, f.W_prototype, f.paramjac,
            f.observed, f.colorvec, f.sys, f.initializeprob,
            f.update_initializeprob!, f.initializeprobmap,
            f.initializeprobpmap)
    else
        ODEFunction{isinplace(f), specialization(f), typeof(newf), typeof(f.mass_matrix),
            typeof(f.analytic), typeof(f.tgrad),
            typeof(f.jac), typeof(f.jvp), typeof(f.vjp), typeof(f.jac_prototype),
            typeof(f.sparsity), typeof(f.Wfact), typeof(f.Wfact_t), typeof(f.W_prototype),
            typeof(f.paramjac),
            typeof(f.observed), typeof(f.colorvec),
            typeof(f.sys), typeof(f.initializeprob), typeof(f.update_initializeprob!),
            typeof(f.initializeprobmap),
            typeof(f.initializeprobpmap)}(newf, f.mass_matrix, f.analytic, f.tgrad, f.jac,
            f.jvp, f.vjp, f.jac_prototype, f.sparsity, f.Wfact,
            f.Wfact_t, f.W_prototype, f.paramjac,
            f.observed, f.colorvec, f.sys, f.initializeprob, f.update_initializeprob!,
            f.initializeprobmap, f.initializeprobpmap)
    end
end

"""
$(SIGNATURES)

Converts a NonlinearFunction into an ODEFunction.
"""
function ODEFunction(f::NonlinearFunction)
    iip = isinplace(f)
    ODEFunction{iip}(f)
end

function ODEFunction{iip}(f::NonlinearFunction) where {iip}
    _f = iip ? (du, u, p, t) -> (f.f(du, u, p); nothing) : (u, p, t) -> f.f(u, p)
    if f.analytic !== nothing
        _analytic = (u0, p, t) -> f.analytic(u0, p)
    else
        _analytic = nothing
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
        jac = _jac,
        jvp = _jvp,
        vjp = _vjp,
        jac_prototype = f.jac_prototype,
        sparsity = f.sparsity,
        paramjac = f.paramjac,
        sys = f.sys,
        observed = f.observed,
        colorvec = f.colorvec)
end

"""
$(SIGNATURES)

Converts an ODEFunction into a NonlinearFunction.
"""
function NonlinearFunction(f::ODEFunction)
    iip = isinplace(f)
    NonlinearFunction{iip}(f)
end

function NonlinearFunction{iip}(f::ODEFunction) where {iip}
    _f = iip ? (du, u, p) -> (f.f(du, u, p, Inf); nothing) : (u, p) -> f.f(u, p, Inf)
    if f.analytic !== nothing
        _analytic = (u0, p) -> f.analytic(u0, p, Inf)
    else
        _analytic = nothing
    end
    if f.jac !== nothing
        _jac = iip ? (J, u, p) -> (f.jac(J, u, p, Inf); nothing) :
               (u, p) -> f.jac(u, p, Inf)
    else
        _jac = nothing
    end
    if f.jvp !== nothing
        _jvp = iip ? (Jv, u, p) -> (f.jvp(Jv, u, p, Inf); nothing) :
               (u, p) -> f.jvp(u, p, Inf)
    else
        _jvp = nothing
    end
    if f.vjp !== nothing
        _vjp = iip ? (vJ, u, p) -> (f.vjp(vJ, u, p, Inf); nothing) :
               (u, p) -> f.vjp(u, p, Inf)
    else
        _vjp = nothing
    end

    NonlinearFunction{iip, specialization(f)}(_f;
        analytic = _analytic,
        jac = _jac,
        jvp = _jvp,
        vjp = _vjp,
        jac_prototype = f.jac_prototype,
        sparsity = f.sparsity,
        paramjac = f.paramjac,
        sys = f.sys,
        observed = f.observed,
        colorvec = f.colorvec)
end

@add_kwonly function SplitFunction(f1, f2, mass_matrix, cache, analytic, tgrad, jac, jvp,
        vjp, jac_prototype, W_prototype, sparsity, Wfact, Wfact_t, paramjac,
        observed, colorvec, sys, initializeprob, update_initializeprob!,
        initializeprobmap, initializeprobpmap)
    f1 = ODEFunction(f1)
    f2 = ODEFunction(f2)

    if !(f1 isa AbstractSciMLOperator || f1.f isa AbstractSciMLOperator) &&
       isinplace(f1) != isinplace(f2)
        throw(NonconformingFunctionsError(["f2"]))
    end

    SplitFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
        typeof(mass_matrix),
        typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
        typeof(vjp), typeof(jac_prototype), typeof(W_prototype), typeof(sparsity),
        typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed), typeof(colorvec),
        typeof(sys), typeof(initializeprob), typeof(update_initializeprob!), typeof(initializeprobmap),
        typeof(initializeprobpmap)}(
        f1, f2, mass_matrix,
        cache, analytic, tgrad, jac, jvp, vjp,
        jac_prototype, W__prototype, sparsity, Wfact, Wfact_t, paramjac, observed, colorvec, sys,
        initializeprob, update_initializeprob!, initializeprobmap, initializeprobpmap)
end
function SplitFunction{iip, specialize}(f1, f2;
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
        W_prototype = __has_W_prototype(f1) ?
                      f1.W_prototype :
                      nothing,
        sparsity = __has_sparsity(f1) ? f1.sparsity :
                   jac_prototype,
        Wfact = __has_Wfact(f1) ? f1.Wfact : nothing,
        Wfact_t = __has_Wfact_t(f1) ? f1.Wfact_t : nothing,
        paramjac = __has_paramjac(f1) ? f1.paramjac :
                   nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f1) ? f1.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f1) ? f1.colorvec :
                   nothing,
        sys = __has_sys(f1) ? f1.sys : nothing,
        initializeprob = __has_initializeprob(f1) ? f1.initializeprob : nothing,
        update_initializeprob! = __has_update_initializeprob!(f1) ?
                                 f1.update_initializeprob! : nothing,
        initializeprobmap = __has_initializeprobmap(f1) ? f1.initializeprobmap : nothing,
        initializeprobpmap = __has_initializeprobpmap(f1) ? f1.initializeprobpmap : nothing
) where {iip,
        specialize
}
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)
    @assert typeof(initializeprob) <:
            Union{Nothing, NonlinearProblem, NonlinearLeastSquaresProblem}

    if specialize === NoSpecialize
        SplitFunction{iip, specialize, Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any}(f1, f2, mass_matrix, _func_cache,
            analytic,
            tgrad, jac, jvp, vjp, jac_prototype, W_prototype,
            sparsity, Wfact, Wfact_t, paramjac,
            observed, colorvec, sys, initializeprob.update_initializeprob!, initializeprobmap,
            initializeprobpmap, initializeprobpmap)
    else
        SplitFunction{iip, specialize, typeof(f1), typeof(f2), typeof(mass_matrix),
            typeof(_func_cache), typeof(analytic),
            typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
            typeof(jac_prototype), typeof(W_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(colorvec),
            typeof(sys), typeof(initializeprob), typeof(update_initializeprob!),
            typeof(initializeprobmap),
            typeof(initializeprobpmap)}(f1, f2,
            mass_matrix, _func_cache, analytic, tgrad, jac,
            jvp, vjp, jac_prototype, W_prototype,
            sparsity, Wfact, Wfact_t, paramjac, observed, colorvec, sys,
            initializeprob, update_initializeprob!, initializeprobmap, initializeprobpmap)
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
        Wfact_t, paramjac,
        observed, colorvec, sys) where {iip}
    f1 = f1 isa AbstractSciMLOperator ? f1 : ODEFunction(f1)
    f2 = ODEFunction(f2)

    if isinplace(f1) != isinplace(f2)
        throw(NonconformingFunctionsError(["f2"]))
    end
    DynamicalODEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
        typeof(mass_matrix),
        typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
        typeof(vjp),
        typeof(jac_prototype),
        typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
        typeof(colorvec),
        typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
        vjp, jac_prototype, sparsity, Wfact, Wfact_t,
        paramjac, observed,
        colorvec, sys)
end

function DynamicalODEFunction{iip, specialize}(f1, f2;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f1) ? f1.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f1) ? f1.colorvec :
                   nothing,
        sys = __has_sys(f1) ? f1.sys : nothing) where {
        iip,
        specialize
}
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        DynamicalODEFunction{iip, specialize, Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any,
            Any, Any, Any, Any}(f1, f2, mass_matrix,
            analytic,
            tgrad,
            jac, jvp, vjp,
            jac_prototype,
            sparsity,
            Wfact, Wfact_t, paramjac,
            observed, colorvec, sys)
    else
        DynamicalODEFunction{iip, specialize, typeof(f1), typeof(f2), typeof(mass_matrix),
            typeof(analytic),
            typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
            typeof(jac_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(colorvec),
            typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
            vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, observed,
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

function DiscreteFunction{iip, specialize}(f;
        analytic = __has_analytic(f) ? f.analytic :
                   nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        sys = __has_sys(f) ? f.sys : nothing) where {iip,
        specialize
}
    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        DiscreteFunction{iip, specialize, Any, Any, Any, Any}(_f, analytic,
            observed, sys)
    else
        DiscreteFunction{iip, specialize, typeof(_f), typeof(analytic),
            typeof(observed), typeof(sys)}(_f, analytic, observed, sys)
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

function unwrapped_f(f::DiscreteFunction, newf = unwrapped_f(f.f))
    specialize = specialization(f)

    if specialize === NoSpecialize
        DiscreteFunction{isinplace(f), specialize, Any, Any,
            Any, Any}(newf, f.analytic, f.observed, f.sys)
    else
        DiscreteFunction{isinplace(f), specialize, typeof(newf), typeof(f.analytic),
            typeof(f.observed), typeof(f.sys)}(newf, f.analytic,
            f.observed, f.sys)
    end
end

function ImplicitDiscreteFunction{iip, specialize}(f;
        analytic = __has_analytic(f) ?
                   f.analytic :
                   nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ?
                   f.observed :
                   DEFAULT_OBSERVED,
        sys = __has_sys(f) ? f.sys : nothing) where {
        iip,
        specialize
}
    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        ImplicitDiscreteFunction{iip, specialize, Any, Any, Any, Any}(_f,
            analytic,
            observed,
            sys)
    else
        ImplicitDiscreteFunction{
            iip, specialize, typeof(_f), typeof(analytic), typeof(observed), typeof(sys)}(
            _f, analytic, observed, sys)
    end
end

function ImplicitDiscreteFunction{iip}(f; kwargs...) where {iip}
    ImplicitDiscreteFunction{iip, FullSpecialize}(f; kwargs...)
end
ImplicitDiscreteFunction{iip}(f::ImplicitDiscreteFunction; kwargs...) where {iip} = f
function ImplicitDiscreteFunction(f; kwargs...)
    ImplicitDiscreteFunction{isinplace(f, 5), FullSpecialize}(f; kwargs...)
end
ImplicitDiscreteFunction(f::ImplicitDiscreteFunction; kwargs...) = f

function unwrapped_f(f::ImplicitDiscreteFunction, newf = unwrapped_f(f.f))
    specialize = specialization(f)

    if specialize === NoSpecialize
        ImplicitDiscreteFunction{isinplace(f, 6), specialize, Any, Any,
            Any, Any}(newf, f.analytic, f.observed, f.sys)
    else
        ImplicitDiscreteFunction{isinplace(f, 6), specialize, typeof(newf),
            typeof(f.analytic),
            typeof(f.observed), typeof(f.sys)}(newf, f.analytic,
            f.observed, f.sys)
    end
end

function SDEFunction{iip, specialize}(f, g;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing) where {iip,
        specialize
}
    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
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

    _f = prepare_function(f)
    _g = prepare_function(g)

    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        SDEFunction{iip, specialize, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any,
            typeof(_colorvec), typeof(sys)}(_f, _g, mass_matrix, analytic,
            tgrad, jac, jvp, vjp,
            jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, ggprime, observed,
            _colorvec, sys)
    else
        SDEFunction{iip, specialize, typeof(_f), typeof(_g),
            typeof(mass_matrix), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
            typeof(paramjac), typeof(ggprime), typeof(observed), typeof(_colorvec), typeof(sys)}(
            _f, _g, mass_matrix,
            analytic, tgrad, jac,
            jvp, vjp,
            jac_prototype,
            sparsity, Wfact,
            Wfact_t,
            paramjac, ggprime,
            observed, _colorvec,
            sys)
    end
end

function unwrapped_f(f::SDEFunction, newf = unwrapped_f(f.f),
        newg = unwrapped_f(f.g))
    specialize = specialization(f)

    if specialize === NoSpecialize
        SDEFunction{isinplace(f), specialize, Any, Any,
            typeof(f.mass_matrix), Any, Any,
            Any, Any, Any, typeof(f.jac_prototype),
            typeof(f.sparsity), Any, Any,
            Any, Any,
            typeof(f.observed), typeof(f.colorvec), typeof(f.sys)}(newf, newg,
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
            f.observed,
            f.colorvec,
            f.sys)
    else
        SDEFunction{isinplace(f), specialize, typeof(newf), typeof(newg),
            typeof(f.mass_matrix), typeof(f.analytic), typeof(f.tgrad),
            typeof(f.jac), typeof(f.jvp), typeof(f.vjp), typeof(f.jac_prototype),
            typeof(f.sparsity), typeof(f.Wfact), typeof(f.Wfact_t),
            typeof(f.paramjac), typeof(f.ggprime),
            typeof(f.observed), typeof(f.colorvec), typeof(f.sys)}(newf, newg,
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
        colorvec, sys)
    f1 = f1 isa AbstractSciMLOperator ? f1 : SDEFunction(f1)
    f2 = SDEFunction(f2)

    SplitFunction{isinplace(f2), typeof(f1), typeof(f2), typeof(g), typeof(mass_matrix),
        typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
        typeof(vjp),
        typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
        typeof(colorvec),
        typeof(sys)}(f1, f2, mass_matrix, cache, analytic, tgrad, jac,
        jac_prototype, Wfact, Wfact_t, paramjac, observed, colorvec, sys)
end

function SplitSDEFunction{iip, specialize}(f1, f2, g;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f1) ? f1.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f1) ? f1.colorvec :
                   nothing,
        sys = __has_sys(f1) ? f1.sys : nothing) where {
        iip,
        specialize
}
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        SplitSDEFunction{iip, specialize, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any}(f1, f2, g, mass_matrix, _func_cache,
            analytic,
            tgrad, jac, jvp, vjp, jac_prototype,
            sparsity,
            Wfact, Wfact_t, paramjac, observed,
            colorvec, sys)
    else
        SplitSDEFunction{iip, specialize, typeof(f1), typeof(f2), typeof(g),
            typeof(mass_matrix), typeof(_func_cache),
            typeof(analytic),
            typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
            typeof(jac_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(colorvec),
            typeof(sys)}(f1, f2, g, mass_matrix, _func_cache, analytic,
            tgrad, jac, jvp, vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac,
            observed, colorvec, sys)
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
        observed, colorvec,
        sys)
    f1 = f1 isa AbstractSciMLOperator ? f1 : SDEFunction(f1)
    f2 = SDEFunction(f2)

    DynamicalSDEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2), typeof(g),
        typeof(mass_matrix),
        typeof(cache), typeof(analytic), typeof(tgrad), typeof(jac),
        typeof(jvp), typeof(vjp),
        typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
        typeof(colorvec),
        typeof(sys)}(f1, f2, g, mass_matrix, cache, analytic, tgrad,
        jac, jac_prototype, Wfact, Wfact_t, paramjac, observed, colorvec, sys)
end

function DynamicalSDEFunction{iip, specialize}(f1, f2, g;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f1) ? f1.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f1) ? f1.colorvec :
                   nothing,
        sys = __has_sys(f1) ? f1.sys : nothing) where {
        iip,
        specialize
}
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        DynamicalSDEFunction{iip, specialize, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any}(f1, f2, g, mass_matrix,
            _func_cache,
            analytic, tgrad, jac, jvp, vjp,
            jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, observed,
            colorvec, sys)
    else
        DynamicalSDEFunction{iip, specialize, typeof(f1), typeof(f2), typeof(g),
            typeof(mass_matrix), typeof(_func_cache),
            typeof(analytic),
            typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
            typeof(jac_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(colorvec),
            typeof(sys)}(f1, f2, g, mass_matrix, _func_cache, analytic,
            tgrad, jac, jvp, vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, observed, colorvec, sys)
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

function RODEFunction{iip, specialize}(f;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        analytic_full = __has_analytic_full(f) ?
                        f.analytic_full : false) where {iip,
        specialize
}
    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
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

    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        RODEFunction{iip, specialize, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any,
            Any,
            typeof(_colorvec), Any}(_f, mass_matrix, analytic,
            tgrad,
            jac, jvp, vjp,
            jac_prototype,
            sparsity, Wfact, Wfact_t,
            paramjac, observed,
            _colorvec, sys,
            analytic_full)
    else
        RODEFunction{iip, specialize, typeof(_f), typeof(mass_matrix),
            typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
            typeof(paramjac),
            typeof(observed), typeof(_colorvec),
            typeof(sys)}(_f, mass_matrix, analytic, tgrad,
            jac, jvp, vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac,
            observed, _colorvec, sys, analytic_full)
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

function DAEFunction{iip, specialize}(f;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        initializeprob = __has_initializeprob(f) ? f.initializeprob : nothing,
        update_initializeprob! = __has_update_initializeprob!(f) ?
                                 f.update_initializeprob! : nothing,
        initializeprobmap = __has_initializeprobmap(f) ? f.initializeprobmap : nothing,
        initializeprobpmap = __has_initializeprobpmap(f) ? f.initializeprobpmap : nothing) where {
        iip,
        specialize
}
    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
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

    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    @assert typeof(initializeprob) <:
            Union{Nothing, NonlinearProblem, NonlinearLeastSquaresProblem}

    if specialize === NoSpecialize
        DAEFunction{iip, specialize, Any, Any, Any,
            Any, Any, Any, Any, Any,
            Any, Any, Any,
            Any, typeof(_colorvec), Any, Any, Any, Any, Any}(_f, analytic, tgrad, jac, jvp,
            vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, observed,
            _colorvec, sys, initializeprob, update_initializeprob!,
            initializeprobmap, initializeprobpmap)
    else
        DAEFunction{iip, specialize, typeof(_f), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
            typeof(paramjac),
            typeof(observed), typeof(_colorvec),
            typeof(sys), typeof(initializeprob), typeof(update_initializeprob!),
            typeof(initializeprobmap),
            typeof(initializeprobpmap)}(
            _f, analytic, tgrad, jac, jvp, vjp,
            jac_prototype, sparsity, Wfact, Wfact_t,
            paramjac, observed,
            _colorvec, sys, initializeprob, update_initializeprob!,
            initializeprobmap, initializeprobpmap)
    end
end

function DAEFunction{iip}(f; kwargs...) where {iip}
    DAEFunction{iip, FullSpecialize}(f; kwargs...)
end
DAEFunction{iip}(f::DAEFunction; kwargs...) where {iip} = f
DAEFunction(f; kwargs...) = DAEFunction{isinplace(f, 5), FullSpecialize}(f; kwargs...)
DAEFunction(f::DAEFunction; kwargs...) = f

function DDEFunction{iip, specialize}(f;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing) where {iip,
        specialize
}
    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    jaciip = jac !== nothing ? isinplace(jac, 5, "jac", iip) : iip
    tgradiip = tgrad !== nothing ? isinplace(tgrad, 5, "tgrad", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 6, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 6, "vjp", iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact, 6, "Wfact", iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t, 6, "Wfact_t", iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac, 5, "paramjac", iip) : iip

    nonconforming = (jaciip, tgradiip, jvpiip, vjpiip, Wfactiip, Wfact_tiip,
        paramjaciip) .!= iip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["jac", "tgrad", "jvp", "vjp", "Wfact", "Wfact_t", "paramjac"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        DDEFunction{iip, specialize, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any,
            Any,
            Any, typeof(_colorvec), Any}(_f, mass_matrix,
            analytic,
            tgrad,
            jac, jvp, vjp,
            jac_prototype,
            sparsity, Wfact,
            Wfact_t,
            paramjac,
            observed,
            _colorvec, sys)
    else
        DDEFunction{iip, specialize, typeof(_f), typeof(mass_matrix), typeof(analytic),
            typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
            typeof(paramjac),
            typeof(observed),
            typeof(_colorvec), typeof(sys)}(_f, mass_matrix, analytic,
            tgrad, jac, jvp, vjp,
            jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac,
            observed,
            _colorvec, sys)
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
        observed,
        colorvec) where {iip}
    f1 = f1 isa AbstractSciMLOperator ? f1 : DDEFunction(f1)
    f2 = DDEFunction(f2)

    DynamicalDDEFunction{isinplace(f2), FullSpecialize, typeof(f1), typeof(f2),
        typeof(mass_matrix),
        typeof(analytic), typeof(tgrad), typeof(jac), typeof(jvp),
        typeof(vjp),
        typeof(jac_prototype),
        typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
        typeof(colorvec),
        typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
        vjp, jac_prototype, sparsity, Wfact, Wfact_t,
        paramjac, observed,
        colorvec, sys)
end
function DynamicalDDEFunction{iip, specialize}(f1, f2;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f1) ? f1.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f1) ? f1.colorvec :
                   nothing,
        sys = __has_sys(f1) ? f1.sys : nothing) where {
        iip,
        specialize
}
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        DynamicalDDEFunction{iip, specialize, Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any}(f1, f2, mass_matrix,
            analytic,
            tgrad,
            jac, jvp, vjp,
            jac_prototype,
            sparsity,
            Wfact, Wfact_t,
            paramjac,
            observed, colorvec,
            sys)
    else
        DynamicalDDEFunction{iip, typeof(f1), typeof(f2), typeof(mass_matrix),
            typeof(analytic),
            typeof(tgrad), typeof(jac), typeof(jvp), typeof(vjp),
            typeof(jac_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(colorvec),
            typeof(sys)}(f1, f2, mass_matrix, analytic, tgrad, jac, jvp,
            vjp, jac_prototype, sparsity,
            Wfact, Wfact_t, paramjac, observed,
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

function SDDEFunction{iip, specialize}(f, g;
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
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing) where {iip,
        specialize
}
    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    _f = prepare_function(f)
    _g = prepare_function(g)
    sys = sys_or_symbolcache(sys, syms, paramsyms, indepsym)

    if specialize === NoSpecialize
        SDDEFunction{iip, specialize, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any,
            Any, Any,
            Any, typeof(_colorvec), Any}(_f, _g, mass_matrix,
            analytic, tgrad,
            jac,
            jvp,
            vjp,
            jac_prototype,
            sparsity, Wfact,
            Wfact_t,
            paramjac, ggprime,
            observed,
            _colorvec,
            sys)
    else
        SDDEFunction{iip, specialize, typeof(_f), typeof(_g),
            typeof(mass_matrix), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact), typeof(Wfact_t),
            typeof(paramjac), typeof(ggprime), typeof(observed),
            typeof(_colorvec), typeof(sys)}(_f, _g, mass_matrix,
            analytic, tgrad, jac,
            jvp, vjp, jac_prototype,
            sparsity, Wfact,
            Wfact_t,
            paramjac, ggprime,
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

function NonlinearFunction{iip, specialize}(f;
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
        Wfact_t = __has_Wfact_t(f) ? f.Wfact_t :
                  nothing,
        paramjac = __has_paramjac(f) ? f.paramjac :
                   nothing,
        syms = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED_NO_TIME,
        colorvec = __has_colorvec(f) ? f.colorvec :
                   nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        resid_prototype = __has_resid_prototype(f) ? f.resid_prototype : nothing) where {
        iip, specialize}
    if mass_matrix === I && f isa Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if jac === nothing && isa(jac_prototype, AbstractSciMLOperator)
        if iip
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterface.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterface.matrix_colors(jac_prototype)
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

    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms)
    if specialize === NoSpecialize
        NonlinearFunction{iip, specialize,
            Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any,
            Any, Any, Any,
            typeof(_colorvec), Any, Any}(_f, mass_matrix,
            analytic, tgrad, jac,
            jvp, vjp,
            jac_prototype,
            sparsity, Wfact,
            Wfact_t, paramjac,
            observed,
            _colorvec, sys, resid_prototype)
    else
        NonlinearFunction{iip, specialize,
            typeof(_f), typeof(mass_matrix), typeof(analytic), typeof(tgrad),
            typeof(jac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(sparsity), typeof(Wfact),
            typeof(Wfact_t), typeof(paramjac),
            typeof(observed),
            typeof(_colorvec), typeof(sys), typeof(resid_prototype)}(_f, mass_matrix,
            analytic, tgrad, jac,
            jvp, vjp, jac_prototype, sparsity,
            Wfact,
            Wfact_t, paramjac,
            observed, _colorvec, sys, resid_prototype)
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

function IntervalNonlinearFunction{iip, specialize}(f;
        analytic = __has_analytic(f) ?
                   f.analytic :
                   nothing,
        syms = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ?
                   f.observed :
                   DEFAULT_OBSERVED_NO_TIME,
        sys = __has_sys(f) ? f.sys : nothing) where {
        iip,
        specialize
}
    _f = prepare_function(f)
    sys = sys_or_symbolcache(sys, syms, paramsyms)

    if specialize === NoSpecialize
        IntervalNonlinearFunction{iip, specialize,
            Any, Any, Any, Any}(_f, analytic, observed, sys)
    else
        IntervalNonlinearFunction{iip, specialize,
            typeof(_f), typeof(analytic),
            typeof(observed),
            typeof(sys)}(_f, analytic,
            observed, sys)
    end
end

function IntervalNonlinearFunction{iip}(f; kwargs...) where {iip}
    IntervalNonlinearFunction{iip, FullSpecialize}(f; kwargs...)
end
IntervalNonlinearFunction{iip}(f::IntervalNonlinearFunction; kwargs...) where {iip} = f
function IntervalNonlinearFunction(f; kwargs...)
    IntervalNonlinearFunction{isinplace(f, 3), FullSpecialize}(f; kwargs...)
end
IntervalNonlinearFunction(f::IntervalNonlinearFunction; kwargs...) = f

struct NoAD <: AbstractADType end

(f::OptimizationFunction)(args...) = f.f(args...)
OptimizationFunction(args...; kwargs...) = OptimizationFunction{true}(args...; kwargs...)

function OptimizationFunction{iip}(f, adtype::AbstractADType = NoAD();
        grad = nothing, fg = nothing, hess = nothing, hv = nothing, fgh = nothing,
        cons = nothing, cons_j = nothing, cons_jvp = nothing,
        cons_vjp = nothing, cons_h = nothing,
        hess_prototype = nothing,
        cons_jac_prototype = __has_jac_prototype(f) ?
                             f.jac_prototype : nothing,
        cons_hess_prototype = nothing,
        syms = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED_NO_TIME,
        expr = nothing, cons_expr = nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        lag_h = nothing, lag_hess_prototype = nothing,
        hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        cons_jac_colorvec = __has_colorvec(f) ? f.colorvec :
                            nothing,
        cons_hess_colorvec = __has_colorvec(f) ? f.colorvec :
                             nothing,
        lag_hess_colorvec = nothing) where {iip}
    isinplace(f, 2; has_two_dispatches = false, isoptimization = true)
    sys = sys_or_symbolcache(sys, syms, paramsyms)
    OptimizationFunction{
        iip, typeof(adtype), typeof(f), typeof(grad), typeof(fg), typeof(hess),
        typeof(fgh), typeof(hv),
        typeof(cons), typeof(cons_j), typeof(cons_jvp),
        typeof(cons_vjp), typeof(cons_h),
        typeof(hess_prototype),
        typeof(cons_jac_prototype), typeof(cons_hess_prototype),
        typeof(observed),
        typeof(expr), typeof(cons_expr), typeof(sys), typeof(lag_h),
        typeof(lag_hess_prototype), typeof(hess_colorvec),
        typeof(cons_jac_colorvec), typeof(cons_hess_colorvec),
        typeof(lag_hess_colorvec)
    }(f, adtype, grad, fg, hess, fgh,
        hv, cons, cons_j, cons_jvp,
        cons_vjp, cons_h,
        hess_prototype, cons_jac_prototype,
        cons_hess_prototype, observed, expr, cons_expr, sys,
        lag_h, lag_hess_prototype, hess_colorvec, cons_jac_colorvec,
        cons_hess_colorvec, lag_hess_colorvec)
end

# Function call operator for MultiObjectiveOptimizationFunction
(f::MultiObjectiveOptimizationFunction)(args...) = f.f(args...)

# Convenience constructor
function MultiObjectiveOptimizationFunction(args...; kwargs...)
    MultiObjectiveOptimizationFunction{true}(args...; kwargs...)
end

# Constructor with keyword arguments
function MultiObjectiveOptimizationFunction{iip}(f, adtype::AbstractADType = NoAD();
        jac = nothing, hess = Vector{Nothing}(undef, 0), hv = nothing,
        cons = nothing, cons_j = nothing, cons_jvp = nothing,
        cons_vjp = nothing, cons_h = nothing,
        hess_prototype = nothing,
        cons_jac_prototype = __has_jac_prototype(f) ?
                             f.jac_prototype : nothing,
        cons_hess_prototype = nothing,
        syms = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed :
                   DEFAULT_OBSERVED_NO_TIME,
        expr = nothing, cons_expr = nothing,
        sys = __has_sys(f) ? f.sys : nothing,
        lag_h = nothing, lag_hess_prototype = nothing,
        hess_colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        cons_jac_colorvec = __has_colorvec(f) ? f.colorvec :
                            nothing,
        cons_hess_colorvec = __has_colorvec(f) ? f.colorvec :
                             nothing,
        lag_hess_colorvec = nothing) where {iip}
    isinplace(f, 2; has_two_dispatches = false, isoptimization = true)
    sys = sys_or_symbolcache(sys, syms, paramsyms)
    MultiObjectiveOptimizationFunction{
        iip, typeof(adtype), typeof(f), typeof(jac), typeof(hess),
        typeof(hv),
        typeof(cons), typeof(cons_j), typeof(cons_jvp),
        typeof(cons_vjp), typeof(cons_h),
        typeof(hess_prototype),
        typeof(cons_jac_prototype), typeof(cons_hess_prototype),
        typeof(observed),
        typeof(expr), typeof(cons_expr), typeof(sys), typeof(lag_h),
        typeof(lag_hess_prototype), typeof(hess_colorvec),
        typeof(cons_jac_colorvec), typeof(cons_hess_colorvec),
        typeof(lag_hess_colorvec)
    }(f, adtype, jac, hess,
        hv, cons, cons_j, cons_jvp,
        cons_vjp, cons_h,
        hess_prototype, cons_jac_prototype,
        cons_hess_prototype, observed, expr, cons_expr, sys,
        lag_h, lag_hess_prototype, hess_colorvec, cons_jac_colorvec,
        cons_hess_colorvec, lag_hess_colorvec)
end

function BVPFunction{iip, specialize, twopoint}(f, bc;
        mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
        analytic = __has_analytic(f) ? f.analytic : nothing,
        tgrad = __has_tgrad(f) ? f.tgrad : nothing,
        jac = __has_jac(f) ? f.jac : nothing,
        bcjac = __has_jac(bc) ? bc.jac : nothing,
        jvp = __has_jvp(f) ? f.jvp : nothing,
        vjp = __has_vjp(f) ? f.vjp : nothing,
        jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
        bcjac_prototype = __has_jac_prototype(bc) ? bc.jac_prototype : nothing,
        bcresid_prototype = nothing,
        sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
        Wfact = __has_Wfact(f) ? f.Wfact : nothing,
        Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
        paramjac = __has_paramjac(f) ? f.paramjac : nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed : DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        bccolorvec = __has_colorvec(bc) ? bc.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing) where {iip, specialize, twopoint}
    if mass_matrix === I && f isa Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if (specialize === FunctionWrapperSpecialize) &&
       !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
        error("FunctionWrapperSpecialize must be used on the problem constructor for access to u0, p, and t types!")
    end

    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip_f
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if bcjac === nothing && isa(bcjac_prototype, AbstractDiffEqLinearOperator)
        if iip_bc
            bcjac = update_coefficients! #(J,u,p,t)
        else
            bcjac = (u, p, t) -> update_coefficients!(deepcopy(bcjac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    if bcjac_prototype !== nothing && bccolorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(bcjac_prototype)
        _bccolorvec = ArrayInterfaceCore.matrix_colors(bcjac_prototype)
    else
        _bccolorvec = bccolorvec
    end

    bciip = if !twopoint
        try
            isinplace(bc, 4, "bc", iip)
        catch e
            isinplace(bc, 5, "bc", iip)
        end
    else
        @assert length(bc) == 2
        bc = Tuple(bc)
        if isinplace(first(bc), 3, "bc", iip) != isinplace(last(bc), 3, "bc", iip)
            throw(NonconformingFunctionsError(["bc[1]", "bc[2]"]))
        end
        isinplace(first(bc), 3, "bc", iip)
    end
    jaciip = jac !== nothing ? isinplace(jac, 4, "jac", iip) : iip
    bcjaciip = if bcjac !== nothing
        if !twopoint
            isinplace(bcjac, 4, "bcjac", bciip)
        else
            @assert length(bcjac) == 2
            bcjac = Tuple(bcjac)
            if isinplace(first(bcjac), 3, "bcjac", bciip) !=
               isinplace(last(bcjac), 3, "bcjac", bciip)
                throw(NonconformingFunctionsError(["bcjac[1]", "bcjac[2]"]))
            end
            isinplace(bcjac, 3, "bcjac", iip)
        end
    else
        bciip
    end
    tgradiip = tgrad !== nothing ? isinplace(tgrad, 4, "tgrad", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 5, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 5, "vjp", iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact, 5, "Wfact", iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t, 5, "Wfact_t", iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac, 4, "paramjac", iip) : iip

    nonconforming = (bciip, jaciip, tgradiip, jvpiip, vjpiip, Wfactiip, Wfact_tiip,
        paramjaciip) .!= iip
    bc_nonconforming = bcjaciip .!= bciip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["bc", "jac", "bcjac", "tgrad", "jvp", "vjp", "Wfact", "Wfact_t",
            "paramjac"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if twopoint
        if iip && (bcresid_prototype === nothing || length(bcresid_prototype) != 2)
            error("bcresid_prototype must be a tuple / indexable collection of length 2 for a inplace TwoPointBVPFunction")
        end
        if bcresid_prototype !== nothing && length(bcresid_prototype) == 2
            bcresid_prototype = ArrayPartition(first(bcresid_prototype),
                last(bcresid_prototype))
        end

        bccolorvec !== nothing && length(bccolorvec) == 2 &&
            (bccolorvec = Tuple(bccolorvec))

        bcjac_prototype !== nothing && length(bcjac_prototype) == 2 &&
            (bcjac_prototype = Tuple(bcjac_prototype))
    end

    if any(bc_nonconforming)
        bc_nonconforming = findall(bc_nonconforming)
        functions = ["bcjac"][bc_nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    _f = prepare_function(f)

    sys = something(sys, SymbolCache(syms, paramsyms, indepsym))

    if specialize === NoSpecialize
        BVPFunction{iip, specialize, twopoint, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any,
            Any, typeof(_colorvec), typeof(_bccolorvec), Any}(_f, bc, mass_matrix,
            analytic, tgrad, jac, bcjac, jvp, vjp, jac_prototype,
            bcjac_prototype, bcresid_prototype,
            sparsity, Wfact, Wfact_t, paramjac, observed,
            _colorvec, _bccolorvec, sys)
    else
        BVPFunction{iip, specialize, twopoint, typeof(_f), typeof(bc),
            typeof(mass_matrix), typeof(analytic), typeof(tgrad), typeof(jac),
            typeof(bcjac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(bcjac_prototype), typeof(bcresid_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(_colorvec), typeof(_bccolorvec), typeof(sys)}(
            _f, bc, mass_matrix, analytic,
            tgrad, jac, bcjac, jvp, vjp,
            jac_prototype, bcjac_prototype, bcresid_prototype, sparsity,
            Wfact, Wfact_t, paramjac,
            observed,
            _colorvec, _bccolorvec, sys)
    end
end

function BVPFunction{iip}(f, bc; twopoint::Union{Val, Bool} = Val(false),
        kwargs...) where {iip}
    BVPFunction{iip, FullSpecialize, _unwrap_val(twopoint)}(f, bc; kwargs...)
end
BVPFunction{iip}(f::BVPFunction, bc; kwargs...) where {iip} = f
function BVPFunction(f, bc; twopoint::Union{Val, Bool} = Val(false), kwargs...)
    BVPFunction{isinplace(f, 4), FullSpecialize, _unwrap_val(twopoint)}(f, bc; kwargs...)
end
BVPFunction(f::BVPFunction; kwargs...) = f

function DynamicalBVPFunction{iip, specialize, twopoint}(f, bc;
        mass_matrix = __has_mass_matrix(f) ? f.mass_matrix : I,
        analytic = __has_analytic(f) ? f.analytic : nothing,
        tgrad = __has_tgrad(f) ? f.tgrad : nothing,
        jac = __has_jac(f) ? f.jac : nothing,
        bcjac = __has_jac(bc) ? bc.jac : nothing,
        jvp = __has_jvp(f) ? f.jvp : nothing,
        vjp = __has_vjp(f) ? f.vjp : nothing,
        jac_prototype = __has_jac_prototype(f) ? f.jac_prototype : nothing,
        bcjac_prototype = __has_jac_prototype(bc) ? bc.jac_prototype : nothing,
        bcresid_prototype = nothing,
        sparsity = __has_sparsity(f) ? f.sparsity : jac_prototype,
        Wfact = __has_Wfact(f) ? f.Wfact : nothing,
        Wfact_t = __has_Wfact_t(f) ? f.Wfact_t : nothing,
        paramjac = __has_paramjac(f) ? f.paramjac : nothing,
        syms = nothing,
        indepsym = nothing,
        paramsyms = nothing,
        observed = __has_observed(f) ? f.observed : DEFAULT_OBSERVED,
        colorvec = __has_colorvec(f) ? f.colorvec : nothing,
        bccolorvec = __has_colorvec(bc) ? bc.colorvec : nothing,
        sys = __has_sys(f) ? f.sys : nothing) where {iip, specialize, twopoint}
    if mass_matrix === I && f isa Tuple
        mass_matrix = ((I for i in 1:length(f))...,)
    end

    if (specialize === FunctionWrapperSpecialize) &&
       !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
        error("FunctionWrapperSpecialize must be used on the problem constructor for access to u0, p, and t types!")
    end

    if jac === nothing && isa(jac_prototype, AbstractDiffEqLinearOperator)
        if iip_f
            jac = update_coefficients! #(J,u,p,t)
        else
            jac = (u, p, t) -> update_coefficients!(deepcopy(jac_prototype), u, p, t)
        end
    end

    if bcjac === nothing && isa(bcjac_prototype, AbstractDiffEqLinearOperator)
        if iip_bc
            bcjac = update_coefficients! #(J,u,p,t)
        else
            bcjac = (u, p, t) -> update_coefficients!(deepcopy(bcjac_prototype), u, p, t)
        end
    end

    if jac_prototype !== nothing && colorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(jac_prototype)
        _colorvec = ArrayInterfaceCore.matrix_colors(jac_prototype)
    else
        _colorvec = colorvec
    end

    if bcjac_prototype !== nothing && bccolorvec === nothing &&
       ArrayInterfaceCore.fast_matrix_colors(bcjac_prototype)
        _bccolorvec = ArrayInterfaceCore.matrix_colors(bcjac_prototype)
    else
        _bccolorvec = bccolorvec
    end

    bciip = if !twopoint
        isinplace(bc, 5, "bc", iip)
    else
        @assert length(bc) == 2
        bc = Tuple(bc)
        if isinplace(first(bc), 4, "bc", iip) != isinplace(last(bc), 4, "bc", iip)
            throw(NonconformingFunctionsError(["bc[1]", "bc[2]"]))
        end
        isinplace(first(bc), 4, "bc", iip)
    end
    jaciip = jac !== nothing ? isinplace(jac, 5, "jac", iip) : iip
    bcjaciip = if bcjac !== nothing
        if !twopoint
            isinplace(bcjac, 5, "bcjac", bciip)
        else
            @assert length(bcjac) == 2
            bcjac = Tuple(bcjac)
            if isinplace(first(bcjac), 4, "bcjac", bciip) !=
               isinplace(last(bcjac), 4, "bcjac", bciip)
                throw(NonconformingFunctionsError(["bcjac[1]", "bcjac[2]"]))
            end
            isinplace(bcjac, 4, "bcjac", iip)
        end
    else
        bciip
    end
    tgradiip = tgrad !== nothing ? isinplace(tgrad, 5, "tgrad", iip) : iip
    jvpiip = jvp !== nothing ? isinplace(jvp, 6, "jvp", iip) : iip
    vjpiip = vjp !== nothing ? isinplace(vjp, 6, "vjp", iip) : iip
    Wfactiip = Wfact !== nothing ? isinplace(Wfact, 6, "Wfact", iip) : iip
    Wfact_tiip = Wfact_t !== nothing ? isinplace(Wfact_t, 6, "Wfact_t", iip) : iip
    paramjaciip = paramjac !== nothing ? isinplace(paramjac, 5, "paramjac", iip) : iip

    nonconforming = (bciip, jaciip, tgradiip, jvpiip, vjpiip, Wfactiip, Wfact_tiip,
        paramjaciip) .!= iip
    bc_nonconforming = bcjaciip .!= bciip
    if any(nonconforming)
        nonconforming = findall(nonconforming)
        functions = ["bc", "jac", "bcjac", "tgrad", "jvp", "vjp", "Wfact", "Wfact_t",
            "paramjac"][nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    if twopoint
        if iip && (bcresid_prototype === nothing || length(bcresid_prototype) != 2)
            error("bcresid_prototype must be a tuple / indexable collection of length 2 for a inplace TwoPointBVPFunction")
        end
        if bcresid_prototype !== nothing && length(bcresid_prototype) == 2
            bcresid_prototype = ArrayPartition(first(bcresid_prototype),
                last(bcresid_prototype))
        end

        bccolorvec !== nothing && length(bccolorvec) == 2 &&
            (bccolorvec = Tuple(bccolorvec))

        bcjac_prototype !== nothing && length(bcjac_prototype) == 2 &&
            (bcjac_prototype = Tuple(bcjac_prototype))
    end

    if any(bc_nonconforming)
        bc_nonconforming = findall(bc_nonconforming)
        functions = ["bcjac"][bc_nonconforming]
        throw(NonconformingFunctionsError(functions))
    end

    _f = prepare_function(f)

    sys = something(sys, SymbolCache(syms, paramsyms, indepsym))

    if specialize === NoSpecialize
        DynamicalBVPFunction{iip, specialize, twopoint, Any, Any, Any, Any, Any,
            Any, Any, Any, Any, Any, Any, Any, Any, Any, Any,
            Any,
            Any, typeof(_colorvec), typeof(_bccolorvec), Any}(_f, bc, mass_matrix,
            analytic, tgrad, jac, bcjac, jvp, vjp, jac_prototype,
            bcjac_prototype, bcresid_prototype,
            sparsity, Wfact, Wfact_t, paramjac, observed,
            _colorvec, _bccolorvec, sys)
    else
        DynamicalBVPFunction{iip, specialize, twopoint, typeof(_f), typeof(bc),
            typeof(mass_matrix), typeof(analytic), typeof(tgrad), typeof(jac),
            typeof(bcjac), typeof(jvp), typeof(vjp), typeof(jac_prototype),
            typeof(bcjac_prototype), typeof(bcresid_prototype), typeof(sparsity),
            typeof(Wfact), typeof(Wfact_t), typeof(paramjac), typeof(observed),
            typeof(_colorvec), typeof(_bccolorvec), typeof(sys)}(
            _f, bc, mass_matrix, analytic,
            tgrad, jac, bcjac, jvp, vjp,
            jac_prototype, bcjac_prototype, bcresid_prototype, sparsity,
            Wfact, Wfact_t, paramjac,
            observed,
            _colorvec, _bccolorvec, sys)
    end
end

function DynamicalBVPFunction{iip}(f, bc; twopoint::Union{Val, Bool} = Val(false),
        kwargs...) where {iip}
    DynamicalBVPFunction{iip, FullSpecialize, _unwrap_val(twopoint)}(f, bc; kwargs...)
end
DynamicalBVPFunction{iip}(f::DynamicalBVPFunction, bc; kwargs...) where {iip} = f
function DynamicalBVPFunction(f, bc; twopoint::Union{Val, Bool} = Val(false), kwargs...)
    DynamicalBVPFunction{isinplace(f, 5), FullSpecialize, _unwrap_val(twopoint)}(
        f, bc; kwargs...)
end
DynamicalBVPFunction(f::DynamicalBVPFunction; kwargs...) = f

function IntegralFunction{iip, specialize}(f, integrand_prototype) where {iip, specialize}
    _f = prepare_function(f)
    IntegralFunction{iip, specialize, typeof(_f), typeof(integrand_prototype)}(_f,
        integrand_prototype)
end

function IntegralFunction{iip}(f, integrand_prototype) where {iip}
    IntegralFunction{iip, FullSpecialize}(f, integrand_prototype)
end
function IntegralFunction(f)
    calculated_iip = isinplace(f, 3, "integral", true)
    if calculated_iip
        throw(IntegrandMismatchFunctionError(calculated_iip, false))
    end
    IntegralFunction{false}(f, nothing)
end
function IntegralFunction(f, integrand_prototype)
    calculated_iip = isinplace(f, 3, "integral", true)
    IntegralFunction{calculated_iip}(f, integrand_prototype)
end

function BatchIntegralFunction{iip, specialize}(f, integrand_prototype;
        max_batch::Integer = typemax(Int)) where {iip, specialize}
    _f = prepare_function(f)
    BatchIntegralFunction{
        iip,
        specialize,
        typeof(_f),
        typeof(integrand_prototype)
    }(_f,
        integrand_prototype,
        max_batch)
end

function BatchIntegralFunction{iip}(f,
        integrand_prototype;
        kwargs...) where {iip}
    return BatchIntegralFunction{iip, FullSpecialize}(f,
        integrand_prototype;
        kwargs...)
end

function BatchIntegralFunction(f; kwargs...)
    calculated_iip = isinplace(f, 3, "batchintegral", true)
    if calculated_iip
        throw(IntegrandMismatchFunctionError(calculated_iip, false))
    end
    BatchIntegralFunction{false}(f, nothing; kwargs...)
end
function BatchIntegralFunction(f, integrand_prototype; kwargs...)
    calculated_iip = isinplace(f, 3, "batchintegral", true)
    BatchIntegralFunction{calculated_iip}(f, integrand_prototype; kwargs...)
end

########## Utility functions

function sys_or_symbolcache(sys, syms, paramsyms, indepsym = nothing)
    if sys === nothing &&
       (syms !== nothing || paramsyms !== nothing || indepsym !== nothing)
        Base.depwarn(
            "The use of keyword arguments `syms`, `paramsyms` and `indepsym` for `SciMLFunction`s is deprecated. Pass `sys = SymbolCache(syms, paramsyms, indepsym)` instead.",
            :syms)
        sys = SymbolCache(syms, paramsyms, indepsym)
    end
    return sys
end

########## Existence Functions

# Check that field/property exists (may be nothing)
__has_jac(f) = isdefined(f, :jac)
__has_jvp(f) = isdefined(f, :jvp)
__has_vjp(f) = isdefined(f, :vjp)
__has_tgrad(f) = isdefined(f, :tgrad)
__has_Wfact(f) = isdefined(f, :Wfact)
__has_Wfact_t(f) = isdefined(f, :Wfact_t)
__has_W_prototype(f) = isdefined(f, :W_prototype)
__has_paramjac(f) = isdefined(f, :paramjac)
__has_jac_prototype(f) = isdefined(f, :jac_prototype)
__has_sparsity(f) = isdefined(f, :sparsity)
__has_mass_matrix(f) = isdefined(f, :mass_matrix)
__has_syms(f) = isdefined(f, :syms)
__has_indepsym(f) = isdefined(f, :indepsym)
__has_paramsyms(f) = isdefined(f, :paramsyms)
__has_observed(f) = isdefined(f, :observed)
__has_analytic(f) = isdefined(f, :analytic)
__has_colorvec(f) = isdefined(f, :colorvec)
__has_sys(f) = isdefined(f, :sys)
__has_analytic_full(f) = isdefined(f, :analytic_full)
__has_resid_prototype(f) = isdefined(f, :resid_prototype)
__has_initializeprob(f) = isdefined(f, :initializeprob)
__has_update_initializeprob!(f) = isdefined(f, :update_initializeprob!)
__has_initializeprobmap(f) = isdefined(f, :initializeprobmap)
__has_initializeprobpmap(f) = isdefined(f, :initializeprobpmap)

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
has_sys(f::AbstractSciMLFunction) = __has_sys(f) && f.sys !== nothing
function has_initializeprob(f::AbstractSciMLFunction)
    __has_initializeprob(f) && f.initializeprob !== nothing
end
function has_update_initializeprob!(f::AbstractSciMLFunction)
    __has_update_initializeprob!(f) && f.update_initializeprob! !== nothing
end
function has_initializeprobmap(f::AbstractSciMLFunction)
    __has_initializeprobmap(f) && f.initializeprobmap !== nothing
end
function has_initializeprobpmap(f::AbstractSciMLFunction)
    __has_initializeprobpmap(f) && f.initializeprobpmap !== nothing
end

function has_syms(f::AbstractSciMLFunction)
    if __has_syms(f)
        f.syms !== nothing
    else
        !isempty(variable_symbols(f))
    end
end
function has_indepsym(f::AbstractSciMLFunction)
    if __has_indepsym(f)
        f.indepsym !== nothing
    else
        !isempty(independent_variable_symbols(f))
    end
end
function has_paramsyms(f::AbstractSciMLFunction)
    if __has_paramsyms(f)
        f.paramsyms !== nothing
    else
        !isempty(parameter_symbols(f))
    end
end
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

has_jac(f::DynamicalBVPFunction) = has_jac(f.f)
has_jvp(f::DynamicalBVPFunction) = has_jvp(f.f)
has_vjp(f::DynamicalBVPFunction) = has_vjp(f.f)
has_tgrad(f::DynamicalBVPFunction) = has_tgrad(f.f)
has_Wfact(f::DynamicalBVPFunction) = has_Wfact(f.f)
has_Wfact_t(f::DynamicalBVPFunction) = has_Wfact_t(f.f)
has_paramjac(f::DynamicalBVPFunction) = has_paramjac(f.f)
has_colorvec(f::DynamicalBVPFunction) = has_colorvec(f.f)

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

islinear(::AbstractDiffEqFunction) = false
islinear(f::ODEFunction) = islinear(f.f)
islinear(f::SplitFunction) = islinear(f.f1)

struct IncrementingODEFunction{iip, specialize, F} <: AbstractODEFunction{iip}
    f::F
end

function IncrementingODEFunction{iip, specialize}(f) where {iip, specialize}
    _f = prepare_function(f)
    IncrementingODEFunction{iip, specialize, typeof(_f)}(_f)
end

function IncrementingODEFunction{iip}(f) where {iip}
    IncrementingODEFunction{iip, FullSpecialize}(f)
end
function IncrementingODEFunction(f)
    IncrementingODEFunction{isinplace(f, 7), FullSpecialize}(f)
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
          :IntervalNonlinearFunction
          :IncrementingODEFunction
          :BVPFunction
          :DynamicalBVPFunction
          :IntegralFunction
          :BatchIntegralFunction]
    @eval begin
        function ConstructionBase.constructorof(::Type{<:$S{iip}}) where {
                iip,
        }
            (args...) -> $S{iip, FullSpecialize, map(typeof, args)...}(args...)
        end
    end
end

function SymbolicIndexingInterface.symbolic_container(fn::AbstractSciMLFunction)
    has_sys(fn) ? fn.sys : SymbolCache()
end

function SymbolicIndexingInterface.is_observed(fn::AbstractSciMLFunction, sym)
    has_sys(fn) ? is_observed(fn.sys, sym) : has_observed(fn)
end

function SymbolicIndexingInterface.observed(fn::AbstractSciMLFunction, sym)
    if has_observed(fn) && fn.observed !== DEFAULT_OBSERVED &&
       fn.observed !== DEFAULT_OBSERVED_NO_TIME
        if hasmethod(fn.observed, Tuple{Any})
            return fn.observed(sym)
        else
            return (args...) -> fn.observed(sym, args...)
        end
    end
    if has_sys(fn) &&
       hasmethod(SymbolicIndexingInterface.observed, Tuple{typeof(fn.sys), typeof(sym)})
        return SymbolicIndexingInterface.observed(fn.sys, sym)
    end
    error("SciMLFunction does not have observed")
end

function SymbolicIndexingInterface.observed(fn::AbstractSciMLFunction, sym::Symbol)
    return SymbolicIndexingInterface.observed(fn, getproperty(fn.sys, sym))
end

SymbolicIndexingInterface.constant_structure(::AbstractSciMLFunction) = true
