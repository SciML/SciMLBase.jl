const allowedkeywords = (
    :dense,
    :saveat,
    :save_idxs,
    :save_discretes,
    :tstops,
    :tspan,
    :d_discontinuities,
    :save_everystep,
    :save_on,
    :save_start,
    :save_end,
    :initialize_save,
    :adaptive,
    :abstol,
    :reltol,
    :dt,
    :dtmax,
    :dtmin,
    :force_dtmin,
    :internalnorm,
    :controller,
    :gamma,
    :beta1,
    :beta2,
    :qmax,
    :qmin,
    :qsteady_min,
    :qsteady_max,
    :qoldinit,
    :failfactor,
    :calck,
    :alias_u0,
    :maxiters,
    :maxtime,
    :callback,
    :isoutofdomain,
    :unstable_check,
    :verbose,
    :merge_callbacks,
    :progress,
    :progress_steps,
    :progress_name,
    :progress_message,
    :progress_id,
    :timeseries_errors,
    :dense_errors,
    :weak_timeseries_errors,
    :weak_dense_errors,
    :wrap,
    :calculate_error,
    :initializealg,
    :alg,
    :save_noise,
    :delta,
    :seed,
    :alg_hints,
    :kwargshandle,
    :trajectories,
    :batch_size,
    :sensealg,
    :advance_to_tstop,
    :stop_at_next_tstop,
    :u0,
    :p,
    # These two are from the default algorithm handling
    :default_set,
    :second_time,
    # This is for DiffEqDevTools
    :prob_choice,
    # Jump problems
    :alias_jump,
    # This is for copying/deepcopying noise in StochasticDiffEq
    :alias_noise,
    # This is for SimpleNonlinearSolve handling for batched Nonlinear Solves
    :batch,
    # Shooting method in BVP needs to differentiate between these two categories
    :nlsolve_kwargs,
    :odesolve_kwargs,
    # If Solvers which internally use linsolve
    :linsolve_kwargs,
    # Solvers internally using EnsembleProblem
    :ensemblealg,
    # Fine Grained Control of Tracing (Storing and Logging) during Solve
    :show_trace,
    :trace_level,
    :store_trace,
    # Termination condition for solvers
    :termination_condition,
    # For AbstractAliasSpecifier
    :alias,
    # Parameter estimation with BVP
    :tune_parameters,
)


const KWARGWARN_MESSAGE = """
Unrecognized keyword arguments found.
The only allowed keyword arguments to `solve` are:
$allowedkeywords

See <https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts> for more details.

Set kwargshandle=KeywordArgError for an error message.
Set kwargshandle=KeywordArgSilent to ignore this message.
"""

const KWARGERROR_MESSAGE = """
Unrecognized keyword arguments found.
The only allowed keyword arguments to `solve` are:
$allowedkeywords

See <https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts> for more details.
"""

struct CommonKwargError <: Exception
    kwargs::Any
end

function Base.showerror(io::IO, e::CommonKwargError)
    println(io, KWARGERROR_MESSAGE)
    notin = collect(map(x -> x ∉ allowedkeywords, keys(e.kwargs)))
    unrecognized = collect(keys(e.kwargs))[notin]
    print(io, "Unrecognized keyword arguments: ")
    return printstyled(io, unrecognized; bold = true, color = :red)
end

@enum KeywordArgError KeywordArgWarn KeywordArgSilent

const INCOMPATIBLE_U0_MESSAGE = """
Initial condition incompatible with functional form.
Detected an in-place function with an initial condition of type Number or SArray.
This is incompatible because Numbers cannot be mutated, i.e.
`x = 2.0; y = 2.0; x .= y` will error.

If using a immutable initial condition type, please use the out-of-place form.
I.e. define the function `du=f(u,p,t)` instead of attempting to "mutate" the immutable `du`.

If your differential equation function was defined with multiple dispatches and one is
in-place, then the automatic detection will choose in-place. In this case, override the
choice in the problem constructor, i.e. `ODEProblem{false}(f,u0,tspan,p,kwargs...)`.

For a longer discussion on mutability vs immutability and in-place vs out-of-place, see:
<https://docs.sciml.ai/DiffEqDocs/stable/tutorials/faster_ode_example#Example-Accelerating-a-Non-Stiff-Equation:-The-Lorenz-Equation>
"""

struct IncompatibleInitialConditionError <: Exception end

function Base.showerror(io::IO, e::IncompatibleInitialConditionError)
    return print(io, INCOMPATIBLE_U0_MESSAGE)
end

const NO_DEFAULT_ALGORITHM_MESSAGE = """
Default algorithm choices require DifferentialEquations.jl.
Please specify an algorithm (e.g., `solve(prob, Tsit5())` or
`init(prob, Tsit5())` for an ODE) or import DifferentialEquations
directly.

You can find the list of available solvers at <https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve>
and its associated pages.
"""

struct NoDefaultAlgorithmError <: Exception end

function Base.showerror(io::IO, e::NoDefaultAlgorithmError)
    return print(io, NO_DEFAULT_ALGORITHM_MESSAGE)
end

const NO_TSPAN_MESSAGE = """
No tspan is set in the problem or chosen in the init/solve call
"""

struct NoTspanError <: Exception end

function Base.showerror(io::IO, e::NoTspanError)
    return print(io, NO_TSPAN_MESSAGE)
end

const NAN_TSPAN_MESSAGE = """
NaN tspan is set in the problem or chosen in the init/solve call.
Note that -Inf and Inf values are allowed in the timespan for solves
which are terminated via callbacks, however NaN values are not allowed
since the direction of time is undetermined.
"""

struct NaNTspanError <: Exception end

function Base.showerror(io::IO, e::NaNTspanError)
    return print(io, NAN_TSPAN_MESSAGE)
end

const NON_SOLVER_MESSAGE = """
The arguments to solve are incorrect.
The second argument must be a solver choice, `solve(prob,alg)`
where `alg` is a `<: AbstractDEAlgorithm`, e.g. `Tsit5()`.

Please double check the arguments being sent to the solver.

You can find the list of available solvers at <https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve>
and its associated pages.
"""

struct NonSolverError <: Exception end

function Base.showerror(io::IO, e::NonSolverError)
    return print(io, NON_SOLVER_MESSAGE)
end

const NOISE_SIZE_MESSAGE = """
Noise sizes are incompatible. The expected number of noise terms in the defined
`noise_rate_prototype` does not match the number of noise terms in the defined
`AbstractNoiseProcess`. Please ensure that
`size(prob.noise_rate_prototype,2) == length(prob.noise.W[1])`.

Note: Noise process definitions require that users specify `u0`, and this value is
directly used in the definition. For example, if `noise = WienerProcess(0.0,0.0)`,
then the noise process is a scalar with `u0=0.0`. If `noise = WienerProcess(0.0,[0.0])`,
then the noise process is a vector with `u0=0.0`. If `noise_rate_prototype = zeros(2,4)`,
then the noise process must be a 4-dimensional process, for example
`noise = WienerProcess(0.0,zeros(4))`. This error is a sign that the user definition
of `noise_rate_prototype` and `noise` are not aligned in this manner and the definitions should
be double checked.
"""

struct NoiseSizeIncompatabilityError <: Exception
    prototypesize::Int
    noisesize::Int
end

function Base.showerror(io::IO, e::NoiseSizeIncompatabilityError)
    println(io, NOISE_SIZE_MESSAGE)
    println(io, "size(prob.noise_rate_prototype,2) = $(e.prototypesize)")
    return println(io, "length(prob.noise.W[1]) = $(e.noisesize)")
end

const PROBSOLVER_PAIRING_MESSAGE = """
Incompatible problem+solver pairing.
For example, this can occur if an ODE solver is passed with an SDEProblem.
Solvers are only capable of handling specific problem types. Please double
check that the chosen pairing is capable for handling the given problems.
"""

struct ProblemSolverPairingError <: Exception
    prob::Any
    alg::Any
end

function Base.showerror(io::IO, e::ProblemSolverPairingError)
    println(io, PROBSOLVER_PAIRING_MESSAGE)
    println(io, "Problem type: $(SciMLBase.__parameterless_type(typeof(e.prob)))")
    println(io, "Solver type: $(SciMLBase.__parameterless_type(typeof(e.alg)))")
    return println(
        io,
        "Problem types compatible with the chosen solver: $(compatible_problem_types(e.prob, e.alg))"
    )
end

function compatible_problem_types(prob, alg)
    return if alg isa AbstractODEAlgorithm
        ODEProblem
    elseif alg isa AbstractSDEAlgorithm
        (SDEProblem, SDDEProblem)
    elseif alg isa AbstractDDEAlgorithm # StochasticDelayDiffEq.jl just uses the SDE alg
        DDEProblem
    elseif alg isa AbstractDAEAlgorithm
        DAEProblem
    elseif alg isa AbstractSteadyStateAlgorithm
        SteadyStateProblem
    end
end

const DIRECT_AUTODIFF_INCOMPATABILITY_MESSAGE = """
Incompatible solver + automatic differentiation pairing.
The chosen automatic differentiation algorithm requires the ability
for compiler transforms on the code which is only possible on pure-Julia
solvers such as those from OrdinaryDiffEq.jl. Direct differentiation methods
which require this ability include:

- Direct use of ForwardDiff.jl on the solver
- `ForwardDiffSensitivity`, `ReverseDiffAdjoint`, `TrackerAdjoint`, and `ZygoteAdjoint`
  sensealg choices for adjoint differentiation.

Either switch the choice of solver to a pure Julia method, or change the automatic
differentiation method to one that does not require such transformations.

For more details on automatic differentiation, adjoint, and sensitivity analysis
of differential equations, see the documentation page:

<https://docs.sciml.ai/SciMLSensitivity/>
"""

struct DirectAutodiffError <: Exception end

function Base.showerror(io::IO, e::DirectAutodiffError)
    return println(io, DIRECT_AUTODIFF_INCOMPATABILITY_MESSAGE)
end

const NONNUMBER_ELTYPE_MESSAGE = """
Non-Number element type inside of an `Array` detected.
Arrays with non-number element types, such as
`Array{Array{Float64}}`, are not supported by the
solvers.

If you are trying to use an array of arrays structure,
look at the tools in RecursiveArrayTools.jl. For example:

If this was a mistake, promote the element types to be
all the same. If this was intentional, for example,
using Unitful.jl with different unit values, then use
an array type which has fast broadcast support for
heterogeneous values such as the ArrayPartition
from RecursiveArrayTools.jl. For example:

```julia
using RecursiveArrayTools
u0 = ArrayPartition([1.0,2.0],[3.0,4.0])
u0 = VectorOfArray([1.0,2.0],[3.0,4.0])
```

are both initial conditions which would be compatible with
the solvers. Or use ComponentArrays.jl for more complex
nested structures.

Element type:
"""

struct NonNumberEltypeError <: Exception
    eltype::Any
end

function Base.showerror(io::IO, e::NonNumberEltypeError)
    print(io, NONNUMBER_ELTYPE_MESSAGE)
    return print(io, e.eltype)
end

const GENERIC_NUMBER_TYPE_ERROR_MESSAGE = """
Non-standard number type (i.e. not Float32, Float64,
ComplexF32, or ComplexF64) detected as the element type
for the initial condition or time span. These generic
number types are only compatible with the pure Julia
solvers which support generic programming, such as
OrdinaryDiffEq.jl. The chosen solver does not support
this functionality. Please double check that the initial
condition and time span types are correct, and check that
the chosen solver was correct.
"""

struct GenericNumberTypeError <: Exception
    alg::Any
    uType::Any
    tType::Any
end

function Base.showerror(io::IO, e::GenericNumberTypeError)
    println(io, GENERIC_NUMBER_TYPE_ERROR_MESSAGE)
    println(io, "Solver: $(e.alg)")
    println(io, "u0 type: $(e.uType)")
    return print(io, "Timespan type: $(e.tType)")
end

const COMPLEX_SUPPORT_ERROR_MESSAGE = """
Complex number type (i.e. ComplexF32, or ComplexF64)
detected as the element type for the initial condition
with an algorithm that does not support complex numbers.
Please check that the initial condition type is correct.
If complex number support is needed, try different solvers
such as those from OrdinaryDiffEq.jl.
"""

struct ComplexSupportError <: Exception
    alg::Any
end

function Base.showerror(io::IO, e::ComplexSupportError)
    println(io, COMPLEX_SUPPORT_ERROR_MESSAGE)
    return println(io, "Solver: $(e.alg)")
end

const COMPLEX_TSPAN_ERROR_MESSAGE = """
Complex number type (i.e. ComplexF32, or ComplexF64)
detected as the element type for the independent variable
(i.e. time span). Please check that the tspan type is correct.
No solvers support complex time spans. If this is required,
please open an issue.
"""

struct ComplexTspanError <: Exception end

function Base.showerror(io::IO, e::ComplexTspanError)
    return println(io, COMPLEX_TSPAN_ERROR_MESSAGE)
end

const TUPLE_STATE_ERROR_MESSAGE = """
Tuple type used as a state. Since a tuple does not have vector
properties, it will not work as a state type in equation solvers.
Instead, change your equation from using tuple constructors `()`
to static array constructors `SA[]`. For example, change:

```julia
function ftup((a,b),p,t)
  return b,-a
end
u0 = (1.0,2.0)
tspan = (0.0,1.0)
ODEProblem(ftup,u0,tspan)
```

to:

```julia
using StaticArrays
function fsa(u,p,t)
    SA[u[2],u[1]]
end
u0 = SA[1.0,2.0]
tspan = (0.0,1.0)
ODEProblem(ftup,u0,tspan)
```

This will be safer and fast for small ODEs. For more information, see:
<https://docs.sciml.ai/DiffEqDocs/stable/tutorials/faster_ode_example/#Further-Optimizations-of-Small-Non-Stiff-ODEs-with-StaticArrays>
"""

struct TupleStateError <: Exception end

function Base.showerror(io::IO, e::TupleStateError)
    return println(io, TUPLE_STATE_ERROR_MESSAGE)
end

const MASS_MATRIX_ERROR_MESSAGE = """
Mass matrix size is incompatible with initial condition
sizing. The mass matrix must represent the `vec`
form of the initial condition `u0`, i.e.
`size(mm,1) == size(mm,2) == length(u)`
"""

struct IncompatibleMassMatrixError <: Exception
    sz::Int
    len::Int
end

function Base.showerror(io::IO, e::IncompatibleMassMatrixError)
    println(io, MASS_MATRIX_ERROR_MESSAGE)
    print(io, "size(prob.f.mass_matrix,1): ")
    println(io, e.sz)
    print(io, "length(u0): ")
    return println(e.len)
end

const LATE_BINDING_TSTOPS_ERROR_MESSAGE = """
This solver does not support providing `tstops` as a function.
Consider using a different solver or providing `tstops` as an array
of times.
"""

struct LateBindingTstopsNotSupportedError <: Exception end

function Base.showerror(io::IO, e::LateBindingTstopsNotSupportedError)
    return println(io, LATE_BINDING_TSTOPS_ERROR_MESSAGE)
end
