using SciMLBase, Test

module ExternalSolverUtilities
    using SciMLBase: @def, _unwrap_val

    @def affine_preamble begin
        shifted = x + offset
    end

    function evaluate(x, offset)
        @affine_preamble
        return shifted
    end

    unwrap(x) = _unwrap_val(x)
end

struct ProblemTypeTestProblem <: SciMLBase.AbstractSciMLProblem end
struct ProblemTypeTestMarker end
struct ProblemTypeTestSolution
    prob::ProblemTypeTestProblem
end

SciMLBase.problem_type(::ProblemTypeTestProblem) = ProblemTypeTestMarker()
SciMLBase.wrap_sol(::ProblemTypeTestSolution, ::ProblemTypeTestMarker) = :wrapped

struct ConcretizationHookProblem{F, U, P} <: SciMLBase.AbstractSciMLProblem
    f::F
    u0::U
    p::P
end
struct ConcretizationHookAlgorithm end

SciMLBase.isinplace(::ConcretizationHookProblem) = false
SciMLBase.get_concrete_problem(
    prob::ConcretizationHookProblem, isadapt; alg = nothing, kwargs...
) = (; prob, isadapt, alg, kwargs)
SciMLBase.check_prob_alg_pairing(::ConcretizationHookProblem, ::ConcretizationHookAlgorithm) =
    nothing

struct GenericSciMLFunction{iip} <: SciMLBase.AbstractSciMLFunction{iip} end

struct GenericGlobalErrorAlgorithm <: SciMLBase.AbstractDEAlgorithm end
struct GenericGlobalErrorReportingAlgorithm <: SciMLBase.AbstractDEAlgorithm end

SciMLBase.has_global_error(::GenericGlobalErrorReportingAlgorithm) = true

struct GenericProblem <: SciMLBase.AbstractNonlinearProblem{Vector{Float64}, false}
    u0::Vector{Float64}
    p::NamedTuple
end

struct GenericFunction <: SciMLBase.AbstractSciMLFunction{false}
    f::Function
end

(f::GenericFunction)(u, p, t) = f.f(u, p, t)

struct GenericAlgorithm <: SciMLBase.AbstractODEAlgorithm end

SciMLBase.isadaptive(::GenericAlgorithm) = false
SciMLBase.isdiscrete(::GenericAlgorithm) = false
SciMLBase.allowscomplex(::GenericAlgorithm) = true
SciMLBase.alg_order(::GenericAlgorithm) = 2

struct GenericSolution <: SciMLBase.AbstractNonlinearSolution{Float64, 1}
    u::Vector{Float64}
    retcode::SciMLBase.ReturnCode.T
end

struct ConcreteSolveContractProblem end
struct ConcreteSolveContractAlgorithm end
struct ConcreteSolveContractSenseAlg end
struct ConcreteSolveContractOriginator <: SciMLBase.ADOriginator end

function SciMLBase._concrete_solve_adjoint(
        ::ConcreteSolveContractProblem, ::ConcreteSolveContractAlgorithm,
        ::ConcreteSolveContractSenseAlg, u0, p, ::ConcreteSolveContractOriginator,
        args...; kwargs...
    )
    return (; u0, p, args, kwargs = (; kwargs...)), Δ -> (:adjoint, Δ)
end

function SciMLBase._concrete_solve_forward(
        ::ConcreteSolveContractProblem, ::ConcreteSolveContractAlgorithm,
        ::ConcreteSolveContractSenseAlg, u0, p, ::ConcreteSolveContractOriginator,
        args...; kwargs...
    )
    return (; u0, p, args, kwargs = (; kwargs...)), Δ -> (:forward, Δ)
end

struct SymbolicRemakeContractRoot end
struct SymbolicRemakeContractProblem
    u0
    p
end
struct PreparedStateContract{T}
    value::T
end
struct PreparedFunctionContract{F}
    f::F
end
struct SolverOptionsContract <: SciMLBase.AbstractDEOptions end

function SciMLBase.remake_initialization_data(
        ::SymbolicRemakeContractRoot, scimlfn, u0, t0, p, newu0, newp,
        ::SciMLBase.RemakeInitializationDataContext
    )
    return (; scimlfn, u0, t0, p, newu0, newp)
end

function SciMLBase.get_new_A_b(
        ::SymbolicRemakeContractRoot, f, p, A, b; scale = one(p), kwargs...
    )
    return scale .* A, scale .* b
end

function SciMLBase.detect_cycles(
        ::SymbolicRemakeContractRoot, varmap, syms
    )
    return any(sym -> get(varmap, sym, nothing) === sym, syms)
end

function SciMLBase.get_updated_symbolic_problem(
        ::SymbolicRemakeContractRoot, prob::SymbolicRemakeContractProblem;
        u0 = prob.u0, p = prob.p, kwargs...
    )
    return SymbolicRemakeContractProblem(u0, p)
end

SciMLBase.prepare_initial_state(state::PreparedStateContract) = state.value
SciMLBase.prepare_function(f::PreparedFunctionContract) = f.f

@testset "Common keyword interface documentation" begin
    common_keywords = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Common_Keywords.md"),
        String
    )
    algorithms = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Algorithms.md"),
        String
    )

    for keyword in (
            ":auto", ":nonstiff", ":stiff", ":additive", ":commutative",
            ":stratonovich",
        )
        @test occursin("`$keyword`", common_keywords)
    end
    @test occursin("save_everystep && isempty(saveat)", common_keywords)
    @test !occursin("save_everystep && !isempty(saveat)", common_keywords)
    @test occursin("1_000_000", common_keywords)
    @test occursin("ProgressLogging.jl", common_keywords)
    @test occursin("common keyword interface", algorithms)
    @test !occursin("Commonly used algorithm keyword arguments are:\n\n", algorithms)
end

@testset "PDE interface documentation" begin
    pde_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "PDE.md"), String
    )
    normalized_pde_docs = join(split(pde_docs), " ")

    @test !occursin("WIP", pde_docs)
    @test occursin("does not require subtyping", normalized_pde_docs)
    @test occursin("SciMLBase.discretize(sys, discretizer", pde_docs)
    @test occursin("SciMLBase.symbolic_discretize(sys, discretizer", pde_docs)
    @test occursin("AbstractDiscretizationMetadata{Val(true)}", pde_docs)
    @test occursin("AbstractDiscretizationMetadata{Val(false)}", pde_docs)
    @test occursin("`NonlinearProblem`", pde_docs)
    @test occursin("`OptimizationProblem`", pde_docs)
end

@testset "Array and number interface documentation" begin
    array_number_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Array_and_Number.md"),
        String
    )

    @test !occursin("interface_checks", array_number_docs)
    @test !occursin("as of 2023", array_number_docs)
    @test occursin("problem-algorithm pair", array_number_docs)
    @test occursin("allows_arbitrary_number_types", array_number_docs)
    @test occursin("allowscomplex", array_number_docs)
    @test occursin("ArrayInterface.zeromatrix", array_number_docs)
    @test occursin("SciMLOperators.jl", array_number_docs)
    @test occursin("LinearProblem(A, b)", array_number_docs)
end

@testset "Problem and function interface documentation" begin
    problem_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Problems.md"), String
    )
    function_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "SciMLFunctions.md"),
        String
    )

    @test occursin("`use_defaults = true`", problem_docs)
    @test !occursin("`default_values = true`", problem_docs)
    @test occursin("`2.0`, `0.1`", problem_docs)
    @test !occursin("### `problem_type`", problem_docs)
    @test occursin("Differential Equation Problem Types", problem_docs)
    @test !occursin("SciMLBase.ImmutableODEProblem", problem_docs)
    @test !occursin("will always make a deep copy", function_docs)
    @test occursin("selected differentiation and linear solver", function_docs)
    @test occursin("`update_coefficients` for the out-of-place form", function_docs)
    @test occursin("ODE and Discrete Function Types", function_docs)
    @test !occursin("SciMLBase.ODEFunction\n", function_docs)
end

@testset "AbstractSciMLFunction in-place trait contract" begin
    @test SciMLBase.isinplace(GenericSciMLFunction{true}()) === true
    @test SciMLBase.isinplace(GenericSciMLFunction{false}()) === false
    @test !occursin("No documentation found", sprint(show, @doc SciMLBase.isinplace))
end

@testset "AbstractSciMLAlgorithm global-error trait contract" begin
    @test !SciMLBase.has_global_error(GenericGlobalErrorAlgorithm())
    @test SciMLBase.has_global_error(GenericGlobalErrorReportingAlgorithm())
    @test (@doc SciMLBase.has_global_error) !== nothing
end

@testset "Generic abstract interface contracts" begin
    problem = GenericProblem([1.0], (; rate = 2.0))
    @test SciMLBase.isinplace(problem) === false
    @test SciMLBase.problem_type(problem) === nothing
    @test SciMLBase.is_diagonal_noise(problem) === false

    f = GenericFunction((u, p, t) -> p.rate .* u .+ t)
    @test SciMLBase.isinplace(f) === false
    @test f([1.0], (; rate = 2.0), 0.5) == [2.5]
    @test !SciMLBase.has_analytic(f)
    @test !SciMLBase.has_jac(f)
    @test !SciMLBase.has_jvp(f)
    @test !SciMLBase.has_vjp(f)
    @test !SciMLBase.has_paramjac(f)
    @test !SciMLBase.has_observed(f)

    alg = GenericAlgorithm()
    @test !SciMLBase.isadaptive(alg)
    @test !SciMLBase.isdiscrete(alg)
    @test SciMLBase.allowscomplex(alg)
    @test SciMLBase.alg_order(alg) == 2

    sol = GenericSolution([1.0, 2.0], SciMLBase.ReturnCode.Success)
    @test size(sol) == (2,)
    @test sol[2] == 2.0
    @test SciMLBase.successful_retcode(sol)
    @test SciMLBase.plottable_indices(sol.u) == 1:2
end

@testset "Problem layout marker interface" begin
    problem_trait_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Problem_Traits.md"),
        String
    )
    problem_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Problems.md"), String
    )

    ode_prob = ODEProblem((u, p, t) -> u, 1.0, (0.0, 1.0))
    linear_prob = LinearProblem(ones(1, 1), ones(1))
    @test SciMLBase.problem_type(ode_prob) isa SciMLBase.StandardODEProblem
    @test SciMLBase.problem_type(linear_prob) === nothing
    @test SciMLBase.problem_type(ProblemTypeTestProblem()) isa ProblemTypeTestMarker
    @test SciMLBase.wrap_sol(ProblemTypeTestSolution(ProblemTypeTestProblem())) === :wrapped
    @test occursin("SciMLBase.problem_type", problem_trait_docs)
    @test occursin("Delay, Boundary, and Noise Problem Types", problem_docs)
    @test occursin("Algebraic Problem Types", problem_docs)
end

@testset "Solver concretization developer interface" begin
    prob = ConcretizationHookProblem(nothing, [1.0], :default_parameter)
    alg = ConcretizationHookAlgorithm()

    @test SciMLBase.get_concrete_p(prob, (;)) === :default_parameter
    @test SciMLBase.get_concrete_p(prob, (; p = :override_parameter)) === :override_parameter
    @test SciMLBase.get_concrete_u0(prob, false, 0.0, (;)) === prob.u0
    @test SciMLBase.get_concrete_u0(prob, false, 0.0, (; u0 = [2.0])) == [2.0]
    @test SciMLBase.isconcreteu0(prob, 0.0, (;))
    @test SciMLBase.promote_u0([1.0], prob.p, 0.0) == [1.0]

    concrete = SciMLBase.get_concrete_problem(prob, true; alg, p = :override_parameter)
    @test concrete.prob === prob
    @test concrete.isadapt
    @test concrete.alg === alg
    @test concrete.kwargs[:p] === :override_parameter
    @test SciMLBase.check_prob_alg_pairing(prob, alg) === nothing

    SciMLBase.@add_kwonly function concretization_kwonly(x; offset = 1)
        return x + offset
    end
    @test concretization_kwonly(2) == 3
    @test concretization_kwonly(; x = 2, offset = 4) == 6
end

@testset "Concrete solve AD developer interface" begin
    prob = ConcreteSolveContractProblem()
    alg = ConcreteSolveContractAlgorithm()
    sensealg = ConcreteSolveContractSenseAlg()
    originator = ConcreteSolveContractOriginator()

    primal, pullback = SciMLBase._concrete_solve_adjoint(
        prob, alg, sensealg, :u0, :p, originator, :extra; saveat = :saved
    )
    @test primal == (; u0 = :u0, p = :p, args = (:extra,), kwargs = (; saveat = :saved))
    @test pullback(:cotangent) == (:adjoint, :cotangent)

    primal, pushforward = SciMLBase._concrete_solve_forward(
        prob, alg, sensealg, :u0, :p, originator, :extra; saveat = :saved
    )
    @test primal == (; u0 = :u0, p = :p, args = (:extra,), kwargs = (; saveat = :saved))
    @test pushforward(:tangent) == (:forward, :tangent)
end

@testset "Symbolic-system developer interface" begin
    root = SymbolicRemakeContractRoot()
    prob = SymbolicRemakeContractProblem([1.0], [2.0])

    @test SciMLBase.RemakeInitializationDataContext() isa
        SciMLBase.RemakeInitializationDataContext
    @test SciMLBase.LateBindingUpdateU0PContext() isa
        SciMLBase.LateBindingUpdateU0PContext

    initdata = SciMLBase.remake_initialization_data(
        root, :function, :old_u0, 0.0, :old_p, :new_u0, :new_p
    )
    @test initdata == (;
        scimlfn = :function, u0 = :old_u0, t0 = 0.0, p = :old_p,
        newu0 = :new_u0, newp = :new_p,
    )

    @test SciMLBase.get_new_A_b(nothing, nothing, 2.0, [1.0], [3.0]) ==
        ([1.0], [3.0])
    @test SciMLBase.get_new_A_b(root, nothing, 2.0, [1.0], [3.0]; scale = 2.0) ==
        ([2.0], [6.0])

    @test !SciMLBase.detect_cycles(nothing, Dict(:x => :x), [:x])
    @test SciMLBase.detect_cycles(root, Dict(:x => :x), [:x])
    @test !SciMLBase.detect_cycles(root, Dict(:x => 1), [:x])

    @test SciMLBase.get_updated_symbolic_problem(nothing, prob) === prob
    updated = SciMLBase.get_updated_symbolic_problem(root, prob; u0 = [3.0], p = [4.0])
    @test updated.u0 == [3.0]
    @test updated.p == [4.0]

    update_Ab = (A, b, p) -> (A .= p[1]; b .= p[2]; (A, b))
    symbolic_linear = SciMLBase.SymbolicLinearInterface(
        ; update_Ab, sys = root, observed = nothing, metadata = :metadata
    )
    A, b = symbolic_linear.update_Ab(zeros(1, 1), zeros(1), (2.0, 3.0))
    @test A == fill(2.0, 1, 1)
    @test b == [3.0]
    @test symbolic_linear.sys === root
    @test symbolic_linear.metadata === :metadata
end

@testset "Input-preparation developer interface" begin
    state = PreparedStateContract([1.0, 2.0])
    callable = PreparedFunctionContract(x -> x + 1)

    @test SciMLBase.prepare_initial_state(:unchanged) === :unchanged
    @test SciMLBase.prepare_initial_state(state) === state.value
    @test SciMLBase.prepare_function(identity) === identity
    @test SciMLBase.prepare_function(callable)(2) == 3
end

@testset "Solver support developer types" begin
    @test SolverOptionsContract() isa SciMLBase.AbstractDEOptions

    nlstep = SciMLBase.ODENLStepData(
        :nlprob, :u0perm, :set_gamma_c, :set_outer_tmp, :set_inner_tmp, :nlprobmap
    )
    @test nlstep.nlprob === :nlprob
    @test nlstep.u0perm === :u0perm
    @test nlstep.set_γ_c === :set_gamma_c
    @test nlstep.set_outer_tmp === :set_outer_tmp
    @test nlstep.set_inner_tmp === :set_inner_tmp
    @test nlstep.nlprobmap === :nlprobmap

    wrapper = SciMLBase.JacobianWrapper((u, p) -> u .- p, [1.0, 2.0])
    @test wrapper([3.0, 5.0]) == [2.0, 3.0]
    residual = zeros(2)
    @test wrapper(residual, [4.0, 7.0]) === residual
    @test residual == [3.0, 5.0]
end

@testset "Concrete interface reference documentation" begin
    interfaces_dir = joinpath(@__DIR__, "..", "docs", "src", "interfaces")
    bindings = String[]
    for path in filter(path -> endswith(path, ".md"), readdir(interfaces_dir; join = true))
        append!(bindings, strip.(readlines(path)))
    end

    concrete_bindings = (
        # Problems
        "SciMLBase.LinearProblem",
        "SciMLBase.EigenvalueProblem",
        "SciMLBase.EigenvalueTarget",
        "SciMLBase.EigenvalueTarget.LargestMagnitude",
        "SciMLBase.EigenvalueTarget.SmallestMagnitude",
        "SciMLBase.EigenvalueTarget.LargestRealPart",
        "SciMLBase.EigenvalueTarget.SmallestRealPart",
        "SciMLBase.EigenvalueTarget.LargestImaginaryPart",
        "SciMLBase.EigenvalueTarget.SmallestImaginaryPart",
        "SciMLBase.NonlinearProblem",
        "SciMLBase.StandardNonlinearProblem",
        "SciMLBase.IntervalNonlinearProblem",
        "SciMLBase.NonlinearLeastSquaresProblem",
        "SciMLBase.SCCNonlinearProblem",
        "SciMLBase.HomotopyProblem",
        "SciMLBase.IntegralProblem",
        "SciMLBase.SampledIntegralProblem",
        "SciMLBase.OptimizationProblem",
        "SciMLBase.SteadyStateProblem",
        "SciMLBase.AnalyticalProblem",
        "SciMLBase.ODEProblem",
        "SciMLBase.ImmutableODEProblem",
        "SciMLBase.StandardODEProblem",
        "SciMLBase.DynamicalODEProblem",
        "SciMLBase.SecondOrderODEProblem",
        "SciMLBase.AbstractSplitODEProblem",
        "SciMLBase.SplitODEProblem",
        "SciMLBase.IncrementingODEProblem",
        "SciMLBase.DiscreteProblem",
        "SciMLBase.ImplicitDiscreteProblem",
        "SciMLBase.RODEProblem",
        "SciMLBase.SDEProblem",
        "SciMLBase.SplitSDEProblem",
        "SciMLBase.DynamicalSDEProblem",
        "SciMLBase.DAEProblem",
        "SciMLBase.StandardDAEProblem",
        "SciMLBase.DDEProblem",
        "SciMLBase.StandardDDEProblem",
        "SciMLBase.AbstractDynamicalDDEProblem",
        "SciMLBase.DynamicalDDEProblem",
        "SciMLBase.SecondOrderDDEProblem",
        "SciMLBase.SDDEProblem",
        "SciMLBase.BVProblem",
        "SciMLBase.StandardBVProblem",
        "SciMLBase.TwoPointBVProblem",
        "SciMLBase.SecondOrderBVProblem",
        "SciMLBase.StandardSecondOrderBVProblem",
        "SciMLBase.TwoPointSecondOrderBVProblem",
        "SciMLBase.NoiseProblem",
        # Functions
        "SciMLBase.ODEFunction",
        "SciMLBase.DynamicalODEFunction",
        "SciMLBase.SplitFunction",
        "SciMLBase.IncrementingODEFunction",
        "SciMLBase.ODEInputFunction",
        "SciMLBase.DiscreteFunction",
        "SciMLBase.ImplicitDiscreteFunction",
        "SciMLBase.SDEFunction",
        "SciMLBase.SplitSDEFunction",
        "SciMLBase.DynamicalSDEFunction",
        "SciMLBase.RODEFunction",
        "SciMLBase.DDEFunction",
        "SciMLBase.DynamicalDDEFunction",
        "SciMLBase.SDDEFunction",
        "SciMLBase.DAEFunction",
        "SciMLBase.NonlinearFunction",
        "SciMLBase.HomotopyNonlinearFunction",
        "SciMLBase.IntervalNonlinearFunction",
        "SciMLBase.IntegralFunction",
        "SciMLBase.BatchIntegralFunction",
        "SciMLBase.OptimizationFunction",
        "SciMLBase.MultiObjectiveOptimizationFunction",
        "SciMLBase.BVPFunction",
        "SciMLBase.TwoPointBVPFunction",
        "SciMLBase.TwoPointDynamicalBVPFunction",
        "SciMLBase.DynamicalBVPFunction",
        # Solutions
        "SciMLBase.LinearSolution",
        "SciMLBase.EigenvalueSolution",
        "SciMLBase.NonlinearSolution",
        "SciMLBase.SteadyStateSolution",
        "SciMLBase.IntegralSolution",
        "SciMLBase.OptimizationSolution",
        "SciMLBase.ODESolution",
        "SciMLBase.RODESolution",
        "SciMLBase.DAESolution",
    )

    for binding in concrete_bindings
        @test count(==(binding), bindings) == 1
    end

    for binding in (
            "SciMLBase.Clocks",
            "SciMLBase.EnsembleAnalysis",
            "SciMLBase.NullParameters",
            "SciMLBase.check_keywords",
            "SciMLBase.warn_compat",
            "SciMLBase.u_modified!",
            "SciMLBase.NoRootFind",
            "SciMLBase.LeftRootFind",
            "SciMLBase.RightRootFind",
        )
        @test count(==(binding), bindings) == 1
    end
end

@testset "Solution interface documentation" begin
    solution_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Solutions.md"), String
    )

    @test occursin("a union of four array-like solution families", solution_docs)
    @test occursin("length(sol.t)", solution_docs)
    @test !occursin("length(sol))", solution_docs)
    @test occursin("100_000", solution_docs)
    @test occursin("`100` for a discrete problem", solution_docs)
    @test occursin("sol.tslocation != 0", solution_docs)
    @test occursin("1000 * sol.tslocation", solution_docs)
    @test occursin("`SensitivityInterpolation`", solution_docs)
end

@testset "Generic abstract interface documentation" begin
    problem_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Problems.md"), String
    )
    function_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "SciMLFunctions.md"),
        String,
    )
    algorithm_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Algorithms.md"), String
    )
    solution_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Solutions.md"), String
    )

    @test occursin("### Generic Usage Rules", problem_docs)
    @test occursin("problem_type(prob)", problem_docs)
    @test occursin("### Generic Usage Rules", function_docs)
    @test occursin("has_paramjac", function_docs)
    @test occursin("### Generic Usage Rules", algorithm_docs)
    @test occursin("capability traits", algorithm_docs)
    @test occursin("### Generic Usage Rules", solution_docs)
    @test occursin("successful_retcode(sol)", solution_docs)
end

@testset "Ensemble interface documentation" begin
    ensemble_docs = read(
        joinpath(@__DIR__, "..", "docs", "src", "interfaces", "Ensembles.md"), String
    )

    @test !occursin("AbstractEnsembleSimulation", ensemble_docs)
    @test !occursin("EnsembleSimulation", ensemble_docs)
    @test !occursin("`linspace`", ensemble_docs)
    @test occursin("rand(ctx.rng)", ensemble_docs)
    @test occursin("rand(ctx.rng, 2)", ensemble_docs)
    @test occursin("sqrt(var(u) / last(I))", ensemble_docs)
    @test occursin("julia --threads=auto", ensemble_docs)
    @test occursin("SciMLBase.EnsembleAnalysis.EnsembleSummary", ensemble_docs)
end

@testset "Downstream-rendered integrator docstrings" begin
    for doc in (
            (@doc SciMLBase.last_step_failed),
            (@doc SciMLBase.check_error),
            (@doc SciMLBase.check_error!),
        )
        text = sprint(show, doc)
        @test occursin(
            "https://docs.sciml.ai/SciMLBase/stable/interfaces/Init_Solve/", text
        )
        @test !occursin("](@ref)", text)
    end
end

@testset "Solver code-generation developer API" begin
    @test ExternalSolverUtilities.evaluate(2, 3) == 5
    @test ExternalSolverUtilities.unwrap(Val(:compile_time)) === :compile_time

    runtime_value = Ref(1)
    @test ExternalSolverUtilities.unwrap(runtime_value) === runtime_value
end

if isdefined(Base, :ispublic)
    @testset "Extension hooks public API" begin
        for name in (
                :parameterless_type, :updated_u0_p, :isdenseplot, :plottable_indices,
                :done, :postamble!, :enable_interpolation_sensitivitymode,
                :get_root_indp, :has_initializeprob, :late_binding_update_u0_p,
                :strip_interpolation, :unitfulvalue, :value, :last_step_failed,
                Symbol("@def"), :_unwrap_val,
                :get_concrete_p, :get_concrete_u0, :isconcreteu0, :promote_u0,
                :get_concrete_problem, :check_prob_alg_pairing, :KeywordArgError,
                :keyword_arg_silent, Symbol("@add_kwonly"),
                :is_overdetermined_initialization, :sensitivity_solution,
                :has_paramjac, :has_vjp_p, :has_observed, :ParamJacobianWrapper, :Void,
                :ADOriginator, :ChainRulesOriginator, :EnzymeOriginator,
                :ReverseDiffOriginator, :TrackerOriginator, :MooncakeOriginator,
                :set_mooncakeoriginator_if_mooncake,
                :_concrete_solve_adjoint, :_concrete_solve_forward,
                :RemakeInitializationDataContext, :remake_initialization_data,
                :LateBindingUpdateU0PContext, :detect_cycles,
                :get_updated_symbolic_problem, :SymbolicLinearInterface, :get_new_A_b,
                :widen_bounded_type_params, :prepare_initial_state, :prepare_function,
                :AbstractDEOptions, :ODENLStepData, :JacobianWrapper,
            )
            @test Base.ispublic(SciMLBase, name)
            @test Base.Docs.hasdoc(SciMLBase, name)
        end
    end

    @testset "Clocks manual public API" begin
        for name in (
                :AbstractClock,
                :ContinuousClock,
                :PeriodicClock,
                :SolverStepClock,
                :EventClock,
                :TimeDomain,
                :Continuous,
                :Clock,
                :isclock,
                :issolverstepclock,
                :iscontinuous,
                :iseventclock,
                :is_discrete_time_domain,
                :first_clock_tick_time,
                :IndexedClock,
                :canonicalize_indexed_clock,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "SciMLFunctions manual public API" begin
        for name in (
                :AbstractSciMLFunction,
                :AbstractDiffEqFunction,
                :AbstractODEFunction,
                :AbstractSDEFunction,
                :AbstractDDEFunction,
                :AbstractDAEFunction,
                :AbstractRODEFunction,
                :AbstractDiscreteFunction,
                :AbstractSDDEFunction,
                :AbstractNonlinearFunction,
                :AbstractIntervalNonlinearFunction,
                :AbstractIntegralFunction,
                :AbstractOptimizationFunction,
                :AbstractODEInputFunction,
                :AbstractBVPFunction,
                :AbstractParameterizedFunction,
                :AbstractHistoryFunction,
                :IncrementingODEFunction,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Solutions manual public API" begin
        for name in (
                :AbstractSciMLSolution,
                :AbstractNoTimeSolution,
                :AbstractTimeseriesSolution,
                :AbstractNoiseProcess,
                :AbstractEnsembleSolution,
                :AbstractLinearSolution,
                :AbstractEigenvalueSolution,
                :AbstractNonlinearSolution,
                :AbstractIntegralSolution,
                :AbstractOptimizationSolution,
                :AbstractSteadyStateSolution,
                :AbstractAnalyticalSolution,
                :AbstractODESolution,
                :AbstractDDESolution,
                :AbstractRODESolution,
                :AbstractDAESolution,
                :AbstractPDETimeSeriesSolution,
                :AbstractPDENoTimeSolution,
                :AbstractPDESolution,
                :AbstractSensitivitySolution,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Problems manual public API" begin
        for name in (
                :AbstractSpecialization,
                :specialization,
                :isfunctionwrapper,
                :wrapfun_oop,
                :wrapfun_iip,
                :unwrap_fw,
                :problem_type,
                :AbstractLinearProblem,
                :AbstractEigenvalueProblem,
                :AbstractIntervalNonlinearProblem,
                :AbstractIntegralProblem,
                :AbstractOptimizationProblem,
                :AbstractDynamicOptProblem,
                :AbstractAnalyticalProblem,
                :AbstractConstantLagDDEProblem,
                :AbstractSecondOrderODEProblem,
                :AbstractConstantLagSDDEProblem,
                :AbstractPDEProblem,
                :AbstractOptimizationCache,
                :IncrementingODEProblem,
                :StandardDDEProblem,
                :StandardNonlinearProblem,
                :StandardBVProblem,
                :StandardSecondOrderBVProblem,
                :AbstractDynamicalODEProblem,
                :AbstractSplitODEProblem,
                :AbstractDynamicalDDEProblem,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Algorithms manual public API" begin
        for name in (
                :isautodifferentiable,
                :allows_arbitrary_number_types,
                :allowscomplex,
                :isadaptive,
                :isdiscrete,
                :forwarddiffs_model,
                :forwarddiffs_model_time,
                :forwarddiff_chunksize,
                :has_lazy_interpolation,
                :has_global_error,
                :allows_late_binding_tstops,
                :supports_opt_cache_interface,
                :has_init,
                :has_step,
                :alg_order,
                :allowsbounds,
                :requiresbounds,
                :allowsconstraints,
                :requiresconstraints,
                :requiresgradient,
                :allowsfg,
                :requireshessian,
                :allowsfgh,
                :requiresconsjac,
                :allowsconsjvp,
                :allowsconsvjp,
                :requiresconshess,
                :requireslagh,
                :allowscallback,
                :allows_non_wiener_noise,
                :requires_additive_noise,
                :AlgorithmInterpretation,
                :alg_interpretation,
                :AbstractSciMLAlgorithm,
                :AbstractDEAlgorithm,
                :AbstractLinearAlgorithm,
                :AbstractNonlinearAlgorithm,
                :AbstractIntervalNonlinearAlgorithm,
                :AbstractIntegralAlgorithm,
                :AbstractOptimizationAlgorithm,
                :AbstractSteadyStateAlgorithm,
                :AbstractBVPAlgorithm,
                :AbstractODEAlgorithm,
                :AbstractSecondOrderODEAlgorithm,
                :AbstractRODEAlgorithm,
                :AbstractSDEAlgorithm,
                :AbstractDAEAlgorithm,
                :AbstractDDEAlgorithm,
                :AbstractSDDEAlgorithm,
                :EnsembleAlgorithm,
                :DAEInitializationAlgorithm,
                :AbstractDiscretization,
                :AbstractDiscretizationMetadata,
                :NoInit,
                :CheckInit,
                :OverrideInit,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Algorithm interpretation enum" begin
        @test instances(SciMLBase.AlgorithmInterpretation.T) ==
            (
            SciMLBase.AlgorithmInterpretation.Ito,
            SciMLBase.AlgorithmInterpretation.Stratonovich,
        )
        @test Int(SciMLBase.AlgorithmInterpretation.Ito) == 0
        @test Int(SciMLBase.AlgorithmInterpretation.Stratonovich) == 1
    end

    @testset "Init-solve manual public API" begin
        for name in (
                :DEIntegrator,
                :AbstractSteadyStateIntegrator,
                :AbstractODEIntegrator,
                :AbstractSecondOrderODEIntegrator,
                :AbstractSDEIntegrator,
                :AbstractRODEIntegrator,
                :AbstractDDEIntegrator,
                :AbstractDAEIntegrator,
                :AbstractSDDEIntegrator,
                :DECache,
                :addat!,
                :get_tmp_cache,
                :user_cache,
                :u_cache,
                :du_cache,
                :full_cache,
                :resize_non_user_cache!,
                :deleteat_non_user_cache!,
                :addat_non_user_cache!,
                :terminate!,
                :add_tstop!,
                :has_tstop,
                :first_tstop,
                :pop_tstop!,
                :add_saveat!,
                :get_du,
                :get_du!,
                :get_proposed_dt,
                :set_proposed_dt!,
                :derivative_discontinuity!,
                :savevalues!,
                :reinit!,
                :auto_dt_reset!,
                :change_t_via_interpolation!,
                :reeval_internals_due_to_modification!,
                :set_t!,
                :set_u!,
                :set_ut!,
                :get_sol,
                :check_error!,
                :initialize_dae!,
                :has_reinit,
                :OverrideInitData,
                :get_initial_values,
                :numargs,
                :FunctionArgumentsError,
                :TooFewArgumentsError,
                :TooManyArgumentsError,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Callbacks manual public API" begin
        for name in (
                :DECallback,
                :AbstractContinuousCallback,
                :AbstractDiscreteCallback,
                :ContinuousCallback,
                :VectorContinuousCallback,
                :DiscreteCallback,
                :CallbackSet,
                :RootfindOpt,
                :NoRootFind,
                :LeftRootFind,
                :RightRootFind,
                :split_callbacks,
                :save_final_discretes!,
                :save_discretes_if_enabled!,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Ensembles manual public API" begin
        for name in (
                :AbstractEnsembleProblem,
                :EnsembleProblem,
                :WeightedEnsembleProblem,
                :DEFAULT_PROB_FUNC,
                :DEFAULT_OUTPUT_FUNC,
                :DEFAULT_REDUCTION,
                :__solve,
                :EnsembleContext,
                :generate_sim_seeds,
                :default_rng_func,
                :BasicEnsembleAlgorithm,
                :EnsembleSerial,
                :EnsembleThreads,
                :EnsembleDistributed,
                :EnsembleSplitThreads,
                :AbstractEnsembleEstimator,
                :EnsembleSolution,
                :EnsembleTestSolution,
                :WeightedEnsembleSolution,
                :calculate_ensemble_errors,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Symbolic save_idxs manual public API" begin
        for name in (
                :get_saved_subsystem,
                :SavedSubsystem,
                :get_saved_state_idxs,
                :SavedSubsystemWithFallback,
                :get_save_idxs_and_saved_subsystem,
                :create_parameter_timeseries_collection,
                :get_saveable_values,
                :save_discretes!,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "PDE manual public API" begin
        for name in (
                :AbstractPDEProblem,
                :PDEProblem,
                :discretize,
                :symbolic_discretize,
                :PDETimeSeriesSolution,
                :PDENoTimeSolution,
                :wrap_sol,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end

    @testset "Public abstract interface documentation" begin
        interface_docs = join(
            read(joinpath(root, file), String)
                for (root, _, files) in walkdir(joinpath(@__DIR__, "..", "docs", "src", "interfaces"))
                for file in files if endswith(file, ".md")
        )
        interface_api_bindings = String[]
        in_docs_block = false
        for line in strip.(split(interface_docs, '\n'))
            if line == "```@docs"
                in_docs_block = true
            elseif in_docs_block && startswith(line, "```")
                in_docs_block = false
            elseif in_docs_block && !isempty(line)
                push!(interface_api_bindings, line)
            end
        end
        @test !in_docs_block
        abstract_public_names = filter(
            names(SciMLBase, all = true, imported = false)
        ) do name
            Base.ispublic(SciMLBase, name) &&
                isdefined(SciMLBase, name) &&
                let value = getfield(SciMLBase, name)
                value isa DataType && isabstracttype(value)
            end
        end

        for name in abstract_public_names
            @test Base.Docs.hasdoc(SciMLBase, name)
            @test "SciMLBase.$name" in interface_api_bindings
        end
    end
end
