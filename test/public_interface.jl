using SciMLBase, Test

struct ProblemTypeTestProblem <: SciMLBase.AbstractSciMLProblem end
struct ProblemTypeTestMarker end
struct ProblemTypeTestSolution
    prob::ProblemTypeTestProblem
end

SciMLBase.problem_type(::ProblemTypeTestProblem) = ProblemTypeTestMarker()
SciMLBase.wrap_sol(::ProblemTypeTestSolution, ::ProblemTypeTestMarker) = :wrapped

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
    @test occursin("SciMLBase.ImmutableODEProblem", problem_docs)
    @test !occursin("will always make a deep copy", function_docs)
    @test occursin("selected differentiation and linear solver", function_docs)
    @test occursin("`update_coefficients` for the out-of-place form", function_docs)
    @test occursin("SciMLBase.ODEFunction", function_docs)
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
    @test occursin("SciMLBase.StandardDDEProblem", problem_docs)
    @test occursin("SciMLBase.StandardBVProblem", problem_docs)
    @test occursin("SciMLBase.StandardNonlinearProblem", problem_docs)
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

if isdefined(Base, :ispublic)
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
                :step!,
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
end
