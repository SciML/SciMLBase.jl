using SciMLBase, Test, Logging, LinearAlgebra
using SciMLBase: checkkwargs, can_honor, keyword_class, keyword_status, KeywordVerbosity,
    KeywordArgError, KeywordArgWarn, KeywordArgSilent, CommonKwargError
using SciMLLogging: None, ErrorLevel, WarnLevel, InfoLevel, Silent

struct NoMaxtimeAlg end
SciMLBase.can_honor(::NoMaxtimeAlg, ::Val{:maxtime}) = false
struct HonorsAllAlg end

struct MockJumpProblem{P} <: SciMLBase.AbstractJumpProblem{P, Nothing}
    prob::P
end

nlprob = NonlinearProblem((u, p) -> u .- 1, [0.0])
odeprob = ODEProblem((u, p, t) -> u, [1.0], (0.0, 1.0))
discprob = DiscreteProblem((u, p, t) -> u, [1.0], (0.0, 1.0))

const UNDEF = r"not defined for a NonlinearProblem"
const UNHONORED = r"cannot honor keyword\(s\) `maxtime`"

@testset "Class mapping" begin
    mm = ODEFunction((du, u, p, t) -> nothing; mass_matrix = [1.0 0; 0 0])
    cases = [
        odeprob => :ODE,
        ODEProblem(mm, [1.0, 1.0], (0.0, 1.0)) => :DAE,
        SDEProblem((u, p, t) -> u, (u, p, t) -> u, [1.0], (0.0, 1.0)) => :ODE,
        DDEProblem((u, h, p, t) -> u, [1.0], (p, t) -> [1.0], (0.0, 1.0)) => :ODE,
        DAEProblem((du, u, p, t) -> du - u, [1.0], [1.0], (0.0, 1.0)) => :DAE,
        discprob => :Discrete,
        ImplicitDiscreteProblem((u, un, p, t) -> u - un, [1.0], (0.0, 1.0)) => :ODE,
        MockJumpProblem(discprob) => :Discrete,
        MockJumpProblem(odeprob) => :ODE,
        BVProblem((u, p, t) -> u, (u, p, t) -> u[1][1], [1.0], (0.0, 1.0)) => :BVP,
        SteadyStateProblem(odeprob) => :SteadyState,
        nlprob => :Nonlinear,
        NonlinearLeastSquaresProblem((u, p) -> u .- 1, [0.0]) => :NLLS,
        IntervalNonlinearProblem((u, p) -> u, (0.0, 1.0)) => :Interval,
        OptimizationProblem(OptimizationFunction((u, p) -> sum(u)), [0.0]) => :Optimization,
        LinearProblem([1.0;;], [1.0]) => :Linear,
        IntegralProblem((x, p) -> x, (0.0, 1.0)) => :Integral,
        EnsembleProblem(odeprob) => nothing,
    ]
    for (prob, class) in cases
        @test keyword_class(prob) === class
    end
    mmprob(M) = ODEProblem(
        ODEFunction((du, u, p, t) -> nothing; mass_matrix = M), [1.0, 1.0], (0.0, 1.0)
    )
    @test keyword_class(mmprob(I)) === :ODE
    @test keyword_class(mmprob(Diagonal([1.0, 1.0]))) === :ODE
    @test keyword_class(mmprob(0 * I)) === :DAE
    @test keyword_class(mmprob(Diagonal([1.0, 0.0]))) === :DAE
end

@testset "Three states" begin
    @test keyword_status(nlprob, :abstol) === :defined
    @test keyword_status(nlprob, :gtol) === :defined
    @test keyword_status(nlprob, :saveat) === :undefined
    @test keyword_status(nlprob, :dtmin) === :undefined
    @test keyword_status(nlprob, :not_a_keyword) === :unrecognized
    @test keyword_status(odeprob, :saveat) === :defined
    @test keyword_status(odeprob, :gtol) === :undefined
    @test keyword_status(IntervalNonlinearProblem((u, p) -> u, (0.0, 1.0)), :reltol) ===
        :undefined
    opt = OptimizationProblem(OptimizationFunction((u, p) -> sum(u)), [0.0])
    @test keyword_status(opt, :callback) === :defined
    @test keyword_status(opt, :saveat) === :undefined
    @test keyword_status(discprob, :abstol) === :undefined
    @test keyword_status(discprob, :internalnorm) === :undefined
    for kw in (:saveat, :dt, :tstops, :save_everystep, :callback)
        @test keyword_status(discprob, kw) === :defined
    end
    @test keyword_status(IntervalNonlinearProblem((u, p) -> u, (0.0, 1.0)), :internalnorm) ===
        :undefined
    bvp = BVProblem((u, p, t) -> u, (u, p, t) -> u[1][1], [1.0], (0.0, 1.0))
    @test keyword_status(bvp, :dtmin) === :defined
    @test_logs checkkwargs(bvp, nothing; dtmin = 1.0e-8)
    @test keyword_status(EnsembleProblem(odeprob), :gtol) === :unclassified
    @test keyword_status(EnsembleProblem(odeprob), :not_a_keyword) === :unrecognized

    @test_logs checkkwargs(nlprob, HonorsAllAlg(); abstol = 1.0e-3, maxiters = 10)
    @test_logs (:warn, UNDEF) checkkwargs(nlprob, HonorsAllAlg(); saveat = 0.1)
    @test_logs (:warn, UNHONORED) checkkwargs(nlprob, NoMaxtimeAlg(); maxtime = 1.0)
    @test_throws CommonKwargError checkkwargs(nlprob, HonorsAllAlg(); not_a_keyword = 1)
    @test_logs checkkwargs(nlprob, HonorsAllAlg(); saveat = nothing, maxtime = nothing)
    @test_logs (:warn, r"`dtmax`, `dtmin`") checkkwargs(
        nlprob, HonorsAllAlg(); abstol = 1.0, dtmax = 1.0, dtmin = 1.0
    )
    # can_honor is only consulted for keywords defined for the class
    @test_logs (:warn, r"not defined for a DiscreteProblem") checkkwargs(
        discprob, NoMaxtimeAlg(); abstol = 1.0
    )
end

@testset "Severities" begin
    err = KeywordVerbosity(undefined_keyword = ErrorLevel, unhonored_keyword = ErrorLevel)
    @test_logs (:error, UNDEF) @test_throws ErrorException checkkwargs(
        nlprob, nothing; saveat = 0.1, kwargs_verbosity = err
    )
    @test_logs (:error, UNHONORED) @test_throws ErrorException checkkwargs(
        nlprob, NoMaxtimeAlg(); maxtime = 1.0, kwargs_verbosity = err
    )
    info = KeywordVerbosity(undefined_keyword = InfoLevel)
    @test_logs (:info, UNDEF) checkkwargs(nlprob, nothing; saveat = 0.1, kwargs_verbosity = info)
    silent = KeywordVerbosity(undefined_keyword = Silent, unhonored_keyword = Silent)
    @test_logs min_level = Logging.Debug checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0, kwargs_verbosity = silent
    )
    mixed = KeywordVerbosity(undefined_keyword = Silent)
    @test_logs (:warn, UNHONORED) checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0, kwargs_verbosity = mixed
    )
    @test KeywordVerbosity().undefined_keyword == WarnLevel
    @test KeywordVerbosity().unhonored_keyword == WarnLevel
    @test_throws ArgumentError checkkwargs(nlprob, nothing; kwargs_verbosity = 1)
end

@testset "Bool verbose does not disable validation" begin
    @test_logs (:warn, UNDEF) checkkwargs(nlprob, nothing; saveat = 0.1, verbose = false)
end

@testset "kwargshandle precedence" begin
    @test_logs (:error, UNDEF) @test_throws ErrorException checkkwargs(
        nlprob, nothing; saveat = 0.1, kwargshandle = KeywordArgError
    )
    @test_logs (:warn, UNDEF) checkkwargs(
        nlprob, nothing; saveat = 0.1, kwargshandle = KeywordArgWarn,
        kwargs_verbosity = KeywordVerbosity(undefined_keyword = ErrorLevel)
    )
    @test_logs min_level = Logging.Debug checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0,
        kwargshandle = KeywordArgSilent,
        kwargs_verbosity = KeywordVerbosity(undefined_keyword = ErrorLevel)
    )
    @test_throws ArgumentError checkkwargs(nlprob, nothing; kwargshandle = :error)
end

@testset "Internal-caller escape hatch" begin
    internal = KeywordVerbosity(None())
    @test internal isa KeywordVerbosity{false}
    @test_logs min_level = Logging.Debug checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0, kwargs_verbosity = internal
    )
    @test_logs min_level = Logging.Debug checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0, kwargs_verbosity = None()
    )
    # It disables the class checks even when a forwarded `kwargshandle` is present...
    @test_logs min_level = Logging.Debug checkkwargs(
        nlprob, NoMaxtimeAlg(); saveat = 0.1, maxtime = 1.0,
        kwargshandle = KeywordArgError, kwargs_verbosity = internal
    )
    # ...but not the unrecognized-keyword check.
    @test_throws CommonKwargError checkkwargs(
        nlprob, nothing; not_a_keyword = 1, kwargs_verbosity = internal
    )
end

@testset "can_honor" begin
    @test can_honor(HonorsAllAlg(), Val(:maxtime))
    @test can_honor(nothing, Val(:abstol))
    @test !can_honor(NoMaxtimeAlg(), Val(:maxtime))
    @test can_honor(NoMaxtimeAlg(), Val(:abstol))
    @test_logs checkkwargs(nlprob, HonorsAllAlg(); maxtime = 1.0)
end

@testset "Legacy checkkwargs is unchanged" begin
    @test checkkwargs(KeywordArgError; abstol = 1.0, saveat = 0.1) === nothing
    @test checkkwargs(KeywordArgError; kwargs_verbosity = KeywordVerbosity()) === nothing
    @test_throws CommonKwargError checkkwargs(KeywordArgError; gtol = 1.0)
    @test_throws CommonKwargError checkkwargs(KeywordArgError; not_a_keyword = 1)
    @test checkkwargs(KeywordArgSilent; not_a_keyword = 1) === nothing
    @test :kwargs_verbosity in SciMLBase.allowedkeywords
end
