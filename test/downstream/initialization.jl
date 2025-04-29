using ModelingToolkit, NonlinearSolve, OrdinaryDiffEq, Sundials, SciMLBase, Test
using SymbolicIndexingInterface
using ModelingToolkit: t_nounits as t, D_nounits as D
using StochasticDiffEq, OrdinaryDiffEq, NonlinearSolve, SymbolicIndexingInterface,
      LinearAlgebra, Test
@testset "CheckInit" begin
    abstol = 1e-10
    @testset "Sundials + DAEProblem" begin
        function daerhs(du, u, p, t)
            return [du[1] - u[1] * t - p, u[1]^2 - u[2]^2]
        end
        function daerhs!(resid, du, u, p, t)
            resid[1] = du[1] - u[1] * t - p
            resid[2] = u[1]^2 - u[2]^2
        end

        oopfn = DAEFunction{false}(daerhs)
        iipfn = DAEFunction{true}(daerhs!)

        @testset "Inplace = $(SciMLBase.isinplace(f))" for f in [oopfn, iipfn]
            prob = DAEProblem(f, [1.0, 0.0], [1.0, 1.0], (0.0, 1.0), 1.0)
            integ = init(prob, Sundials.IDA())
            u0, _,
            success = SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(), Val(SciMLBase.isinplace(f)); abstol)
            @test success
            @test u0 == prob.u0

            integ.u[2] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(), Val(SciMLBase.isinplace(f)); abstol)

            integ.u[2] = 1.0
            integ.du[1] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(), Val(SciMLBase.isinplace(f)); abstol)
        end
    end
end

@testset "OverrideInit with MTK" begin
    abstol = 1e-10
    reltol = 1e-8

    @variables x(t) [guess = 1.0] y(t) [guess = 1.0]
    @parameters p=missing [guess=1.0] q=missing [guess=1.0]
    @mtkbuild sys = ODESystem([D(x) ~ p * y + q * t, D(y) ~ 5x + q], t;
        initialization_eqs = [p^2 + q^2 ~ 3, x^3 + y^3 ~ 5])
    prob = ODEProblem(
        sys, [x => 1.0], (0.0, 1.0), [p => 1.0]; initializealg = SciMLBase.NoInit())

    @test prob.f.initialization_data isa SciMLBase.OverrideInitData
    integ = init(prob, Tsit5())
    u0, pobj,
    success = SciMLBase.get_initial_values(
        prob, integ, prob.f, SciMLBase.OverrideInit(), Val(true);
        nlsolve_alg = NewtonRaphson(), abstol, reltol)

    @test getu(sys, x)(u0) ≈ 1.0
    @test getu(sys, y)(u0) ≈ cbrt(4)
    @test getp(sys, p)(pobj) ≈ 1.0
    @test getp(sys, q)(pobj) ≈ sqrt(2)
end

@testset "Initialization info printed" begin
    @parameters g
    @variables x(t) y(t) [state_priority = 10] λ(t)
    eqs = [D(D(x)) ~ λ * x
           D(D(y)) ~ λ * y - g
           x^2 + y^2 ~ 1]
    @mtkbuild pend = ODESystem(eqs, t)

    prob = ODEProblem(pend, [x => 1, y => 0], (0.0, 1.5), [g => 1], guesses = [λ => 1])
    @test occursin("Initialization status: FULLY_DETERMINED", sprint(summary, prob))

    prob = ODEProblem(pend, [], (0.0, 1.5), [g => 1], guesses = [λ => 1, x => 1, y => 0])
    @test occursin("Initialization status: UNDERDETERMINED", sprint(summary, prob))

    prob = ODEProblem(pend, [x => 1, y => 0, λ => 2], (0.0, 1.5), [g => 1])
    @test occursin("Initialization status: OVERDETERMINED", sprint(summary, prob))
end

@testset "CheckInit" begin
    @testset "ODEProblem" begin
        function rhs(u, p, t)
            return [u[1] * t, u[1]^2 - u[2]^2]
        end
        function rhs!(du, u, p, t)
            du[1] = u[1] * t
            du[2] = u[1]^2 - u[2]^2
        end

        oopfn = ODEFunction{false}(rhs, mass_matrix = [1 0; 0 0])
        iipfn = ODEFunction{true}(rhs!, mass_matrix = [1 0; 0 0])

        @testset "Inplace = $(SciMLBase.isinplace(f))" for f in [oopfn, iipfn]
            prob = ODEProblem(f, [1.0, 1.0], (0.0, 1.0))
            integ = init(prob)
            u0, _,
            success = SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
            @test success
            @test u0 == prob.u0

            integ.u[2] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
        end

        @testset "With I mass matrix" begin
            function rhs(u, p, t)
                return u
            end
            prob = ODEProblem(ODEFunction(rhs; mass_matrix = I), ones(2), (0.0, 1.0))
            integ = init(prob)
            u0, _,
            success = SciMLBase.get_initial_values(
                prob, integ, prob.f, SciMLBase.CheckInit(),
                Val(false); abstol = 1e-10
            )
            @test success
            @test u0 == prob.u0
        end
    end

    @testset "DAEProblem" begin
        function daerhs(du, u, p, t)
            return [du[1] - u[1] * t - p, u[1]^2 - u[2]^2]
        end
        function daerhs!(resid, du, u, p, t)
            resid[1] = du[1] - u[1] * t - p
            resid[2] = u[1]^2 - u[2]^2
        end

        oopfn = DAEFunction{false}(daerhs)
        iipfn = DAEFunction{true}(daerhs!)

        @testset "Inplace = $(SciMLBase.isinplace(f))" for f in [oopfn, iipfn]
            prob = DAEProblem(f, [1.0, 0.0], [1.0, 1.0], (0.0, 1.0), 1.0)
            integ = init(prob, DImplicitEuler())
            u0, _,
            success = SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
            @test success
            @test u0 == prob.u0

            integ.u[2] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)

            integ.u[2] = 1.0
            integ.du[1] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
        end
    end

    @testset "SDEProblem" begin
        mm_A = [1 0 0; 0 1 0; 0 0 0]
        function sdef!(du, u, p, t)
            du[1] = u[1]
            du[2] = u[2]
            du[3] = u[1] + u[2] + u[3] - 1
        end
        function sdef(u, p, t)
            du = similar(u)
            sdef!(du, u, p, t)
            du
        end

        function g!(du, u, p, t)
            @. du = 0.1
        end
        function g(u, p, t)
            du = similar(u)
            g!(du, u, p, t)
            du
        end
        iipfn = SDEFunction{true}(sdef!, g!; mass_matrix = mm_A)
        oopfn = SDEFunction{false}(sdef, g; mass_matrix = mm_A)

        @testset "Inplace = $(SciMLBase.isinplace(f))" for f in [oopfn, iipfn]
            prob = SDEProblem(f, [1.0, 1.0, -1.0], (0.0, 1.0))
            integ = init(prob, ImplicitEM())
            u0, _,
            success = SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
            @test success
            @test u0 == prob.u0

            integ.u[2] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(),
                Val(SciMLBase.isinplace(f)); abstol = 1e-10)
        end
    end
end

@testset "OverrideInit" begin
    function rhs2(u, p, t)
        return [u[1] * t + p, u[1]^2 - u[2]^2]
    end

    @testset "No-op without `initialization_data`" begin
        prob = ODEProblem(rhs2, [1.0, 2.0], (0.0, 1.0), 1.0)
        @test SciMLBase.initialization_status(prob) === nothing
        integ = init(prob)
        integ.u[2] = 3.0
        u0, p,
        success = SciMLBase.get_initial_values(
            prob, integ, prob.f, SciMLBase.OverrideInit(), Val(false))
        @test u0 ≈ [1.0, 3.0]
        @test success
    end

    # unknowns are u[2], p. Parameter is u[1]
    initprob = NonlinearProblem([1.0, 1.0], [1.0]) do x, _u1
        u2, p = x
        u1 = _u1[1]
        return [u1^2 - u2^2, p^2 - 2p + 1]
    end
    update_initializeprob! = function (iprob, integ)
        iprob.p[1] = integ.u[1]
    end
    initprobmap = function (nlsol)
        return [parameter_values(nlsol)[1], nlsol.u[1]]
    end
    initprobpmap = function (_, nlsol)
        return nlsol.u[2]
    end
    initialization_data = SciMLBase.OverrideInitData(
        initprob, update_initializeprob!, initprobmap, initprobpmap)
    fn = ODEFunction(rhs2; initialization_data)
    prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
    @test SciMLBase.initialization_status(prob) == SciMLBase.FULLY_DETERMINED
    integ = init(prob; initializealg = NoInit())

    @testset "Errors without `nlsolve_alg`" begin
        @test_throws SciMLBase.OverrideInitMissingAlgorithm SciMLBase.get_initial_values(
            prob, integ, fn, SciMLBase.OverrideInit(), Val(false))
    end

    abstol = 1e-10
    reltol = 1e-10
    @testset "Solves" begin
        @testset "with explicit alg" begin
            u0, p,
            success = SciMLBase.get_initial_values(
                prob, integ, fn, SciMLBase.OverrideInit(),
                Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol)

            @test u0 ≈ [2.0, 2.0]
            @test p ≈ 1.0
            @test success

            initprob.p[1] = 1.0
        end
        @testset "with alg in `OverrideInit`" begin
            u0, p,
            success = SciMLBase.get_initial_values(
                prob, integ, fn,
                SciMLBase.OverrideInit(; nlsolve = NewtonRaphson(), abstol, reltol),
                Val(false))

            @test u0 ≈ [2.0, 2.0]
            @test p ≈ 1.0
            @test success

            initprob.p[1] = 1.0
        end
        @testset "with trivial problem and no alg" begin
            iprob = NonlinearProblem((u, p) -> 0.0, nothing, 1.0)
            iprobmap = (_) -> [1.0, 1.0]
            initdata = SciMLBase.OverrideInitData(iprob, nothing, iprobmap, nothing)
            _fn = ODEFunction(rhs2; initialization_data = initdata)
            _prob = ODEProblem(_fn, [2.0, 0.0], (0.0, 1.0), 1.0)
            _integ = init(_prob; initializealg = NoInit())

            u0, p,
            success = SciMLBase.get_initial_values(
                _prob, _integ, _fn, SciMLBase.OverrideInit(), Val(false); abstol, reltol)

            @test u0 ≈ [1.0, 1.0]
            @test p ≈ 1.0
            @test success
        end
        @testset "with kwargs provided to `get_initial_values`" begin
            u0, p,
            success = SciMLBase.get_initial_values(
                prob, integ, fn, SciMLBase.OverrideInit(),
                Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol, u0 = [-1.0, 1.0])
            @test u0 ≈ [2.0, -2.0]
            @test p ≈ 1.0
            @test success
        end
    end

    @testset "Solves with non-integrator value provider" begin
        _integ = ProblemState(; u = integ.u, p = parameter_values(integ), t = integ.t)
        u0, p,
        success = SciMLBase.get_initial_values(
            prob, _integ, fn, SciMLBase.OverrideInit(),
            Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol)

        @test u0 ≈ [2.0, 2.0]
        @test p ≈ 1.0
        @test success

        initprob.p[1] = 1.0
    end

    @testset "Solves without `update_initializeprob!`" begin
        initdata = SciMLBase.@set initialization_data.update_initializeprob! = nothing
        fn = ODEFunction(rhs2; initialization_data = initdata)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        integ = init(prob; initializealg = NoInit())

        u0, p,
        success = SciMLBase.get_initial_values(
            prob, integ, fn, SciMLBase.OverrideInit(),
            Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol)
        @test u0 ≈ [1.0, 1.0]
        @test p ≈ 1.0
        @test success
    end

    @testset "Solves without `initializeprobmap`" begin
        initdata = SciMLBase.@set initialization_data.initializeprobmap = nothing
        fn = ODEFunction(rhs2; initialization_data = initdata)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        integ = init(prob; initializealg = NoInit())

        u0, p,
        success = SciMLBase.get_initial_values(
            prob, integ, fn, SciMLBase.OverrideInit(),
            Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol)

        @test u0 ≈ [2.0, 0.0]
        @test p ≈ 1.0
        @test success
    end

    @testset "Solves without `initializeprobpmap`" begin
        initdata = SciMLBase.@set initialization_data.initializeprobpmap = nothing
        fn = ODEFunction(rhs2; initialization_data = initdata)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        integ = init(prob; initializealg = NoInit())

        u0, p,
        success = SciMLBase.get_initial_values(
            prob, integ, fn, SciMLBase.OverrideInit(),
            Val(false); nlsolve_alg = NewtonRaphson(), abstol, reltol)

        @test u0 ≈ [2.0, 2.0]
        @test p ≈ 0.0
        @test success
    end

    @testset "Initialization status for `SCCNonlinearProblem`" begin
        initprob = SCCNonlinearProblem([initprob], [Returns(nothing)])
        initialization_data = SciMLBase.OverrideInitData(
            initprob, nothing, nothing, nothing)
        fn = ODEFunction(rhs2; initialization_data)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        @test SciMLBase.initialization_status(prob) == SciMLBase.FULLY_DETERMINED
    end

    @testset "Trivial initialization" begin
        initprob = NonlinearProblem(Returns(nothing), nothing, [1.0])
        update_initializeprob! = function (iprob, integ)
            # just to access the current time and use it as a number, so this errors
            # if run on a problem with `current_time(prob) === nothing`
            iprob.p[1] = current_time(integ) + 1
            iprob.p[1] = state_values(integ)[1]
        end
        initprobmap = function (nlsol)
            u1 = parameter_values(nlsol)[1]
            return [u1, u1]
        end
        initprobpmap = function (_, nlsol)
            return 0.0
        end
        initialization_data = SciMLBase.OverrideInitData(
            initprob, update_initializeprob!, initprobmap, initprobpmap)
        fn = ODEFunction(rhs2; initialization_data)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        @test SciMLBase.initialization_status(prob) == SciMLBase.FULLY_DETERMINED
        integ = init(prob; initializealg = NoInit())

        u0, p,
        success = SciMLBase.get_initial_values(
            prob, integ, fn, SciMLBase.OverrideInit(), Val(false)
        )
        @test u0 ≈ [2.0, 2.0]
        @test p ≈ 0.0
        @test success

        @testset "Doesn't run in `remake` if `tspan == (nothing, nothing)`" begin
            prob = ODEProblem(fn, [2.0, 0.0], (nothing, nothing), 0.0)
            @test_nowarn remake(prob)
        end
    end

    @testset "Initialization status for overdetermined case" begin
        initfn = NonlinearFunction(; resid_prototype = ones(3)) do u, p
            return [u[1] - 1.0, u[2] - 1.0, u[1] * u[2] - 1.0]
        end
        initprob = NonlinearLeastSquaresProblem(initfn, ones(2))
        initialization_data = SciMLBase.OverrideInitData(
            initprob, nothing, nothing, nothing)
        fn = ODEFunction(rhs2; initialization_data)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        @test SciMLBase.initialization_status(prob) == SciMLBase.OVERDETERMINED
    end

    @testset "Initialization status for underdetermined case" begin
        initfn = NonlinearFunction(; resid_prototype = ones(1)) do u, p
            return [u[1] - 1.0]
        end
        initprob = NonlinearLeastSquaresProblem(initfn, ones(2))
        initialization_data = SciMLBase.OverrideInitData(
            initprob, nothing, nothing, nothing)
        fn = ODEFunction(rhs2; initialization_data)
        prob = ODEProblem(fn, [2.0, 0.0], (0.0, 1.0), 0.0)
        @test SciMLBase.initialization_status(prob) == SciMLBase.UNDERDETERMINED
    end
end

@testset "NoInit" begin
    prob = ODEProblem(ones(2), (0.0, 1.0), ones(2)) do u, p, t
        return u
    end
    u, p,
    success = SciMLBase.get_initial_values(
        prob, prob, prob.f, SciMLBase.NoInit(), Val(true))
    @test u == ones(2)
    @test p == ones(2)
    @test success
end
