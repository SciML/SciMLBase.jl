using ModelingToolkit, NonlinearSolve, OrdinaryDiffEq, Sundials, SciMLBase, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

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
            u0, _, success = SciMLBase.get_initial_values(
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
    @parameters p=missing [guess = 1.0] q=missing [guess = 1.0]
    @mtkbuild sys = ODESystem([D(x) ~ p * y + q * t, D(y) ~ 5x + q], t;
        initialization_eqs = [p^2 + q^2 ~ 3, x^3 + y^3 ~ 5])
    prob = ODEProblem(
        sys, [x => 1.0], (0.0, 1.0), [p => 1.0]; initializealg = SciMLBase.NoInit())

    @test prob.f.initialization_data isa SciMLBase.OverrideInitData
    integ = init(prob, Tsit5())
    u0, pobj, success = SciMLBase.get_initial_values(
        prob, integ, prob.f, SciMLBase.OverrideInit(), Val(true);
        nlsolve_alg = NewtonRaphson(), abstol, reltol)

    @test getu(sys, x)(u0) ≈ 1.0
    @test getu(sys, y)(u0) ≈ cbrt(4)
    @test getp(sys, p)(pobj) ≈ 1.0
    @test getp(sys, q)(pobj) ≈ sqrt(2)
end
