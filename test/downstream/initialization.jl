using OrdinaryDiffEq, Sundials, SciMLBase, Test

@testset "CheckInit" begin
    abstol = 1e-10
    @testset "Sundials + ODEProblem" begin
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
            integ = init(prob, Sundials.ARKODE())
            u0, _, success = SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(), Val(SciMLBase.isinplace(f)); abstol)
            @test success
            @test u0 == prob.u0

            integ.u[2] = 2.0
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.get_initial_values(
                prob, integ, f, SciMLBase.CheckInit(), Val(SciMLBase.isinplace(f)); abstol)
        end
    end

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
