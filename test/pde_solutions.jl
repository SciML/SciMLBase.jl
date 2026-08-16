using Test, SciMLBase

struct TestTimeMetadata <: SciMLBase.AbstractDiscretizationMetadata{Val(true)} end

struct TestNoTimeMetadata <: SciMLBase.AbstractDiscretizationMetadata{Val(false)} end

SciMLBase.PDETimeSeriesSolution(sol, metadata::TestTimeMetadata) =
    (:time_series, sol, metadata)

SciMLBase.PDENoTimeSolution(sol, metadata::TestNoTimeMetadata) =
    (:no_time, sol, metadata)

@testset "wrap_sol routes by discretization metadata time flag" begin
    sol = Any[:discretized_solution]

    time_metadata = TestTimeMetadata()
    wrapped_time = SciMLBase.wrap_sol(sol, time_metadata)
    @test wrapped_time[1] === :time_series
    @test wrapped_time[2] === sol
    @test wrapped_time[3] === time_metadata

    no_time_metadata = TestNoTimeMetadata()
    wrapped_no_time = SciMLBase.wrap_sol(sol, no_time_metadata)
    @test wrapped_no_time[1] === :no_time
    @test wrapped_no_time[2] === sol
    @test wrapped_no_time[3] === no_time_metadata
end

@testset "wrap_sol reaches a DAEProblem's discretization metadata" begin
    daef(du, u, p, t) = du .- u
    metadata = TestTimeMetadata()
    prob = DAEProblem(daef, [1.0], [1.0], (0.0, 1.0), SciMLBase.NullParameters(), metadata)

    @test SciMLBase.problem_type(prob) === metadata
    @test SciMLBase.problem_type(remake(prob; u0 = [2.0])) === metadata
    @test SciMLBase.problem_type(DAEProblem(daef, [1.0], [1.0], (0.0, 1.0))) isa
        SciMLBase.StandardDAEProblem

    sol = SciMLBase.build_solution(prob, nothing, [0.0, 1.0], [[1.0], [2.0]])
    wrapped = SciMLBase.wrap_sol(sol)
    @test wrapped[1] === :time_series
    @test wrapped[3] === metadata
end
