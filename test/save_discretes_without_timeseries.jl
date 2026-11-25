using SciMLBase, Test

# A solution without a parameter time series collection has nowhere to save discrete values, so
# `save_discretes!` returns before computing them.
struct NoTimeseriesIntegrator <: SciMLBase.DEIntegrator{Nothing, true, Vector{Float64}, Float64}
    sol::Nothing
end
@testset "save_discretes! without a parameter time series" begin
    integ = NoTimeseriesIntegrator(nothing)
    # past the early return, computing the values to save would throw for this integrator
    @test SciMLBase.save_discretes!(integ, 1) === nothing
    @test SciMLBase.save_discretes!(integ, 2; skip_duplicates = true) === nothing
end
