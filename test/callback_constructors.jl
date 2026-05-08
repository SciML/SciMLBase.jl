using SciMLBase
using Test

# Regression: the 4-arg `VectorContinuousCallback` keyword constructor used to
# forward arguments to the inner positional constructor in the wrong order,
# placing `maybe_discontinuity` in the slot expected for `abstol`.
@testset "VectorContinuousCallback 4-arg kwarg constructor" begin
    cond(out, u, t, integ) = (out[1] = u[1]; out[2] = u[2])
    aff!(integ, idx) = nothing

    cb = VectorContinuousCallback(cond, aff!, nothing, 2)
    @test cb isa VectorContinuousCallback

    cb_kw = VectorContinuousCallback(cond, aff!, nothing, 2;
        save_positions = (true, false),
        rootfind = SciMLBase.LeftRootFind,
        abstol = 1e-9, reltol = 0.0,
        saved_clock_partitions = (),
        maybe_discontinuity = false,
        initialize_save_discretes = false)
    @test cb_kw.save_positions == BitVector([true, false])
    @test cb_kw.abstol == 1e-9
    @test cb_kw.maybe_discontinuity == false
    @test cb_kw.initialize_save_discretes == false
    @test cb_kw.saved_clock_partitions == ()
end

# Sanity check: the 3-arg form (which already had the right order) still works.
@testset "VectorContinuousCallback 3-arg kwarg constructor" begin
    cond(out, u, t, integ) = (out[1] = u[1])
    aff!(integ, idx) = nothing

    cb = VectorContinuousCallback(cond, aff!, 1; abstol = 1e-12)
    @test cb.abstol == 1e-12
    @test cb.maybe_discontinuity == true
end
