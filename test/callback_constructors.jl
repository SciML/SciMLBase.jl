using SciMLBase
using Test

@testset "VectorContinuousCallback 3-arg kwarg constructor" begin
    cond(out, u, t, integ) = (out[1] = u[1])
    aff!(integ, idx) = nothing

    cb = VectorContinuousCallback(cond, aff!, 1; abstol = 1.0e-12)
    @test cb.abstol == 1.0e-12
    @test cb.maybe_discontinuity == true

    cb_kw = VectorContinuousCallback(
        cond, aff!, 2;
        save_positions = (true, false),
        rootfind = SciMLBase.LeftRootFind,
        abstol = 1.0e-9, reltol = 0.0,
        saved_clock_partitions = (),
        maybe_discontinuity = false,
        initialize_save_discretes = false
    )
    @test cb_kw.save_positions == BitVector([true, false])
    @test cb_kw.abstol == 1.0e-9
    @test cb_kw.maybe_discontinuity == false
    @test cb_kw.initialize_save_discretes == false
    @test cb_kw.saved_clock_partitions == ()
end

# `VectorContinuousCallback` should not accept `affect_neg!`. The
# 4-arg positional form was a leftover from the v6 `ContinuousCallback`
# API that was supposed to be removed in the v7 update; downstream
# code that passed `affect_neg!` had it silently ignored, so the
# constructor itself is now removed (and so is the kwarg).
@testset "VectorContinuousCallback rejects affect_neg!" begin
    cond(out, u, t, integ) = (out[1] = u[1])
    aff!(integ, idx) = nothing
    aff_neg!(integ, idx) = nothing

    @test_throws MethodError VectorContinuousCallback(cond, aff!, aff_neg!, 1)
    @test_throws MethodError VectorContinuousCallback(
        cond, aff!, 1; affect_neg! = aff_neg!
    )
end
