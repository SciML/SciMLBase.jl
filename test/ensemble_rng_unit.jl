using Test, SciMLBase, Random

@testset "Ensemble RNG Unit Tests" begin
    @testset "EnsembleContext construction and field access" begin
        ctx = EnsembleContext(3, 1, 0, UInt64(12345), nothing, nothing)
        @test ctx.sim_id == 3
        @test ctx.repeat == 1
        @test ctx.worker_id == 0
        @test ctx.sim_seed === UInt64(12345)
        @test ctx.rng === nothing
        @test ctx.master_rng === nothing

        rng = Random.Xoshiro(99)
        ctx2 = EnsembleContext(7, 2, 1, nothing, rng, rng)
        @test ctx2.sim_id == 7
        @test ctx2.repeat == 2
        @test ctx2.worker_id == 1
        @test ctx2.sim_seed === nothing
        @test ctx2.rng === rng
        @test ctx2.master_rng === rng
    end

    @testset "generate_sim_seeds" begin
        # Determinism: same seed twice → identical arrays
        seeds1 = SciMLBase.generate_sim_seeds(nothing, UInt64(42), 10)
        seeds2 = SciMLBase.generate_sim_seeds(nothing, UInt64(42), 10)
        @test seeds1 == seeds2
        @test length(seeds1) == 10
        @test eltype(seeds1) === UInt64

        # rng kwarg produces same results as seed when using same Xoshiro state
        seeds_from_rng = SciMLBase.generate_sim_seeds(Random.Xoshiro(42), nothing, 10)
        @test seeds_from_rng == seeds1

        # Different seeds produce different arrays
        seeds_other = SciMLBase.generate_sim_seeds(nothing, UInt64(123), 10)
        @test seeds_other != seeds1

        # All seeds within a batch are distinct (deterministic check against known values)
        @test length(Set(seeds1)) == length(seeds1)
    end

    @testset "default_rng_func" begin
        seed_val = UInt64(42)
        ctx = EnsembleContext(1, 1, 0, seed_val, nothing, nothing)

        rng = SciMLBase.default_rng_func(ctx)
        @test rng isa Random.AbstractRNG
        @test rng === Random.default_rng()

        # Same context → same initial random stream
        val1 = rand(rng, Float64)
        rng2 = SciMLBase.default_rng_func(ctx)
        val2 = rand(rng2, Float64)
        @test val1 == val2

        # nothing seed → no error, still returns default_rng
        ctx_noseed = EnsembleContext(1, 1, 0, nothing, nothing, nothing)
        rng3 = SciMLBase.default_rng_func(ctx_noseed)
        @test rng3 isa Random.AbstractRNG
    end
end
