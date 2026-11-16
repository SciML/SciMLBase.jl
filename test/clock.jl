using Test
using SciMLBase
using SciMLBase: Clock, PeriodicClock, Continuous, ContinuousClock, SolverStepClock,
    first_clock_tick_time, IndexedClock, canonicalize_indexed_clock

@testset "Clock" begin
    @test PeriodicClock(nothing, 0.2) isa TimeDomain
    @test SolverStepClock() isa TimeDomain
    @test ContinuousClock() isa TimeDomain
    @test Continuous() isa TimeDomain
    @test Continuous === ContinuousClock()

    @test Clock(1) isa TimeDomain
    @test Clock(24.0; phase = 0.1) == PeriodicClock(24.0, 0.1)
    @test Clock(1 // 24) == PeriodicClock(1 // 24, 0.0)
    @test Clock(; phase = 0.2) == PeriodicClock(nothing, 0.2)
    @test PeriodicClock(1 // 30, 1 // 60).phase === 1 // 60
    @test PeriodicClock(1 // 30, 1 // 60) isa PeriodicClock{Rational{Int}, Rational{Int}}
    @test PeriodicClock(big(1) // 30, 0.0) isa PeriodicClock{Rational{BigInt}, Float64}
    @test PeriodicClock(nothing, 0.2) isa PeriodicClock{Nothing, Float64}
    @test PeriodicClock(1, 0).dt === 1.0
    @test Clock(1 // 30; phase = 1 // 60) == PeriodicClock(1 // 30, 1 // 60)
    @test Clock(0.1; phase = 1).phase === 1.0
    @test Clock(0.1; phase = 0 // 1) == Clock(0.1)
    @test isequal(Clock(0.1; phase = 1 // 2), Clock(0.1; phase = 0.5))
    @test hash(Clock(0.1; phase = 1 // 2)) == hash(Clock(0.1; phase = 0.5))
    @test Clock(1 // 2) == Clock(0.5)
    @test hash(Clock(1 // 2)) == hash(Clock(0.5))
    @test Clock(0.1; phase = 1 // 3) != Clock(0.1; phase = 1 / 3)
    @test Clock(; phase = 0.2) == Clock(; phase = 0.2)

    @test isclock(PeriodicClock(; dt = 1.0))
    @test !isclock(Continuous())
    @test !isclock(SolverStepClock())
    @test !isclock(nothing)

    @test !issolverstepclock(PeriodicClock(; dt = 1.0))
    @test !issolverstepclock(Continuous())
    @test issolverstepclock(SolverStepClock())
    @test !issolverstepclock(nothing)

    @test !iscontinuous(PeriodicClock(; dt = 1.0))
    @test iscontinuous(Continuous())
    @test !iscontinuous(SolverStepClock())
    @test !iscontinuous(nothing)

    @test is_discrete_time_domain(PeriodicClock(; dt = 1.0))
    @test !is_discrete_time_domain(Continuous())
    @test is_discrete_time_domain(SolverStepClock())
    @test !is_discrete_time_domain(nothing)

    # the ticks are t0 + phase + k*dt
    @test first_clock_tick_time(PeriodicClock(; dt = 2.0), 5.0) === 5.0
    @test first_clock_tick_time(PeriodicClock(2.0, 0.5), 0.0) === 0.5
    @test first_clock_tick_time(PeriodicClock(2.0, 0.5), 5.0) === 5.5
    @test first_clock_tick_time(PeriodicClock(1.0, 2.5), 0.0) === 2.5
    @test first_clock_tick_time(PeriodicClock(1 // 30, 1 // 60), 0.0) === 1 / 60
    @test first_clock_tick_time(PeriodicClock(1 // 30, 1 // 60), sqrt(2.0)) === sqrt(2.0) + 1 / 60
    @test_throws ErrorException first_clock_tick_time(Continuous(), 5.0)
    @test first_clock_tick_time(SolverStepClock(), 5.0) === 5.0

    ic = Clock(1)[5]
    @test ic === IndexedClock(Clock(1), 5)
end

@testset "canonicalize_indexed_clock with phase and start time" begin
    prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), [1.0], (0.0, 2.0))
    sol = SciMLBase.build_solution(prob, nothing, [0.0, 2.0], [[1.0], [1.0]])
    # rational clocks: tick times computed exactly and rounded once
    @test canonicalize_indexed_clock(Clock(1 // 30; phase = 1 // 60)[41], sol) === 1.35
    @test canonicalize_indexed_clock(Clock(1 // 200)[271], sol) === 1.35
    @test canonicalize_indexed_clock(Clock(1 // 30; phase = 1 // 60)[1:2], sol) == [1 / 60, 0.05]
    @test canonicalize_indexed_clock(Clock(0.5; phase = 0.25)[[1, 3]], sol) == [0.25, 1.25]
    @test canonicalize_indexed_clock(Clock(0.1)[3], sol) === 0.2
    # the ticks are relative to the start time of the simulation
    sol2 = SciMLBase.build_solution(SciMLBase.remake(prob; tspan = (1.0, 2.0)), nothing, [1.0, 2.0], [[1.0], [1.0]])
    @test canonicalize_indexed_clock(Clock(0.5; phase = 0.25)[1:2], sol2) == [1.25, 1.75]
    @test canonicalize_indexed_clock(Clock(0.3)[1:2], sol2) == [1.0, 1.3]
    # equal clocks give equal tick times
    sol3 = SciMLBase.build_solution(SciMLBase.remake(prob; tspan = (0.1, 2.0)), nothing, [0.1, 2.0], [[1.0], [1.0]])
    @test canonicalize_indexed_clock(Clock(1 // 10; phase = 1 // 2)[1:5], sol3) ==
        canonicalize_indexed_clock(Clock(1 // 10; phase = 0.5)[1:5], sol3)
    # the tick of a clock with zero phase at the start time is the start time
    @test first_clock_tick_time(Clock(0.1), big"0.1") == big"0.1"
end
