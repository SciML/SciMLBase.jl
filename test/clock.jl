using Test
using SciMLBase
using SciMLBase: Clock, PeriodicClock, Continuous, ContinuousClock, SolverStepClock,
                 first_clock_tick_time, IndexedClock, canonicalize_indexed_clock
using MLStyle: @match

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

    @test isclock(PeriodicClock(; dt = 1.0))
    @test !isclock(Continuous())
    @test !isclock(SolverStepClock())

    @test !issolverstepclock(PeriodicClock(; dt = 1.0))
    @test !issolverstepclock(Continuous())
    @test issolverstepclock(SolverStepClock())

    @test !iscontinuous(PeriodicClock(; dt = 1.0))
    @test iscontinuous(Continuous())
    @test !iscontinuous(SolverStepClock())

    @test is_discrete_time_domain(PeriodicClock(; dt = 1.0))
    @test !is_discrete_time_domain(Continuous())
    @test is_discrete_time_domain(SolverStepClock())

    @test first_clock_tick_time(PeriodicClock(; dt = 2.0), 5.0) === 6.0
    @test_throws ErrorException first_clock_tick_time(Continuous(), 5.0)
    @test first_clock_tick_time(SolverStepClock(), 5.0) === 5.0

    ic = Clock(1)[5]
    @test ic === IndexedClock(Clock(1), 5)
end

@testset "MLStyle" begin
    sampletime(c) = @match c begin
        PeriodicClock(dt, _...) => dt
        _ => nothing
    end

    @test sampletime(PeriodicClock(1 // 2, 3.14)) === 1 // 2
    @test sampletime(ContinuousClock()) === nothing
    @test sampletime(missing) === nothing

    function clocktype(c)
        @match c begin
            Continuous() => "continuous"
            SolverStepClock() => "solver_step_clock"
            PeriodicClock(dt, phase) => (dt, phase)
            _ => "other"
        end
    end

    @test clocktype(Continuous()) === "continuous"
    @test clocktype(ContinuousClock()) === "continuous"
    @test clocktype(Continuous) === "continuous"
    @test clocktype(SolverStepClock()) === "solver_step_clock"
    @test clocktype(PeriodicClock(1 // 2, 3.14)) === (1 // 2, 3.14)
    @test clocktype(pi)==="other" broken=true
end
