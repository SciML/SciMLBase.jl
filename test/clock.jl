using Test
using SciMLBase
using ForwardDiff: Dual, value, partials
using SciMLBase: Clock, PeriodicClock, Continuous, ContinuousClock, SolverStepClock,
    first_clock_tick_time, IndexedClock, canonicalize_indexed_clock

# nearest Float64 to the rational number `x`, ties to even
function nearest(x)
    c = (prevfloat(Float64(x)), Float64(x), nextfloat(Float64(x)))
    dist = map(y -> abs(Rational{BigInt}(y) - x), c)
    return c[argmin(i -> (dist[i], isodd(reinterpret(UInt64, c[i]))), 1:3)]
end

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
    # rationals are stored as Rational{Int}, or as Rational{BigInt} if they do not fit
    @test PeriodicClock(big(1) // 30, 0.0) isa PeriodicClock{Rational{Int}, Float64}
    @test PeriodicClock(big(1) // big(2)^70, 0.0) isa PeriodicClock{Rational{BigInt}, Float64}
    # a rational phase of a floating-point period is converted to Float64
    @test PeriodicClock(0.5, 1 // 10) isa PeriodicClock{Float64, Float64}
    @test PeriodicClock(nothing, 1 // 10).phase === 1 // 10
    @test PeriodicClock(nothing, 0.2) isa PeriodicClock{Nothing, Float64}
    @test PeriodicClock(1, 0).dt === 1.0
    @test Clock(1 // 30; phase = 1 // 60) == PeriodicClock(1 // 30, 1 // 60)
    @test Clock(0.1; phase = 1).phase === 1.0
    @test Clock(0.1; phase = 0 // 1) == Clock(0.1)
    @test isequal(Clock(0.1; phase = 1 // 2), Clock(0.1; phase = 0.5))
    @test hash(Clock(0.1; phase = 1 // 2)) == hash(Clock(0.1; phase = 0.5))
    # a rational and a floating-point period give different tick times
    @test Clock(1 // 2) != Clock(0.5)
    @test !isequal(Clock(1 // 2), Clock(0.5))
    @test Clock(1 // 2; phase = 0.5) == Clock(1 // 2; phase = 1 // 2)
    @test Clock(1 // 10; phase = 1 // 3) != Clock(1 // 10; phase = 1 / 3)
    @test Clock(0.1; phase = 1 // 3) == Clock(0.1; phase = 1 / 3)
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
    # rational clocks: t0 + phase + k*dt computed exactly and rounded once
    @test first_clock_tick_time(PeriodicClock(1 // 30, 1 // 60), sqrt(2.0)) ===
        nearest(Rational{BigInt}(sqrt(2.0)) + 1 // 60)
    @test first_clock_tick_time(PeriodicClock(1 // 10, 1), -1.0) === 0.0
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
    for (a, b) in (
            (Clock(1 // 10; phase = 1 // 2), Clock(1 // 10; phase = 0.5)),
            (Clock(0.5; phase = 1 // 10), Clock(0.5; phase = big(1) // 10)),
            (Clock(big(1) // 2; phase = 1 // 10), Clock(1 // 2; phase = 1 // 10)),
        )
        @test a == b
        @test hash(a) == hash(b)
        @test canonicalize_indexed_clock(a[1:5], sol3) == canonicalize_indexed_clock(b[1:5], sol3)
    end
    # the tick of a clock with zero phase at the start time is the start time
    @test first_clock_tick_time(Clock(0.1), big"0.1") == big"0.1"
end

@testset "Tick times: rounding, precision and generic time types" begin
    tick(c, t0, k) = SciMLBase._periodic_tick_time(c, t0, k)
    # The tick is rounded once to the nearest Float64, independently of the BigFloat
    # precision, on both sides of the midpoint between 1.0 and nextfloat(1.0).
    c = Clock(1 // Int64(2)^53; phase = 1)
    @test tick(c, 0x1p-300, 1) === nextfloat(1.0) === nearest(1 // big(2)^300 + 1 + 1 // big(2)^53)
    @test tick(c, -0x1p-300, 1) === 1.0
    @test tick(c, 0.0, 1) === 1.0
    @test setprecision(() -> tick(Clock(1 // 10), 0.0, 1), BigFloat, 24) === 0.1
    # A BigFloat tick is rounded once; rounding the numerator first gives the next value.
    setprecision(BigFloat, 24) do
        @test tick(Clock(1 // 1; phase = 2 // 3), nextfloat(BigFloat(1)), 0) ==
            ldexp(BigFloat(13981014), -23)
    end
    # The precision of a BigFloat start time is retained.
    t0 = big(2.0)^53 + 1
    ic = Clock(1.0)[1:3]
    prob = SciMLBase.ODEProblem((u, p, t) -> zero(u), [1.0], (t0, t0 + 3))
    sol = SciMLBase.build_solution(prob, nothing, [t0, t0 + 3], [[1.0], [1.0]])
    @test canonicalize_indexed_clock(ic, sol) == t0 .+ (0:2)
    @test canonicalize_indexed_clock(Clock(1 // 3)[1:3], sol) == t0 .+ (0:2) .// 3
    # Dual-number start times are supported and keep their partial derivatives.
    for c in (Clock(0.25), Clock(1 // 4; phase = 1 // 8))
        ticks = [tick(c, Dual(1.0, 1.0), k) for k in 0:2]
        @test value.(ticks) == [tick(c, 1.0, k) for k in 0:2]
        @test all(t -> partials(t)[1] == 1, ticks)
    end
    # A clock without period has only a first tick, at the start time plus phase.
    @test first_clock_tick_time(Clock(), 5.0) === 5.0
    @test first_clock_tick_time(Clock(; phase = 1 // 4), 5.0) === 5.25
    @test first_clock_tick_time(Clock(; phase = 0.25), 5.0) === 5.25
    @test_throws ArgumentError tick(Clock(), 5.0, 1)
    # An infinite phase or period gives infinite tick times.
    @test tick(Clock(1 // 4; phase = Inf), 0.0, 0) === Inf
    @test tick(Clock(1 // 0), 0.0, 1) === Inf
    @test tick(Clock(-1 // 0), 1.0, 1) === -Inf
end

@testset "periodic_schedule and periodic_tick_time" begin
    using SciMLBase: periodic_schedule, periodic_schedule_type, periodic_tick_time
    s = periodic_schedule(Float64, 0.0, 1 // 60, 1 // 30)
    @test s isa periodic_schedule_type(Float64, Rational{Int}, Rational{Int})
    @test periodic_tick_time(Float64, s, 40) === 1.35
    @test periodic_tick_time(Float64, periodic_schedule(Float64, 0.0, 0, 1 // 200), 270) === 1.35
    s2 = periodic_schedule(Float64, 0.3, 0.05, 0.1)
    @test s2 isa periodic_schedule_type(Float64, Float64, Float64) === Tuple{Float64, Float64}
    @test (@inferred periodic_tick_time(Float64, s2, 3)) === (0.3 + 0.05) + 3 * 0.1
    s3 = periodic_schedule(Float32, 1.0f0, 0, 1 // 4)
    @test (@inferred periodic_tick_time(Float32, s3, 2)) === 1.5f0
end
