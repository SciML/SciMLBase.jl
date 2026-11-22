# The tick times of an indexed periodic clock are the times at which
# `PeriodicCallback` fires with the period and phase of the clock, and the solution
# evaluated at these times contains the state updated by the callback.
using OrdinaryDiffEq, DiffEqCallbacks, Test
using SciMLBase: Clock, canonicalize_indexed_clock

@testset "Indexed clock ticks equal PeriodicCallback firings" begin
    for (clock, tspan) in (
            (Clock(1 // 30; phase = 1 // 60), (0.0, 1.0)),
            (Clock(1 // 200), (0.0, 1.5)),
            (Clock(1 // 10; phase = 1 // 2), (nextfloat(1.0), 3.0)),
            (Clock(0.1; phase = 0.05), (0.3, 2.0)),
            (Clock(0.25), (sqrt(2.0), 4.0)),
        )
        firings = Float64[]
        affect! = function (integrator)
            integrator.u[1] += 1
            push!(firings, integrator.t)
        end
        # a clock with zero phase ticks at the start time, as in ModelingToolkit
        cb = PeriodicCallback(
            affect!, clock.dt; phase = clock.phase, initial_affect = iszero(clock.phase)
        )
        prob = ODEProblem((du, u, p, t) -> (du[1] = 0), [0.0], tspan)
        sol = solve(prob, Tsit5(); callback = cb)
        @test sol.retcode == ReturnCode.Success
        n = length(firings)
        @test n > 2
        ticks = canonicalize_indexed_clock(clock[1:n], sol)
        @test ticks == firings
        @test [sol(t; continuity = :right)[1] for t in ticks] == 1:n
    end
end
