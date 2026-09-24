using SciMLBase, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

f_oop(u, p, t) = 1.01u - u .^ 3
function f_iip(du, u, p, t)
    @. du = 1.01u - u^3
    return nothing
end

u0 = rand(rng, 20)
tspan = (0.0, 1.0)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["problem_construct"] = BenchmarkGroup()

SUITE["problem_construct"]["ODEProblem_iip"] = @benchmarkable ODEProblem(
    $f_iip, $u0, $tspan
)
SUITE["problem_construct"]["ODEProblem_oop"] = @benchmarkable ODEProblem(
    $f_oop, $u0, $tspan
)
SUITE["problem_construct"]["ODEProblem_ODEFunction"] = @benchmarkable ODEProblem(
    $(ODEFunction(f_iip)), $u0, $tspan
)
function f_nl(resid, u, p)
    @. resid = u^2 - 2
    return nothing
end
SUITE["problem_construct"]["NonlinearProblem"] = @benchmarkable NonlinearProblem(
    $f_nl, $u0
)
SUITE["problem_construct"]["SDEProblem"] = @benchmarkable SDEProblem(
    $f_iip, (du, u, p, t) -> du .= 0.1 .* u, $u0, $tspan
)
SUITE["problem_construct"]["IntervalNonlinearProblem"] = @benchmarkable IntervalNonlinearProblem(
    (u, p) -> u^2 - 2, (0.0, 2.0)
)

# =============================================================================
# remake
# =============================================================================

SUITE["remake"] = BenchmarkGroup()

ode_prob = ODEProblem(f_iip, u0, tspan)
u0_new = rand(rng, 20)

SUITE["remake"]["u0"] = @benchmarkable remake($ode_prob; u0 = $u0_new)
SUITE["remake"]["tspan"] = @benchmarkable remake($ode_prob; tspan = (0.0, 2.0))
SUITE["remake"]["p"] = @benchmarkable remake($ode_prob; p = [1.0])

# =============================================================================
# Ensemble machinery
# =============================================================================

SUITE["ensemble"] = BenchmarkGroup()

prob_func(prob, i, repeat) = remake(prob; u0 = u0 .* (1 + 0.01 * i))
SUITE["ensemble"]["construct"] = @benchmarkable EnsembleProblem(
    $ode_prob; prob_func = $prob_func
)

# =============================================================================
# Return codes and traits
# =============================================================================

SUITE["interface"] = BenchmarkGroup()

SUITE["interface"]["successful_retcode"] = @benchmarkable SciMLBase.successful_retcode(
    ReturnCode.Success
)
SUITE["interface"]["isinplace"] = @benchmarkable isinplace($ode_prob)
