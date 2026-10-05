using StochasticDiffEq, SciMLBase, SciMLSensitivity, Test
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff

# Reverse-mode Enzyme through an EnsembleProblem of SDEProblems
# (SciML/SciMLSensitivity.jl#1696). batch_func used to join the safetycopy
# deepcopy with the caller's problem into one problem, a pointer phi Enzyme
# cannot keep rooted. Each trajectory gets a fixed noise seed through
# prob_func, so ForwardDiff solves the same trajectories and is a reference.
# safetycopy = false keeps the default no-copy path that a custom prob_func
# would otherwise turn off; Enzyme does not yet differentiate the deepcopy of
# an SDEProblem itself.
function lotka_volterra!(du, u, p, t)
    x, y = u
    du[1] = p[1] * x - p[2] * x * y
    du[2] = p[4] * x * y - p[3] * y
    return nothing
end
function multiplicative_noise!(du, u, p, t)
    du[1] = p[5] * u[1]
    du[2] = p[6] * u[2]
    return nothing
end
prob = SDEProblem(
    lotka_volterra!, multiplicative_noise!, [1.0, 1.0], (0.0, 1.0),
    [1.5, 1.0, 3.0, 1.0, 0.3, 0.3]
)
seeded(prob, ctx) = remake(prob; seed = UInt64(1000 + ctx.sim_id))
p0 = [1.2, 0.8, 2.5, 0.8, 0.1, 0.1]

@testset "Enzyme reverse over EnsembleProblem of SDEProblems: $(nameof(typeof(ensemblealg)))" for ensemblealg in
    (EnsembleSerial(), EnsembleThreads())
    loss(p) = sum(
        Array(
            solve(
                EnsembleProblem(remake(prob; p); prob_func = seeded, safetycopy = false), SOSRI(), ensemblealg;
                saveat = 0.1, trajectories = 3
            )
        )
    )
    g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss, p0)[1]
    @test g ≈ ForwardDiff.gradient(loss, p0) rtol = 1.0e-8
end
