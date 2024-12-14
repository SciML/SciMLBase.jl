using Test, SciMLBase, SymbolicIndexingInterface, Accessors

function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = u[1][1] + pi / 2
    residual[2] = u[end][1] - pi / 2
end
prob_bvp = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], (0, 1.0))
@test prob_bvp.tspan === (0.0, 1.0)

@testset "`constructorof` tests" begin
    probs = []

    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = [1.0; 2.0; 3.0]
    du0 = similar(u0)
    p = [10.0, 20.0, 30.0]
    tspan = (0.0, 100.0)
    lorenz!(du0, u0, p, tspan[1])
    sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t)
    fn = ODEFunction(lorenz!; sys)
    push!(probs, ODEProblem(fn, u0, tspan, p))

    function daelorenz!(resid, du, u, p, t)
        lorenz!(resid, u, p, t)
        resid .-= du
    end
    fn = DAEFunction(daelorenz!; sys)
    push!(probs, DAEProblem(fn, du0, u0, tspan, p))

    function ddelorenz!(du, u, h, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end

    function history(p, t)
        return u0 .- t
    end

    fn = DDEFunction(ddelorenz!; sys)
    push!(probs, DDEProblem(fn, u0, history, tspan, p))

    function noise!(du, u, p, t)
        du .= 0.1u
    end
    fn = SDEFunction(lorenz!, noise!; sys)
    push!(probs, SDEProblem(fn, u0, tspan, p))

    fn = SDDEFunction(ddelorenz!, noise!; sys)
    push!(probs, SDDEProblem(fn, noise!, u0, history, tspan, p))

    function nllorenz!(du, u, p)
        lorenz!(du, u, p, 0.0)
    end

    fn = NonlinearFunction(nllorenz!; sys)
    push!(probs, NonlinearProblem(fn, u0, p))
    push!(probs, NonlinearLeastSquaresProblem(fn, u0, p))

    @testset "$(SciMLBase.parameterless_type(typeof(prob)))" for prob in probs
        newprob = @reset prob.u0 = u0 .+ 1
        @test typeof(newprob) == typeof(prob)
    end
end
