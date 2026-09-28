using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqBDF: DFBDF
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
using StochasticDiffEq: SOSRI
using Zygote: Zygote

# Manual cotangent for `loss(sol) = sol[idx]^2` under VectorOfArray layout:
# scatter `2 * sol[idx]` into the matching state/time slot of `sol.u`.
function integer_getindex_square_grad(sol, idx)
    inds = idx isa CartesianIndex ? Tuple(idx) : Tuple(CartesianIndices(size(sol))[idx])
    front_inds = Base.front(inds)
    step_idx = last(inds)
    val = sol[idx]
    return map(enumerate(sol.u)) do (k, x)
        if k == step_idx
            δu = zero(x)
            δu[front_inds...] = 2 * val
            δu
        else
            zero(x)
        end
    end
end

# Normalize Zygote cotangents: AbstractArray layout is (state × time); the
# SciML Integer adjoint returns a length-ntime vector of state vectors.
function normalize_sol_grad(g, sol)
    if g isa AbstractMatrix
        return [g[:, j] for j in axes(g, 2)]
    elseif hasproperty(g, :u)
        return normalize_sol_grad(g.u, sol)
    else
        return g
    end
end

function zygote_u_grad(loss, sol)
    _, back = Zygote.pullback(loss, sol)
    return normalize_sol_grad(back(1)[1], sol)
end

@testset "Symbolic solution indexing agrees with ForwardDiff (#1594)" begin
    @variables x(t)[1:2]
    @parameters p[1:2] = [1.0, 2.0]
    @mtkcompile sys = System(
        [D(x[1]) ~ -p[1] * x[1], D(x[2]) ~ -p[2] * x[2]], t
    )
    prob = ODEProblem(sys, [x => [1.0, 1.0]], (0.0, 1.0))
    set_p = SymbolicIndexingInterface.setp_oop(prob, p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    ode_loss(sym, ps) = sum(
        abs2, reduce(
            hcat, solve(
                remake(prob; p = set_p(prob, ps)), Tsit5();
                saveat = 0.1, reltol = 1.0e-10, abstol = 1.0e-12, sensealg
            )[sym]
        )
    )

    @mtkcompile dsys = System(
        [0 ~ -p[1] * x[1] - D(x[1]), 0 ~ -p[2] * x[2] - D(x[2])], t
    )
    dprob = DAEProblem(dsys, [x => [1.0, 1.0]], (0.0, 1.0))
    set_dp = SymbolicIndexingInterface.setp_oop(dprob, p)
    dae_loss(f, ps) = f(
        solve(
            remake(dprob; p = set_dp(dprob, ps)), DFBDF();
            saveat = 0.1, reltol = 1.0e-10, abstol = 1.0e-12
        )
    )

    losses = (
        ps -> ode_loss(x, ps),
        ps -> ode_loss(x[1], ps),
        ps -> dae_loss(sol -> sum(abs2, reduce(hcat, sol[x])), ps),
        ps -> dae_loss(sol -> sum(abs2, reduce(hcat, [sol[x[1]], sol[x[2]]])), ps),
    )
    for loss in losses
        @test Zygote.gradient(loss, [1.0, 2.0])[1] ≈ ForwardDiff.gradient(loss, [1.0, 2.0])
    end
end

@testset "DAE/RODE integer solution indexing under Zygote (#1594)" begin
    # Non-symbolic indexing of DAESolution / RODESolution must not be captured by
    # the symbolic AbstractODESolution getindex adjoint.
    @variables x(t)[1:2]
    @parameters p[1:2] = [1.0, 2.0]
    @mtkcompile dsys = System(
        [0 ~ -p[1] * x[1] - D(x[1]), 0 ~ -p[2] * x[2] - D(x[2])], t
    )
    dprob = DAEProblem(dsys, [x => [1.0, 1.0]], (0.0, 1.0))
    dsol = solve(dprob, DFBDF(); saveat = 0.1, reltol = 1.0e-10, abstol = 1.0e-12)

    f!(du, u, p, t) = (du[1] = -p[1] * u[1]; du[2] = -p[2] * u[2])
    g!(du, u, p, t) = (du .= 0)
    sprob = SDEProblem(f!, g!, [1.0, 1.0], (0.0, 1.0), [1.0, 2.0])
    ssol = solve(sprob, SOSRI(); saveat = 0.1, seed = 1)

    for sol in (dsol, ssol)
        @test sol isa SciMLBase.AbstractODESolution
        @test !(sol isa ODESolution)
        for idx in (3, lastindex(sol), CartesianIndex(2, 3))
            loss = s -> s[idx]^2
            @test zygote_u_grad(loss, sol) ≈ integer_getindex_square_grad(sol, idx)
        end
    end
end
