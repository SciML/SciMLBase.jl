using ModelingToolkit, OrdinaryDiffEq, SymbolicIndexingInterface, Test, ForwardDiff, SciMLBase
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqBDF: DFBDF
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
using StochasticDiffEq: SOSRI
using ChainRulesCore: ChainRulesCore
using Zygote: Zygote

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
        ps -> dae_loss(sol -> sum(abs2, sol[x[2]]), ps),
        ps -> dae_loss(sol -> sum(abs2, reduce(hcat, [sol[x[1]], sol[x[2]]])), ps),
    )
    for loss in losses
        @test Zygote.gradient(loss, [1.0, 2.0])[1] ≈ ForwardDiff.gradient(loss, [1.0, 2.0])
    end
end

# Non-symbolic indices on DAE/RODE solutions must keep the generic `AbstractArray`
# rules: a correct gradient, or an error, never a different answer.
@testset "Non-symbolic indexing of DAE/RODE solutions under AD" begin
    f(u, p, t) = -p .* u
    fd(du, u, p, t) = du .+ p .* u
    dprob = DAEProblem(
        fd, -[1.0, 4.0], [1.0, 2.0], (0.0, 1.0), [1.0, 2.0];
        differential_vars = [true, true]
    )
    dsol = solve(dprob, DFBDF(); saveat = 0.25)
    g(u, p, t) = 0.1 .* u
    sprob = SDEProblem(f, g, [1.0, 2.0], (0.0, 1.0), [1.0, 2.0])
    ssol = solve(sprob, SOSRI(); saveat = 0.25, seed = 1)

    idxs = (
        (3,), (10,), (CartesianIndex(2, 3),), (1:2,), ([1, 3],), (2:2:4,),
        (trues(10),), (:,), (:, 2), (1, 2), (1:2, 3), (1, :),
    )
    loss(idx) = s -> sum(abs2, Array(s[idx...]))
    ones_like(y::Number) = 1.0
    ones_like(y) = ones(size(y))
    grad_matrix(sol, g::AbstractMatrix) = Array(g)
    grad_matrix(sol, g::AbstractVector{<:AbstractVector}) = reduce(hcat, g)
    grad_matrix(sol, g::NamedTuple) = grad_matrix(sol, g.u)
    # Central differences of `loss` in each saved state value. The losses are
    # quadratic in `sol.u`, so this is exact up to roundoff.
    function fd_grad(loss, sol; h = 1.0e-4)
        G = zeros(size(Array(sol)))
        for j in eachindex(sol.u), k in eachindex(sol.u[j])
            sp, sm = deepcopy(sol), deepcopy(sol)
            sp.u[j][k] += h
            sm.u[j][k] -= h
            G[k, j] = (loss(sp) - loss(sm)) / 2h
        end
        return G
    end
    function outcome(thunk)
        return try
            thunk()
        catch e
            e
        end
    end

    for sol in (dsol, ssol), idx in idxs
        @test sol isa SciMLBase.AbstractODESolution && !(sol isa ODESolution)

        # ChainRules: the rule applied is exactly the generic `AbstractArray` one.
        generic = outcome(
            () -> invoke(
                ChainRulesCore.rrule,
                Tuple{typeof(getindex), AbstractArray, map(typeof, idx)...},
                getindex, sol, idx...
            )
        )
        rule = outcome(() -> ChainRulesCore.rrule(getindex, sol, idx...))
        if generic isa Exception
            @test typeof(rule) == typeof(generic)
        else
            @test rule[1] == generic[1]
            dgeneric = outcome(() -> ChainRulesCore.unthunk.(generic[2](ones_like(generic[1]))))
            drule = outcome(() -> ChainRulesCore.unthunk.(rule[2](ones_like(rule[1]))))
            if dgeneric isa Exception
                @test typeof(drule) == typeof(dgeneric)
            else
                @test drule == dgeneric
            end
        end

        # Zygote: the gradient is the true derivative of the loss, or an error.
        l = loss(idx)
        primal = outcome(() -> l(sol))
        zg = outcome(() -> Zygote.gradient(l, sol)[1])
        if primal isa Exception
            @test zg isa Exception
        elseif !(zg isa Exception)
            @test grad_matrix(sol, zg) ≈ fd_grad(l, sol) atol = 1.0e-6
        end
    end
end
