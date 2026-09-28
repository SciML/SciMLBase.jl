using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqBDF: DFBDF
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
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
        ps -> dae_loss(sol -> sum(abs2, reduce(hcat, sol[x])), ps),
        ps -> dae_loss(sol -> sum(abs2, reduce(hcat, [sol[x[1]], sol[x[2]]])), ps),
    )
    for loss in losses
        @test Zygote.gradient(loss, [1.0, 2.0])[1] ≈ ForwardDiff.gradient(loss, [1.0, 2.0])
    end
end
