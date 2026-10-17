using SciMLBase, Test, SymbolicIndexingInterface

struct SCCLinearTestSys
    sc::SymbolCache
end
SymbolicIndexingInterface.symbolic_container(s::SCCLinearTestSys) = s.sc
function SciMLBase.get_new_A_b(::SCCLinearTestSys, f, p, A, b; kwargs...)
    A .= p[1]
    return A, b
end

function scc_blocks(container)
    f1 = NonlinearFunction{false}(
        (u, p) -> [1 - u[1]];
        sys = SymbolCache([:x], [:a])
    )
    f2 = NonlinearFunction{false}(
        (u, p) -> [u[1] - p[1], u[2]^2 - 4];
        sys = SymbolCache([:y, :z], [:b])
    )
    prob1 = NonlinearProblem(f1, [0.0], [2.0])
    prob2 = NonlinearProblem(f2, [1.0, 1.5], [3.0])
    probs = container([prob1, prob2])
    funs = container([Returns(nothing), Returns(nothing)])
    return probs, funs
end

@testset "SCCSteadyStateProblem" begin
    @testset "construction" begin
        for wrap in (x -> Tuple(x), identity)
            probs, funs = scc_blocks(wrap)
            sys = SymbolCache([:x, :y, :z], [:a, :b])
            p = [2.0, 3.0]
            prob = SCCSteadyStateProblem(
                probs, funs, p, Val(true); sys = sys
            )

            @test prob isa SciMLBase.AbstractSteadyStateProblem
            @test !SciMLBase.isinplace(prob)
            @test state_values(prob) == [0.0, 1.0, 1.5]
            @test parameter_values(prob) === p
            @test symbolic_container(prob) === prob.f
            @test sprint(show, MIME"text/plain"(), prob) isa String
        end
    end

    @testset "conversions" begin
        probs, funs = scc_blocks(identity)
        sys = SymbolCache([:x, :y, :z])
        p = [2.0, 3.0]
        ssprob = SCCSteadyStateProblem(probs, funs, p, Val(true); sys = sys)

        nlprob = SCCNonlinearProblem(ssprob)
        @test nlprob isa SCCNonlinearProblem
        @test nlprob.probs === probs
        @test nlprob.f === ssprob.f
        @test nlprob.p === p

        ssprob2 = SCCSteadyStateProblem(nlprob)
        @test ssprob2 isa SCCSteadyStateProblem
        @test ssprob2.probs === probs
        @test ssprob2.f === ssprob.f
        @test ssprob2.p === p

        nlprob2 = SCCNonlinearProblem(ssprob2)
        ssprob3 = SCCSteadyStateProblem(nlprob2)
        @test ssprob3.probs === probs
        @test ssprob3.f === ssprob.f
    end

    @testset "symbolic indexing" begin
        probs, funs = scc_blocks(identity)
        sys = SymbolCache([:x, :y, :z], [:a, :b])
        p = [2.0, 3.0]
        prob = SCCSteadyStateProblem(probs, funs, p, Val(true); sys = sys)

        @test getu(prob, :y)(prob) == 1.0
        @test getu(prob, :x)(prob) == 0.0
        @test getsym(prob, :z)(prob) == 1.5

        # index 2 lies in block 2
        set_state!(prob, 9.0, 2)
        @test state_values(prob) == [0.0, 9.0, 1.5]

        # alias true: updating `prob`'s parameters stops at the shared object
        set_parameter!(prob, 7.0, 1)
        @test parameter_values(prob)[1] == 7.0
        @test parameter_values(prob.probs[1])[1] == 2.0
    end

    @testset "set_parameter! without alias" begin
        probs, funs = scc_blocks(identity)
        sys = SymbolCache([:x, :y, :z], [:a, :b])
        p = [2.0, 3.0]
        prob = SCCSteadyStateProblem(probs, funs, p, Val(false); sys = sys)

        set_parameter!(prob, 8.0, 1)
        @test parameter_values(prob)[1] == 8.0
        # index 1 is a valid parameter index in both blocks' local parameter spaces
        @test parameter_values(prob.probs[1])[1] == 8.0
        @test parameter_values(prob.probs[2])[1] == 8.0
    end

    @testset "remake" begin
        for wrap in (x -> Tuple(x), identity)
            probs, funs = scc_blocks(wrap)
            sys = SymbolCache([:x, :y, :z], [:a, :b])
            p = [2.0, 3.0]
            prob = SCCSteadyStateProblem(probs, funs, p, Val(true); sys = sys)

            prob2 = remake(prob; u0 = [5.0, 6.0, 7.0])
            @test prob2 isa SCCSteadyStateProblem
            @test state_values(prob2) == [5.0, 6.0, 7.0]
            @test state_values(prob2.probs[1]) == [5.0]
            @test state_values(prob2.probs[2]) == [6.0, 7.0]

            # parameters_alias pushes the new p into every block
            newp = [4.0, 5.0]
            prob3 = remake(prob; p = newp)
            @test prob3 isa SCCSteadyStateProblem
            @test parameter_values(prob3) == newp
            @test parameter_values(prob3.probs[1]) == newp
            @test parameter_values(prob3.probs[2]) == newp
        end

        # parameters_alias false + p without probs throws
        probs, funs = scc_blocks(identity)
        prob = SCCSteadyStateProblem(
            probs, funs, [2.0, 3.0], Val(false);
            sys = SymbolCache([:x, :y, :z], [:a, :b])
        )
        @test_throws ArgumentError remake(prob; p = [9.0, 9.0])
    end

    @testset "build_solution" begin
        probs, funs = scc_blocks(identity)
        sys = SymbolCache([:x, :y, :z])
        prob = SCCSteadyStateProblem(probs, funs, [2.0, 3.0]; sys = sys)

        u = [1.0, 2.0, 3.0]
        resid = [0.1, 0.2, 0.3]
        sol = SciMLBase.build_solution(prob, nothing, u, resid)
        @test sol isa SteadyStateSolution
        @test sol.prob === prob
        @test sol.u == u
        @test sol.resid == resid
        @test sol[:y] == 2.0
        @test sol[[:z, :x]] == [3.0, 1.0]
        @test sol[:(x + z)] == 4.0
    end

    @testset "remake parameters of guess-less linear blocks" begin
        for T in (SCCSteadyStateProblem, SCCNonlinearProblem),
                wrap in (x -> Tuple(x), identity)
            p = [1.0]
            lin = LinearProblem(
                [1.0;;], [4.0], p;
                f = SciMLBase.SymbolicLinearInterface(;
                    update_Ab = nothing,
                    sys = SCCLinearTestSys(SymbolCache([:x], [:k])),
                    observed = nothing, metadata = nothing
                )
            )
            prob = T(wrap([lin]), wrap([Returns(nothing)]), p, Val(true))

            prob2 = remake(prob; p = [2.0])
            @test prob2 isa T
            @test parameter_values(prob2) == [2.0]
            @test parameter_values(prob2.probs[1]) === parameter_values(prob2)
            @test prob2.probs[1].A == [2.0;;]
            @test prob2.probs[1].A \ prob2.probs[1].b ≈ [2.0]
            @test state_values(prob2) === nothing
            @test prob.probs[1].A == [1.0;;]

            prob3 = remake(prob; u0 = [3.0])
            @test state_values(prob3) == [3.0]
            @test state_values(prob3.probs[1]) == [3.0]
        end
    end

    @testset "mixed guess and guess-less blocks" begin
        for T in (SCCSteadyStateProblem, SCCNonlinearProblem),
                wrap in (x -> Tuple(x), identity)
            lin = LinearProblem(Float64[1 0; 0 1], [1.0, 2.0])
            nl = NonlinearProblem(
                NonlinearFunction{false}((u, p) -> u .- 3), [3.0], nothing
            )
            funs = () -> wrap([Returns(nothing), Returns(nothing)])
            @test_throws ArgumentError T(wrap([lin, nl]), funs())
            @test_throws ArgumentError T(wrap([nl, lin]), funs())

            linu = LinearProblem(Float64[1 0; 0 1], [1.0, 2.0]; u0 = [0.5, 0.25])
            for (blocks, expected) in (
                    ([linu, nl], [0.5, 0.25, 3.0]), ([nl, linu], [3.0, 0.5, 0.25]),
                )
                u = state_values(T(wrap(blocks), funs()))
                @test u == expected
                @test eltype(u) == Float64
            end
        end
    end

    @testset "all-LinearProblem blocks" begin
        A1 = Float64[1 0; 0 1]
        b1 = [1.0, 2.0]
        A2 = Float64[2 0; 0 2]
        b2 = [4.0, 6.0]
        prob1 = LinearProblem(A1, b1)
        prob2 = LinearProblem(A2, b2)
        prob = SCCSteadyStateProblem(
            [prob1, prob2], [Returns(nothing), Returns(nothing)]
        )
        @test prob isa SCCSteadyStateProblem{Nothing}
        @test state_values(prob) === nothing
    end
end
