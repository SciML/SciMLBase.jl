using SciMLBase, Test
using Zygote

# The `getproperty(::NonlinearProblem)` rrule must emit full-width cotangents.
# Partial NamedTuples like `(p = dp,)` cannot be `Zygote.accum`ed with
# cotangents for the same problem coming from other pullbacks, throwing
# `ArgumentError: ... keys must be a subset of ... keys`.
# https://github.com/SciML/SciMLBase.jl test/downstream/remake_autodiff.jl hit
# this via `remake(::ODEProblem)` -> trivial initialization of the MTK
# initialization `NonlinearProblem`.

fnl(u, p) = u .^ 2 .- p

@testset "getproperty(::NonlinearProblem) cotangent shape" begin
    prob = NonlinearProblem{false}(NonlinearFunction{false}(fnl), [1.0, 2.0], [3.0, 4.0])
    _, back = Zygote.pullback(p -> getproperty(p, :p), prob)
    dprob = back(ones(2))[1]
    @test dprob isa NamedTuple
    @test keys(dprob) == fieldnames(NonlinearProblem)
    @test dprob.p == ones(2)
end

@testset "accumulating cotangents from different fields" begin
    prob = NonlinearProblem{false}(NonlinearFunction{false}(fnl), [1.0, 2.0], [3.0, 4.0])

    # reads two different fields of the same problem (remake reads kwargs,
    # problem_type, lb, ub; user code reads p)
    function loss(p)
        prob2 = NonlinearProblem{false}(NonlinearFunction{false}(fnl), [1.0, 2.0], p)
        prob3 = SciMLBase.remake(prob2; u0 = [2.0, 3.0])
        return sum(prob3.p) + sum(prob2.p)
    end
    @test Zygote.gradient(loss, [3.0, 4.0])[1] == [2.0, 2.0]

    # two remakes of the same problem
    function loss2(p)
        prob2 = NonlinearProblem{false}(NonlinearFunction{false}(fnl), [1.0, 2.0], p)
        prob3 = SciMLBase.remake(prob2; u0 = [2.0, 3.0])
        prob4 = SciMLBase.remake(prob2; u0 = [4.0, 5.0])
        return sum(prob3.p) + sum(prob4.p)
    end
    @test Zygote.gradient(loss2, [3.0, 4.0])[1] == [2.0, 2.0]
end
