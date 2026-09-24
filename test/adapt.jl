using Adapt, SciMLBase, Test

f(u, p) = u .* u .- p
u0 = [1.0f0, 1.0f0]

for iip in (true, false)
    prob = SciMLBase.ImmutableNonlinearProblem{iip}(f, u0, 2.0f0)
    adapted = Adapt.adapt(nothing, prob)
    @test typeof(prob) === typeof(adapted)
    @test SciMLBase.isinplace(adapted) === iip
end
