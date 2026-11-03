using SciMLBase, Test
using SciMLBase: AutoDespecialize, AutoSpecialize, DespecializedParameters,
    FullSpecialize, NoAD, specialization
using ConstructionBase: setproperties

struct OptimizationSpecializationParams
    target::Vector{Float64}
end

objective(x, p) = sum(abs2, x .- p.target)
gradient!(G, x, p) = (G .= 2 .* (x .- p.target); nothing)

@testset "specialization marker" begin
    @test specialization(OptimizationFunction(objective)) === FullSpecialize
    @test specialization(OptimizationFunction{true}(objective, NoAD())) === FullSpecialize
    @test specialization(OptimizationFunction) === FullSpecialize
    for spec in (FullSpecialize, AutoSpecialize, AutoDespecialize)
        f = OptimizationFunction{true, spec}(objective)
        @test specialization(f) === spec
        @test specialization(typeof(f)) === spec
        @test f.adtype === NoAD()
        @test SciMLBase.isinplace(f)
    end
    # the AD type is the second type parameter, not the marker
    f = OptimizationFunction{true, AutoDespecialize}(objective, NoAD())
    @test typeof(f).parameters[2] === NoAD
end

@testset "constructors keep the marker" begin
    f = OptimizationFunction{true, AutoSpecialize}(objective, NoAD(); grad = gradient!)
    @test f.grad === gradient!
    # adtype as a keyword, the spelling `remake` and `ConstructionBase` rebuild with
    @test specialization(OptimizationFunction{true, AutoSpecialize}(objective; adtype = NoAD())) ===
        AutoSpecialize
    # the old spelling with the AD type as the second type parameter
    old = OptimizationFunction{true, NoAD}(objective; adtype = NoAD())
    @test specialization(old) === FullSpecialize
    @test old.adtype === NoAD()

    remade = remake(f; expr = :(a + b))
    @test specialization(remade) === AutoSpecialize
    @test remade.expr == :(a + b)
    @test remade.grad === gradient!

    set = setproperties(f, (; grad = nothing))
    @test specialization(set) === AutoSpecialize
    @test set.grad === nothing
    @test set.f === objective
end

@testset "problems keep the marker" begin
    p = OptimizationSpecializationParams([1.0, 2.0])
    prob = OptimizationProblem(
        OptimizationFunction{true, AutoDespecialize}(objective), [0.0, 0.0], p
    )
    @test specialization(prob.f) === AutoDespecialize
    @test specialization(remake(prob; u0 = [1.0, 1.0]).f) === AutoDespecialize
end

@testset "calls with despecialized parameters" begin
    f = OptimizationFunction{true, AutoDespecialize}(objective)
    p = OptimizationSpecializationParams([1.0, 2.0])
    @test f([1.0, 2.0], DespecializedParameters(p)) == 0.0
    @test f([0.0, 0.0], DespecializedParameters(p)) == f([0.0, 0.0], p)
end
