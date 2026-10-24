using SciMLBase, AllocCheck, Test

struct DiscreteSaveCounter <: SciMLBase.DEIntegrator{Nothing, true, Vector{Float64}, Float64}
    count::Base.RefValue{Int}
end
function SciMLBase.save_discretes!(
        integ::DiscreteSaveCounter, ::Union{ContinuousCallback, DiscreteCallback};
        skip_duplicates = false
    )
    integ.count[] += 1
    return nothing
end

save_all!(integ, cbs) = SciMLBase.save_discretes_if_enabled!(integ, cbs; skip_duplicates = true)

@testset "save_discretes_if_enabled! over a CallbackSet of different callback types" begin
    cbs = CallbackSet(
        ContinuousCallback((u, t, i) -> u[1], i -> nothing; save_positions = (false, true)),
        DiscreteCallback((u, t, i) -> false, i -> nothing; save_positions = (false, true)),
        DiscreteCallback((u, t, i) -> true, i -> nothing; save_positions = (false, false))
    )
    integ = DiscreteSaveCounter(Ref(0))
    save_all!(integ, cbs)
    @test integ.count[] == 2
    allocs = check_allocs(save_all!, (DiscreteSaveCounter, typeof(cbs)); ignore_throw = true)
    @test !any(a -> a isa AllocCheck.DynamicDispatch, allocs)
end
