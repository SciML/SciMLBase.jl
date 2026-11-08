using SciMLBase, AllocCheck, Test

# Records which callbacks save, in order, and the `skip_duplicates` each save receives.
struct DiscreteSaveRecorder <: SciMLBase.DEIntegrator{Nothing, true, Vector{Float64}, Float64}
    saves::Vector{Tuple{UInt, Bool}}
end
function SciMLBase.save_discretes!(
        integ::DiscreteSaveRecorder, cb::Union{ContinuousCallback, DiscreteCallback};
        skip_duplicates = false
    )
    push!(integ.saves, (objectid(cb), skip_duplicates))
    return nothing
end

save_all!(integ, cbs) = SciMLBase.save_discretes_if_enabled!(integ, cbs; skip_duplicates = true)

@testset "save_discretes_if_enabled! over a CallbackSet of different callback types" begin
    c = ContinuousCallback((u, t, i) -> u[1], i -> nothing; save_positions = (false, true))
    d1 = DiscreteCallback((u, t, i) -> false, i -> nothing; save_positions = (false, true))
    d2 = DiscreteCallback((u, t, i) -> true, i -> nothing; save_positions = (false, false))
    expected = [(objectid(c), true), (objectid(d1), true)]
    cbs = CallbackSet(c, d1, d2)
    integ = DiscreteSaveRecorder(Tuple{UInt, Bool}[])
    save_all!(integ, cbs)
    @test integ.saves == expected
    allocs = check_allocs(save_all!, (DiscreteSaveRecorder, typeof(cbs)); ignore_throw = true)
    @test !any(a -> a isa AllocCheck.DynamicDispatch, allocs)
    # the same callbacks held in vectors
    vinteg = DiscreteSaveRecorder(Tuple{UInt, Bool}[])
    save_all!(vinteg, CallbackSet(Any[c], Any[d1, d2]))
    @test vinteg.saves == expected
end
