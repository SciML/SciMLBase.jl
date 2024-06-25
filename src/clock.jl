@adt TimeDomain begin
    Inferred
    InferredDiscrete
    Continuous
    struct Clock
        t
        dt::Union{Nothing, Float64}
    end
    struct SolverStepClock
        t
    end
    struct IntegerSequence
        t
    end
end

clock(t, dt) = Clock(t, dt)
function clock(arg)
    if symbolic_type(arg) == NotSymbolic()
        return Clock(nothing, arg)
    else
        return Clock(arg, nothing)
    end
end
clock() = Clock(nothing, nothing)
solver_step_clock(t) = SolverStepClock(t)
integer_sequence(t) = IntegerSequence(t)

sampletime(c) = @match c begin
    Clock(_, dt) => dt
    _ => nothing
end
Base.hash(c::TimeDomain, seed::UInt) = @match c begin
    Clock(_, dt) => hash(dt, seed ⊻ 0x953d7a9a18874b90)
    SolverStepClock(_) => seed ⊻ 0x953d7a9a18874b90
    IntegerSequence(_) => seed ⊻ 0x56f6815845619670
    &Inferred => seed ⊻ 0xb7e0b94acc9a8b8d
    &InferredDiscrete => seed ⊻ 0x308e0b97d61e6900
    &Continuous => seed ⊻ 0xfd2a3dfeb13318e5
end

function Base.:(==)(c1::TimeDomain, c2::TimeDomain)
    @match c1 begin
        Clock(t1, dt1) => @match c2 begin
            Clock(t2, dt2) => (t1 === nothing || t2 === nothing || isequal(t1, t2)) && dt1 == dt2
            _ => false
        end
        SolverStepClock(t1) => @match c2 begin
            SolverStepClock(t2) => t1 === nothing || t2 === nothing || isequal(t1, t2)
            _ => false
        end
        IntegerSequence(t1) => @match c2 begin
            IntegerSequence(t2) => t1 === nothing || t2 === nothing || isequal(t1, t2)
            _ => false
        end
        _ => c1 === c2
    end
end

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = IndexedClock(c, idx)

function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock
    
    return @match c begin
        Clock(_, dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
        SolverStepClock(_) => begin
            ssc_idx = findfirst(eachindex(sol.discretes)) do i
                !isa(sol.discretes[i].t, AbstractRange)
            end
            sol.discretes[ssc_idx].t[ic.idx]
        end
        IntegerSequence(_) => ic.idx
        &Continuous => sol.t[ic.idx]
        _ => error("Cannot index $c")
    end
end
