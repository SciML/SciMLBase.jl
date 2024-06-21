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

function Clock(arg = nothing)
    if symbolic_type(arg) == NotSymbolic()
        return Clock(nothing, arg)
    else
        return Clock(arg, nothing)
    end
end

sampletime(c) = @match c begin
    Clock(; dt) = dt
    _ => nothing
end
Base.hash(c::TimeDomain, seed::UInt) = @match c begin
    Clock(; dt) => hash(dt, seed ⊻ 0x953d7a9a18874b90)
    SolverStepClock(;) => seed ⊻ 0x953d7a9a18874b90
    IntegerSequence(;) => seed ⊻ 0x56f6815845619670
    Inferred => seed ⊻ 0xb7e0b94acc9a8b8d
    InferredDiscrete => seed ⊻ 0x308e0b97d61e6900
    Continuous => seed ⊻ 0xfd2a3dfeb13318e5
end

function Base.:(==)(c1::TimeDomain, c2::TimeDomain)
    @match c1 begin
        &Clock => @match c2 begin
            &Clock => (c1.t === nothing || c2.t === nothing || isequal(c1.t, c2.t)) && c1.dt == c2.dt
            _ => false
        end
        &SolverStepClock => @match c2 begin
            &SolverStepClock => c1.t === nothing || c2.t === nothing || isequal(c1.t, c2.t)
            _ => false
        end
        &IntegerSequence => @match c2 begin
            &IntegerSequence => c1.t === nothing || c2.t === nothing || isequal(c1.t, c2.t)
            _ => false
        end
        _ => c1 == c2
    end
end

Base.Broadcast.broadcastable(d::TimeDomain) = Ref(d)

struct IndexedClock{I}
    clock::TimeDomain
    idx::I
end

Base.getindex(c::TimeDomain, idx) = Indexedclock(c, idx)

function canonicalize_indexed_clock(ic::IndexedClock, sol::AbstractTimeseriesSolution)
    c = ic.clock
    return @match c begin
        &Clock => ic.idx .* c.dt
        &SolverStepClock => sol.t[ic.idx]
        &IntegerSequence => ic.idx
        &Continuous => sol.t[ic.idx]
        _ => error("Cannot index $c")
    end
end
