@adt TimeDomain begin
    Inferred
    InferredDiscrete
    Continuous
    struct PeriodicClock
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

"""
    clock(t, dt)
    clock(t)
    clock(dt)
    clock()

The default periodic clock with symbolic independent variable `t` and tick interval `dt`.
If `dt` is left unspecified, it will be inferred (if possible). If `t` is not specified,
it is `nothing`.
"""
clock(t, dt) = PeriodicClock(t, dt)
function clock(arg)
    if symbolic_type(arg) == NotSymbolic()
        return PeriodicClock(nothing, arg)
    else
        return PeriodicClock(arg, nothing)
    end
end
clock() = PeriodicClock(nothing, nothing)

"""
    solver_step_clock(t)
    solver_step_clock()

A clock that ticks at each solver step (sometimes referred to as "continuous sample time").
This clock **does generally not have equidistant tick intervals**, instead, the tick
interval depends on the adaptive step-size selection of the continuous solver, as well as
any continuous event handling. If adaptivity of the solver is turned off and there are no
continuous events, the tick interval will be given by the fixed solver time step `dt`.

Due to possibly non-equidistant tick intervals, this clock should typically not be used with
discrete-time systems that assume a fixed sample time, such as PID controllers and digital
filters.
"""
solver_step_clock(t) = SolverStepClock(t)
solver_step_clock() = SolverStepClock(nothing)
integer_sequence(t) = IntegerSequence(t)
integer_sequence() = IntegerSequence(nothing)

function is_discrete_time_domain(c::TimeDomain)
    return @match c begin
        PeriodicClock(_...) => true
        SolverStepClock(_...) => true
        IntegerSequence(_...) => true
        &InferredDiscrete => true
        _ => false
    end
end

function is_concrete_time_domain(c)
    return @match c begin
        PeriodicClock(_...) => true
        SolverStepClock(_...) => true
        IntegerSequence(_...) => true
        &Continuous => true
        _ => false
    end
end

Base.hash(c::TimeDomain, seed::UInt) = @match c begin
    PeriodicClock(_, dt) => hash(dt, seed ⊻ 0x953d7a9a18874b90)
    SolverStepClock(_) => seed ⊻ 0x953d7a9a18874b90
    IntegerSequence(_) => seed ⊻ 0x56f6815845619670
    &Inferred => seed ⊻ 0xb7e0b94acc9a8b8d
    &InferredDiscrete => seed ⊻ 0x308e0b97d61e6900
    &Continuous => seed ⊻ 0xfd2a3dfeb13318e5
end

function Base.:(==)(c1::TimeDomain, c2::TimeDomain)
    @match c1 begin
        PeriodicClock(t1, dt1) => @match c2 begin
            PeriodicClock(t2, dt2) => (t1 === nothing || t2 === nothing || isequal(t1, t2)) && dt1 == dt2
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
        PeriodicClock(_, dt) => ceil(sol.prob.tspan[1] / dt) * dt .+ (ic.idx .- 1) .* dt
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
