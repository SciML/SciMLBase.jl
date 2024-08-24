using SciMLBase
using SymbolicIndexingInterface
using StaticArrays
using ForwardDiff

probs = []
containerTypes = [Vector, Tuple, SVector{3}, MVector{3}, SizedVector{3}]
# ODE
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end
u0 = [1.0; 2.0; 3.0]
tspan = (0.0, 100.0)
p = [10.0, 20.0, 30.0]
sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t)
fn = ODEFunction(lorenz!; sys)
for T in containerTypes
    push!(probs, ODEProblem(fn, u0, tspan, T(p)))
end

function residual!(resid, u, p, t)
    resid[1] = u[1] - 0.5
    resid[2] = u[2] - 0.5
    resid[3] = u[3] - 0.5
end
fn = BVPFunction(lorenz!, residual!; sys)
for T in containerTypes
    push!(probs, BVProblem(fn, u0, tspan, T(p)))
end

function noise!(du, u, p, t)
    du .= 0.1u
end
fn = SDEFunction(lorenz!, noise!; sys)
for T in containerTypes
    push!(probs, SDEProblem(fn, u0, tspan, T(p)))
end

function loss(x, p)
    du = similar(x)
    lorenz!(du, u, p, 0.0)
    return sum(du)
end

fn = OptimizationFunction(loss; sys)
for T in containerTypes
    push!(probs, OptimizationProblem(fn, u0, T(p)))
end

function nllorenz!(du, u, p)
    lorenz!(du, u, p, 0.0)
end

fn = NonlinearFunction(nllorenz!; sys)
for T in containerTypes
    push!(probs, NonlinearProblem(fn, u0, T(p)))
end

for T in containerTypes
    push!(probs, NonlinearLeastSquaresProblem(fn, u0, T(p)))
end

for prob in deepcopy(probs)
    prob2 = @inferred remake(prob)
    @test prob2.u0 == u0
    @test prob2.p == typeof(prob.p)(p)
    baseType = Base.typename(typeof(prob)).wrapper
    for T in containerTypes
        if T !== Tuple
            local u0 = T([2.0, 3.0, 4.0])
            prob2 = @inferred baseType remake(prob; u0 = deepcopy(u0))
            @test prob2.u0 == u0
            @test prob2.u0 isa T
        end
        local p = T([11.0, 12.0, 13.0])
        prob2 = @inferred baseType remake(prob; p = deepcopy(p))
        @test prob2.p == p
        @test prob2.p isa T
    end

    for T in [Float32, Float64]
        local u0 = [:x => T(2.0), :z => T(4.0), :y => T(3.0)]
        prob2 = @inferred baseType remake(prob; u0)
        @test all(prob2.u0 .≈ T[2.0, 3.0, 4.0])
        @test eltype(prob2.u0) == T

        local u0 = [:x => T(2.0)]
        prob2 = @inferred baseType remake(prob; u0)
        @test all(prob2.u0 .≈ [2.0, 2.0, 3.0])
        @test eltype(prob2.u0) == Float64 # partial update promotes, since fallback is Float64

        local p = [:a => T(11.0), :b => T(12.0), :c => T(13.0)]
        prob2 = @inferred baseType remake(prob; p)
        @test all(prob2.p .≈ T[11.0, 12.0, 13.0])
        @test eltype(prob2.p) == T

        local p = [:a => T(11.0)]
        prob2 = @inferred baseType remake(prob; p)
        @test all(prob2.p .≈ [11.0, 20.0, 30.0])
        if prob.p isa Tuple
            @test prob2.p isa Tuple{T, Float64, Float64}
        else
            @test eltype(prob2.p) == Float64
        end
    end

    # constant defaults
    begin
        prob.f.sys.defaults[:a] = 0.1
        prob.f.sys.defaults[:x] = 0.1
        # remake with no updates should use existing values
        prob2 = @inferred baseType remake(prob)
        @test prob2.u0 == u0
        @test prob2.p == typeof(prob.p)(p)

        # not passing use_defaults ignores defaults
        prob2 = @inferred baseType remake(prob; u0 = [:y => 0.2])
        @test prob2.u0 == [1.0, 0.2, 3.0]
        @test prob2.p == typeof(prob.p)(p)

        # respect defaults (:x), fallback to existing value (:z)
        prob2 = @inferred baseType remake(prob; u0 = [:y => 0.2], use_defaults = true)
        @test prob2.u0 ≈ [0.1, 0.2, 3.0]
        @test prob2.p == typeof(prob.p)(p) # params unaffected

        # override defaults
        prob2 = @inferred baseType remake(prob; u0 = [:x => 0.2], use_defaults = true)
        @test prob2.u0 ≈ [0.2, 2.0, 3.0]
        @test prob2.p == typeof(prob.p)(p)

        prob2 = @inferred baseType remake(prob; p = [:b => 0.2], use_defaults = true)
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [0.1, 0.2, 30.0])

        prob2 = @inferred baseType remake(prob; p = [:a => 0.2], use_defaults = true)
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [0.2, 20.0, 30.0])

        empty!(prob.f.sys.defaults)
    end

    # dependent defaults
    begin
        prob.f.sys.defaults[:b] = :(3a)
        prob.f.sys.defaults[:y] = :(3x)
        prob.f.sys.defaults[:z] = 9.0
        prob.f.sys.defaults[:c] = 0.9
        # remake with no updates should use existing values
        prob2 = @inferred baseType remake(prob)
        @test prob2.u0 == u0
        @test prob2.p == typeof(prob.p)(p)

        prob2 = @inferred baseType remake(prob; u0 = [:x => 0.2])
        @test prob2.u0 ≈ [0.2, 0.6, 3.0]
        @test prob2.p == typeof(prob.p)(p)

        # respect numeric defaults (:z)
        prob2 = @inferred baseType remake(prob; u0 = [:x => 0.2], use_defaults = true)
        @test prob2.u0 ≈ [0.2, 0.6, 9.0]
        @test prob2.p == typeof(prob.p)(p) # params unaffected

        # override defaults
        prob2 = @inferred baseType remake(prob; u0 = [:y => 0.2])
        @test prob2.u0 ≈ [1.0, 0.2, 3.0]
        @test prob2.p == typeof(prob.p)(p)
        prob2 = @inferred baseType remake(prob; u0 = [:y => 0.2], use_defaults = true)
        @test prob2.u0 ≈ [1.0, 0.2, 9.0]
        @test prob2.p == typeof(prob.p)(p)

        prob2 = @inferred baseType remake(prob; p = [:a => 0.2])
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [0.2, 0.6, 30.0])

        prob2 = @inferred baseType remake(prob; p = [:a => 0.2], use_defaults = true)
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [0.2, 0.6, 0.9])

        prob2 = @inferred baseType remake(prob; p = [:b => 0.2])
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [10.0, 0.2, 30.0])
        prob2 = @inferred baseType remake(prob; p = [:b => 0.2], use_defaults = true)
        @test prob2.u0 == u0
        @test all(prob2.p .≈ [10.0, 0.2, 0.9])

        empty!(prob.f.sys.defaults)
    end

    # defaults dependent on each other (params <-> states)
    begin
        prob.f.sys.defaults[:b] = :(3x)
        prob.f.sys.defaults[:y] = :(3a)
        prob.f.sys.defaults[:x] = 0.1
        prob.f.sys.defaults[:a] = 1.0
        # remake with no updates should use existing values
        prob2 = @inferred baseType remake(prob)
        @test prob2.u0 == u0
        @test prob2.p == typeof(prob.p)(p)

        # Dependency ignored since `p` was not changed
        prob2 = @inferred baseType remake(prob; u0 = [:x => 0.2])
        @test prob2.u0 ≈ [0.2, 30.0, 3.0]
        @test prob2.p == typeof(prob.p)(p)

        # need to pass empty `Dict()` to prevent defaulting to existing values
        prob2 = @inferred baseType remake(
            prob; u0 = [:x => 0.2], p = Dict())
        @test prob2.u0 ≈ [0.2, 30.0, 3.0]
        @test all(prob2.p .≈ [10.0, 0.6, 30.0])

        prob2 = @inferred baseType remake(
            prob; u0 = [:x => 0.2], p = Dict(), use_defaults = true)
        @test prob2.u0 ≈ [0.2, 3.0, 3.0]
        @test all(prob2.p .≈ [1.0, 0.6, 30.0])

        # override defaults
        prob2 = @inferred baseType remake(
            prob; u0 = [:y => 0.2], p = Dict())
        @test prob2.u0 ≈ [1.0, 0.2, 3.0]
        @test all(prob2.p .≈ [10.0, 3.0, 30.0])
        prob2 = @inferred baseType remake(
            prob; u0 = [:y => 0.2], p = Dict(), use_defaults = true)
        @test prob2.u0 ≈ [0.1, 0.2, 3.0]
        @test all(prob2.p .≈ [1.0, 0.3, 30.0])

        prob2 = @inferred baseType remake(
            prob; p = [:a => 0.2], u0 = Dict())
        @test prob2.u0 ≈ [1.0, 0.6, 3.0]
        @test all(prob2.p .≈ [0.2, 3.0, 30.0])
        prob2 = @inferred baseType remake(
            prob; p = [:a => 0.2], u0 = Dict(), use_defaults = true)
        @test prob2.u0 ≈ [0.1, 0.6, 3.0]
        @test all(prob2.p .≈ [0.2, 0.3, 30.0])

        prob2 = @inferred baseType remake(
            prob; p = [:b => 0.2], u0 = Dict())
        @test prob2.u0 ≈ [1.0, 30.0, 3.0]
        @test all(prob2.p .≈ [10.0, 0.2, 30.0])
        prob2 = @inferred baseType remake(
            prob; p = [:b => 0.2], u0 = Dict(), use_defaults = true)
        @test prob2.u0 ≈ [0.1, 3.0, 3.0]
        @test all(prob2.p .≈ [1.0, 0.2, 30.0])

        empty!(prob.f.sys.defaults)
    end

    if !isa(prob.p, Tuple)
        function fakeloss!(p)
            prob2 = @inferred baseType remake(prob; p = [:a => p])
            @test eltype(prob2.p) <: ForwardDiff.Dual
            return prob2.ps[:a]
        end
        ForwardDiff.derivative(fakeloss!, 1.0)
    end
end

# eltype(()) <: Pair, so ensure that this doesn't error
function lorenz!(du, u, _, t)
    du[1] = 1 * (u[2] - u[1])
    du[2] = u[1] * (2 - u[3]) - u[2]
    du[3] = u[1] * u[2] - 3 * u[3]
end
u0 = [1.0; 2.0; 3.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan, nothing)
@test_nowarn remake(prob, p = (), interpret_symbolicmap = true)

# IntervalNonlinearProblem doesn't have a u0
# Issue#726
interval_f(u, p) = u * u - 2.0 + p[1]
uspan = (1.0, 2.0)
interval_prob = IntervalNonlinearProblem(interval_f, uspan)
new_prob = @inferred IntervalNonlinearProblem remake(interval_prob; p = [0])
@test new_prob.p == [0]

# SDEProblem specific
function noise2!(du, u, p, t)
    du .= 0.2u
end
fn = SDEFunction(lorenz!, noise!; sys)
sdeprob = SDEProblem(fn, u0, tspan, Tuple(p))
newprob = remake(sdeprob; g = noise2!)
@test newprob.f isa SDEFunction
tmp = newprob.g([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], nothing, 0.0)
@test tmp≈[0.2, 0.4, 0.6] atol=1e-6

struct Remake_Test1
    p::Any
    args::Any
    kwargs::Any
end
Remake_Test1(args...; p, kwargs...) = Remake_Test1(p, args, kwargs)
a = Remake_Test1(p = 1)
@test @inferred remake(a, p = 2) == Remake_Test1(p = 2)
@test @inferred remake(a, args = 1) == Remake_Test1(1, p = 1)
@test @inferred remake(a, kwargs = (; a = 1)) == Remake_Test1(p = 1, a = 1)
