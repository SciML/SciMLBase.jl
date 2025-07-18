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
indep_sys = SymbolCache([:x, :y, :z], [:a, :b, :c])
fn = ODEFunction(lorenz!; sys)
for T in containerTypes
    push!(probs, ODEProblem(fn, u0, tspan, T(p)))
end
for T in containerTypes
    push!(probs, SteadyStateProblem(fn, u0, T(p)))
end

function ddelorenz!(du, u, h, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

function history(p, t)
    return u0 .- t
end

fn = DDEFunction(ddelorenz!; sys)
for T in containerTypes
    push!(probs, DDEProblem(fn, u0, history, tspan, T(p)))
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

fn = SDDEFunction(ddelorenz!, noise!; sys)
for T in containerTypes
    push!(probs, SDDEProblem(fn, noise!, u0, history, tspan, T(p)))
end

function loss(x, p)
    du = similar(x)
    lorenz!(du, u, p, 0.0)
    return sum(du)
end

fn = OptimizationFunction(loss; sys = indep_sys)
for T in containerTypes
    push!(probs, OptimizationProblem(fn, u0, T(p)))
end

function nllorenz!(du, u, p)
    lorenz!(du, u, p, 0.0)
end

fn = NonlinearFunction(nllorenz!; sys = indep_sys)
for T in containerTypes
    push!(probs, NonlinearProblem(fn, u0, T(p)))
end

for T in containerTypes
    push!(probs, NonlinearLeastSquaresProblem(fn, u0, T(p)))
end

update_A! = function (A, p)
    A[1, 1] = p[1]
    A[2, 2] = p[2]
    A[3, 3] = p[3]
end
update_b! = function (b, p)
    b[1] = p[3]
    b[2] = -8p[2] - p[1]
end
f = SciMLBase.SymbolicLinearInterface(update_A!, update_b!, indep_sys, nothing, nothing)
for T in containerTypes
    push!(probs, LinearProblem(rand(3, 3), rand(3), T(p); u0, f))
end

# temporary definition to test this functionality
function SciMLBase.late_binding_update_u0_p(
        prob, u0, p::SciMLBase.NullParameters, t0, newu0, newp)
    return newu0, ones(3)
end

@testset "$(SciMLBase.parameterless_type(prob)) - $(typeof(prob.p))" for prob in
                                                                         deepcopy(probs)
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

    # test late_binding_update_u0_p
    prob2 = remake(prob; p = SciMLBase.NullParameters())
    @test prob2.p ≈ ones(3)
end

# delete the method defined here to prevent breaking other tests
Base.delete_method(only(methods(SciMLBase.late_binding_update_u0_p, @__MODULE__)))

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

@testset "fill_u0 and fill_p ignore identical variables with different names" begin
    struct SCWrapper{S}
        sc::S
    end
    SymbolicIndexingInterface.symbolic_container(s::SCWrapper) = s.sc
    function SymbolicIndexingInterface.is_variable(s::SCWrapper, i::Symbol)
        if i == :x2
            return is_variable(s.sc, :x)
        end
        is_variable(s.sc, i)
    end
    function SymbolicIndexingInterface.variable_index(s::SCWrapper, i::Symbol)
        if i == :x2
            return variable_index(s.sc, :x)
        end
        variable_index(s.sc, i)
    end
    function SymbolicIndexingInterface.is_parameter(s::SCWrapper, i::Symbol)
        if i == :a2
            return is_parameter(s.sc, :a)
        end
        is_parameter(s.sc, i)
    end
    function SymbolicIndexingInterface.parameter_index(s::SCWrapper, i::Symbol)
        if i == :a2
            return parameter_index(s.sc, :a)
        end
        parameter_index(s.sc, i)
    end
    sys = SCWrapper(SymbolCache(Dict(:x => 1, :y => 2), Dict(:a => 1, :b => 2),
        :t; defaults = Dict(:x => 1, :y => 2, :a => 3, :b => 4)))
    function foo(du, u, p, t)
        du .= u .* p
    end
    prob = ODEProblem(ODEFunction(foo; sys), [1.5, 2.5], (0.0, 1.0), [3.5, 4.5])
    u0 = Dict(:x2 => 2)
    newu0 = SciMLBase.fill_u0(prob, u0; defs = default_values(sys))
    @test length(newu0) == 2
    @test get(newu0, :x, 0) == 2
    @test get(newu0, :y, 0) == 2.5
    p = Dict(:a2 => 3)
    newp = SciMLBase.fill_p(prob, p; defs = default_values(sys))
    @test length(newp) == 2
    @test get(newp, :a, 0) == 3
    @test get(newp, :b, 0) == 4.5
end

@testset "value of `nothing` is ignored" begin
    sys = SymbolCache(Dict(:x => 1, :y => 2), Dict(:a => 1, :b => 2),
        :t; defaults = Dict(:x => 1, :y => 2, :a => 3, :b => 4))
    function foo(du, u, p, t)
        du .= u .* p
    end
    prob = ODEProblem(ODEFunction(foo; sys), [1.5, 2.5], (0.0, 1.0), [3.5, 4.5])
    @test_nowarn remake(prob; u0 = [:x => nothing], p = [:a => nothing])
end

@testset "retain properties of `SciMLFunction` passed to `remake`" begin
    u0 = [1.0; 2.0; 3.0]
    p = [10.0, 20.0, 30.0]
    sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t)
    fn = NonlinearFunction(nllorenz!; sys, resid_prototype = zeros(Float64, 3))
    prob = NonlinearProblem(fn, u0, p)
    fn2 = NonlinearFunction(nllorenz!; resid_prototype = zeros(Float32, 3))
    prob2 = remake(prob; f = fn2)
    @test prob2.f.resid_prototype isa Vector{Float32}
end

@testset "`remake(::HomotopyNonlinearFunction)`" begin
    f! = function (du, u, p)
        du[1] = u[1] * u[1] - p[1] * u[2] + u[2]^3 + 1
        du[2] = u[2]^3 + 2 * p[2] * u[1] * u[2] + u[2]
    end

    fjac! = function (j, u, p)
        j[1, 1] = 2u[1]
        j[1, 2] = -p[1] + 3 * u[2]^2
        j[2, 1] = 2 * p[2] * u[2]
        j[2, 2] = 3 * u[2]^2 + 2 * p[2] * u[1] + 1
    end
    fn = NonlinearFunction(f!; jac = fjac!)
    fn = HomotopyNonlinearFunction(fn)
    prob = NonlinearProblem(fn, ones(2), ones(2))
    @test prob.f.f.jac == fjac!
    prob2 = remake(prob; u0 = zeros(2))
    @test prob2.f.f.jac == fjac!
end

@testset "Issue#925: `remake` retains specialization of explicit `f`" begin
    f = ODEFunction{false, SciMLBase.FullSpecialize}((u, p, t) -> u)
    prob = ODEProblem(f, nothing, nothing)
    @test SciMLBase.specialization(prob.f) == SciMLBase.FullSpecialize
    prob2 = remake(ODEProblem((u, p, t) -> 2 .* u, nothing, nothing); f = f)
    @test SciMLBase.specialization(prob2.f) == SciMLBase.FullSpecialize
end

@testset "`remake(::LinearProblem)` without a system" begin
    prob = LinearProblem{true}(rand(3, 3), rand(3))
    @inferred remake(prob)
    base_allocs = @allocations remake(prob)
    A = ones(3, 3)
    b = ones(3)
    u0 = ones(3)
    p = "P"
    @inferred remake(prob; A, b, u0, p)
    @test (@allocations remake(prob; A, b, u0, p)) <= base_allocs

    prob2 = remake(prob; u0)
    @test prob2.u0 === u0
    prob2 = remake(prob; A = SMatrix{3, 3}(A))
    @test prob2.A  isa SMatrix{3, 3}
end
