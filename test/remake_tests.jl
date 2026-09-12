using SciMLBase
using SymbolicIndexingInterface
using StaticArrays
using DifferentiationInterface
using ADTypes
using ForwardDiff: ForwardDiff
using RecursiveArrayTools

probs = []
containerTypes = [Vector, Tuple, SVector{3}, MVector{3}, SizedVector{3}]
# ODE
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    return du[3] = u[1] * u[2] - p[3] * u[3]
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
    return du[3] = u[1] * u[2] - p[3] * u[3]
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
    return resid[3] = u[3] - 0.5
end
fn = BVPFunction(lorenz!, residual!; sys)
for T in containerTypes
    push!(probs, BVProblem(fn, u0, tspan, T(p)))
end

function noise!(du, u, p, t)
    return du .= 0.1u
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

@testset "`remake(::OptimizationFunction)` works" begin
    expr = :(A + B)
    _fn = remake(fn; expr)
    @test _fn.expr == expr
end

for T in containerTypes
    push!(probs, OptimizationProblem(fn, u0, T(p)))
end

function nllorenz!(du, u, p)
    return lorenz!(du, u, p, 0.0)
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
    return A[3, 3] = p[3]
end
update_b! = function (b, p)
    b[1] = p[3]
    return b[2] = -8p[2] - p[1]
end
f = SciMLBase.SymbolicLinearInterface(update_A!, update_b!, indep_sys, nothing, nothing)
for T in containerTypes
    push!(probs, LinearProblem(rand(3, 3), rand(3), T(p); u0, f))
end

# temporary definition to test this functionality
function SciMLBase.late_binding_update_u0_p(
        prob, u0, p::SciMLBase.NullParameters, t0, newu0, newp
    )
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
            prob; u0 = [:x => 0.2], p = Dict()
        )
        @test prob2.u0 ≈ [0.2, 30.0, 3.0]
        @test all(prob2.p .≈ [10.0, 0.6, 30.0])

        prob2 = @inferred baseType remake(
            prob; u0 = [:x => 0.2], p = Dict(), use_defaults = true
        )
        @test prob2.u0 ≈ [0.2, 3.0, 3.0]
        @test all(prob2.p .≈ [1.0, 0.6, 30.0])

        # override defaults
        prob2 = @inferred baseType remake(
            prob; u0 = [:y => 0.2], p = Dict()
        )
        @test prob2.u0 ≈ [1.0, 0.2, 3.0]
        @test all(prob2.p .≈ [10.0, 3.0, 30.0])
        prob2 = @inferred baseType remake(
            prob; u0 = [:y => 0.2], p = Dict(), use_defaults = true
        )
        @test prob2.u0 ≈ [0.1, 0.2, 3.0]
        @test all(prob2.p .≈ [1.0, 0.3, 30.0])

        prob2 = @inferred baseType remake(
            prob; p = [:a => 0.2], u0 = Dict()
        )
        @test prob2.u0 ≈ [1.0, 0.6, 3.0]
        @test all(prob2.p .≈ [0.2, 3.0, 30.0])
        prob2 = @inferred baseType remake(
            prob; p = [:a => 0.2], u0 = Dict(), use_defaults = true
        )
        @test prob2.u0 ≈ [0.1, 0.6, 3.0]
        @test all(prob2.p .≈ [0.2, 0.3, 30.0])

        prob2 = @inferred baseType remake(
            prob; p = [:b => 0.2], u0 = Dict()
        )
        @test prob2.u0 ≈ [1.0, 30.0, 3.0]
        @test all(prob2.p .≈ [10.0, 0.2, 30.0])
        prob2 = @inferred baseType remake(
            prob; p = [:b => 0.2], u0 = Dict(), use_defaults = true
        )
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
        DifferentiationInterface.derivative(fakeloss!, AutoForwardDiff(), 1.0)
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
    return du[3] = u[1] * u[2] - 3 * u[3]
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
    return du .= 0.2u
end
fn = SDEFunction(lorenz!, noise!; sys)
sdeprob = SDEProblem(fn, u0, tspan, Tuple(p))
newprob = remake(sdeprob; g = noise2!)
@test newprob.f isa SDEFunction
tmp = newprob.g([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], nothing, 0.0)
@test tmp ≈ [0.2, 0.4, 0.6] atol = 1.0e-6

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
    sys = SCWrapper(
        SymbolCache(
            Dict(:x => 1, :y => 2), Dict(:a => 1, :b => 2),
            :t; defaults = Dict(:x => 1, :y => 2, :a => 3, :b => 4)
        )
    )
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
    sys = SymbolCache(
        Dict(:x => 1, :y => 2), Dict(:a => 1, :b => 2),
        :t; defaults = Dict(:x => 1, :y => 2, :a => 3, :b => 4)
    )
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
    prob2 = remake(ODEProblem((u, p, t) -> 2 .* u, nothing, nothing); f)
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
    @test prob2.A isa SMatrix{3, 3}
end

@testset "Issue#1267: `anyeltypedual` ambiguity" begin
    ts = 0.0:0.1:10.0
    f1(t) = t - 1
    f2(t) = t^2
    vals = [[f1(t), f2(t)] for t in ts]
    sol = DiffEqArray(vals, ts)
    @test SciMLBase.anyeltypedual(sol, Val{0}) == Any
end

@testset "`remake` preserves `lb`/`ub` on bounded nonlinear problems" begin
    nlf(u, p) = u .^ 2 .- p
    for P in (NonlinearProblem, NonlinearLeastSquaresProblem)
        prob = P(nlf, [1.0, 1.0], [2.0, 3.0]; lb = [0.0, 0.0], ub = [5.0, 5.0])
        @test prob.lb == [0.0, 0.0]
        @test prob.ub == [5.0, 5.0]

        # `remake` with no bounds override keeps the existing bounds
        prob2 = remake(prob; u0 = [2.0, 2.0])
        @test prob2.lb == [0.0, 0.0]
        @test prob2.ub == [5.0, 5.0]

        # explicit bounds override
        prob3 = remake(prob; lb = [-1.0, -1.0], ub = [1.0, 1.0])
        @test prob3.lb == [-1.0, -1.0]
        @test prob3.ub == [1.0, 1.0]

        # bounds can be cleared
        prob4 = remake(prob; lb = nothing, ub = nothing)
        @test prob4.lb === nothing
        @test prob4.ub === nothing
    end
end

@testset "`remake` `DynamicalODEProblem` with the additional argument" begin
    f1! = function (dv, v, u, p, t)
        dv[1] = -p[1] * u[1]
    end
    f2! = function (du, v, u, p, t)
        du[1] = p[2] * v[1]
    end
    prob = DynamicalODEProblem(f1!, f2!, [1.0], [2.0], 3.0, [4.0, 5.0])
    @test prob.u0 == ArrayPartition([1.0], [2.0])
    @test SciMLBase.problem_type(prob) isa DynamicalODEProblem{true}

    # `remake` with no initial values override
    prob2 = remake(prob; p = [6.0, 7.0])
    @test prob2.u0 == ArrayPartition([1.0], [2.0])
    @test prob2.p == [6.0, 7.0]

    # `remake` v0 override
    prob3 = remake(prob; v0 = [6.0])
    @test prob3.u0 == ArrayPartition([6.0], [2.0])

    # `remake` u0 override
    prob4 = remake(prob; u0 = [6.0])
    @test prob4.u0 == ArrayPartition([1.0], [6.0])

    # `remake` v0 and u0 override
    prob5 = remake(prob; v0 = [6.0], u0 = [7.0])
    @test prob5.u0 == ArrayPartition([6.0], [7.0])
end

@testset "`remake` `SecondOrderODEProblem` with the additional argument" begin
    f! = function (ddu, du, u, p, t)
        ddu[1] = -p[1] * sin(u[1])
    end
    prob = SecondOrderODEProblem(f!, [1.0], [2.0], 3.0, [4.0])
    @test prob.u0 == ArrayPartition([1.0], [2.0])
    @test SciMLBase.problem_type(prob) isa SecondOrderODEProblem{true}

    # `remake` with no initial values override
    prob2 = remake(prob; p = [5.0])
    @test prob2.u0 == ArrayPartition([1.0], [2.0])
    @test prob2.p == [5.0]

    # `remake` du0 override
    prob3 = remake(prob; du0 = [5.0])
    @test prob3.u0 == ArrayPartition([5.0], [2.0])

    # `remake` u0 override
    prob4 = remake(prob; u0 = [5.0])
    @test prob4.u0 == ArrayPartition([1.0], [5.0])

    # `remake` du0 and u0 override
    prob5 = remake(prob; du0 = [5.0], u0 = [6.0])
    @test prob5.u0 == ArrayPartition([5.0], [6.0])
end

@testset "`DynamicalODEProblem` constructors retain the problem type" begin
    f1!(dv, v, u, p, t) = dv .= -p[1] .* u
    f2!(du, v, u, p, t) = du .= p[2] .* v
    wrapped! = DynamicalODEFunction(f1!, f2!)
    iip_probs = (
        DynamicalODEProblem(wrapped!, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
        DynamicalODEProblem(f1!, f2!, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
        DynamicalODEProblem{true}(f1!, f2!, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
    )
    for prob in iip_probs
        @test SciMLBase.isinplace(prob)
        @test SciMLBase.problem_type(prob) isa DynamicalODEProblem{true}
    end

    f1(v, u, p, t) = -p[1] .* u
    f2(v, u, p, t) = p[2] .* v
    wrapped = DynamicalODEFunction(f1, f2)
    oop_probs = (
        DynamicalODEProblem(wrapped, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
        DynamicalODEProblem(f1, f2, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
        DynamicalODEProblem{false}(f1, f2, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]),
    )
    for prob in oop_probs
        @test !SciMLBase.isinplace(prob)
        @test SciMLBase.problem_type(prob) isa DynamicalODEProblem{false}
    end
end

@testset "wrapped `SecondOrderODEProblem` constructors preserve specialization" begin
    acceleration(du, u, p, t) = -p[1] .* u
    analytic(u0, p, t) = :analytic
    for spec in (
            SciMLBase.AutoSpecialize, SciMLBase.AutoDespecialize,
            SciMLBase.AutoRespecialize, SciMLBase.FullSpecialize,
            SciMLBase.NoSpecialize,
        )
        func = DynamicalODEFunction{false, spec}(
            acceleration, nothing; analytic
        )
        prob = SecondOrderODEProblem(
            func, [1.0], [2.0], (0.0, 1.0), [3.0]
        )
        @test SciMLBase.specialization(prob.f) === spec
        @test SciMLBase.specialization(prob.f.f1) === spec
        @test SciMLBase.specialization(prob.f.f2) === spec
        @test prob.f.analytic === analytic
        @test SciMLBase.problem_type(prob) isa SecondOrderODEProblem{false}
        @test prob.f(prob.u0, prob.p, 0.0) == ArrayPartition([-6.0], [1.0])
    end

    kinematic(du, u, p, t) = 2 .* du
    raw_func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        acceleration, kinematic
    )
    raw_prob = SecondOrderODEProblem(
        raw_func, [1.0], [2.0], (0.0, 1.0), [3.0]
    )
    @test SciMLBase.specialization(raw_prob.f) === SciMLBase.FullSpecialize
    @test SciMLBase.specialization(raw_prob.f.f1) === SciMLBase.FullSpecialize
    @test SciMLBase.specialization(raw_prob.f.f2) === SciMLBase.FullSpecialize
    @test raw_prob.f(raw_prob.u0, raw_prob.p, 0.0) ==
        ArrayPartition([-6.0], [2.0])
end

@testset "structured `ODEProblem` remake uses the full ODE pipeline" begin
    f1(v, u, p, t) = -p[1] .* u
    f2(v, u, p, t) = p[2] .* v
    prob = DynamicalODEProblem(
        f1, f2, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]; saveat = 0.1
    )

    same = @inferred remake(prob)
    @test same.u0 === prob.u0
    @test same.f === prob.f
    @test same.p === prob.p
    @test same.tspan === prob.tspan
    @test SciMLBase.problem_type(same) === SciMLBase.problem_type(prob)

    newv0 = [5.0]
    vprob = @inferred remake(prob; v0 = newv0)
    @test vprob.u0.x[1] === newv0
    @test vprob.u0.x[2] === prob.u0.x[2]

    newu0 = [6.0]
    uprob = @inferred remake(prob; u0 = newu0)
    @test uprob.u0.x[1] === prob.u0.x[1]
    @test uprob.u0.x[2] === newu0

    both = @inferred remake(prob; v0 = newv0, u0 = newu0)
    @test both.u0 == ArrayPartition(newv0, newu0)

    packed = ArrayPartition([7.0], [8.0])
    packed_prob = @inferred remake(prob; u0 = packed)
    @test packed_prob.u0 === packed
    nested_prob = remake(prob; v0 = [9.0], u0 = packed)
    @test nested_prob.u0.x[1] == [9.0]
    @test nested_prob.u0.x[2] === packed

    newp = [9.0, 10.0]
    forwarded = remake(
        prob; v0 = newv0, p = newp, tspan = (1.0, 2.0), abstol = 1.0e-8,
        build_initializeprob = false, lazy_initialization = true
    )
    @test forwarded.p === newp
    @test forwarded.tspan == (1.0, 2.0)
    @test values(forwarded.kwargs) == (; saveat = 0.1, abstol = 1.0e-8)
    @test SciMLBase.problem_type(forwarded) === SciMLBase.problem_type(prob)

    replaced_kwargs = remake(prob; kwargs = (; reltol = 1.0e-7))
    @test values(replaced_kwargs.kwargs) == (; reltol = 1.0e-7)
end

@testset "structured ODE function replacement and specialization" begin
    f1(v, u, p, t) = -p[1] .* u
    f2(v, u, p, t) = p[2] .* v
    newf1(v, u, p, t) = fill(11.0, length(v))
    newf2(v, u, p, t) = fill(12.0, length(u))

    for spec in (
            SciMLBase.AutoSpecialize, SciMLBase.AutoDespecialize,
            SciMLBase.AutoRespecialize, SciMLBase.FullSpecialize,
            SciMLBase.NoSpecialize,
        )
        func = DynamicalODEFunction{false, spec}(f1, f2)
        prob = DynamicalODEProblem(func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0])
        remade = remake(prob; v0 = [5.0])
        @test SciMLBase.specialization(remade.f) === spec
        @test SciMLBase.problem_type(remade) === SciMLBase.problem_type(prob)
    end

    wrapped_func = DynamicalODEFunction{
        false, SciMLBase.FunctionWrapperSpecialize,
    }(f1, f2)
    wrapped_prob = DynamicalODEProblem(
        wrapped_func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    wrapped_remade = @inferred remake(wrapped_prob; v0 = [5.0])
    @test SciMLBase.specialization(wrapped_remade.f) ===
        SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.specialization(wrapped_remade.f.f1) ===
        SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.specialization(wrapped_remade.f.f2) ===
        SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.unwrapped_f(wrapped_remade.f.f1.f) === f1
    @test SciMLBase.unwrapped_f(wrapped_remade.f.f2.f) === f2

    old_analytic(u0, p, t) = :old
    new_analytic(u0, p, t) = :new
    oldfunc = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        f1, f2; analytic = old_analytic
    )
    prob = DynamicalODEProblem(oldfunc, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0])

    first_replaced = remake(prob; f = newf1)
    first_result = first_replaced.f(first_replaced.u0, first_replaced.p, 0.0)
    @test first_result.x[1] == [11.0]
    @test first_result.x[2] == [4.0]
    @test first_replaced.f.analytic === old_analytic
    @test SciMLBase.specialization(first_replaced.f) === SciMLBase.FullSpecialize
    @test SciMLBase.problem_type(first_replaced) === SciMLBase.problem_type(prob)

    erased_first = ODEFunction{false, SciMLBase.NoSpecialize}(newf1)
    erased_first_replaced = remake(prob; f = erased_first)
    @test SciMLBase.specialization(erased_first_replaced.f) ===
        SciMLBase.FullSpecialize
    @test SciMLBase.specialization(erased_first_replaced.f.f1) ===
        SciMLBase.FullSpecialize

    newfunc = DynamicalODEFunction{false, SciMLBase.NoSpecialize}(
        newf1, newf2; analytic = new_analytic
    )
    fully_replaced = remake(prob; f = newfunc)
    full_result = fully_replaced.f(fully_replaced.u0, fully_replaced.p, 0.0)
    @test full_result == ArrayPartition([11.0], [12.0])
    @test fully_replaced.f.analytic === new_analytic
    @test SciMLBase.specialization(fully_replaced.f) === SciMLBase.FullSpecialize
    @test SciMLBase.specialization(fully_replaced.f.f1) === SciMLBase.FullSpecialize
    @test SciMLBase.specialization(fully_replaced.f.f2) === SciMLBase.FullSpecialize
    @test SciMLBase.problem_type(fully_replaced) === SciMLBase.problem_type(prob)

    replacement_without_f2 = DynamicalODEFunction(newf1)
    fallback_replaced = remake(prob; f = replacement_without_f2)
    fallback_result = fallback_replaced.f(
        fallback_replaced.u0, fallback_replaced.p, 0.0
    )
    @test fallback_result == ArrayPartition([11.0], [4.0])

    component_jac(u, p, t) = fill(-p[1], length(u))
    component_f1 = ODEFunction{false, SciMLBase.FullSpecialize}(
        f1; jac = component_jac
    )
    metadata_func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        component_f1, f2
    )
    metadata_prob = DynamicalODEProblem(
        metadata_func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    metadata_remade = remake(metadata_prob)
    @test metadata_remade.f.f1.jac === component_jac
    @test SciMLBase.has_jac(metadata_remade.f)
    metadata_raw_replaced = remake(metadata_prob; f = newf1)
    @test metadata_raw_replaced.f.f1.jac === component_jac
    @test SciMLBase.has_jac(metadata_raw_replaced.f)
    metadata_wrapped_replaced = remake(
        metadata_prob;
        f = ODEFunction{false, SciMLBase.NoSpecialize}(newf1)
    )
    @test metadata_wrapped_replaced.f.f1.jac === component_jac
    @test SciMLBase.has_jac(metadata_wrapped_replaced.f)

    widened_component_f1 = SciMLBase.widen_bounded_type_params(component_f1)
    widened_func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        widened_component_f1, f2
    )
    widened_prob = DynamicalODEProblem(
        widened_func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    widened_remade = remake(widened_prob)
    @test typeof(widened_remade.f.f1) === typeof(widened_component_f1)
    @test typeof(widened_remade.f.f1).parameters[end - 1] ===
        Union{Nothing, SciMLBase.OverrideInitData}
    widened_replaced = remake(widened_prob; f = newf1)
    @test widened_replaced.f.f1.jac === component_jac
    @test typeof(widened_replaced.f.f1).parameters[end - 1] ===
        Union{Nothing, SciMLBase.OverrideInitData}

    auto_func = DynamicalODEFunction{false, SciMLBase.AutoSpecialize}(f1, f2)
    auto_prob = DynamicalODEProblem(
        auto_func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    full_func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(newf1, newf2)
    auto_replaced = remake(auto_prob; f = full_func)
    @test SciMLBase.specialization(auto_replaced.f) === SciMLBase.AutoSpecialize
    @test SciMLBase.specialization(auto_replaced.f.f1) === SciMLBase.AutoSpecialize
    @test SciMLBase.specialization(auto_replaced.f.f2) === SciMLBase.AutoSpecialize

    newf1!(dv, v, u, p, t) = dv .= 14.0
    newf2!(du, v, u, p, t) = du .= 15.0
    inplace_func = DynamicalODEFunction{true, SciMLBase.FullSpecialize}(newf1!, newf2!)
    @test_throws ArgumentError remake(prob; f = inplace_func)
    inplace_component = ODEFunction{true, SciMLBase.FullSpecialize}(newf1!)
    @test_throws ArgumentError remake(prob; f = inplace_component)
    @test_throws ArgumentError remake(prob.f; f2 = inplace_component)

    oldf1!(dv, v, u, p, t) = dv .= -p[1] .* u
    oldf2!(du, v, u, p, t) = du .= p[2] .* v
    inplace_prob = DynamicalODEProblem(
        oldf1!, oldf2!, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    inplace_replaced = remake(inplace_prob; f = newf1!)
    derivative = ArrayPartition(zeros(1), zeros(1))
    inplace_replaced.f(derivative, inplace_replaced.u0, inplace_replaced.p, 0.0)
    @test derivative == ArrayPartition([14.0], [4.0])
    @test SciMLBase.problem_type(inplace_replaced) ===
        SciMLBase.problem_type(inplace_prob)

    acceleration(du, u, p, t) = -p[1] .* u
    new_acceleration(du, u, p, t) = fill(13.0, length(u))
    second_order = SecondOrderODEProblem(
        acceleration, [1.0], [2.0], (0.0, 1.0), [3.0]
    )
    second_same = @inferred remake(second_order)
    @test SciMLBase.problem_type(second_same) === SciMLBase.problem_type(second_order)
    second_packed = ArrayPartition([6.0], [7.0])
    @test remake(second_order; u0 = second_packed).u0 === second_packed
    second_remade = @inferred remake(second_order; du0 = [5.0], f = new_acceleration)
    second_result = second_remade.f(second_remade.u0, second_remade.p, 0.0)
    @test second_result.x[1] == [13.0]
    @test second_result.x[2] == [5.0]
    second_full_replacement = remake(
        second_order; f = DynamicalODEFunction(new_acceleration)
    )
    second_full_result = second_full_replacement.f(
        second_full_replacement.u0, second_full_replacement.p, 0.0
    )
    @test second_full_result.x[1] == [13.0]
    @test second_full_result.x[2] == [1.0]
end

@testset "structured `FunctionWrapperSpecialize` remake" begin
    f1(v, u, p, t) = v .* p[1]
    f2(v, u, p, t) = u .* p[2]
    v0 = Float32[1]
    u0 = Int[2]
    p = (Float32(3), 2)
    func = DynamicalODEFunction{false, SciMLBase.FunctionWrapperSpecialize}(f1, f2)
    prob = @inferred DynamicalODEProblem(func, v0, u0, (0.0, 1.0), p)

    @test SciMLBase.specialization(prob.f) === SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.specialization(prob.f.f1) === SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.specialization(prob.f.f2) === SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.unwrapped_f(prob.f.f1.f) === f1
    @test SciMLBase.unwrapped_f(prob.f.f2.f) === f2
    result = @inferred prob.f(prob.u0, prob.p, 0.0)
    @test result == ArrayPartition(Float32[3], Int[4])
    @test result.x[1] isa Vector{Float32}
    @test result.x[2] isa Vector{Int}

    derivative = ForwardDiff.derivative(1.0) do x
        value = prob.f(ArrayPartition([x], prob.u0.x[2]), prob.p, 0.0)
        return only(value.x[1])
    end
    @test derivative == 3.0

    newf1(v, u, p, t) = v .+ p[1]
    newf2(v, u, p, t) = u .+ p[2]
    replacement = DynamicalODEFunction{
        false, SciMLBase.FunctionWrapperSpecialize,
    }(newf1, newf2)
    remade = @inferred remake(prob; f = replacement, v0 = Float32[5])
    @test typeof(remade.f.f1.f) === typeof(prob.f.f1.f)
    @test typeof(remade.f.f2.f) === typeof(prob.f.f2.f)
    @test SciMLBase.unwrapped_f(remade.f.f1.f) === newf1
    @test SciMLBase.unwrapped_f(remade.f.f2.f) === newf2
    @test remade.f(remade.u0, remade.p, 0.0) ==
        ArrayPartition(Float32[8], Int[4])

    raw_remade = @inferred remake(prob; f = newf1, v0 = Float32[6])
    @test SciMLBase.unwrapped_f(raw_remade.f.f1.f) === newf1
    @test SciMLBase.unwrapped_f(raw_remade.f.f2.f) === f2
    @test raw_remade.f(raw_remade.u0, raw_remade.p, 0.0) ==
        ArrayPartition(Float32[9], Int[4])

    component_jac(u, p, t) = fill(p[1], length(u))
    component = ODEFunction{false, SciMLBase.FullSpecialize}(f1; jac = component_jac)
    metadata_func = DynamicalODEFunction{
        false, SciMLBase.FunctionWrapperSpecialize,
    }(component, f2)
    metadata_prob = DynamicalODEProblem(metadata_func, v0, u0, (0.0, 1.0), p)
    metadata_remade = remake(metadata_prob; f = newf1)
    @test metadata_remade.f.f1.jac === component_jac
    @test SciMLBase.has_jac(metadata_remade.f)

    f1!(dv, v, u, p, t) = dv .= p[1] .* u
    f2!(du, v, u, p, t) = du .= p[2] .* v
    ifunc = DynamicalODEFunction{true, SciMLBase.FunctionWrapperSpecialize}(f1!, f2!)
    iprob = @inferred DynamicalODEProblem(
        ifunc, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )
    iremade = @inferred remake(iprob; v0 = [5.0])
    du = ArrayPartition(zeros(1), zeros(1))
    @test iremade.f(du, iremade.u0, iremade.p, 0.0) === nothing
    @test du == ArrayPartition([6.0], [20.0])

    acceleration(du, u, p, t) = -p[1] .* u
    second_func = DynamicalODEFunction{
        false, SciMLBase.FunctionWrapperSpecialize,
    }(acceleration, nothing)
    second = @inferred SecondOrderODEProblem(
        second_func, [1.0], [2.0], (0.0, 1.0), [3.0]
    )
    second_remade = @inferred remake(second; du0 = [4.0])
    @test second_remade.f(second_remade.u0, second_remade.p, 0.0) ==
        ArrayPartition([-6.0], [4.0])
    @test SciMLBase.specialization(second_remade.f.f1) ===
        SciMLBase.FunctionWrapperSpecialize
    @test SciMLBase.specialization(second_remade.f.f2) ===
        SciMLBase.FunctionWrapperSpecialize
end

struct StructuredRemakeInitSystem
    name::Symbol
end

function SciMLBase.remake_initialization_data(
        sys::StructuredRemakeInitSystem, scimlfn, u0, t0, p, newu0, newp,
        ::SciMLBase.RemakeInitializationDataContext
    )
    initdata = scimlfn.initialization_data
    return SciMLBase.OverrideInitData(
        initdata.initializeprob, initdata.update_initializeprob!,
        initdata.initializeprobmap, initdata.initializeprobpmap; metadata = sys
    )
end

@testset "structured ODE initialization data follows full-function replacements" begin
    f1(v, u, p, t) = -p[1] .* u
    f2(v, u, p, t) = p[2] .* v
    newf1(v, u, p, t) = fill(11.0, length(v))
    newf2(v, u, p, t) = fill(12.0, length(u))

    global_initprob = NonlinearProblem(Returns(nothing), nothing, [1.0])
    component_initprob = NonlinearProblem(Returns(nothing), nothing, [2.0])
    replacement_initprob = NonlinearProblem(Returns(nothing), nothing, [3.0])
    global_initdata = SciMLBase.OverrideInitData(
        global_initprob, nothing, nothing, nothing
    )
    component_initdata = SciMLBase.OverrideInitData(
        component_initprob, nothing, nothing, nothing
    )
    replacement_initdata = SciMLBase.OverrideInitData(
        replacement_initprob, nothing, nothing, nothing
    )

    original_sys = StructuredRemakeInitSystem(:original)
    replacement_sys = StructuredRemakeInitSystem(:replacement)
    component_f1 = ODEFunction{false, SciMLBase.FullSpecialize}(
        f1; initialization_data = component_initdata
    )
    func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        component_f1, f2; sys = original_sys, initialization_data = global_initdata
    )
    prob = DynamicalODEProblem(
        func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0]
    )

    same = remake(prob)
    @test same.f.initialization_data.initializeprob === global_initprob
    @test same.f.initialization_data.metadata === original_sys
    @test same.f.f1.initialization_data.initializeprob === component_initprob

    incoming_component = ODEFunction{false, SciMLBase.FullSpecialize}(
        newf1; initialization_data = component_initdata
    )
    component_replaced = remake(prob; f = incoming_component)
    @test component_replaced.f.initialization_data.initializeprob === global_initprob
    @test component_replaced.f.initialization_data.metadata === original_sys
    @test component_replaced.f.f1.initialization_data.initializeprob ===
        component_initprob

    replacement = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        newf1, newf2; sys = replacement_sys, initialization_data = replacement_initdata
    )
    fully_replaced = remake(prob; f = replacement)
    @test fully_replaced.f.initialization_data.initializeprob === replacement_initprob
    @test fully_replaced.f.initialization_data.metadata === replacement_sys

    replacement_without_sys = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        newf1, newf2; initialization_data = replacement_initdata
    )
    replaced_without_sys = remake(prob; f = replacement_without_sys)
    @test replaced_without_sys.f.sys === original_sys
    @test replaced_without_sys.f.initialization_data.initializeprob ===
        replacement_initprob
    @test replaced_without_sys.f.initialization_data.metadata === original_sys

    replacement_without_init = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(
        newf1, newf2; sys = replacement_sys
    )
    replaced_without_init = remake(prob; f = replacement_without_init)
    @test replaced_without_init.f.sys === replacement_sys
    @test replaced_without_init.f.initialization_data.initializeprob === global_initprob
    @test replaced_without_init.f.initialization_data.metadata === replacement_sys

    disabled = remake(prob; build_initializeprob = false)
    @test disabled.f.initialization_data === nothing
end

@testset "structured ODE symbolic remake forwarding" begin
    f1(v, u, p, t) = -p[1] .* u
    f2(v, u, p, t) = p[2] .* v
    sys = SymbolCache([:v, :u], [:a, :b], :t)
    func = DynamicalODEFunction{false, SciMLBase.FullSpecialize}(f1, f2; sys)
    prob = DynamicalODEProblem(func, [1.0], [2.0], (0.0, 1.0), [3.0, 4.0])

    @test remake(prob; v0 = [:v => 5.0]).u0 == ArrayPartition([5.0], [2.0])
    @test remake(prob; u0 = [:u => 6.0]).u0 == ArrayPartition([1.0], [6.0])
    @test remake(prob; v0 = [:v => 5.0], u0 = [:u => 6.0]).u0 ==
        ArrayPartition([5.0], [6.0])
    @test remake(prob; p = [:a => 7.0]).p == [7.0, 4.0]
    @test_throws ArgumentError remake(prob; v0 = [5.0], u0 = [:u => 6.0])
end
