using Adapt
using FunctionWrappersWrappers
using ForwardDiff
using RecursiveArrayTools: ArrayPartition
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
using Test

struct DespecializedCallParameters
    rate::Float64
end

@testset "all callable SciMLFunction families" begin
    params = SciMLBase.DespecializedParameters(DespecializedCallParameters(2.0))
    seen = DataType[]
    function record_parameter_type(p)
        push!(seen, typeof(p))
        return p.rate
    end

    u = [1.0]
    dynamical_u = ArrayPartition([1.0], [2.0])
    calls = (
        () -> NonlinearFunction{false}((u, p) -> record_parameter_type(p))(u, params),
        () -> IntervalNonlinearFunction{false}((u, p) -> record_parameter_type(p))(
            1.0, params
        ),
        () -> IntegralFunction{false}((u, p) -> record_parameter_type(p), nothing)(
            u, params
        ),
        () -> BatchIntegralFunction{false}(
            (u, p) -> record_parameter_type(p), nothing
        )(u, params),
        () -> DiscreteFunction{false}((u, p, t) -> record_parameter_type(p))(
            u, params, 0.0
        ),
        () -> ImplicitDiscreteFunction{false}(
            (unext, u, p, t) -> record_parameter_type(p)
        )(u, u, params, 0.0),
        () -> DAEFunction{false}((du, u, p, t) -> record_parameter_type(p))(
            u, u, params, 0.0
        ),
        () -> DDEFunction{false}((u, h, p, t) -> record_parameter_type(p))(
            u, nothing, params, 0.0
        ),
        () -> SDEFunction{false}(
            (u, p, t) -> record_parameter_type(p), (u, p, t) -> u
        )(u, params, 0.0),
        () -> SDDEFunction{false}(
            (u, h, p, t) -> record_parameter_type(p), (u, h, p, t) -> u
        )(u, nothing, params, 0.0),
        () -> RODEFunction{false}((u, p, t, W) -> record_parameter_type(p))(
            u, params, 0.0, nothing
        ),
        () -> ODEInputFunction{false}((x, u, p, t) -> record_parameter_type(p))(
            u, u, params, 0.0
        ),
        () -> OptimizationFunction((u, p) -> record_parameter_type(p))(u, params),
        () -> MultiObjectiveOptimizationFunction((u, p) -> record_parameter_type(p))(
            u, params
        ),
        () -> HomotopyNonlinearFunction{false}((u, p) -> record_parameter_type(p))(
            u, params
        ),
        () -> BVPFunction{false}(
            (u, p, t) -> record_parameter_type(p), (u, p, t) -> nothing
        )(u, params, 0.0),
        () -> DynamicalBVPFunction{false}(
            (du, u, p, t) -> record_parameter_type(p),
            (du, u, p, t) -> nothing
        )(u, u, params, 0.0),
        () -> IncrementingODEFunction{false}(
            (u, p, t, alpha, beta; scale = 1) ->
            scale * record_parameter_type(p)
        )(u, params, 0.0, 1.0, 2.0; scale = 2),
        () -> SplitFunction{false}(
            (u, p, t) -> record_parameter_type(p),
            (u, p, t) -> record_parameter_type(p)
        )(u, params, 0.0),
        () -> DynamicalODEFunction{false}(
            (x, v, p, t) -> [record_parameter_type(p)],
            (x, v, p, t) -> [record_parameter_type(p)]
        )(dynamical_u, params, 0.0),
        () -> SplitSDEFunction{false}(
            (u, p, t) -> record_parameter_type(p),
            (u, p, t) -> record_parameter_type(p),
            (u, p, t) -> u
        )(u, params, 0.0),
    )

    foreach(call -> call(), calls)
    @test !isempty(seen)
    @test all(==(DespecializedCallParameters), seen)
end

@testset "stable container and forwarded interfaces" begin
    params = SciMLBase.DespecializedParameters([1.0, 2.0])
    other = SciMLBase.DespecializedParameters((rate = 2.0,))

    @test typeof(params) === typeof(other)
    @test fieldtype(typeof(params), :params) === Any
    @test SciMLBase.DespecializedParameters(params) === params
    @test SciMLBase.unwrap_parameters(params) === params.params
    @test SciMLBase.unwrap_parameters(params.params) === params.params
    @test params[2] == 2.0
    @test length(params) == 2
    @test size(params) == (2,)
    @test collect(params) == [1.0, 2.0]
    @test copy(params) == params
    @test hash(copy(params)) == hash(params)

    params[1] = 3.0
    @test params.params == [3.0, 2.0]
    @test SymbolicIndexingInterface.parameter_values(params) === params
    @test SymbolicIndexingInterface.parameter_values(params, 2) == 2.0
    @test SymbolicIndexingInterface.set_parameter!(params, 4.0, 2) == 4.0
    @test params.params == [3.0, 4.0]

    @test SciMLStructures.isscimlstructure(params)
    @test !SciMLStructures.ismutablescimlstructure(params)
    @test SciMLStructures.hasportion(SciMLStructures.Tunable(), params)
    values, repack, aliases = SciMLStructures.canonicalize(
        SciMLStructures.Tunable(), params
    )
    @test values == [3.0, 4.0]
    @test aliases
    repacked = repack([5.0, 6.0])
    @test repacked isa SciMLBase.DespecializedParameters
    @test repacked.params == [5.0, 6.0]
    replaced = SciMLStructures.replace(SciMLStructures.Tunable(), params, [7.0, 8.0])
    @test replaced isa SciMLBase.DespecializedParameters
    @test replaced.params == [7.0, 8.0]

    remade = SymbolicIndexingInterface.remake_buffer(
        nothing, SciMLBase.DespecializedParameters(Dict(:rate => 1.0)), [:rate], [2.0]
    )
    @test remade isa SciMLBase.DespecializedParameters
    @test remade.params == Dict(:rate => 2.0)
    @test Adapt.adapt(Array, params) isa SciMLBase.DespecializedParameters
end

@testset "dynamic function barrier" begin
    params = SciMLBase.DespecializedParameters(DespecializedCallParameters(2.0))
    seen_iip = Ref{DataType}()
    seen_oop = Ref{DataType}()

    function rhs!(du, u, p, t)
        seen_iip[] = typeof(p)
        du[1] = -p.rate * u[1]
        return nothing
    end
    function rhs(u, p, t)
        seen_oop[] = typeof(p)
        return -p.rate .* u
    end

    f_iip = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!)
    f_oop = ODEFunction{false, SciMLBase.AutoSpecialize}(rhs)
    du = zeros(1)
    f_iip(du, [3.0], params, 0.0)
    @test du == [-6.0]
    @test seen_iip[] === DespecializedCallParameters
    @test f_oop([3.0], params, 0.0) == [-6.0]
    @test seen_oop[] === DespecializedCallParameters

    args = (zeros(1), [4.0], params, 0.0)
    SciMLBase.invoke_with_despecialized_parameters(rhs!, args, params, Val(3))
    @test args[1] == [-8.0]

    dual = ForwardDiff.Dual{:despecialized_parameters}(2.0, 1.0)
    @test SciMLBase.anyeltypedual(SciMLBase.DespecializedParameters([dual])) <:
    ForwardDiff.Dual

    function wrapped_rhs!(du, u, p, t)
        return rhs!(du, u, SciMLBase.unwrap_parameters(p), t)
    end
    wrapped = FunctionWrappersWrappers.FunctionWrappersWrapper(
        SciMLBase.Void(wrapped_rhs!), (typeof((du, [3.0], params, 0.0)),), (Nothing,)
    )
    wrapped_f = ODEFunction{true, SciMLBase.AutoSpecialize}(wrapped)
    wrapped_f(du, [3.0], params, 0.0)
    @test du == [-6.0]
end
