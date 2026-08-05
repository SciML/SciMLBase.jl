using Test, SciMLBase
using SciMLBase: __has_jac, __has_tgrad, __has_Wfact, __has_Wfact_t,
    __has_paramjac, __has_analytic, __has_colorvec, has_jac,
    has_tgrad,
    has_Wfact, has_Wfact_t, has_paramjac, has_vjp_p, has_observed, has_analytic,
    has_colorvec, has_sys,
    AbstractDiffEqFunction
using SymbolicIndexingInterface: SymbolCache

struct Foo <: AbstractDiffEqFunction{false}
    jac::Any
    tgrad::Any
    Wfact::Any
    Wfact_t::Any
    paramjac::Any
    analytic::Any
    colorvec::Any
end

struct SensitivityTraitFunction <: AbstractDiffEqFunction{false}
    paramjac::Any
    vjp_p::Any
    observed::Any
end

@testset "Sensitivity function trait interface" begin
    f = SensitivityTraitFunction(:paramjac, :vjp_p, :observed)
    @test has_paramjac(f)
    @test has_vjp_p(f)
    @test has_observed(f)

    missing = SensitivityTraitFunction(nothing, nothing, nothing)
    @test !has_paramjac(missing)
    @test !has_vjp_p(missing)
    @test !has_observed(missing)
end

@testset "Documented function trait interface" begin
    if isdefined(Base, :ispublic)
        for name in (
                :has_analytic, :has_jac, :has_jvp, :has_vjp, :has_tgrad,
                :has_initialization_data,
            )
            @test Base.ispublic(SciMLBase, name)
            @test Base.Docs.hasdoc(SciMLBase, name)
        end
    end
end

@testset "Sensitivity function wrapper interface" begin
    pf = SciMLBase.ParamJacobianWrapper((u, p, t) -> p .* u, 0.0, [2.0])
    @test pf([3.0]) == [6.0]
    out = zeros(1)
    @test pf(out, [4.0]) === out
    @test out == [8.0]

    values = Int[]
    wrapped = SciMLBase.Void(x -> push!(values, x))
    @test wrapped(1) === nothing
    @test values == [1]
end

f = Foo(1, 1, 1, 1, 1, 1, 1)

@test __has_jac(f)
@test __has_tgrad(f)
@test __has_Wfact(f)
@test __has_Wfact_t(f)
@test __has_paramjac(f)
@test __has_analytic(f)
@test __has_colorvec(f)

@test has_jac(f)
@test has_tgrad(f)
@test has_Wfact(f)
@test has_Wfact_t(f)
@test has_paramjac(f)
@test has_analytic(f)
@test has_colorvec(f)

struct Foo2 <: AbstractDiffEqFunction{false}
    jac::Any
    tgrad::Any
    Wfact::Any
    Wfact_t::Any
end

f2 = Foo2(1, 1, nothing, nothing)

@test __has_jac(f2)
@test __has_tgrad(f2)
@test __has_Wfact(f2)
@test __has_Wfact_t(f2)
@test !__has_paramjac(f2)
@test !__has_analytic(f2)
@test !__has_colorvec(f2)

@test has_jac(f2)
@test has_tgrad(f2)
@test !has_Wfact(f2)
@test !has_Wfact_t(f2)
@test !has_paramjac(f2)
@test !has_analytic(f2)
@test !has_colorvec(f2)

@testset "has_sys" begin
    fsys = NonlinearFunction((u, p) -> u; sys = SymbolCache([:x], [:a]))
    @test has_sys(fsys)
    @test !has_sys(NonlinearFunction((u, p) -> u))
    # No `sys` field at all, so the field-existence branch has to short-circuit.
    @test !has_sys(f)
end
