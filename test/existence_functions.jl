using Test, SciMLBase
using SciMLBase: __has_jac, __has_tgrad, __has_Wfact, __has_Wfact_t,
                 __has_paramjac, __has_analytic, __has_colorvec, has_jac,
                 has_tgrad,
                 has_Wfact, has_Wfact_t, has_paramjac, has_analytic, has_colorvec,
                 AbstractDiffEqFunction

struct Foo <: AbstractDiffEqFunction{false}
    jac::Any
    tgrad::Any
    Wfact::Any
    Wfact_t::Any
    paramjac::Any
    analytic::Any
    colorvec::Any
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
