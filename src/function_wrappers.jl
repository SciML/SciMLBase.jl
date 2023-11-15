mutable struct TimeGradientWrapper{iip, fType, uType, P} <: AbstractSciMLFunction{iip}
    f::fType
    uprev::uType
    p::P
end

function TimeGradientWrapper(f::F, uprev, p) where {F}
    return TimeGradientWrapper{isinplace(f, 4), F, typeof(uprev), typeof(p)}(f, uprev, p)
end

(ff::TimeGradientWrapper{true})(t) = (du2 = similar(ff.uprev); ff.f(du2, ff.uprev, ff.p, t); du2)
(ff::TimeGradientWrapper{true})(du2, t) = ff.f(du2, ff.uprev, ff.p, t)

(ff::TimeGradientWrapper{false})(t) = ff.f(ff.uprev, ff.p, t)

mutable struct UJacobianWrapper{iip, fType, tType, P} <: AbstractSciMLFunction{iip}
    f::fType
    t::tType
    p::P
end

function UJacobianWrapper(f::F, t, p) where {F}
    return UJacobianWrapper{isinplace(f, 4), F, typeof(t), typeof(p)}(f, t, p)
end

(ff::UJacobianWrapper{true})(du1, uprev) = ff.f(du1, uprev, ff.p, ff.t)
(ff::UJacobianWrapper{true})(uprev) = (du1 = similar(uprev); ff.f(du1, uprev, ff.p, ff.t); du1)
(ff::UJacobianWrapper{true})(du1, uprev, p, t) = ff.f(du1, uprev, p, t)
(ff::UJacobianWrapper{true})(uprev, p, t) = (du1 = similar(uprev); ff.f(du1, uprev, p, t); du1)

(ff::UJacobianWrapper{false})(uprev) = ff.f(uprev, ff.p, ff.t)
(ff::UJacobianWrapper{false})(uprev, p, t) = ff.f(uprev, p, t)

mutable struct TimeDerivativeWrapper{iip, F, uType, P} <: AbstractSciMLFunction{iip}
    f::F
    u::uType
    p::P
end

function TimeDerivativeWrapper(f::F, u, p) where {F}
    return TimeDerivativeWrapper{isinplace(f, 4), F, typeof(u), typeof(p)}(f, u, p)
end

(ff::TimeDerivativeWrapper{false})(t) = ff.f(ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(du1, t) = ff.f(du1, ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(t) = (du1 = similar(ff.u); ff.f(du1, ff.u, ff.p, t); du1)

mutable struct UDerivativeWrapper{iip, F, tType, P} <: AbstractSciMLFunction{iip}
    f::F
    t::tType
    p::P
end

function UDerivativeWrapper(f::F, t, p) where {F}
    return UDerivativeWrapper{isinplace(f, 4), F, typeof(t), typeof(p)}(f, t, p)
end

(ff::UDerivativeWrapper{false})(u) = ff.f(u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(du1, u) = ff.f(du1, u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(u) = (du1 = similar(u); ff.f(du1, u, ff.p, ff.t); du1)

mutable struct ParamJacobianWrapper{iip, fType, tType, uType} <: AbstractSciMLFunction{iip}
    f::fType
    t::tType
    u::uType
end

function ParamJacobianWrapper(f::F, t, u) where {F}
    return ParamJacobianWrapper{isinplace(f, 4), F, typeof(t), typeof(u)}(f, t, u)
end

(ff::ParamJacobianWrapper{true})(du1, p) = ff.f(du1, ff.u, p, ff.t)
function (ff::ParamJacobianWrapper{true})(p)
    du1 = similar(p, size(ff.u))
    ff.f(du1, ff.u, p, ff.t)
    return du1
end
(ff::ParamJacobianWrapper{false})(p) = ff.f(ff.u, p, ff.t)

mutable struct JacobianWrapper{iip, fType, pType} <: AbstractSciMLFunction{iip}
    f::fType
    p::pType
end

function JacobianWrapper(f::F, p) where {F}
    return JacobianWrapper{isinplace(f, 4), F, typeof(p)}(f, p)
end

(uf::JacobianWrapper{false})(u) = uf.f(u, uf.p)
(uf::JacobianWrapper{true})(res, u) = uf.f(res, u, uf.p)
