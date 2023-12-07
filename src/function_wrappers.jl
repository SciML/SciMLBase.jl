mutable struct TimeGradientWrapper{iip, fType, uType, P} <: AbstractSciMLFunction{iip}
    f::fType
    uprev::uType
    p::P
end

function TimeGradientWrapper{iip}(f::F, uprev, p) where {F, iip}
    return TimeGradientWrapper{iip, F, typeof(uprev), typeof(p)}(f, uprev, p)
end
function TimeGradientWrapper(f::F, uprev, p) where {F}
    return TimeGradientWrapper{isinplace(f, 4)}(f, uprev, p)
end

function (ff::TimeGradientWrapper{true})(t)
    (du2 = similar(ff.uprev); ff.f(du2, ff.uprev, ff.p, t); du2)
end
(ff::TimeGradientWrapper{true})(du2, t) = ff.f(du2, ff.uprev, ff.p, t)

(ff::TimeGradientWrapper{false})(t) = ff.f(ff.uprev, ff.p, t)

mutable struct UJacobianWrapper{iip, fType, tType, P} <: AbstractSciMLFunction{iip}
    f::fType
    t::tType
    p::P
end

function UJacobianWrapper{iip}(f::F, t, p) where {F, iip}
    return UJacobianWrapper{iip, F, typeof(t), typeof(p)}(f, t, p)
end
UJacobianWrapper(f::F, t, p) where {F} = UJacobianWrapper{isinplace(f, 4)}(f, t, p)

(ff::UJacobianWrapper{true})(du1, uprev) = ff.f(du1, uprev, ff.p, ff.t)
function (ff::UJacobianWrapper{true})(uprev)
    (du1 = similar(uprev); ff.f(du1, uprev, ff.p, ff.t); du1)
end
(ff::UJacobianWrapper{true})(du1, uprev, p, t) = ff.f(du1, uprev, p, t)
function (ff::UJacobianWrapper{true})(uprev, p, t)
    (du1 = similar(uprev); ff.f(du1, uprev, p, t); du1)
end

(ff::UJacobianWrapper{false})(uprev) = ff.f(uprev, ff.p, ff.t)
(ff::UJacobianWrapper{false})(uprev, p, t) = ff.f(uprev, p, t)

mutable struct TimeDerivativeWrapper{iip, F, uType, P} <: AbstractSciMLFunction{iip}
    f::F
    u::uType
    p::P
end

function TimeDerivativeWrapper{iip}(f::F, u, p) where {F, iip}
    return TimeDerivativeWrapper{iip, F, typeof(u), typeof(p)}(f, u, p)
end
function TimeDerivativeWrapper(f::F, u, p) where {F}
    return TimeDerivativeWrapper{isinplace(f, 4)}(f, u, p)
end

(ff::TimeDerivativeWrapper{false})(t) = ff.f(ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(du1, t) = ff.f(du1, ff.u, ff.p, t)
(ff::TimeDerivativeWrapper{true})(t) = (du1 = similar(ff.u); ff.f(du1, ff.u, ff.p, t); du1)

mutable struct UDerivativeWrapper{iip, F, tType, P} <: AbstractSciMLFunction{iip}
    f::F
    t::tType
    p::P
end

function UDerivativeWrapper{iip}(f::F, t, p) where {F, iip}
    return UDerivativeWrapper{iip, F, typeof(t), typeof(p)}(f, t, p)
end
UDerivativeWrapper(f::F, t, p) where {F} = UDerivativeWrapper{isinplace(f, 4)}(f, t, p)

(ff::UDerivativeWrapper{false})(u) = ff.f(u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(du1, u) = ff.f(du1, u, ff.p, ff.t)
(ff::UDerivativeWrapper{true})(u) = (du1 = similar(u); ff.f(du1, u, ff.p, ff.t); du1)

mutable struct ParamJacobianWrapper{iip, fType, tType, uType} <: AbstractSciMLFunction{iip}
    f::fType
    t::tType
    u::uType
end

function ParamJacobianWrapper{iip}(f::F, t, u) where {F, iip}
    return ParamJacobianWrapper{iip, F, typeof(t), typeof(u)}(f, t, u)
end
ParamJacobianWrapper(f::F, t, u) where {F} = ParamJacobianWrapper{isinplace(f, 4)}(f, t, u)

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

JacobianWrapper{iip}(f::F, p) where {F, iip} = JacobianWrapper{iip, F, typeof(p)}(f, p)
JacobianWrapper(f::F, p) where {F} = JacobianWrapper{isinplace(f, 3)}(f, p)

(uf::JacobianWrapper{false})(u) = uf.f(u, uf.p)
(uf::JacobianWrapper{false})(res, u) = (vec(res) .= vec(uf.f(u, uf.p)))
(uf::JacobianWrapper{true})(res, u) = uf.f(res, u, uf.p)
