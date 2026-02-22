using SciMLBase, Test

function test_num_args()
    f(x) = 2x
    f(x, y) = 2xy

    numpar = SciMLBase.numargs(f) # Should be [1,2]
    g = (x, y) -> x^2
    numpar2 = SciMLBase.numargs(g) # [2]
    numpar3 = SciMLBase.numargs(sqrt ∘ g) # [2]
    @show numpar, minimum(numpar) == 1, maximum(numpar) == 2
    return minimum(numpar) == 1 && maximum(numpar) == 2 &&
        maximum(numpar2) == 2 &&
        only(numpar3) == 2
end

@test test_num_args()

# Test isinplace on UnionAll
# https://github.com/SciML/SciMLBase.jl/issues/529

struct Foo{T} end
f = Foo{1}()
(this::Foo{T})(args...) where {T} = 1
@test SciMLBase.isinplace(Foo{Int}(), 4)

@testset "isinplace for FunctionWrappersWrapper" begin
    using FunctionWrappersWrappers

    # IIP: all wrappers return Nothing
    iip_fww = FunctionWrappersWrapper(
        (du, u, p, t) -> (du .= u; nothing),
        (Tuple{Vector{Float64}, Vector{Float64}, Nothing, Float64},),
        (Nothing,)
    )
    @test @inferred SciMLBase.isinplace(iip_fww, 4) === true

    # OOP: wrapper returns non-Nothing
    oop_fww = FunctionWrappersWrapper(
        (u, p, t) -> u .* 2,
        (Tuple{Vector{Float64}, Nothing, Float64},),
        (Vector{Float64},)
    )
    @test @inferred SciMLBase.isinplace(oop_fww, 3) === false

    # Multi-variant IIP (like OrdinaryDiffEq uses with 4 dual variants)
    multi_iip = FunctionWrappersWrapper(
        (du, u, p, t) -> (du .= u; nothing),
        (Tuple{Vector{Float64}, Vector{Float64}, Nothing, Float64},
         Tuple{Vector{Float64}, Vector{Float64}, Nothing, Float64}),
        (Nothing, Nothing)
    )
    @test @inferred SciMLBase.isinplace(multi_iip, 4) === true
end

@testset "widen_bounded_type_params" begin
    f = ODEFunction{true, SciMLBase.AutoSpecialize}((du, u, p, t) -> du .= u)
    @test typeof(f).parameters[end-1] === Nothing  # ID is concrete
    @test typeof(f).parameters[end] === Nothing     # NLP is concrete

    widened = @inferred SciMLBase.widen_bounded_type_params(f)

    # Bounded params are widened to their upper bounds
    @test typeof(widened).parameters[end-1] === Union{Nothing, SciMLBase.OverrideInitData}
    @test typeof(widened).parameters[end] === Union{Nothing, SciMLBase.ODENLStepData}

    # Unbounded params stay concrete
    @test SciMLBase.isinplace(widened) === true
    @test SciMLBase.specialization(widened) === SciMLBase.AutoSpecialize

    # All field values preserved
    for fname in fieldnames(typeof(f))
        @test getfield(f, fname) === getfield(widened, fname)
    end
end

@testset "isinplace accepts an out-of-place version with different numbers of parameters " begin
    f1(u) = 2 * u
    @test !isinplace(f1, 2)
    @test_throws SciMLBase.TooFewArgumentsError SciMLBase.isinplace(f1, 4)
    @test !isinplace(f1, 4; outofplace_param_number = 1)
end

## Problem argument tests

ftoomany(u, p, t, x, y) = 2u
u0 = 0.5
tspan = (0.0, 1.0)
@test_throws SciMLBase.TooManyArgumentsError ODEProblem(ftoomany, u0, tspan)

ftoofew(u, t) = 2u
@test_throws SciMLBase.TooFewArgumentsError ODEProblem(ftoofew, u0, tspan)

fmessedup(u, t) = 2u
fmessedup(u, p, t, x, y) = 2u
@test_throws SciMLBase.FunctionArgumentsError ODEProblem(fmessedup, u0, tspan)

# Test SciMLFunctions

foop(u, p, t) = u
fiip(du, u, p, t) = du .= u

ofboth(u, p, t) = u
ofboth(du, u, p, t) = du .= u

ODEFunction(ofboth)
@inferred ODEFunction{true}(ofboth)
@inferred ODEFunction{false}(ofboth)

jac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, jac = jac)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, jac = jac)
jac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, jac = jac)
ODEFunction(foop, jac = jac)
jac(du, u, p, t) = [1.0]
ODEFunction(fiip, jac = jac)
ODEFunction(foop, jac = jac)

Wfact(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, Wfact = Wfact)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, Wfact = Wfact)
Wfact(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, Wfact = Wfact)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, Wfact = Wfact)
Wfact(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, Wfact = Wfact)
ODEFunction(foop, Wfact = Wfact)
Wfact(du, u, p, gamma, t) = [1.0]
ODEFunction(fiip, Wfact = Wfact)
ODEFunction(foop, Wfact = Wfact)

Wfact_t(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, Wfact_t = Wfact_t)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, Wfact_t = Wfact_t)
Wfact_t(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, Wfact_t = Wfact_t)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, Wfact_t = Wfact_t)
Wfact_t(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, Wfact_t = Wfact_t)
ODEFunction(foop, Wfact_t = Wfact_t)
Wfact_t(du, u, p, gamma, t) = [1.0]
ODEFunction(fiip, Wfact_t = Wfact_t)
ODEFunction(foop, Wfact_t = Wfact_t)

tgrad(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, tgrad = tgrad)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, tgrad = tgrad)
tgrad(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, tgrad = tgrad)
ODEFunction(foop, tgrad = tgrad)
tgrad(du, u, p, t) = [1.0]
ODEFunction(fiip, tgrad = tgrad)
ODEFunction(foop, tgrad = tgrad)

paramjac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, paramjac = paramjac)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, paramjac = paramjac)
paramjac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, paramjac = paramjac)
ODEFunction(foop, paramjac = paramjac)
paramjac(du, u, p, t) = [1.0]
ODEFunction(fiip, paramjac = paramjac)
ODEFunction(foop, paramjac = paramjac)

jvp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, jvp = jvp)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, jvp = jvp)
jvp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, jvp = jvp)
ODEFunction(foop, jvp = jvp)
jvp(du, u, v, p, t) = [1.0]
ODEFunction(fiip, jvp = jvp)
ODEFunction(foop, jvp = jvp)

vjp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(fiip, vjp = vjp)
@test_throws SciMLBase.TooFewArgumentsError ODEFunction(foop, vjp = vjp)
vjp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError ODEFunction(fiip, vjp = vjp)
ODEFunction(foop, vjp = vjp)
vjp(du, u, v, p, t) = [1.0]
ODEFunction(fiip, vjp = vjp)
ODEFunction(foop, vjp = vjp)

# SDE

foop(u, p, t) = u
goop(u, p, t) = u

fiip(du, u, p, t) = du .= u
giip(du, u, p, t) = du .= u

SDEFunction(fiip, giip)
SDEFunction(foop, goop)
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(foop, giip)
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, goop)

sfboth(u, p, t) = u
sfboth(du, u, p, t) = du .= u
sgboth(u, p, t) = u
sgboth(du, u, p, t) = du .= u

SDEFunction(sfboth, sgboth)
SDEFunction{true}(sfboth, sgboth)
SDEFunction{false}(sfboth, sgboth)

sjac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, jac = sjac)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, jac = sjac)
sjac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip, jac = sjac)
SDEFunction(foop, goop, jac = sjac)
sjac(du, u, p, t) = [1.0]
SDEFunction(fiip, giip, jac = sjac)
SDEFunction(foop, goop, jac = sjac)

sWfact(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, Wfact = sWfact)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, Wfact = sWfact)
sWfact(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, Wfact = sWfact)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, goop, Wfact = sWfact)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, Wfact = sWfact)
sWfact(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip, Wfact = sWfact)
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, goop, Wfact = sWfact)
SDEFunction(foop, goop, Wfact = sWfact)
sWfact(du, u, p, gamma, t) = [1.0]
SDEFunction(fiip, giip, Wfact = sWfact)
SDEFunction(foop, goop, Wfact = sWfact)

sWfact_t(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, Wfact_t = sWfact_t)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, giip, Wfact_t = sWfact_t)
sWfact_t(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, Wfact_t = sWfact_t)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, giip, Wfact_t = sWfact_t)
sWfact_t(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(
    fiip, giip,
    Wfact_t = sWfact_t
)
SDEFunction(foop, goop, Wfact_t = sWfact_t)
sWfact_t(du, u, p, gamma, t) = [1.0]
SDEFunction(fiip, giip, Wfact_t = sWfact_t)
SDEFunction(foop, goop, Wfact_t = sWfact_t)

stgrad(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, tgrad = stgrad)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, tgrad = stgrad)
stgrad(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip, tgrad = stgrad)
SDEFunction(foop, goop, tgrad = stgrad)
stgrad(du, u, p, t) = [1.0]
SDEFunction(fiip, giip, tgrad = stgrad)
SDEFunction(foop, goop, tgrad = stgrad)

sparamjac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, paramjac = sparamjac)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, paramjac = sparamjac)
sparamjac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(
    fiip, giip,
    paramjac = sparamjac
)
SDEFunction(foop, goop, paramjac = sparamjac)
sparamjac(du, u, p, t) = [1.0]
SDEFunction(fiip, giip, paramjac = sparamjac)
SDEFunction(foop, goop, paramjac = sparamjac)

sjvp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, jvp = sjvp)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, jvp = sjvp)
sjvp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip, jvp = sjvp)
SDEFunction(foop, goop, jvp = sjvp)
sjvp(du, u, v, p, t) = [1.0]
SDEFunction(fiip, giip, jvp = sjvp)
SDEFunction(foop, goop, jvp = sjvp)

svjp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(fiip, giip, vjp = svjp)
@test_throws SciMLBase.TooFewArgumentsError SDEFunction(foop, goop, vjp = svjp)
svjp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip, vjp = svjp)
SDEFunction(foop, goop, vjp = svjp)
svjp(du, u, v, p, t) = [1.0]
SDEFunction(fiip, giip, vjp = svjp)
SDEFunction(foop, goop, vjp = svjp)

# RODEFunction

froop(u, p, t, W) = W
friip(du, p, t, W) = (du .= W)

RODEFunction(froop)
RODEFunction(friip)

frboth(u, p, t, W) = W
frboth(du, u, p, t, W) = (du .= W)

@test_nowarn RODEFunction(frboth)
@test_nowarn RODEFunction{true}(frboth)
@test_nowarn RODEFunction{false}(frboth)

frode(u, p, t, W) = p * u
rode_analytic(u0, t, p, W) = u0 * exp(p * t)
function rode_analytic!(sol)
    empty!(sol.u_analytic)
    return append!(sol.u_analytic, sol.prob.u0 * exp.(sol.prob.p * sol.t))
end
@test_nowarn RODEFunction(frode)
@test_nowarn RODEFunction(frode, analytic = rode_analytic)
@test_nowarn RODEFunction(frode, analytic = rode_analytic!, analytic_full = true)
@test_throws MethodError RODEFunction(
    frode, analytic = rode_analytic!,
    analytic_full = nothing
)

# DAEFunction

dfoop(du, u, p, t) = du .+ u
dfiip(res, du, u, p, t) = res .= du .+ u

dfboth(du, u, p, t) = du .+ u
dfboth(res, du, u, p, t) = res .= du .+ u

DAEFunction(dfboth)
DAEFunction{true}(dfboth)
DAEFunction{false}(dfboth)

djac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jac = djac)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jac = djac)
djac(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jac = djac)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jac = djac)
djac(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jac = djac)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jac = djac)
djac(du, u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DAEFunction(dfiip, jac = djac)
DAEFunction(dfoop, jac = djac)
djac(res, du, u, p, gamma, t) = [1.0]
DAEFunction(dfiip, jac = djac)
DAEFunction(dfoop, jac = djac)

djvp(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jvp = djvp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jvp = djvp)
djvp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jvp = djvp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jvp = djvp)
djvp(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jvp = djvp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jvp = djvp)
djvp(du, u, v, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, jvp = djvp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, jvp = djvp)
djvp(du, u, v, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DAEFunction(dfiip, jvp = djvp)
DAEFunction(dfoop, jvp = djvp)
djvp(res, du, u, v, p, gamma, t) = [1.0]
DAEFunction(dfiip, jvp = djvp)
DAEFunction(dfoop, jvp = djvp)

dvjp(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, vjp = dvjp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, vjp = dvjp)
dvjp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, vjp = dvjp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, vjp = dvjp)
dvjp(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, vjp = dvjp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, vjp = dvjp)
dvjp(du, u, v, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfiip, vjp = dvjp)
@test_throws SciMLBase.TooFewArgumentsError DAEFunction(dfoop, vjp = dvjp)
dvjp(du, u, v, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DAEFunction(dfiip, vjp = dvjp)
DAEFunction(dfoop, vjp = dvjp)
dvjp(res, du, u, v, p, gamma, t) = [1.0]
DAEFunction(dfiip, vjp = dvjp)
DAEFunction(dfoop, vjp = dvjp)
DAEFunction{true, SciMLBase.NoSpecialize}(dfiip, observed = 1)

# DDEFunction

ddefoop(u, h, p, t) = u
ddefiip(du, u, h, p, t) = du .= u

ddeofboth(u, h, p, t) = u
ddeofboth(du, u, h, p, t) = du .= u

DDEFunction(ddeofboth)
DDEFunction{true}(ddeofboth)
DDEFunction{false}(ddeofboth)

ddejac(u, h, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, jac = ddejac)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, jac = ddejac)
ddejac(u, h, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(ddefiip, jac = ddejac)
DDEFunction(ddefoop, jac = ddejac)
ddejac(du, u, h, p, t) = [1.0]
DDEFunction(ddefiip, jac = ddejac)
DDEFunction(ddefoop, jac = ddejac)

ddeWfact(u, h, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, Wfact = ddeWfact)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, Wfact = ddeWfact)
ddeWfact(u, h, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, Wfact = ddeWfact)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, Wfact = ddeWfact)
ddeWfact(u, h, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(ddefiip, Wfact = ddeWfact)
DDEFunction(ddefoop, Wfact = ddeWfact)
ddeWfact(du, u, h, p, gamma, t) = [1.0]
DDEFunction(ddefiip, Wfact = ddeWfact)
DDEFunction(ddefoop, Wfact = ddeWfact)

ddeWfact_t(u, h, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, Wfact_t = ddeWfact_t)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, Wfact_t = ddeWfact_t)
ddeWfact_t(u, h, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, Wfact_t = ddeWfact_t)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, Wfact_t = ddeWfact_t)
ddeWfact_t(u, h, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(
    ddefiip,
    Wfact_t = ddeWfact_t
)
DDEFunction(ddefoop, Wfact_t = Wfact_t)
ddeWfact_t(du, u, h, p, gamma, t) = [1.0]
DDEFunction(ddefiip, Wfact_t = ddeWfact_t)
DDEFunction(ddefoop, Wfact_t = ddeWfact_t)

ddetgrad(u, h, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, tgrad = ddetgrad)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, tgrad = ddetgrad)
ddetgrad(u, h, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(ddefiip, tgrad = ddetgrad)
DDEFunction(ddefoop, tgrad = ddetgrad)
ddetgrad(du, u, h, p, t) = [1.0]
DDEFunction(ddefiip, tgrad = ddetgrad)
DDEFunction(ddefoop, tgrad = ddetgrad)

ddeparamjac(u, h, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, paramjac = ddeparamjac)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, paramjac = ddeparamjac)
ddeparamjac(u, h, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(
    ddefiip,
    paramjac = ddeparamjac
)
DDEFunction(ddefoop, paramjac = paramjac)
ddeparamjac(du, u, h, p, t) = [1.0]
DDEFunction(ddefiip, paramjac = ddeparamjac)
DDEFunction(ddefoop, paramjac = ddeparamjac)

ddejvp(u, h, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, jvp = ddejvp)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, jvp = ddejvp)
ddejvp(u, v, h, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(ddefiip, jvp = ddejvp)
DDEFunction(ddefoop, jvp = ddejvp)
ddejvp(du, u, v, h, p, t) = [1.0]
DDEFunction(ddefiip, jvp = ddejvp)
DDEFunction(ddefoop, jvp = ddejvp)

ddevjp(u, h, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefiip, vjp = ddevjp)
@test_throws SciMLBase.TooFewArgumentsError DDEFunction(ddefoop, vjp = ddevjp)
ddevjp(u, v, h, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DDEFunction(ddefiip, vjp = ddevjp)
DDEFunction(ddefoop, vjp = ddevjp)
ddevjp(du, u, v, h, p, t) = [1.0]
DDEFunction(ddefiip, vjp = ddevjp)
DDEFunction(ddefoop, vjp = ddevjp)

# NonlinearFunction

nfoop(u, p) = u
nfiip(du, u, p) = du .= u

nfboth(u, p) = u
nfboth(du, u, p) = du .= u

NonlinearFunction(nfboth)
@inferred NonlinearFunction{true}(nfboth)
@inferred NonlinearFunction{false}(nfboth)

njac(u) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfiip, jac = njac)
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfoop, jac = njac)
njac(u, p) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError NonlinearFunction(nfiip, jac = njac)
NonlinearFunction(nfoop, jac = njac)
njac(du, u, p) = [1.0]
NonlinearFunction(nfiip, jac = njac)
NonlinearFunction(nfoop, jac = njac)

njvp(u) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfiip, jvp = njvp)
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfoop, jvp = njvp)
njvp(u, p) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfiip, jvp = njvp)
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfoop, jvp = njvp)
njvp(u, v, p) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError NonlinearFunction(nfiip, jvp = njvp)
NonlinearFunction(nfoop, jvp = njvp)
njvp(du, u, v, p) = [1.0]
NonlinearFunction(nfiip, jvp = njvp)
NonlinearFunction(nfoop, jvp = njvp)

nvjp(u) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfiip, vjp = nvjp)
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfoop, vjp = nvjp)
nvjp(u, p) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfiip, vjp = nvjp)
@test_throws SciMLBase.TooFewArgumentsError NonlinearFunction(nfoop, vjp = nvjp)
nvjp(u, v, p) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError NonlinearFunction(nfiip, vjp = nvjp)
NonlinearFunction(nfoop, vjp = nvjp)
nvjp(du, u, v, p) = [1.0]
NonlinearFunction(nfiip, vjp = nvjp)
NonlinearFunction(nfoop, vjp = nvjp)

# Integrals
intfew(u) = 1.0
@test_throws SciMLBase.TooFewArgumentsError IntegralProblem(intfew, (0.0, 1.0))
@test_throws SciMLBase.TooFewArgumentsError IntegralFunction(intfew)
@test_throws SciMLBase.TooFewArgumentsError IntegralFunction(intfew, zeros(3))
@test_throws SciMLBase.TooFewArgumentsError BatchIntegralFunction(intfew)
@test_throws SciMLBase.TooFewArgumentsError BatchIntegralFunction(intfew, zeros(3))
intf(u, p) = 1.0
p = 2.0
intfiip(y, u, p) = y .= 1.0

for (f, kws, iip) in (
            (intf, (;), false),
            (IntegralFunction(intf), (;), false),
            (IntegralFunction(intf, 1.0), (;), false),
            (IntegralFunction(intfiip, zeros(3)), (;), true),
        ),
        domain in (((0.0, 1.0),), (([0.0], [1.0]),))

    IntegralProblem(f, domain...; kws...)
    IntegralProblem(f, domain..., p; kws...)
    IntegralProblem{iip}(f, domain...; kws...)
    IntegralProblem{iip}(f, domain..., p; kws...)
end

x = [1.0, 2.0]
y = rand(2, 2)
SampledIntegralProblem(y, x)
SampledIntegralProblem(y, x; dim = 2)

# Optimization

optf(u) = 1.0
@test_throws SciMLBase.TooFewArgumentsError OptimizationFunction(optf)
@test_throws SciMLBase.TooFewArgumentsError OptimizationProblem(optf, 1.0)
optf(u, p) = 1.0
OptimizationFunction(optf)
OptimizationProblem(optf, 1.0)

# Test OptimizationFunction{iip} respects explicit iip parameter
# This function would fail isinplace check due to method ambiguity
struct ProblematicOptFunction end
(::ProblematicOptFunction)(x, p) = sum(x .^ 2)
# But when iip is explicitly provided, it should work without calling isinplace
@test_nowarn OptimizationFunction{false}(ProblematicOptFunction())
@test_nowarn OptimizationFunction{true}(ProblematicOptFunction())

# BVPFunction

bfoop(u, p, t) = u
bfiip(du, u, p, t) = du .= u

bfboth(u, p, t) = u
bfboth(du, u, p, t) = du .= u

bcoop(u, p, t) = u
bciip(res, u, p, t) = res .= u

bcfboth(u, p, t) = u
bcfboth(du, u, p, t) = du .= u

BVPFunction(bfboth, bcfboth)
@inferred BVPFunction{true}(bfboth, bcfboth)
@inferred BVPFunction{false}(bfboth, bcfboth)

bjac(u, t) = [1.0]
bcjac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(
    bfiip,
    bciip,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(
    bfoop,
    bciip,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(
    bfiip,
    bcoop,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(
    bfoop,
    bcoop,
    jac = bjac,
    bcjac = bcjac
)
bjac(u, p, t) = [1.0]
bcjac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip,
    bcoop,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip,
    bciip,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    jac = bjac,
    bcjac = bcjac
)
BVPFunction(bfoop, bcoop, jac = bjac)
bjac(du, u, p, t) = [1.0]
bcjac(du, u, p, t) = [1.0]
BVPFunction(bfiip, bciip, jac = bjac, bcjac = bcjac)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    jac = bjac,
    bcjac = bcjac
)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip,
    bcoop,
    jac = bjac,
    bcjac = bcjac
)
BVPFunction(bfoop, bcoop, jac = bjac, bcjac = bcjac)

bWfact(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, Wfact = bWfact)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, Wfact = bWfact)
bWfact(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, Wfact = bWfact)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, Wfact = bWfact)
bWfact(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfiip, bciip, Wfact = bWfact)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, Wfact = bWfact)
bWfact(du, u, p, gamma, t) = [1.0]
BVPFunction(bfiip, bciip, Wfact = bWfact)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, Wfact = bWfact)

bWfact_t(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, Wfact_t = bWfact_t)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, Wfact_t = bWfact_t)
bWfact_t(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, Wfact_t = bWfact_t)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, Wfact_t = bWfact_t)
bWfact_t(u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip,
    bciip,
    Wfact_t = bWfact_t
)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    Wfact_t = bWfact_t
)
bWfact_t(du, u, p, gamma, t) = [1.0]
BVPFunction(bfiip, bciip, Wfact_t = bWfact_t)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    Wfact_t = bWfact_t
)

btgrad(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, tgrad = btgrad)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, tgrad = btgrad)
btgrad(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfiip, bciip, tgrad = btgrad)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, tgrad = btgrad)
btgrad(du, u, p, t) = [1.0]
BVPFunction(bfiip, bciip, tgrad = btgrad)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, tgrad = btgrad)

bparamjac(u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, paramjac = bparamjac)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, paramjac = bparamjac)
bparamjac(u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip,
    bciip,
    paramjac = bparamjac
)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    paramjac = bparamjac
)
bparamjac(du, u, p, t) = [1.0]
BVPFunction(bfiip, bciip, paramjac = bparamjac)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfoop,
    bciip,
    paramjac = bparamjac
)

bjvp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, jvp = bjvp)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, jvp = bjvp)
bjvp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfiip, bciip, jvp = bjvp)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, jvp = bjvp)
bjvp(du, u, v, p, t) = [1.0]
BVPFunction(bfiip, bciip, jvp = bjvp)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, jvp = bjvp)

bvjp(u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfiip, bciip, vjp = bvjp)
@test_throws SciMLBase.TooFewArgumentsError BVPFunction(bfoop, bciip, vjp = bvjp)
bvjp(u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfiip, bciip, vjp = bvjp)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, vjp = bvjp)
bvjp(du, u, v, p, t) = [1.0]
BVPFunction(bfiip, bciip, vjp = bvjp)

@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfoop, bciip, vjp = bvjp)

BVPFunction(bfiip, bciip, cost = (x, p) -> 0.0)
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(bfiip, bciip, cost = x -> 0.0)
equality(u, p) = u
inequality(u, p) = u
@test_throws SciMLBase.NonconformingFunctionsError BVPFunction(
    bfiip, bciip, cost = (x, p) -> 0.0, equality = equality, inequality = inequality
)
equality(res, u, p) = (res .= u)
inequality(res, u, p) = (res .= u)
BVPFunction(
    bfiip, bciip, cost = (x, p) -> 0.0, equality = equality, inequality = inequality
)

# DynamicalBVPFunction

dbfoop(du, u, p, t) = u
dbfiip(ddu, du, u, p, t) = ddu .= du .- u

dbfboth(du, u, p, t) = u
dbfboth(ddu, du, u, p, t) = ddu .= du .- u

dbcoop(du, u, p, t) = u
dbciip(res, du, u, p, t) = res .= du .- u

dbcfboth(du, u, p, t) = u
dbcfboth(res, du, u, p, t) = res .= du .- u

DynamicalBVPFunction(dbfboth, dbcfboth)
@inferred DynamicalBVPFunction{true}(dbfboth, dbcfboth)
@inferred DynamicalBVPFunction{false}(dbfboth, dbcfboth)

dbjac(du, u, t) = [1.0]
dbcjac(du, u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip,
    dbciip,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip,
    dbcoop,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop,
    dbcoop,
    jac = dbjac,
    bcjac = dbcjac
)
dbjac(du, u, p, t) = [1.0]
dbcjac(du, u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip,
    dbcoop,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip,
    dbciip,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    jac = dbjac,
    bcjac = dbcjac
)
DynamicalBVPFunction(dbfoop, dbcoop, jac = dbjac)
dbjac(ddu, du, u, p, t) = [1.0]
dbcjac(ddu, du, u, p, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, jac = dbjac, bcjac = dbcjac)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    jac = dbjac,
    bcjac = dbcjac
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip,
    dbcoop,
    jac = dbjac,
    bcjac = dbcjac
)
DynamicalBVPFunction(dbfoop, dbcoop, jac = dbjac, bcjac = dbcjac)

dbWfact(du, u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, Wfact = dbWfact
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact = dbWfact
)
dbWfact(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, Wfact = dbWfact
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact = dbWfact
)
dbWfact(du, u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip, dbciip, Wfact = dbWfact
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact = dbWfact
)
dbWfact(ddu, du, u, p, gamma, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, Wfact = dbWfact)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact = dbWfact
)

dbWfact_t(du, u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, Wfact_t = dbWfact_t
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact_t = dbWfact_t
)
dbWfact_t(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, Wfact_t = dbWfact_t
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, Wfact_t = dbWfact_t
)
dbWfact_t(du, u, p, gamma, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip,
    dbciip,
    Wfact_t = dbWfact_t
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    Wfact_t = dbWfact_t
)
dbWfact_t(ddu, du, u, p, gamma, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, Wfact_t = dbWfact_t)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    Wfact_t = dbWfact_t
)

dbtgrad(du, u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, tgrad = dbtgrad
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, tgrad = dbtgrad
)
dbtgrad(du, u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip, dbciip, tgrad = dbtgrad
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, tgrad = dbtgrad
)
dbtgrad(ddu, du, u, p, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, tgrad = dbtgrad)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, tgrad = dbtgrad
)

dbparamjac(du, u, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, paramjac = dbparamjac
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, paramjac = dbparamjac
)
dbparamjac(du, u, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip,
    dbciip,
    paramjac = dbparamjac
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    paramjac = dbparamjac
)
dbparamjac(ddu, du, u, p, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, paramjac = dbparamjac)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop,
    dbciip,
    paramjac = dbparamjac
)

dbjvp(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, jvp = dbjvp
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, jvp = dbjvp
)
dbjvp(du, u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip, dbciip, jvp = dbjvp
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, jvp = dbjvp
)
dbjvp(ddu, du, u, v, p, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, jvp = dbjvp)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, jvp = dbjvp
)

dbvjp(du, u, p, t) = [1.0]
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfiip, dbciip, vjp = dbvjp
)
@test_throws SciMLBase.TooFewArgumentsError DynamicalBVPFunction(
    dbfoop, dbciip, vjp = dbvjp
)
dbvjp(du, u, v, p, t) = [1.0]
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfiip, dbciip, vjp = dbvjp
)
@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, vjp = dbvjp
)
dbvjp(ddu, du, u, v, p, t) = [1.0]
DynamicalBVPFunction(dbfiip, dbciip, vjp = dbvjp)

@test_throws SciMLBase.NonconformingFunctionsError DynamicalBVPFunction(
    dbfoop, dbciip, vjp = dbvjp
)

# IntegralFunction

ioop(u, p) = p * u
iiip(y, u, p) = y .= u * p
i1(u) = u
itoo(y, u, p, a) = y .= u * p

IntegralFunction(ioop)
IntegralFunction(ioop, 0.0)
IntegralFunction(iiip, Float64[])

@test_throws SciMLBase.IntegrandMismatchFunctionError IntegralFunction(iiip)
@test_throws SciMLBase.TooFewArgumentsError IntegralFunction(i1)
@test_throws SciMLBase.TooManyArgumentsError IntegralFunction(itoo)
@test_throws SciMLBase.TooManyArgumentsError IntegralFunction(itoo, Float64[])

# BatchIntegralFunction

boop(u, p) = p .* u
biip(y, u, p) = y .= p .* u
bi1(u) = u
bitoo(y, u, p, a) = y .= p .* u

BatchIntegralFunction(boop)
BatchIntegralFunction(boop, max_batch = 20)
BatchIntegralFunction(boop, Float64[])
BatchIntegralFunction(boop, Float64[], max_batch = 20)
BatchIntegralFunction(biip, Float64[])
BatchIntegralFunction(biip, Float64[], max_batch = 20)

@test_throws SciMLBase.IntegrandMismatchFunctionError BatchIntegralFunction(biip)
@test_throws SciMLBase.TooFewArgumentsError BatchIntegralFunction(bi1)
@test_throws SciMLBase.TooManyArgumentsError BatchIntegralFunction(bitoo)
@test_throws SciMLBase.TooManyArgumentsError BatchIntegralFunction(bitoo, Float64[])
