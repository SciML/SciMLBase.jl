using SciMLBase, Test

function test_num_args()
    f(x) = 2x
    f(x, y) = 2xy

    numpar = SciMLBase.numargs(f) # Should be [1,2]
    g = (x, y) -> x^2
    numpar2 = SciMLBase.numargs(g) # [2]
    @show numpar, minimum(numpar) == 1, maximum(numpar) == 2
    minimum(numpar) == 1 && maximum(numpar) == 2 &&
        maximum(numpar2) == 2 &&
        minimum(numpar2) == 2
end

@test test_num_args()

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
ODEFunction{true}(ofboth)
ODEFunction{false}(ofboth)

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
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip,
                                                               Wfact_t = sWfact_t)
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
@test_throws SciMLBase.NonconformingFunctionsError SDEFunction(fiip, giip,
                                                               paramjac = sparamjac)
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
    append!(sol.u_analytic, sol.prob.u0 * exp.(sol.prob.p * sol.t))
end
@test_nowarn RODEFunction(frode)
@test_nowarn RODEFunction(frode, analytic = rode_analytic)
@test_nowarn RODEFunction(frode, analytic = rode_analytic!, analytic_full = true)
@test_throws MethodError RODEFunction(frode, analytic = rode_analytic!,
                                      analytic_full = nothing)

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

# NonlinearFunction

nfoop(u, p) = u
nfiip(du, u, p) = du .= u

nfboth(u, p) = u
nfboth(du, u, p) = du .= u

NonlinearFunction(nfboth)
NonlinearFunction{true}(nfboth)
NonlinearFunction{false}(nfboth)

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
intf(u) = 1.0
@test_throws SciMLBase.TooFewArgumentsError IntegralProblem(intf, 0.0, 1.0)
intf(u, p) = 1.0
IntegralProblem(intf, 0.0, 1.0)

# Optimization

optf(u) = 1.0
@test_throws SciMLBase.TooFewArgumentsError OptimizationFunction(optf)
@test_throws SciMLBase.TooFewArgumentsError OptimizationProblem(optf, 1.0)
optf(u, p) = 1.0
OptimizationFunction(optf)
OptimizationProblem(optf, 1.0)
