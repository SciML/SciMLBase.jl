using OrdinaryDiffEq
import Unitful: @u_str

# w/o Unitful
rhs(x,p,t) = p*x
p = 0.5
x0 = 1.0
tspan = (0.0, 5.0)
prob = ODE.ODEProblem(rhs, x0, tspan, p)
sol = ODE.solve(prob, ODE.Tsit5())
t = range(tspan[1], tspan[2], length=20)
x = sol(t)

# w Unitful
rhsU(x,p,t) = p*x
pU = 0.5u"1/hr"
xU0 = 1.0u"g/L"
tspanU = (0.0u"hr", 5.0u"hr")
probU = ODEProblem(rhsU, xU0, tspanU, pU)
solU = solve(probU, ODE.Tsit5())
tU = range(tspanU[1], tspanU[2], length=20)
xU = solU(tU)
