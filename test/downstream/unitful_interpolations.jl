using OrdinaryDiffEq
import Unitful: @u_str

# w Unitful
rhsU(x, p, t) = p * x
pU = 0.5u"1/hr"
xU0 = 1.0u"g/L"
tspanU = (0.0u"hr", 5.0u"hr")
probU = ODEProblem(rhsU, xU0, tspanU, pU)
solU = solve(probU, Tsit5())
tU = range(tspanU[1], tspanU[2], length = 20)
xU = solU(tU)
