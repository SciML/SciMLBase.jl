using SciMLBase
using SymbolicIndexingInterface

# ODE
function lorenz!(du,u,p,t)
 du[1] = p[1] * (u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
p = [10.0, 28.0, 8/3]
fn = ODEFunction(lorenz!; sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t))
prob = ODEProblem(fn, u0, tspan, p)

@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0, 4.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; u0 = [:x => 2.0, :z => 4.0, :y => 3.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; p = [11.0, 12.0, 13.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = [:a => 11.0, :c => 13.0, :b => 12.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = (11.0, 12.0, 13)).p == (11.0, 12.0, 13)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, 0.0, 0.0]
@test remake(prob; p = [:b => 11.0]).p == [10.0, 11.0, 8/3]

# BVP
g = 9.81
L = 1.0
tspan = (0.0, pi / 2)
function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(p[1] / p[2]) * sin(θ)
end
function bc1!(residual, u, p, t)
    residual[1] = u[end ÷ 2][1] + pi / 2 # the solution at the middle of the time span should be -pi/2
    residual[2] = u[end][1] - pi / 2 # the solution at the end of the time span should be pi/2
end
u0 = [pi / 2, pi / 2]
p = [g, L]
fn = BVPFunction(simplependulum!, bc1!; sys = SymbolCache([:x, :y], [:a, :b], :t) )
prob = BVProblem(fn, u0, tspan, p)

@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0]).u0 == [2.0, 3.0]
@test remake(prob; u0 = [:x => 2.0, :y => 3.0]).u0 == [2.0, 3.0]
@test remake(prob; p = [11.0, 12.0]).p == [11.0, 12.0]
@test remake(prob; p = [:a => 11.0, :b => 12.0]).p == [11.0, 12.0]
@test remake(prob; p = (11.0, 12.0)).p == (11.0, 12.0)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, pi / 2]
@test remake(prob; p = [:b => 11.0]).p == [g, 11.0]

# SDE
function sdef(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end
function sdeg(du, u, p, t)
    du .= 0.1u
end

u0 = [1.0, 0.0, 0.0]
p = [10, 2.33, 26]
tspan = (0, 100)
fn = SDEFunction(sdef, sdeg; sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t))
prob = SDEProblem(fn, u0, tspan, p)

@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0, 4.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; u0 = [:x => 2.0, :z => 4.0, :y => 3.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; p = [11.0, 12.0, 13.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = [:a => 11.0, :c => 13.0, :b => 12.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = (11.0, 12.0, 13)).p == (11.0, 12.0, 13)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, 0.0, 0.0]
@test remake(prob; p = [:b => 11.0]).p == [10.0, 11.0, 26.0]

# OptimizationProblem
function loss(u, p)
    return (p[1] - u[1]) ^ 2 + p[2] * (u[2] - u[1] ^ 2) ^ 2
end
u0 = [1.0, 2.0]
p = [1.0, 100.0]
fn = OptimizationFunction(loss; sys = SymbolCache([:x, :y], [:a, :b], :t) )
prob = OptimizationProblem(fn, u0, p)
@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0]).u0 == [2.0, 3.0]
@test remake(prob; u0 = [:x => 2.0, :y => 3.0]).u0 == [2.0, 3.0]
@test remake(prob; p = [11.0, 12.0]).p == [11.0, 12.0]
@test remake(prob; p = [:a => 11.0, :b => 12.0]).p == [11.0, 12.0]
@test remake(prob; p = (11.0, 12.0)).p == (11.0, 12.0)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, 2.0]
@test remake(prob; p = [:b => 11.0]).p == [1.0, 11.0]

# NonlinearProblem
function nlf(du, u, p)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end
u0 = [1.0, 0.0, 0.0]
p = [10, 2.33, 26]
fn = NonlinearFunction(nlf; sys = SymbolCache([:x, :y, :z], [:a, :b, :c], :t))
prob = NonlinearProblem(fn, u0, p)

@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0, 4.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; u0 = [:x => 2.0, :z => 4.0, :y => 3.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; p = [11.0, 12.0, 13.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = [:a => 11.0, :c => 13.0, :b => 12.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = (11.0, 12.0, 13)).p == (11.0, 12.0, 13)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, 0.0, 0.0]
@test remake(prob; p = [:b => 11.0]).p == [10.0, 11.0, 26.0]

# NonlinearLeastSquaresProblem
prob = NonlinearLeastSquaresProblem(fn, u0, p)
@test remake(prob).u0 == u0
@test remake(prob).p == p
@test remake(prob; u0 = [2.0, 3.0, 4.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; u0 = [:x => 2.0, :z => 4.0, :y => 3.0]).u0 == [2.0, 3.0, 4.0]
@test remake(prob; p = [11.0, 12.0, 13.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = [:a => 11.0, :c => 13.0, :b => 12.0]).p == [11.0, 12.0, 13.0]
@test remake(prob; p = (11.0, 12.0, 13)).p == (11.0, 12.0, 13)
@test remake(prob; u0 = [:x => 2.0]).u0 == [2.0, 0.0, 0.0]
@test remake(prob; p = [:b => 11.0]).p == [10.0, 11.0, 26.0]

