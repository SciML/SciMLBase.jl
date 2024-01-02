using ModelingToolkit, OrdinaryDiffEq, Test
using SymbolicIndexingInterface

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs)

sys = structural_simplify(sys)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)

# ODEProblem.
oprob = ODEProblem(sys, u0, tspan, p, jac = true)

@test_throws Exception oprob[σ]
@test_throws Exception oprob[sys.σ]
@test_throws Exception oprob[:σ]
getσ1 = getp(sys, σ)
getσ2 = getp(sys, sys.σ)
getσ3 = getp(sys, :σ)
@test getσ1(oprob) == getσ2(oprob) == getσ3(oprob) == 28.0
getρ1 = getp(sys, ρ)
getρ2 = getp(sys, sys.ρ)
getρ3 = getp(sys, :ρ)
@test getρ1(oprob) == getρ2(oprob) == getρ3(oprob) == 10.0
getβ1 = getp(sys, β)
getβ2 = getp(sys, sys.β)
getβ3 = getp(sys, :β)
@test getβ1(oprob) == getβ2(oprob) == getβ3(oprob) == 8 / 3

@test oprob[x] == oprob[sys.x] == oprob[:x] == 1.0
@test oprob[y] == oprob[sys.y] == oprob[:y] == 0.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 0.0
@test oprob[[x, y]] == oprob[[sys.x, sys.y]] == oprob[[:x, :y]] == [1.0, 0.0]
@test oprob[(x, y)] == oprob[(sys.x, sys.y)] == oprob[(:x, :y)] == (1.0, 0.0)
@test oprob[solvedvariables] == oprob[variable_symbols(sys)]
@test oprob[allvariables] == oprob[all_variable_symbols(sys)]

setσ = setp(sys, σ)
setσ(oprob, 10.0)
@test getσ1(oprob) == getσ2(oprob) == getσ3(oprob) == 10.0
setρ = setp(sys, sys.ρ)
setρ(oprob, 20.0)
@test getρ1(oprob) == getρ2(oprob) == getρ3(oprob) == 20.0
setβ = setp(sys, :β)
setβ(oprob, 30.0)
@test getβ1(oprob) == getβ2(oprob) == getβ3(oprob) == 30.0

oprob[x] = 10.0
@test oprob[x] == oprob[sys.x] == oprob[:x] == 10.0
oprob[sys.y] = 10.0
@test oprob[sys.y] == oprob[sys.y] == oprob[:y] == 10.0
oprob[:z] = 1.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 1.0

# SDEProblem.
noiseeqs = [0.1 * x,
    0.1 * y,
    0.1 * z]
@named noise_sys = SDESystem(sys, noiseeqs)
sprob = SDEProblem(noise_sys, u0, (0.0, 100.0), p)
u0

@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == 28.0
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == 10.0
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == 8 / 3

@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 1.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 0.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 0.0

setσ(sprob, 10.0)
@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == 10.0
setρ(sprob, 20.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == 20.0
setp(noise_sys, noise_sys.ρ)(sprob, 25.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == 25.0
setβ(sprob, 30.0)
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == 30.0

sprob[x] = 10.0
@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 10.0
sprob[noise_sys.y] = 10.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 10.0
sprob[:z] = 1.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 1.0
