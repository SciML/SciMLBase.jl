using ModelingToolkit, OrdinaryDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@mtkbuild sys = ODESystem(eqs, t)

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
@test getσ1(oprob) == getσ2(oprob) == getσ3(oprob) == oprob.ps[σ] == oprob.ps[sys.σ] ==
      oprob.ps[:σ] == 28.0
getρ1 = getp(sys, ρ)
getρ2 = getp(sys, sys.ρ)
getρ3 = getp(sys, :ρ)
@test getρ1(oprob) == getρ2(oprob) == getρ3(oprob) == oprob.ps[ρ] == oprob.ps[sys.ρ] ==
      oprob.ps[:ρ] == 10.0
getβ1 = getp(sys, β)
getβ2 = getp(sys, sys.β)
getβ3 = getp(sys, :β)
@test getβ1(oprob) == getβ2(oprob) == getβ3(oprob) == oprob.ps[β] == oprob.ps[sys.β] ==
      oprob.ps[:β] == 8 / 3

@test oprob[x] == oprob[sys.x] == oprob[:x] == 1.0
@test oprob[y] == oprob[sys.y] == oprob[:y] == 0.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 0.0
@test oprob[solvedvariables] == oprob[variable_symbols(sys)]
@test oprob[allvariables] == oprob[all_variable_symbols(sys)]

setσ = setp(sys, σ)
setσ(oprob, 10.0)
@test getσ1(oprob) == getσ2(oprob) == getσ3(oprob) == oprob.ps[σ] == oprob.ps[sys.σ] ==
      oprob.ps[:σ] == 10.0
setρ = setp(sys, sys.ρ)
setρ(oprob, 20.0)
@test getρ1(oprob) == getρ2(oprob) == getρ3(oprob) == oprob.ps[ρ] == oprob.ps[sys.ρ] ==
      oprob.ps[:ρ] == 20.0
setβ = setp(sys, :β)
setβ(oprob, 30.0)
@test getβ1(oprob) == getβ2(oprob) == getβ3(oprob) == oprob.ps[β] == oprob.ps[sys.β] ==
      oprob.ps[:β] == 30.0

oprob.ps[σ] = 11.0
@test getσ1(oprob) == getσ2(oprob) == getσ3(oprob) == oprob.ps[σ] == oprob.ps[sys.σ] ==
      oprob.ps[:σ] == 11.0
oprob.ps[sys.ρ] = 21.0
@test getρ1(oprob) == getρ2(oprob) == getρ3(oprob) == oprob.ps[ρ] == oprob.ps[sys.ρ] ==
      oprob.ps[:ρ] == 21.0
oprob.ps[:β] = 31.0
@test getβ1(oprob) == getβ2(oprob) == getβ3(oprob) == oprob.ps[β] == oprob.ps[sys.β] ==
      oprob.ps[:β] == 31.0

oprob[x] = 10.0
@test oprob[x] == oprob[sys.x] == oprob[:x] == 10.0
oprob[sys.y] = 10.0
@test oprob[sys.y] == oprob[sys.y] == oprob[:y] == 10.0
oprob[:z] = 1.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 1.0

getx = getu(oprob, x)
gety = getu(oprob, :y)
get_arr = getu(oprob, [x, y])
get_tuple = getu(oprob, (y, z))
get_obs = getu(oprob, sys.x + sys.z + t + σ)
@test getx(oprob) == 10.0
@test gety(oprob) == 10.0
@test get_arr(oprob) == [10.0, 10.0]
@test get_tuple(oprob) == (10.0, 1.0)
@test get_obs(oprob) == 22.0

setx! = setu(oprob, x)
sety! = setu(oprob, :y)
set_arr! = setu(oprob, [x, y])
set_tuple! = setu(oprob, (y, z))

setx!(oprob, 11.0)
@test getx(oprob) == 11.0
sety!(oprob, 12.0)
@test gety(oprob) == 12.0
set_arr!(oprob, [11.0, 12.0])
@test get_arr(oprob) == [11.0, 12.0]
set_tuple!(oprob, [10.0, 10.0])
@test get_tuple(oprob) == (10.0, 10.0)

# SDEProblem.
noiseeqs = [0.1 * x,
    0.1 * y,
    0.1 * z]
@named noise_sys = SDESystem(sys, noiseeqs)
sprob = SDEProblem(noise_sys, u0, (0.0, 100.0), p)
u0

@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == sprob.ps[σ] == sprob.ps[sys.σ] ==
      sprob.ps[:σ] == 28.0
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 10.0
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == sprob.ps[β] == sprob.ps[sys.β] ==
      sprob.ps[:β] == 8 / 3

@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 1.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 0.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 0.0

setσ(sprob, 10.0)
@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == sprob.ps[σ] == sprob.ps[sys.σ] ==
      sprob.ps[:σ] == 10.0
setρ(sprob, 20.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 20.0
setp(noise_sys, noise_sys.ρ)(sprob, 25.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 25.0
setβ(sprob, 30.0)
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == sprob.ps[β] == sprob.ps[sys.β] ==
      sprob.ps[:β] == 30.0
sprob.ps[σ] = 11.0
@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == sprob.ps[σ] == sprob.ps[sys.σ] ==
      sprob.ps[:σ] == 11.0
sprob.ps[sys.ρ] = 21.0
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 21.0
sprob.ps[:β] = 31.0
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == sprob.ps[β] == sprob.ps[sys.β] ==
      sprob.ps[:β] == 31.0

sprob[x] = 10.0
@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 10.0
sprob[noise_sys.y] = 10.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 10.0
sprob[:z] = 1.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 1.0

getx = getu(sprob, x)
gety = getu(sprob, :y)
get_arr = getu(sprob, [x, y])
get_tuple = getu(sprob, (y, z))
get_obs = getu(sprob, sys.x + sys.z + t + σ)
@test getx(sprob) == 10.0
@test gety(sprob) == 10.0
@test get_arr(sprob) == [10.0, 10.0]
@test get_tuple(sprob) == (10.0, 1.0)
@test get_obs(sprob) == 22.0

setx! = setu(sprob, x)
sety! = setu(sprob, :y)
set_arr! = setu(sprob, [x, y])
set_tuple! = setu(sprob, (y, z))

setx!(sprob, 11.0)
@test getx(sprob) == 11.0
sety!(sprob, 12.0)
@test gety(sprob) == 12.0
set_arr!(sprob, [11.0, 12.0])
@test get_arr(sprob) == [11.0, 12.0]
set_tuple!(sprob, [10.0, 10.0])
@test get_tuple(sprob) == (10.0, 10.0)

using LinearAlgebra
sts = @variables x(t)[1:3]=[1, 2, 3.0] y(t)=1.0
ps = @parameters p[1:3] = [1, 2, 3]
D = Differential(t)
eqs = [collect(D.(x) .~ x)
       D(y) ~ norm(x) * y - x[1]]
@mtkbuild sys = ODESystem(eqs, t, [sts...;], [ps...;])
prob = ODEProblem(sys, [], (0, 1.0))
@test getp(sys, p)(prob) == prob.ps[p] == [1, 2, 3]
setp(sys, p)(prob, [4, 5, 6])
@test getp(sys, p)(prob) == prob.ps[p] == [4, 5, 6]
prob.ps[p] = [7, 8, 9]
@test getp(sys, p)(prob) == prob.ps[p] == [7, 8, 9]
