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

getx = getsym(oprob, x)
gety = getsym(oprob, :y)
get_arr = getsym(oprob, [x, y])
get_tuple = getsym(oprob, (y, z))
get_obs = getsym(oprob, sys.x + sys.z + t + σ)
@test getx(oprob) == 10.0
@test gety(oprob) == 10.0
@test get_arr(oprob) == [10.0, 10.0]
@test get_tuple(oprob) == (10.0, 1.0)
@test get_obs(oprob) == 22.0

setx! = setsym(oprob, x)
sety! = setsym(oprob, :y)
set_arr! = setsym(oprob, [x, y])
set_tuple! = setsym(oprob, (y, z))

setx!(oprob, 11.0)
@test getx(oprob) == 11.0
sety!(oprob, 12.0)
@test gety(oprob) == 12.0
set_arr!(oprob, [11.0, 12.0])
@test get_arr(oprob) == [11.0, 12.0]
set_tuple!(oprob, [10.0, 10.0])
@test get_tuple(oprob) == (10.0, 10.0)

# SDEProblem.
noiseeqs = [
    0.1 * D(x),
    0.1 * x,
    0.1 * y,
    0.1 * z]
@named noise_sys = SDESystem(sys, noiseeqs)
noise_sys = complete(noise_sys)
sprob = SDEProblem(noise_sys, u0, (0.0, 100.0), p)
u0

getσ1 = getp(noise_sys, σ)
getσ2 = getp(noise_sys, sys.σ)
getσ3 = getp(noise_sys, :σ)
@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == sprob.ps[σ] == sprob.ps[sys.σ] ==
      sprob.ps[:σ] == 28.0
getρ1 = getp(noise_sys, ρ)
getρ2 = getp(noise_sys, sys.ρ)
getρ3 = getp(noise_sys, :ρ)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 10.0
getβ1 = getp(noise_sys, β)
getβ2 = getp(noise_sys, sys.β)
getβ3 = getp(noise_sys, :β)
@test getβ1(sprob) == getβ2(sprob) == getβ3(sprob) == sprob.ps[β] == sprob.ps[sys.β] ==
      sprob.ps[:β] == 8 / 3

@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 1.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 0.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 0.0

setσ = setp(noise_sys, σ)
setσ(sprob, 10.0)
@test getσ1(sprob) == getσ2(sprob) == getσ3(sprob) == sprob.ps[σ] == sprob.ps[sys.σ] ==
      sprob.ps[:σ] == 10.0
setρ = setp(noise_sys, sys.ρ)
setρ(sprob, 20.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 20.0
setp(noise_sys, noise_sys.ρ)(sprob, 25.0)
@test getρ1(sprob) == getρ2(sprob) == getρ3(sprob) == sprob.ps[ρ] == sprob.ps[sys.ρ] ==
      sprob.ps[:ρ] == 25.0
setβ = setp(noise_sys, :β)
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

getx = getsym(sprob, x)
gety = getsym(sprob, :y)
get_arr = getsym(sprob, [x, y])
get_tuple = getsym(sprob, (y, z))
get_obs = getsym(sprob, sys.x + sys.z + t + σ)
@test getx(sprob) == 10.0
@test gety(sprob) == 10.0
@test get_arr(sprob) == [10.0, 10.0]
@test get_tuple(sprob) == (10.0, 1.0)
@test get_obs(sprob) == 22.0

setx! = setsym(sprob, x)
sety! = setsym(sprob, :y)
set_arr! = setsym(sprob, [x, y])
set_tuple! = setsym(sprob, (y, z))

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
eqs = [collect(D.(x) .~ x)
       D(y) ~ norm(x) * y - x[1]]
@mtkbuild sys = ODESystem(eqs, t, [sts...;], [ps...;])
prob = ODEProblem(sys, [], (0, 1.0))
@test getp(sys, p)(prob) == prob.ps[p] == [1, 2, 3]
setp(sys, p)(prob, [4, 5, 6])
@test getp(sys, p)(prob) == prob.ps[p] == [4, 5, 6]
prob.ps[p] = [7, 8, 9]
@test getp(sys, p)(prob) == prob.ps[p] == [7, 8, 9]

# EnsembleProblem indexing
# Issue#661
@parameters p d
@variables X(t) X2(t)
eqs = [
    D(X) ~ p - d * X,
    X2 ~ 2 * X
]
@mtkbuild osys = ODESystem(eqs, t)

u0 = [X => 0.1]
tspan = (0.0, 10.0)
ps = [p => 1.0, d => 0.2]
oprob = ODEProblem(osys, u0, tspan, ps)
eprob = EnsembleProblem(oprob)

@test eprob[X] == 0.1
@test eprob[:X] == 0.1
@test eprob[osys.X] == 0.1
@test eprob.ps[p] == 1.0
@test eprob.ps[:p] == 1.0
@test eprob.ps[osys.p] == 1.0
@test getsym(eprob, X)(eprob) == 0.1
@test getsym(eprob, :X)(eprob) == 0.1
@test getsym(eprob, osys.X)(eprob) == 0.1
@test getp(eprob, p)(eprob) == 1.0
@test getp(eprob, :p)(eprob) == 1.0
@test getp(eprob, osys.p)(eprob) == 1.0

@test_nowarn eprob[X] = 0.0
@test eprob[X] == 0.0
@test_nowarn eprob[:X] = 0.1
@test eprob[:X] == 0.1
@test_nowarn eprob[osys.X] = 0.0
@test eprob[osys.X] == 0.0
@test_nowarn eprob.ps[p] = 0.0
@test eprob.ps[p] == 0.0
@test_nowarn eprob.ps[:p] = 0.1
@test eprob.ps[:p] == 0.1
@test_nowarn eprob.ps[osys.p] = 0.0
@test eprob.ps[osys.p] == 0.0
@test_nowarn setsym(eprob, X)(eprob, 0.1)
@test eprob[X] == 0.1
@test_nowarn setsym(eprob, :X)(eprob, 0.0)
@test eprob[:X] == 0.0
@test_nowarn setsym(eprob, osys.X)(eprob, 0.1)
@test eprob[osys.X] == 0.1
@test_nowarn setp(eprob, p)(eprob, 0.1)
@test eprob.ps[p] == 0.1
@test_nowarn setp(eprob, :p)(eprob, 0.0)
@test eprob.ps[:p] == 0.0
@test_nowarn setp(eprob, osys.p)(eprob, 0.1)
@test eprob.ps[osys.p] == 0.1

@test state_values(remake(eprob; u0 = [X => 0.1])) == [0.1]
@test_broken state_values(remake(eprob; u0 = [:X => 0.2])) == [0.2]
@test state_values(remake(eprob; u0 = [osys.X => 0.3])) == [0.3]

@test_broken remake(eprob; p = [d => 0.4]).ps[d] == 0.4
@test_broken remake(eprob; p = [:d => 0.5]).ps[d] == 0.5
@test_broken remake(eprob; p = [osys.d => 0.6]).ps[d] == 0.6

# SteadyStateProblem Indexing
# Issue#660
@parameters p d
@variables X(t) X2(t)
eqs = [
    D(X) ~ p - d * X,
    X2 ~ 2 * X
]
@mtkbuild osys = ODESystem(eqs, t)

u0 = [X => 0.1]
ps = [p => 1.0, d => 0.2]
prob = SteadyStateProblem(osys, u0, ps)

@test prob[X] == prob[osys.X] == prob[:X] == 0.1
@test prob[X2] == prob[osys.X2] == prob[:X2] == 0.2
@test prob[[X, X2]] == prob[[osys.X, osys.X2]] == prob[[:X, :X2]] == [0.1, 0.2]
@test getsym(prob, X)(prob) == getsym(prob, osys.X)(prob) == getsym(prob, :X)(prob) == 0.1
@test getsym(prob, X2)(prob) == getsym(prob, osys.X2)(prob) == getsym(prob, :X2)(prob) ==
      0.2
@test getsym(prob, [X, X2])(prob) == getsym(prob, [osys.X, osys.X2])(prob) ==
      getsym(prob, [:X, :X2])(prob) == [0.1, 0.2]
@test getsym(prob, (X, X2))(prob) == getsym(prob, (osys.X, osys.X2))(prob) ==
      getsym(prob, (:X, :X2))(prob) == (0.1, 0.2)

@testset "SCCNonlinearProblem" begin
    # TODO: Rewrite this example when the MTK codegen is merged

    function fullf!(du, u, p)
        du[1] = cos(u[2]) - u[1]
        du[2] = sin(u[1] + u[2]) + u[2]
        du[3] = 2u[4] + u[3] + p[1]
        du[4] = u[5]^2 + u[4]
        du[5] = u[3]^2 + u[5]
        du[6] = u[1] + u[2] + u[3] + u[4] + u[5] + 2.0u[6] + 2.5u[7] + 1.5u[8]
        du[7] = u[1] + u[2] + u[3] + 2.0u[4] + u[5] + 4.0u[6] - 1.5u[7] + 1.5u[8]
        du[8] = u[1] + 2.0u[2] + 3.0u[3] + 5.0u[4] + 6.0u[5] + u[6] - u[7] - u[8]
    end
    @variables u[1:8]=zeros(8) [irreducible = true]
    u2 = collect(u)
    @parameters p = 1.0
    eqs = Any[0 for _ in 1:8]
    fullf!(eqs, u, [p])
    @named model = NonlinearSystem(0 .~ eqs, [u...], [p])
    model = complete(model; split = false)

    cache = zeros(4)
    cache[1] = 1.0

    function f1!(du, u, p)
        du[1] = cos(u[2]) - u[1]
        du[2] = sin(u[1] + u[2]) + u[2]
    end
    explicitfun1(cache, sols) = nothing

    f1!(eqs, u2[1:2], [p])
    @named subsys1 = NonlinearSystem(0 .~ eqs[1:2], [u2[1:2]...], [p])
    subsys1 = complete(subsys1; split = false)
    prob1 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1!; sys = subsys1),
        zeros(2), copy(cache))

    function f2!(du, u, p)
        du[1] = 2u[2] + u[1] + p[1]
        du[2] = u[3]^2 + u[2]
        du[3] = u[1]^2 + u[3]
    end
    explicitfun2(cache, sols) = nothing

    f2!(eqs, u2[3:5], [p])
    @named subsys2 = NonlinearSystem(0 .~ eqs[1:3], [u2[3:5]...], [p])
    subsys2 = complete(subsys2; split = false)
    prob2 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f2!; sys = subsys2),
        zeros(3), copy(cache))

    function f3!(du, u, p)
        du[1] = p[2] + 2.0u[1] + 2.5u[2] + 1.5u[3]
        du[2] = p[3] + 4.0u[1] - 1.5u[2] + 1.5u[3]
        du[3] = p[4] + +u[1] - u[2] - u[3]
    end
    function explicitfun3(cache, sols)
        cache[2] = sols[1][1] + sols[1][2] + sols[2][1] + sols[2][2] + sols[2][3]
        cache[3] = sols[1][1] + sols[1][2] + sols[2][1] + 2.0sols[2][2] + sols[2][3]
        cache[4] = sols[1][1] + 2.0sols[1][2] + 3.0sols[2][1] + 5.0sols[2][2] +
                   6.0sols[2][3]
    end

    @parameters tmpvar[1:3]
    f3!(eqs, u2[6:8], [p, tmpvar...])
    @named subsys3 = NonlinearSystem(0 .~ eqs[1:3], [u2[6:8]...], [p, tmpvar...])
    subsys3 = complete(subsys3; split = false)
    prob3 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f3!; sys = subsys3),
        zeros(3), copy(cache))

    prob = NonlinearProblem(model, [])
    sccprob = SciMLBase.SCCNonlinearProblem([prob1, prob2, prob3],
        SciMLBase.Void{Any}.([explicitfun1, explicitfun2, explicitfun3]),
        copy(cache); sys = model)

    for sym in [u, u..., u[2] + u[3], p * u[1] + u[2]]
        @test prob[sym] ≈ sccprob[sym]
    end

    for sym in [p, 2p + 1]
        @test prob.ps[sym] ≈ sccprob.ps[sym]
    end

    for (i, sym) in enumerate([u[1], u[3], u[6]])
        sccprob[sym] = 0.5i
        @test sccprob[sym] ≈ 0.5i
        @test sccprob.probs[i].u0[1] ≈ 0.5i
    end
    sccprob.ps[p] = 2.5
    @test sccprob.ps[p] ≈ 2.5
    @test sccprob.p[1] ≈ 2.5
    for scc in sccprob.probs
        @test parameter_values(scc)[1] ≈ 2.5
    end
end
