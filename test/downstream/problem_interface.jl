using ModelingToolkit, OrdinaryDiffEq, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
]

@mtkcompile sys = System(eqs, t)

u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]

tspan = (0.0, 100.0)

# ODEProblem.
oprob = ODEProblem(sys, [u0; p], tspan, jac = true)

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
    0.1 * z,
]
@named noise_sys = SDESystem(sys, noiseeqs)
noise_sys = complete(noise_sys)
sprob = SDEProblem(noise_sys, [u0; p], (0.0, 100.0))
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
sts = @variables x(t)[1:3] = [1, 2, 3.0] y(t) = 1.0
ps = @parameters p[1:3] = [1, 2, 3]
eqs = [
    collect(D.(x) .~ x)
    D(y) ~ norm(x) * y - x[1]
]
@mtkcompile sys = System(eqs, t, [sts...;], [ps...;])
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
    X2 ~ 2 * X,
]
@mtkcompile osys = System(eqs, t)

u0 = [X => 0.1]
tspan = (0.0, 10.0)
ps = [p => 1.0, d => 0.2]
oprob = ODEProblem(osys, [u0; ps], tspan)
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
@test state_values(remake(eprob; u0 = [:X => 0.2])) == [0.2]
@test state_values(remake(eprob; u0 = [osys.X => 0.3])) == [0.3]

@test remake(eprob; p = [d => 0.4]).ps[d] == 0.4
@test remake(eprob; p = [:d => 0.5]).ps[d] == 0.5
@test remake(eprob; p = [osys.d => 0.6]).ps[d] == 0.6

# SteadyStateProblem Indexing
# Issue#660
@parameters p d
@variables X(t) X2(t)
eqs = [
    D(X) ~ p - d * X,
    X2 ~ 2 * X,
]
@mtkbuild osys = System(eqs, t)

u0 = [X => 0.1]
ps = [p => 1.0, d => 0.2]
prob = SteadyStateProblem(osys, [u0; ps])

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
    @variables u[1:8] = zeros(8) [irreducible = true]
    u2 = collect(u)
    @parameters p = 1.0
    eqs = Any[0 for _ in 1:8]
    fullf!(eqs, u, [p])
    @mtkcompile model = System(0 .~ eqs, [u...], [p])

    prob = NonlinearProblem(model, [])
    sccprob = SCCNonlinearProblem(model, [])
    @test sccprob isa SCCNonlinearProblem
end

@testset "LinearProblem" begin
    # TODO update when MTK codegen exists
    sys = SymbolCache([:x, :y, :z], [:p, :q, :r])
    update_A! = function (A, p)
        A[1, 1] = p[1]
        A[2, 2] = p[2]
        A[3, 3] = p[3]
    end
    update_b! = function (b, p)
        b[1] = p[3]
        b[2] = -8p[2] - p[1]
    end
    f = SciMLBase.SymbolicLinearInterface(update_A!, update_b!, sys, nothing, nothing)
    A = Float64[1 1 1; 6 -4 5; 5 2 2]
    b = Float64[2, 31, 13]
    p = Float64[1, -4, 2]
    u0 = Float64[1, 2, 3]
    prob = LinearProblem(A, b, p; u0, f)
    @test prob[:x] ≈ 1.0
    @test prob[:y] ≈ 2.0
    @test prob[:z] ≈ 3.0
    @test prob.ps[:p] ≈ 1.0
    @test prob.ps[:q] ≈ -4.0
    @test prob.ps[:r] ≈ 2.0
    prob.ps[:p] = 2.0
    @test prob.ps[:p] ≈ 2.0
    @test prob.A[1, 1] ≈ 2.0
    @test prob.b[2] ≈ 30.0

    prob2 = remake(prob; u0 = 2u0)
    @test prob2.u0 ≈ 2u0
    prob2 = remake(prob; p = 2p)
    @test prob2.p ≈ 2p
    prob2 = remake(prob; u0 = [:x => 3.0], p = [:q => 1.5])
    @test prob2.u0[1] ≈ 3.0
    @test prob2.p[2] ≈ 1.5

    # no u0
    prob = LinearProblem(A, b, p; f)
    prob2 = remake(prob; p = 2p)
    @test prob2.p ≈ 2p
    prob2 = remake(prob; p = [:q => 1.5])
    @test prob2.p[2] ≈ 1.5
end
