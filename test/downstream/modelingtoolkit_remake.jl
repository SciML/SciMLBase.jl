using ModelingToolkit, SymbolicIndexingInterface
using JumpProcesses
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)
u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
oprob = ODEProblem(sys, u0, tspan, p, jac = true)

@inferred typeof(oprob) remake(oprob; u0 = [x => 2.0], p = [σ => 29.0])
oprob2 = remake(
    oprob;
    u0 = [x => 2.0, sys.y => 1.2, :z => 1.0],
    p = [σ => 29.0, sys.ρ => 11.0, :β => 3.0]
)
@test oprob2.u0 isa Vector{<:Number}
@test oprob2.p isa ModelingToolkit.MTKParameters
@test oprob2[x] == oprob2[sys.x] == oprob2[:x] == 2.0
@test oprob2[y] == oprob2[sys.y] == oprob2[:y] == 1.2
@test oprob2[z] == oprob2[sys.z] == oprob2[:z] == 1.0
@test getp(sys, σ)(oprob2) == 29.0
@test getp(sys, sys.ρ)(oprob2) == 11.0
@test getp(sys, :β)(oprob2) == 3.0

oprob3 = remake(oprob; u0 = [x => 3.0], p = [σ => 30.0]) # partial update
@test oprob3[x] == 3.0
@test getp(sys, σ)(oprob3) == 30.0

# SDEProblem.
noiseeqs = [0.1 * x,
    0.1 * y,
    0.1 * z]
@named noise_sys = SDESystem(sys, noiseeqs)
noise_sys = complete(noise_sys)
sprob = SDEProblem(noise_sys, u0, (0.0, 100.0), p)

@inferred typeof(sprob) remake(sprob; u0 = [x => 2.0], p = [σ => 29.0])
sprob2 = remake(
    sprob;
    u0 = [x => 2.0, sys.y => 1.2, :z => 1.0],
    p = [σ => 29.0, sys.ρ => 11.0, :β => 3.0]
)
@test sprob2.u0 isa Vector{<:Number}
@test sprob2.p isa ModelingToolkit.MTKParameters
@test sprob2[x] == sprob2[sys.x] == sprob2[:x] == 2.0
@test sprob2[y] == sprob2[sys.y] == sprob2[:y] == 1.2
@test sprob2[z] == sprob2[sys.z] == sprob2[:z] == 1.0
@test getp(sys, σ)(sprob2) == 29.0
@test getp(sys, sys.ρ)(sprob2) == 11.0
@test getp(sys, :β)(sprob2) == 3.0

sprob3 = remake(sprob; u0 = [x => 3.0], p = [σ => 30.0]) # partial update
@test sprob3[x] == 3.0
@test getp(sys, σ)(sprob3) == 30.0

# DiscreteProblem
# @named de = DiscreteSystem(
#     [D(x) ~ σ*(y-x),
#     D(y) ~ x*(ρ-z)-y,
#     D(z) ~ x*y - β*z],
#     t,
#     [x, y, z],
#     [σ, ρ, β],
# )
# dprob = DiscreteProblem(de, u0, tspan, p)

# @inferred typeof(dprob) remake(dprob; u0 = [x => 2.0], p = [σ => 29.0])
# dprob2 = remake(
#     dprob;
#     u0 = [x => 2.0, sys.y => 1.2, :z => 1.0],
#     p = [σ => 29.0, sys.ρ => 11.0, :β => 3.0]
# )
# @test dprob2.u0 isa Vector{<:Number}
# @test dprob2.p isa ModelingToolkit.MTKParameters
# @test dprob2[x] == dprob2[sys.x] == dprob2[:x] == 2.0
# @test dprob2[y] == dprob2[sys.y] == dprob2[:y] == 1.2
# @test dprob2[z] == dprob2[sys.z] == dprob2[:z] == 1.0
# @test getp(de, σ)(dprob2) == 29.0
# @test getp(de, sys.ρ)(dprob2) == 11.0
# @test getp(de, :β)(dprob2) == 3.0

# dprob3 = remake(dprob; u0 = [x => 3.0], p = [σ => 30.0]) # partial update
# @test dprob3[x] == 3.0
# @test getp(de, σ)(dprob3) == 30.0

# NonlinearProblem
@named ns = NonlinearSystem(
    [0 ~ σ*(y-x),
    0 ~ x*(ρ-z)-y,
    0 ~ x*y - β*z],
    [x,y,z],
    [σ,ρ,β]
)
ns = complete(ns)
nlprob = NonlinearProblem(ns, u0, p)

@inferred typeof(nlprob) remake(nlprob; u0 = [x => 2.0], p = [σ => 29.0])
nlprob2 = remake(
    nlprob;
    u0 = [x => 2.0, sys.y => 1.2, :z => 1.0],
    p = [σ => 29.0, sys.ρ => 11.0, :β => 3.0]
)
@test nlprob2.u0 isa Vector{<:Number}
@test nlprob2.p isa ModelingToolkit.MTKParameters
@test nlprob2[x] == nlprob2[sys.x] == nlprob2[:x] == 2.0
@test nlprob2[y] == nlprob2[sys.y] == nlprob2[:y] == 1.2
@test nlprob2[z] == nlprob2[sys.z] == nlprob2[:z] == 1.0
@test getp(ns, σ)(nlprob2) == 29.0
@test getp(ns, sys.ρ)(nlprob2) == 11.0
@test getp(ns, :β)(nlprob2) == 3.0

nlprob3 = remake(nlprob; u0 = [x => 3.0], p = [σ => 30.0]) # partial update
@test nlprob3[x] == 3.0
@test getp(ns, σ)(nlprob3) == 30.0

@parameters β γ
@variables S(t) I(t) R(t)
rate₁   = β*S*I
affect₁ = [S ~ S - 1, I ~ I + 1]
rate₂   = γ*I
affect₂ = [I ~ I - 1, R ~ R + 1]
j₁      = ConstantRateJump(rate₁,affect₁)
j₂      = ConstantRateJump(rate₂,affect₂)
j₃      = MassActionJump(2*β+γ, [R => 1], [S => 1, R => -1])
@named js      = JumpSystem([j₁,j₂,j₃], t, [S,I,R], [β,γ])
js = complete(js)
u₀map = [S => 999, I => 1, R => 0.0]
parammap = [β => 0.1 / 1000, γ => 0.01]
tspan = (0.0, 250.0)
jump_dprob = DiscreteProblem(js, u₀map, tspan, parammap)
jprob = JumpProblem(js, jump_dprob, Direct())

@inferred typeof(jprob) remake(jprob; u0 = [S => 900], p = [β => 0.2e-3])
jprob2 = remake(
    jprob;
    u0 = [S => 900, js.I => 2, :R => 0.1],
    p = [β => 0.2 / 1000, js.γ => 11.0]
)
@test jprob2.prob.u0 isa Vector{<:Number}
@test jprob2.prob.p isa ModelingToolkit.MTKParameters
@test jprob2[S] == jprob2[js.S] == jprob2[:S] == 900.0
@test jprob2[I] == jprob2[js.I] == jprob2[:I] == 2.0
@test jprob2[R] == jprob2[js.R] == jprob2[:R] == 0.1
@test getp(js, β)(jprob2) == 0.2 / 1000
@test getp(js, js.γ)(jprob2) == 11.0

jprob3 = remake(jprob; u0 = [S => 901], p = [:β => 0.3 / 1000]) # partial update
@test jprob3[S] == 901
@test getp(js, β)(jprob3) == 0.3 / 1000
