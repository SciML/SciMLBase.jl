using ModelingToolkit, OrdinaryDiffEq, Test

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

@test oprob[σ] == oprob[sys.σ] ==oprob[:σ] == 28.0
@test oprob[ρ] == oprob[sys.ρ] ==oprob[:ρ] == 10.0
@test oprob[β] == oprob[sys.β] ==oprob[:β] == 8 / 3

@test oprob[x] == oprob[sys.x] == oprob[:x] == 1.0
@test oprob[y] == oprob[sys.y] == oprob[:y] == 0.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 0.0

oprob[σ] = 10.0
@test oprob[σ] == oprob[sys.σ] ==oprob[:σ] == 10.0
oprob[sys.ρ] = 20.0
@test oprob[ρ] == oprob[sys.ρ] ==oprob[:ρ] == 20.0
oprob[σ] = 30.0
@test oprob[σ] == oprob[sys.σ] ==oprob[:σ] == 30.0

oprob[x] = 10.0
@test oprob[x] == oprob[sys.x] == oprob[:x] == 10.0
oprob[sys.y] = 10.0
@test oprob[sys.y] == oprob[sys.y] == oprob[:y] == 10.0
oprob[:z] = 1.0
@test oprob[z] == oprob[sys.z] == oprob[:z] == 1.0

# SDEProblem.
noiseeqs = [0.1*x,
            0.1*y,
            0.1*z]
@named noise_sys = SDESystem(sys, noiseeqs)
sprob = SDEProblem(noise_sys, u0, (0.0, 100.0), p)
u0

@test sprob[σ] == sprob[noise_sys.σ] ==sprob[:σ] == 28.0
@test sprob[ρ] == sprob[noise_sys.ρ] ==sprob[:ρ] == 10.0
@test sprob[β] == sprob[noise_sys.β] ==sprob[:β] == 8 / 3

@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 1.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 0.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 0.0

sprob[σ] = 10.0
@test sprob[σ] == sprob[noise_sys.σ] ==sprob[:σ] == 10.0
sprob[noise_sys.ρ] = 20.0
@test sprob[ρ] == sprob[noise_sys.ρ] ==sprob[:ρ] == 20.0
sprob[σ] = 30.0
@test sprob[σ] == sprob[noise_sys.σ] ==sprob[:σ] == 30.0

sprob[x] = 10.0
@test sprob[x] == sprob[noise_sys.x] == sprob[:x] == 10.0
sprob[noise_sys.y] = 10.0
@test sprob[y] == sprob[noise_sys.y] == sprob[:y] == 10.0
sprob[:z] = 1.0
@test sprob[z] == sprob[noise_sys.z] == sprob[:z] == 1.0