using OrdinaryDiffEq, Test

# https://github.com/SciML/OrdinaryDiffEq.jl/issues/2719

# set up functions
function f1!(du, u , p, t)
    du .= -u.^2
    return nothing
end

function f2!(du, u , p, t)
    du .= 2u
    return nothing
end

function f!(du, u, p, t)
    du .= -u.^2 .+ 2u
    return nothing
end

#create problems
u0 = ones(2)
tspan = (0.0, 1.0)
prob = ODEProblem(f!, u0, tspan)
f_split! = SplitFunction(f1!, f2!)
prob_split = SplitODEProblem(f_split!, u0, tspan)

#solve
sol = solve(prob, Rodas5P())
sol_split = solve(prob_split, Rodas5P())

#tests
@test sol_split == sol