# Skip the DiffEqBase handling
function solve(prob::OptimizationProblem, opt, args...;kwargs...)
    __solve(prob, opt, args...; kwargs...)
end
