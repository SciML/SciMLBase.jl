function getsyms(sol::AbstractSciMLSolution)
    syms = variable_symbols(sol)
    if isempty(syms)
        syms = keys(sol.u[1])
    end
    return syms
end

function getsyms(prob::AbstractSciMLProblem)
    return variable_symbols(prob.f)
end

function getsyms(sol::AbstractOptimizationSolution)
    syms = variable_symbols(sol)
    if isempty(syms)
        syms = keys(sol.u[1])
    end
    return syms
end

"""
    getindepsym(prob_or_sol_or_integrator)

Return the primary symbolic independent variable for a problem, solution, or
integrator.

For problems, this queries
`SymbolicIndexingInterface.independent_variable_symbols` on the problem's
function and returns the first symbol when one is available. For solutions and
integrators, the query delegates to the underlying problem. If no symbolic
independent variable is defined, the result is `nothing`.

Use [`getindepsym_defaultt`](@ref) when caller code needs a plotting or display
fallback for time-dependent solutions that do not carry symbolic metadata.
"""
function getindepsym(prob::AbstractSciMLProblem)
    syms = independent_variable_symbols(prob.f)
    if isempty(syms)
        return nothing
    else
        return syms[1]
    end
end

getindepsym(sol::AbstractSciMLSolution) = getindepsym(sol.prob)

function getparamsyms(prob::AbstractSciMLProblem)
    psyms = parameter_symbols(prob.f)
    if isempty(psyms)
        return nothing
    end
    return psyms
end

getparamsyms(sol) = getparamsyms(sol.prob)

function getparamsyms(sol::AbstractOptimizationSolution)
    psyms = parameter_symbols(sol)
    if isempty(psyms)
        return nothing
    end
    return psyms
end

"""
    getindepsym_defaultt(sol)

Return the primary symbolic independent variable, falling back to `:t`.

This is a display and plotting helper for code paths that historically assumed a
time variable is always present. It returns `getindepsym(sol)` when symbolic
metadata defines an independent variable and `:t` otherwise. Use
[`getindepsym`](@ref) when `nothing` is the correct representation for "no
symbolic independent variable".
"""
function getindepsym_defaultt(sol)
    return something(getindepsym(sol), :t)
end

function getobserved(prob::AbstractSciMLProblem)
    if has_observed(prob.f)
        return prob.f.observed
    else
        return DEFAULT_OBSERVED
    end
end

getobserved(sol) = getobserved(sol.prob)

function getobserved(sol::AbstractOptimizationSolution)
    if has_observed(sol)
        return get_observed(sol)
    else
        return DEFAULT_OBSERVED
    end
end
