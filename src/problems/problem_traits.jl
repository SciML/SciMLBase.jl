"""
    is_diagonal_noise(prob::AbstractSciMLProblem)

Return whether a stochastic or random problem should be treated as diagonal
noise.

For SDE-like problems, diagonal noise means the noise-rate callback supplies one
independent noise channel per state component rather than a full noise-rate
matrix or operator. SciML problem constructors encode this through their
noise-rate prototype type parameter: `noise_rate_prototype === nothing` (or the
corresponding random-prototype metadata for RODE/SDDE problems) is interpreted
as diagonal by the common interface. Non-stochastic problem types return
`false`.

Solver packages use this trait when choosing storage layouts, validating noise
dimensions, and selecting algorithms that require diagonal or non-diagonal
noise. Concrete problem subtypes should overload this trait if diagonal-noise
status is not represented by the standard prototype type parameter.
"""
is_diagonal_noise(prob::AbstractSciMLProblem) = false
function is_diagonal_noise(
        prob::AbstractRODEProblem{
            uType,
            tType,
            iip,
            Nothing,
        }
    ) where {
        uType,
        tType,
        iip,
    }
    return true
end
function is_diagonal_noise(
        prob::AbstractSDDEProblem{
            uType,
            tType,
            lType,
            iip,
            Nothing,
        }
    ) where {
        uType,
        tType,
        lType,
        iip,
    }
    return true
end

"""
    isinplace(prob::AbstractSciMLProblem)

Return the in-place convention encoded by a SciML problem.

`true` means the primary model callback mutates its first argument, such as
`f(du, u, p, t)` for an ODE or `f(resid, u, p)` for a nonlinear problem.
`false` means the callback returns its computed value, such as `f(u, p, t)`.
Concrete problem types store this choice as a type parameter so solvers can
dispatch without re-inspecting user methods.

Constructors infer this value from raw callables with [`isinplace`](@ref),
but users and problem builders can specify the type parameter explicitly for
type stability. Remake, problem conversion, display, initialization, and solver
setup code should query this trait instead of re-detecting callback arity.

Subtypes of [`AbstractSciMLProblem`](@ref) should implement this by returning
their stored in-place type parameter. Wrapper problems should forward to the
wrapped problem when the wrapper does not define its own model convention.
"""
function isinplace(prob::AbstractSciMLProblem) end
isinplace(prob::AbstractLinearProblem{bType, iip}) where {bType, iip} = iip
isinplace(prob::AbstractNonlinearProblem{uType, iip}) where {uType, iip} = iip
isinplace(prob::AbstractIntegralProblem{iip}) where {iip} = iip
isinplace(prob::AbstractODEProblem{uType, tType, iip}) where {uType, tType, iip} = iip
function isinplace(
        prob::AbstractRODEProblem{
            uType,
            tType,
            iip,
            ND,
        }
    ) where {
        uType, tType,
        iip, ND,
    }
    return iip
end
function isinplace(
        prob::AbstractDDEProblem{
            uType,
            tType,
            lType,
            iip,
        }
    ) where {
        uType, tType,
        lType, iip,
    }
    return iip
end
function isinplace(
        prob::AbstractDAEProblem{
            uType,
            duType,
            tType,
            iip,
        }
    ) where {
        uType,
        duType,
        tType, iip,
    }
    return iip
end
isinplace(prob::AbstractNoiseProblem) = isinplace(prob.noise)
isinplace(::SplitFunction{iip}) where {iip} = iip
function isinplace(
        prob::AbstractSDDEProblem{
            uType,
            tType,
            lType,
            iip,
            ND,
        }
    ) where {
        uType,
        tType,
        lType,
        iip, ND,
    }
    return iip
end
