"""
    is_diagonal_noise(prob::AbstractSciMLProblem)
"""
is_diagonal_noise(prob::AbstractSciMLProblem) = false
function is_diagonal_noise(prob::AbstractRODEProblem{
        uType,
        tType,
        iip,
        Nothing
}) where {
        uType,
        tType,
        iip
}
    true
end
function is_diagonal_noise(prob::AbstractSDDEProblem{
        uType,
        tType,
        lType,
        iip,
        Nothing
}) where {
        uType,
        tType,
        lType,
        iip
}
    true
end

"""
    isinplace(prob::AbstractSciMLProblem)

Determine whether the function of the given problem operates in place or not.
"""
function isinplace(prob::AbstractSciMLProblem) end
isinplace(prob::AbstractLinearProblem{bType, iip}) where {bType, iip} = iip
isinplace(prob::AbstractNonlinearProblem{uType, iip}) where {uType, iip} = iip
isinplace(prob::AbstractIntegralProblem{iip}) where {iip} = iip
isinplace(prob::AbstractODEProblem{uType, tType, iip}) where {uType, tType, iip} = iip
function isinplace(prob::AbstractRODEProblem{
        uType,
        tType,
        iip,
        ND
}) where {uType, tType,
        iip, ND}
    iip
end
function isinplace(prob::AbstractDDEProblem{
        uType,
        tType,
        lType,
        iip
}) where {uType, tType,
        lType, iip}
    iip
end
function isinplace(prob::AbstractDAEProblem{
        uType,
        duType,
        tType,
        iip
}) where {uType,
        duType,
        tType, iip}
    iip
end
isinplace(prob::AbstractNoiseProblem) = isinplace(prob.noise)
isinplace(::SplitFunction{iip}) where {iip} = iip
function isinplace(prob::AbstractSDDEProblem{
        uType,
        tType,
        lType,
        iip,
        ND
}) where {uType,
        tType,
        lType,
        iip, ND}
    iip
end
