EnumX.@enumx(ReturnCode, Default, Success, Terminated, DtNaN, MaxIters, DtLessThanMin,
             Unstable,
             InitialFailure, ConvergenceFailure, Failure, ExactSolutionLeft,
             ExactSolutionRight, FloatingPointLimit)

Base.convert(::Type{Symbol}, retcode::ReturnCode.T) = Symbol(retcode)

Base.:(==)(retcode::ReturnCode.T, s::Symbol) = Symbol(retcode) == s
Base.:(!=)(retcode::ReturnCode.T, s::Symbol) = Symbol(retcode) != s

const symtrue = Symbol("true")
const symfalse = Symbol("false")

function Base.convert(::Type{ReturnCode.T}, retcode::Symbol)
    if retcode == :Default || retcode == :DEFAULT
        ReturnCode.Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT ||
           retcode == :FLOATING_POINT_LIMIT || retcode == symtrue
        ReturnCode.Success
    elseif retcode == :Terminated
        ReturnCode.Terminated
    elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED
        ReturnCode.MaxIters
    elseif retcode == :DtLessThanMin
        ReturnCode.DtLessThanMin
    elseif retcode == :Unstable
        ReturnCode.Unstable
    elseif retcode == :InitialFailure
        ReturnCode.InitialFailure
    elseif retcode == :ConvergenceFailure || retcode == :ITERATION_LIMIT
        ReturnCode.ConvergenceFailure
    elseif retcode == :Failure || retcode == symfalse
        ReturnCode.Failure
    else
        ReturnCode.Failure
    end
end

function Base.convert(::Type{ReturnCode.T}, bool::Bool)
    bool ? ReturnCode.Success : ReturnCode.Failure
end

function successful_retcode(retcode::ReturnCode.T)
    retcode == :Success || retcode == :Terminated || retcode == :ExactSolutionLeft ||
        retcode == :ExactSolutionRight || retcode == :FloatingPointLimit
end
