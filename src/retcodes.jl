EnumX.@enumx(ReturnCode,Default,Success,Terminated,MaxIters,DtLessThanMin,Unstable,
      InitialFailure,ConvergenceFailure,Failure)

function Base.Symbol(retcode::ReturnCode.T)
    if retcode == Default
        :Default
    elseif retcode == Success
        :Success
    elseif retcode == Terminated
        :Terminated
    elseif retcode == MaxIters
        :MaxIters
    elseif retcode == DtLessThanMin
        :DtLessThanMin
    elseif retcode == Unstable
        :Unstable
    elseif retcode == InitialFailure
        :InitialFailure
    elseif retcode == ConvergenceFailure
        :ConvergenceFailure
    elseif retcode == Failure
        :Failure
    end
end

Base.:(==)(retcode::ReturnCode.T, s::Symbol) = Symbol(retcode) == s
Base.:(!=)(retcode::ReturnCode.T, s::Symbol) = Symbol(retcode) != s

function Base.convert(::Type{ReturnCode.T}, retcode::Symbol)
    if retcode == :Default
        ReturnCode.Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT || retcode == :FLOATING_POINT_LIMIT
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
    elseif retcode == :ConvergenceFailure
        ReturnCode.ConvergenceFailure
    elseif retcode == :Failure
        ReturnCode.Failure
    else
        error("$retcode is not a valid return code")
    end
end

successful_retcode(retcode::ReturnCode.T) = retcode == :Success || retcode == :Terminated
