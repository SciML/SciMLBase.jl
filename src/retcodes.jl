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
        Default
    elseif retcode == :Success || retcode == :EXACT_SOLUTION_LEFT || retcode == :FLOATING_POINT_LIMIT
        Success
    elseif retcode == :Terminated
        Terminated
    elseif retcode == :MaxIters || retcode == :MAXITERS_EXCEED
        MaxIters
    elseif retcode == :DtLessThanMin
        DtLessThanMin
    elseif retcode == :Unstable
        Unstable
    elseif retcode == :InitialFailure
        InitialFailure
    elseif retcode == :ConvergenceFailure
        ConvergenceFailure
    elseif retcode == :Failure
        Failure
    else
        error("$retcode is not a valid return code")
    end
end

successful_retcode(retcode::ReturnCode.T) = retcode == :Success || retcode == :Terminated
