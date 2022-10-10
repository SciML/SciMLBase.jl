@enum(ReturnCode,Default,Success,Terminated,MaxIters,DtLessThanMin,Unstable,
      InitialFailure,ConvergenceFailure,Failure)

function Base.Symbol(retcode::ReturnCode)
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

Base.:(==)(retcode::ReturnCode, s::Symbol) = Symbol(retcode) == s
function Base.convert(::Type{ReturnCode}, retcode::Symbol)
    if retcode == :Default
        Default
    elseif retcode == :Success
        Success
    elseif retcode == :Terminated
        Terminated
    elseif retcode == :MaxIters
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
    end
end

successful_retcode(retcode::ReturnCode) = retcode == :Success || retcode == :Terminated
