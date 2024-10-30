"""
    $(TYPEDEF)

A collection of all the data required for `OverrideInit`.
"""
struct OverrideInitData{IProb, UIProb, IProbMap, IProbPmap}
    """
    The `AbstractNonlinearProblem` to solve for initialization.
    """
    initializeprob::IProb
    """
    A function which takes `(initializeprob, prob)` and updates
    the parameters of the former with their values in the latter.
    """
    update_initializeprob!::UIProb
    """
    A function which takes the solution of `initializeprob` and returns
    the state vector of the original problem.
    """
    initializeprobmap::IProbMap
    """
    A function which takes the solution of `initializeprob` and returns
    the parameter object of the original problem.
    """
    initializeprobpmap::IProbPmap

    function OverrideInitData(initprob::I, update_initprob!::J, initprobmap::K,
            initprobpmap::L) where {I, J, K, L}
        @assert initprob isa Union{NonlinearProblem, NonlinearLeastSquaresProblem}
        return new{I, J, K, L}(initprob, update_initprob!, initprobmap, initprobpmap)
    end
end
