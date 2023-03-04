module SciMLBaseChainRulesCoreExt

using SciMLBase
isdefined(Base, :get_extension) ? (import ChainRulesCore) : (import ..ChainRulesCore)

function ChainRulesCore.rrule(::Type{
                                     <:SciMLBase.PDETimeSeriesSolution{T, N, uType, Disc, Sol, DType, tType, domType, ivType, dvType,
                                     P, A,
                                     IType}}, u,
                              args...) where {T, N, uType, Disc, Sol, DType, tType, domType, ivType, dvType,
                              P, A,
                              IType}
    function PDETimeSeriesSolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    SciMLBase.PDETimeSeriesSolution{T, N, uType, Disc, Sol, DType, tType, domType, ivType, dvType,
    P, A,
    IType}(u, args...), PDETimeSeriesSolutionAdjoint
end

function ChainRulesCore.rrule(::Type{
                                     <:SciMLBase.PDENoTimeSolution{T, N, uType, Disc, Sol, domType, ivType, dvType, P, A,
                                     IType}}, u,
                              args...) where {T, N, uType, Disc, Sol, domType, ivType, dvType, P, A,
                              IType}
    function PDENoTimeSolutionAdjoint(ȳ)
        (NoTangent(), ȳ, ntuple(_ -> NoTangent(), length(args))...)
    end

    SciMLBase.PDENoTimeSolution{T, N, uType, Disc, Sol, domType, ivType, dvType, P, A,
    IType}(u, args...), PDENoTimeSolutionAdjoint
end

end
