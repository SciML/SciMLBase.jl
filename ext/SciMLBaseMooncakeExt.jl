module SciMLBaseMooncakeExt

using SciMLBase, Mooncake
using SciMLBase: ADOriginator, ChainRulesOriginator, MooncakeOriginator
import Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_rrule, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback

# OverrideInitData and ODENLStepData are solver/initialization infrastructure
# embedded in ODEFunction type parameters. They are not differentiable, but their
# deeply nested type parameters (NonlinearProblem, RuntimeGeneratedFunction,
# InitializationMetadata, etc.) cause Mooncake's abstract interpretation to hang
# during type inference. Marking them NoTangent avoids generating tangent code
# for these fields entirely.
Mooncake.tangent_type(::Type{<:SciMLBase.OverrideInitData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:SciMLBase.ODENLStepData}) = Mooncake.NoTangent

@zero_adjoint MinimalCtx Tuple{typeof(SciMLBase.numargs), Any}
@is_primitive MinimalCtx Tuple{
    typeof(SciMLBase.set_mooncakeoriginator_if_mooncake), SciMLBase.ChainRulesOriginator,
}

@mooncake_overlay SciMLBase.set_mooncakeoriginator_if_mooncake(x::SciMLBase.ADOriginator) = SciMLBase.MooncakeOriginator()

function rrule!!(
        f::CoDual{typeof(SciMLBase.set_mooncakeoriginator_if_mooncake)},
        X::CoDual{SciMLBase.ChainRulesOriginator}
    )
    return zero_fcodual(SciMLBase.MooncakeOriginator()), NoPullback(f, X)
end


end
