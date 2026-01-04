module SciMLBaseMooncakeExt

using SciMLBase, Mooncake
using SciMLBase: ADOriginator, ChainRulesOriginator, MooncakeOriginator
import Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_rrule, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback

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
