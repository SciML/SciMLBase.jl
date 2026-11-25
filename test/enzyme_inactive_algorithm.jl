using SciMLBase
using Enzyme
using Test

import Enzyme: EnzymeRules

# Regression test for SciMLSensitivity.jl#1499.
#
# Solver algorithms are pure configuration and are never differentiated with
# respect to, so the whole `AbstractDEAlgorithm` hierarchy must be declared
# `EnzymeRules.inactive_type`. Without this, on Julia 1.12+ a nested
# polyalgorithm (e.g. the `CompositeAlgorithm` behind `DefaultODEAlgorithm`)
# gets split by the "inline-roots" calling convention and trips Enzyme's
# `roots_activep != activep` assertion inside the `solve_up` custom rule.
struct Issue1499ODEAlg <: SciMLBase.AbstractODEAlgorithm end
struct Issue1499SDEAlg <: SciMLBase.AbstractSDEAlgorithm end
struct Issue1499DAEAlg <: SciMLBase.AbstractDAEAlgorithm end

@testset "AbstractDEAlgorithm is Enzyme-inactive (SciMLSensitivity#1499)" begin
    @test EnzymeRules.inactive_type(Issue1499ODEAlg)
    @test EnzymeRules.inactive_type(Issue1499SDEAlg)
    @test EnzymeRules.inactive_type(Issue1499DAEAlg)
    @test EnzymeRules.inactive_type(SciMLBase.AbstractDEAlgorithm)
end

if isdefined(Base, :ispublic)
    @testset "Sensitivity algorithm supertypes are public" begin
        for name in (
                :AbstractSensitivityAlgorithm,
                :AbstractOverloadingSensitivityAlgorithm,
                :AbstractForwardSensitivityAlgorithm,
                :AbstractAdjointSensitivityAlgorithm,
                :AbstractSecondOrderSensitivityAlgorithm,
                :AbstractShadowingSensitivityAlgorithm,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end
end
