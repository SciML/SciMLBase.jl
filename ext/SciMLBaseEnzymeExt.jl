module SciMLBaseEnzymeExt

using SciMLBase
import Enzyme: EnzymeRules

# Enzyme rules for SciMLBase abstract types
#
# Sensitivity algorithms define HOW to compute derivatives, not WHAT to differentiate.
# They should be treated as inactive (constant) during Enzyme differentiation to prevent
# errors when they are stored in problem structures that Enzyme differentiates through.
#
# This fixes issues like SciMLSensitivity.jl#1225 where passing `sensealg` to ODEProblem 
# constructor would fail with Enzyme.

# All sensitivity algorithm types should be inactive for Enzyme differentiation
EnzymeRules.inactive_type(::Type{<:AbstractSensitivityAlgorithm}) = true

end