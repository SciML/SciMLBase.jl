module SciMLBaseDistributionsExt

using Distributions: Distributions
using SciMLBase: SciMLBase

SciMLBase.handle_distribution_u0(_u0::Distributions.Sampleable) = rand(_u0)
SciMLBase.isdistribution(_u0::Distributions.Sampleable) = true

end
