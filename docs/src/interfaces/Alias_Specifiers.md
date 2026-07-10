# [Alias Specifier Interface](@id alias_specifier_interface)

An [`AbstractAliasSpecifier`](@ref SciMLBase.AbstractAliasSpecifier) is
associated with each SciML problem type that allows solver caches to reuse
problem inputs. Alias specifiers hold `Union{Bool, Nothing}` fields named after
the input they control. For example, to tell an ODE solver that it may keep a
reference to the initial state, pass
`solve(prob, alias = ODEAliasSpecifier(alias_u0 = true))`.

The common rules are:

  - `true` permits aliasing the corresponding input;
  - `false` requests that the solver avoid aliasing the corresponding input;
  - `nothing` delegates the choice to the solver and algorithm default; and
  - `alias = true` or `alias = false` in a specifier constructor applies the
    same policy to every stored field of that concrete specifier.

Alias specifiers are ownership hints rather than solver guarantees. Solvers may
still copy inputs when the algorithm requires a different memory layout or when
copying is required for correctness.

```@docs
SciMLBase.AbstractAliasSpecifier
SciMLBase.LinearAliasSpecifier
SciMLBase.NonlinearAliasSpecifier
SciMLBase.ODEAliasSpecifier
SciMLBase.RODEAliasSpecifier
SciMLBase.SDEAliasSpecifier
SciMLBase.DAEAliasSpecifier
SciMLBase.DDEAliasSpecifier
SciMLBase.SDDEAliasSpecifier
SciMLBase.BVPAliasSpecifier
SciMLBase.OptimizationAliasSpecifier
SciMLBase.IntegralAliasSpecifier
SciMLBase.DiscreteAliasSpecifier
SciMLBase.ImplicitDiscreteAliasSpecifier
SciMLBase.AnalyticalAliasSpecifier
SciMLBase.SteadyStateAliasSpecifier
```
