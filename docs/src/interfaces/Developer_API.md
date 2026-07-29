# Developer API

!!! warning "Developer API, not user API"
    The contracts on this page are versioned for SciML solver, symbolic-system,
    and numeric-wrapper packages. Application code should use `solve`,
    `remake`, solution interpolation, and return-code interfaces instead of
    calling or extending these hooks.

## Interpolation Hooks

```@docs
SciMLBase.strip_interpolation
```

## Symbolic Initialization And Remake Hooks

```@docs
SciMLBase.get_root_indp
SciMLBase.has_initializeprob
SciMLBase.late_binding_update_u0_p
```

## Numeric Wrapper Hooks

```@docs
SciMLBase.value
SciMLBase.unitfulvalue
```

## Integrator Hook

```@docs
SciMLBase.last_step_failed
```

## Solver Preparation Hooks

Solver packages use these hooks to derive the effective problem data for an individual
solve call and to reject unsupported problem-algorithm pairings. They are not an
application-facing replacement for `solve`, `init`, or `remake`.

```@docs
SciMLBase.get_concrete_p
SciMLBase.get_concrete_u0
SciMLBase.isconcreteu0
SciMLBase.promote_u0
SciMLBase.get_concrete_problem
SciMLBase.check_prob_alg_pairing
SciMLBase.KeywordArgError
SciMLBase.keyword_arg_silent
SciMLBase.@add_kwonly
```
