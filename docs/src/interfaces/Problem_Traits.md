# [Problem Traits](@id problem_traits)

Problem traits expose properties that are stored on concrete problem types and
used by solver dispatch. Solver packages should query these traits instead of
reconstructing the answer from fields or callback method tables.

- `isinplace(prob)` reports the mutating convention selected by the problem
  constructor. This value is part of the problem type and should be preserved by
  `remake`, problem conversion, and wrapper problems.
- `problem_type(prob)` returns public construction-layout metadata when several
  convenience constructors share one concrete problem representation. It returns
  `nothing` when no separate marker is needed.
- `is_diagonal_noise(prob)` reports whether an SDE-like problem should be
  treated as diagonal noise. This is usually determined by the absence of a
  `noise_rate_prototype`, while non-stochastic problem types return `false`.

```@docs
SciMLBase.isinplace(prob::SciMLBase.AbstractDEProblem)
SciMLBase.problem_type
SciMLBase.is_diagonal_noise
```
