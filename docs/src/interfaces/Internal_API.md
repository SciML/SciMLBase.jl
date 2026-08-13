# Internal Implementation Reference

!!! warning "Private implementation details"
    The bindings on this page are not public API. They are rendered so that every
    intentional SciMLBase docstring is checked by Documenter, not to establish an
    extension contract. Downstream packages must use the public interfaces documented
    on the other interface pages.

## Problem Representation

```@docs
SciMLBase.AbstractSplitSDEProblem
SciMLBase.AbstractDynamicalSDEProblem
SciMLBase.StandardSDEProblem
SciMLBase.AbstractIncrementingODEProblem
SciMLBase.DEElement
SciMLBase.DESensitivity
SciMLBase.AbstractWrappedFunction
SciMLBase.AbstractReactionNetwork
```

## Construction And Remake

```@docs
SciMLBase.num_types_in_tuple
SciMLBase._get_new_A_b
SciMLBase.UpdateABWrapper
SciMLBase._similar_namedtuple_merge_ignore_nothing
SciMLBase._has_type_erased_params
SciMLBase._reconstruct_as_type
SciMLBase.handle_varmap
SciMLBase.warn_paramtype
```

## Initialization

```@docs
SciMLBase.evaluate_f
SciMLBase._evaluate_f
SciMLBase._vec
```

## Interpolation

```@docs
SciMLBase.tspan_indices
SciMLBase.interpolant
SciMLBase.interpolant!
SciMLBase.interpolation
SciMLBase.interpolation!
SciMLBase.linear_interpolant
SciMLBase.linear_interpolant!
SciMLBase.hermite_interpolant
SciMLBase.hermite_interpolant!
```

## Symbolic Save Selection

```@docs
SciMLBase.translate_symbolic_save_idxs
SciMLBase._invalid_save_idxs_symbol_error
```

## Diagnostics And Maintenance

```@docs
SciMLBase.@CSI_str
SciMLBase.strip_solution
SciMLBase.controller_kwargs
SciMLBase.undefined_exports
SciMLBase._has_sciml_in_stacktrace
```
