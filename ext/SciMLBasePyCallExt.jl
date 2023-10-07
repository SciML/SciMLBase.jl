module SciMLBasePyCallExt

using PyCall: PyCall, PyObject, PyAny, pyfunctionret, pyimport, hasproperty
using SciMLBase: SciMLBase, solve

# SciML uses a function's arity (number of arguments) to determine if it operates in place.
# PyCall does not preserve arity, so we inspect Python functions to find their arity.
function SciMLBase.numargs(f::PyObject)
    inspect = pyimport("inspect")
    f2 = hasproperty(f, :py_func) ? f.py_func : f
    # if `f` is a bound method (i.e., `self.f`), `getfullargspec` includes
    # `self` in the `args` list. So, we subtract 1 in that case:
    length(first(inspect.getfullargspec(f2))) - inspect.ismethod(f2)
end

# differential equation solutions can be converted to lists, this tells PyCall not
# to perform that conversion automatically when a solution is returned from `solve`
PyCall.PyObject(::typeof(solve)) = pyfunctionret(solve, Any, Vararg{PyAny})

end
