using SciMLBase, Test
using SciMLBase: should_warn_paramtype, info_paramtype, PARAMTYPE_INFO_MESSAGE

@test should_warn_paramtype([1,"2"]) == true
@test should_warn_paramtype([1,2.0]) == false
@test should_warn_paramtype(SciMLBase.NullParameters()) == false
@test should_warn_paramtype((1,"2")) == false
@test should_warn_paramtype(Dict(:a => 1, :b => "2")) == false


f(x,p,t) = x
x0 = [0.0]
tspan = (0.0,1.0)

@test_logs (:info, PARAMTYPE_INFO_MESSAGE) ODEProblem(f, x0, tspan, [1,"2"])