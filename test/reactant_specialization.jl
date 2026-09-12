using SciMLBase, Reactant, Test
using SciMLBase: AutoSpecialize, FullSpecialize, NoSpecialize, specialization

f(u, p, t) = u
function specialization_checks(u)
    f_auto = ODEFunction{false, AutoSpecialize}(f)
    f_full = ODEFunction{false, FullSpecialize}(f)
    f_none = ODEFunction{false, NoSpecialize}(f)
    return u .* (
        specialization(f_auto) === FullSpecialize &&
            specialization(typeof(f_auto)) === FullSpecialize &&
            specialization(f_full) === FullSpecialize &&
            specialization(f_none) === NoSpecialize
    )
end

f_auto = ODEFunction{false, AutoSpecialize}(f)
@test specialization(f_auto) === AutoSpecialize
@test specialization(typeof(f_auto)) === AutoSpecialize
u = Reactant.to_rarray(Float32[1, 2])
@test Array(Reactant.@jit(specialization_checks(u))) == Float32[1, 2]
@test specialization(f_auto) === AutoSpecialize
