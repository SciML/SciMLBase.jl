using SciMLBase, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SciMLBase)
    Aqua.test_ambiguities(SciMLBase, recursive = false)
    Aqua.test_deps_compat(SciMLBase)
    Aqua.test_piracies(SciMLBase)
    Aqua.test_project_extras(SciMLBase)
    Aqua.test_stale_deps(SciMLBase)
    Aqua.test_unbound_args(SciMLBase)
    Aqua.test_undefined_exports(SciMLBase)
end
