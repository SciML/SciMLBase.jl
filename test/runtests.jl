using Pkg
using SafeTestsets
using Test
using SciMLBase

@test isempty(detect_ambiguities(SciMLBase))

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )

@time begin
@time @safetestset "Existence functions" begin include("existence_functions.jl") end
@time @safetestset "Integrator interface" begin include("integrator_tests.jl") end
end
