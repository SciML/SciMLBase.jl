using SciMLBase
using Test

# Synthetic saved data for a two-state trajectory.
t = [0.0, 1.0, 2.0, 3.0]
u = [[1.0, 2.0], [1.5, 2.4], [2.1, 2.9], [2.8, 3.5]]
du = [[0.4, 0.3], [0.5, 0.4], [0.6, 0.5], [0.7, 0.6]]

hermite = SciMLBase.HermiteInterpolation(t, u, du)
linear = SciMLBase.LinearInterpolation(t, u)

# Non-dense `du` must share the container type of the dense case so that the
# concrete type is invariant across dense/non-dense.
empty_du = similar(du, 0)
basic_dense = SciMLBase.BasicInterpolation(t, u, du, true)
basic_nondense = SciMLBase.BasicInterpolation(t, u, empty_du, false)

# points to probe: interior nodes and interval midpoints (non-node values force
# the interpolant math rather than exact node lookup)
scalar_tvals = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0]
vector_tvals = [0.5, 1.5, 2.5]

oop(interp, tvals, idxs, deriv, cont) = interp(tvals, idxs, deriv, nothing, cont)

getu(x) = x isa SciMLBase.AbstractDiffEqArray ? x.u : x

@testset "Type invariance across dense/non-dense" begin
    @test typeof(basic_dense) == typeof(basic_nondense)
    # sanity: the switched type differs from the delegates it wraps
    @test basic_dense isa SciMLBase.BasicInterpolation
    @test basic_nondense isa SciMLBase.BasicInterpolation
end

@testset "interp_summary" begin
    @test SciMLBase.interp_summary(basic_dense) == "3rd order Hermite"
    @test SciMLBase.interp_summary(basic_nondense) == "1st order linear"
    @test SciMLBase.interp_summary(basic_dense) ==
        SciMLBase.interp_summary(hermite)
    @test SciMLBase.interp_summary(basic_nondense) ==
        SciMLBase.interp_summary(linear)
end

@testset "Out-of-place correctness vs delegates" begin
    for cont in (:left, :right)
        for deriv in (Val{0}, Val{1})
            for idxs in (nothing, 1, 2, [1], [1, 2])
                # dense == Hermite
                for tv in scalar_tvals
                    @test getu(oop(basic_dense, tv, idxs, deriv, cont)) ==
                        getu(oop(hermite, tv, idxs, deriv, cont))
                end
                @test getu(oop(basic_dense, vector_tvals, idxs, deriv, cont)) ==
                    getu(oop(hermite, vector_tvals, idxs, deriv, cont))
                # non-dense == Linear
                for tv in scalar_tvals
                    @test getu(oop(basic_nondense, tv, idxs, deriv, cont)) ==
                        getu(oop(linear, tv, idxs, deriv, cont))
                end
                @test getu(oop(basic_nondense, vector_tvals, idxs, deriv, cont)) ==
                    getu(oop(linear, vector_tvals, idxs, deriv, cont))
            end
        end
    end
end

@testset "In-place scalar correctness vs delegates" begin
    for cont in (:left, :right)
        for deriv in (Val{0}, Val{1})
            for tv in scalar_tvals
                ob = zeros(2)
                oh = zeros(2)
                basic_dense(ob, tv, nothing, deriv, nothing, cont)
                hermite(oh, tv, nothing, deriv, nothing, cont)
                @test ob == oh

                ob2 = zeros(2)
                ol = zeros(2)
                basic_nondense(ob2, tv, nothing, deriv, nothing, cont)
                linear(ol, tv, nothing, deriv, nothing, cont)
                @test ob2 == ol
            end
        end
    end
end

@testset "In-place vector correctness vs delegates" begin
    for cont in (:left, :right)
        for deriv in (Val{0}, Val{1})
            ob = [zeros(2) for _ in vector_tvals]
            oh = [zeros(2) for _ in vector_tvals]
            basic_dense(ob, vector_tvals, nothing, deriv, nothing, cont)
            hermite(oh, vector_tvals, nothing, deriv, nothing, cont)
            @test ob == oh

            ob2 = [zeros(2) for _ in vector_tvals]
            ol = [zeros(2) for _ in vector_tvals]
            basic_nondense(ob2, vector_tvals, nothing, deriv, nothing, cont)
            linear(ol, vector_tvals, nothing, deriv, nothing, cont)
            @test ob2 == ol
        end
    end
end

@testset "strip_interpolation" begin
    @test SciMLBase.strip_interpolation(basic_dense) === basic_dense
    @test SciMLBase.strip_interpolation(basic_nondense) === basic_nondense
end

@testset "Inference and allocations" begin
    for interp in (basic_dense, basic_nondense)
        for deriv in (Val{0}, Val{1})
            @inferred interp(1.5, nothing, deriv, nothing, :left)
            @inferred interp(1.5, 1, deriv, nothing, :left)
            @inferred interp(1.5, [1, 2], deriv, nothing, :left)
            out = zeros(2)
            @inferred interp(out, 1.5, nothing, deriv, nothing, :left)
        end
    end

    # The scalar in-place path must be allocation-free post-warmup for the
    # switched type in both modes AND the legacy types: no per-call wrapper
    # construction, no dynamic dispatch from the Type-argument
    # non-specialization heuristic on `deriv` (hence `deriv::D where {D}` in
    # this harness too — an unannotated Type slot would re-box in the harness
    # itself and be misattributed to the library), and no boxed ordering from
    # a runtime-`rev` sorted search.
    function scalar_inplace_allocs(itp, out, deriv::D) where {D}
        itp(out, 1.5, nothing, deriv, nothing, :left)
        itp(out, 1.5, nothing, deriv, nothing, :left)
        return @allocated itp(out, 1.5, nothing, deriv, nothing, :left)
    end
    out = zeros(2)
    for deriv in (Val{0}, Val{1})
        @test scalar_inplace_allocs(basic_dense, out, deriv) == 0
        @test scalar_inplace_allocs(basic_nondense, out, deriv) == 0
        @test scalar_inplace_allocs(hermite, out, deriv) == 0
        @test scalar_inplace_allocs(linear, out, deriv) == 0
    end

    # Vector-tvals calls return a DiffEqArray whose two SymbolCache metadata
    # type parameters are not inferred for ANY interpolation type (a
    # pre-existing property of the DiffEqArray constructor; verified identical
    # for HermiteInterpolation/LinearInterpolation on master). The data
    # parameters ARE fully inferred. So for the vector path, pin inference
    # PARITY: the runtime dense branch must add zero inference degradation
    # over the legacy type it corresponds to.
    argtypes(idxs) = (Vector{Float64}, idxs, Type{Val{0}}, Nothing, Symbol)
    for idxs_T in (Nothing, Int)
        rt_basic_dense = Base.return_types(basic_dense, argtypes(idxs_T))
        rt_basic_nondense = Base.return_types(basic_nondense, argtypes(idxs_T))
        rt_hermite = Base.return_types(hermite, argtypes(idxs_T))
        rt_linear = Base.return_types(linear, argtypes(idxs_T))
        @test only(rt_basic_dense) == only(rt_hermite)
        @test only(rt_basic_nondense) == only(rt_linear)
        # and the inferred type is a DiffEqArray, not Any or a Union
        @test only(rt_basic_dense) <: SciMLBase.AbstractDiffEqArray
        @test only(rt_basic_nondense) <: SciMLBase.AbstractDiffEqArray
    end
end

@testset "sensitivitymode toggling" begin
    sdense = SciMLBase.enable_interpolation_sensitivitymode(basic_dense)
    snondense = SciMLBase.enable_interpolation_sensitivitymode(basic_nondense)
    @test sdense isa SciMLBase.BasicInterpolation
    @test sdense.sensitivitymode
    @test sdense.dense           # dense flag preserved
    @test snondense.sensitivitymode
    @test !snondense.dense
    # type still invariant after enabling sensitivity mode
    @test typeof(sdense) == typeof(snondense)
    # exact node lookups still work under sensitivity mode
    @test getu(oop(sdense, 1.0, nothing, Val{0}, :left)) == u[2]
    # interpolating between nodes must error under sensitivity mode
    @test_throws ErrorException oop(sdense, 1.5, nothing, Val{0}, :left)
    @test_throws ErrorException oop(snondense, 1.5, nothing, Val{0}, :left)
end
