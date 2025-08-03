# Clock functionality is provided by the Moshi extension
# This file only contains error messages when Moshi is not available

function Clock(args...; kwargs...)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function isclock(::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function issolverstepclock(::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function iscontinuous(::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function is_discrete_time_domain(::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function first_clock_tick_time(::Any, ::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end

function canonicalize_indexed_clock(::Any, ::Any)
    error("Clock functionality requires Moshi.jl. Please run `using Moshi` or `import Moshi` to enable clock features.")
end
