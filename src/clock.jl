# Clock system implementation
# 
# When Moshi is available as an extension, it provides advanced @data/@match functionality.
# When Moshi is not loaded, we fall back to simple struct-based implementations.

# Include fallback implementations (always available)
include("clock_fallback.jl")
