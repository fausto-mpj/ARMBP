"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ListViterbiStorage{R}
    "k-most likely state sequence `q[t,l] = argmaxᵢ ℙ(X[t]=i | Y[1:T])` if `l = 1`"
    q::Matrix{Int}
    "one joint loglikelihood per pair of observation sequence and k-most likely state sequence"
    logL::Matrix{R}
    logB::Matrix{R}
    ϕ::Array{R}
    ψ::Array{Int}
    "ranking for the k-most likely paths"
    ρ::Array{Int}
end

"""
$(SIGNATURES)
"""
function initialize_list_viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    list_len::Integer,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    N, T, K = size(hmm, control_seq[1]), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    q = Matrix{Int}(undef, T, list_len)
    logL = Matrix{R}(undef, K, list_len)
    logB = Matrix{R}(undef, N, T)
    ϕ = Array{R}(undef, N, T, list_len)
    ψ = Array{Int}(undef, N, T, list_len)
    ρ = Array{Int}(undef, N, T, list_len)
    return ListViterbiStorage(q, logL, logB, ϕ, ψ, ρ)
end

function _list_viterbi!(
    storage::ListViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    list_len::Integer,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
) where {R}
    (; q, logB, ϕ, ψ, logL, ρ) = storage
    t1, t2 = seq_limits(seq_ends, k)
    logBₜ₁ = view(logB, :, t1)
    obs_logdensities!(logBₜ₁, hmm, obs_seq[t1], control_seq[t1], missing)
    loginit = log_initialization(hmm, control_seq[t1])
    ϕ[:, t1, :] .= loginit .+ logBₜ₁

    for t in (t1 + 1):t2
        logBₜ = view(logB, :, t)
        obs_logdensities!(
            logBₜ, hmm, obs_seq[t], control_seq[t], previous_obs(hmm, obs_seq, t)
        )
        logtrans = log_transition_matrix(hmm, control_seq[t]) # See forward.jl, line 106.
        ϕₜ, ϕₜ₋₁ = view(ϕ, :, t, :), view(ϕ, :, t - 1, :)
        ψₜ = view(ψ, :, t, :)
        ρₜ = view(ρ, :, t, :)
        argmaxplus_transmul!(ϕₜ, ψₜ, logtrans, ϕₜ₋₁, ρₜ, list_len)
        ϕₜ .+= logBₜ
    end

    z = Tuple{R, Int, Int}[]
    for i ∈ 1:size(hmm, control_seq[t1])
        for l ∈ 1:list_len
            push!(z, (ϕ[i, t2, l], i, l))
        end
    end
    sort!(z, rev = true)
    
    for l ∈ 1:list_len
        ρₜ = z[l][3]
        logL[k, l] = ϕ[z[l][2], t2, l]
        q[t2, l] = z[l][2]
        for t ∈ (t2-1):-1:t1
            q[t, l] = ψ[q[t+1, l], t+1, ρₜ]
            ρₜ = ρ[q[t+1, l], t+1, ρₜ]
        end
    end
    @argcheck all(isfinite.(logL[k,:]))
    return nothing
end


"""
$(SIGNATURES)
"""
function list_viterbi!(
    storage::ListViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    list_len::Integer,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
) where {R}
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _list_viterbi!(storage, hmm, obs_seq, list_len, control_seq, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            _list_viterbi!(storage, hmm, obs_seq, list_len, control_seq, seq_ends, k)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

Return a tuple `(storage.q, storage.logL)` where `storage` is of type [`ViterbiStorage`](@ref).
"""
function list_viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    list_len::Integer,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    storage = initialize_list_viterbi(hmm, obs_seq, list_len, control_seq; seq_ends)
    list_viterbi!(storage, hmm, obs_seq, list_len, control_seq; seq_ends)
    return storage.q, storage.logL
end
