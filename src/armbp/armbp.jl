function iniciar_armbp_local(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, aas::SVector{N, UInt8}, controls::SVector{N, UInt8}, hmm::ARMBPModel, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	buffer = AllocBuffer()
	lru = LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}(maxsize = 500000)
	hlru = LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}(maxsize = 500000)
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x00, buffer, Val(N)), 1))
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x01, buffer, Val(N)), 1))
	resposta = Channel{NTuple{M, Float64}}(1)
	smoothed = MMatrix{2, N, Float64}(first(forward_backward(hmm.hmm, aas, controls))) # <-- ESTÁ ALOCANDO 18.5KiB!
	xs = MVector{N, NTuple{N, UInt8}}(i for i ∈ Iterators.repeated(Tuple(UInt8.(map(argmax, eachcol(smoothed))) .- 0x01), N))
	vΔ_localidade = abs.(vec(diff(smoothed, dims = 1)))
	localidade = Tuple(sortperm(vΔ_localidade))
	Δ_localidade = ntuple(x -> vΔ_localidade[getfield(localidade, x)], Val(N))
	maxheaps = ntuple(x-> BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}(), Val(N))
	sizehint!.(maxheaps, 50 * N)
	ns = @MVector ones(Int64, N)
	params = @MVector ones(Int64, 3)
	visited1 = @MVector zeros(Int64, 1)
	visited2 = @MVector zeros(Int64, 1)
	visited3 = @MVector zeros(Int64, 1)
	visited4 = @MVector zeros(Int128, 1)
	previous_inicial = @MVector ones(Int64, N)
	previous_final = @MVector ones(Int64, N)
	buscar_localmente000(D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, xs, visited1, visited4, smoothed, localidade, Δ_localidade, maxheaps, hlru, ns, params, previous_inicial, previous_final, Val(N), Val(M), Val(3))
	if !isready(resposta)
		put!(resposta, ntuple(x -> 0.0, Val(M)))
	end
	return(fetch(resposta), only(visited1), only(visited2), only(visited3), only(visited4))
end


function iniciar_armbp_localm(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, aas::SVector{N, UInt8}, controls::SVector{N, UInt8}, hmm::ARMBPModel, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	buffer = AllocBuffer()
	lru = LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}(maxsize = 500000)
	hlru = LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}(maxsize = 500000)
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x00, buffer, Val(N)), 1))
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x01, buffer, Val(N)), 1))
	resposta = Channel{NTuple{M, Float64}}(1)
	smoothed = MMatrix{2, N, Float64}(first(forward_backward(hmm.hmm, aas, controls))) # <-- ESTÁ ALOCANDO 18.5KiB!
	xs = MVector{N, NTuple{N, UInt8}}(i for i ∈ Iterators.repeated(Tuple(UInt8.(map(argmax, eachcol(smoothed))) .- 0x01), N))
	vΔ_localidade = abs.(vec(diff(smoothed, dims = 1)))
	localidade = Tuple(sortperm(vΔ_localidade))
	Δ_localidade = ntuple(x -> vΔ_localidade[getfield(localidade, x)], Val(N))
	maxheaps = ntuple(x-> BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}(), Val(N))
	sizehint!.(maxheaps, 50 * N)
	ns = @MVector ones(Int64, N)
	params = @MVector ones(Int64, 3)
	visited1 = @MVector zeros(Int64, 1)
	visited2 = @MVector zeros(Int64, 1)
	visited3 = @MVector zeros(Int64, 1)
	visited4 = @MVector zeros(Int128, 1)
	visited5 = @MVector zeros(Int128, 1)
	previous_inicial = @MVector ones(Int64, N)
	previous_final = @MVector ones(Int64, N)
	buscar_localmente000m(D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, xs, visited1, visited4, visited5, smoothed, localidade, Δ_localidade, maxheaps, hlru, ns, params, previous_inicial, previous_final, Val(N), Val(M), Val(3))
	if !isready(resposta)
		put!(resposta, ntuple(x -> 0.0, Val(M)))
	end
	return(fetch(resposta), only(visited1), only(visited2), only(visited3), only(visited4), only(visited5))
end



function iniciar_armbp_total(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, aas::SVector{N, UInt8}, controls::SVector{N, UInt8}, hmm::ARMBPModel, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	buffer = AllocBuffer()
	lru = LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}(maxsize = 500000)
	hlru = LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}(maxsize = 500000)
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x00, buffer, Val(N)), 1))
	setindex!(lru, (ntuple(x -> 0.0, Val(M)), Quaternion{Float64}(1.0, 0.0, 0.0, 0.0)), (gerar_tupla_inicial(0x01, buffer, Val(N)), 1))
	resposta = Channel{NTuple{M, Float64}}(1)
	smoothed = MMatrix{2, N, Float64}(first(forward_backward(hmm.hmm, aas, controls))) # <-- ESTÁ ALOCANDO 18.5KiB!
	xs = MVector{N, NTuple{N, UInt8}}(i for i ∈ Iterators.repeated(Tuple(UInt8.(map(argmax, eachcol(smoothed))) .- 0x01), N))
	vΔ_localidade = abs.(vec(diff(smoothed, dims = 1)))
	localidade = Tuple(sortperm(vΔ_localidade))
	Δ_localidade = ntuple(x -> vΔ_localidade[getfield(localidade, x)], Val(N))
	maxheaps = ntuple(x-> BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}(), Val(N))
	sizehint!.(maxheaps, 10 * N)
	ns = @MVector ones(Int64, N)
	params = @MVector ones(Int64, 3)
	visited1 = @MVector zeros(Int64, 1)
	visited2 = @MVector zeros(Int64, 1)
	visited3 = @MVector zeros(Int64, 1)
	visited4 = @MVector zeros(Int128, 1)
	previous_inicial = @MVector ones(Int64, N)
	previous_final = @MVector ones(Int64, N)
	@sync begin
		Dagger.@spawn buscar_globalmente00(D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, getindex(xs, 1), visited3, visited4, aas, controls, hmm.hmm, Val(N), Val(M), Val(3))
		Dagger.@spawn buscar_localmente00(D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, xs, visited1, visited2, visited4, aas, controls, hmm, smoothed, localidade, Δ_localidade, maxheaps, hlru, ns, params, previous_inicial, previous_final, Val(N), Val(M), Val(3))
	end
	final = fetch(resposta)
	return(final, only(visited1), only(visited2), only(visited3), only(visited4))
	#if !Dagger.in_task() && !isready(resposta)
	#	put!(resposta, ntuple(x -> 0.0, Val(M)))	
	#end
end