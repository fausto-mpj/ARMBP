function buscar_globalmente00(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, xs::NTuple{N, UInt8}, visited::MVector{1, Int64}, visited2::MVector{1, Int64}, aas::SVector{N, UInt8}, controls::SVector{N, UInt8}, hmm::ARHMM, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	#println("Iniciando Busca Global...")
	#println("Viterbi: $(first(list_viterbi(hmm, aas, 10*N, controls)))")
	for caminho ∈ eachcol(first(list_viterbi(hmm, aas, 10*N, controls)))
		#println("(Global) caminho: $(caminho)")
		if isready(resposta)
			#println("...fim da Busca Global.")
			return(nothing)
		end
		ys = converter_caminho(xs, caminho, buffer, Val(N))
		#println("(Global) ys: $(ys)")
		if !isready(resposta) && obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, Val(N), Val(M), Val(3)) == N+1
			put!(resposta, first(getindex(lru, (ys, N))))
			break
		end
	end
	return(nothing)
	#println("...fim da Busca Global.")
end