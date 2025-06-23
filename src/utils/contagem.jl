function contar_dfs(xs::NTuple{N, Bool}) where {N}
    binary2int = [Int64(x) for x ∈ xs]
    exp_base_2 = [2^i for i ∈ N-1:-1:0]
    return(trunc(Int128, sum([sum(binary2int[1:N-i+1] .* exp_base_2[i:N]) + 1 for i ∈ 1:N])))
end


function tempo_esperado_primeira_passagem0(xs::NTuple{N, Bool}) where {N}
	P = zeros(Float64, (N+1, N+1))
	aux = collect(xs)
	for b ∈ 1:lastindex(xs)
		P[b, b+1] = 0.5 # Tentar com outros valores
		teste = vcat(aux[1:b-1], !aux[b])
		for d ∈ lastindex(teste):-1:0
			if d ≠ 0 && last(teste, d) == first(xs, d)
				P[b, d+1] = 0.5 # Tentar com outros valores
				break
			elseif d == 0
				P[b, 1] = 0.5 # Tentar com outros valores
			end
		end
	end
	P[N+1, N+1] = 1.0
	trunc(Int128, first(inv(I - view(P, 1:N, 1:N)) * ones(Float64, N)))
end


function tempo_esperado_primeira_passagem(xs::NTuple{N, Bool}) where {N}
	P = zeros(Float64, (N+1, N+1))
	for b ∈ 1:lastindex(xs)
		P[b, b+1] = 0.5 # Tentar com outros valores
		if b == 1
			P[b, b] = 0.5 # Tentar com outros valores
		else
			P[b, b-1] = 0.5 # Tentar com outros valores
		end
	end
	P[N+1, N+1] = 1.0
	trunc(Int128, first(inv(I - view(P, 1:N, 1:N)) * ones(Float64, N)))
end