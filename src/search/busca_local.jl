function buscar_localmente00(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, xs::MVector{N, NTuple{N, UInt8}}, visited::MVector{1, Int64}, visited2::MVector{1, Int64}, visited3::MVector{1, Int128}, aas::SVector{N, UInt8}, controls::SVector{N, UInt8}, hmm::ARMBPModel, smoothed::MMatrix{2, N, Float64}, localidade::NTuple{N, Int64}, Δ_localidade::NTuple{N, Float64}, maxheaps::NTuple{N, BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}}, hlru::LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}, ns::MVector{N, Int64}, params::MVector{3, Int64}, previous_inicial::MVector{N, Int64}, previous_final::MVector{N, Int64}, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	flag = false
	setindex!(params, obter_coordenadas00(getindex(xs, getindex(params, 1)), lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited3, Val(N), Val(M), Val(3)), 2)
	setindex!(params, getindex(params, 2), 3)
	setindex!(previous_inicial, getindex(params, 1), getindex(params, 1)) # <-- `previous_inicial` base na posição inicial
	setindex!(previous_final, getindex(params, 3), getindex(params, 1)) # <-- `previous_final` base na posição inicial
	while getindex(params, 3) ≠ N+1 && !isready(resposta)
		flag = false
		let idxtuple = achar_indices(localidade, getindex(params, 1), getindex(params, 3), buffer, Val(N))
			if !isready(resposta) && isnothing(get(hlru, (1, getindex(params, 1), getindex(params, 3)), nothing))
				if !isempty(getfield(maxheaps, getindex(params, 3)))
					extract_all!(getfield(maxheaps, getindex(params, 3)))
				end
				if !isnothing(findfirst(idxtuple))
					first_idx = findfirst(idxtuple) # <-- Zerado de alocação
					push!(getfield(maxheaps, getindex(params, 3)), (sum(map(maximum, eachcol(view(smoothed, :, getindex(params, 1):getindex(params, 3))))) - getfield(Δ_localidade, first_idx), flipat(ntuple(x -> false, Val(N)), first_idx, buffer, Val(N))))
					while !isempty(getfield(maxheaps, getindex(params, 3))) && !isready(resposta)
						resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
							next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
							if !isnothing(findlast(next_base))
								last_idx = findlast(next_base) # <-- Zerado de alocação
								if !isnothing(findnext(idxtuple, last_idx+1))
									next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
								end
							end
							next_base
						end
						ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
						setindex!(params, obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited3, Val(N), Val(M), Val(3)), 2)
						ns[getindex(params, 1)] += 1
						if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
							setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
							setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
							setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
							setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao final
							setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
							if getindex(params, 2) < N
								Dagger.@spawn buscar_restritamente00(getindex(params, 2) - 1, D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, ys, visited2, visited3, aas, controls, hmm, Val(N), Val(M), Val(3))
							end
							flag = true
							break
						end
					end
				end
			elseif !isready(resposta)
				while (!isempty(getfield(maxheaps, getindex(params, 3))) || !isnothing(get(hlru, (getindex(ns, getindex(params, 1)), getindex(params, 1), getindex(params, 3)), nothing))) && !isready(resposta)
					resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
						next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
						if !isnothing(findlast(next_base))
							last_idx = findlast(next_base) # <-- Zerado de alocação
							if !isnothing(findnext(idxtuple, last_idx+1))
								next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
							end
						end
						next_base
					end
					ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
					setindex!(params, obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited3, Val(N), Val(M), Val(3)), 2)
					ns[getindex(params, 1)] += 1
					if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
						setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
						setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
						setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
						setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao início
						setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
						flag = true
						if getindex(params, 2) < N
							Dagger.@spawn buscar_restritamente00(getindex(params, 2) - 1, D, d, sinθ, cosθ, sinω, cosω, F, buffer, lru, resposta, ys, visited2, visited3, aas, controls, hmm, Val(N), Val(M), Val(3))
						end
						break
					end
				end
			end
			if !flag && getindex(params, 1) ≠ 1 && !isready(resposta)
				setindex!(ns, 1, getindex(params, 1))
				setindex!(params, getindex(previous_final, getindex(params, 1)), 3)
				setindex!(params, getindex(previous_inicial, getindex(params, 1)), 1)
			elseif !flag && getindex(params, 1) == 1 && !isready(resposta)
				return(nothing)
			end
		end
	end
end





function buscar_localmente000(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, xs::MVector{N, NTuple{N, UInt8}}, visited::MVector{1, Int64}, visited2::MVector{1, Int128}, smoothed::MMatrix{2, N, Float64}, localidade::NTuple{N, Int64}, Δ_localidade::NTuple{N, Float64}, maxheaps::NTuple{N, BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}}, hlru::LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}, ns::MVector{N, Int64}, params::MVector{3, Int64}, previous_inicial::MVector{N, Int64}, previous_final::MVector{N, Int64}, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	flag = false
	setindex!(params, obter_coordenadas00(getindex(xs, getindex(params, 1)), lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, Val(N), Val(M), Val(3)), 2)
	setindex!(params, getindex(params, 2), 3)
	setindex!(previous_inicial, getindex(params, 1), getindex(params, 1)) # <-- `previous_inicial` base na posição inicial
	setindex!(previous_final, getindex(params, 3), getindex(params, 1)) # <-- `previous_final` base na posição inicial
	while getindex(params, 3) ≠ N+1 && !isready(resposta)
		flag = false
		let idxtuple = achar_indices(localidade, getindex(params, 1), getindex(params, 3), buffer, Val(N))
			if !isready(resposta) && isnothing(get(hlru, (1, getindex(params, 1), getindex(params, 3)), nothing))
				if !isempty(getfield(maxheaps, getindex(params, 3)))
					extract_all!(getfield(maxheaps, getindex(params, 3)))
				end
				if !isnothing(findfirst(idxtuple))
					first_idx = findfirst(idxtuple) # <-- Zerado de alocação
					push!(getfield(maxheaps, getindex(params, 3)), (sum(map(maximum, eachcol(view(smoothed, :, getindex(params, 1):getindex(params, 3))))) - getfield(Δ_localidade, first_idx), flipat(ntuple(x -> false, Val(N)), first_idx, buffer, Val(N))))
					while !isempty(getfield(maxheaps, getindex(params, 3))) && !isready(resposta)
						resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
							next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
							if !isnothing(findlast(next_base))
								last_idx = findlast(next_base) # <-- Zerado de alocação
								if !isnothing(findnext(idxtuple, last_idx+1))
									next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
								end
							end
							next_base
						end
						ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
						setindex!(params, obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, Val(N), Val(M), Val(3)), 2)
						ns[getindex(params, 1)] += 1
						if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
							setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
							setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
							setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
							setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao final
							setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
							flag = true
							break
						end
					end
				end
			elseif !isready(resposta)
				while (!isempty(getfield(maxheaps, getindex(params, 3))) || !isnothing(get(hlru, (getindex(ns, getindex(params, 1)), getindex(params, 1), getindex(params, 3)), nothing))) && !isready(resposta)
					resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
						next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
						if !isnothing(findlast(next_base))
							last_idx = findlast(next_base) # <-- Zerado de alocação
							if !isnothing(findnext(idxtuple, last_idx+1))
								next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
							end
						end
						next_base
					end
					ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
					setindex!(params, obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, Val(N), Val(M), Val(3)), 2)
					ns[getindex(params, 1)] += 1
					if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
						setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
						setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
						setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
						setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao início
						setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
						flag = true
						break
					end
				end
			end
			if !flag && getindex(params, 1) ≠ 1 && !isready(resposta)
				setindex!(ns, 1, getindex(params, 1))
				setindex!(params, getindex(previous_final, getindex(params, 1)), 3)
				setindex!(params, getindex(previous_inicial, getindex(params, 1)), 1)
			elseif !flag && getindex(params, 1) == 1 && !isready(resposta)
				return(nothing)
			end
		end
	end
end


function buscar_localmente000m(D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, xs::MVector{N, NTuple{N, UInt8}}, visited::MVector{1, Int64}, visited2::MVector{1, Int128}, visited3::MVector{1, Int128}, smoothed::MMatrix{2, N, Float64}, localidade::NTuple{N, Int64}, Δ_localidade::NTuple{N, Float64}, maxheaps::NTuple{N, BinaryMaxHeap{Tuple{Float64, NTuple{N, Bool}}}}, hlru::LRU{Tuple{Int64, Int64, Int64}, NTuple{N, Bool}}, ns::MVector{N, Int64}, params::MVector{3, Int64}, previous_inicial::MVector{N, Int64}, previous_final::MVector{N, Int64}, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	flag = false
	setindex!(params, obter_coordenadas00m(getindex(xs, getindex(params, 1)), lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, visited3, Val(N), Val(M), Val(3)), 2)
	setindex!(params, getindex(params, 2), 3)
	setindex!(previous_inicial, getindex(params, 1), getindex(params, 1)) # <-- `previous_inicial` base na posição inicial
	setindex!(previous_final, getindex(params, 3), getindex(params, 1)) # <-- `previous_final` base na posição inicial
	while getindex(params, 3) ≠ N+1 && !isready(resposta)
		flag = false
		let idxtuple = achar_indices(localidade, getindex(params, 1), getindex(params, 3), buffer, Val(N))
			if !isready(resposta) && isnothing(get(hlru, (1, getindex(params, 1), getindex(params, 3)), nothing))
				if !isempty(getfield(maxheaps, getindex(params, 3)))
					extract_all!(getfield(maxheaps, getindex(params, 3)))
				end
				if !isnothing(findfirst(idxtuple))
					first_idx = findfirst(idxtuple) # <-- Zerado de alocação
					push!(getfield(maxheaps, getindex(params, 3)), (sum(map(maximum, eachcol(view(smoothed, :, getindex(params, 1):getindex(params, 3))))) - getfield(Δ_localidade, first_idx), flipat(ntuple(x -> false, Val(N)), first_idx, buffer, Val(N))))
					while !isempty(getfield(maxheaps, getindex(params, 3))) && !isready(resposta)
						resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
							next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
							if !isnothing(findlast(next_base))
								last_idx = findlast(next_base) # <-- Zerado de alocação
								if !isnothing(findnext(idxtuple, last_idx+1))
									next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
									push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
								end
							end
							next_base
						end
						ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
						setindex!(params, obter_coordenadas00m(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, visited3, Val(N), Val(M), Val(3)), 2)
						ns[getindex(params, 1)] += 1
						if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
							setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
							setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
							setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
							setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao final
							setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
							flag = true
							break
						end
					end
				end
			elseif !isready(resposta)
				while (!isempty(getfield(maxheaps, getindex(params, 3))) || !isnothing(get(hlru, (getindex(ns, getindex(params, 1)), getindex(params, 1), getindex(params, 3)), nothing))) && !isready(resposta)
					resp = get!(hlru, (ns[getindex(params, 1)], getindex(params, 1), getindex(params, 3))) do
						next_sum, next_base = pop!(getfield(maxheaps, getindex(params, 3))) # <-- Zerado de alocação
						if !isnothing(findlast(next_base))
							last_idx = findlast(next_base) # <-- Zerado de alocação
							if !isnothing(findnext(idxtuple, last_idx+1))
								next_valid = findnext(idxtuple, last_idx+1) # <-- Zerado de alocação
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum - getfield(Δ_localidade, next_valid), flipat(next_base, next_valid, buffer, Val(N))))
								push!(getfield(maxheaps, getindex(params, 3)), (next_sum + getfield(Δ_localidade, last_idx) - getfield(Δ_localidade, next_valid), switchat(next_base, last_idx, next_valid, buffer, Val(N))))
							end
						end
						next_base
					end
					ys = flipwith(getindex(xs, getindex(params, 1)), localidade, resp, buffer, Val(N)) # <-- Zerado de alocação
					setindex!(params, obter_coordenadas00(ys, lru, resposta, D, d, sinθ, cosθ, sinω, cosω, F, buffer, N, visited, visited2, Val(N), Val(M), Val(3)), 2)
					ns[getindex(params, 1)] += 1
					if !isready(resposta) && getindex(params, 2) > getindex(params, 3)
						setindex!(previous_inicial, getindex(params, 1), getindex(params, 3)+1) # <-- Registrando o inicial antecessor no que será o novo início
						setindex!(previous_final, getindex(params, 3), getindex(params, 3)+1) # <-- Registrando o final antecessor no que será o novo início
						setindex!(xs, ys, getindex(params, 3)+1) # <-- Atualizando o valor de xs no que será o novo início
						setindex!(params, getindex(params, 3)+1, 1) # <-- Atualizando o valor do início para ser igual ao início
						setindex!(params, getindex(params, 2), 3) # <-- Atualizando o valor do final para ser igual ao da nova poda
						flag = true
						break
					end
				end
			end
			if !flag && getindex(params, 1) ≠ 1 && !isready(resposta)
				setindex!(ns, 1, getindex(params, 1))
				setindex!(params, getindex(previous_final, getindex(params, 1)), 3)
				setindex!(params, getindex(previous_inicial, getindex(params, 1)), 1)
			elseif !flag && getindex(params, 1) == 1 && !isready(resposta)
				return(nothing)
			end
		end
	end
end