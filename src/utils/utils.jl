# Função para achar a chave de um dicionário/hashtable a partir do valor
function contraimagem(dict::Dict, valor::Any)
    for chave in keys(dict)
        if dict[chave] == valor
            return (chave)
        end
    end
    error("ERRO: Esse dicionário não é uma função inversível.")
end


function find_nth(f::Function, v::Vector{Union{Nothing, Char}}, n::Int64)
	aux = findnext(f, v, 1)
	for i ∈ 1:(n-1)
		aux = findnext(f, v, aux + 5)
		isnothing(aux) && break
	end
	return(aux)
end


function extrair_trecho(v::Vector{Union{Nothing, Char}}, n::Int64)
	lbound = find_nth(isnothing, v, n)
	if isnothing(lbound)
		return(nothing)
	else
		hbound = findnext(isnothing, v, lbound+5)
		if isnothing(hbound)
			return(lbound:lastindex(v))
		else
			return(lbound:(hbound-1))
		end
	end
end


function extrair_segmento(m::Int64, n::Int64, lbound::Int64, hbound::Int64)
	resp = Tuple{Int64, UnitRange{Int64}}[]
	for i ∈ m:n
		df = entry2df(i)
		flag = true
		j = 1
		while flag
			aux = extrair_trecho(df.Prev_Residue, j)
			if !isnothing(aux) && (last(aux) - first(aux) ≤ hbound) && (last(aux) - first(aux) ≥ lbound)
				push!(resp, (i, aux))
				j += 1
			elseif !isnothing(aux) && last(aux) ≠ lastindex(df.Prev_Residue)
				j += 1
			else isnothing(aux) || last(aux) ≠ lastindex(df.Prev_Residue)
				flag = false
			end
		end
	end
	return(resp)
end


function verificando_binario(df::DataFrame)
	vetor = Bool[]
	for i ∈ 1:length(df.Coordinate)
		if i ≤ 3
			push!(vetor, false)
		else
			aux = point_plane_distance(collect(df.Coordinate[i]), [collect(df.Coordinate[i-1]), collect(df.Coordinate[i-2]), collect(df.Coordinate[i-3])]) > 0.0 ? true : false
			push!(vetor, aux)
		end
	end
	return(vetor)
end


function verificando_binario(X::Vector{Vector{Float64}})
	vetor = Bool[]
	for i ∈ 1:length(X)
		if i ≤ 3
			push!(vetor, false)
		else
			aux = point_plane_distance(X[i], [X[i-1], X[i-2], X[i-3]]) > 0.0 ? true : false
			push!(vetor, aux)
		end
	end
	return(vetor)
end

# `zs` é uma NTuple de booleanos que, se na i-ésima posição tiver `true`, significa que o i-ésimo termo de `ys` é um índice de `xs` que devemos dar flip
function flipwith(xs::NTuple{N, UInt8}, ys::NTuple{N, Int64}, zs::NTuple{N, Bool}, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
		y = @alloc(UInt8, N)
		copyto!(y, xs)
		for i ∈ eachindex(zs)
			if getfield(zs, i)
				y[getfield(ys, i)] = getfield(xs, getfield(ys, i)) == 0x00 ? 0x01 : 0x00
			end
		end
		ntuple(k -> getindex(y, k), Val(N))
	end
end


function flipat(xs::NTuple{N, Bool}, i::Int64, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
		y = @alloc(UInt8, N)
		copyto!(y, xs)
		y[i] = getfield(xs, i) == 0x00 ? 0x01 : 0x00
		ntuple(k -> getindex(y, k), Val(N))
	end
end


function switchat(xs::NTuple{N, Bool}, i::Int64, j::Int64, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
		y = @alloc(UInt8, N)
		copyto!(y, xs)
		y[i] = getfield(xs, i) == 0x00 ? 0x01 : 0x00
		y[j] = getfield(xs, j) == 0x00 ? 0x01 : 0x00
		ntuple(k -> getindex(y, k), Val(N))
	end
end

#==
function flipwithat(xs::NTuple{N, UInt8}, ys::NTuple{N, UInt8}, i::Int64, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
		y = @alloc(UInt8, N)
		copyto!(y, xs)
		y[i] = getfield(ys, i)
		ntuple(k -> getindex(y, k), Val(N))
	end
end
==#

function dᵢ(D::Matrix{Float64})
  n = size(D,1)
	d = zeros(Float64, n)
	for i ∈ 2:n
		d[i] = D[i-1, i]
	end
	return(d)
end


function dᵢ(D::SMatrix{N, N, Float64}) where {N}
	d = zeros(Float64, N)
	for i ∈ 2:N
		d[i] = D[i-1, i]
	end
	return(d)
end


### Função para gerar os quaterniões de translação tᵢ no BP Quaternião
function tᵢ(d::NTuple{N, Float64}, i::Int64) where {N}
	if i == 1
		return(Quaternion{Float64}(0.0, 0.0, 0.0, 0.0))
	else
		return(Quaternion{Float64}(0.0, d[i-1], 0.0, 0.0))
	end
end


function tᵢ(d::AbstractVector{Float64}, i::Int64)
	return(Quaternion{Float64}(0.0, d[i], 0.0, 0.0))
end

### Função para calcular os θᵢ a partir da matriz de distâncias
function θᵢ(M::Matrix{Float64})
	θ = zeros(Float64, size(M,1))
	for i ∈ 3:size(M,1)
		d₍ᵢ₋₂₎₍ᵢ₋₁₎ = M[i-2,i-1]
		d₍ᵢ₋₂₎₍ᵢ₎ = M[i-2,i]
		d₍ᵢ₋₁₎₍ᵢ₎ = M[i-1,i]
		θ[i] = acos(
			#### Numerador
			(d₍ᵢ₋₂₎₍ᵢ₋₁₎^2 + d₍ᵢ₋₁₎₍ᵢ₎^2 - d₍ᵢ₋₂₎₍ᵢ₎^2) /
			#### Denominador
			(2 * d₍ᵢ₋₂₎₍ᵢ₋₁₎ * d₍ᵢ₋₁₎₍ᵢ₎)
		)
	end
	return(θ)
end



function ωᵢ(M::Matrix{Float64})
	ω = zeros(Float64, size(M,1))
	for i ∈ 4:size(M,1)
		d₍ᵢ₋₃₎₍ᵢ₎² = M[i-3, i]^2
		d₍ᵢ₋₃₎₍ᵢ₋₁₎² = M[i-3, i-1]^2
		d₍ᵢ₋₃₎₍ᵢ₋₂₎² = M[i-3, i-2]^2
		d₍ᵢ₋₂₎₍ᵢ₋₁₎² = M[i-2, i-1]^2
		d₍ᵢ₋₂₎₍ᵢ₎² = M[i-2, i]^2
		d₍ᵢ₋₁₎₍ᵢ₎² = M[i-1, i]^2
		d₍ᵢ₋₃₎₍ᵢ₋₂₎₍ᵢ₋₁₎ = d₍ᵢ₋₃₎₍ᵢ₋₂₎² + d₍ᵢ₋₂₎₍ᵢ₋₁₎² - d₍ᵢ₋₃₎₍ᵢ₋₁₎²
		d₍ᵢ₋₂₎₍ᵢ₋₁₎₍ᵢ₎ = d₍ᵢ₋₂₎₍ᵢ₋₁₎² + d₍ᵢ₋₂₎₍ᵢ₎² - d₍ᵢ₋₁₎₍ᵢ₎²
		### Fórmula de ωᵢ₋₃,ᵢ
		ω[i] = acos(
			#### Numerador
			(2 * d₍ᵢ₋₂₎₍ᵢ₋₁₎² * (d₍ᵢ₋₃₎₍ᵢ₋₂₎² + d₍ᵢ₋₂₎₍ᵢ₎² - d₍ᵢ₋₃₎₍ᵢ₎²) - (d₍ᵢ₋₃₎₍ᵢ₋₂₎₍ᵢ₋₁₎ * d₍ᵢ₋₂₎₍ᵢ₋₁₎₍ᵢ₎)) /
			#### Denominador
			(√((4 * d₍ᵢ₋₃₎₍ᵢ₋₂₎² * d₍ᵢ₋₂₎₍ᵢ₋₁₎² - (d₍ᵢ₋₃₎₍ᵢ₋₂₎₍ᵢ₋₁₎)^2) * (4 * d₍ᵢ₋₂₎₍ᵢ₋₁₎² * d₍ᵢ₋₂₎₍ᵢ₎² - (d₍ᵢ₋₂₎₍ᵢ₋₁₎₍ᵢ₎)^2)))
		)
	end
	return(ω)
end


function qᵢ(τ::Tuple{Vector{Float64},Vector{Tuple{Float64,Float64}},Vector{Tuple{Float64,Float64}}}, i::Int64, λ::Bool=false)
  # Guia para τ:
  # τ[1][i] = d_i
  # τ[2][i][1] = sen(θ_i)
  # τ[2][i][2] = cos(θ_i)
  # τ[3][i][1] = sen(ω_i)
  # τ[3][i][2] = cos(ω_i)
  if i == 1
    return (quat(1.0, 0.0, 0.0, 0.0))
  elseif i == 2
    return (quat(0.0, 0.0, -1.0, 0.0))
  elseif i == 3
    return (quat(τ[2][i][1], 0.0, 0.0, τ[2][i][2]))
  elseif !λ
    return (quat(
      τ[2][i][1] * τ[3][i][2],
      τ[2][i][1] * τ[3][i][1],
      -τ[2][i][2] * τ[3][i][1],
      τ[2][i][2] * τ[3][i][2]
    ))
  else
    return (quat(
      τ[2][i][1] * τ[3][i][2],
      -τ[2][i][1] * τ[3][i][1],
      τ[2][i][2] * τ[3][i][1],
      τ[2][i][2] * τ[3][i][2]
    ))
  end
end


function qᵢ(θ::Tuple{Float64, Float64}, ω::Tuple{Float64, Float64}, i::Int64, λ::Bool = true)
	#### As tuplas de θ e ω contém os valores (sin(θ), cos(θ)) e (sin(ω), cos(ω)). Ou seja, θ[1] = sin(θ), ω[2] = cos(ω) e assim por diante.
	if i == 1
		return(quat(1.0, 0.0, 0.0, 0.0))
	elseif i == 2
		return(quat(0.0, 0.0, -1.0, 0.0))
	elseif i == 3
		return(quat(θ[1], 0.0, 0.0, θ[2]))
	elseif λ
		return(quat(
			θ[1]*ω[2],
			θ[1]*ω[1],
			-θ[2]*ω[1],
			θ[2]*ω[2]
		))
	else
		return(quat(
			θ[1]*ω[2],
			-θ[1]*ω[1],
			θ[2]*ω[1],
			θ[2]*ω[2]
		))
	end
end


function calcular_qᵢ(sinθ::Float64, cosθ::Float64, sinω::Float64, cosω::Float64, i::Int64, λ::Bool=false)
	if i == 1
	  return (Quaternion{Float64}(1.0, 0.0, 0.0, 0.0))
	elseif i == 2
	  return (Quaternion{Float64}(0.0, 0.0, -1.0, 0.0))
	elseif i == 3
	  return (Quaternion{Float64}(sinθ, 0.0, 0.0, cosθ))
	elseif !λ
	  return (Quaternion{Float64}(
		sinθ * cosω,
		sinθ * sinω,
		-cosθ * sinω,
		cosθ * cosω
	  ))
	else
	  return (Quaternion{Float64}(
		sinθ * cosω,
		-sinθ * sinω,
		cosθ * sinω,
		cosθ * cosω
	  ))
	end
end


function gerar_tupla_parcial(xs::NTuple{N, UInt8}, i::Int64, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
        y = @alloc(UInt8, N)
		copyto!(y, xs)
        y[(i+1):N] .= 0x07
        #NTuple{N, UInt8}(y)
		ntuple(k -> getindex(y, k), Val(N))
    end
end


function gerar_tupla_inicial(i::UInt8, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
        y = @alloc(UInt8, N)
		y .= 0x07
		y[1] = i
        ntuple(k -> y[k], Val(N))
    end
end


function calcular_coordenada(x::UInt8, ls::NTuple{M, Float64}, q::QuaternionF64, d::Float64, sinθ::Float64, cosθ::Float64, sinω::Float64, cosω::Float64, buffer::AllocBuffer{Vector{UInt8}}, i::Int64, ::Val{N}, ::Val{M}, ::Val{3}) where {N, M}
	#### Construindo xᵢ = qᵢ * (xᵢ₋₁ + tᵢ) * ̄qᵢ -> xᵢ₊₁ = qᵢ₊₁ * (xᵢ + tᵢ₊₁) *  ̄qᵢ₊₁ = qᵢ₊₁ * ([qᵢ * (xᵢ₋₁ + tᵢ) * ̄qᵢ] + tᵢ₊₁) *  ̄qᵢ₊₁
	#println("(CC) Valor do $(i)-ésimo termo de $(xs): Elemento $(xs[i])")
	q *= calcular_qᵢ(sinθ, cosθ, sinω, cosω, i, Bool(x))
	#println("(CC) Posição $(i) com $(lᵢ) e $(q)")
	@no_escape buffer begin
		y = @alloc(Float64, M)
		copyto!(y, ls)
		y[(3 * i - 2):(3 * i)] .= map(+, imag_part(q * Quaternion{Float64}(0.0, d, 0.0, 0.0) * conj(q)), ntuple(k -> getfield(ls, 3 * (i-1) - 3 + k), Val(3)))
		ntuple(k -> getindex(y, k), Val(M)), q
	end
end


@inline function verificar_podas(ls::NTuple{M, Float64}, D::SizedMatrix{N, N, Float64}, F::Vector{Int64}, i::Int64, ::Val{3})::Bool where {N, M}
    if isempty(F)
        return(false)
    else
        for j ∈ eachindex(F) # <-- Zerado de alocação
		    if abs(norm(map(-, ntuple(k -> getfield(ls, 3 * i - 3 + k), Val(3)), ntuple(k -> getfield(ls, 3 * getindex(F, j) - 3 + k), Val(3)))) - getindex(D, i, getindex(F, j))) > 1e-7 # <-- Zerado de alocação
			    return(true)
            end
	    end
    	return(false)
    end
end


function converter_caminho(xs::NTuple{N, UInt8}, candidate::AbstractArray{Int64, 1}, buffer::AllocBuffer{Vector{UInt8}}, ::Val{N}) where {N}
	@no_escape buffer begin
		y = @alloc(UInt8, N)
		copyto!(y, xs)
		y[(N - length(candidate) + 1):N] .= UInt8.(candidate .== 2)
		ntuple(x -> y[x], Val(N))
	end
end


function obter_coordenadas00(xs::NTuple{N, UInt8}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, i::Int64, visited::MVector{1, Int64}, visited2::MVector{1, Int128}, ::Val{N}, ::Val{M}, ::Val{3})::Int64 where {N, M}
	init = 1
	for k ∈ (i-1):-1:1
		if haskey(lru, (gerar_tupla_parcial(xs, k, buffer, Val(N)), k)) # <- Zerado de alocação
			init = k + 1
			break
		end
	end
	for idx ∈ init:i
		ys = gerar_tupla_parcial(xs, idx, buffer, Val(N)) # <-- Zerado de alocação
		ls, q = calcular_coordenada(getfield(ys, idx), getindex(lru, (gerar_tupla_parcial(ys, idx-1, buffer, Val(N)), idx-1))..., getfield(d, idx), getfield(sinθ, idx), getfield(cosθ, idx), getfield(sinω, idx), getfield(cosω, idx), buffer, idx, Val(N), Val(M), Val(3)) # <-- Zerado de alocação
		visited[1] += 1
		if isready(resposta)
			return(0)
		elseif verificar_podas(ls, D, getfield(F, idx), idx, Val(3))
			return(idx)
		elseif idx == N && !isready(resposta)
			put!(resposta, ls)
			#visited2[1] = contar_dfs(Bool.(ys))
			#visited2[1] = tempo_esperado_primeira_passagem(NTuple{N-3, Bool}(last(ys, N-3)))
		else
			setindex!(lru, (ls, q), (ys, idx))
		end
	end
	N+1
end


function obter_coordenadas00m(xs::NTuple{N, UInt8}, lru::LRU{Tuple{NTuple{N, UInt8},Int64}, Tuple{NTuple{M, Float64}, QuaternionF64}}, resposta::Channel{NTuple{M, Float64}}, D::SizedMatrix{N, N, Float64}, d::NTuple{N, Float64}, sinθ::NTuple{N, Float64}, cosθ::NTuple{N, Float64}, sinω::NTuple{N, Float64}, cosω::NTuple{N, Float64}, F::NTuple{N, Vector{Int64}}, buffer::AllocBuffer{Vector{UInt8}}, i::Int64, visited::MVector{1, Int64}, visited2::MVector{1, Int128}, visited3::MVector{1, Int128}, ::Val{N}, ::Val{M}, ::Val{3})::Int64 where {N, M}
	init = 1
	for k ∈ (i-1):-1:1
		if haskey(lru, (gerar_tupla_parcial(xs, k, buffer, Val(N)), k)) # <- Zerado de alocação
			init = k + 1
			break
		end
	end
	for idx ∈ init:i
		ys = gerar_tupla_parcial(xs, idx, buffer, Val(N)) # <-- Zerado de alocação
		ls, q = calcular_coordenada(getfield(ys, idx), getindex(lru, (gerar_tupla_parcial(ys, idx-1, buffer, Val(N)), idx-1))..., getfield(d, idx), getfield(sinθ, idx), getfield(cosθ, idx), getfield(sinω, idx), getfield(cosω, idx), buffer, idx, Val(N), Val(M), Val(3)) # <-- Zerado de alocação
		visited[1] += 1
		if isready(resposta)
			return(0)
		elseif verificar_podas(ls, D, getfield(F, idx), idx, Val(3))
			return(idx)
		elseif idx == N && !isready(resposta)
			put!(resposta, ls)
			visited2[1] = contar_dfs(Bool.(ys))
			visited3[1] = tempo_esperado_primeira_passagem(NTuple{N-3, Bool}(last(ys, N-3)))
		else
			setindex!(lru, (ls, q), (ys, idx))
		end
	end
	N+1
end