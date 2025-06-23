function extrair_dados_instancia2(entryid::Int64, idxrange::UnitRange{Int64})
	df = getindex(entry2df(entryid; coordinates = true), idxrange, :)
	RD, aas, controls = extrair_dados_entrada(entryid, idxrange);
	D = ajustar_matriz(deepcopy(RD))
	N = size(df, 1)
	SD = SizedMatrix{N, N}(D)
	TD = SizedMatrix{N, N}(RD)
	d = NTuple{N, Float64}(dᵢ(D))
	sinθ, cosθ = NTuple{N, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})))
	sinω, cosω = NTuple{N, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})))
	F = NTuple{N, Vector{Int64}}(particionar_arestas(D))
	TF = NTuple{N, Vector{Int64}}(particionar_arestas(RD))
	aas = SVector{N, UInt8}(UInt8.(aas))
	controls = SVector{N, UInt8}(UInt8.(controls))
	return(df, TD, SD, d, sinθ, cosθ, sinω, cosω, TF, F, aas, controls, N)
end

function point_plane_distance(point::Vector{Float64}, plane_points::Vector{Vector{Float64}})
  nᵥ = cross(plane_points[2] - plane_points[1], plane_points[3] - plane_points[1])
  norm_nᵥ = norm(nᵥ)
  if isapprox(norm_nᵥ, 0.0; atol=eps())
    println(norm_nᵥ)
    println("Vetor normal é nulo! Os pontos são colineares!")
    @warn("Vetor normal é nulo! Os pontos são colineares!")
    return (0.0)
  else
    return (dot(nᵥ, point - plane_points[1]) / norm_nᵥ)
  end
end


function point_plane_distance2(point::NTuple{3, Float64}, plane_points::NTuple{3, NTuple{3, Float64}})
  @no_escape begin
    xs = @chain begin
      @alloc(Float64, 3, 4)
      @aside _[:, 1] .= plane_points[1]
      @aside _[:, 2] .= plane_points[2]
      @aside _[:, 3] .= plane_points[3]
      @aside _[:, 4] .= point
      SMatrix{3,4, Float64}(_)
    end
  end
  dot(cross(xs[:, 2] - xs[:, 1], xs[:, 3] - xs[:, 1]), xs[:, 4] - xs[:, 1])
end


# NÃO OBEDECE A REGRA DO N: Usando i-1, i-2 e i-3 ao invés de i-1, i-2 e i-4 no caso de N. No caso de (HA, HA2, HA3), (H1, H2, H3) e (H, HD2, HD3), escolher o átomos mais próximo!
function extrair_pontos(i::Int64, current::Residue, previous::Union{Residue,Nothing})
  if i == 1
    # [!] ÁTOMO H
    # Colocando xᵢ₋₁ primeiro para poder fazer o teste da menor norma em xᵢ
    xᵢ₋₁ = coords(atoms(previous)["C"])
    xᵢ₋₃ = coords(atoms(previous)["CA"])
    xᵢ₋₄ = coords(atoms(current)["N"])
    if haskey(atoms(current), "H")
      xᵢ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
          if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₃)
            xᵢ = aux₁
          else
            xᵢ = aux₃
          end
        else
          if norm(xᵢ₋₄ - aux₂) ≤ norm(xᵢ₋₄ - aux₃)
            xᵢ = aux₂
          else
            xᵢ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    if haskey(atoms(previous), "HA")
      xᵢ₋₂ = coords(atoms(previous)["HA"])
    elseif haskey(atoms(previous), "HA2") && haskey(atoms(previous), "HA3")
      aux₁ = coords(atoms(previous)["HA2"])
      aux₂ = coords(atoms(previous)["HA3"])
      # (!) HA^{i-1} comparado com CA^{i-1}
      if norm(xᵢ₋₃ - aux₁) ≤ norm(xᵢ₋₃ - aux₂)
        xᵢ₋₂ = aux₁
      else
        xᵢ₋₂ = aux₂
      end
    elseif haskey(atoms(previous), "HA2") && !haskey(atoms(previous), "HA3")
      xᵢ₋₂ = coords(atoms(previous)["HA2"])
    elseif !haskey(atoms(previous), "HA2") && haskey(atoms(previous), "HA3")
      xᵢ₋₂ = coords(atoms(previous)["HA3"])
    else
      println(atoms(previous))
    end
    return (xᵢ, [xᵢ₋₁, xᵢ₋₂, xᵢ₋₃])
  elseif i == 2
    # [!] ÁTOMO N
    xᵢ = coords(atoms(current)["N"])
    xᵢ₋₂ = coords(atoms(previous)["C"])
    xᵢ₋₄ = coords(atoms(previous)["CA"])
    if haskey(atoms(current), "H")
      xᵢ₋₁ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ₋₁ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ₋₁ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ₋₁ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₂)
          xᵢ₋₁ = aux₁
        else
          xᵢ₋₁ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₂)
          xᵢ₋₁ = aux₁
        else
          xᵢ₋₁ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₂)
          xᵢ₋₁ = aux₁
        else
          xᵢ₋₁ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₂)
          if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₃)
            xᵢ₋₁ = aux₁
          else
            xᵢ₋₁ = aux₃
          end
        else
          if norm(xᵢ - aux₂) ≤ norm(xᵢ - aux₃)
            xᵢ₋₁ = aux₂
          else
            xᵢ₋₁ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ - aux₁) ≤ norm(xᵢ - aux₂)
          xᵢ₋₁ = aux₁
        else
          xᵢ₋₁ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ₋₁ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ₋₁ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    if haskey(atoms(previous), "HA")
      xᵢ₋₃ = coords(atoms(previous)["HA"])
    elseif haskey(atoms(previous), "HA2") && haskey(atoms(previous), "HA3")
      aux₁ = coords(atoms(previous)["HA2"])
      aux₂ = coords(atoms(previous)["HA3"])
      # (!) HA^{i} comparado com CA^{i}
      if norm(xᵢ₋₄ - aux₁) ≤ norm(xᵢ₋₄ - aux₂)
        xᵢ₋₃ = aux₁
      else
        xᵢ₋₃ = aux₂
      end
    elseif haskey(atoms(previous), "HA2") && !haskey(atoms(previous), "HA3")
      xᵢ₋₃ = coords(atoms(previous)["HA2"])
    elseif !haskey(atoms(previous), "HA2") && haskey(atoms(previous), "HA3")
      xᵢ₋₃ = coords(atoms(previous)["HA3"])
    else
      println(atoms(previous))
    end
    return (xᵢ, [xᵢ₋₁, xᵢ₋₂, xᵢ₋₃])
  elseif i == 3
    # [!] ÁTOMO CA
    xᵢ = coords(atoms(current)["CA"])
    xᵢ₋₁ = coords(atoms(current)["N"])
    xᵢ₋₃ = coords(atoms(previous)["C"])
    if haskey(atoms(current), "H")
      xᵢ₋₂ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ₋₂ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ₋₂ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ₋₂ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ₋₂ = aux₁
        else
          xᵢ₋₂ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ₋₂ = aux₁
        else
          xᵢ₋₂ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ₋₂ = aux₁
        else
          xᵢ₋₂ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ₋₂ = aux₁
          else
            xᵢ₋₂ = aux₃
          end
        else
          if norm(xᵢ₋₁ - aux₂) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ₋₂ = aux₂
          else
            xᵢ₋₂ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ₋₂ = aux₁
        else
          xᵢ₋₂ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ₋₂ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ₋₂ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    return (xᵢ, [xᵢ₋₁, xᵢ₋₂, xᵢ₋₃])
  elseif i == 4
    # [!] ÁTOMO HA
    # Colocando xᵢ₋₁ primeiro para poder fazer o teste da menor norma em xᵢ
    xᵢ₋₁ = coords(atoms(current)["CA"])
    xᵢ₋₂ = coords(atoms(current)["N"])
    if haskey(atoms(current), "HA")
      xᵢ = coords(atoms(current)["HA"])
    elseif haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      aux₁ = coords(atoms(current)["HA2"])
      aux₂ = coords(atoms(current)["HA3"])
      # (!) HA^{i} comparado com CA^{i}
      if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
        xᵢ = aux₁
      else
        xᵢ = aux₂
      end
    elseif haskey(atoms(current), "HA2") && !haskey(atoms(current), "HA3")
      xᵢ = coords(atoms(current)["HA2"])
    elseif !haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      xᵢ = coords(atoms(current)["HA3"])
    else
      println(atoms(current))
    end
    if haskey(atoms(current), "H")
      xᵢ₋₃ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ₋₃ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ₋₃ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ₋₃ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
          xᵢ₋₃ = aux₁
        else
          xᵢ₋₃ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
          xᵢ₋₃ = aux₁
        else
          xᵢ₋₃ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
          xᵢ₋₃ = aux₁
        else
          xᵢ₋₃ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
          if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₃)
            xᵢ₋₃ = aux₁
          else
            xᵢ₋₃ = aux₃
          end
        else
          if norm(xᵢ₋₂ - aux₂) ≤ norm(xᵢ₋₂ - aux₃)
            xᵢ₋₃ = aux₂
          else
            xᵢ₋₃ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
          xᵢ₋₃ = aux₁
        else
          xᵢ₋₃ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ₋₃ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ₋₃ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    return (xᵢ, [xᵢ₋₁, xᵢ₋₂, xᵢ₋₃])
  elseif i == 5
    # [!] ÁTOMO C
    xᵢ = coords(atoms(current)["C"])
    xᵢ₋₂ = coords(atoms(current)["CA"])
    xᵢ₋₃ = coords(atoms(current)["N"])
    if haskey(atoms(current), "HA")
      xᵢ₋₁ = coords(atoms(current)["HA"])
    elseif haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      aux₁ = coords(atoms(current)["HA2"])
      aux₂ = coords(atoms(current)["HA3"])
      # (!) HA^{i} comparado com CA^{i}
      if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
        xᵢ₋₁ = aux₁
      else
        xᵢ₋₁ = aux₂
      end
    elseif haskey(atoms(current), "HA2") && !haskey(atoms(current), "HA3")
      xᵢ₋₁ = coords(atoms(current)["HA2"])
    elseif !haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      xᵢ₋₁ = coords(atoms(current)["HA3"])
    else
      println(atoms(current))
    end
    return (xᵢ, [xᵢ₋₁, xᵢ₋₂, xᵢ₋₃])
  end
end


function extrair_coordenadas(current::Residue, previous::Union{Residue,Nothing})
  if isnothing(previous)
    # Colocando xᵢ₋₁ primeiro para poder fazer o teste da menor norma em xᵢ
    xᵢ₋₁ = coords(atoms(current)["N"])
    xᵢ₋₂ = coords(atoms(current)["CA"])
    xᵢ₋₄ = coords(atoms(current)["C"])
    if haskey(atoms(current), "H")
      xᵢ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ = aux₁
          else
            xᵢ = aux₃
          end
        else
          if norm(xᵢ₋₁ - aux₂) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ = aux₂
          else
            xᵢ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    if haskey(atoms(current), "HA")
      xᵢ₋₃ = coords(atoms(current)["HA"])
    elseif haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      aux₁ = coords(atoms(current)["HA2"])
      aux₂ = coords(atoms(current)["HA3"])
      # (!) HA^{i} comparado com CA^{i}
      if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
        xᵢ₋₃ = aux₁
      else
        xᵢ₋₃ = aux₂
      end
    elseif haskey(atoms(current), "HA2") && !haskey(atoms(current), "HA3")
      xᵢ₋₃ = coords(atoms(current)["HA2"])
    elseif !haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      xᵢ₋₃ = coords(atoms(current)["HA3"])
    else
      println(atoms(current))
    end
    return (xᵢ, xᵢ₋₁, xᵢ₋₂, xᵢ₋₃, xᵢ₋₄)
  else
    # [!] Quando há resíduo anterior
    xᵢ₋₁ = coords(atoms(current)["N"])
    xᵢ₋₂ = coords(atoms(current)["CA"])
    xᵢ₋₄ = coords(atoms(current)["C"])
    if haskey(atoms(current), "H")
      xᵢ = coords(atoms(current)["H"])
    else
      if haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H1"])
      elseif haskey(atoms(current), "H2") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H3")
        xᵢ = coords(atoms(current)["H2"])
      elseif haskey(atoms(current), "H3") && !haskey(atoms(current), "H1") && !haskey(atoms(current), "H2")
        xᵢ = coords(atoms(current)["H3"])
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && !haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && !haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif !haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H2"])
        aux₂ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "H1") && haskey(atoms(current), "H2") && haskey(atoms(current), "H3")
        aux₁ = coords(atoms(current)["H1"])
        aux₂ = coords(atoms(current)["H2"])
        aux₃ = coords(atoms(current)["H3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ = aux₁
          else
            xᵢ = aux₃
          end
        else
          if norm(xᵢ₋₁ - aux₂) ≤ norm(xᵢ₋₁ - aux₃)
            xᵢ = aux₂
          else
            xᵢ = aux₃
          end
        end
      elseif haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        aux₁ = coords(atoms(current)["HD2"])
        aux₂ = coords(atoms(current)["HD3"])
        # (!) H^{i} comparado com N^{i}
        if norm(xᵢ₋₁ - aux₁) ≤ norm(xᵢ₋₁ - aux₂)
          xᵢ = aux₁
        else
          xᵢ = aux₂
        end
      elseif haskey(atoms(current), "HD2") && !haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD2"])
      elseif !haskey(atoms(current), "HD2") && haskey(atoms(current), "HD3")
        xᵢ = coords(atoms(current)["HD3"])
      else
        println(atoms(current))
      end
    end
    if haskey(atoms(current), "HA")
      xᵢ₋₃ = coords(atoms(current)["HA"])
    elseif haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      aux₁ = coords(atoms(current)["HA2"])
      aux₂ = coords(atoms(current)["HA3"])
      # (!) HA^{i} comparado com CA^{i}
      if norm(xᵢ₋₂ - aux₁) ≤ norm(xᵢ₋₂ - aux₂)
        xᵢ₋₃ = aux₁
      else
        xᵢ₋₃ = aux₂
      end
    elseif haskey(atoms(current), "HA2") && !haskey(atoms(current), "HA3")
      xᵢ₋₃ = coords(atoms(current)["HA2"])
    elseif !haskey(atoms(current), "HA2") && haskey(atoms(current), "HA3")
      xᵢ₋₃ = coords(atoms(current)["HA3"])
    else
      println(atoms(current))
    end
    return (xᵢ, xᵢ₋₁, xᵢ₋₂, xᵢ₋₃, xᵢ₋₄)
  end
end

function atom_binary(current::Residue, previous::Union{Residue,Nothing}; coordinates::Bool=false)
  if coordinates
    vectorbinary = Bool[]
    vectorcoordinates = Vector{Float64}[]
    if isnothing(previous)
      append!(
        vectorbinary,
        [
          false,
          false,
          false,
          point_plane_distance(extrair_pontos(4, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(5, current, previous)...) > 0 ? true : false
        ]
      )
      append!(vectorcoordinates, extrair_coordenadas(current, previous))
      return (vectorbinary, vectorcoordinates)
    else
      append!(
        vectorbinary,
        [
          point_plane_distance(extrair_pontos(1, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(2, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(3, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(4, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(5, current, previous)...) > 0 ? true : false
        ]
      )
      append!(vectorcoordinates, extrair_coordenadas(current, previous))
      return (vectorbinary, vectorcoordinates)
    end
  else
    vectorbinary = Bool[]
    if isnothing(previous)
      append!(
        vectorbinary,
        [
          false,
          false,
          false,
          point_plane_distance(extrair_pontos(4, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(5, current, previous)...) > 0 ? true : false
        ]
      )
      return (vectorbinary)
    else
      append!(
        vectorbinary,
        [
          point_plane_distance(extrair_pontos(1, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(2, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(3, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(4, current, previous)...) > 0 ? true : false,
          point_plane_distance(extrair_pontos(5, current, previous)...) > 0 ? true : false
        ]
      )
      return (vectorbinary)
    end
  end
end

function validation_criterion(residue::Residue)
  aux = atoms(residue)
  haskey(aux, "CA") && haskey(aux, "C") && haskey(aux, "N") && (haskey(aux, "H") || haskey(aux, "H1") || haskey(aux, "H2") || haskey(aux, "H3") || haskey(aux, "HD2") || haskey(aux, "HD3")) && (haskey(aux, "HA") || haskey(aux, "HA2") || haskey(aux, "HA3"))
end


function entry2df(entry_id::Int64; coordinates::Bool=false)
  if coordinates
    df = DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[], Coordinate=NTuple{3,Float64}[])
    for model ∈ read(path_PDB * "/$(ids_PDB[entry_id]).mmtf", MMTFFormat)
      for chain ∈ model
        previous_residue = nothing
        for residue ∈ chain
          if !isnothing(previous_residue) && !ishetero(residue) && sequentialresidues(previous_residue, residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # N-ésimo elemento de um segmento
            aux_binary, aux_coordinates = atom_binary(residue, previous_residue; coordinates=coordinates)
            for id_binary ∈ eachindex(aux_binary)
              push!(df, [code2iupac(resname(previous_residue)), code2iupac(resname(residue)), aux_binary[id_binary], NTuple{3,Float64}(aux_coordinates[id_binary])])
            end
            previous_residue = residue
          elseif !isnothing(previous_residue) && (!sequentialresidues(previous_residue, residue) || ishetero(residue))
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
          elseif !ishetero(residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # Primeiro elemento de um segmento
            aux_binary, aux_coordinates = atom_binary(residue, previous_residue; coordinates=coordinates)
            for id_binary ∈ eachindex(aux_binary)
              push!(df, [nothing, code2iupac(resname(residue)), aux_binary[id_binary], NTuple{3,Float64}(aux_coordinates[id_binary])])
            end
            previous_residue = residue
          else
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
          end
        end
      end
    end
    return (df)
  else
    df = DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[])
    for model ∈ read(path_PDB * "/$(ids_PDB[entry_id]).mmtf", MMTFFormat)
      for chain ∈ model
        previous_residue = nothing
        for residue ∈ chain
          if !isnothing(previous_residue) && !ishetero(residue) && sequentialresidues(previous_residue, residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # N-ésimo elemento de um segmento
            aux_binary = atom_binary(residue, previous_residue)
            for id_binary ∈ eachindex(aux_binary)
              push!(df, [code2iupac(resname(previous_residue)), code2iupac(resname(residue)), aux_binary[id_binary]])
            end
            previous_residue = residue
          elseif !isnothing(previous_residue) && (!sequentialresidues(previous_residue, residue) || ishetero(residue))
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
          elseif !ishetero(residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # Primeiro elemento de um segmento
            aux_binary = atom_binary(residue, previous_residue)
            for id_binary ∈ eachindex(aux_binary)
              push!(df, [nothing, code2iupac(resname(residue)), aux_binary[id_binary]])
            end
            previous_residue = residue
          else
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
          end
        end
      end
    end
    return (df)
  end
end


function entry2df(pid::String; coordinates::Bool=false)
  if coordinates
      df = DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[], Coordinate=NTuple{3,Float64}[])
      models = read(path_PDB * "/$(pid).cif", MMCIFFormat)
      for model ∈ models
          for chain ∈ model
              previous_residue = nothing
              for residue ∈ chain
                  if !isnothing(previous_residue) && !ishetero(residue) && sequentialresidues(previous_residue, residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
                      # N-ésimo elemento de um segmento
                      aux_binary, aux_coordinates = atom_binary(residue, previous_residue; coordinates=coordinates)
                      for id_binary ∈ eachindex(aux_binary)
                          push!(df, [code2iupac(resname(previous_residue)), code2iupac(resname(residue)), aux_binary[id_binary], NTuple{3,Float64}(aux_coordinates[id_binary])])
                      end
                      previous_residue = residue
                  elseif !isnothing(previous_residue) && (!sequentialresidues(previous_residue, residue) || ishetero(residue))
                      # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
                      previous_residue = nothing
                  elseif !ishetero(residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
                      # Primeiro elemento de um segmento
                      aux_binary, aux_coordinates = atom_binary(residue, previous_residue; coordinates=coordinates)
                      for id_binary ∈ eachindex(aux_binary)
                          push!(df, [nothing, code2iupac(resname(residue)), aux_binary[id_binary], NTuple{3,Float64}(aux_coordinates[id_binary])])
                      end
                      previous_residue = residue
                  else
                      # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
                      previous_residue = nothing
                  end
              end
          end
      end
      return (df)
  else
      df = DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[])
      for model ∈ read(path_PDB * "/$(pid).cif", MMCIFFormat)
          for chain ∈ model
              previous_residue = nothing
              for residue ∈ chain
                  if !isnothing(previous_residue) && !ishetero(residue) && sequentialresidues(previous_residue, residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
                  # N-ésimo elemento de um segmento
                      aux_binary = atom_binary(residue, previous_residue)
                      for id_binary ∈ eachindex(aux_binary)
                          push!(df, [code2iupac(resname(previous_residue)), code2iupac(resname(residue)), aux_binary[id_binary]])
                      end
                      previous_residue = residue
                  elseif !isnothing(previous_residue) && (!sequentialresidues(previous_residue, residue) || ishetero(residue))
                      # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
                      previous_residue = nothing
                  elseif !ishetero(residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
                      # Primeiro elemento de um segmento
                      aux_binary = atom_binary(residue, previous_residue)
                      for id_binary ∈ eachindex(aux_binary)
                          push!(df, [nothing, code2iupac(resname(residue)), aux_binary[id_binary]])
                      end
                      previous_residue = residue
                  else
                      # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
                      previous_residue = nothing
                  end
              end
          end
      end
      return (df)
  end
end



function save_PDB_dataframe(names::Vector{String}; save2archive::Bool=false, coordinates::Bool=false)
  if coordinates
    data = @chain begin
      [DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[], Coordinate=NTuple{3,Float64}[]) for i ∈ 1:Threads.nthreads()]
      @aside Threads.@threads for i ∈ eachindex(names)
        append!(_[Threads.threadid()], entry2df(i; coordinates=coordinates))
      end
      reduce(vcat, _)
    end
    if save2archive
      save_object("PDB_HMM_Data_Coordinates.jld2", data)
    end
    return (data)
  else
    data = @chain begin
      [DataFrame(Prev_Residue=Union{Nothing,Char}[], Residue=Char[], Binary=Bool[]) for i ∈ 1:Threads.nthreads()]
      @aside Threads.@threads for i ∈ eachindex(names)
        append!(_[Threads.threadid()], entry2df(i; coordinates=coordinates))
      end
      reduce(vcat, _)
    end
    if save2archive
      save_object("PDB_HMM_Data.jld2", data)
    end
    return (data)
  end
end

# Fazer uma versão para colocar diretamente em um dicionário `Dict{Tuple{Char, Char}, Dict{Tuple{Bool, Bool, Char}, Int64}}()` para ttʰᵢⱼ[(row.Prev_Residue, row.Atom)][(row.Prev_Binary, row.Binary, row.Residue)] += 1
function entry2dictionary(entry_id::Int64, counter::Vector{Int64})
  db = Dict(vec(collect(Iterators.product(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*'], ['H', 'N', 'Ċ', 'Ḣ', 'C']))) .=> [Dict(vec(collect(Iterators.product([0x00, 0x01, 0x07], [0x00, 0x01], ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*']))) .=> 0) for i ∈ 1:115])
  counter[1] += 1
  for model ∈ read(path_PDB * "/$(ids_PDB[entry_id]).mmtf", MMTFFormat)
    counter[2] += 1
    for chain ∈ model
      counter[3] += 1
      previous_residue = nothing
      let previous_binary
        previous_binary = nothing
        for residue ∈ chain
          if !isnothing(previous_residue) && !ishetero(residue) && sequentialresidues(previous_residue, residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # N-ésimo elemento de um segmento
            counter[4] += 1
            counter[5] += 5
            aux_binary = atom_binary(residue, previous_residue)
            for id_binary ∈ eachindex(aux_binary)
              db[(code2iupac(resname(previous_residue)), num2atom(id_binary))][(isnothing(previous_binary) ? 0x07 : UInt8(previous_binary), UInt8(aux_binary[id_binary]), code2iupac(resname(residue)))] += 1
              previous_binary = aux_binary[id_binary]
            end
            previous_residue = residue
          elseif !isnothing(previous_residue) && (!sequentialresidues(previous_residue, residue) || ishetero(residue))
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
            previous_binary = nothing
          elseif !ishetero(residue) && !(code2iupac(resname(residue)) ∈ 'X') && validation_criterion(residue)
            # Primeiro elemento de um segmento
            counter[4] += 1
            counter[5] += 5
            aux_binary = atom_binary(residue, previous_residue)
            for id_binary ∈ eachindex(aux_binary)
              db[('*', num2atom(id_binary))][(isnothing(previous_binary) ? 0x07 : UInt8(previous_binary), UInt8(aux_binary[id_binary]), code2iupac(resname(residue)))] += 1
              previous_binary = aux_binary[id_binary]
            end
            previous_residue = residue
          else
            # Reiniciando o segmento (hétero-resíduo ou não-contíguo)
            previous_residue = nothing
            previous_binary = nothing
          end
        end
      end
    end
  end
  return (db)
end


function merge_composite!(dict1::Dict{Tuple{Char,Char},Dict{Tuple{UInt8,UInt8,Char},Int64}}, dict2::Dict{Tuple{Char,Char},Dict{Tuple{UInt8,UInt8,Char},Int64}})
  for key ∈ keys(dict1)
    mergewith!(+, dict1[key], dict2[key])
  end
  return (dict1)
end


function save_PDB_count_data(names::Vector{String}; save2archive::Bool=false)
  counter = [[0, 0, 0, 0, 0] for i ∈ 1:Threads.nthreads()]
  db = @chain begin
    [Dict(vec(collect(Iterators.product(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*'], ['H', 'N', 'Ċ', 'Ḣ', 'C']))) .=> [Dict(vec(collect(Iterators.product([0x00, 0x01, 0x07], [0x00, 0x01], ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*']))) .=> 0) for i ∈ 1:115]) for i ∈ 1:Threads.nthreads()]
    @aside Threads.@threads for i ∈ eachindex(names)
      merge_composite!(_[Threads.threadid()], entry2dictionary(i, counter[Threads.threadid()]))
    end
    foldr(merge_composite!, _; init=Dict(vec(collect(Iterators.product(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*'], ['H', 'N', 'Ċ', 'Ḣ', 'C']))) .=> [Dict(vec(collect(Iterators.product([0x00, 0x01, 0x07], [0x00, 0x01], ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*']))) .=> 0) for i ∈ 1:115]))
  end
  if save2archive
    save_object("PDB_count_data.jld2", db)
    save_object("PDB_count_number.jld2", reduce(+, counter))
  end
  return (db, reduce(+, counter))
end


function code2iupac(name::SubString{String})
  dict = Dict(
    "ALA" => 'A',
    "CYS" => 'C',
    "ASP" => 'D',
    "GLU" => 'E',
    "PHE" => 'F',
    "GLY" => 'G',
    "HIS" => 'H',
    "ILE" => 'I',
    "LYS" => 'K',
    "LEU" => 'L',
    "MET" => 'M',
    "ASN" => 'N',
    "PYL" => 'O',
    "PRO" => 'P',
    "GLN" => 'Q',
    "ARG" => 'R',
    "SER" => 'S',
    "THR" => 'T',
    "SEC" => 'U',
    "VAL" => 'V',
    "TRP" => 'W',
    "TYR" => 'Y',
    # Aberrações que aconteceram ao processar o PDB
    "UUU" => 'F',
    "UUC" => 'F',
    "UUA" => 'L',
    "UUG" => 'L',
    "UCU" => 'S',
    "UCC" => 'S',
    "UCA" => 'S',
    "UCG" => 'S',
    "UAU" => 'Y',
    "UAC" => 'Y',
    "UAA" => 'X',
    "UAG" => 'X',
    "UGU" => 'C',
    "UGC" => 'C',
    "UGA" => 'X',
    "UGG" => 'W',
    "CUU" => 'L',
    "CUC" => 'L',
    "CUA" => 'L',
    "CUG" => 'L',
    "CCU" => 'P',
    "CCC" => 'P',
    "CCA" => 'P',
    "CCG" => 'P',
    "CAU" => 'H',
    "CAC" => 'H',
    "CAA" => 'Q',
    "CAG" => 'Q',
    "CGU" => 'R',
    "CGC" => 'R',
    "CGA" => 'R',
    "CGG" => 'R',
    "AUU" => 'I',
    "AUC" => 'I',
    "AUA" => 'I',
    "AUG" => 'M',
    "ACU" => 'T',
    "ACC" => 'T',
    "ACA" => 'T',
    "ACG" => 'T',
    "AAU" => 'N',
    "AAC" => 'N',
    "AAA" => 'K',
    "AAG" => 'K',
    "AGU" => 'S',
    "AGC" => 'S',
    "AGA" => 'R',
    "AGG" => 'R',
    "GUU" => 'V',
    "GUC" => 'V',
    "GUA" => 'V',
    "GUG" => 'V',
    "GCU" => 'A',
    "GCC" => 'A',
    "GCA" => 'A',
    "GCG" => 'A',
    "GAU" => 'D',
    "GAC" => 'D',
    "GAA" => 'E',
    "GAG" => 'E',
    "GGU" => 'G',
    "GGC" => 'G',
    "GGA" => 'G',
    "GGG" => 'G',
    # D-Aminoácidos
    "DAL" => 'A',
    "DSN" => 'S',
    "DCY" => 'C',
    "DPR" => 'P',
    "DVA" => 'V',
    "DTH" => 'T',
    "DLE" => 'L',
    "DIL" => 'I',
    "DSG" => 'N',
    "DAS" => 'D',
    "MED" => 'M',
    "DGN" => 'Q',
    "DGL" => 'E',
    "DLY" => 'K',
    "DHI" => 'H',
    "DPN" => 'F',
    "DAR" => 'R',
    "DTY" => 'Y',
    "DTR" => 'W'
  )
  return (get!(dict, uppercase(name), 'X'))
end


function code2iupac(name::Union{String,Nothing})
  dict = Dict(
    nothing => '*',
    "ALA" => 'A',
    "CYS" => 'C',
    "ASP" => 'D',
    "GLU" => 'E',
    "PHE" => 'F',
    "GLY" => 'G',
    "HIS" => 'H',
    "ILE" => 'I',
    "LYS" => 'K',
    "LEU" => 'L',
    "MET" => 'M',
    "ASN" => 'N',
    "PYL" => 'O',
    "PRO" => 'P',
    "GLN" => 'Q',
    "ARG" => 'R',
    "SER" => 'S',
    "THR" => 'T',
    "SEC" => 'U',
    "VAL" => 'V',
    "TRP" => 'W',
    "TYR" => 'Y',
    # Aberrações que aconteceram ao processar o PDB
    "UUU" => 'F',
    "UUC" => 'F',
    "UUA" => 'L',
    "UUG" => 'L',
    "UCU" => 'S',
    "UCC" => 'S',
    "UCA" => 'S',
    "UCG" => 'S',
    "UAU" => 'Y',
    "UAC" => 'Y',
    "UAA" => 'X',
    "UAG" => 'X',
    "UGU" => 'C',
    "UGC" => 'C',
    "UGA" => 'X',
    "UGG" => 'W',
    "CUU" => 'L',
    "CUC" => 'L',
    "CUA" => 'L',
    "CUG" => 'L',
    "CCU" => 'P',
    "CCC" => 'P',
    "CCA" => 'P',
    "CCG" => 'P',
    "CAU" => 'H',
    "CAC" => 'H',
    "CAA" => 'Q',
    "CAG" => 'Q',
    "CGU" => 'R',
    "CGC" => 'R',
    "CGA" => 'R',
    "CGG" => 'R',
    "AUU" => 'I',
    "AUC" => 'I',
    "AUA" => 'I',
    "AUG" => 'M',
    "ACU" => 'T',
    "ACC" => 'T',
    "ACA" => 'T',
    "ACG" => 'T',
    "AAU" => 'N',
    "AAC" => 'N',
    "AAA" => 'K',
    "AAG" => 'K',
    "AGU" => 'S',
    "AGC" => 'S',
    "AGA" => 'R',
    "AGG" => 'R',
    "GUU" => 'V',
    "GUC" => 'V',
    "GUA" => 'V',
    "GUG" => 'V',
    "GCU" => 'A',
    "GCC" => 'A',
    "GCA" => 'A',
    "GCG" => 'A',
    "GAU" => 'D',
    "GAC" => 'D',
    "GAA" => 'E',
    "GAG" => 'E',
    "GGU" => 'G',
    "GGC" => 'G',
    "GGA" => 'G',
    "GGG" => 'G',
    # D-Aminoácidos
    "DAL" => 'A',
    "DSN" => 'S',
    "DCY" => 'C',
    "DPR" => 'P',
    "DVA" => 'V',
    "DTH" => 'T',
    "DLE" => 'L',
    "DIL" => 'I',
    "DSG" => 'N',
    "DAS" => 'D',
    "MED" => 'M',
    "DGN" => 'Q',
    "DGL" => 'E',
    "DLY" => 'K',
    "DHI" => 'H',
    "DPN" => 'F',
    "DAR" => 'R',
    "DTY" => 'Y',
    "DTR" => 'W'
  )
  return (get!(dict, isnothing(name) ? nothing : uppercase(name), 'X'))
end


function atoms2code(name::String)
  dict = Dict(
    "H" => 'H',
    "H1" => 'H',
    "H2" => 'H',
    "H3" => 'H',
    "HD2" => 'H',
    "HD3" => 'H',
    "N" => 'N',
    "CA" => 'Ċ',
    "HA" => 'Ḣ',
    "HA2" => 'Ḣ',
    "HA3" => 'Ḣ',
    "C" => 'C'
  )
  return (get!(dict, name, 'X'))
end


function num2atom(num::Int64)
  dict = Dict(
    1 => 'H',
    2 => 'N',
    3 => 'Ċ',
    4 => 'Ḣ',
    5 => 'C'
  )
  return (get!(dict, num, 'X'))
end


function something2other(dict::Dict{Char, Int64}, item::Union{Char, Nothing})
	return(get!(dict, !isnothing(item) ? item : '*', 23))
end


function ajustar_matriz(H::Matrix{Float64}; prob::Float64=0.75)
	for i ∈ 1:size(H, 2)
		for j ∈ i:size(H, 2)
			if i == j || !(H[i,j] < 5.0) || (j > i + 3 && rand() < prob)
				H[i,j] = 0.0
				H[j,i] = 0.0
			end
		end
	end
	return(H)
end


function particionar_arestas(D::Matrix{Float64})
	F = zeros(Int64, size(D))
	for i ∈ 1:(size(D,1)-1)
		for j ∈ (i+1):size(D,1)
			if D[i,j] ≠ zero(Float64) && j - i > 3
				F[j,i] = 1
			end
		end
	end
	return(findall.(!iszero, eachrow(F)))
end


function obter_controles(aas::AbstractVector{Int64})
	dict = Dict(reduce(vcat, Iterators.product(1:23, 1:5)) .=> 1:115)
	control = Int64[]
	for aa ∈ aas
		for element ∈ 1:5
			push!(control, dict[(aa, element)])
		end
	end
	return(control)
end


function extrair_dados_entrada(entry_id::Int64, range::UnitRange{Int64})
  df = entry2df(entry_id; coordinates=true)[range, :]
  return (
    pairwise(Euclidean(), df.Coordinate),
    [something2other(res2ind, x) for x ∈ df.Residue],
    obter_controles([something2other(res2ind, x) for x ∈ (df.Prev_Residue[1:5:end])])
  )
end


function extrair_dados_entrada(entry_id::String, range::UnitRange{Int64})
  df = entry2df(entry_id; coordinates=true)[range, :]
  return (
    pairwise(Euclidean(), df.Coordinate),
    [something2other(res2ind, x) for x ∈ df.Residue],
    obter_controles([something2other(res2ind, x) for x ∈ (df.Prev_Residue[1:5:end])])
  )
end


function matrix_rowwise_division(M::Matrix)
	v = sum(M; dims = 2)
	Δ = [maximum(M[i,:]) for i ∈ 1:size(M, 1)]
	try
		float.((M .// Δ) .* (Δ .// v))
	catch
		@warn "Erro na matriz $(M)"
	end
	return(float.((M .// Δ) .* (Δ .// v)))
end


function get_transition_matrix(db::Dict{Tuple{Char, Char}, Dict{Tuple{Bool, Bool}, Int64}})
	T = [ones(Int64, 2, 2) for i ∈ 1:115]
	for (prv_res, cur_atom) ∈ keys(db)
		for ((prv_bool, cur_bool), num) ∈ db[(prv_res, cur_atom)]
			T[cont2ind[(prv_res, cur_atom)]][bool2ind[prv_bool], bool2ind[cur_bool]] += num
		end
	end
	return(matrix_rowwise_division.(T))
end


function get_transition_matrix(db::Dict{Tuple{Char, Char}, Dict{Tuple{UInt8, UInt8}, Int64}})
	T = [ones(Int64, 2, 2) for i ∈ 1:115]
	for (prv_res, cur_atom) ∈ keys(db)
		for ((prv_bool, cur_bool), num) ∈ db[(prv_res, cur_atom)]
			if prv_bool ≠ 0x07
				T[cont2ind[(prv_res, cur_atom)]][uint2ind[prv_bool], uint2ind[cur_bool]] += num
      end
		end
	end
	return(matrix_rowwise_division.(T))
end


function get_initial_distribution(db::Dict{Tuple{Char, Char}, Dict{Tuple{UInt8, UInt8}, Int64}})
	δ = [ones(Int64, 2) for i ∈ 1:115]
	for (prv_res, cur_atom) ∈ keys(db)
		for ((prv_bool, cur_bool), num) ∈ db[(prv_res, cur_atom)]
			δ[cont2ind[(prv_res, cur_atom)]][uint2ind[cur_bool]] += num
		end
	end
	return(δ ./ sum.(δ))
end


function get_observation_matrix(db::Dict{Tuple{Char, Char}, Dict{Tuple{Bool, Char}, Int64}})
	E = [ones(Int64, 115, 23) for i ∈ 1:2]
	for (prv_res, cur_atom) ∈ keys(db)
		for ((cur_bool, cur_res), num) ∈ db[(prv_res, cur_atom)]
			E[bool2ind[cur_bool]][cont2ind[(prv_res, cur_atom)], res2ind[cur_res]] += num
		end
	end
	return(matrix_rowwise_division.(E))
end


function get_observation_matrix(db::Dict{Tuple{Char, Char}, Dict{Tuple{UInt8, Char}, Int64}})
	E = [ones(Int64, 115, 23) for i ∈ 1:2]
	for (prv_res, cur_atom) ∈ keys(db)
		for ((cur_bool, cur_res), num) ∈ db[(prv_res, cur_atom)]
			E[uint2ind[cur_bool]][cont2ind[(prv_res, cur_atom)], res2ind[cur_res]] += num
		end
	end
	return(matrix_rowwise_division.(E))
end


function verificar_binario_inicial(coords::AbstractVector{Vector{Float64}}, D::AbstractMatrix{Float64}, F::AbstractVector{Vector{Int64}}, n::Int64 = 5)
	vetor = UInt8[]
	for i ∈ 1:length(coords)
		if i ≤ 3
			push!(vetor, 0x00)
		else
			if !isempty(F[i])
				for j ∈ F[i]
					if abs(norm(coords[i] - coords[j]) - D[i,j]) > 1e-3
						return(nothing)
					end
				end
			end
			aux = point_plane_distance(coords[i], [coords[i-1], coords[i-2], coords[i-3]]) ≥ 0 ? 0x01 : 0x00
			push!(vetor, aux)
		end
	end
	return(length(vetor) == n ? vetor : UInt8[])
end

