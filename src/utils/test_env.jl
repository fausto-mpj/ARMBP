const entradas_disponiveis = load("/data/entradas_disponiveis.jld2")["entradas_disponiveis"];
const entradas_disponiveis2 = load("/data/entradas_disponiveis2.jld2")["entradas_disponiveis2"];
const entradas_disponiveis3 = load("/data/entradas_disponiveis3.jld2")["entradas_disponiveis3"];
const entradas_disponiveis4 = load("/data/entradas_disponiveis4.jld2")["entradas_disponiveis4"];
const count_data = load_object("/data/PDB_count_data.jld2");


T = Dict(vec(collect(Iterators.product(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*'], ['H', 'N', 'Ċ', 'Ḣ', 'C']))) .=> [Dict(vec(collect(Iterators.product([0x00, 0x01, 0x07], [0x00, 0x01]))) .=> 0) for i ∈ 1:115]);
for key₀ ∈ keys(count_data)
  for key₁ ∈ keys(count_data[key₀])
    T[key₀][Tuple(first(key₁, 2))] += values(count_data[key₀][key₁]);
  end
end
δ = get_initial_distribution(T);
T = get_transition_matrix(T);
E = Dict(vec(collect(Iterators.product(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*'], ['H', 'N', 'Ċ', 'Ḣ', 'C']))) .=> [Dict(vec(collect(Iterators.product([0x00, 0x01], ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*']))) .=> 0) for i ∈ 1:115]);
for key₀ ∈ keys(count_data)
  for key₁ ∈ keys(count_data[key₀])
    E[key₀][Tuple(last(key₁, 2))] += values(count_data[key₀][key₁]);
  end
end
E = get_observation_matrix(E);
thmm = ARHMM(δ, T, E);
thmm₁₀ = ARHMM(repeat([[1.0, 0.0]], 115), thmm.trans, thmm.dists);
thmm₀₁ = ARHMM(repeat([[0.0, 1.0]], 115), thmm.trans, thmm.dists);
hmm = ARMBPModel(thmm, thmm₁₀, thmm₀₁);




# #=
testebuffer = AllocBuffer()
entry_id = findfirst(==("1FW5"), ids_PDB);
df8 = entry2df(entry_id);
RD8, aas8, controls8 = extrair_dados_entrada(entry_id, 1:size(df8, 1));
D8 = ajustar_matriz(deepcopy(RD8));
tamanho8 = size(df8, 1);
#SD8 = SMatrix{tamanho8, tamanho8, Float64}(D8);
SD8 = SizedMatrix{tamanho8,tamanho8}(D8);
d8 = SVector{tamanho8, Float64}(dᵢ(D8));
sinθ8, cosθ8 = SVector{tamanho8, Float64}.(map(x -> getfield.(sincos.(θᵢ(D8) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
sinω8, cosω8 = SVector{tamanho8, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D8) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
F8 = NTuple{tamanho8, Vector{Int64}}(particionar_arestas(D8));
td8 = NTuple{tamanho8, Float64}(dᵢ(D8));
tsinθ8, tcosθ8 = NTuple{tamanho8, Float64}.(map(x -> getfield.(sincos.(θᵢ(D8) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tsinω8, tcosω8 = NTuple{tamanho8, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D8) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tF8 = particionar_arestas(D8)
Ns8 = size(df8, 1)
taas8 = SVector{tamanho8, UInt8}(UInt8.(aas8));
tcontrols8 = SVector{tamanho8, UInt8}(UInt8.(controls8));
#=
data8 = Instance3(
	SD8,
	d8,
	sinθ8,
	cosθ8,
	sinω8,
	cosω8,
	F8,
	SVector{tamanho8, UInt8}(UInt8.(aas8)),
	SVector{tamanho8, UInt8}(UInt8.(controls8))
);
params8 = Parameters3(
	SVector{tamanho8, UInt8}(gerar_tupla_inicial3(data8, testebuffer)),
	SVector{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho8)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
altparams8 = Parameters4(
	NTuple{tamanho8, UInt8}(gerar_tupla_inicial3(data8, testebuffer)),
	NTuple{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho8)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
alt2params8 = Parameters5(
	NTuple{tamanho8, UInt8}(gerar_tupla_inicial3(data8, testebuffer)),
	ntuple(i -> 0.0, 3 * tamanho8),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
=#
dfresp8 = entry2df(entry_id; coordinates = true);
# =#




# #=
# Das `entradas_disponiveis`:
# Entrada (10069, 2611:2630) com problema
# Entrada (909, 4356:4375) com problema
# Entrada (12129, 2976:2995) com problema
# Entrada (8395, 7981:8000) com problema
# Entrada (11520, 781:800) com problema

# Carregando uma instância teste
testebuffer = AllocBuffer()
entrada_aleatoria = rand(entradas_disponiveis);
df = entry2df(entrada_aleatoria[1])[entrada_aleatoria[2], :];
RD, aas, controls = extrair_dados_entrada(entrada_aleatoria...);
D = ajustar_matriz(deepcopy(RD));
tamanho = size(df, 1);
SD = SizedMatrix{tamanho,tamanho}(D);
d = SVector{tamanho, Float64}(dᵢ(D));
sinθ, cosθ = SVector{tamanho, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
sinω, cosω = SVector{tamanho, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
F = NTuple{tamanho, Vector{Int64}}(particionar_arestas(D));
td = NTuple{tamanho, Float64}(dᵢ(D));
tsinθ, tcosθ = NTuple{tamanho, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tsinω, tcosω = NTuple{tamanho, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tF = particionar_arestas(D)
Ns = size(df, 1)
taas = SVector{tamanho, UInt8}(UInt8.(aas));
tcontrols = SVector{tamanho, UInt8}(UInt8.(controls));
testexs3 = NTuple{Ns, UInt8}(UInt8.(df.Binary))
testelru3 = LRU{Tuple{NTuple{Ns, UInt8},Int64}, Tuple{NTuple{3*Ns, Float64}, QuaternionF64}}(maxsize = 150000);
#=
data = Instance3(
	SD,
	d,
	sinθ,
	cosθ,
	sinω,
	cosω,
	F,
	SVector{tamanho, UInt8}(UInt8.(aas)),
	SVector{tamanho, UInt8}(UInt8.(controls))
);
params = Parameters3(
	SVector{tamanho, UInt8}(gerar_tupla_inicial3(data, testebuffer)),
	SVector{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
altparams = Parameters4(
	NTuple{tamanho, UInt8}(gerar_tupla_inicial3(data, testebuffer)),
	NTuple{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
alt2params = Parameters5(
	NTuple{tamanho, UInt8}(gerar_tupla_inicial3(data, testebuffer)),
	ntuple(i -> 0.0, 3 * tamanho),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
setindex!(testelru3, (alt2params3.ls₀, alt2params3.q₀), (gerar_tupla_inicial4(data3, 0x00, testebuffer, Val(Ns)), 1))
setindex!(testelru3, (alt2params3.ls₀, alt2params3.q₀), (gerar_tupla_inicial4(data3, 0x01, testebuffer, Val(Ns)), 1))
testels3r = alt2params3.ls₀
testels3 = params3.ls₀
=#
dfresp = entry2df(entrada_aleatoria[1]; coordinates = true)[entrada_aleatoria[2], :];
# =#





# #=
# Carregando uma instância teste

# `entrada_aleatoria3` com valores (8264, 401:600) tem tamanho 200
# `entrada_aleatoria3` com valores (1122, 2801:3000) tem tamanho 200
# `entrada_aleatoria3` com valores (2642, 26671:26880) tem tamanho 210
# `entrada_aleatoria3` com valores (12038, 3601:3840) tem tamanho 240
entrada_aleatoria3 = rand(entradas_disponiveis3);
df3 = entry2df(entrada_aleatoria3[1])[entrada_aleatoria3[2], :];
RD3, aas3, controls3 = extrair_dados_entrada(entrada_aleatoria3...);
D3 = ajustar_matriz(deepcopy(RD3));
tamanho3 = size(df3, 1);
SD3 = SizedMatrix{tamanho3,tamanho3}(D3);
d3 = SVector{tamanho3, Float64}(dᵢ(D3));
sinθ3, cosθ3 = SVector{tamanho3, Float64}.(map(x -> getfield.(sincos.(θᵢ(D3) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
sinω3, cosω3 = SVector{tamanho3, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D3) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
F3 = NTuple{tamanho3, Vector{Int64}}(particionar_arestas(D3));
td3 = NTuple{tamanho3, Float64}(dᵢ(D3));
tsinθ3, tcosθ3 = NTuple{tamanho3, Float64}.(map(x -> getfield.(sincos.(θᵢ(D3) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tsinω3, tcosω3 = NTuple{tamanho3, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D3) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tF3 = particionar_arestas(D3)
Ns3 = size(df3, 1)
taas3 = SVector{tamanho3, UInt8}(UInt8.(aas3));
tcontrols3 = SVector{tamanho3, UInt8}(UInt8.(controls3));
#=
data3 = Instance3(
	SD3,
	d3,
	sinθ3,
	cosθ3,
	sinω3,
	cosω3,
	F3,
	SVector{tamanho3, UInt8}(UInt8.(aas3)),
	SVector{tamanho3, UInt8}(UInt8.(controls3))
);
params3 = Parameters3(
	SVector{tamanho3, UInt8}(gerar_tupla_inicial3(data3, testebuffer)),
	SVector{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho3)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
altparams3 = Parameters4(
	NTuple{tamanho3, UInt8}(gerar_tupla_inicial3(data3, testebuffer)),
	NTuple{3, Float64}.(ntuple(i -> (0.0, 0.0, 0.0), tamanho3)),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
alt2params3 = Parameters5(
	NTuple{tamanho3, UInt8}(gerar_tupla_inicial3(data3, testebuffer)),
	ntuple(i -> 0.0, 3 * tamanho3),
	Quaternion{Float64}(1.0, 0.0, 0.0, 0.0),
);
=#
dfresp3 = entry2df(entrada_aleatoria3[1]; coordinates = true)[entrada_aleatoria3[2], :];
# =#