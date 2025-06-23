module ARMBP

# Dependencies
using Base.ScopedValues
using LinearAlgebra
using StaticArrays
using StrideArrays
using Random
using Quaternions
using MKL
using ThreadPinning
using Bumper
using LRUCache
using Memoization
using Accessors
using DataStructures

import Dagger

# Auxiliaries
using BioStructures
using BioSequences
using MMTF
using Chain
using DataFrames
using DelimitedFiles
using Distances
using Distributions
using JLD2
using Statistics
using StatsBase

# HiddenMarkovModels
using ArgCheck: @argcheck
using Base: RefValue
using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity, logdensityof
using DocStringExtensions
using FillArrays: Fill
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, nonzeros, nnz, nzrange, rowvals
using StatsAPI: StatsAPI, fit, fit!
using StatsFuns: log2π

# HiddenMarkovModels.jl `include`
include("types/abstract_hmm.jl")

include("utils/linalg.jl")
include("utils/valid.jl")
include("utils/probvec_transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")
include("utils/lightcategorical.jl")
include("utils/limits.jl")

include("inference/predict.jl")
include("inference/forward.jl")
include("inference/viterbi.jl")
include("inference/list_viterbi.jl")
include("inference/forward_backward.jl")
include("inference/baum_welch.jl")
include("inference/logdensity.jl")
include("inference/chainrules.jl")

include("types/hmm.jl")

# Básico
include("armbp/armbp.jl")

include("search/busca_local.jl")
include("search/busca_global.jl")

include("utils/utils.jl")

include("utils/processamento.jl")

include("utils/contagem.jl")


# Constantes
const path_PDB::String = "/PDB"
const ids_PDB::Vector{String} = vec(readdlm("2024-03-05_rcsb_pdb_prot_nmr.txt", ',', String))
const dict_PDB::Dict{Int64,String} = Dict(i => ids_PDB[i] for i ∈ 1:length(ids_PDB))
const selected_atomnames::Vector{Char} = ['H', 'Ḥ', 'N', 'Ċ', 'Ḣ', 'Ḧ', 'C'] #Anteriormente: ["C", "CA", "N", "H", "H1", "HA", "HA1"]
const atomos::Vector{Char} = ['H', 'N', 'Ċ', 'Ḣ', 'C']
const aminos::Vector{Char} = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '*']
const aminoXatomos::Vector{Tuple{Char,Char}} = vcat(collect(Iterators.product(aminos, atomos))...)
const ftXft::Vector{Tuple{Bool,Bool}} = vcat(collect(Iterators.product([false, true], [false, true]))...)
const ftXres::Vector{Tuple{Bool,Char}} = vcat(collect(Iterators.product([false, true], aminos))...)
const ind2atom = Dict(1:5 .=> atomos)
const ind2bool = Dict(1:2 .=> [false, true])
const ind2res = Dict(1:23 .=> aminos)
const ind2cont = Dict(1:115 .=> aminoXatomos)
const bool2ind = Dict([false, true] .=> 1:2)
const uint2ind = Dict([0x00, 0x01] .=> 1:2)
const atom2ind = Dict(atomos .=> 1:5)
const res2ind = Dict(aminos .=> 1:23)
const cont2ind = Dict(aminoXatomos .=> 1:115)


# `struct` com os modelos, o primeiro com distribuição inicial aleatórias, os demais com distribuição inicial fixa
struct ARMBPModel
  hmm::ARHMM
  hmm₁₀::ARHMM
  hmm₀₁::ARHMM
end



# Debugging Packages
using Infiltrator
using Cthulhu
using ProfileView
using JET
using BenchmarkTools

# Exports
export path_PDB, ids_PDB, dict_PDB, selected_atomnames, atomos, aminos, aminoXatomos, ftXft, ftXres, ind2atom, ind2bool, ind2res, ind2cont, bool2ind, atom2ind, res2ind, cont2ind, ARMBPModel
# busca_global.jl
export buscar_globalmente00
# busca_local.jl
export buscar_localmente00, buscar_localmente000, buscar_localmente000m
# armbp.jl
export iniciar_armbp_local, iniciar_armbp_localm, iniciar_armbp_total
# utils.jl
export contraimagem, find_nth, extrair_trecho, extrair_segmento, verificando_binario, flipwith, flipat, switchat, dᵢ, tᵢ, θᵢ, ωᵢ, qᵢ, calcular_qᵢ, gerar_tupla_parcial, gerar_tupla_inicial, calcular_coordenada, verificar_podas, converter_caminho, obter_coordenadas00, obter_coordenadas00m
# processamento.jl
export extrair_dados_instancia2, point_plane_distance, point_plane_distance2, extrair_pontos, extrair_coordenadas, atom_binary, validation_criterion, entry2df, save_PDB_dataframe, entry2dictionary, merge_composite!, save_PDB_count_data, code2iupac, atoms2code, num2atom, something2other, ajustar_matriz, particionar_arestas, obter_controles, extrair_dados_entrada, matrix_rowwise_division, get_transition_matrix, get_transition_matrix, get_initial_distribution, get_observation_matrix, verificar_binario_inicial
# contagem.jl
export contar_dfs, tempo_esperado_primeira_passagem, tempo_esperado_primeira_passagem0

# HiddenMarkovModels.jl exports
export AbstractHMM, HMM, ARHMM, AbstractVectorOrNTuple
export initialization, transition_matrix, obs_distributions
export fit!, logdensityof, joint_logdensityof
export viterbi, forward, forward_backward, baum_welch, list_viterbi, list_viterbi2
export seq_limits


end
