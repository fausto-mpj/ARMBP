#cd(pwd() * "/Benchmarks")

pwd()

#=
function teste_armbp_local(entrada::Int64, idxrange::UnitRange{Int64}, hmm::ARMBPModel, ids_PDB::Vector{String}, num_dict::Dict{Tuple{String, UnitRange{Int64}}, Int64}, fail_dict::Dict{Tuple{String, UnitRange{Int64}}, UInt8})
    println("Iniciando entrada $(entrada): $(ids_PDB[entrada]), com índices $(idxrange)")
    dft, SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, Nt = extrair_dados_instancia2(entrada, idxrange)
    println("...dados extraídos.")
    if isnothing(Nt) || Nt < 5
        println("$(ids_PDB[entrada]) com índices $(idxrange) degenerada com tamanho $(Nt). Abortando.")
        push!(fail_dict, (ids_PDB[entrada], idxrange) => 0x02)
        return(nothing)
    end
    bpresp, num_bl, num_br, num_bi = iniciar_armbp_local(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    println("...ARBMBP realizado.")
    num_total = sum([num_bl, num_br, num_bi])
    GC.gc(true)
    if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
        println("$(ids_PDB[entrada]) com índices $(idxrange) bem-sucedido!")
        push!(num_dict, (ids_PDB[entrada], idxrange) => num_total)
        teste = @benchmarkable iniciar_armbp_local($SDt, $dt, $sintt, $costt, $sinot, $cosot, $Ft, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
        resultado =  BenchmarkTools.run(teste, verbose = true)
        BenchmarkTools.save("$(ids_PDB[entrada])($(entrada),$(idxrange))[Local]($(Nt))[$(length(findall(≠(0.0), SDt)))].json", resultado)
    else
        println("$(ids_PDB[entrada]) com índices $(idxrange) FALHOU!")
        push!(fail_dict, (ids_PDB[entrada], idxrange) => 0x01)
    end
    GC.gc(true)
end



function teste_armbp_total(entrada::Int64, idxrange::UnitRange{Int64}, hmm::ARMBPModel, ids_PDB::Vector{String}, num_dict::Dict{Tuple{String, UnitRange{Int64}}, Int64}, fail_dict::Dict{Tuple{String, UnitRange{Int64}}, UInt8})
    println("Iniciando entrada $(entrada): $(ids_PDB[entrada]), com índices $(idxrange)")
    dft, SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, Nt = extrair_dados_instancia2(entrada, idxrange)
    println("...dados extraídos.")
    if isnothing(Nt) || Nt < 5
        println("$(ids_PDB[entrada]) com índices $(idxrange) degenerada com tamanho $(Nt). Abortando.")
        push!(fail_dict, (ids_PDB[entrada], idxrange) => 0x02)
        return(nothing)
    end
    bpresp, num_bl, num_br, num_bi = iniciar_armbp_total(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    println("...ARBMBP realizado.")
    num_total = sum([num_bl, num_br, num_bi])
    GC.gc(true)
    if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
        println("$(ids_PDB[entrada]) com índices $(idxrange) bem-sucedido!")
        push!(num_dict, (ids_PDB[entrada], idxrange) => num_total)
        teste = @benchmarkable iniciar_armbp_total($SDt, $dt, $sintt, $costt, $sinot, $cosot, $Ft, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
        resultado =  BenchmarkTools.run(teste, verbose = true)
        BenchmarkTools.save("$(ids_PDB[entrada])($(entrada),$(idxrange))[Total]($(Nt))[$(length(findall(≠(0.0), SDt)))].json", resultado)
    else
        println("$(ids_PDB[entrada]) com índices $(idxrange) FALHOU!")
        push!(fail_dict, (ids_PDB[entrada], idxrange) => 0x01)
    end
    GC.gc(true)
end
=#


function teste_armbp(entrada::Int64, idxrange::UnitRange{Int64}, hmm::ARMBPModel, ids_PDB::Vector{String}, num_dict_local::Dict{Tuple{String, UnitRange{Int64}}, Int64}, fail_dict_local::Dict{Tuple{String, UnitRange{Int64}}, UInt8}, num_dict_total::Dict{Tuple{String, UnitRange{Int64}}, Int64}, fail_dict_total::Dict{Tuple{String, UnitRange{Int64}}, UInt8}, num_dict_bp::Dict{Tuple{String, UnitRange{Int64}}, Int128}, num_dict_clocal::Dict{Tuple{String, UnitRange{Int64}}, Int64}, num_dict_etfp::Dict{Tuple{String, UnitRange{Int64}}, Int128})
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 480
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
    BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true
    BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
    println("Iniciando entrada $(entrada): $(ids_PDB[entrada]), com índices $(idxrange) e tamanho $(length(idxrange))")
    dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(entrada, idxrange)
    println("...dados extraídos.")
    if isnothing(Nt) || Nt < 5
        println("$(ids_PDB[entrada]) com índices $(idxrange) degenerada com tamanho $(Nt). Abortando.")
        push!(fail_dict_local, (ids_PDB[entrada], idxrange) => 0x02)
        #push!(fail_dict_total, (ids_PDB[entrada], idxrange) => 0x02)
        return(nothing)
    end
    GC.gc(true)
    println("...iniciando ARMBP local com matriz reduzida.")
    bpresp, num_bl, num_br, num_bi, num_bp, num_tepp = iniciar_armbp_localm(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    println("...iniciando ARMBP local com matriz completa.")
    GC.gc(true)
    bpresp2, num_bl2, num_br2, num_bi2, num_bp2, num_tepp2 = iniciar_armbp_localm(TDt, dt, sintt, costt, sinot, cosot, TF, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    println("...ARBMBP local realizado.")
    num_total = sum([num_bl, num_br, num_bi])
    num_total2 = sum([num_bl2, num_br2, num_bi2])
    GC.gc(true)
    if pairwise(Euclidean(), reshape(collect(bpresp2), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
        println("(Local) Entrada $(ids_PDB[entrada]), número $(entrada) com índices $(idxrange) SUCESSO!")
        push!(num_dict_local, (ids_PDB[entrada], idxrange) => num_total)
        push!(num_dict_clocal, (ids_PDB[entrada], idxrange) => num_total2)
        push!(num_dict_bp, (ids_PDB[entrada], idxrange) => num_bp)
        push!(num_dict_etfp, (ids_PDB[entrada], idxrange) => num_tepp)
        teste = @benchmarkable iniciar_armbp_local($SDt, $dt, $sintt, $costt, $sinot, $cosot, $Ft, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
        println("(Local) Estabelecido o benchmark com matriz reduzida. Rodando...")
        resultado =  BenchmarkTools.run(teste, verbose = true)
        println("(Local) Salvando o resultado...")
        BenchmarkTools.save(pwd() * "/Benchmarks/$(ids_PDB[entrada])($(entrada),$(idxrange))[Local]($(Nt))[$(length(findall(≠(0.0), collect(SDt))))].json", resultado)
        GC.gc(true)
        teste = @benchmarkable iniciar_armbp_local($TDt, $dt, $sintt, $costt, $sinot, $cosot, $TF, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
        println("(CLocal) Estabelecido o benchmark com matriz completa. Rodando...")
        resultado =  BenchmarkTools.run(teste, verbose = true)
        println("(CLocal) Salvando o resultado...")
        BenchmarkTools.save(pwd() * "/Benchmarks/$(ids_PDB[entrada])($(entrada),$(idxrange))[CLocal]($(Nt))[$(length(findall(≠(0.0), collect(TDt))))].json", resultado)
        GC.gc(true)
    else
        println("(Local) Entrada $(ids_PDB[entrada]), número $(entrada) com índices $(idxrange) FALHOU!")
        push!(fail_dict_local, (ids_PDB[entrada], idxrange) => 0x01)
    end
    #=
    println("...iniciando ARMBP total.")
    bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_total(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    println("...ARBMBP total realizado.")
    num_total = sum([num_bl, num_br, num_bi])
    GC.gc(true)
    if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
        println("(Total) Entrada $(ids_PDB[entrada]), número $(entrada) com índices $(idxrange) SUCESSO!")
        push!(num_dict_total, (ids_PDB[entrada], idxrange) => num_total)
        teste = @benchmarkable iniciar_armbp_total($SDt, $dt, $sintt, $costt, $sinot, $cosot, $Ft, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
        println("(Total) Estabelecido o benchmark. Rodando...")
        resultado =  BenchmarkTools.run(teste, verbose = true)
        println("(Total) Salvando o resultado...")
        BenchmarkTools.save(pwd() * "/Benchmarks/$(ids_PDB[entrada])($(entrada),$(idxrange))[Total]($(Nt))[$(length(findall(≠(0.0), collect(SDt))))].json", resultado)
    else
        println("(Total) Entrada $(ids_PDB[entrada]), número $(entrada) com índices $(idxrange) FALHOU!")
        push!(fail_dict_total, (ids_PDB[entrada], idxrange) => 0x01)
    end
    println("...entrada $(ids_PDB[entrada]), número $(entrada) com índices $(idxrange) FINALIZADA.\n\n")
    GC.gc(true)
    =#
end









num_dict_local = Dict{Tuple{String, UnitRange{Int64}}, Int64}()
num_dict_clocal = Dict{Tuple{String, UnitRange{Int64}}, Int64}()
fail_dict_local = Dict{Tuple{String, UnitRange{Int64}}, UInt8}() # 0x00: Erro de domínio no pré-processamento; 0x01: Falhou em achar as coordenadas; 0x02: Instância degenerada
num_dict_bp = Dict{Tuple{String, UnitRange{Int64}}, Int128}()
num_dict_etfp = Dict{Tuple{String, UnitRange{Int64}}, Int128}()
num_dict_global = Dict{Tuple{String, UnitRange{Int64}}, Int64}()
fail_dict_global = Dict{Tuple{String, UnitRange{Int64}}, UInt8}() # 0x00: Erro de domínio no pré-processamento; 0x01: Falhou em achar as coordenadas; 0x02: Instância degenerada

# #=
num_dict_clocal = load("num_dict_clocal.jld2")["num_dict_clocal"]
num_dict_local = load("num_dict_local.jld2")["num_dict_local"]
fail_dict_local = load("fail_dict_local.jld2")["fail_dict_local"]
num_dict_global = load("num_dict_global.jld2")["num_dict_global"]
fail_dict_global = load("fail_dict_global.jld2")["fail_dict_global"]
num_dict_bp = load("num_dict_bp.jld2")["num_dict_bp"]
num_dict_etfp = load("num_dict_etfp.jld2")["num_dict_etfp"]
benchdf = load("benchdf.jld2")["benchdf"]
# =#

# Deu ERRO: Entrada 23 com índices 9316:9330
# Consumo EXCESSIVO de memória: Entrada 23 com índices 6076:6090
# Consumo EXCESSIVO de memória: Entrada 23 com índices 9721:9735 (Não conseguiu terminar)


instancias_de_teste = collect(keys(num_dict_local))
push!(instancias_de_teste, ("1FW5", 1:100))

rand()

# Faltam 0 de tamanho 15, 0 de tamanho 20 até 75, 1 de tamanho 85 até 250



MIPS = 320440

IPS = MIPS * 1000000

# Instância alega 3393957174821980710 e fez em 0.009715191 segundos.

# Deu ERRO: 6SAP, Entrada 11344 com índices 8261:8280
# Deu ERRO: 1PDC, Entrada 2233 com índices 1:20
# Deu ERRO: 2E3F, com índices 646:860

TF
Ft

entrada, idxrange = rand(entradas_disponiveis3)
ids_PDB[entrada]
dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(entrada, idxrange);
GC.gc(true)
bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_local(TDt, dt, sintt, costt, sinot, cosot, TF, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
push!(instancias_de_teste, (ids_PDB[entrada], idxrange))
#bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_local(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))

jldsave("instancias_de_teste.jld2"; instancias_de_teste)

dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(findfirst(==("5JTP"), ids_PDB), 70361:70470);

#bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_local(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_localm(TDt, dt, sintt, costt, sinot, cosot, TF, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)

pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt))

pairwise(Euclidean(), dft.Coordinate)

fail_dict_local

num_dict_local
num_dict_global
num_dict_bp
num_dict_cbp
num_dict_etfp
num_dict_clocal


# Entrada 2W9O, número 9795 com índices 461:690 FALHOU!
# Entrada 1NPQ, número 2038 com índices 5266:5350 FALHOU!
# Entrada 6YHI, número 11538 com índices 1801:1950 FALHOU! <-- Depois foi!
# Entrada 2NPV, número 9366 com índices 71:105 FALHOU! <-- Depois foi!
# Entrada 1HY9, número 1175 com índices 1026:1230 FALHOU! <-- Depois foi!
# Entrada 2RR4, número 9645 com índices 5416:5430 FALHOU!
# Entrada 2MF6, número 8396 com índices 1:15 FALHOU! <-- Depois foi!
# Entrada 1F0D, número 782 com índices 301:400 FALHOU! <-- Depois foi!
# Entrada 6OFA, número 11208 com índices 5921:6080 FALHOU!
# Entrada 1QKL, número 2410 com índices 1621:1685 FALHOU! <-- Depois foi!
# Entrada 5JTP, número 10206 com índices 70361:70470 FALHOU!
# Entrada 2N4K, número 9078 com índices 1:215 FALHOU!

collect(keys(fail_dict_local))

instancias_de_teste

GC.gc(true)
# REFAZENDO AS CONTAS PARA 45 ELEMENTOS!
for (entrada, idxrange) ∈ instancias_de_teste
    println("Rodando $(entrada), com índices $(idxrange)...")
    teste_armbp(findfirst(==(entrada), ids_PDB), idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_clocal, num_dict_etfp)
    GC.gc(true)
end


fail_dict_local


GC.gc(true)
# REFAZENDO AS CONTAS PARA OS QUE DERAM ERRO!
for (entrada, idxrange) ∈ collect(keys(fail_dict_local))
    println("Rodando $(entrada), com índices $(idxrange)...")
    teste_armbp(findfirst(==(entrada), ids_PDB), idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_clocal, num_dict_etfp)
    GC.gc(true)
end

# Entrada 2W9O, número 9795 com índices 461:690 FALHOU! <-- Depois foi!
# Entrada 5JTP, número 10206 com índices 70361:70470 FALHOU!
# Entrada 1NPQ, número 2038 com índices 5266:5350 FALHOU! <-- Depois foi!
# Entrada 2N4K, número 9078 com índices 1:215 FALHOU! <-- Depois foi!
# Entrada 2RR4, número 9645 com índices 5416:5430 FALHOU! <-- Depois foi!
# Entrada 6OFA, número 11208 com índices 5921:6080 FALHOU! <-- Depois foi!

ainda_falhando = [("2W9O", 461:690), ("5JTP", 70361:70470), ("1NPQ", 5266:5350), ("2N4K", 1:215), ("2RR4", 5416:5430), ("6OFA", 5921:6080)]
GC.gc(true)
for (entrada, idxrange) ∈ ainda_falhando
    println("Rodando $(entrada), com índices $(idxrange)...")
    teste_armbp(findfirst(==(entrada), ids_PDB), idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_clocal, num_dict_etfp)
    GC.gc(true)
end


ainda_falhando = [("5JTP", 70361:70470)]
GC.gc(true)
for (entrada, idxrange) ∈ ainda_falhando
    println("Rodando $(entrada), com índices $(idxrange)...")
    teste_armbp(findfirst(==(entrada), ids_PDB), idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_clocal, num_dict_etfp)
    GC.gc(true)
end


jldsave("num_dict_local.jld2"; num_dict_local)
jldsave("num_dict_clocal.jld2"; num_dict_clocal)
jldsave("num_dict_bp.jld2"; num_dict_bp)
jldsave("num_dict_etfp.jld2"; num_dict_etfp)
jldsave("fail_dict_local.jld2"; fail_dict_local)

#jldsave("fail_dict_global.jld2"; fail_dict_global)
#jldsave("num_dict_global.jld2"; num_dict_global)

# #=
#for (entrada, idxrange) ∈ collect(keys(num_dict_bp))
for (entrada, idxrange) ∈ collect(keys(fail_dict_local))
    println("Rodando $(entrada), com índices $(idxrange)...")
    id = findfirst(==(entrada), ids_PDB)
    println("...obtendo dados de entrada")
    dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(id, idxrange)
    GC.gc(true)
    println("...iniciando ARMBP")
    #bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_local(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    bpresp, num_bl, num_br, num_bi, num_bp, num_tepp = iniciar_armbp_local(TDt, dt, sintt, costt, sinot, cosot, TF, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    GC.gc(true)
    println(bpresp)
    println("...testando resposta")
    if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)
        println("SUCESSO!")
    else
        println("FRACASSO!")
    end
    #println("...salvando no dicionário")
    println("...terminado.\n\n")
    #push!(num_dict_etfp, (entrada, idxrange) => num_bp)
end



jldsave("num_dict_etfp.jld2"; num_dict_etfp)
# =#


#=
for (entrada, idxrange) ∈ collect(keys(num_dict_bp))
    println("Rodando $(entrada), com índices $(idxrange)...")
    id = findfirst(==(entrada), ids_PDB)
    println("...obtendo dados de entrada")
    dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(id, idxrange)
    GC.gc(true)
    println("...iniciando ARMBP")
    bpresp, num_bl, num_br, num_bi, num_bp = iniciar_armbp_local(TDt, dt, sintt, costt, sinot, cosot, TF, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
    GC.gc(true)
    println("...salvando no dicionário")
    push!(num_dict_clocal, (entrada, idxrange) => sum([num_bl, num_br, num_bi]))
end
jldsave("num_dict_clocal.jld2"; num_dict_clocal)
=#


#=
# Criando lista de entradas entre lowerbound e upperbound átomos!
lowerbound = 250
upperbound = 10000
entradas_disponiveis4 = extrair_segmento(1, length(ids_PDB), lowerbound, upperbound)
jldsave("entradas_disponiveis4.jld2"; entradas_disponiveis4)
=#


# #=
num = 0
while num < 10
    entryid, idxrange = rand(entradas_disponiveis4)
    println("Entrada $(entryid) com índices $(idxrange)")
    teste_armbp(entryid, idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_cbp, num_dict_cbp, num_dict_etfp)
    num += 1
end
# =#

# Usando instâncias menores!

# #=
num = 0
while num < 20
    entryid, idxrange = rand(entradas_disponiveis)
    teste_armbp(entryid, idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_cbp, num_dict_etfp)
    num += 1
end
num = 0
while num < 20
    entryid, idxrange = rand(entradas_disponiveis2)
    teste_armbp(entryid, idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_cbp, num_dict_etfp)
    num += 1
end
num = 0
while num < 20
    entryid, idxrange = rand(entradas_disponiveis3)
    teste_armbp(entryid, idxrange, hmm, ids_PDB, num_dict_local, fail_dict_local, num_dict_global, fail_dict_global, num_dict_bp, num_dict_cbp, num_dict_etfp)
    num += 1
end
#jldsave("num_dict_local.jld2"; num_dict_local)
#jldsave("fail_dict_local.jld2"; fail_dict_local)
#jldsave("num_dict_global.jld2"; num_dict_global)
#jldsave("fail_dict_global.jld2"; fail_dict_global)
#jldsave("num_dict_bp.jld2"; num_dict_bp)
#jldsave("num_dict_cbp.jld2"; num_dict_cbp)
# =#



# Entrada 11878: 7OSC, com índices 1956:2070 <-- Erro ao calcular as coordenadas, indicando ocorrência de 0x07 em contexto de Bool. VERIFICAR!
# Entrada 2KUX, número 6813 com índices 2551:2700 <-- Erro ao calcular as coordenadas, indicando ocorrência de 0x07 em contexto de Bool. VERIFICAR!
# Entrada 9066 com índices 12806:13790 <-- Faltou memória! VERIFICAR!

#=
# Ou usar a junção das `entradas_disponiveis` produzidas até agora
entradas = vcat(entradas_disponiveis, entradas_disponiveis2, entradas_disponiveis3)
# PROBLEMA 1: Falha na Entrada 73 com índices 196:215
idxprob = findfirst(==((73, 196:215)), entradas)
# PROBLEMA 1: Falha na Entrada 94 com índices 531:545
idxprob = findfirst(==((94, 531:545)), entradas)
# PROBLEMA 3: Parei na Entrada 128 com índices 1171:1185
idxprob = findfirst(==((128, 1171:1185)), entradas)


SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, Nt = extrair_dados_instancia(entradas[idxprob]...)

@time iniciar_armbp_local(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))



for (entryid, idxrange) ∈ entradas[idxprob:end]
    println("Entrada $(entryid) com índices $(idxrange)")extrair_dados_entrada
    teste_armbp(entryid, idxrange, hmm, ids_PDB, num_dict, fail_dict)
end
=#

#@time iniciar_armbp_local(SD8, td8, tsinθ8, tcosθ8, tsinω8, tcosω8, F8, taas8, tcontrols8, hmm, Val(Ns8), Val(3 * Ns8), Val(3))

benchdf = DataFrame(ID = Int64[], Nome = String[], Inicial = Int64[], Final = Int64[], Modo = String[], Visitados = Int128[], BP = Int128[], EFTP = Int128[], Media = Float64[], Mediana = Float64[], Minimo = Float64[], Maximo = Float64[], Desvio = Float64[], Aloc = Int64[], Mem = Int64[], TGC = Float64[], Vertices = Int64[], Arestas = Int64[])
for teststring ∈ readdir(pwd() * "/Benchmarks")
    resultado = only(BenchmarkTools.load(pwd() * "/Benchmarks/" * teststring))
    entryid = match(r"^.{1,4}", teststring).match
    entrycode, idxrange = split(match(r"\d{1,},\d{1,}:\d{1,}", teststring).match, ',')
    inicial, final = split(idxrange, ':')
    mode = match(r"[a-zA-Z]{5,6}", teststring).match
    visited = 0
    if mode == "Local"
        visited = get(num_dict_local, (entryid, parse(Int64, inicial):parse(Int64, final)), 0)
    elseif mode == "CLocal"
        visited = get(num_dict_clocal, (entryid, parse(Int64, inicial):parse(Int64, final)), 0)
    else
        visited = get(num_dict_global, (entryid, parse(Int64, inicial):parse(Int64, final)), 0)
    end
    bp_visited = get(num_dict_bp, (entryid, parse(Int64, inicial):parse(Int64, final)), 0)
    eftp_visited = get(num_dict_etfp, (entryid, parse(Int64, inicial):parse(Int64, final)), 0)    
    n_nodes = match(r"(?<=\()\d{1,}(?=\))", teststring).match
    n_edges = match(r"(?<=\[)\d{1,}(?=\])", teststring).match
    push!(benchdf, (
        parse(Int64, entrycode),
        entryid,
        parse(Int64, inicial),
        parse(Int64, final),
        mode,
        visited,
        bp_visited,
        eftp_visited,
        mean(resultado).time,
        median(resultado).time,
        minimum(resultado).time,
        maximum(resultado).time,
        std(resultado).time,
        mean(resultado).allocs,
        mean(resultado).memory,
        mean(resultado).gctime,
        parse(Int64, n_nodes),
        parse(Int64, n_edges)
        )
    )
end
sort!(benchdf, [:Vertices, :Nome, :Inicial, :Modo, :Arestas])

benchdf
#jldsave("benchdf.jld2"; benchdf)




#=
mean(resultado).time
mean(resultado).allocs
mean(resultado).gctime
mean(resultado).memory


minimum(resultado)
maximum(resultado)
std(resultado)

resultado = only(BenchmarkTools.load("1C4E(346,3231:3400)[Local](170)[1574].json"))
resultado2 = only(BenchmarkTools.load("1C4E(346,3231:3400)[Total](170)[1574].json"))

judge(mean(resultado), mean(resultado2))
judge(median(resultado), median(resultado2))
judge(std(resultado), std(resultado2))
judge(minimum(resultado), minimum(resultado2))
judge(maximum(resultado), maximum(resultado2))

# O tempo no `dump` está em nanosegundos!
dump(resultado)visited4 = @MVector zeros(Int64, 1)
=#

pids = uppercase.(["1n6t", "1fw5", "1adx", "1bdo", "1all", "6s61", "4wua", "1fhl", "6czf", "5ijn", "6rn2", "1cza", "6bco", "1epw", "5np0", "5nug", "4rh7", "3vkh"])


println("$(pids[1])")


downloadpdb(pids[2])
visited4 = @MVector zeros(Int64, 1)
entry2df2(pids[2]; coordinates = true)

modelo1 = read(path_PDB * "/1BDO.pdb", PDBFormat)

modelo2 = read(path_PDB * "/1BDO.mmtf", MMTFFormat)

modelo3 = read(path_PDB * "/1BDO.cif", MMCIFFormat)visited4 = @MVector zeros(Int64, 1)

collectatoms(collectresidues(modelo2[1])[1])
collectatoms(collectresidues(modelo3[1])[1])


atoms(collectresidues(modelo1[1])[1])

atoms(collectresidues(modelo2[1])[4])
atoms(collectresidues(modelo3[1])[4])


strip.(keys(atoms(collectresidues(modelo1[1])[1])))



validation_criterion(collectresidues(modelo1[1])[1])

read(path_PDB * "/$(pids[1]).pdb", PDBFormat)

ids_PDB[1]


downloadpdb(pids[4], dir = path_PDB, format = MMCIFFormat)



entry2df(ids_PDB[1]; coordinates = true)

entry2df(1)

df1 = entry2df(pids[1]; coordinates = true)



findnext(isnothing, df1.Prev_Residue, 1)
findnext(isnothing, df1.Prev_Residue, 6)

entry2df("1BDO")

for pid ∈ pids
    downloadpdb(pid, dir = path_PDB, format = MMCIFFormat)
    df = entry2df(pid; coordinates = true)
    println("PID $(pid)...")
    if isempty(df)
        continue
    end
    inicial = findnext(isnothing, df.Prev_Residue, 1)
    final = isnothing(findnext(isnothing, df.Prev_Residue, inicial + 5)) ? size(df, 1) : findnext(isnothing, df.Prev_Residue, inicial + 5) - 1
    df = df[inicial:final, :]
    println("...com tamanho $(size(df, 1))")
    RD, aas, controls = extrair_dados_entrada(pid, inicial:final);
    D = ajustar_matriz(deepcopy(RD));
    Nt = size(df, 1);
    SD = SizedMatrix{Nt,Nt}(D);
    d = SVector{Nt, Float64}(dᵢ(D));
    sinθ, cosθ = SVector{Nt, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
    sinω, cosω = SVector{Nt, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
    F = NTuple{Nt, Vector{Int64}}(particionar_arestas(D));
    td = NTuple{Nt, Float64}(dᵢ(D));
    tsinθ, tcosθ = NTuple{Nt, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
    tsinω, tcosω = NTuple{Nt, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
    Ns = size(df, 1)
    taas = SVector{Nt, UInt8}(UInt8.(aas));
    tcontrols = SVector{Nt, UInt8}(UInt8.(controls));
    bpresp, num_bl, num_br, num_bi = iniciar_armbp_local(SD, td, tsinθ, tcosθ, tsinω, tcosω, F, taas, tcontrols, hmm, Val(Nt), Val(3 * Nt), Val(3))
    if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), df.Coordinate)
        println("$(pid) SUCESSO!\n\n")
    else
        println("$(pid) FRACASSO!\n\n")
    end
end





pid = "6S61"
downloadpdb(pid, dir = path_PDB, format = MMCIFFormat)
df = entry2df(pid; coordinates = true)


idx = UnitRange{Int64}[]
target = 1
while !isnothing(target)
    aux = findnext(isnothing, df.Prev_Residue, target)
    if !isnothing(aux)
        target = findnext(isnothing, df.Prev_Residue, aux + 5)
    else
        target = nothing
    end
    if !isnothing(target)
        push!(idx, aux:(target-1))
    end
end
idx



println("PID $(pid)...")
inicial = findnext(isnothing, df.Prev_Residue, 1)
final = findnext(isnothing, df.Prev_Residue, inicial + 5) - 1
df = df[inicial:final, :]
println("...com tamanho $(size(df, 1))")
RD, aas, controls = extrair_dados_entrada(pid, inicial:final);
D = ajustar_matriz(deepcopy(RD));
Nt = size(df, 1);
SD = SizedMatrix{Nt,Nt}(D);
d = SVector{Nt, Float64}(dᵢ(D));
sinθ, cosθ = SVector{Nt, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
sinω, cosω = SVector{Nt, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
F = NTuple{Nt, Vector{Int64}}(particionar_arestas(D));
td = NTuple{Nt, Float64}(dᵢ(D));
tsinθ, tcosθ = NTuple{Nt, Float64}.(map(x -> getfield.(sincos.(θᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
tsinω, tcosω = NTuple{Nt, Float64}.(map(x -> getfield.(sincos.(ωᵢ(D) ./ 2), x), fieldnames(Tuple{Float64, Float64})));
Ns = size(df, 1)
taas = SVector{Nt, UInt8}(UInt8.(aas));
tcontrols = SVector{Nt, UInt8}(UInt8.(controls));
bpresp, num_bl, num_br, num_bi = iniciar_armbp_local(SD, td, tsinθ, tcosθ, tsinω, tcosω, F, taas, tcontrols, hmm, Val(Nt), Val(3 * Nt), Val(3))
if pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), df.Coordinate)
    println("$(pid) SUCESSO!\n\n")
else
    println("$(pid) FRACASSO!\n\n")
end


############################

findlast(>(10^7), benchdf.Media)

benchdf[48:74, :]

benchdf[1:48, :]

benchdf

resultado = only(BenchmarkTools.load(pwd() * "/Benchmarks/" * "6N68(11160,1021:1080)[Local](60)[510].json"))

gctime(resultado)
allocs
memory
judge(mean(resultado), mean(resultado2))
judge(gctime(resultado), gctime(resultado2))
#judge(allocs(resultado), allocs(resultado2))
#judge(memory(resultado), memory(resultado2))

@show resultado

resultado2 = only(BenchmarkTools.load(pwd() * "/Benchmarks/" * "6N68(11160,1021:1080)[CLocal](60)[3540].json"))

judge(mean(resultado), mean(resultado2)).memory
judge(median(resultado), median(resultado2))
judge(std(resultado), std(resultado2))
judge(minimum(resultado), minimum(resultado2))
judge(maximum(resultado), maximum(resultado2))

for i ∈ 1:2:size(benchdf, 1)
    println(i)
end


for i ∈ 1:2:size(benchdf, 1)
    resultado = only(BenchmarkTools.load(pwd() * "/Benchmarks/" * "$(benchdf.Nome[i])($(benchdf.ID[i]),$(benchdf.Inicial[i]):$(benchdf.Final[i]))[$(benchdf.Modo[i])]($(benchdf.Vertices[i]))[$(benchdf.Arestas[i])].json"))
    resultado2 = only(BenchmarkTools.load(pwd() * "/Benchmarks/" * "$(benchdf.Nome[i+1])($(benchdf.ID[i+1]),$(benchdf.Inicial[i+1]):$(benchdf.Final[i+1]))[$(benchdf.Modo[i+1])]($(benchdf.Vertices[i+1]))[$(benchdf.Arestas[i+1])].json"))
    println("Analisando $(benchdf.Nome[i])...")
    println(benchdf.Vertices[i])
    println(judge(mean(resultado), mean(resultado2)))
    println(judge(median(resultado), median(resultado2)))
    println(judge(std(resultado), std(resultado2)))
    println(judge(minimum(resultado), minimum(resultado2)))
    println(judge(maximum(resultado), maximum(resultado2)))
    println(judge(mean(resultado), mean(resultado2)).memory)
    println(judge(gctime(resultado), gctime(resultado2)))
    println("...finalizado.\n\n")
end

benchdf


# O SBBU nas instâncias testadas até vértices 120 apresentaram tempo na escala de 10^{5} nanosegundos; acima de 120 e até 1000, apresentaram tempo na escala de 10^{6} nanosegundos.

# No ARMBP Local conseguimos manter o tempo na escala de 10^{5} nanosegundos até 60 vértices, incluindo algumas instâncias na escala 10^{4}. Acima de 60 e abaixo de 200, conseguimos manter o tempo na escala de 10^{6} nanosegundos, no entanto, acima de 200 vimos um aumento significativo na quantidade de memória e de alocações na heap, elevando o tempo de processamento para além de 10^{7} nanosegundos

# Para cada instância, fizemos 100 ensaios utilizando o ARMBP.

# Característica interessante: testando com a matriz de distâncias completas, em nenhum caso o ARMBP visitou mais vértices do que com o a matriz de distâncias incompletas. No pior caso, visitaram o mesmo número de vértices. 

# Razão calculada usando o valor do primeiro dividido pelo valor do segundo. Se (ratio - tolerance) > 1.0, então regressão. Se (ratio + tolerance) < 1.0, então melhora.

# Comparando CLocal com Local. Regressão significa que Local foi melhor!

#=
Analisando 2DN4...
15
TrialJudgement(+0.52% => invariant)
TrialJudgement(+0.78% => invariant)
TrialJudgement(-8.28% => improvement)
TrialJudgement(+0.84% => invariant)
TrialJudgement(+3.58% => invariant)
invariant
improvement
...finalizado.


Analisando 2EKF...
15
TrialJudgement(+3.56% => invariant)
TrialJudgement(+1.73% => invariant)
TrialJudgement(+16.03% => regression)
TrialJudgement(+1.56% => invariant)
TrialJudgement(+0.51% => invariant)
invariant
improvement
...finalizado.


Analisando 2JND...
15
TrialJudgement(+0.92% => invariant)
TrialJudgement(-0.16% => invariant)
TrialJudgement(+4.34% => invariant)
TrialJudgement(-0.06% => invariant)
TrialJudgement(+0.10% => invariant)
invariant
improvement
...finalizado.


Analisando 2L6K...
15
TrialJudgement(+1.23% => invariant)
TrialJudgement(+1.36% => invariant)
TrialJudgement(+4.25% => invariant)
TrialJudgement(+0.73% => invariant)
TrialJudgement(+0.30% => invariant)
invariant
improvement
...finalizado.


Analisando 2L6K...
15
TrialJudgement(-10.26% => improvement)
TrialJudgement(-11.56% => improvement)
TrialJudgement(-6.23% => improvement)
TrialJudgement(-12.22% => improvement)
TrialJudgement(-15.60% => improvement)
improvement
improvement
...finalizado.


Analisando 2L6N...
15
TrialJudgement(-0.11% => invariant)
TrialJudgement(+0.41% => invariant)
TrialJudgement(-8.98% => improvement)
TrialJudgement(+1.88% => invariant)
TrialJudgement(-5.10% => improvement)
invariant
improvement
...finalizado.


Analisando 2MF6...
15
TrialJudgement(-20.27% => improvement)
TrialJudgement(-18.12% => improvement)
TrialJudgement(-29.00% => improvement)
TrialJudgement(-18.83% => improvement)
TrialJudgement(-27.33% => improvement)
improvement
improvement
...finalizado.


Analisando 2QL0...
15
TrialJudgement(-2.00% => invariant)
TrialJudgement(+0.14% => invariant)
TrialJudgement(-20.99% => improvement)
TrialJudgement(-0.09% => invariant)
TrialJudgement(-1.09% => invariant)
invariant
improvement
...finalizado.


Analisando 2RLL...
15
TrialJudgement(-2.30% => invariant)
TrialJudgement(-0.34% => invariant)
TrialJudgement(-19.59% => improvement)
TrialJudgement(-0.38% => invariant)
TrialJudgement(-3.93% => invariant)
invariant
improvement
...finalizado.


Analisando 2RR4...
15
TrialJudgement(-1.33% => invariant)
TrialJudgement(-2.32% => invariant)
TrialJudgement(+10.41% => regression)
TrialJudgement(-5.41% => improvement)
TrialJudgement(+3.90% => invariant)
invariant
improvement
...finalizado.


Analisando 5M9Y...
15
TrialJudgement(+1.79% => invariant)
TrialJudgement(+0.32% => invariant)
TrialJudgement(+16.79% => regression)
TrialJudgement(+1.62% => invariant)
TrialJudgement(+16.83% => regression)
invariant
improvement
...finalizado.


Analisando 6ZOM...
15
TrialJudgement(-42.90% => improvement)
TrialJudgement(-43.71% => improvement)
TrialJudgement(-39.01% => improvement)
TrialJudgement(-44.42% => improvement)
TrialJudgement(-43.62% => improvement)
improvement
improvement
...finalizado.


Analisando 7JU9...
15
TrialJudgement(-9.14% => improvement)
TrialJudgement(-8.54% => improvement)
TrialJudgement(-3.50% => invariant)
TrialJudgement(-12.36% => improvement)
TrialJudgement(-15.84% => improvement)
invariant
improvement
...finalizado.


Analisando 7TV7...
15
TrialJudgement(-17.43% => improvement)
TrialJudgement(-15.09% => improvement)
TrialJudgement(-34.86% => improvement)
TrialJudgement(-15.83% => improvement)
TrialJudgement(-19.95% => improvement)
improvement
improvement
...finalizado.


Analisando 8DIJ...
15
TrialJudgement(+2.35% => invariant)
TrialJudgement(+0.91% => invariant)
TrialJudgement(+23.54% => regression)
TrialJudgement(+0.11% => invariant)
TrialJudgement(+9.54% => regression)
invariant
improvement
...finalizado.


Analisando 2JUY...
20
TrialJudgement(-6.21% => improvement)
TrialJudgement(-4.55% => invariant)
TrialJudgement(-10.14% => improvement)
TrialJudgement(-6.51% => improvement)
TrialJudgement(+2.46% => invariant)
invariant
improvement
...finalizado.


Analisando 2L6N...
20
TrialJudgement(-3.85% => invariant)
TrialJudgement(-1.96% => invariant)
TrialJudgement(-13.61% => improvement)
TrialJudgement(-0.66% => invariant)
TrialJudgement(-3.70% => invariant)
invariant
improvement
...finalizado.


Analisando 2N5W...
20
TrialJudgement(-5.04% => improvement)
TrialJudgement(-3.75% => invariant)
TrialJudgement(-20.40% => improvement)
TrialJudgement(-2.07% => invariant)
TrialJudgement(-3.56% => invariant)
invariant
improvement
...finalizado.


Analisando 2N5W...
20
TrialJudgement(+1.00% => invariant)
TrialJudgement(+2.56% => invariant)
TrialJudgement(+5.51% => regression)
TrialJudgement(-0.48% => invariant)
TrialJudgement(+1.04% => invariant)
invariant
improvement
...finalizado.


Analisando 2G46...
30
TrialJudgement(-9.55% => improvement)
TrialJudgement(-8.01% => improvement)
TrialJudgement(-24.07% => improvement)
TrialJudgement(-9.50% => improvement)
TrialJudgement(-18.58% => improvement)
invariant
improvement
...finalizado.


Analisando 2BBU...
35
TrialJudgement(+3.50% => invariant)
TrialJudgement(+3.29% => invariant)
TrialJudgement(+5.23% => regression)
TrialJudgement(+2.91% => invariant)
TrialJudgement(+4.62% => invariant)
invariant
improvement
...finalizado.


Analisando 2NPV...
35
TrialJudgement(-37.05% => improvement)
TrialJudgement(-36.97% => improvement)
TrialJudgement(-38.54% => improvement)
TrialJudgement(-37.91% => improvement)
TrialJudgement(-37.84% => improvement)
improvement
improvement
...finalizado.


Analisando 1JLP...
40
TrialJudgement(-12.06% => improvement)
TrialJudgement(-11.28% => improvement)
TrialJudgement(-31.38% => improvement)
TrialJudgement(-10.15% => improvement)
TrialJudgement(-18.58% => improvement)
improvement
improvement
...finalizado.


Analisando 2N08...
55
TrialJudgement(-21.14% => improvement)
TrialJudgement(-21.28% => improvement)
TrialJudgement(+2.14% => invariant)
TrialJudgement(-24.12% => improvement)
TrialJudgement(-21.60% => improvement)
invariant
improvement
...finalizado.


Analisando 2N9A...
55
TrialJudgement(-0.24% => invariant)
TrialJudgement(-0.02% => invariant)
TrialJudgement(+8.83% => regression)
TrialJudgement(-2.28% => invariant)
TrialJudgement(+1.06% => invariant)
invariant
improvement
...finalizado.


Analisando 6N68...
60
TrialJudgement(-18.20% => improvement)
TrialJudgement(-18.98% => improvement)
TrialJudgement(+2.87% => invariant)
TrialJudgement(-20.70% => improvement)
TrialJudgement(-13.42% => improvement)
invariant
improvement
...finalizado.


Analisando 7TVQ...
60
TrialJudgement(-4.69% => invariant)
TrialJudgement(-3.65% => invariant)
TrialJudgement(+2.98% => invariant)
TrialJudgement(-4.80% => invariant)
TrialJudgement(-1.79% => invariant)
invariant
improvement
...finalizado.


Analisando 8DGH...
60
TrialJudgement(-44.23% => improvement)
TrialJudgement(-43.97% => improvement)
TrialJudgement(-46.98% => improvement)
TrialJudgement(-44.54% => improvement)
TrialJudgement(-48.34% => improvement)
invariant
improvement
...finalizado.


Analisando 1QKL...
65
TrialJudgement(+4.19% => invariant)
TrialJudgement(+4.32% => invariant)
TrialJudgement(+3.59% => invariant)
TrialJudgement(+5.27% => regression)
TrialJudgement(+7.25% => regression)
invariant
improvement
...finalizado.


Analisando 6M19...
75
TrialJudgement(-47.61% => improvement)
TrialJudgement(-48.06% => improvement)
TrialJudgement(-62.33% => improvement)
TrialJudgement(-45.84% => improvement)
TrialJudgement(-56.12% => improvement)
improvement
improvement
...finalizado.


Analisando 1NPQ...
85
TrialJudgement(-8.75% => improvement)
TrialJudgement(-7.05% => improvement)
TrialJudgement(-9.57% => improvement)
TrialJudgement(-10.17% => improvement)
TrialJudgement(-8.87% => improvement)
invariant
improvement
...finalizado.


Analisando 1PAO...
85
TrialJudgement(-53.69% => improvement)
TrialJudgement(-54.20% => improvement)
TrialJudgement(-29.84% => improvement)
TrialJudgement(-54.41% => improvement)
TrialJudgement(-48.70% => improvement)
improvement
improvement
...finalizado.


Analisando 1F0D...
100
TrialJudgement(-0.53% => invariant)
TrialJudgement(-0.76% => invariant)
TrialJudgement(+25.79% => regression)
TrialJudgement(-2.48% => invariant)
TrialJudgement(+7.51% => regression)
invariant
improvement
...finalizado.


Analisando 1FW5...
100
TrialJudgement(-14.05% => improvement)
TrialJudgement(-14.24% => improvement)
TrialJudgement(+2.23% => invariant)
TrialJudgement(-14.19% => improvement)
TrialJudgement(-11.94% => improvement)
invariant
improvement
...finalizado.


Analisando 5ZYX...
100
TrialJudgement(+10.90% => regression)
TrialJudgement(+11.63% => regression)
TrialJudgement(+19.20% => regression)
TrialJudgement(+8.85% => regression)
TrialJudgement(+13.75% => regression)
invariant
improvement
...finalizado.


Analisando 5JTP...
110
TrialJudgement(-24.45% => improvement)
TrialJudgement(-24.88% => improvement)
TrialJudgement(+4.59% => invariant)
TrialJudgement(-24.53% => improvement)
TrialJudgement(-8.22% => improvement)
improvement
improvement
...finalizado.


Analisando 1WQC...
130
TrialJudgement(-99.61% => improvement)
TrialJudgement(-99.62% => improvement)
TrialJudgement(-98.57% => improvement)
TrialJudgement(-99.63% => improvement)
TrialJudgement(-99.57% => improvement)
improvement
improvement
...finalizado.


Analisando 2K6T...
130
TrialJudgement(-3.12% => invariant)
TrialJudgement(-2.93% => invariant)
TrialJudgement(-1.08% => invariant)
TrialJudgement(-2.08% => invariant)
TrialJudgement(-6.30% => improvement)
invariant
improvement
...finalizado.


Analisando 2AIY...
150
TrialJudgement(-13.36% => improvement)
TrialJudgement(-13.17% => improvement)
TrialJudgement(-17.19% => improvement)
TrialJudgement(-13.65% => improvement)
TrialJudgement(-17.26% => improvement)
improvement
improvement
...finalizado.


Analisando 6YHI...
150
TrialJudgement(-15.34% => improvement)
TrialJudgement(-14.99% => improvement)
TrialJudgement(-7.92% => improvement)
TrialJudgement(-15.83% => improvement)
TrialJudgement(-16.92% => improvement)
invariant
improvement
...finalizado.


Analisando 6OFA...
160
TrialJudgement(+2.77% => invariant)
TrialJudgement(+3.34% => invariant)
TrialJudgement(+14.40% => regression)
TrialJudgement(+1.96% => invariant)
TrialJudgement(+3.79% => invariant)
improvement
improvement
...finalizado.


Analisando 2DCO...
170
TrialJudgement(-5.08% => improvement)
TrialJudgement(-5.20% => improvement)
TrialJudgement(+2.49% => invariant)
TrialJudgement(-5.55% => improvement)
TrialJudgement(-4.60% => invariant)
invariant
improvement
...finalizado.


Analisando 1HY9...
205
TrialJudgement(-22.70% => improvement)
TrialJudgement(-22.74% => improvement)
TrialJudgement(+45.27% => regression)
TrialJudgement(-23.42% => improvement)
TrialJudgement(-0.63% => invariant)
invariant
invariant
...finalizado.


Analisando 2N4K...
215
TrialJudgement(-88.45% => improvement)
TrialJudgement(-88.43% => improvement)
TrialJudgement(-82.02% => improvement)
TrialJudgement(-88.65% => improvement)
TrialJudgement(-88.25% => improvement)
improvement
regression
...finalizado.


Analisando 2W9O...
230
TrialJudgement(-49.07% => improvement)
TrialJudgement(-49.30% => improvement)
TrialJudgement(-41.87% => improvement)
TrialJudgement(-49.19% => improvement)
TrialJudgement(-49.00% => improvement)
invariant
regression
...finalizado.
=#

# Na maioria dos casos o CLocal foi melhor que o Local no uso do GC. Isso indica que há algo no código Local que pode ser melhorado, porque não há motivo teórico para o GC ficar mais agressivo quando há menos arestas! Note que apesar do GC ter sido significativo, o uso de memória foi invariante entre CLocal e Local.

# Somente em um caso (60FA) o CLocal foi significativamente melhor do que o Local em tempo médio. Fora neste caso, vemos que o tempo médio costuma a ser bem próximo. Isso mesmo o Local em geral visitando mais vértices do que o CLocal.

benchdf

# Tomando 0.15 de tolerância
any((benchdf.Visitados ./ benchdf.EFTP) .+ 0.15 .> 1.0)

benchfil_local.Visitados ./ benchfil_clocal.Visitados

# Razão calculada usando o valor do primeiro dividido pelo valor do segundo. 
# Se (ratio - tolerance) > 1.0, então regressão. 
# Se (ratio + tolerance) < 1.0, então melhora.

# Comparando CLocal com Local. Regressão significa que Local foi melhor!


sum((benchfil_clocal.Visitados ./ benchfil_local.Visitados) .+ 0.15 .< 1.0)

sum((benchfil_clocal.Visitados ./ benchfil_local.Visitados) .≥ 1.0)

benchfil_clocal.Media ./ benchfil_local.Media

(benchfil_clocal.Media ./ benchfil_local.Media)

(benchfil_clocal.Media ./ benchfil_local.Media) .≥ 1.0

sum((benchfil_clocal.Media ./ benchfil_local.Media) .- 0.15 .> 1.0)

sum((benchfil_clocal.Media ./ benchfil_local.Media) .+ 0.15 .< 1.0)

sum((benchfil_clocal.Media ./ benchfil_local.Media) .≥ 1.0)

sum((benchfil_clocal.Media ./ benchfil_local.Media) .- 0.15 .> 1.0)

benchfil_clocal.Visitados ./ benchfil_local.Visitados

benchdf.Visitados ./ benchdf.EFTP

# Fazer boxplot!
mean(benchdf.Visitados ./ benchdf.EFTP)
std(benchdf.Visitados ./ benchdf.EFTP)
median(benchdf.Visitados ./ benchdf.EFTP)
maximum(benchdf.Visitados ./ benchdf.EFTP)
minimum(benchdf.Visitados ./ benchdf.EFTP)


benchdf = load("benchdf.jld2")["benchdf"]
benchfil = select(benchdf, [:Nome, :Modo, :Visitados, :EFTP, :Media, :Desvio, :Aloc, :Mem, :TGC, :Vertices, :Arestas, :Inicial, :Final, :BP, :Mediana, :Minimo, :Maximo])


benchfil.Media .= benchfil.Media .* 10^-9;
benchfil.Desvio .= benchfil.Desvio .* 10^-9;
benchfil.Mediana .= benchfil.Mediana .* 10^-9;
benchfil.Minimo .= benchfil.Minimo .* 10^-9;
benchfil.Maximo .= benchfil.Maximo .* 10^-9;
benchfil.TGC .= benchfil.TGC .* 10^-9;




@show benchfil


# O GC só foi ativado em 6OFA, 2DCO, 1HY9 e 2W9O, ou seja, nas instâncias acima de 150 vértices. No caso de 2DCO, não ativou o GC no caso de CLocal, mas sim no caso de Local. Não está claro o motivo do GC ter ficado mais agressivo nestas instâncias maiores.

show(benchdf; allrows = true)

show(benchfil; allrows = true)
# 


benchfil_local = filter(:Modo => x -> x == "Local", benchfil);

benchfil_clocal = filter(:Modo => x -> x == "CLocal", benchfil);


benchtable1 = select(benchfil_local, [:Nome, :Inicial, :Final, :Visitados, :BP, :EFTP, :Aloc, :Mem, :TGC, :Vertices, :Arestas]);

benchctable1 = select(benchfil_clocal, [:Nome, :Inicial, :Final, :Visitados, :BP, :EFTP, :Aloc, :Mem, :TGC, :Vertices, :Arestas]);


benchtable2 = select(benchfil_local, [:Nome, :Media, :Mediana, :Minimo, :Maximo, :Desvio, :Vertices, :Arestas]);

benchctable2 = select(benchfil_clocal, [:Nome, :Media, :Mediana, :Minimo, :Maximo, :Desvio, :Vertices, :Arestas]);

replace!(benchfil.Modo, "CLocal" => "Completa", "Local" => "Incompleta");


using Latexify

println(latexify(benchtable1; env = :tabular))

println(latexify(benchtable2; env = :tabular, fmt = x -> round(x, sigdigits=5)))

benchtable1
benchtable2

benchfil_local

visitados = filter(:Modo => x -> x == "Local", benchfil).Visitados

estimados = filter(:Modo => x -> x == "Local", benchfil).EFTP

estimados2 = filter(:Modo => x -> x == "Local", benchfil).EFTP ./ 2

mean(estimados - visitados)

median(estimados - visitados)

using HypothesisTests
testesinal = SignTest(visitados, estimados)

testesinal2 = SignTest(visitados, estimados2)

confint(SignTest(visitados, estimados); level = 0.99, tail = :both, method = :clopper_pearson)

confint(testesinal; level = 0.99, tail = :both)

confint(testesinal2; level = 0.99, tail = :both)

dft, TDt, SDt, dt, sintt, costt, sinot, cosot, TF, Ft, aast, controlst, Nt = extrair_dados_instancia2(findfirst(==("1FW5"), ids_PDB), 1:100);
bpresp, num_bl, num_br, num_bi, num_bp, num_tepp = @time iniciar_armbp_localm(SDt, dt, sintt, costt, sinot, cosot, Ft, aast, controlst, hmm, Val(Nt), Val(3 * Nt), Val(3))
GC.gc()
teste = @benchmarkable iniciar_armbp_local($SDt, $dt, $sintt, $costt, $sinot, $cosot, $Ft, $aast, $controlst, $hmm, Val($Nt), Val(3 * $Nt), Val(3))
resultado =  BenchmarkTools.run(teste, verbose = true)
pairwise(Euclidean(), reshape(collect(bpresp), 3, Nt)) ≈ pairwise(Euclidean(), dft.Coordinate)





nx, ny = length(x), length(y)
xbar = mean(x)-mean(y)
varx, vary = var(x), var(y)
stderr = sqrt(varx/nx + vary/ny)
t = (xbar-μ0)/stderr
df = (varx / nx + vary / ny)^2 / ((varx / nx)^2 / (nx - 1) + (vary / ny)^2 / (ny - 1))
UnequalVarianceTTest(nx, ny, xbar, df, stderr, t, μ0)

using Plots
using StatsPlots
benchfil
@df benchfil violin(:Modo, :Media, line = 0, fill = (0.2, :blue), ylims = (0.0, 0.005))

boxplot_compVincomp = @df benchfil boxplot(:Modo, :Media, line = (2, :black), fill = (0.5, :purple), ylims = (0.0, 0.005), legend = nothing, xlab = "Matriz", ylab = "Média do tempo de processamento", color = :black, dpi = 300)
savefig(boxplot_compVincomp, "boxplot_compVincomp.pdf")

scatterplot_compVincomp = @df benchfil scatter(:Vertices, :Media, group = :Modo, ylims = (0.0, 0.015), alpha = 0.6, dpi = 300, xlab = "Vértices", ylab = "Média do tempo de processamento", palette = palette(:tab10))
savefig(scatterplot_compVincomp, "scatterplot_compVincomp.pdf")


scatterplot_visVeftp = @df benchtable1 scatter(:Vertices, :Visitados, ylims = (0, 1500), alpha = 0.6, dpi = 300, xlab = "Vértices", ylab = "Vértices Visitados", palette = palette(:tab10), label = "Visitados")
@df benchtable1 scatter!(:Vertices, :EFTP, ylims = (0, 1500), alpha = 0.6, dpi = 300, xlab = "Vértices", ylab = "Vértices Visitados", palette = palette(:tab10), label = "MTPP")
@df benchtable1 scatter!(:Vertices, :BP, ylims = (0, 1500), alpha = 0.6, dpi = 300, xlab = "Vértices", ylab = "Vértices Visitados", palette = palette(:tab10), label = "EDFS")
savefig(scatterplot_visVeftp, "scatterplot_visVeftp.pdf")


histogram_vis = histogram(filter!(x -> x < 100.0, benchtable1.Visitados ./ benchctable1.Vertices), bins = :scott, legend = nothing, dpi = 300, color = :purple, alpha = 0.6, xlab = "Razão de Vis. por Vert.", ylab = "Frequência")
savefig(histogram_vis, "histogram_vis.pdf")


boxplot_time = @df benchtable2 boxplot(:Media, line = (2, :black), fill = (0.5, :purple), ylims = (0.0, 0.005), legend = nothing, xlab = "Matriz", ylab = "Média do tempo de processamento", color = :black, dpi = 300)
savefig(boxplot_compVincomp, "boxplot_compVincomp.pdf")



benchtable2
benchtable1

