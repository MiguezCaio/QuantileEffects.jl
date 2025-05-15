using DataFrames, Random, StatsBase, KernelDensity, FixedEffectModels,StatFiles,Statistics,CategoricalArrays,JSON
const Root = "C:\\Users\\migue\\OneDrive - Fundacao Getulio Vargas - FGV\\Projetos\\Julia Servidor"

const RawData = joinpath(Root, "Raw Data")
const TempData = joinpath(Root, "Temp Data")
const FinalData = joinpath(Root, "Final Datasets")

const Codes = joinpath(Root, "Codes", "Estimation")
const Tests = joinpath("E:\\Projetos\\Julia Packages\\QuantileEffects", "Results")
const CodesRobustness = joinpath(Codes, "Robustness Tests")

cd(FinalData)
df = DataFrame(load("final_data_mt.dta"))
println(names(df))
df.municipio_coorte = string.(df.municipio, "_", df.coorte)
using Random
function cdfinv_bracket(y, P, YS)

    # Informação Geral
    # Esta função estima o inverso de uma função de distribuição cumulativa
    # usando a definição alternativa da equação (24) do artigo
    # para uma variável aleatória discreta.

    # INPUT
    # y é um escalar entre zero e um.
    # P são as probabilidades cumulativas nos pontos de suporte
    # YS é o vetor de pontos de suporte
    # P e YS devem ter o mesmo comprimento

    # OUTPUT
    # FINVY é o inverso da distribuição empírica de x avaliada em y.
    # Se o valor do inverso for menos infinito, o valor usado
    # será o menor ponto de suporte menos 100 vezes a diferença entre
    # o maior e o menor ponto de suporte.

    cc = 0.000001
    NS = length(YS)  # número de pontos de suporte
    t = collect(1:NS)

    if y >= P[1] - cc/2
        RANK = maximum(t[P .<= y + cc])
        FINVY = YS[RANK]
    else
        FINVY = YS[1] - 100 * (1 + YS[NS] - YS[1])
    end

    return FINVY
end
using StatsBase
@inline function cdfinv_bracket_new(y::Float64, P::AbstractVector{Float64}, YS::AbstractVector{Float64})
    cc = 1e-6
    n  = length(YS)
    p1 = P[1]
    if y >= p1 - cc/2
        # find the largest index i with P[i] <= y + cc
        i = searchsortedlast(P, y + cc)
        # if y+cc < P[1], searchsortedlast gives 0, so clamp to 1:
        i = clamp(i, 1, n)
        return YS[i]
    else
        # “minus infinity” rule: go a little below the minimum support
        return YS[1] - 100 * (YS[n] - YS[1] + 1)
    end
end
"a_trat","a_coorte_post","a_coorte_pre","z_nota_mt_b2013"
trat_var="a_trat"
post_var="a_coorte_post"
pre_var="a_coorte_pre"
outcome="z_nota_mt_b2013"
sub_sample_factor=1
percentiles=collect(0.01:0.01:0.99)
cd(Tests)
test_model=cic(df,trat_var,post_var,pre_var,outcome,percentiles,1,1,true,20,"municipio")

