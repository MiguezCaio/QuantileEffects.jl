module QuantileEffects

using Distributions
using DataFrames, Random, StatsBase, KernelDensity, FixedEffectModels,Statistics,CategoricalArrays,JSON
using LinearAlgebra
using Pkg
export DataFrame

function cdf_empirical(y, P, YS)
    # YS is a sorted vector of distinct support points.
    if y < first(YS)
        return 0.0
    elseif y ≥ last(YS)
        return 1.0
    else
        idx = searchsortedlast(YS, y)  # O(log(n)) lookup of the largest index where YS[idx] ≤ y
        return clamp(P[idx], 0.0, 1.0)
    end
end
function cdfinv(y, P, YS)
    # GENERAL INFORMATION
    # This function calculates the inverse of the cumulative distribution function 
    # for a discrete random variable with cumulative distribution function P at the support points YS.

    # INPUT
    # y is a scalar between zero and one.
    # P is a vector of cumulative probabilities corresponding to YS.

    # OUTPUT
    # FINVY is the inverse of the empirical distribution of YS evaluated at y.

    # Tolerance level to avoid issues at actual support points
    cc = 1e-6

     # P assumed sorted increasing, YS sorted support
     if y <= 0.0
        return first(YS)
    elseif y >= 1.0
        return last(YS)
    else
        # find smallest index i with P[i] >= y - cc
        i = searchsortedfirst(P, y - cc)
        # clamp just in case y-cc falls below P[1] or above P[end]
        i = clamp(i, 1, length(YS))
        return YS[i]
    end
end 
function cdfinv_bracket(y, P, YS)
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
function supp(Y)
    # Initialize an empty array for the support points
    YS = []
    
    # While there are still elements in Y
    while length(Y) > 0
        # Append the minimum value of Y to YS
        push!(YS, minimum(Y))
        
        # Filter out elements that are equal to the minimum value
        Y = Y[Y .> minimum(Y)]
    end
    
    # Return the vector of ordered support points
    return YS
end
function supp2(Y)
    # Sort the vector and remove duplicates
    return unique(sort(Y))
end
function prob(Y, YS)
    # Calculate the tolerance (half the minimum difference between support points)
    mdys = minimum(abs.(YS[2:end] .- YS[1:end-1])) / 2

    # Initialize the vector of frequencies (proportions)
    pi = zeros(length(YS))

    # Calculate the frequency (or count) for each support point in YS
    for i in 1:length(YS)
        pi[i] = sum(abs.(Y .- YS[i]) .< (mdys / 100))
        #println(i)
    end

    # Convert counts to proportions
    pi /= sum(pi)

    return pi
end
import KernelDensity: kernel_dist

# Register Epanechnikov with KernelDensity

function fden(ys::AbstractVector, Y::AbstractVector)
    Yc = collect(skipmissing(Y))
    n = length(Yc)
    # Silverman's rule-of-thumb bandwidth
    h = 1.06 * std(Yc) * n^(-1/5)
    # 3) force into (n×1) and (1×m)
    Ycol = reshape(Yc, n, 1)     # n×1
    yrow = reshape(ys, 1, :)     # 1×m
    # Build the matrix of scaled differences: size n×m
    D = (Ycol .- yrow) ./ h
     # Epanechnikov kernel: (3/(4√5))·(1 - u^2/5)  for |u| < √5, else 0
    K = @. (abs(D) <= √5) * (1 - D^2/5) * (3/(4√5))
    # Average over rows and rescale by 1/h
    return vec(sum(K, dims=1)) ./ (n * h)
end
import KernelDensity: kernel_dist
kernel_dist(::Type{Epanechnikov},w::Real) = Epanechnikov(0.0,w)
function fden_package(ys::AbstractVector, Y::AbstractVector)
    # 1) build the KDE object once (Epanechnikov is the default)
    Yc = collect(skipmissing(Y))
    kde_est = kde(Yc; kernel=Epanechnikov)
    # 2) then get density for any vector `ys` of length 321k without memory blowup
    densities = pdf.(Ref(kde_est), ys) 
    return densities   
end

fden_package(y::Real, Y::AbstractVector) = fden_package([y], Y)[1]
function prob2(Y, YS)
    # Calculate the tolerance (half the minimum difference between support points)
    mdys = minimum(abs.(YS[2:end] .- YS[1:end-1])) / 2

    # Use a vectorized approach to count frequencies for each support point in YS
    pi = [count(y -> abs(y - ys) < (mdys / 100), Y) for ys in YS]

    # Convert counts to proportions
    pi /= sum(pi)

    return pi
end
function prob3(Y, YS)
    # Calculate the tolerance (half the minimum difference between support points)
    mdys = minimum(abs.(YS[2:end] .- YS[1:end-1])) / 2

    # Use `fit` to create a histogram with custom bin edges based on YS and tolerance
    bin_edges = vcat(YS .- mdys / 100, maximum(YS) + mdys / 100)
    hist = fit(Histogram, Y, bin_edges)

    # Convert histogram counts to proportions
    pi = hist.weights / sum(hist.weights)

    return pi
end
function cic_con(f00, f01, f10, f11, qq, YS, YS01)

    # INFORMAÇÕES GERAIS
    # Esta função calcula o estimador contínuo CIC. Primeiro
    # estimamos a CDF de Y^N_11 usando a equação (9) do artigo e
    # depois usamos isso para calcular o efeito médio do tratamento.

    # ENTRADAS
    # f00 é o vetor de probabilidades no grupo (0,0)
    # f01 é o vetor de probabilidades no grupo (0,1)
    # f10 é o vetor de probabilidades no grupo (1,0)
    # f11 é o vetor de probabilidades no grupo (1,1)
    # qq é o vetor de quantis nos quais a estimativa precisa ser calculada
    # além do efeito médio
    # YS é o vetor de pontos de suporte
    # YS01 é o vetor de pontos de suporte para Y01
    # f00, f01, f10, f11 e YS têm o mesmo comprimento, mas YS01 pode ser menor

    # SAÍDA
    # est é um escalar com as estimativas CIC contínuas

    F00 = cumsum(f00)  # funções de distribuição acumulada
    F01 = cumsum(f01)  # funções de distribuição acumulada
    F10 = cumsum(f10)  # funções de distribuição acumulada
    F11 = cumsum(f11)  # funções de distribuição acumulada

    #ccc = min(max(abs(cumsum(f00) - cumsum(f01))) / 1000, 1e-9)
    # tolerância que será usada para distinguir entre probabilidades

    # Para cada valor de y em YS01, que é o suporte de Y01
    # calculamos:
    # 1. F_01(y)
    # 2. F_00^-1(F_01(y))
    # 3. FCO(y) = F_10(F^-1_00(F_01(y)))

    ns01 = length(YS01)
    FCO = zeros(ns01)
    # Vectorize the first loop
    F01y = cdf_empirical.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F10F00invF01y = cdf_empirical.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
    FCO = F10F00invF01y
    FCO[end] = 1  # Set the last element to 1

    # Continuous mean estimate
    est = (F11 - vcat(0, F11[1:end-1]))' * YS - (FCO - vcat(0, FCO[1:end-1]))' * YS01

    # Estimate quantiles
    Nq = length(qq)
    est = [est zeros(1, Nq)]

    # Vectorize the quantile calculation
    est[1, 2:end] .= cdfinv.(qq, Ref(F11), Ref(YS)) .- cdfinv.(qq, Ref(FCO), Ref(YS01))

    return est
end
function cic_dci(f00, f01, f10, f11, qq, YS, YS01)
    # INFORMAÇÕES GERAIS
    # Esta função calcula o estimador CIC discreto. Primeiro, estimamos a CDF de Y^N_11 usando a equação (29) do artigo,
    # e depois usamos isso para calcular o efeito médio do tratamento.

    # ENTRADAS
    # f00 é o vetor de probabilidades no grupo (0,0)
    # f01 é o vetor de probabilidades no grupo (0,1)
    # f10 é o vetor de probabilidades no grupo (1,0)
    # f11 é o vetor de probabilidades no grupo (1,1)
    # qq é o vetor de quantis nos quais a estimativa precisa ser calculada, além do efeito médio
    # YS é o vetor de pontos de suporte
    # YS01 é o vetor de pontos de suporte para Y01
    # f00, f01, f10, f11 e YS têm o mesmo comprimento, mas YS01 pode ser menor

    # SAÍDA
    # est é um escalar com as estimativas de independência condicional discreta

    F00 = cumsum(f00)  # Funções de distribuição acumulada
    F01 = cumsum(f01)  # Funções de distribuição acumulada
    F10 = cumsum(f10)  # Funções de distribuição acumulada
    F11 = cumsum(f11)  # Funções de distribuição acumulada

    ccc = min(maximum(abs.(cumsum(f00) - cumsum(f01))) / 1000, 1e-9)
    # Tolerância que será usada para distinguir entre probabilidades

    # Para cada valor de y em YS01, que é o suporte de Y01, calculamos:
    # 1. FUB(y) = FCO(y) = F_10(F^-1_00(F_01(y)))
    # 2. FLB(y) = F_10(F^-1_00(F_01(y))) (inversa alternativa)
    # 3. F_00(F^-1_00(F_01(y)))
    # 4. F_00(F^-1_00(F_01(y))) (inversa alternativa)
    # 5. F_01(y)
    # 6. FDCI_weight(y) (peso para independência condicional discreta)

    ns01 = length(YS01)
    FUB =  zeros(ns01)  # Estimativa do CDF para o limite superior
    FLB = zeros(ns01)  # Estimativa do CDF para o limite inferior
    FDCI = zeros(ns01)
    # Vectorized computation of cdf and inverse cdf operations for YS01
    F01y = cdf_empirical.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F00invbF01y=cdfinv_bracket.(F01y,Ref(F00),Ref(YS))
    F10F00invF01y = cdf_empirical.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
    F10F00invbF01y=cdf_empirical.(F00invbF01y,Ref(F10),Ref(YS))
    F00F00invF01y = cdf_empirical.(F00invF01y, Ref(F00), Ref(YS))
    F00F00invbF01y = cdf_empirical.(F00invbF01y, Ref(F00), Ref(YS))
    FLB = F10F00invbF01y
    FUB = F10F00invF01y

    # now compute the weights vectorized, watching out for the ccc‐threshold:
    Δ     = F00F00invF01y .- F00F00invbF01y
    numer = F01y           .- F00F00invbF01y

    w = ifelse.(
        Δ .> ccc,
        numer ./ Δ,
        zero.(Δ)           # produces a vector of zeros, same type/size as Δ
    )


    # Finally build FDCI in one shot:
    FDCI = FLB .+ (FUB .- FLB) .* w

    # Adjust FDCI at the last value
    FDCI[end] = 1

    # Conditional independence estimate
    est = (F11 - vcat(0, F11[1:end-1]))' * YS - (FDCI - vcat(0, FDCI[1:end-1]))' * YS01

    # Quantile estimates
    Nq = length(qq)
    est = [est zeros(1, Nq)]
    est[1, 2:end] .= cdfinv.(qq, Ref(F11), Ref(YS)) .- cdfinv.(qq, Ref(FDCI), Ref(YS01))

    return est
end

function cic_upp(f00, f01, f10, f11, qq, YS, YS01)

    # INFORMAÇÕES GERAIS
    # Esta função calcula o estimador CIC upperbound. Primeiro, estimamos a CDF de Y^N_11 usando a equação (25) do artigo,
    # e depois usamos isso para calcular o efeito médio do tratamento.

    # ENTRADAS
    # f00 é o vetor de probabilidades no grupo (0,0)
    # f01 é o vetor de probabilidades no grupo (0,1)
    # f10 é o vetor de probabilidades no grupo (1,0)
    # f11 é o vetor de probabilidades no grupo (1,1)
    # qq é o vetor de quantis nos quais a estimativa precisa ser calculada, além do efeito médio
    # YS é o vetor de pontos de suporte
    # YS01 é o vetor de pontos de suporte para Y01
    # f00, f01, f10, f11 e YS têm o mesmo comprimento, mas YS01 pode ser menor

    # SAÍDA
    # est é um escalar com as estimativas do CIC contínuo

    F00 = cumsum(f00)  # Funções de distribuição acumulada
    F01 = cumsum(f01)  # Funções de distribuição acumulada
    F10 = cumsum(f10)  # Funções de distribuição acumulada
    F11 = cumsum(f11)  # Funções de distribuição acumulada

    #ccc = min(maximum(abs.(cumsum(f00) - cumsum(f01))) / 1000, 1e-9)
    # Tolerância que será usada para distinguir entre probabilidades

    # Para cada valor de y em YS01, que é o suporte de Y01, calculamos:
    # 1. FUB(y) = FCO(y) = F_10(F^-1_00(F_01(y)))
    # 2. FLB(y) = F_10(F^-1_00(F_01(y))) (inversa alternativa)
    # 3. F_00(F^-1_00(F_01(y)))
    # 4. F_00(F^-1_00(F_01(y))) (inversa alternativa)
    # 5. F_01(y)
    # 6. FDCI_weight(y) (peso para independência condicional discreta)

    ns01 = length(YS01)
    FUB =  zeros(ns01) 
    # Vectorized computation of cdf and inverse cdf operations for YS01
    F01y = cdf_empirical.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F10F00invF01y = cdf_empirical.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
    FUB = F10F00invF01y
    # Ajustando as estimativas das CDFs de Y^N_11 
    # em valores maiores ou iguais ao máximo de Y01.
    FUB[ns01] = 1

    # Dado as funções de distribuição, as estimativas são:
    # Deixe F^N_11 ser a função de distribuição
    # tau = (F11 - vcat(0, F11[1:end-1]))' * YS
    #      - (F^N_11 - vcat(0, FUB[1:end-1]))' * YS01

    # Estimativa de independência condicional
    est = (F11 - vcat(0, F11[1:end-1]))' * YS - (FUB - vcat(0, FUB[1:end-1]))' * YS01

    # Estimativas dos quantis
    Nq = length(qq)
    est = [est zeros(1, Nq)]
    est[1, 2:end] .= cdfinv.(qq, Ref(F11), Ref(YS)) .- cdfinv.(qq, Ref(FUB), Ref(YS01))

    return est
end
function cic_low(f00, f01, f10, f11, qq, YS, YS01)

    # INFORMAÇÕES GERAIS
    # Esta função calcula o estimador CIC contínuo para o limite inferior. 
    # Estimamos a CDF de Y^N_11 usando a equação (25) do artigo, 
    # e depois utilizamos isso para calcular o efeito médio do tratamento.

    # ENTRADAS
    # f00 é o vetor de probabilidades do grupo (0,0)
    # f01 é o vetor de probabilidades do grupo (0,1)
    # f10 é o vetor de probabilidades do grupo (1,0)
    # f11 é o vetor de probabilidades do grupo (1,1)
    # qq é o vetor de quantis nos quais a estimativa precisa ser calculada 
    # além do efeito médio
    # YS é o vetor de pontos de suporte
    # YS01 é o vetor de pontos de suporte para Y01
    # f00, f01, f10, f11 e YS têm o mesmo comprimento, mas YS01 pode ser menor

    # SAÍDA
    # est é um escalar com as estimativas do limite inferior

    F00 = cumsum(f00)  # Funções de distribuição acumulada
    F01 = cumsum(f01)  # Funções de distribuição acumulada
    F10 = cumsum(f10)  # Funções de distribuição acumulada
    F11 = cumsum(f11)  # Funções de distribuição acumulada

    #ccc = min(maximum(abs.(cumsum(f00) - cumsum(f01))) / 1000, 1e-9)
    # Tolerância para distinguir entre probabilidades

    # Para cada valor de y em YS01, que é o suporte de Y01, calculamos:
    # 1. FUB(y) = FCO(y) = F_10(F^-1_00(F_01(y)))
    # 2. FLB(y) = F_10(F^-1_00(F_01(y))) (inversa alternativa)
    # 3. F_00(F^-1_00(F_01(y)))
    # 4. F_00(F^-1_00(F_01(y))) (inversa alternativa)
    # 5. F_01(y)
    # 6. FDCI_weight(y) (peso para independência condicional discreta)

    ns01 = length(YS01)
    FLB = zeros(ns01)  # Estimador do limite inferior


    F01y = cdf_empirical.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invbF01y = cdfinv_bracket.(F01y, Ref(F00), Ref(YS))
    F10F00invbF01y = cdf_empirical.(F00invbF01y, Ref(F10), Ref(YS))
    FLB = F10F00invbF01y

    # Ajustando as estimativas das CDFs de Y^N_11 
    # em valores maiores ou iguais ao máximo de Y01
    FLB[ns01] = 1

    # Dado as funções de distribuição, as estimativas são:
    # tau = (F11 - vcat(0, F11[1:end-1]))' * YS 
    #      - (F^N_11 - vcat(0, FLB[1:end-1]))' * YS01

    # Estimativa de independência condicional
    est = (F11 - vcat(0, F11[1:end-1]))' * YS - (FLB - vcat(0, FLB[1:end-1]))' * YS01
    #est = 0 
    # Estimativas dos quantis
    Nq = length(qq)
    est = [est zeros(1, Nq)]
    est[1, 2:end] .= cdfinv.(qq, Ref(F11), Ref(YS)) .- cdfinv.(qq, Ref(FLB), Ref(YS01))


    return est
end
"""
    make_se_estimator(Y00, Y01, Y10, Y11; cc=1e-8)

Precompute everything you need for the standard‐error function,
and return a function `se(q)` that only loops over `q`.
"""
function make_se_estimator(
    Y00::AbstractVector,
    Y01::AbstractVector,
    Y10::AbstractVector,
    Y11::AbstractVector;
    cc::Float64 = 1e-8
)
    # 1) build the big support & probs once
    YS   = supp2(vcat(Y00, Y01, Y10, Y11))
    f00  = prob3(Y00, YS)
    f01  = prob3(Y01, YS)
    f10  = prob3(Y10, YS)
    f11  = prob3(Y11, YS)

    # 2) normalize to get CDFs
    F00 = cumsum(f00) / maximum(cumsum(f00))
    F01 = cumsum(f01) / maximum(cumsum(f01))
    F10 = cumsum(f10) / maximum(cumsum(f10))
    F11 = cumsum(f11) / maximum(cumsum(f11))

    # 3) distinct supports and map‐indices
    YS00 = supp2(Y00); N00 = length(Y00)
    YS01 = supp2(Y01); N01 = length(Y01)
    YS10 = supp2(Y10); N10 = length(Y10)
    YS11 = supp2(Y11); N11 = length(Y11)

    map_idx = Dict(v=>i for (i,v) in enumerate(YS))
    idx00   = map(i->map_idx[i], YS00)
    idx01   = map(i->map_idx[i], YS01)
    idx10   = map(i->map_idx[i], YS10)
    idx11   = map(i->map_idx[i], YS11)

    # 4) masked densities
    f00m = f00[f00 .> cc]
    f01m = f01[f01 .> cc]
    f10m = f10[f10 .> cc]
    f11m = f11[f11 .> cc]

    # 5) now return a simple `se(q)` that closes over all of the above:
    function se(q::Float64)
        # back‐transforms
        x10 = cdfinv(q, F10, YS)
        a00 = cdf_empirical(x10, F00, YS)
        x01 = cdfinv(a00, F01, YS)

        # “counterfactual” density
        d01 = fden_package(x01, Y01)

        # V00
        P00 = ((YS00[f00[idx00] .> cc] .<= x10) .- a00) ./ d01
        V00 = sum(P00.^2 .* f00m) / N00

        # V01
        emp01 = cdf_empirical.(YS01[f01[idx01] .> cc], Ref(F01), Ref(YS))
        Q01   = (-(emp01 .<= a00) .+ a00) ./ d01
        V01   = sum(Q01.^2 .* f01m) / N01

        # V10
        emp10 = cdf_empirical.(YS10[f10[idx10] .> cc], Ref(F10), Ref(YS))
        den10 = fden_package(x10, Y10)
        R10   = ( - fden_package(x10, Y00) * ((emp10 .<= q) .- q) ) / (d01 * den10)
        V10   = sum(R10.^2 .* f10m) / N10

        # V11
        z11 = cdfinv(q, F11, YS)
        emp11 = (YS11[f11[idx11] .> cc] .<= z11)
        S11   = (emp11 .- q) ./ fden_package(z11, Y11)
        V11   = sum(S11.^2 .* f11m) / N11

        return sqrt(V00 + V01 + V10 + V11)
    end

    return se
end

function cic(df,
    trat_var::AbstractString,
    post_var::AbstractString,
    pre_var::AbstractString,
    outcome::AbstractString,
    qq::AbstractVector{<:Real},
    standard_error::Integer,
    se_avg::Integer,
    bootstrap::Bool,
    bootstrap_reps::Integer,
    cluster::Union{AbstractString,Missing};
    sub_sample_factor::Float64 = 1.0)
@assert 0.0 < sub_sample_factor ≤ 1.0 "sub_sample_factor must be in (0,1]"
    # INFORMAÇÕES GERAIS
    # Este programa calcula quatro conjuntos de estimativas CIC
    # (para independência condicional contínua, discreta, limite inferior e limite superior)
    # Se necessário, também calcula erros padrão analíticos e erros padrão bootstrap.
    ###First we remove any missing values
    df2=df[.!ismissing.(df[!,Symbol(trat_var)]) .& .!ismissing.(df[!,Symbol(pre_var)]) .& .!ismissing.(df[!,Symbol(post_var)]) .& .!ismissing.(df[!,Symbol(outcome)]),:]
    subsample_indices = sample(1:size(df2, 1), Int(floor(size(df2, 1) * sub_sample_factor)), replace=false)
    if sub_sample_factor ==1 
        df3=df2
    else
        df3 = df2[subsample_indices, :]
    end
    ###then we define the 4 groups
    #Y00 is vector of outcomes in (0,0) group
    #   Y01 is vector of outcomes in (0,1) group
    # Y10 is vector of outcomes in (1,0) group
    # Y11 is vector of outcomes in (1,1) group
    # ENTRADAS
    Y00=df3[(df3[!,Symbol(trat_var)] .== 0) .& (df3[!,Symbol(pre_var)] .== 1), :][!,Symbol(outcome)]
    Y01=df3[(df3[!,Symbol(trat_var)] .== 0) .& (df3[!,Symbol(post_var)] .== 1), :][!,Symbol(outcome)]
    Y10=df3[(df3[!,Symbol(trat_var)] .== 1) .& (df3[!,Symbol(pre_var)] .== 1), :][!,Symbol(outcome)]
    Y11=df3[(df3[!,Symbol(trat_var)] .== 1) .& (df3[!,Symbol(post_var)] .== 1), :][!,Symbol(outcome)]
    # qq é um vetor de quantis para os quais os efeitos quantílicos serão calculados
    # standard_error é um indicador 0/1 que define se os erros padrão devem ser calculados
    # bootstrap é um indicador se os erros padrão bootstrap devem ser calculados
    # se for positivo, os erros padrão bootstrap serão calculados com o número de repetições igual ao valor de bootstrap.

    # SAÍDAS
    # est é um vetor de estimativas:
    # 1. contínua
    
    # se é um vetor de estimativas dos erros padrão:
    # 1. contínua
    # Manipulação preliminar dos dados
    YS = supp2(vcat(Y00, Y01, Y10, Y11))  # vetor de pontos de suporte distintos
    YS00 = supp2(Y00)                     # vetor de pontos de suporte distintos para Y00
    #NYS00 = length(YS00)
    YS01 = supp2(Y01)                     # vetor de pontos de suporte distintos para Y01
    #NYS01 = length(YS01)
    YS10 = supp2(Y10)                     # vetor de pontos de suporte distintos para Y10
    #NYS10 = length(YS10)
    YS11 = supp2(Y11)                     # vetor de pontos de suporte distintos para Y11
    f00 = prob3(Y00, YS)                  # vetor de probabilidades
    f01 = prob3(Y01, YS)                  # vetor de probabilidades
    f10 = prob3(Y10, YS)                  # vetor de probabilidades
    f11 = prob3(Y11, YS)                  # vetor de probabilidades
    # Estimativa contínua
    est_con = cic_con(f00, f01, f10, f11, qq, YS, YS01)

    # Estimativa de independência condicional
    est_dci = cic_dci(f00, f01, f10, f11, qq, YS, YS01)

    # Estimativa do limite inferior
    est_low = cic_low(f00, f01, f10, f11, qq, YS, YS01)

    # Estimativa do limite superior
    est_upp = cic_upp(f00, f01, f10, f11, qq, YS, YS01)
    
   est = [est_con, est_dci, est_low, est_upp]
   #est = est_con
    # Cálculo dos erros padrão
   se = [ zeros(size(e)) for e in est ]

    if standard_error == 1
        println("running the analytical calculations")
        F00 = cumsum(f00) / maximum(cumsum(f00))  # normalização para 1
        F01 = cumsum(f01) / maximum(cumsum(f01))
        F10 = cumsum(f10) / maximum(cumsum(f10))
        F11 = cumsum(f11) / maximum(cumsum(f11))
        cc = 1e-8
        map_idx = Dict(v => i for (i,v) in enumerate(YS))
        idx00   = [ map_idx[v] for v in YS00 ]
        idx10   = [ map_idx[v] for v in YS10 ]
        idx01   = [ map_idx[v] for v in YS01 ]
        idx11   = [ map_idx[v] for v in YS11 ]

       

        if se_avg ==1
            #We create the kernel density of  Y01, it will be useful for later
            Y01c    = collect(skipmissing(Y01))
            kde01 = kde(Y01c; kernel=Epanechnikov)
            fden_r01(ys) = pdf.(Ref(kde01), ys)

            # A. Erro padrão para o estimador contínuo (delta method)
            # pre-mask your densities once
            f10m   = f10[f10 .> cc]                # vector of length L10
            f00m   = f00[f00 .> cc]                # vector of length L00
            f01m   = f01[f01 .> cc]                # vector of length L01
            f11m   = f11[f11 .> cc]                # vector of length L11
            F00_10 = cdf_empirical.(YS10[f10[idx10] .> cc], Ref(F00), Ref(YS))
            F01invF00_10 = cdfinv.(F00_10, Ref(F01), Ref(YS))
            f01F01invF00_10 = fden_r01(F01invF00_10)
            ϵ = 1e-4
            f01F01invF00_10_fixed = ifelse.(f01F01invF00_10 .== 0.0, ϵ, f01F01invF00_10)
        
            # 1. Contribution of Y00
            Y00s = YS00[f00[idx00] .> cc]        # length n00
            N00=length(Y00s)
            Y10s = YS10[f10[idx10] .> cc]        # length n10
            w       = f10m ./ f01F01invF00_10_fixed                       # length n10
            B       = sum(F00_10 .* w)   # constant term
            # wsuf[k] = sum_{j=k..end} w[j]
            wsuf = reverse!(cumsum(reverse(w)))     
            P00 = zeros(Float64, N00)
            j    = 1
            for i in eachindex(Y00s)
            # advance j until Y10s[j] ≥ Y00s[i]
            while j ≤ length(Y10s) && Y10s[j] < Y00s[i]
                j += 1
            end
        
            P00[i] = (j > length(Y10s) ? 0 : wsuf[j]) - B
            end
            V00 = sum(P00.^2 .* f00m) / length(Y00)
                #2. Contribution of Y01

            #=We originally calculated using matrix adjoint calculation, but ran out of memory in large Dataset
            M01 = -((C01 .<= F00_10') .- F00_10') ./ f01F01invF00_10'
            P01 =  M01 * f10m                     # length NYS01
            =#
            #We do the same conditional weighted sum as before
            Y01s=YS01[f01[idx01] .> cc ]
            N01=length(Y01s)
            C01 =cdf_empirical.(Y01s,Ref(F01),Ref(YS))
            # 2.a) weights and constant B
            w = f10m ./ f01F01invF00_10_fixed                    # length n10
            B = sum(F00_10 .* w)                  # scalar   
            
            # --- 2) prepare suffix sums w.r.t. F0010 ---

            # build suffix‐sum: wsuf[k] = sum_{j=k..end} w_s[j]
            wsuf   = reverse!(cumsum(reverse(w)))  # length n10

            # --- 3) evaluate P01 without any big matrix ---
            #P01 = similar(Y01s)                  # length n01
            P01 = zeros(Float64, N01)

            j = 1
            for (k, c) in enumerate(C01)
                # advance j until F_s[j] >= c
                while j ≤ length(F00_10) && F00_10[j] < c
                    j += 1
                end
                # suffix‐sum at j is sum of all w_s[j..end]
                a = (j > length(F00_10) ? zero(w[i]) : wsuf[j])
                P01[k] = -(a - B)
            end

            # --- 4) compute the variance contribution ---
            V01 = sum(P01.^2 .* f01m) / length(Y01)
            
            # 3. Contribution of Y10
            P10 = F01invF00_10 .- sum(F01invF00_10.* f10m)
            V10 = sum(P10.^2 .* f10m) / length(Y10)
            
            # 4. Contribution of Y11
            P11 = YS11 .- sum(YS .* f11)
            V11 = sum(P11.^2 .* f11m) / length(Y11)
            se_con = sqrt(V00 + V01 + V10 + V11)
            se[1][1] = se_con
        end

        #### Now the quantile analytical standard errors
        # 2) write a small helper that computes se for a single q
        
        # do this *once* per bootstrap sample:
        se_boot = make_se_estimator(Y00, Y01, Y10, Y11; cc=1e-8)
        se_vec = [ se_boot(q) for q in qq ]
        se[1][2:end]=se_vec
    end
    se_mat = Array{Float64}(undef, bootstrap_reps, length(qq)+1)
    if bootstrap
        #boot = zeros(size(est))
        Nboot = bootstrap_reps
        boot_est = zeros(Nboot,length(est_con))
        boot_est2 = zeros(Nboot,length(est_con))
        boot_est3 = zeros(Nboot,length(est_con))
        boot_est4 = zeros(Nboot,length(est_con))
        #N00 = length(Y00)
        #N01 = length(Y01)
        #N10 = length(Y10)
        #N11 = length(Y11)
        n = nrow(df2)
        if cluster === missing
            clusters = missing
        else
            clusters = unique(df2[!, Symbol(cluster)])
        end
        for i in 1:Nboot
            println("rep $i")
            if cluster === missing
                idx    = rand(1:n, n)           # sample rows with replacement
                boot_df = df2[idx, :]
            else 
                indices = rand(1:length(clusters), length(clusters))
                aa = clusters[indices]
                sampled_clusters_set = Set(aa)
                boot_df = df2[in.(df2[!, Symbol(cluster)], Ref(sampled_clusters_set)), :]
            end
            boot_df2=boot_df
            subsample_indices = sample(1:size(boot_df2, 1), Int(floor(size(boot_df2, 1) * sub_sample_factor)), replace=false)
            if sub_sample_factor ==1 
                boot_df3=boot_df2
            else
                boot_df3 = boot_df2[subsample_indices, :]
            end
            Y00b=boot_df3[(boot_df3[!,Symbol(trat_var)] .== 0) .& (boot_df3[!,Symbol(pre_var)] .== 1), :][!,Symbol(outcome)]
            Y01b=boot_df3[(boot_df3[!,Symbol(trat_var)] .== 0) .& (boot_df3[!,Symbol(post_var)] .== 1), :][!,Symbol(outcome)]
            Y10b=boot_df3[(boot_df3[!,Symbol(trat_var)] .== 1) .& (boot_df3[!,Symbol(pre_var)] .== 1), :][!,Symbol(outcome)]
            Y11b=boot_df3[(boot_df3[!,Symbol(trat_var)] .== 1) .& (boot_df3[!,Symbol(post_var)] .== 1), :][!,Symbol(outcome)]      
            YSb=supp2(vcat(Y00b, Y01b, Y10b, Y11b))
            f00b = prob3(Y00b, YSb)  # Vetor de probabilidades
            f01b = prob3(Y01b, YSb)  # Vetor de probabilidades
            f10b = prob3(Y10b, YSb)  # Vetor de probabilidades
            f11b = prob3(Y11b, YSb)  # Vetor de probabilidades
            YS01b = supp2(Y01b)
            boot_est[i, 1:end] = cic_con(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est2[i, 1:end] = cic_dci(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est3[i, 1:end] = cic_low(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est4[i, 1:end] = cic_upp(f00b, f01b, f10b, f11b, qq, YSb, YS01b)

            se_bootb = make_se_estimator(Y00b, Y01b, Y10b, Y11b; cc=1e-8)
            se_vecb = [ se_bootb(q) for q in qq ]
            se_mat[i, 1] = 0
            se_mat[i, 2:end] = se_vecb
        
        end
        #se_boot = std(boot_est, dims=1)
        #se = vcat(se, se_boot)
        # Inicializar a matriz 3D para armazenar todas as estimativas
        n_repetitions = size(boot_est, 1)  # Número de repetições de bootstrap
        n_estimations = size(boot_est, 2)  # Número de estimativas por repetição
        n_functions = 4                     # Número de funções diferentes (cic_con, cic_dci, cic_low, cic_upp)

        # Cria uma matriz 3D para armazenar todas as estimativas
        boot_estimates = zeros(n_repetitions, n_estimations, n_functions)

        # Armazenar as estimativas em diferentes "fatias" da 3ª dimensão
        boot_estimates[:, :, 1] = boot_est
        boot_estimates[:, :, 2] = boot_est2
        boot_estimates[:, :, 3] = boot_est3
        boot_estimates[:, :, 4] = boot_est4
        if standard_error==1
            est_by_boot=2:4
        else
            est_by_boot=1:4
        end
        for k in est_by_boot
            cov_matrix = cov(boot_estimates[:, :, k]; dims=1)
            se[k][:]=sqrt.(diag(cov_matrix))
        end
        # Calcular o erro padrão (standard error) ao longo da 1ª dimensão (repetições de bootstrap)
        #se_boot = std(boot_est, dims=1)

        # Para concatenar os erros padrão de cada estimativa, transformamos `se_boot` para 2D
        #se_boot_2d = dropdims(se_boot, dims=1)

        # Para salvar em `data`:
        data = Dict(
            "est" => est,  # Supondo que você já tenha a variável est
            "boot_est" => boot_estimates,
            "se" => se,
            "se_boot" => se_mat

        )
    else 
        data = Dict(
            "est" => est,  # Supondo que você já tenha a variável est
            "boot_est" => 0,
            "se" => se,
            "se_boot" => 0
        )
    end
    # Convert the matrices to arrays
    filename="CIC_results_nocovariates_nboot_$(bootstrap_reps)_size_$(sub_sample_factor)_cluster_$(cluster)_full_estimation.json"
    # Write the data to a JSON file
    open(filename, "w") do file
        JSON.print(file, data)
    end
    return data
end
export cic

end
