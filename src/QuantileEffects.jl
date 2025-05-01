module QuantileEffects

using Expectations, Distributions
using DataFrames, Random, StatsBase, KernelDensity, FixedEffectModels,StatFiles,Statistics,CategoricalArrays,JSON

function foo(mu = 1., sigma = 2.)
    println("Modified foo definition")
    d = Normal(mu, sigma)
    E = expectation(d)
    return E(x -> cos(x))
end
export DataFrame, load
export foo

function cdf(y, P, YS)
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
function fden(y, Y)
    # Silverman's optimal bandwidth
    h = 1.06 * std(Y) * length(Y)^(-1/5)
    
    # Calculate scaled differences
    d = (Y .- y) / h
    
    # Epanechnikov kernel
    kd = (abs.(d) .< sqrt(5)) .* (1 .- d.^2 / 5) * (3 / (4 * sqrt(5)))
    
    # Estimated density
    fdensity = mean(kd) / h
    
    return fdensity
end
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
    F01y = cdf.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F10F00invF01y = cdf.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
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
    F01y = cdf.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F00invbF01y=cdfinv_bracket.(F01y,Ref(F00),Ref(YS))
    F10F00invF01y = cdf.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
    F10F00invbF01y=cdf.(F00invbF01y,Ref(F10),Ref(YS))
    F00F00invF01y = cdf.(F00invF01y, Ref(F00), Ref(YS))
    F00F00invbF01y = cdf.(F00invbF01y, Ref(F00), Ref(YS))
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
    F01y = cdf.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invF01y = cdfinv.(F01y, Ref(F00), Ref(YS))    # Apply cdfinv to the result
    F10F00invF01y = cdf.(F00invF01y, Ref(F10), Ref(YS)) # Apply cdf to the result
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


    F01y = cdf.(YS01, Ref(F01), Ref(YS))             # Apply cdf to each element of YS01
    F00invbF01y = cdfinv_bracket.(F01y, Ref(F00), Ref(YS))
    F10F00invbF01y = cdf.(F00invbF01y, Ref(F10), Ref(YS))
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


function cic(df,
    trat_var::AbstractString,
    post_var::AbstractString,
    pre_var::AbstractString,
    outcome::AbstractString,
    qq::AbstractVector{<:Real},
    standard_error::Integer,
    bootstrap::Bool,
    bootstrap_reps::Integer,
    cluster::Union{AbstractString,Missing};
    sub_sample_factor::Float64 = 1.0)

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
    NYS00 = length(YS00)
    YS01 = supp2(Y01)                     # vetor de pontos de suporte distintos para Y01
    NYS01 = length(YS01)
    YS10 = supp2(Y10)                     # vetor de pontos de suporte distintos para Y10
    NYS10 = length(YS10)
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
    se = zeros(length(est))

    if standard_error == 1
        F00 = cumsum(f00) / maximum(cumsum(f00))  # normalização para 1
        F01 = cumsum(f01) / maximum(cumsum(f01))
        F10 = cumsum(f10) / maximum(cumsum(f10))
        F11 = cumsum(f11) / maximum(cumsum(f11))

        # A. Erro padrão para o estimador contínuo (delta method)
        cc = 1e-8
        F00_10 = zeros(NYS10)
        F01invF00_10 = zeros(NYS10)
        for i in 1:NYS10
            F00_10[i] = cdf(YS10[i], F00, YS)
            F01invF00_10[i] = cdfinv(F00_10[i], F01, YS)
            f01F01invF00_10[i] = fden(F01invF00_10[i], Y01)
        end
        
        # 1. Contribution of Y00
        P = zeros(NYS00)
        for i in 1:NYS00
            PY00 = ((YS00[i] <= YS10) .- F00_10) ./ f01F01invF00_10
            P[i] = PY00' * f10(f10 .> cc)
        end
        V00 = sum(P .* P .* f00(f00 .> cc)) / length(Y00)
        
        # 2. Contribution of Y01
        P = zeros(NYS01)
        for i in 1:NYS01
            PY01 = -((cdf(YS01[i], F01, YS) .<= F00_10) .- F00_10) ./ f01F01invF00_10
            P[i] = PY01' * f10(f10 .> cc)
        end
        V01 = sum(P .* P .* f01(f01 .> cc)) / length(Y01)
        
        # 3. Contribution of Y10
        P = F01invF00_10 .- F01invF00_10' * f10(f10 .> cc)
        V10 = sum(P .* P .* f10(f10 .> cc)) / length(Y10)
        
        # 4. Contribution of Y11
        P = YS11 .- YS' * f11
        V11 = sum(P .* P .* f11(f11 .> cc)) / length(Y11)
        
        se_con = sqrt(V00 + V01 + V10 + V11)
        se[1] = se_con
    end

    if bootstrap > 0
        #boot = zeros(size(est))
        Nboot = bootstrap
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
            
            YSb = supp2([Y00b; Y01b; Y10b; Y11b])
            YS01b = supp2(Y01b)
            
            
            f00b = prob3(Y00b, YSb)  # Vetor de probabilidades
            f01b = prob3(Y01b, YSb)  # Vetor de probabilidades
            f10b = prob3(Y10b, YSb)  # Vetor de probabilidades
            f11b = prob3(Y11b, YSb)  # Vetor de probabilidades
            
            boot_est[i, 1:end] = cic_con(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est2[i, 1:end] = cic_dci(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est3[i, 1:end] = cic_low(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            boot_est4[i, 1:end] = cic_upp(f00b, f01b, f10b, f11b, qq, YSb, YS01b)
            
            
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

        # Calcular o erro padrão (standard error) ao longo da 1ª dimensão (repetições de bootstrap)
        #se_boot = std(boot_est, dims=1)

        # Para concatenar os erros padrão de cada estimativa, transformamos `se_boot` para 2D
        #se_boot_2d = dropdims(se_boot, dims=1)

        # Para salvar em `data`:
        data = Dict(
            "est" => est,  # Supondo que você já tenha a variável est
            "boot_est" => boot_estimates

        )
    else 
        data = Dict(
            "est" => est,  # Supondo que você já tenha a variável est
            "boot_est" => 0

        )
    end
    # Convert the matrices to arrays
    filename="CIC_results_nocovariates_nboot_$(bootstrap)_size_$(sub_sample_factor)_full_estimation.json"
    # Write the data to a JSON file
    open(filename, "w") do file
        JSON.print(file, data)
    end
    return est
end
export cic

end
