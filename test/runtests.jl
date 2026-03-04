using Test
using CSV
using Statistics
using Random
using QuantileEffects
file = joinpath(@__DIR__, "mvd.dat")

#"C:\Users\migue\OneDrive - Fundacao Getulio Vargas - FGV\Projetos\Julia Servidor\cicprograms"
# read it in, collapsing repeated spaces into one
df = CSV.read(file, DataFrame;
    delim = ' ',
    ignorerepeated = true,
    header = false,            # set to `true` if your file has a header row
)

# peek
df[!,:new_col] = 1 .- df[!,:Column3]
df=df[df[!,:Column19] .== 16,:]
#model=cic(df,"Column5","Column4","new_col","Column2",percentiles,0,200,"municipio_coorte",1)
#cic v2, group(v5) time(v4) quantiles(0.05(0.05)0.9) reps(200)  saving(cic_visconti_1995, replace)
trat_var="Column4"
post_var="Column3"
pre_var="new_col"
outcome="Column1"
nboot=20
sub_sample_factor=1
percentiles=collect(0.01:0.01:0.99)
test_rif=calculate_rif(df,outcome,0.5)
@testset "CIC smoke‐tests" begin
    result = cic(df,trat_var,post_var,pre_var,outcome, percentiles,1,1,true,nboot,missing)
# 3) Basic sanity checks
    @testset "keys exist" begin
        @test sort(collect(keys(result))) == sort(["est","boot_est","se","se_boot"])
    end
     # 4) No NaN/Inf and correct dimensions
    for key in ["est","boot_est","se","se_boot"]
        arr = result[key]
        # "est" and "se" are Vector{Matrix}, so iterate mat→elements.
        # "boot_est" (3-D) and "se_boot" (2-D) are plain numeric arrays; just flatten.
        arr_vec = arr isa AbstractVector ? [x for mat in arr for x in mat] : vec(arr)
        @testset "checking $key" begin
            # a) all entries finite (no NaN, no Inf)
            @test all(isfinite, arr_vec)

            # b) last dimension matches length(percentiles)
            if key in ("est","se")
                for i in arr
                    lastdim = size(i, ndims(i))
                    @test lastdim == length(percentiles)+1
                end
            elseif key =="boot_est"
                # must be 3D
                @test ndims(arr) == 3

                # exact shape check
                @test size(arr) == (nboot, length(percentiles)+1, 4)
            else
                lastdim = size(arr, ndims(arr))
                @test lastdim == length(percentiles)+1
            end
        end
    end                         # no NaNs/Infs
end

@testset "RIF smoke-tests" begin
    col = collect(skipmissing(df[!, Symbol(outcome)]))

    @testset "calculate_rif" begin
        # Output length matches number of rows in df
        rif50 = calculate_rif(df, outcome, 0.5)
        @test length(rif50) == size(df, 1)

        # All values are finite
        @test all(isfinite, rif50)

        # Works for a different quantile
        rif90 = calculate_rif(df, outcome, 0.9)
        @test length(rif90) == size(df, 1)
        @test all(isfinite, rif90)
    end

    @testset "calculate_rif – E[RIF] = q_τ on continuous data" begin
        # E[RIF(Y; q_τ, F_Y)] = q_τ is a population property that holds for
        # continuous distributions (Firpo, Fortin & Lemieux 2009, eq. 3).
        # It requires P(Y ≤ q_τ) = τ exactly, which fails for discrete outcomes
        # like the Mauritius wage data. We verify it on synthetic Normal data.
        Random.seed!(1234)
        df_cont = DataFrame(y = randn(2000))
        for τ in (0.25, 0.5, 0.75)
            rif = calculate_rif(df_cont, "y", τ)
            @test isapprox(mean(rif), quantile(df_cont.y, τ), rtol=0.05)
        end
    end

    @testset "rif_did – unconditional and 1|1 distributions" begin
        qq_rif = collect(0.1:0.1:0.9)
        res = rif_did(df, trat_var, post_var, pre_var, outcome, qq_rif)

        # Must return a DataFrame with the right number of rows
        @test res isa DataFrame
        @test size(res, 1) == length(qq_rif)

        # All expected columns must be present
        for col_name in [:percentil, :nota, :DID_pct, :density, :RIF_DID,
                         :nota11, :DID_pct11, :density11, :RIF_DID11]
            @test col_name in propertynames(res)
        end

        # percentil column must equal qq * 100
        @test res[!, :percentil] ≈ qq_rif .* 100

        # Numeric columns must be finite
        for col_name in [:nota, :DID_pct, :density, :RIF_DID,
                         :nota11, :DID_pct11, :density11, :RIF_DID11]
            @test all(isfinite, res[!, col_name])
        end

        # Densities must be strictly positive
        @test all(>(0), res[!, :density])
        @test all(>(0), res[!, :density11])

        # Unconditional quantiles (nota) are pooled over all four groups,
        # so they should differ from the 1|1 quantiles (nota11) in general
        # (they are not guaranteed equal; just check they are both monotone)
        @test issorted(res[!, :nota])
        @test issorted(res[!, :nota11])
    end
end
