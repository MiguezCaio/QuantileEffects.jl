using Test
using CSV
using QuantileEffects
file =joinpath("mvd.dat")

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

@testset "CIC smoke‐tests" begin
    result = cic(df,trat_var,post_var,pre_var,outcome, percentiles,1,1,true,nboot,missing)
# 3) Basic sanity checks
    @testset "keys exist" begin
        @test sort(collect(keys(result))) == sort(["est","boot_est","se","se_boot"])
    end
     # 4) No NaN/Inf and correct dimensions
    for key in ["est","boot_est","se","se_boot"]
        arr = result[key]
        arr_vec = [ x for mat in arr for x in mat ]
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
