using QuantileEffects
using Test

const Root = "C:\\Users\\migue\\OneDrive - Fundacao Getulio Vargas - FGV\\Projetos\\Julia Servidor"

const RawData = joinpath(Root, "Raw Data")
const TempData = joinpath(Root, "Temp Data")
const FinalData = joinpath(Root, "Final Datasets")

const Codes = joinpath(Root, "Codes", "Estimation")
const CodesResults = joinpath(Codes, "Results")
const CodesRobustness = joinpath(Codes, "Robustness Tests")

const Tables = joinpath(Root, "Tables")
# Assuming df contains your data
# Carregando o arquivo SAEB MT
cd(FinalData)
df = DataFrame(load("final_data_lp.dta"))
println(names(df))
df.municipio_coorte = string.(df.municipio, "_", df.coorte)
percentiles = collect(0.01:0.01:0.99)
path=joinpath(TempData, "new_cic_julia_lp")
cd(path)
@testset "check cic" begin
    #@test foo(0)<1E-16
    @test cic(df,"a_trat","a_coorte_post","a_coorte_pre","z_nota_lp_b2013",percentiles,0,true,20,"municipio_coorte";sub_sample_factor=0.01)
    @test cic(df,"a_trat","a_coorte_post","a_coorte_pre","z_nota_lp_b2013",percentiles,0,true,20,missing;sub_sample_factor=0.01)
end

@testset "CIC smoke‐tests" begin
    res = cic(df, "a_trat", "a_coorte_post", "a_coorte_pre",
              "z_nota_lp_b2013", percentiles, 0, true, 20, "municipio_coorte";
              sub_sample_factor = 0.01)
println(size(res))
#println(res)
     # 1) Should be a Vector of length 4
     @test isa(res, Vector)
     @test length(res) == 4
 
     # 2) Each element is a Float64 vector of the same length as percentiles
     @test all(x -> isa(x, AbstractVector{Float64}), res)
     @test all(x -> length(x) == length(percentiles), res)
 
     # 3) All numbers finite
     @test all(x -> all(isfinite, x), res)                                # no NaNs/Infs
end
res