using DataFrames, Random, StatsBase, KernelDensity, FixedEffectModels,StatFiles,Statistics,CategoricalArrays,JSON
const Root = pwd()
const Tests = joinpath(Root,"test")
const Results = joinpath(Root,"Results")


using CSV
using QuantileEffects
cd(Tests)
file =joinpath(Tests,"mvd.dat")
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
percentiles = collect(0.05:0.05:0.90)
cd(Results)
#model=cic(df,"Column5","Column4","new_col","Column2",percentiles,0,200,"municipio_coorte",1)
#cic v2, group(v5) time(v4) quantiles(0.05(0.05)0.9) reps(200)  saving(cic_visconti_1995, replace)
trat_var="Column4"
post_var="Column3"
pre_var="new_col"
outcome="Column1"
sub_sample_factor=1
percentiles=collect(0.01:0.01:0.99)
cd(Tests)
test_model=cic(df,trat_var,post_var,pre_var,outcome,percentiles,1,1,true,20,missing)

