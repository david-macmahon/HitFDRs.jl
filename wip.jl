using HitFDRs
using ProgressBars
using DataFrames
using StatsBase
using Distributions
using Plots

#using SeticoreCapnp
#using FrequencyDriftRateTransforms
#using Memoize
#using DimensionalData
#using DataFrames: groupby
#using Plots
#using LinearAlgebra # for `normalize`, which is extended by Plots and StatsBase
#
#using StatsPlots

##

obsid = "20240809T065515Z-20240628-0003"
dfs = map(ProgressBar(0:63)) do i
    loadhitsdf("/mnt/blpn$i/scratch/data/20240809T065515Z-20240628-0003/seticore_search/guppi_60531_24915_002762_GJ367_0001.hits")
end |> filter(!isempty)

##

hosts = [df.host[1] for df in dfs]
nhits = nrow.(dfs)
nunique = [length(unique(tuple.(df.frequency, df.driftRate))) for df in dfs]

df1 = reduce(vcat, dfs)
gdf = groupby(df1, :host);

foff = df1.foff[1]
tsamp = df1.tsamp[1]
nsamp = df1.numTimesteps[1]
dfdt = 1e6 * foff / tsamp
# This is the inherent drift rate resolution of the Fast Taylor Tree search
δr = 1e6 * foff / (tsamp * (nextpow(2, nsamp) - 1))
δrn = δr / dfdt
# Cheater/imperfect way to get `nfpc` (number of fine channels per coarse channel)
nfpc = nextpow(2, maximum(df1.index))
nsti = round(Int, 1e6 * foff * tsamp)
scaling = Float32(nfpc)^2 * 4 * nsti

##

host = "blpn11"
df = gdf[(;host)]
datas = loadhitsdata(obsid, host, scaling);

##

#=
fdrrates=calcfdr.(datas, δrn; own=true)
fdrs=first.(fdrrates);
ratess=last.(fdrrates);

df.fdrrate = map(fdrrates) do (fdr, rates)
    rates[findmax(fdr)[2][2]]
end
=#

# Compute FDRs
fdrs = calcfdr.(datas, δrn, median; own=true)
fdrszero = calcfdr.(datas, δrn, zero∘eltype; own=true)
fdrsmean = calcfdr.(datas, δrn, mean; own=true)
fdrsrand = calcfdr.(datas, δrn, datas; own=true)
fdrsgamma = calcfdr.(datas, δrn, Gamma; own=true)

# Add fdrrate column to df for peak rates from FDRs
df.fdrrate = driftrate.(fdrs)
df.fdrratezeropad = driftrate.(fdrszero)
df.fdrratemeanpad = driftrate.(fdrsmean)
df.fdrraterandpad = driftrate.(fdrsrand)
df.fdrrategammapad = driftrate.(fdrsgamma)

df.fdrsnr = calcsnr.(fdrs);
df.fdrsnrzeropad = calcsnr.(fdrszero);
df.fdrsnrmeanpad = calcsnr.(fdrsmean);
df.fdrsnrrandpad = calcsnr.(fdrsrand);
df.fdrsnrgammapad = calcsnr.(fdrsgamma);

dfnz=df[df.driftRate.!=0, :]

##

splots = ["zero",   "median", "mean",   "rand",   "gamma"  ] .=>
         [fdrszero, fdrs,     fdrsmean, fdrsrand, fdrsgamma]

scsnrmax = maximum(df.snr)
snrthreshold = 6
#=
snrplots = (["zero" "median" "mean" "rand" "gamma"] .* " fill") .=>
           (collect∘skipmissing).([df.fdrsnrzeropad df.fdrsnr df.fdrsnrmeanpad df.fdrsnrrandpad df.fdrsnrgammapad])
=#
snrplots = (["zero" "median" "mean" "rand" "gamma"] .* " fill") .=>
            [Ref(df.fdrsnrzeropad) Ref(df.fdrsnr) Ref(df.fdrsnrmeanpad) Ref(df.fdrsnrrandpad) Ref(df.fdrsnrgammapad)]

# Pick random hit from df
i=rand(df.idx)

for (l,f) in splots
    plotspectrogram(datas[i], df[i,:], f[i]; extra_title=" ($l fill)")|>display
end

for (l,f) in splots
    histogram(vec(f[i]);
        title="$host Histogram FDR for hit $i ($l fill)",
        xlabel="FDR value",
        ylabel="Counts (log10 scale)",
        formatter=:plain, legend=:topright,
        label="$l fill", lc=:match, yscale=:log10
    )|>display
end

p = plot([0,scsnrmax],[0,scsnrmax], primary=false, lc=:gray, xlims=(0,scsnrmax), widen=true, xlabel="Seticore SNR", ylabel="FDR SNR");
vline!(p, [snrthreshold], primary=false, lc=:gray, ls=:dash);
for (l,f) in snrplots
    scatter!(p, df.snr, float.(f); label=l)
end
display(p)

