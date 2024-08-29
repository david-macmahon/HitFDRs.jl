module HitFDRs

using SeticoreCapnp
using DataFrames
using FrequencyDriftRateTransforms
using Memoize
using StatsBase
using StatsBase: normalize!
using DimensionalData
using Plots
using Distributions
using Distributions: rand!

#using LinearAlgebra # for `normalize`, which is extended by Plots and StatsBase
#using StatsPlots

export loadhitsdf, loadhitsfb, loadhitsdata
export calcfdr, driftfreq, driftrate, driftfreqrate, calcsnr
export plotspectrogram, plothist

SpectrogramDims = Tuple{Dim{:Frequency},Dim{:Time}}
AbstractDimSpectrogram = AbstractDimArray{T,2,<:SpectrogramDims} where T

FDRDims = Tuple{Dim{:Frequency},Dim{:DriftRate}}
AbstractDimFDR = AbstractDimArray{T,2,<:FDRDims} where T

##

function loadhitsdf(filename)
    isfile(filename) || return DataFrame()
    stat(filename).size == 0 && return DataFrame()

    reader = CapnpReader(SeticoreCapnp.nodata_factory, Hit, filename)
    df = DataFrame(Iterators.map(NamedTuple, reader))
    finalize(reader)
    df.idx = axes(df, 1)
    # Cheater way to get `nfpc` (number of fine channels per coarse channel)
    nfpc = nextpow(2, maximum(df.index))
    df.fineChannel = df.coarseChannel .* nfpc .+ df.index;
    # Set host column if filename contains a `blpn[0-9]+`
    if contains(filename, r"blpn[0-9]+")
        df.host .= replace(filename, r".*(blpn[0-9]+).*"=>s"\1")
    end
    df
end

function loadhitsfb(filename, rescale::Bool=true)
    isfile(filename) || return (DataFrame(), Matrix{Float32}[])
    stat(filename).size == 0 && return (DataFrame(), Matrix{Float32}[])

    reader = CapnpReader(Hit, filename)
    df = DataFrame()
    fbs = map(reader) do hit
        push!(df, NamedTuple(hit))
        # Create DimArry of hit's filterbank data
        fb = hit.filterbank
        freqs = range(fb.fch1, step=fb.foff, length=fb.numChannels)|>Dim{:Frequency}
        times = range(0.0, step=fb.tsamp, length=fb.numTimesteps)  |>Dim{:Time}
        DimArray(fb.data, (freqs, times))
    end
    finalize(reader)

    # Add auxilliary columns
    df.idx = axes(df, 1)
    # Cheater way to get `nfpc` (number of fine channels per coarse channel)
    nfpc = nextpow(2, maximum(df.index))
    df.fineChannel = df.coarseChannel .* nfpc .+ df.index;
    # Set host column if filename contains a `blpn[0-9]+`
    if contains(filename, r"blpn[0-9]+")
        df.host .= replace(filename, r".*(blpn[0-9]+).*"=>s"\1")
    end

    if rescale
        foff = df.foff[1]
        tsamp = df.tsamp[1]
        nsti = round(Int, 1e6 * foff * tsamp)
        scaling = Float32(nfpc)^2 * 4 * nsti
        scaling != 0 && foreach(d->d./=scaling, fbs)
    end

    df, fbs
end

function loadhitsdata(filename, scaling::Real=1)
    isfile(filename) || return Matrix{Float32}[]
    stat(filename).size == 0 && return Matrix{Float32}[]

    reader = CapnpReader(Hit, filename)
    fbs = map(reader) do hit
        # Create DimArry of hit's filterbank data
        fb = hit.filterbank
        freqs = range(fb.fch1, step=fb.foff, length=fb.numChannels)|>Dim{:Frequency}
        times = range(0.0, step=fb.tsamp, length=fb.numTimesteps)  |>Dim{:Time}
        DimArray(fb.data, (freqs, times))
    end
    finalize(reader)

    if scaling != 1 && scaling != 0 && !isempty(fbs)
        # Ensure that we don't scale by a wider float type
        typed_scaling = convert(eltype(fbs[1]), scaling)
        foreach(d->d./=typed_scaling, fbs)
    end

    fbs
end

@memoize function getzdtworkspace(T::Type, nf, nt, r0n, δrn, Nr)
    spectrogram = zeros(T, nf, nt)
    zdtws = ZDTWorkspace(spectrogram, r0n, δrn, Nr)
    spectrogram, zdtws, zeros(T, zdtws.Nf, zdtws.Nr)
end

function getzdtworkspace(spectrogram, δrn, T::Type=eltype(spectrogram))
    nf, nt = size(spectrogram)
    rmaxn = (nf-1) / (nt-1)
    # TODO Make sure this doesn't drift too far
    Nr = Int(fld(rmaxn, abs(δrn)))
    r0n = -Nr * abs(δrn)
    # Make fdr input be twice as wide for zero padding to avoid "wrap around"
    getzdtworkspace(T, nextprod((2,3,5), 2nf), nt, r0n, δrn, 2Nr+1)
end

# TODO Move to FrequencyDriftRateTransforms
function getrates(zdtws, dfdt=1)
    range(zdtws.r0*dfdt, step=zdtws.δr*dfdt, length=zdtws.Nr)
end

function padded_copyto!(dst, src, paddist::Distribution)
    overlap = intersect(CartesianIndices(src), CartesianIndices(dst))
    extra_chans = CartesianIndices((size(src,1)+1:size(dst,1), axes(src,2)))
    extra_times = CartesianIndices((axes(dst,1), size(src,2)+1:size(dst,2)))
    dst[overlap] .= src[overlap]
    rand!(paddist, @view dst[extra_chans])
    dst[extra_times] .= 0
    dst
end

function padded_copyto!(dst, src, paddisttype::Type{<:Distribution})
    padded_copyto!(dst, src, fit(paddisttype, src))
end

function padded_copyto!(dst, src, paddata::AbstractArray)
    overlap = intersect(CartesianIndices(src), CartesianIndices(dst))
    extra_chans = CartesianIndices((size(src,1)+1:size(dst,1), axes(src,2)))
    extra_times = CartesianIndices((axes(dst,1), size(src,2)+1:size(dst,2)))
    dst[overlap] .= src[overlap]
    dst[extra_chans] .= (rand(paddata) for _ in extra_chans)
    dst[extra_times] .= 0
    dst
end

function padded_copyto!(dst, src, padfunc::Function)
    overlap = intersect(CartesianIndices(src), CartesianIndices(dst))
    extra_chans = CartesianIndices((size(src,1)+1:size(dst,1), axes(src,2)))
    extra_times = CartesianIndices((axes(dst,1), size(src,2)+1:size(dst,2)))
    dst[overlap] .= src[overlap]
    dst[extra_chans] .= padfunc(src)
    dst[extra_times] .= 0
    dst
end

function calcfdr(spectrogram, δrn, pad=median; own=false)
    fdrin, fdrws, fdrout = getzdtworkspace(spectrogram, δrn)
    padded_copyto!(fdrin, spectrogram, pad)
    zdtfdr!(fdrout, fdrws, fdrin)
    (own ? copy(fdrout) : fdrout), getrates(fdrws)
end

function calcfdr(spectrogram::AbstractDimSpectrogram, δrn, pad=median; own=false)
    fdrin, fdrws, fdrout = getzdtworkspace(spectrogram, δrn)
    padded_copyto!(fdrin, spectrogram, pad)
    zdtfdr!(fdrout, fdrws, fdrin)
    freqs, times = dims(spectrogram)
    dfdt = 1e6 * step(freqs) / step(times)
    rates = Dim{:DriftRate}(getrates(fdrws, dfdt))
    fdrvw = view(fdrout, axes(parent(spectrogram),1), :)
    fdr = (own ? copy(fdrvw) : fdrvw)
    DimArray(fdr, (freqs, rates))
end

function driftchans(dfrow::DataFrameRow)
    nt = dfrow[:numTimesteps]
    c1 = dfrow[:index] - dfrow[:startChannel] + 1
    c2 = c1 + dfrow[:driftSteps] * nt / nextpow(2, nt)
    [c1, c2]
end

function driftfreqs(dfrow::DataFrameRow)
    nt = dfrow[:numTimesteps]
    f1 = dfrow[:frequency]
    f2 = f1 + dfrow[:driftRate] * dfrow[:tsamp] * (nt-1) / 1e6
    [f1, f2]
end

function driftchans(fdr::Matrix, steps)
    ij = findmax(fdr)[2]
    c1, ridx = Tuple(ij)
    c2 = c1 + steps[ridx]
    [c1, c2]
end

function driftchans(fdr::Matrix, rates, nsamp)
    driftchans(fdr, rates.*(nsamp-1))
end

function driftfreq(fdr::AbstractDimFDR)
    peakij = findmax(fdr)[2]
    dims(fdr,1)[peakij[1]]
end

function driftrate(fdr::AbstractDimFDR)
    peakij = findmax(fdr)[2]
    dims(fdr,2)[peakij[2]]
end

function driftfreqrate(fdr::AbstractDimFDR)
    peakij = findmax(fdr)[2]
    getindex.(dims(fdr), Tuple(peakij))
end

function plotspectrogram(spectrogram, dfrow, fdr; extra_title="", kwargs...)
    # TODO Add DimArray types for spectrogram and fdr input?
    # TODO Get tlast from time dimension of spectrogram?
    tlast = dfrow[:tsamp] * (dfrow[:numTimesteps] - 1)
    sf1, sf2 = driftfreqs(dfrow)
    p = heatmap(spectrogram';
        xticks=val(dims(spectrogram,1)[[1,end]]),
        yflip=true, tickdir=:out, kwargs...)
    plot!(p, [sf1, sf2], [0, tlast]; lw=1, la=0.5, widen=false, label="SC")
    fdrf1, fdrrate = driftfreqrate(fdr)
    fdrf2 = fdrf1 + fdrrate * tlast / 1e6
    plot!(p, [fdrf1, fdrf2], [0, tlast]; lw=1, la=0.5, ls=:dash, widen=false, label="FDR")
    hm = heatmap(fdr';
        xticks=val(dims(fdr,1)[[1,end]]), tickdir=:out
    )
    plot(p, hm; layout=(1,2), widen=false,
        plot_title="$(dfrow[:host]): Hit $(dfrow[:idx])$(extra_title)"
    )
end

function calcsnr(x; normalize=true)
    pkval = findmax(x)[1]
    m = median(x)
    s = mad(x; center=m, normalize)
    (pkval - m) / s
end

function fithist(d; nbins=200)
    fit(Histogram, vec(d); nbins) |> normalize!
end

function fithist(d::AbstractArray{<:AbstractFloat}; nbins=200)
    fit(Histogram, vec(d); nbins) |> normalize!
end

function plothist(d; nbins=200, kwargs...)
    h = fithist(d; nbins)
    g = fit(Gamma, vec(d))
    xx = midpoints(h.edges[1])
    @show length(xx) length(h.weights)
    p = scatter(xx, h.weights; label="Data", kwargs...)
    plot!(p, xx, pdf.(g, xx); label="Gamma fit")
end

end # module HitFDRs