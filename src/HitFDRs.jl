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

export loadhitsmetadata, loadhitsdata, loadhits, loadhitsfdrs
export calcfdr, driftfreq, driftrate, driftfreqrate, calcsnr
export plotspectrogram, plothist

SpectrogramDims = Tuple{Dim{:Frequency},Dim{:Time}}
AbstractDimSpectrogram = AbstractDimArray{T,2,<:SpectrogramDims} where T

FDRDims = Tuple{Dim{:Frequency},Dim{:DriftRate}}
AbstractDimFDR = AbstractDimArray{T,2,<:FDRDims} where T

##

function loadhitsmetadata(reader::CapnpReader)
    df = DataFrame(Iterators.map(NamedTuple, reader))
    df.fileidx = axes(df, 1)
    # Cheater way to get `nfpc` (number of fine channels per coarse channel)
    df.nfpc .= nextpow(2, maximum(df.index))
    df.fineChannel = df.coarseChannel .* df.nfpc .+ df.index;
    # Add dfdt and drstepn columns
    df.dfdt = 1e6 * df.foff ./ df.tsamp
    df.drstepn = df.dfdt ./ (nextpow.(2, df.numTimesteps).-1)
    # Add nint column
    df.nint = round.(Int, 1e6 .* df.foff .* df.tsamp)

    df
end

"""
    loadhitsmetadata(filename) -> DataFrame

Load all the `Hit` metadata from hits file `filename` and return as a
`DataFrame`.
"""
function loadhitsmetadata(filename::AbstractString)
    isfile(filename) || return DataFrame()
    stat(filename).size == 0 && return DataFrame()

    reader = CapnpReader(SeticoreCapnp.nodata_factory, Hit, filename)
        df = loadhitsmetadata(reader)
    finalize(reader)

    # Set hitsfile column
    df.hitsfile .= filename

    df
end

"""
    loadhits(filename; rescale=true) -> metadata, data

Load all the filterbank metadata and data matrices from hits file `filename` and
return them as `(DataFrame, Vector{DimMatrix})`.  The `DimMatrix` elements will
have frequency and time axes.  If `rescale` is `true`, then a scaling factor
will be computed from the metadata and the filterbank data values will all be
divided by `scaling`.  The default is to rescale.
"""
function loadhits(filename::AbstractString; rescale::Bool=true)
    isfile(filename) || return (DataFrame(), Matrix{Float32}[])
    stat(filename).size == 0 && return (DataFrame(), Matrix{Float32}[])

    reader = CapnpReader(Hit, filename)
        df = loadhitsmetadata(reader)
        # Calc scaling if rescale is true
        scaling = [1f0]
        if rescale
            scaling = Float32.(df.nfpc).^2 .* 4 .* df.nint
        end
        fbs = loadhitsdata(reader; scaling)
    finalize(reader)

    # Set hitsfile column
    df.hitsfile .= filename

    df, fbs
end

function loadhitsdata(reader::CapnpReader; scaling=1)
    # Load hits data
    fbs = map(reader) do hit
        # Create DimArry of hit's filterbank data
        fb = hit.filterbank
        freqs = range(fb.fch1, step=fb.foff, length=fb.numChannels)|>Dim{:Frequency}
        times = range(0.0, step=fb.tsamp, length=fb.numTimesteps)  |>Dim{:Time}
        DimArray(fb.data, (freqs, times))
    end

    # Scale hits data
    foreach(zip(fbs, Iterators.cycle(scaling))) do (d,s)
        if s != 1 && s != 0
            # Ensure that we don't scale by a wider float type
            typed_scaling = convert(eltype(d), s)
            d ./= typed_scaling
        end
    end

    fbs
end

"""
    loadhitsdata(filename; scaling=1) -> Vector{DimMatrix}

Load all the filterbank data matrices from hits file `filename` and return as a
`Vector{DimMatrix}` with frequency and time axes.  If `scaling` is non-zero,
(and not 1), then the filterbank data values will all be divided by `scaling`.
The default is no scaling.  Note that `scaling` can be a scalar value to be
applied to all filterbank data or a list of values to scale each `DimMatrix`
independently.
"""
function loadhitsdata(filename; scaling::Real=1)
    isfile(filename) || return Matrix{Float32}[]
    stat(filename).size == 0 && return Matrix{Float32}[]

    reader = CapnpReader(Hit, filename)
        fbs = loadhitsdata(reader; scaling)
    finalize(reader)

    fbs
end

"""
    loadhits(filename; rescale=true, pad=Gamma) -> metadata, data, fdrs

Load all the filterbank metadata and data matrices from hits file `filename` and
compute the frequency drift rate (FDR) plane for each hit.  The results are
returned as `(filterbank, data, fdrs)`.  `data` and `fdrs` will be
`Vector[DimMatrix}`.  The `data` elements will have frequency and time axes.
The `fdrs` elements will have frequency and drift rate axes.

If `rescale` is `true` (the default), then a scaling factor will be computed
from the metadata and applied to the filterbank data values (before FDR
calculation).  The frequency drift rate matrices of each hit is computed using a
drift rate resolution that is calculated from that hit's metatdata.  Each FDR
will use the padding convention specified by `pad`, which defaults to `Gamma`
(i.e. pad using random samples from a Gamma distribution fitted to the input
data).  See `calcfdr` for more info.
"""
function loadhitsfdrs(filename; rescale::Bool=true, pad=Gamma)
    df, fbs = loadhits(filename; rescale)

    fdrs = calcfdr.(fbs, df.drstepn; pad, own=true);

    df.fdrsnr = calcsnr.(fdrs)
    freqs_drs = driftfreqrate.(fdrs)
    df.fdrfreq = first.(freqs_drs)
    df.fdrrate = last.(freqs_drs)

    # Move `hitsfile` column to the end
    select!(df, Cols(Not(:hitsfile), :hitsfile))

    df, fbs, fdrs
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

function calcfdr(spectrogram, δrn, pad=Gamma, own=false)
    fdrin, fdrws, fdrout = getzdtworkspace(spectrogram, δrn)
    padded_copyto!(fdrin, spectrogram, pad)
    zdtfdr!(fdrout, fdrws, fdrin)
    (own ? copy(fdrout) : fdrout), getrates(fdrws)
end

function calcfdr(spectrogram::AbstractDimSpectrogram, δrn; pad=Gamma, own=false)
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

function driftchans(hitmeta)
    nt = hitmeta[:numTimesteps]
    c1 = hitmeta[:index] - hitmeta[:startChannel] + 1
    c2 = c1 + hitmeta[:driftSteps] * nt / nextpow(2, nt)
    [c1, c2]
end

function driftfreqs(hitmeta)
    nt = hitmeta[:numTimesteps]
    f1 = hitmeta[:frequency]
    f2 = f1 + hitmeta[:driftRate] * hitmeta[:tsamp] * (nt-1) / 1e6
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

function plotspectrogram(hitmeta, spectrogram, fdr; extra_title="", kwargs...)
    # TODO Add DimArray types for spectrogram and fdr input?
    # TODO Get tlast from time dimension of spectrogram?
    tlast = hitmeta[:tsamp] * (hitmeta[:numTimesteps] - 1)
    sf1, sf2 = driftfreqs(hitmeta)
    p = heatmap(spectrogram';
        xticks=val(dims(spectrogram,1)[[1,end]]),
        yflip=true, tickdir=:out, kwargs...)
    plot!(p, [sf1, sf2], [0, tlast]; lw=1, la=0.5, widen=false, label="SC")
    fdrf1, fdrrate = driftfreqrate(fdr)
    fdrf2 = fdrf1 + fdrrate * tlast / 1e6
    fdrf2 = clamp(fdrf2, extrema(dims(fdr,1))...)
    plot!(p, [fdrf1, fdrf2], [0, tlast]; lw=1, la=0.5, ls=:dash, widen=false, label="FDR")
    hm = heatmap(fdr';
        xticks=val(dims(fdr,1)[[1,end]]), tickdir=:out
    )
    plot(p, hm; layout=(1,2), widen=false,
        plot_title="Hit $(get(hitmeta,:idx,hitmeta[:fileidx]))$(extra_title)"
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