# HitFDRs

A package for calculating and plotting the *frequency-drift-rate* (FDR) matrix
corresponding to a *Hit* in a Cap'n Proto file.

## Loading hits

A `Hit` contains metadata about a Doppler drifting signal that had a *signal to
noise ratio* (SNR) above a certain threshold.  A `Hit` also contains a small
snippet of spectrogram data (aka *filterbank data*) the spans the frequency and
time range of the drifting signal.  A *hits file* is a Cap'n Proto file that
contains multiple `Hit`s.  `HitFDRs` provides the following functions to load
the metadata and spectrogram snippets of a hits file:

---
`loadhitsdf(filename) -> DataFrame`

This function loads the metadata of all the `Hit`s in the hits file named by
`filename` and returns a `DataFrame` containing one row per `Hit`.

---
`loadhitsdata(filename, scaling::Real=1) -> Vector{DimMatrix}`

This function loads the filterbank data of all the `Hit`s in the hits file named
by `filename`, returning a Vector of matrices, one matrix of filterbank per
`Hit`.  Each returned matrix is actually a `DimMatrix`, which is a matrix with
dimensional axes that specify the frequencies (in MHz) and relative time (in
seconds) of the frequency and time axes of the filterbank data matrix.  This
integrates nicely with plotting packages and also the calculation of the FDR
matrix (which is also a `DimMatrix`, but with frequency and drift rate axes).

If the `scaling` parameter is not equal to 1, the filterbank data values will
be divided by the `scaling` value.  Rescaling can help avoid floating point
computation errors when processing filterbank data that contain large values.

---
`loadhitsfb(filename, rescale::Bool=true) -> (DataFrame, Vector{DimMatrix})`

This function returns both the `DataFrame` of hits metadata that `loadhitsdf`
returns and the filterbank data that `loadhitsdata` returns.  If the `rescale`
parameter is `true`, then the filterbank data will be rescaled by an amount
determined from the `Hit` metadata.

## Calculating the FDR matrix

`calcfdr(spectrogram, δrn, pad=median; own=false) -> FDR`

Each pixel in a frequency drift rate matrix represents the integrated power of
a single drift line for a specific starting frequency and drift rate.
Conceptually, the integrated power for all drift lines of a certain drift rate
can be calculated by shifting the spectrum at each time step by a `time *
drift_rate` amount such that the drift lines for each starting channel become
aligned in frequency with the starting channel.  Then the shifted spectrogram
can be summed along the time axis to produce the FDR points of the given drift
rate.  Repeating this process for all drift rates gives the FDR matrix.  The two
axes of the FDR matrix are starting frequency and drift rate.

The `calfdr` function calculates the FDR matrix for a spectrogram using the ZDT
algorithm implemented in the `FrequencyDriftRateTransforms` package.  `calcfdr`
uses the drift rates corresponding to the diagonals of the spectrogram as the
minimum and maximum drift rates.  The drift rate resolution parameter, `δrn`,
specifies the step size that will be used to evenly divide that range to
determine drift rate axis of the FDR matrix.  Note that the `δrn` parameter
should be normalized by the spectrogram's channel width and time step duration
to give the drift rate resolution in `channels per time step` and will almost
certainly be non-integer.  The use of the spectrogram's minimum and maximum
drift rates becomes impractical as these limits grow, but for the spectrograms
of `Hit`s this is not a problem (for wider bandwidth spectrograms, use
`FrequencyDriftRateTransforms` directly).

The `pad` option specifies how `calcfdr` should synthesize the virtual samples
that are shifted into the spectrogram when (conceptually) frequency aligning the
drift lines.  Three different modes of padding can be used.

- Padding by function

  Passing a function for `pad` will use the return value of `pad(spectrogram)`
  for all the virtual samples.  Some functions that one might want to consider
  are `median` to use the median, `mean` to use the mean, or `zero∘eltype`
  (which is equivalent to the anonymous function `s->zero(eltype(s))`) to use
  zero.

- Padding by sampling

  When `pad` is given as an array of values the virtual samples will be randomly
  taken from that array.  One common use of this mode is to pass `spectrogram`
  for `pad` so that the virtual samples are randomly taken from the same set of
  values as the input itself.

- Padding by `Distribution`

  When `pad` is given as a `Distribution` (from the `Distributions` package),
  the virtual samples will be taken from that `Distribution`.  This can be given
  as a `Distribution` instance, e.g. `Normal(0,1)` to use a standard normal
  distribution, or a `Distribution` type, e.g. `Normal` or `Gamma`.  When `pad`
  is given as a `Distribution` type, an instance of that type will be obtained
  by calling `fit(pad, spectrogram)`.
  
Differently sized spectrograms require differently sized resources to calculate
the FDR matrix.  These resources are cached and reused for spectrograms of the
same size.  This includes the FDR matrix itself.  Passing `own=true` to
`calcfdr` will return a copy of the FDR matrix (which will not be overwritten by
future `calcfdr` calls) rather than the internally cached FDR matrix (which may
be overwritten by future `calcfdr` calls).

## Getting FDR properties

    driftfreq(fdr) -> starting_frequency
    driftrate(fdr) -> drift_rate
    driftfreqrate(fdr) -> (starting_frequency, drift_rate)

For any `DimArray` with frequency and drift rates axes (e.g. an FDR returned by
`calcfdr`), the starting frequency and drift rate (or both) corresponding to the
peak value of the FDR matrix can be obtained by calling one of these functions.


## Calculating SNR

    calcsnr(x; normalize=true) -> SNR

The SNR of array `x` is computed here as:

    snr = (peak_value(x) - median(x)) / mad(x; normalize)

See the documentation for `mad` for more information about the `normalize`
option.

## Plotting spectrograms and FDRs

    plotspectrogram(spectrogram, dfrow, fdr; extra_title="", kwargs...)

This function plots `spectrogram` and `fdr` as side-by-side heatmaps with
drift lines corresponding to the `Hit` metadata (from `dfrow`) and the FDR peak
overlaid on the `spectrogram`.  `dfrow` can be a `DataFrameRow` (or any other
`Hit` metadata container that can be indexed by `Symbol`, e.g. `:frequency`).

---
    function plothist(d; nbins=200, kwargs...)

`plothist` is a simple utility function that fits a histogram with `nbins` bins
to `d`, scatter plots the weights, then overlays a plot of the *probability
density function* of a Gamma distribution fitted to `d`.
