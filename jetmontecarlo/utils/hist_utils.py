import numpy as np
import scipy.fftpack
from scipy import interpolate

from warnings import warn

def histDerivative(hist, bins, giveHist=False, binInput='lin'):
    """Takes in a histogram assuming linear/uniform bins.
    Returns interpolating function for derivative of the hist
    up to accuracy (deltabin)^2.
    Uses a forward difference scheme in the first bin,
    a central difference scheme in the 'bulk' bins,
    and a backward difference scheme in the last bin.
    See, for example,
    https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
    #page=2&zoom=200,0,560

    Note that even if the bins are logarithmically input,
    this still returns a linear derivative of the histogram.
    For example, if the histogram is a set of positions in
    units of meters, and the bin centers correspond to logs
    of times in seconds at which those positions were measured,
    this method will still return a velocity.
    binInput must be set to 'log' to use logarithically
    spaced bins.

    Parameters
    ----------
    hist : array
        The set of y-values of an input histogram, for which we
        want an approximate numerical derivative.
    bins : array
        A set of bin **edges** for the histogram with y-values
        given by hist.
    giveHist : bool
        Determines whether to give a histogram corresponding
        to the derivative found by this method.
        If False, the method only returns an interpolating function.
    binInput : str
        A description of the spacing of the given bins
        ('lin' or 'log')

    Returns
    -------
    interpolate.interp1d (interpolate.interp1d, array)
        An interpolating function for the derivative of the
        histogram with the given bin edges.
        (An interpolating function for the derivative as well as
        its values in the bin centers, if giveHist is True)
    """
    if binInput == 'log':
        bins = np.log(bins)

    deltabin = bins[-1] - bins[-2]
    # Finding the bin width in the final bins, to accomodate underflow

    # Forward difference scheme in the center of the first bin
    firstBinVal = np.array([(-hist[2] + 4.* hist[1] - 3.*hist[0])
                            /(2.*deltabin)])
    # Central difference scheme in the bulk
    bulkBinVals = (hist[2:] - hist[:-2])/(2.*deltabin)
    # Backward difference scheme in the center of the final bin
    lastBinVal = np.array([(3.*hist[-1] - 4.*hist[-2] + hist[-3])
                           /(2.*deltabin)])

    if binInput == 'lin':
        # This is used if the input bins are linearly spaced
        xs = (bins[1:]+bins[:-1])/2.
        derivHist = np.concatenate((firstBinVal, bulkBinVals, lastBinVal))

    elif binInput == 'log':
        # This is used if the input bins are logarithmically spaced:
        # it returns a (linear) derivative w.r.t. the binned variable,
        # rather than its log.
        # First, take the log derivative, dY/dlogX = X dY/dX.
        # Then, divide by exp(logX) = X to get dY/dX.
        # Assuming the bins are already logX values,
        # and not logarithmically spaced X values
        xs = np.exp((bins[1:]+bins[:-1])/2.)
        derivHist = np.concatenate((firstBinVal, bulkBinVals,
                                    lastBinVal)) / xs

    else: raise AssertionError(
        "The style of the input bins must be either 'lin' or 'log'.")

    interp = interpolate.interp1d(x=xs, y=derivHist,
                                  fill_value="extrapolate")

    if giveHist:
        return interp, derivHist
    return interp


def vals_to_pdf(vals, num_bins, weights=None,
                bin_space='log', log_cutoff=None,
                make_finite=True):
    """Takes in a set of values and returns the associated
    xs and probability density.
    """
    if weights is None:
        weights = np.ones_like(vals)

    if make_finite:
        good_val_inds = np.isfinite(vals)
        good_weight_inds = np.isfinite(weights)
        good_inds = np.logical_and(good_val_inds, good_weight_inds)

        vals = vals[good_inds]
        weights = weights[good_inds]
        if not any(good_val_inds):
            warn("No finite values in `vals`, the values given to "
                 "vals_to_pdf. Returning empty arrays.")
            return np.array([]), np.array([])
        if not any(good_weight_inds):
            warn("No finite values in `weights`, the weights given to "
                 "vals_to_pdf. Returning empty arrays.")
            return np.array([]), np.array([])

    if bin_space == 'lin':
        bins = np.linspace(0, 1, num_bins)
        xs = (bins[:-1] + bins[1:])/2

    elif bin_space == 'log':
        assert log_cutoff is not None,\
            "log cutoff required for logarithmic"\
            " bins (you could try, say, -10)."
        bins = np.logspace(np.log10(log_cutoff),
                           0, num_bins)
        bins = np.insert(bins, 0, 1e-100)  # zero bin
        xs = np.sqrt(bins[1:-1] * bins[2:])
        xs = np.insert(xs, 0, 1e-50)  # zero bin

    pdf, _ = np.histogram(vals, bins, weights=weights)

    # Normalizing the pdf so its integral is 1
    dx = np.array(np.diff(bins), float)
    pdf = pdf / dx / pdf.sum()

    if bin_space == 'lin':
        pass
    elif bin_space == 'log':
        pdf = xs * pdf * np.log(10)

    return xs, pdf


def nan_info(array, name):
    """Print info associated with nans in an array."""
    print(f'nan {name}')
    print(len([a for a in array if np.isnan(a)]))


def smooth_data(x, y):
    """Smooths data using scipy functions"""
    N = len(y)
    assert len(x) == N, "x and y must be the same length."

    w = scipy.fftpack.rfft(y)
    spectrum = w**2

    cutoff_idx = spectrum < (spectrum.max()*5e-2)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    y2 = scipy.fftpack.irfft(w2)

    return y2
