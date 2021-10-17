import warnings
import random
import numpy as np
from scipy import interpolate
from pynverse import inversefunc

def getLinSample(sample_min, sample_max):
    """
    Parameters
    ----------
    sample_min : float
        Lower bound of the phase space sampling.
    sample_max : float
        Upper bound of the phase space sampling.

    Returns
    -------
    float
        A random number corresponding to a single sample,
        picked from a distribution which is uniform between
        sample_min and sample_max.
    """
    return sample_min + random.random()*(sample_max-sample_min)

def getLogSample(sample_min, sample_max, epsilon=1e-8):
    """
    Parameters
    ----------
    sample_min : float
        Lower bound of the phase space sampling.
    sample_max : float
        Upper bound of the phase space sampling.
    epsilon : float
        A lower cutoff on the logarithmic sampling.

    Returns
    -------
    float
        A random number corresponding to a single sample.
        In particular, generates a random number picked
        from a distribution which is log uniform between
        0 and sample_max-sample_min, then adds sample_min
        to this random number.

        The result is between sample_min and sample_max.
    """
    logsample = (np.log(epsilon)*random.random()
                 + np.log(sample_max-sample_min))
    return sample_min + np.exp(logsample)


def getLogSample_zerobin(sample_min, sample_max, cum_dist,
                         epsilon=1e-8):
    """
    Parameters
    ----------
    sample_min : float
        Lower bound of the phase space sampling.
    sample_max : float
        Upper bound of the phase space sampling.
    epsilon : float
        A lower cutoff on the logarithmic sampling.
    cum_dist : function
        A function corresponding to a CDF from which we
        want to sample and for which we want to consider
        zero binning.

    Returns
    -------
    float
        A random number corresponding to a single sample.
        In particular, generates a random number picked
        from a distribution which is log uniform between
        0 and sample_max-sample_min, then adds sample_min
        to this random number.

        The result is between sample_min and sample_max.
    """
    if random.random() < cum_dist(epsilon):
        return 1e-50

    logsample = (np.log(epsilon)*random.random()
                 + np.log(sample_max-sample_min))
    return sample_min + np.exp(logsample)

def samples_from_cdf(cdf, num_samples, domain=None,
                     monotone_verbose=0):
    """A function which takes in a functional form for a cdf,
    inverts it, and generates samples using the inverse transform
    method.

    Note that any cdf must be monotonic.

    Parameters
    ----------
    cdf : function
        A functional form for the cdf.
    num_samples : int
        Number of samples to be generated.
    domain : array (optional)
        An array containing the domain of the cdf

    Returns
    -------
    np.array
        An array of num_samples samples generated from the
        given cdf.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                message="Results obtained with less than 2 decimal"
                +" digits of accuracy")
        try:
            inv_cdf = inversefunc(cdf, domain=domain, image=[0,1])
        except ValueError:
            # If it is always 1, simply return zero!
            pnts = np.linspace(domain[0], domain[1], 1000)
            if all(cdf(pnts)==1):
                return np.zeros(num_samples)

            # Otherwise, find where it is not monotone
            monotone = cdf(pnts[:-1]) <= cdf(pnts[1:])

            # Verbose comments pointing out features of cdf
            if monotone_verbose > 1:
                print("Found cdf of " + str(cdf(pnts[:-1]))+"at"+str(pnts[:-1][0]))
                print("Found cdf of " + str(cdf(pnts[1:]))+"at"+str(pnts[1:][0]))
                print("Testing monotonicity in domain...")
                print(monotone)

            bad_xvals_low = np.array([pnts[i] for i in range(len(monotone))
                                      if not monotone[i]])
            bad_xvals_high = np.array([pnts[i+1] for i in range(len(monotone))
                                       if not monotone[i]])
            bad_cdf_low, bad_cdf_high = cdf(bad_xvals_low), cdf(bad_xvals_high)

            # If the lowest fluctation is at the lower bound of the domain,
            # and the cdf appears to be 1 near there
            if bad_xvals_low[0] == pnts[0] and bad_cdf_low[0] == 1:
                if monotone_verbose>1:
                    print("Found non-monotone cdf behavior at minimum "
                          +"value of domain. Drawing from zero bin.")
                return np.zeros(num_samples)

            # Otherwise, if the bad cdf appears to just be fluctuating around 1
            # when monotonicity fails, we can ignore the region where the
            # bad fluctuations, which are presumably numerical artifacts, occur.
            # We can then do the same inverse transform method to sample from
            # the cdf in the valid region of the domain.

            # Finding the inconsistent cdf values
            if monotone_verbose > 0:
                print('[lower, higher] cdf_val where monotonicity is broken: '
                        + str([bad_cdf_low, bad_cdf_high]))

            if len(bad_cdf_low) > 0:
                if bad_cdf_low[0] == 1:
                    # Finding the highest point above which there are no
                    # bad cdf fluctuations:
                    high_good_xval = domain[1]

                    while high_good_xval == domain[1]:
                        if not monotone[i+1]:
                            high_good_xval = pnts[i]
                        i += 1
                        if i > len(monotone)-1:
                            raise AssertionError("Function is not strictly"+
                                "monotonic. Unable to correct this feature.")

                    if monotone_verbose > 0:
                        print("Found numerical fluctuations."
                              +"Correcting recursively...")

                    # Sample from the valid/consistent phase space region
                    return samples_from_cdf(cdf, num_samples,
                                            domain=[domain[0], high_good_xval],
                                            monotone_verbose=monotone_verbose)


            inv_cdf = inversefunc(cdf, domain=domain, image=[0,1])

        rands = np.random.rand(num_samples)
        samples = inv_cdf(rands)
        return samples

def inverse_transform_samples(cdf_vals, x_vals, num_samples):
    """A function which takes in a set of cdf values and associated x
    values, and samples from the corresponding pdf using inverse transform
    sampling and interpolating functions.

    Parameters
    ----------
    cdf : array
        Set of cdf values.
    xs : array
        Set of corresponding x values.
    num_samples : type
        Number of samples to generate

    Returns
    -------
    np.array
        Set of num_samples samples generated by the inverse transform
        method to match the given cdf and x values.
    """
    inv_cdf = interpolate.interp1d(cdf_vals, x_vals, fill_value='extrapolate')
    r = np.random.rand(num_samples)
    return inv_cdf(r)
