import warnings
import random
import numpy as np
from scipy import interpolate
from scipy.misc import derivative
from pynverse import inversefunc

from jetmontecarlo.utils.interp_utils import lin_log_mixed_list

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
                     catch_turnpoint=False,
                     backup_cdf=None,
                     verbose=5):
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
    used_backup = False
    with warnings.catch_warnings():
        # Common but usually uniformative warning we would like to ignore:
        warnings.filterwarnings("ignore",
                message="Results obtained with less than 2 decimal"
                +" digits of accuracy")
        try:
            # Using pynverse's inversefunc method to sample from the CDF
            inv_cdf = inversefunc(cdf, domain=domain, image=[0,1])
        except ValueError:
            #==========================================================
            # Logging Errors and Correcting Common Use Cases
            #==========================================================
            # Getting a variety of points from the domain
            if domain[0] < 0:
                pnts = np.linspace(domain[0], domain[1], 1000)
            else:
                if domain[0] == 0:
                    pnts = lin_log_mixed_list(domain[0]+1e-20, domain[1], 1000)
                    pnts = np.array([0, *pnts])
                else:
                    pnts = lin_log_mixed_list(domain[0], domain[1], 1000)

            #----------------------------------------------------
            # If it is always 1, simply return zeros (the zero bin)!
            #----------------------------------------------------
            cdf_vals = np.nan_to_num(cdf(pnts))
            if all(cdf_vals == 1.):
                if verbose > 4:
                    print("CDF always 1. Returning zero bin.")
                return np.zeros(num_samples), np.ones_like(num_samples)

            # Otherwise, find where it is not monotone
            monotone = cdf_vals[:-1] <= cdf_vals[1:]

            bad_xvals_low = np.array([pnts[i] for i in range(len(monotone))
                                      if not monotone[i]])
            bad_xvals_high = np.array([pnts[i+1] for i in range(len(monotone))
                                       if not monotone[i]])
            bad_cdf_low = cdf(np.unique(bad_xvals_low))
            bad_cdf_high = cdf(np.unique(bad_xvals_high))


            # Another situation I've run into is that the CDF is monotone,
            # but close to zero everywhere in the domain, or a part of
            # the domain.
            # We can solve this type of issue by removing the "bad" part
            # of the domain:
            cdf_threshold = 1e-20

            # * if the CDF is _always_ miniscule, I return the highest
            #   value in the domain:
            if all(cdf_vals < cdf_threshold):
                if verbose > 4:
                    print("CDF always negligible. Returning highest point in domain.")
                return np.full(num_samples, domain[1]), np.ones_like(num_samples)

            # * if the CDF is miniscule up to some point in the domain, but
            #   monotone increasing afterwards, I remove the "bad" part
            #   from the domain and try again
            for i, cdf_val, point in enumerate(zip(cdf_vals, pnts)):
                # If we find a point where the CDF is no longer
                # miniscule
                if cdf_val > cdf_threshold:
                    if not all(monotone[i:]):
                        # If the CDF is not monotone after this point,
                        # this approach won't work
                        break
                    # If the CDF is monotone after this point, we can
                    # remove the "bad" part of the domain and try again!
                    if verbose > 4:
                        print("CDF always negligible up to a certain " +
                              "point. Removing 'bad' part from the domain.")
                    return samples_from_cdf(cdf, num_samples,
                                            domain=[point, domain[1]],
                                            catch_turnpoint=catch_turnpoint,
                                            backup_cdf=backup_cdf,
                                            verbose=verbose)

            #----------------------------------------------------
            # Verbose comments pointing out features of cdf
            if verbose > 2:
                print("Points from "+str(domain[0])+" to "+str(domain[1])
                      +": " + str(pnts))

            if verbose > 1:
                print("Found cdf of " + str(cdf(pnts[:-1]))+" at points "+str(pnts[:-1]))
                print("Found cdf of " + str(cdf(pnts[1:]))+" at points "+str(pnts[1:]))
                print("Testing monotonicity at these points in the requested domain:")
                print(monotone)

            if verbose > 2:
                print("Arguments where proposed cdf is not monotone: ")
                print(bad_xvals_low)
                print()
                print("Arguments i+1 where proposed cdf is not monotone: ")
                print(bad_xvals_high)

            # Finding the inconsistent cdf values
            if verbose > 0:
                print('[lower, higher] cdf_val where monotonicity is broken: '
                        + str([bad_cdf_low, bad_cdf_high]))
            #----------------------------------------------------

            #----------------------------------------------------
            # Other common use cases:
            #----------------------------------------------------
            # If the lowest fluctation is at the lower bound of the domain,
            # and the cdf appears to be 1 near there
            if bad_xvals_low[0] == pnts[0] and bad_cdf_low[0] == 1:
                # If the CDF starts out as 1 from the lowest point already,
                # inverse transform should yield all zeros
                if verbose>1:
                    print("Found non-monotone cdf behavior at minimum "
                          +"value of domain. Drawing from zero bin.")
                return np.zeros(num_samples), np.ones_like(num_samples)

            elif bad_cdf_low[0] == 1 and catch_turnpoint:
                # Otherwise, if the CDF reaches 1 at a particular location,
                # and we expect the CDF to be valid only up to that point
                # (where the CDF may `turn around' and start decreasing):
                if verbose>1:
                    print("Found turning point of CDF. Adjusting sampling.")
                return samples_from_cdf(cdf=cdf, num_samples=num_samples,
                                        domain=[domain[0],bad_xvals_low[0]],
                                        catch_turnpoint=catch_turnpoint,
                                        verbose=verbose)

            # After trying to catch the above cases, try to invert the
            # CDF again
            try:
                inv_cdf = inversefunc(cdf, domain=domain, image=[0,1])
            except ValueError as e:
                # If none of the above worked, try to use the backup CDF
                # and adjust the weighting scheme
                if backup_cdf is None:
                    raise e
                used_backup = True
                inv_cdf = inversefunc(backup_cdf, domain=domain, image=[0,1])

        rands = np.random.rand(num_samples)
        samples = inv_cdf(rands)

        if used_backup:
            dxs = [1e-6*min(domain[1]-sample, sample-domain[0])
                   for sample in samples]
            assert all(dx > 0 for dx in dxs)
            weights = np.array([ (derivative(cdf, sample, dx) /
                         derivative(backup_cdf, sample, dx))
                        for sample, dx in zip(samples, dxs)])
        else:
            weights = np.ones_like(samples)

        return samples, weights

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
