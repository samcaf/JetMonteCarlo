from __future__ import absolute_import
import numpy as np
import dill
from scipy.interpolate import griddata, interp2d, NearestNDInterpolator

import matplotlib.pyplot as plt

from sympy import Symbol, S
from sympy.calculus.util import continuous_domain

# Local utils:
from jetmontecarlo.montecarlo.integrator import *

# Local jet tools
from jetmontecarlo.jets.jetSamplers import *
from jetmontecarlo.jets.observables import *
from jetmontecarlo.jets.weights import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *


local_verbose = 1

def lin_log_mixed_list(lower_bound, upper_bound, num_bins):
    # Mixing linear and logarithmic bins for the list of thetas
    mixed_list = np.logspace(np.log10(lower_bound), np.log10(upper_bound),
                             int(num_bins/2)+2)
    mixed_list = np.append(mixed_list,
                           np.linspace(lower_bound, upper_bound,
                                       int(num_bins/2)))
    # Sorting and removing the duplicate values of upper and lower bound
    mixed_list = np.sort(mixed_list[1:-1])
    mixed_list = mixed_list[~np.isnan(mixed_list)]
    return mixed_list


def test_monotonicity(obj, domain=None):
    if hasattr(obj, '__iter__'):
        is_monotone = ((obj[1:] <= obj[:-1]).all() or
            (obj[1:] >= obj[:-1]).all())
        print(f"{is_monotone = }")
        return is_monotone
    elif hasattr(obj, '__call__'):
        assert domain is not None,\
            "Testing montonicity of a function requires a domain."
        xs = lin_log_mixed_list(domain[0], domain[1], 1000)
        vals = obj(xs)
        #print(xs)
        #print(vals)
        return test_monotonicity(vals)
        

###########################################
# Numerical Radiator Calculations:
###########################################
# ------------------------------------
# Radiator Generation:
# ------------------------------------
def gen_numerical_radiator(rad_sampler, emission_type,
                           jet_type='quark',
                           obs_accuracy='LL',
                           splitfn_accuracy='LL',
                           beta = None,
                           bin_space='lin',
                           fixed_coupling=True,
                           save=True,
                           num_bins=100,
                           force_monotone=False):
    """A function which takes in a sampler with generated data,
    and returns the associated numerically integrated radiator
    (dependent on a single parameter).

    Optionally, saves the radiator and the associated interpolation
    function.

    Parameters
    ----------
    rad_sampler : sampler
        A sampler class which has sampled over the phase space
        of a certain type of emission.
    emission_type : str
        The type of emission whose phase space we have sampled.
        Must be 'crit', 'sub', or 'pre'.
    jet_type : str
        The type of jet. Must be 'quark' or 'gluon'.
    obs_accuracy : str
        The accuracy at which we calculate the relevant observable.
        Must be 'LL' or 'MLL'.
    splitfn_accuracy : str
        The accuracy at which we calculate the relevant splitting function.
        Must be 'LL' or 'MLL'.
    fixed_coupling : bool
        A boolean  which determines whether the radiator is calculated
        using fixed coupling (True) or running coupling (False).
    save : bool
        A boolean which determines whether the radiator information,
        and the corresponding interpolation function, are saved.

    Returns
    -------
    xs, radiator, radiator_error
    float, float, float
        The xs, radiator values, and radiator errors obtained by
        numerical integration for the given emission and jet type.
    """
    # Verifying that the generation is proceeding with valid parameters
    assert (beta is None and not emission_type == 'sub')\
        or (beta is not None and emission_type == 'sub'),\
        str(emission_type)+' emissions must have a valid beta value.'
    # Setting up an integrator
    rad_integrator = integrator()

    # Integral is positive, monotonic, and is zero at the last bin
    rad_integrator.setLastBinBndCondition([0., 'plus'])
    rad_integrator.setMonotone()

    # Getting information from sampler
    samples = rad_sampler.getSamples()

    if emission_type == 'crit':
        z = samples[:, 0]
        theta = samples[:, 1]
        obs = theta
    elif emission_type == 'sub':
        z = samples[:, 0]
        theta = samples[:, 1]
        obs = C_ungroomed(z, theta, beta, acc=obs_accuracy)
    else:
        raise AssertionError("Invalid emission type. Emission must be "
                             + "'crit', 'sub', or 'pre', but is"
                             + str(emission_type))

    # Weights, binned observables, and area
    rad_integrator.setBins(num_bins, samples, bin_space)
    weights = radiatorWeight(z, theta, jet_type,
                             fixedcoupling=fixed_coupling,
                             acc=splitfn_accuracy)
    jacs = rad_sampler.jacobians
    area = rad_sampler.area

    # Performing integration
    rad_integrator.setDensity(obs, weights * jacs, area)
    rad_integrator.integrate()

    radiator = rad_integrator.integral
    radiator_error = rad_integrator.integralErr
    xs = rad_integrator.bins[:-1]

    # DEBUG: testing monotonicity
    if local_verbose >= 2:
        print("crit ints:")
        test_monotonicity(radiator)

    # Reminding the radiator where it needs to stop
    if emission_type == 'crit':
        bounds = [1e-15, 1]
    if emission_type == 'sub':
        bounds = [1e-15, .5]
    radiator = np.append(radiator, 0)
    radiator_error = np.append(radiator_error, 0)
    xs = np.append(xs, bounds[1])

    rad_integrator.integral = radiator
    rad_integrator.integralErr = radiator_error
    rad_integrator.bins = np.append(xs, bounds[1])

    # Generating an interpolating function
    rad_integrator.makeInterpolatingFn()

    unbounded_interp_function = rad_integrator.interpFn

    # Checking the interpolating function against our numerics
    # assert(unbounded_interp_function(xs) == radiator).all(),\
    #     "The interpolating radiator is not the same as the radiator "\
    #     +"obtained with numerical integration."

    # Bounding the interpolating function
    def bounded_interp_function(x):
        rad = (bounds[0] < x) * (x < bounds[1]) * unbounded_interp_function(x)
        return rad

    rad_integrator.interpFn = bounded_interp_function

    # DEBUG: testing monotonicity
    if local_verbose >= 1:
        print("crit interps:")
        test_monotonicity(bounded_interp_function, bounds)

    # Saving data and interpolating function
    if save:
        extra_info = ''
        if fixed_coupling:
            extra_info = 'fc_'
        else:
            extra_info = 'rc_'
        if emission_type == 'crit':
            extra_info += 'zc_' + str(rad_sampler.zc)
        elif beta is not None:
            extra_info += 'beta_' + str(beta) + '_'
        filename = 'radiator_'+jet_type+'_'+emission_type\
                   +'_obs'+obs_accuracy+'_splitfn'+splitfn_accuracy\
                   +'_'+str(len(samples))+extra_info\
                   +'_samples.py'
        rad_integrator.saveInterpolatingFn(filename)

    return rad_integrator.interpFn

def gen_pre_num_rad(rad_sampler, crit_rad_sampler,
                    jet_type='quark',
                    obs_accuracy='LL', splitfn_accuracy='LL',
                    bin_space='lin',
                    fixed_coupling=True,
                    num_bins=100,
                    force_monotone=False):
    """A function which takes in a sampler with generated data,
    and returns the associated numerically integrated pre-critical
    radiator (dependent on two parameters).

    More precisely, the pre-critical radiator may be used to find
    the conditional cumulative distribution for the energy fraction z_pre
    of an emission which occurs before a `critical' emission at angle
    theta_crit:
    Sigma(z_pre | theta_crit) = exp[-R(z_pre, theta_crit)]

    gen_pre_num_rad also saves the radiator and the associated interpolation
    function.

    Parameters
    ----------
    rad_sampler : sampler
        A sampler class which has sampled over the phase space
        of pre-critical emissions.
    crit_rad_sampler : sampler
        A sampler class which has sampled over a different phase space,
        corresponding to the second parameter of the radiator.
        This is called crit_rad_sampler because for now (and for all
        cases I can imagine) we will be using this to sample over
        critical emission angles.
    jet_type : str
        The type of jet. Must be 'quark' or 'gluon'.
    accuracy : str
        The accuracy at which we calculate the relevant observable.
        Must be 'LL' or 'MLL'.
    fixed_coupling : bool
        A boolean  which determines whether the radiator is calculated
        using fixed coupling (True) or running coupling (False).
    save : bool
        A boolean which determines whether the radiator information,
        and the corresponding interpolation function, are saved.

    Returns
    -------
    function
        A 2d interpolating function for the pre-critical radiator.
    """
    # Setting up an integrator
    rad_integrator = integrator_2d()

    # Integral is positive, and is zero at the last bin
    rad_integrator.setLastBinBndCondition([0., 'plus'])

    # Getting information from the critical sampler
    theta_crit = crit_rad_sampler.getSamples()[:, 1]
    z_em = rad_sampler.getSamples()

    # Weights, binned observables, and area
    obs = [z_em, theta_crit]

    rad_integrator.setBins(num_bins, obs, bin_space)

    weights = radiatorWeight(z_em, theta_crit, jet_type,
                             fixedcoupling=fixed_coupling,
                             acc=splitfn_accuracy)

    # Accounting for the additional integration over critical energy fractions
    if bin_space == 'lin':
        area = rad_sampler.zc
        jacs = np.ones(len(theta_crit))
    else:
        area = np.log(1./rad_sampler.epsilon)\
               * np.log(1./crit_rad_sampler.epsilon)
        jacs = z_em * theta_crit

    # Performing integration
    rad_integrator.setDensity(obs, weights * jacs, area)
    rad_integrator.integrate()

    radiator = rad_integrator.integral
    radiator_error = rad_integrator.integralErr

    # DEBUG: testing monotonicity
    if local_verbose >= 2:
        print("pre ints:")
        for rad1d in radiator:
            test_monotonicity(rad1d)

    # Generating an interpolating function
    rad_integrator.makeInterpolatingFn()

    unbounded_interp_function = rad_integrator.interpFn

    # Bounding the interpolating function
    def bounded_interp_function(x, theta):
        # Pre-critical emission boundaries
        rad = ((0 <= x) * (x <= rad_sampler.zc) * (0 <= theta)
               * unbounded_interp_function(x, theta))
        return rad

    # DEBUG: testing monotonicity
    if local_verbose >= 1:
        print("pre interps:")
        for theta in lin_log_mixed_list(np.min(theta_crit), np.max(theta_crit), 250):
            rad1d_fn = lambda x: bounded_interp_function(x, theta)
            test_monotonicity(rad1d_fn, [1e-15, rad_sampler.zc])

    rad_integrator.interpFn = bounded_interp_function

    return rad_integrator.interpFn

def gen_crit_sub_num_rad(rad_sampler,
                         jet_type='quark',
                         obs_accuracy='LL', splitfn_accuracy='LL',
                         beta=2., epsilon=1e-15,
                         bin_space='log',
                         fixed_coupling=True,
                         num_bins=1000,
                         force_monotone=False):
    """A function which takes in a sampler with generated data,
    and returns the associated numerically integrated radiator
    dependent on the variables over a sampled phase space as
    well as angles theta of associated critical emissions.

    Saves the radiator and the associated interpolation
    function.

    Parameters
    ----------
    rad_sampler : sampler
        A sampler class which has sampled over the phase space
        of a certain type of emission.
    jet_type : str
        The type of jet. Must be 'quark' or 'gluon'.
    accuracy : str
        The accuracy at which we calculate the relevant observable.
        Must be 'LL' or 'MLL'.
    fixed_coupling : bool
        A boolean  which determines whether the radiator is calculated
        using fixed coupling (True) or running coupling (False).

    Returns
    -------
    function
        A 2d interpolating function for the critical-subsequent radiator.
    """
    # Preparing lists to hold all radiator and angle data
    rads_all = []
    rad_error_all = []
    xs_all = []
    thetas_all = []

    # Getting information from sampler
    samples = rad_sampler.getSamples()

    # Preparing a list of theta_crits on which our radiator will depend:
    theta_calc_list = lin_log_mixed_list(epsilon, 1., num_bins)

    # Preparing observables from the subsequent sampler
    z_em = samples[:, 0]
    theta_samp = samples[:, 1]

    for i, theta_crit in enumerate(theta_calc_list):
        # Setting up an integrator
        rad_integrator = integrator()

        # Integral is positive, monotone, and is zero at the last bin
        rad_integrator.setLastBinBndCondition([0., 'plus'])
        rad_integrator.setMonotone()

        # Preparing to integrate over the sampled phase space
        jacs = rad_sampler.jacobians
        area = rad_sampler.area

        # Rescaling the emission angles relative to the sampled angles:
        theta_em = theta_samp * theta_crit

        if bin_space == 'lin':
            area = area * theta_crit
        if bin_space == 'log':
            jacs = np.array(jacs) * theta_crit

        obs = C_ungroomed(z_em, theta_em, beta, acc=obs_accuracy)

        # Weights, binned observables, and area
        rad_integrator.setBins(num_bins, obs, bin_space,
                               min_log_bin=theta_crit**beta*1e-15)

        weights = radiatorWeight(z_em, theta_em, jet_type,
                                 fixedcoupling=fixed_coupling,
                                 acc=splitfn_accuracy)

        # Performing integration
        rad_integrator.setDensity(obs, weights * jacs, area)
        rad_integrator.integrate()

        # Radiator, given a maximum angle of theta
        radiator = rad_integrator.integral
        # radiator_error = rad_integrator.integralErr
        xs = rad_integrator.bins[:-1]

        # DEBUG: testing monotonicity
        if local_verbose >= 3:
            print("crit sub integral 0:")
            test_monotonicity(radiator)

        radiator = np.append(radiator, 0)
        # radiator_error = np.append(radiator_error, 0)
        xs = np.append(xs, C_ungroomed_max(beta, radius=theta_crit,
                                           acc=obs_accuracy))

        # DEBUG: testing monotonicity
        if local_verbose >= 2:
            print("crit sub int:")
            test_monotonicity(radiator)

        # Saving the function/radiator values, bin edges, and theta value
        rads_all.append(np.array(radiator))
        # rad_error_all.append(np.array(radiator_error))

        xs_all.append(np.array(xs))
        thetas_all.append(np.ones(len(xs))*theta_crit)

    xs_all = np.array(xs_all)
    thetas_all = np.array(thetas_all)
    rads_all = np.array(rads_all)

    points = np.array([xs_all.flatten(), thetas_all.flatten()]).T
    unbounded_interp_function = NearestNDInterpolator(points, rads_all.flatten())

    def bounded_interp_function(x, theta):
        # Subsequent emission boundaries
        bnds = (x <= C_ungroomed_max(beta, radius=theta, acc=obs_accuracy))
        rad = ((0 <= x) * bnds * (0 <= theta)
               * unbounded_interp_function(x, theta))
        return rad

    # DEBUG: testing monotonicity
    if local_verbose >= 1:
        print("crit sub interps:")
        for theta in lin_log_mixed_list(np.min(theta_crit), np.max(theta_crit), 250):
            rad1d_fn = lambda x: bounded_interp_function(x, theta)
            test_monotonicity(rad1d_fn, [1e-15, C_ungroomed_max(beta, radius=theta,
                                                            acc=obs_accuracy)])

    return bounded_interp_function

###########################################
# Splitting Functions:
###########################################
# Generation of Normalizing Factors for Splitting Functions:
def gen_normalized_splitting(num_samples, z_cut,
                             jet_type='quark', accuracy='LL',
                             fixed_coupling=True,
                             bin_space='lin', epsilon=1e-15, num_bins=100):
    # Preparing a list of thetas, and normalizations which will depend on theta
    theta_calc_list, norms = lin_log_mixed_list(epsilon, 1., num_bins), []

    for _, theta in enumerate(theta_calc_list):
        # Preparing the weight we want to normalize
        def weight(z):
            if fixed_coupling:
                alpha = alpha_fixed
            else:
                alpha = alpha_s(z, theta)
            return alpha * splittingFn(z, jet_type, accuracy)
        # Finding the normalization factor
        n, _, _ = integrate_1d(weight, [z_cut, 1./2.],
                               bin_space=bin_space, epsilon=epsilon,
                               num_samples=num_samples)
        norms.append(n)

    # Making an interpolating function for the splitting fn normalization
    normalization = interpolate.interp1d(x=theta_calc_list,
                                         y=norms,
                                         fill_value="extrapolate")

    def normed_splitting_fn(z, theta):
        if fixed_coupling:
            alpha = alpha_fixed
        else:
            alpha = alpha_s(z, theta)
        splitfn =  alpha*splittingFn(z, jet_type, accuracy)/normalization(theta)
        return splitfn * (z_cut < z) * (z < 1./2.)

    return normed_splitting_fn

