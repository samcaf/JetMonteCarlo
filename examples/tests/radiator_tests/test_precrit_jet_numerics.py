from __future__ import absolute_import
import numpy as np
import dill
from scipy.interpolate import griddata, interp2d, NearestNDInterpolator

import matplotlib.pyplot as plt
from matplotlib import cm

# Local utils:
from jetmontecarlo.montecarlo.integrator import *

# Local jet tools
from jetmontecarlo.numerics.radiators.samplers import *
from jetmontecarlo.numerics.observables import *
from jetmontecarlo.numerics.weights import *

# Local analytics
from jetmontecarlo.analytics.radiators.running_coupling import *
from jetmontecarlo.analytics.radiators.fixedcoupling import *

import matplotlib.ticker as mticker

VERBOSE = 0

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation


def tst_pre_num_rad(rad_sampler, crit_rad_sampler,
                    jet_type='quark',
                    obs_accuracy='LL', splitfn_accuracy='LL',
                    bin_space='lin',
                    fixed_coupling=True,
                    num_bins=100):
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

    if VERBOSE > 3:
        # Testing uniform distribution in log space:
        print("Minimum and maximum of theta: ")
        print(min(theta_crit))
        print(max(theta_crit))
        th = np.log(theta_crit)
        print("Minimum and maximum of log(theta): ")
        print(min(th))
        print(max(th))
        fig = plt.figure()
        plt.hist(th)
        fig.savefig('testhist_theta.pdf', format='pdf')
        print()

        print("Minimum and maximum of z: ")
        print(min(z_em))
        print(max(z_em))
        ze = np.log(z_em)
        print("Minimum and maximum of log(z): ")
        print(min(ze))
        print(max(ze))
        fig = plt.figure()
        plt.hist(ze)
        fig.savefig('testhist_z'+str(rad_sampler.zc)+'.pdf', format='pdf')
        print('', flush=True)

    # Weights, binned observables, and area
    obs = [z_em, theta_crit]

    rad_integrator.setBins(num_bins, obs, bin_space)

    weights = radiatorWeight(z_em, theta_crit, jet_type,
                             fixedcoupling=fixed_coupling,
                             acc=splitfn_accuracy)
    if VERBOSE > 4:
        print("weights*zem*theta_em: ")
        print(weights*z_em*theta_crit, flush=True)

    # Accounting for the additional integration over critical energy fractions
    if bin_space == 'lin':
        area = rad_sampler.zc
        jacs = np.ones(len(theta_crit))
    else:
        area = np.log(1./rad_sampler.epsilon)\
               * np.log(1./crit_rad_sampler.epsilon)
        jacs = z_em * theta_crit

    # Performing integration
    #rad_integrator.setDensity(obs, weights, area) #print (delete, old)
    rad_integrator.setDensity(obs, weights*jacs, area)
    rad_integrator.integrate()

    radiator = rad_integrator.integral
    radiator_error = rad_integrator.integralErr

    # Generating an interpolating function
    rad_integrator.makeInterpolatingFn(interpolate_error=True)

    unbounded_interp_function = rad_integrator.interpFn

    # Bounding the interpolating function
    def bounded_interp_function(x, theta):
        # Pre-critical emission boundaries
        rad = ((0 <= x) * (x <= rad_sampler.zc) * (0 <= theta)
               * unbounded_interp_function(x, theta))
        return rad

    rad_integrator.interpFn = bounded_interp_function

    """
    ##########################
    # Plot tests
    xs = rad_integrator.bins[0][1:]
    ys = rad_integrator.bins[1][1:]

    integral_interp = rad_integrator.interpFn(xs, ys)
    xs, ys = np.meshgrid(xs, ys)
    xs, ys = np.log(xs), np.log(ys)

    zs = [radiator, integral_interp, preRadAnalytic_fc_LL(xs, ys, .05)]
    zs.append(abs(zs[0] - zs[1]))

    zlims = [(0, 1), (0, 1), (0, 1), (0, .1)]
    titles = ['Monte Carlo', 'Interpolation',
              'Analytic', '|Difference|']

    projection = '3d'
    figsize = plt.figaspect(0.5)

    fig = plt.figure(figsize=figsize)
    fig.suptitle('MC Integration to determine '
                 + 'precrit radiator')
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection=projection)
        ax.set_title(titles[i])
        my_col = cm.coolwarm(zs[i])
        ax.plot_surface(xs, ys, zs[i],
                        rstride=1, cstride=1,
                        facecolors=my_col,
                        linewidth=0, antialiased=False)
        ax.set_zlim(zlims[i])
        if i == 0 or i == 3:
            # Plotting errorbars
            fx = xs.flatten()
            fy = ys.flatten()
            fz = zs[i].flatten()
            fzerr = radiator_error.flatten()
            fcols = my_col.reshape(fx.shape[0], 4)
            for j in np.arange(0, len(fx)):
                ax.plot([fx[j], fx[j]], [fy[j], fy[j]],
                        [fz[j]+fzerr[j], fz[j]-fzerr[j]],
                        marker="|", color=fcols[j], zorder=5)

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.savefig('precrit_test'+str(rad_sampler.zc)+'.pdf', format='pdf')
    ##########################
    """

    return rad_integrator.interpFn, rad_integrator.interpErr

