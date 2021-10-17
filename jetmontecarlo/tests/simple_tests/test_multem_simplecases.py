from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Local utils:
from jetmontecarlo.montecarlo.integrator import integrator
from jetmontecarlo.utils.plot_utils \
    import aestheticfig, style_dashed, style_yerr, \
    legend_darklight, labelLines
from jetmontecarlo.utils.color_utils import compcolors

# Local test utils:
from jetmontecarlo.tests.simple_tests.test_simpleSampler import simpleSampler

# Parameters
NUM_SAMPLES = 1000
NUM_BINS = 100
NUM_RVS = 3
X_MAX = [NUM_RVS, 2*NUM_RVS]

SHOW_PLOTS = True
SAVE_PLOTS = False

# ------------------------------------
# Test pdfs:
# ------------------------------------
rvtypes = ['Uniform (0,1)', 'Uniform (0,2)', 'Gaussian']

def simple_pdf(samples, algorithm=0):
    """Returns a test pdf for a set of samples, in order
    to test the procedure we are using to generate pdfs
    and cdfs for multiple emissions.
    The particular test pdf depends on the algorithm
    we are using to test the multiple emissions procedure

    Parameters
    ----------
    samples : list
        A list of samples to which
    algorithm : int
        An integer that determines the particular pdf we use
        to test our multiple emissions procedure:
        0 - Uniform pdf from 0 to 1
        1 - Uniform pdf from 0 to 2

    Returns pdf
    -------
    list
        A set of pdfs corresponding to the given samples,
        for the given algorithm.
    """
    assert algorithm in [0, 1, 2], "Unsupported pdf algorithm."
    if algorithm == 0:
        return np.ones(len(samples))
    if algorithm == 1:
        return np.ones(len(samples)) / 2.
    return np.zeros(len(samples))

# ------------------------------------
# Analytic CDFs:
# ------------------------------------
def simple_cdf(x_vals, num_summed, algorithm=0):
    """Analytic expression for the cdf of a sum of
    num_summed random variables distributed according
    to an algorithm.

    Parameters
    ----------
    x_vals : list
        List of x values for which we will return a cdf.
    num_summed : type
        Number of identically distributed random variables
        we consider summing over.
    algorithm : type
        Algorithm which determines the distribution of our
        random variables.

    Returns cdf
    -------
    numpy array
        An array containing a list of cdf values for the given x_vals
    """
    cdf = []
    if algorithm == 0:
        # Uniform RVs from 0 to 1
        # The Irwin-Hall distribution governs the sum of uniform rvs:
        # https://en.wikipedia.org/wiki/Irwin–Hall_distribution
        for x_val in x_vals:
            cdf_x = 0
            for k in range(int(np.floor(x_val)+1)):
                cdf_x_k = (-1)**k*comb(num_summed, k)*(x_val-k)**num_summed
                cdf_x += cdf_x_k/factorial(num_summed)
            cdf.append(cdf_x)
        cdf = np.array(cdf)
        return cdf*(cdf < 1.) + (cdf >= 1.)

    if algorithm == 1:
        # Uniform RVs from 0 to 2
        for x_val in x_vals:
            cdf_x = 0
            x_val = x_val/2.
            for k in range(int(np.floor(x_val)+1)):
                cdf_x_k = (-1)**k*comb(num_summed, k)*(x_val-k)**num_summed
                cdf_x += cdf_x_k/factorial(num_summed)
            cdf.append(cdf_x)
        cdf = np.array(cdf)
        return cdf*(cdf < 1.) + (cdf >= 1.)

    # Gaussian RVs
    cdf = np.zeros(len(x_vals))
    return cdf*(cdf < 1.) + (cdf >= 1.)

# ------------------------------------
# Testing sums of RVs:
# ------------------------------------
def test_lin_sum_of_uniform_rvs(algorithm):
    """Produces pdfs associated with sums of identical random
    variables (rvs), using linear MC integration.
    The distribution of the rvs depends on a chosen algorithm.

    In particular, this method samples over 3 rvs, and integrates
    out 2, 1, or none of these rvs in order to produce several
    distributions.

    This is to make sure that the integration over multiple emissions
    when we go to the case of jet observables does not have problems
    when it comes to the way it integrates over the extra emissions.

    Parameters
    ----------
    algorithm : int
        An integer that determines the particular pdf we use
        to test our multiple emissions procedure:
        0 - Uniform pdf from 0 to 1
        1 - Uniform pdf from 0 to 2

    Returns None
    """
    # Setting up plot
    _, axes = aestheticfig(xlabel='x', ylabel='CDF(x)',
                           title='CDFs of sums of '
                           + str(rvtypes[algorithm]) + ' RVs',
                           xlim=(0, X_MAX[algorithm]),
                           ylim=(0, 1.1),
                           showdate=False,
                           ratio_plot=False)

    # Plotting the analytic result for the sum of i rvs
    pnts = np.linspace(0, X_MAX[algorithm], 1000)
    for i in range(NUM_RVS):
        axes[0].plot(pnts, simple_cdf(pnts, i+1, algorithm),
                     **style_dashed,
                     color=compcolors[(i%4, 'light')],
                     label=str(i+1) + ' RVs' if i != 0 else '1 RV')

    # Labelling the lines for the analytic expressions
    labelLines(axes[0].get_lines(),
               xvals=np.linspace(.5*(algorithm+1), (NUM_RVS-.5)*(algorithm+1),
                                 NUM_RVS))


    # Setting up total integration weight, area, and observables
    weights = 1.
    area = 1.
    # The first entry of obs_all will have rvs distributed by 'algorithm',
    # the second will have the sum of two identically such distributed rvs,
    # etc.
    obs_all = []

    for i in range(NUM_RVS):
        # Generating samples
        test_sampler = simpleSampler('lin')
        test_sampler.generateSamples(NUM_SAMPLES)

        if algorithm == 1:
            test_sampler.samples *= 2.
            test_sampler.area *= 2.

        # Getting integration weight for all samples at once
        weights = (weights * np.array(test_sampler.jacobians)
                   * simple_pdf(test_sampler.getSamples(), algorithm))

        # Getting integration area for all samples at once
        area = area * test_sampler.area

        # Getting the sums of our identically distributed rvs
        if i == 0:
            obs_all.append(test_sampler.getSamples())
        else:
            obs_all.append(obs_all[-1]+test_sampler.getSamples())

    # Setting up integrator
    test_int = integrator()
    test_int.setLastBinBndCondition((1., 'minus'))

    # Enumerating over the relevant sets of samples, jacobians, and areas
    # to find distributions for each possible sum of rvs:
    for i, obs in enumerate(obs_all):
        # Integration
        test_int.setBins(NUM_BINS, obs, 'lin')
        test_int.setDensity(obs, weights, area)
        test_int.integrate()

        # Plotting
        _, _, bars = axes[0].errorbar(x=test_int.bins[:-1],
                                      y=test_int.integral,
                                      yerr=test_int.integralErr,
                                      **style_yerr,
                                      color=compcolors[(i%4, 'dark')],
                                      ecolor=compcolors[(i%4, 'dark')])
        bars = [b.set_alpha(.5) for b in bars]

    # Legend
    legend_darklight(axes[0], errtype='yerr', twosigma=False)

    if SHOW_PLOTS:
        plt.show()
    elif SAVE_PLOTS:
        plt.savefig('rv_lin_sum_'+str(algorithm)+'_test.pdf',
                    format='pdf')

def test_log_sum_of_uniform_rvs(algorithm):
    """Produces pdfs associated with sums of identical random
    variables (rvs), using logarithmic MC integration.
    The distribution of the rvs depends on a chosen algorithm.

    In particular, this method samples over 3 rvs, and integrates
    out 2, 1, or none of these rvs in order to produce several
    distributions.

    This is to make sure that the integration over multiple emissions
    when we go to the case of jet observables does not have problems
    when it comes to the way it integrates over the extra emissions.

    Parameters
    ----------
    algorithm : int
        An integer that determines the particular pdf we use
        to test our multiple emissions procedure:
        0 - Uniform pdf from 0 to 1
        1 - Uniform pdf from 0 to 2

    Returns None
    """
    # Setting up plot
    _, axes = aestheticfig(xlabel='x', ylabel='CDF(x)',
                           title='CDFs of sums of '
                           + str(rvtypes[algorithm]) + ' RVs',
                           xlim=(1e-3, X_MAX[algorithm]),
                           ylim=(0, 1.1),
                           showdate=False,
                           ratio_plot=False)
    axes[0].set_xscale('log')

    # Plotting the analytic result for the sum of i rvs
    pnts = np.logspace(-3, np.log10(X_MAX[algorithm]), 1000)
    for i in range(NUM_RVS):
        axes[0].plot(pnts, simple_cdf(pnts, i+1, algorithm),
                     **style_dashed,
                     color=compcolors[(i%4, 'light')],
                     label=str(i+1) + ' RVs' if i != 0 else '1 RV')

    # Labelling the lines for the analytic expressions
    labelLines(axes[0].get_lines(),
               xvals=np.logspace(np.log10(8e-2),
                                 np.log10((NUM_RVS-2)*(algorithm+1.)),
                                 NUM_RVS))


    # Setting up total integration weight, area, and observables
    weights = 1.
    area = 1.
    # The first entry of obs_all will have rvs distributed by 'algorithm',
    # the second will have the sum of two identically such distributed rvs,
    # etc.
    obs_all = []

    for i in range(NUM_RVS):
        # Generating samples
        test_sampler = simpleSampler('log', epsilon=1e-8)
        test_sampler.generateSamples(NUM_SAMPLES)

        if algorithm == 1:
            test_sampler.samples = np.array(test_sampler.samples)*2.
            test_sampler.jacobians = np.array(test_sampler.jacobians)*2.

        # Getting integration weight for all samples at once
        weights = (weights * np.array(test_sampler.jacobians)
                   * simple_pdf(test_sampler.getSamples(), algorithm))

        # Getting integration area for all samples at once
        area = area * test_sampler.area

        # Getting the sums of our identically distributed rvs
        if i == 0:
            obs_all.append(test_sampler.getSamples())
        else:
            obs_all.append(obs_all[-1]+test_sampler.getSamples())

    # Setting up integrator
    test_int = integrator()
    test_int.setLastBinBndCondition((1., 'minus'))

    # Enumerating over the relevant sets of samples, jacobians, and areas
    # to find distributions for each possible sum of rvs:
    for i, obs in enumerate(obs_all):
        # Integration
        test_int.setBins(NUM_BINS, obs, 'log')
        test_int.setDensity(obs, weights, area)
        test_int.integrate()

        # Plotting
        _, _, bars = axes[0].errorbar(x=test_int.bins[:-1],
                                      y=test_int.integral,
                                      yerr=test_int.integralErr,
                                      **style_yerr,
                                      color=compcolors[(i%4, 'dark')],
                                      ecolor=compcolors[(i%4, 'dark')])
        bars = [b.set_alpha(.5) for b in bars]

    # Legend
    legend_darklight(axes[0], errtype='yerr', twosigma=False)

    if SHOW_PLOTS:
        plt.show()
    elif SAVE_PLOTS:
        plt.savefig('rv_log_sum_'+str(algorithm)+'_test.pdf',
                    format='pdf')

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_lin_sum_of_uniform_rvs(algorithm=0)
    test_lin_sum_of_uniform_rvs(algorithm=1)
    test_log_sum_of_uniform_rvs(algorithm=0)
    test_log_sum_of_uniform_rvs(algorithm=1)
