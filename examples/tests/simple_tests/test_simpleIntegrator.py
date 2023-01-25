import numpy as np

# Local imports
from jetmontecarlo.montecarlo.sampler import *
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *

showPlots = False
savePlots = True

# ------------------------------------
# Linear Integrator:
# ------------------------------------
def test_SimpleLinIntegrator():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    def test_weight(x, n):
        return (n+1.)*x**n

    # Setting up plot
    fig, axes = aestheticfig(xlabel='x', ylabel='f(x)',
                             ratio_plot=False)

    fig.suptitle(r'Linear Monte Carlo Integration, '
                + '{:.0e} Samples'.format(numSamples))

    # Sampling
    testSampler = simpleSampler('lin')
    testSampler.generateSamples(numSamples)
    samples = testSampler.getSamples()

    # Setting up integrator
    testInt = integrator()
    testInt.setFirstBinBndCondition(0.)
    testInt.setBins(numBins, samples, 'lin')

    # Plotting analytic result
    pnts = np.linspace(0, 1, 100)
    for n in range(4):
        if n==0:
            label = r"$\int $d$x$"
        else:
            label = (r"$\int x^{pow}$d$x\ /\ {powplus}$"
                        .format(pow=n,powplus=n+1))
        axes[0].plot(pnts, pnts**(n+1),**style_dashed,
                    color=compcolors[(n,'light')], label=label)
    # Labelling
    labelLines(axes[0].get_lines())

    for n in range(4):
        # Weights, binned observables, and area
        weights = test_weight(samples, n)
        jacs    = testSampler.jacobians
        obs     = samples
        area    = testSampler.area

        testInt.setDensity(obs, weights * jacs, area)
        testInt.integrate()

        integral = testInt.integral
        yerr     = testInt.integralErr
        xs       = testInt.bins[1:]

        col  = compcolors[(n, 'dark')]
        ecol = compcolors[(n, 'dark')]

        _, _, bars = axes[0].errorbar(xs, integral, yerr=yerr,
                        **style_yerr, color=col,ecolor=ecol)
        [bar.set_alpha(.5) for bar in bars]

    # Legend
    legend_darklight(axes[0], errtype='yerr', twosigma=False)

    if showPlots: plt.show()
    elif savePlots:
        filename = ('simpleIntegrator_lin_test.pdf')
        plt.savefig(filename)


# ------------------------------------
# Logarithmic Integrator:
# ------------------------------------
def test_SimpleLogIntegrator():
    # Basic setup
    numSamples = 1000
    numBins    = 100
    epsilons   = [1e-3,1e-5,1e-10]
    lowerlims  = [1e-3, 1e-3, 1e-3]

    def test_weight(x, n):
        return (n+1.)*x**n

    for ieps in range(len(epsilons)):
        eps = epsilons[ieps]

        # Setting up plot
        fig, axes = aestheticfig(xlabel='x', ylabel='f(x)',
                                 xlim = (lowerlims[ieps], 1),
                                 ratio_plot=False)

        fig.suptitle('Logarithmic Monte Carlo Integration,\n'
                    + r'$\epsilon$ = {:.0e}, '.format(eps)
                    + '{:.0e} Samples'.format(numSamples))

        axes[0].set_xscale('log')

        # Sampling
        testSampler = simpleSampler('log', epsilon=eps)
        testSampler.generateSamples(numSamples)
        samples = testSampler.getSamples()

        # Setting up integrator
        testInt = integrator()
        testInt.setFirstBinBndCondition(0.)
        testInt.setBins(numBins, samples, 'log')

        # Plotting analytic result
        pnts = np.linspace(0, 1, 100)
        for n in range(4):
            if n==0:
                label = r"$\int $d$x$"
            else:
                label = (r"$\int x^{pow}$d$x\ /\ {powplus}$"
                            .format(pow=n,powplus=n+1))
            axes[0].plot(pnts, pnts**(n+1),**style_dashed,
                        color=compcolors[(n,'light')], label=label)
        # Labelling
        labelLines(axes[0].get_lines(), xvals=[.05,.5,.3,.58])

        for n in range(4):
            # Weights, binned observables, and area
            weights = test_weight(samples, n)
            jacs    = testSampler.jacobians
            obs     = samples
            area    = testSampler.area

            testInt.setDensity(obs, weights * jacs, area)
            testInt.integrate()

            integral = testInt.integral
            yerr     = testInt.integralErr
            xs       = testInt.bins[1:]

            col  = compcolors[(n, 'dark')]
            ecol = compcolors[(n, 'dark')]

            _, _, bars = axes[0].errorbar(xs, integral, yerr=yerr,
                            **style_yerr, color=col,ecolor=ecol)
            [bar.set_alpha(.5) for bar in bars]

        # Legend
        legend_darklight(axes[0], errtype='yerr', twosigma=False)

        if showPlots: plt.show()
        elif savePlots:
            filename = ('simpleIntegrator_log_test_'
                        +str(ieps)+'.pdf')
            plt.savefig(filename)


#########################################################
# Tests:
#########################################################
if __name__ == '__main__':
    test_SimpleLinIntegrator()
    test_SimpleLogIntegrator()
