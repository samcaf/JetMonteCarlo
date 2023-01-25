import matplotlib.pyplot as plt

# Local imports
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.montecarlo.sampler import *

# Parameters
showPlots = True
savePlots = False

def plotLinSamples(samples, num, icol=0, weights=None):
    """Plots a normalized histogram of the samples"""
    # Getting histogram
    if weights is None: weights = np.ones(len(samples))
    hist, bins = np.histogram(samples, weights=weights, bins=100)
    err = np.sqrt(hist)

    # Normalizing
    err  = err*len(bins)/np.sum(hist)
    hist = hist*len(bins)/np.sum(hist)

    #Plotting
    xs   = (bins[1:]+bins[:-1])/2.
    xerr = (bins[1:]-bins[:-1])/2.
    label = '{:.0e} samples'.format(num)

    plt.errorbar(xs, hist,
                xerr=xerr,yerr=err,
                c=compcolors[(icol,'dark')],
                **modstyle, label=label)

    plt.ylim((0.,np.max(hist)*1.1))

def plotLogSamples(samples, num, icol=0, weights=None):
    """Plots a histogram of the samples,
    normalized by the number in the leftmost bin"""
    # Getting histogram
    if weights is None: weights = np.ones(len(samples))
    hist, bins = np.histogram(samples, weights=weights, bins=100)
    err = np.sqrt(hist)

    # Normalizing
    err  = err*len(bins)/np.sum(hist)
    hist = hist*len(bins)/np.sum(hist)

    #Plotting
    xs   = (bins[1:]+bins[:-1])/2.
    xerr = (bins[1:]-bins[:-1])/2.
    label = '{:.0e} samples'.format(num)

    plt.errorbar(xs, hist,
                xerr=xerr,yerr=err,
                c=compcolors[(icol,'dark')],
                **modstyle, label=label)

    plt.ylim((0.,np.max(hist)*1.1))

# ------------------------------------
# Simple Sampler Tests:
# ------------------------------------
def test_SimpleLinSampling():
    numSamples = [int(1e3),int(1e4),int(1e5),int(1e6)]

    testSampler = simpleSampler('lin')
    if showPlots or savePlots:
        fig, axes = aestheticfig(xlim=(0,1),ylim=(.5,1.5),
                                ratio_plot=False,
                                title='Lin Sampling Test')

        axes[0].set_ylabel('Normalized Sample Distribution',
                            labelpad=5)

    # Check that the area is the expected area
    assert(testSampler.area == 1.)

    # Check that the correct number of samples are being generated
    # and that they reproduce the correct area
    for i in range(len(numSamples)):
        num = numSamples[i]
        # Testing that we generate the correct number of samples
        testSampler.clearSamples()
        testSampler.generateSamples(num)
        assert(len(testSampler.samples) == num)

        samples = testSampler.getSamples()

        # Calculating the area we expect from these samples
        min     = np.min(samples)
        max     = np.max(samples)
        areaCalc = (max-min)
        if showPlots or savePlots:
            plotLinSamples(samples, num, i)
        # Verifying that this area is close to correct
        assert(abs(1. - areaCalc/testSampler.area) < .02)

    pnts = np.linspace(0, 1, 100)
    plt.plot(pnts, np.ones(len(pnts)), **style_dashed,
                color='grey', label='Expected', zorder=3.5)

    if showPlots:
        plt.legend(); plt.show()
    elif savePlots:
        plt.legend()
        filename = ('simpleSampler_lin_test.pdf')
        plt.savefig(filename)


def test_SimpleLogSampling():
    numSamples = [int(1e3),int(1e4),int(1e5),int(1e6)]
    epsilons = [1e-3,1e-5,1e-10]
    for ieps in range(len(epsilons)):
        eps = epsilons[ieps]
        testSampler = simpleSampler('log',epsilon=eps)

        # Check that the area is the expected area
        assert(testSampler.area == np.log(1./eps))

        if showPlots or savePlots:
            title = ('Log Sampling Test, '
                     + r'$\epsilon$ = {:.0e}'.format(eps))
            fig, axes = aestheticfig(xlim=(0,1), ratio_plot=False,
                                    title = title)
            axes[0].set_ylabel("Normalized Sample Distribution",
                                labelpad=5)

        # Check that the correct number of samples are being
        # generated and that they reproduce the correct area
        for i in range(len(numSamples)):
            num = numSamples[i]

            # Testing that we generate the correct number of samples
            testSampler.clearSamples()
            testSampler.generateSamples(num)
            assert(len(testSampler.samples) == num)

            samples = testSampler.getSamples()

            # Calculating the area we expect from these samples
            min     = np.min(samples)
            max     = np.max(samples)
            logmin  = np.log(min); logmax = np.log(max)
            areaCalc = (logmax-logmin)
            if showPlots or savePlots:
                plotLogSamples(samples, num, i)
            # Verifying that this area is close to correct
            assert(abs(1. - areaCalc/testSampler.area) < .02)

        pnts = np.linspace(0, 1, 10000)
        plt.plot(pnts, 1/(pnts * np.log(1/eps)), **style_dashed,
                    color='grey', label='Expected', zorder=3.5)

        if showPlots:
            plt.legend(); plt.show()
        elif savePlots:
            plt.legend()
            filename = ('simpleSampler_log_test_'
                        +str(ieps)+'.pdf')
            plt.savefig(filename)

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_SimpleLinSampling()
    test_SimpleLogSampling()
