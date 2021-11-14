import numpy as np
import matplotlib.backends.backend_pdf

# Local utils:
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *

# Local jet tools
from jetmontecarlo.jets.jetSamplers import *
from jetmontecarlo.jets.observables import *
from jetmontecarlo.jets.weights import *

# Local analytics
from jetmontecarlo.analytics.radiators_fixedcoupling import *

# Parameters
showPlots = False
savePlots = True

zcuts = [.01, .05, .1, .2]


######################################################
# Critical Radiator:
######################################################

# ------------------------------------
# Linear Radiator:
# ------------------------------------
def test_CritLinRadiator():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        # Setting up plot
        fig, axes = aestheticfig(xlabel=r'$\theta$', ylabel=r'R($\theta$)',
                                 title = 'Critical '+jet_type+' radiator, '
                                        + r'fixed $\alpha_s$',
                                 ratio_plot=False)


        for i in range(len(zcuts)):
            zc = zcuts[i]
            # Plotting analytic result
            pnts = np.linspace(0, 1, 100)
            label = (r"$z_c=$" + str(zc))
            axes[0].plot(pnts,
                        critRadAnalytic_fc_LL(pnts, zc,jet_type=jet_type),
                        **style_dashed,
                        color=compcolors[(i,'light')], label=label)
        # Labelling
        labelLines(axes[0].get_lines(), xvals = np.linspace(.23,.65,4))
        for i in range(len(zcuts)):
            zc = zcuts[i]
            # Sampling
            testSampler = criticalSampler('lin', zc=zc)
            testSampler.generateSamples(numSamples)
            samples = testSampler.getSamples()
            z = samples[:,0]; theta = samples[:,1]

            # Weights, binned observables, and area
            testInt.setBins(numBins, samples, 'lin')
            weights = radiatorWeight(z, theta, jet_type,
                            fixedcoupling=True, acc='LL')
            jacs    = testSampler.jacobians
            obs     = theta
            area    = testSampler.area

            testInt.setDensity(obs, weights * jacs, area)
            testInt.integrate()

            integral = testInt.integral
            yerr     = testInt.integralErr
            xs       = testInt.bins[:-1]

            col  = compcolors[(i, 'dark')]
            ecol = compcolors[(i, 'dark')]

            _, _, bars = axes[0].errorbar(xs, integral, yerr=yerr,
                            **style_yerr, color=col,ecolor=ecol)
            [bar.set_alpha(.5) for bar in bars]

        # Legend
        legend_darklight(axes[0], errtype='yerr', twosigma=False)

        if showPlots: plt.show()
        elif savePlots:
            filename = (jet_type+'_fc_linrads_test.pdf')
            plt.savefig(filename)

# ------------------------------------
# Logarithmic Radiator:
# ------------------------------------
def test_CritLogRadiator():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    epsilons   = [1e-3, 1e-5, 1e-10]
    ylims = {'quark': [5.5,3.5,2.5,1.25],
             'gluon': [10,7.5,5,3]}

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        if savePlots and not showPlots:
            filename = jet_type+"_fc_lograds_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
        for izc in range(len(zcuts)):
            zc = zcuts[izc]
            # Setting up plot
            fig, axes = aestheticfig(xlabel=r'$\theta$',
                                ylabel=r'R($\theta$)',
                                xlim=(1e-8,1),
                                ylim=(0,ylims[jet_type][izc]),
                                title = 'Critical '+jet_type+' radiator, '
                                        + r'fixed $\alpha_s$, '
                                        + r'$z_c$='+str(zc),
                                showdate=False,
                                ratio_plot=False)
            axes[0].set_xscale('log')

            # Plotting analytic result
            pnts = np.logspace(-8.5, 0, 1000)
            axes[0].plot(pnts,
                        critRadAnalytic_fc_LL(pnts, zc,jet_type=jet_type),
                        **style_dashed, label='Analytic',
                        color='cornflowerblue')

            # Labelling
            for ieps in range(len(epsilons)):
                eps = epsilons[ieps]

                # Sampling
                testSampler = criticalSampler('log', zc=zc, epsilon=eps)
                testSampler.generateSamples(numSamples)
                samples = testSampler.getSamples()
                z = samples[:,0]; theta = samples[:,1]

                # Weights, binned observables, and area
                testInt.setBins(numBins, samples, 'log')
                weights = radiatorWeight(z, theta, jet_type,
                                fixedcoupling=True, acc='LL')
                jacs    = testSampler.jacobians
                obs     = theta
                area    = testSampler.area

                testInt.setDensity(obs, weights * jacs, area)
                testInt.integrate()

                integral = testInt.integral
                yerr     = testInt.integralErr
                xs       = testInt.bins[:-1]

                col  = compcolors[(ieps, 'dark')]
                ecol = compcolors[(ieps, 'dark')]

                label = r'Log MC, $\epsilon$={:.0e}'.format(eps)
                _, _, bars = axes[0].errorbar(xs, integral, yerr=yerr,
                                **style_yerr, color=col,ecolor=ecol,
                                label=label)
                [bar.set_alpha(.5) for bar in bars]

            # Legend
            legend_yerr(axes[0])

            if showPlots: plt.show()
            elif savePlots: plt.savefig(pdffile, format='pdf')
        if savePlots: pdffile.close()



#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_CritLinRadiator()
    test_CritLogRadiator()
