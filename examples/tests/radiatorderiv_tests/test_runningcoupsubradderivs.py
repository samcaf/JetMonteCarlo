import numpy as np
import matplotlib.backends.backend_pdf

# Local utils:
from jetmontecarlo.montecarlo.integrator import *
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *

# Local jet tools
from jetmontecarlo.numerics.radiators.samplers import *
from jetmontecarlo.numerics.observables import *
from jetmontecarlo.numerics.weights import *

# Local analytics
from jetmontecarlo.analytics.radiators import *

# Parameters
showPlots = False
savePlots = True

radii = [1, .8, .6, .4]
betas = [2, 3, 4]

xlabels = [r'$C_1^{(2)}$', r'$C_1^{(3)}$',
        r'$C_1^{(4)}$']
ylabels = [r"$R'\left(C_1^{(2)}\right)$",
        r"$R'\left(C_1^{(3)}\right)$",
        r"$R'\left(C_1^{(4)}\right)$",]

######################################################
# Ungroomed (or Subsequent) Radiator:
######################################################

# ------------------------------------
# Linear Radiator:
# ------------------------------------
def test_SubLinRadiatorDeriv():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        if savePlots and not showPlots:
            filename = jet_type+"_rc_linradderivs_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
        for ibeta in range(len(betas)):
            beta = betas[ibeta]

            # Setting up plot
            fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                                     ylabel=ylabels[ibeta],
                                     xlim=(0,.5),
                                     ylim=(0, 2.5),
                                     showdate = False,
                                     title = 'Ungroomed '+jet_type+' rad. deriv., '
                                            + r'run. $\alpha_s$',
                                     ratio_plot=False)


            for i in range(len(radii)):
                radius = radii[i]
                # Plotting analytic result
                pnts = np.linspace(0, radius**beta/2., 100)
                label = (r"$R=$" + str(radius))
                axes[0].plot(pnts,
                            subRadPrimeAnalytic(
                                        pnts, beta,jet_type=jet_type,
                                        maxRadius=radius),
                            **style_dashed,
                            color=compcolors[(i,'light')], label=label)
            # Labelling
            labelLines(axes[0].get_lines(), xvals=[.25,.12,.055, .04])

            for i in range(len(radii)):
                radius = radii[i]
                # Sampling
                testSampler = ungroomedSampler('lin',radius=radius)
                testSampler.generateSamples(numSamples)
                samples = testSampler.getSamples()
                z = samples[:,0]; theta = samples[:,1]
                obs     = C_ungroomed(z, theta, beta, acc='LL')

                # Weights, binned observables, and area
                testInt.setBins(numBins, obs, 'lin')
                weights = radiatorWeight(z, theta, jet_type,
                                fixedcoupling=False, acc='MLL')
                jacs    = testSampler.jacobians
                area    = testSampler.area

                testInt.setDensity(obs, weights * jacs, area)

                deriv = testInt.density
                yerr  = testInt.densityErr
                xs    = (testInt.bins[1:]+testInt.bins[:-1])/2.
                xerr  = (testInt.bins[1:]-testInt.bins[:-1])/2.

                col  = compcolors[(i, 'dark')]

                _, _, bars = axes[0].errorbar(xs, deriv,
                                xerr=xerr, yerr=yerr,
                                **modstyle, color=col)

            # Legend
            legend_darklight(axes[0], errtype='modstyle', twosigma=False)

            if showPlots: plt.show()
            elif savePlots: plt.savefig(pdffile, format='pdf')
        if savePlots: pdffile.close()


# ------------------------------------
# Logarithmic Radiator:
# ------------------------------------
def test_SubLogRadiatorDeriv():
    # Basic setup
    numSamples = 1000
    numBins    = 100
    epsilons   = [1e-3, 1e-5, 1e-10]

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark','gluon']:
        for ibeta in range(len(betas)):
            beta = betas[ibeta]
            if savePlots and not showPlots:
                filename = jet_type+"_rc_logradderivs_test_"+str(beta)+".pdf"
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
            for i in range(len(radii)):
                radius = radii[i]
                # Setting up plot
                fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                             ylabel=ylabels[ibeta],
                             xlim=(1e-5,.5),
                             ylim=(0, 1000),
                             showdate = False,
                             title = 'Ungroomed '+jet_type+' radiator, '
                                    + r'run. $\alpha_s$, '
                                    + r"$R=$" + str(radius),
                             ratio_plot=False)
                axes[0].set_xscale('log')
                axes[0].set_ylabel(ylabels[ibeta], labelpad=0)

                # Plotting analytic result
                pnts = np.logspace(-8.5, 0, 1000)
                axes[0].plot(pnts,
                            subRadPrimeAnalytic(
                                            pnts, beta,
                                            jet_type=jet_type,
                                            maxRadius=radius),
                            **style_dashed, label='Analytic',
                            color='cornflowerblue')
                for ieps in range(len(epsilons)):
                    eps = epsilons[ieps]
                    # Sampling
                    testSampler = ungroomedSampler('log',radius=radius,
                                                    epsilon=eps)
                    testSampler.generateSamples(numSamples)
                    samples = testSampler.getSamples()
                    z = samples[:,0]; theta = samples[:,1]
                    obs     = C_ungroomed(z, theta, beta, acc='LL')

                    # Weights, binned observables, and area
                    testInt.setBins(numBins, obs, 'log')
                    weights = radiatorWeight(z, theta, jet_type,
                                    fixedcoupling=False, acc='MLL')
                    jacs    = testSampler.jacobians
                    area    = testSampler.area

                    testInt.setDensity(obs, weights * jacs, area)

                    deriv = testInt.density
                    yerr  = testInt.densityErr
                    xs    = np.sqrt(testInt.bins[1:] * testInt.bins[:-1])
                    xerr  = (np.abs(xs-testInt.bins[:-1]),
                             np.abs(testInt.bins[1:]-xs))

                    col  = compcolors[(ieps, 'dark')]
                    ecol = compcolors[(ieps, 'dark')]

                    label = r'Log MC, $\epsilon$={:.0e}'.format(eps)
                    _, _, bars = axes[0].errorbar(xs, deriv,
                                    xerr=xerr, yerr=yerr,
                                    **modstyle, color=col,
                                    label=label)

                # Legend
                axes[0].legend()

                if showPlots: plt.show()
                elif savePlots: plt.savefig(pdffile, format='pdf')
            if savePlots: pdffile.close()

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_SubLinRadiatorDeriv()
    test_SubLogRadiatorDeriv()
