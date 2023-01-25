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
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *
# Make this better:
from examples.tests.partonshower_tests.test_partonshower_angularities import *

# Parameters
showPlots = False
savePlots = True

xlabels = [r'$C_1^{(2)}$', r'$C_1^{(3)}$',
           r'$C_1^{(4)}$']
ylabels = [r'$\Sigma\left(C_1^{(2)}\right)$',
           r'$\Sigma\left(C_1^{(3)}\right)$',
           r'$\Sigma\left(C_1^{(4)}\right)$',]

betas = [2, 3, 4]
zcuts = [.05, .1, .2]

# ------------------------------------
# Lin Sampling Sudakov Exponent:
# ------------------------------------
def test_CritLinSudakov():
    # Basic setup
    numSamples = 1000
    numBins = 100

    # Setting up integrator
    testInt = integrator()

    # Integrating to find a CDF
    testInt.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        if savePlots and not showPlots:
            filename = jet_type+"_fc_linsud_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
        for ibeta in range(len(betas)):
            beta = betas[ibeta]
            # Setting up plot
            fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                                     ylabel=ylabels[ibeta],
                                     ylim=(.7, 1.1),
                                     xlim=(0, .5),
                                     title='Critical '+jet_type
                                     +' Sudakov factor, '
                                     + r'fixed $\alpha_s$',
                                     showdate=False,
                                     ratio_plot=False)

            for i in range(len(zcuts)):
                zc = zcuts[i]
                # Plotting analytic result
                pnts = np.linspace(0, .5, 100)
                label = (r"$z_c=$" + str(zc))
                axes[0].plot(pnts,
                             critSudakov_fc_LL(pnts, zc, beta,
                                               jet_type=jet_type),
                             **style_dashed,
                             color=compcolors[(i, 'light')],
                             label=label)

            # Labelling
            labelLines(axes[0].get_lines(), xvals=[.1, .2, .3])

            for i in range(len(zcuts)):
                zc = zcuts[i]
                # Sampling
                testSampler = criticalSampler('lin', zc=zc)
                testSampler.generateSamples(numSamples)
                samples = testSampler.getSamples()
                z = samples[:, 0]
                theta = samples[:, 1]
                obs = C_groomed(z, theta, zc, beta)

                # Weights, binned observables, and area
                testInt.setBins(numBins, obs, 'lin')
                weights = criticalEmissionWeight(z, theta, zc, jet_type,
                                                 fixedcoupling=True)
                jacs = testSampler.jacobians
                area = testSampler.area

                testInt.setDensity(obs, weights * jacs, area)
                testInt.integrate()

                integral = testInt.integral
                yerr = testInt.integralErr
                xs = testInt.bins[:-1]

                col = compcolors[(i, 'dark')]
                ecol = compcolors[(i, 'dark')]

                _, _, bars = axes[0].errorbar(xs, integral, yerr=yerr,
                                              **style_yerr,
                                              color=col,
                                              ecolor=ecol)
                bars = [b.set_alpha(.5) for b in bars]

            # Legend
            legend_darklight(axes[0], errtype='yerr', twosigma=False)

            if showPlots:
                plt.show()
            elif savePlots:
                plt.savefig(pdffile, format='pdf')
        if savePlots:
            pdffile.close()

# ------------------------------------
# Log Sampling Sudakov Exponent:
# ------------------------------------
def test_CritLogSudakov():
    # Basic setup
    numSamples = 1000
    numBins = 100

    epsilons = [1e-3, 1e-5, 1e-10]
    # Setting up integrator
    testInt = integrator()

    # Integrating to find a CDF
    testInt.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        for ibeta in range(len(betas)):
            beta = betas[ibeta]

            if savePlots and not showPlots:
                filename = jet_type+"_fc_logsud_test_"+str(beta)+".pdf"
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

            for izc in range(len(zcuts)):
                zc = zcuts[izc]

                # Setting up plot
                fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                            ylabel=ylabels[ibeta],
                            ylim  = (0, 1),
                            xlim  = (1e-8, .5),
                            title = 'Critical LL '+ jet_type
                                    +' Sudakov factor,'
                                    + r' $z_{\rm cut}$='
                                    + str(zc),
                            showdate   = False,
                            ratio_plot = True)
                axes[0].set_xscale('log')
                axes[1].set_xscale('log')

                # Plotting analytic result
                pnts = np.logspace(-8, np.log10(.5), 100)
                axes[0].plot(pnts,
                            critSudakov_fc_LL(pnts, zc,beta,
                                              jet_type=jet_type),
                            **style_dashed, color='dimgrey',
                            label='Analytic')
                axes[1].plot(pnts, np.ones(len(pnts)),**style_dashed,
                             color='dimgrey')

                for ieps in range(len(epsilons)):
                    eps = epsilons[ieps]

                    # Sampling
                    testSampler = criticalSampler('log', zc=zc,
                                                  epsilon=eps)
                    testSampler.generateSamples(numSamples)
                    samples = testSampler.getSamples()
                    z = samples[:,0]; theta = samples[:,1]
                    obs     = C_groomed(z, theta, zc, beta)

                    # Weights, binned observables, and area
                    testInt.setBins(numBins, obs, 'log')
                    weights = criticalEmissionWeight(z, theta, zc,
                                                     jet_type,
                                                     fixedcoupling=True)
                    jacs = testSampler.jacobians
                    area = testSampler.area

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
                    bars = [b.set_alpha(.5) for b in bars]

                    analytic = critSudakov_fc_LL(xs, zc,beta,jet_type=jet_type)
                    analytic = [float(a) for a in analytic]

                    _, _, bars = axes[1].errorbar(xs,
                                    integral/analytic,
                                    yerr=yerr/analytic,
                                    color=col,ecolor=ecol,
                                    **style_yerr, label=label)
                    bars = [b.set_alpha(.5) for b in bars]

                legend_yerr(axes[0])

                if showPlots: plt.show()
                elif savePlots: plt.savefig(pdffile, format='pdf')
            if savePlots and not showPlots: pdffile.close()




#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_CritLinSudakov()
    test_CritLogSudakov()
