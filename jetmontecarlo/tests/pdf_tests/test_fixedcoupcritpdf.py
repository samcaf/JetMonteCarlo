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
zlist = {.01: [.05, .1, .2, .4],
         .05: [.1, .2, .3, .4],
         .1 : [.15, .25, .35, .45],
         .2 : [.25, .3, .4, .45]}
ylims = [25, 20, 10, 7]

######################################################
# Critical pdf:
######################################################

# ------------------------------------
# Linear pdf:
# ------------------------------------
def test_CritLinPDF():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        if savePlots and not showPlots:
            filename = (jet_type + "_fc_linpdf_test.pdf")
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
        for izcut in range(len(zcuts)):
            zc = zcuts[izcut]
            zs_pdf = zlist[zc]

            # Setting up plot
            fig, axes = aestheticfig(xlabel=r'$\theta$',
                                     ylabel=r'$\rho(z, \theta)$',
                                     title = 'Critical '+jet_type+' pdf, '
                                            + r'fixed $\alpha_s$, '
                                            + r'$z_c=$'+str(zc),
                                     xlim=(0,1), ylim=(0,ylims[izcut]),
                                     showdate=False,
                                     ratio_plot=False)

            for j in range(len(zs_pdf)):
                # z_pdf is the z that corresponds to a plot of rho(z, theta)
                z_pdf = zs_pdf[j]
                # Plotting analytic result
                pnts = np.linspace(0, 1, 100)
                label = (r"$z=$" + str(z_pdf))
                axes[0].plot(pnts,
                            critPDFAnalytic_fc_LL(z_pdf, pnts, zc,
                                                jet_type=jet_type),
                            **style_dashed, label=label,
                            color=compcolors[(j,'light')])
            # Labelling
            labelLines(axes[0].get_lines(), xvals = [.08,.1,.18,.33])

            for j in range(len(zs_pdf)):
                # z_pdf is the z that corresponds to a plot of rho(z, theta)
                z_pdf = zs_pdf[j]
                # Sampling
                testSampler = criticalSampler('lin', zc=zc)
                testSampler.generateSamples(numSamples)
                samples = testSampler.getSamples()
                # z is the set of critical z samples
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
                interr   = testInt.integralErr
                xs       = testInt.bins[:-1]

                col  = compcolors[(j, 'dark')]
                ecol = compcolors[(j, 'dark')]

                PDF  = ( splittingFn(z_pdf, jet_type, 'LL')
                        * alpha_fixed / (np.pi * xs)
                        * np.exp(-integral) )
                yerr = ( splittingFn(z_pdf, jet_type, 'LL')
                        * alpha_fixed / (np.pi * xs)
                        * np.exp(-integral) ) * interr

                _, _, bars = axes[0].errorbar(xs, PDF, yerr=yerr,
                                **style_yerr, color=col,ecolor=ecol)
                [bar.set_alpha(.5) for bar in bars]

            # Legend
            legend_darklight(axes[0], errtype='yerr', twosigma=False)

            if showPlots: plt.show()
            elif savePlots: plt.savefig(pdffile, format='pdf')
        if savePlots: pdffile.close()


# ------------------------------------
# Linear pdf:
# ------------------------------------
def test_CritLogPDF():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    epsilons   = [1e-3, 1e-5, 1e-10]

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        for izcut in range(len(zcuts)):
            if savePlots and not showPlots:
                filename = (jet_type + "_fc_logpdf_test_"
                            + str(izcut) + ".pdf")
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
            zc = zcuts[izcut]
            zs_pdf = zlist[zc]

            for j in range(len(zs_pdf)):
                z_pdf = zs_pdf[j]

                # Setting up plot
                fig, axes = aestheticfig(xlabel=r'$\theta$',
                                    ylabel=r'$\rho(z, \theta)$',
                                    xlim=(5e-5,1),
                                    ylim=(0,100),
                                    title = 'Critical (fixed) '
                                           +jet_type+' pdf, '
                                           + r'$z_c=$'+str(zc)
                                           + r', $z=$'+str(z_pdf),
                                    showdate=False,
                                    ratio_plot=False)
                axes[0].set_xscale('log')

                # Plotting analytic result
                pnts = np.logspace(-8.5, 0, 1000)
                axes[0].plot(pnts,
                            critPDFAnalytic_fc_LL(z_pdf, pnts,
                                                  zc,jet_type=jet_type),
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
                    interr   = testInt.integralErr
                    xs       = testInt.bins[:-1]

                    col  = compcolors[(ieps, 'dark')]
                    ecol = compcolors[(ieps, 'dark')]

                    PDF  = ( splittingFn(z_pdf, jet_type, 'LL')
                            * alpha_fixed / (np.pi * xs)
                            * np.exp(-integral) )
                    yerr = ( splittingFn(z_pdf, jet_type, 'LL')
                            * alpha_fixed / (np.pi * xs)
                            * np.exp(-integral) ) * interr


                    label = r'Log MC, $\epsilon$={:.0e}'.format(eps)
                    _, _, bars = axes[0].errorbar(xs, PDF, yerr=yerr,
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
    test_CritLinPDF()
    test_CritLogPDF()
