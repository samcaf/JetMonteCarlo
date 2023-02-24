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
from jetmontecarlo.analytics.radiators_fixedcoupling import *

# Parameters
showPlots = False
savePlots = True

xlabels = [r'$C_1^{(2)}$', r'$C_1^{(3)}$',
           r'$C_1^{(4)}$']
ylabels = [r'$\rho\left(C_1^{(2)}\right)$',
           r'$\rho\left(C_1^{(3)}\right)$',
           r'$\rho\left(C_1^{(4)}\right)$',]

radii = [1, .8, .6, .4]
ylims = [3,1,.3]
betas = [2,3,4]

######################################################
# Ungroomed pdf:
######################################################

# ------------------------------------
# Linear pdf:
# ------------------------------------
def test_SubLinPDF():
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
        for ibeta in range(len(betas)):
            beta = betas[ibeta]

            # Setting up plot
            fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                                 ylabel=ylabels[ibeta],
                                 title = 'Ungroomed (fixed) '
                                        +jet_type+' pdf',
                                 xlim=(0,1),
                                 ylim=(0,ylims[ibeta]),
                                 showdate=False,
                                 ratio_plot=False)


            for j in range(len(radii)):
                radius = radii[j]
                # Plotting analytic result
                pnts = np.linspace(0, .5, 100)
                label = (r"$R=$" + str(radius))
                axes[0].plot(pnts,
                    subPDFAnalytic_fc_LL(pnts, beta,
                                        jet_type=jet_type, maxRadius=radius),
                     label=label, **style_dashed,
                     color=compcolors[(j,'light')])
            # Labelling
            if ibeta==0:
                labelLines(axes[0].get_lines(), xvals = [.07,.085,.085,.068])

            for j in range(len(radii)):
                radius = radii[j]
                # Sampling
                testSampler = ungroomedSampler('lin', radius=radius)
                testSampler.generateSamples(numSamples)
                samples = testSampler.getSamples()
                z = samples[:,0]; theta = samples[:,1]

                # Weights, binned observables, and area
                weights = radiatorWeight(z, theta, jet_type,
                                fixedcoupling=True, acc='LL')
                jacs    = testSampler.jacobians
                obs     = C_ungroomed(z, theta, beta, acc='LL')
                area    = testSampler.area

                testInt.setBins(numBins, obs, 'lin')
                testInt.setDensity(obs, weights * jacs, area)
                testInt.integrate()

                deriv    = testInt.density
                deriverr = testInt.densityErr
                integral = testInt.integral
                interr   = testInt.integralErr
                xs       = testInt.bins[:-1]

                col  = compcolors[(j, 'dark')]
                ecol = compcolors[(j, 'dark')]

                PDF  = ( deriv * np.exp(-integral) )
                yerr = ( np.exp(-integral)
                       * np.sqrt(
                            np.square(deriverr) + np.square(interr)
                            )
                        )

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
def test_SubLogPDF():
    # Basic setup
    numSamples = 1000
    numBins    = 100

    epsilons   = [1e-3, 1e-5, 1e-10]

    # Setting up integrator
    testInt = integrator()

    # Integral is positive, and is zero at the last bin
    testInt.setLastBinBndCondition([0.,'plus'])

    for jet_type in ['quark', 'gluon']:
        for ibeta in range(len(betas)):
            beta = betas[ibeta]
            if savePlots and not showPlots:
                filename = (jet_type + "_fc_logpdf_test_"
                            + str(beta) + ".pdf")
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

            for j in range(len(radii)):
                radius=radii[j]
                # Setting up plot
                fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                                    ylabel=ylabels[ibeta],
                                    xlim=(5e-5,1),
                                    ylim=(0,50),
                                    title = 'Ungroomed (fixed) '
                                           +jet_type+' pdf, '
                                           + r'$R=$'+str(radius),
                                    showdate=False,
                                    ratio_plot=False)
                axes[0].set_xscale('log')

                # Plotting analytic result
                pnts = np.logspace(-8.5, 0., 1000)
                axes[0].plot(pnts,
                            subPDFAnalytic_fc_LL(pnts, beta,
                                    jet_type=jet_type, maxRadius=radius),
                            **style_dashed, label='Analytic',
                            color='cornflowerblue')

                # Labelling
                for ieps in range(len(epsilons)):
                    eps = epsilons[ieps]

                    # Sampling
                    testSampler = ungroomedSampler('lin',
                                        epsilon=eps, radius=radius)
                    testSampler.generateSamples(numSamples)
                    samples = testSampler.getSamples()
                    z = samples[:,0]; theta = samples[:,1]

                    # Weights, binned observables, and area
                    weights = radiatorWeight(z, theta, jet_type,
                                    fixedcoupling=True, acc='LL')
                    jacs    = testSampler.jacobians
                    obs     = C_ungroomed(z, theta, beta, acc='LL')
                    area    = testSampler.area

                    testInt.setBins(numBins, obs, 'lin')
                    testInt.setDensity(obs, weights * jacs, area)
                    testInt.integrate()

                    deriv    = testInt.density
                    integral = testInt.integral
                    deriverr = testInt.densityErr
                    interr   = testInt.integralErr
                    xs       = testInt.bins[:-1]

                    col  = compcolors[(ieps, 'dark')]
                    ecol = compcolors[(ieps, 'dark')]

                    PDF  = ( deriv * np.exp(-integral) )
                    yerr = ( np.exp(-integral)
                           * np.sqrt(
                                np.square(deriverr) + np.square(interr)
                                )
                            )

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
    test_SubLinPDF()
    test_SubLogPDF()
