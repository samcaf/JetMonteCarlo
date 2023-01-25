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
SHOW_PLOTS = False
SAVE_PLOTS = True

# Basic setup
NUM_SAMPLES = 1000
NUM_BINS = 100

radii = [1, .8, .6, .4]
betas = [2, 3, 4]

xlabels = [r'$C_1^{(2)}$', r'$C_1^{(3)}$',
           r'$C_1^{(4)}$']
ylabels = [r'$R\left(C_1^{(2)}\right)$',
           r'$R\left(C_1^{(3)}\right)$',
           r'$R\left(C_1^{(4)}\right)$',]
ylims = {'quark': [.07, .06, .01],
         'gluon': [.5, .25, .05]}
ylimslog = {'quark': [10, 5, 2],
            'gluon': [10, 5, 2]}

######################################################
# Ungroomed (or Subsequent) Radiator:
######################################################

# ------------------------------------
# Linear Radiator:
# ------------------------------------
def test_SubLinRadiator():
    # Setting up integrator
    num_int = integrator()

    # Integral is positive, and is zero at the last bin
    num_int.setLastBinBndCondition([0., 'plus'])

    for jet_type in ['quark', 'gluon']:
        if SAVE_PLOTS and not SHOW_PLOTS:
            filename = jet_type+"_fc_linrads_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
        for ibeta, beta in enumerate(betas):
            # Setting up plot
            _, axes = aestheticfig(xlabel=xlabels[ibeta],
                                   ylabel=ylabels[ibeta],
                                   xlim=(0, .3),
                                   ylim=(0, ylims[jet_type][ibeta]),
                                   showdate=False,
                                   title='Ungroomed '+jet_type+' radiator, '
                                   + r'fixed $\alpha_s$',
                                   ratio_plot=False)

            for i, radius in enumerate(radii):
                radius = radii[i]
                # Plotting analytic result
                pnts = np.linspace(0, radius**beta/2., 100)
                label = (r"$R=$" + str(radius))
                axes[0].plot(pnts,
                             subRadAnalytic_fc_LL(pnts, beta,
                                                  jet_type=jet_type,
                                                  maxRadius=radius),
                             **style_dashed,
                             color=compcolors[(i, 'light')],
                             label=label)
            # Labelling
            labelLines(axes[0].get_lines(), xvals=[.25, .12, .035, .02])

            for i, radius in enumerate(radii):
                radius = radii[i]
                # Sampling
                test_sampler = ungroomedSampler('lin', radius=radius)
                test_sampler.generateSamples(NUM_SAMPLES)
                samples = test_sampler.getSamples()
                z = samples[:, 0]
                theta = samples[:, 1]
                obs = C_ungroomed(z, theta, beta, acc='LL')

                # Weights, binned observables, and area
                num_int.setBins(NUM_BINS, obs, 'lin')
                weights = radiatorWeight(z, theta, jet_type,
                                         fixedcoupling=True, acc='LL')
                jacs = test_sampler.jacobians
                area = test_sampler.area

                num_int.setDensity(obs, weights * jacs, area)
                num_int.integrate()

                integral = num_int.integral
                yerr = num_int.integralErr
                x_vals = num_int.bins[:-1]

                col = compcolors[(i, 'dark')]
                ecol = compcolors[(i, 'dark')]

                _, _, bars = axes[0].errorbar(x_vals, integral, yerr=yerr,
                                              **style_yerr,
                                              color=col, ecolor=ecol)
                bars = [b.set_alpha(.5) for b in bars]

            # Legend
            legend_darklight(axes[0], errtype='yerr', twosigma=False)

            if SHOW_PLOTS:
                plt.show()
            elif SAVE_PLOTS:
                plt.savefig(pdffile, format='pdf')
        if SAVE_PLOTS:
            pdffile.close()


# ------------------------------------
# Logarithmic Radiator:
# ------------------------------------
def test_SubLogRadiator():
    # Basic setup
    NUM_SAMPLES = 1000
    NUM_BINS = 100
    epsilons = [1e-3, 1e-5, 1e-10]

    # Setting up integrator
    num_int = integrator()

    # Integral is positive, and is zero at the last bin
    num_int.setLastBinBndCondition([0., 'plus'])

    for jet_type in ['quark', 'gluon']:
        for ibeta, beta in enumerate(betas):
            if SAVE_PLOTS and not SHOW_PLOTS:
                filename = jet_type+"_fc_lograds_test_"+str(beta)+".pdf"
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)
            for i, radius in enumerate(radii):
                radius = radii[i]
                # Setting up plot
                fig, axes = aestheticfig(xlabel=xlabels[ibeta],
                                         ylabel=ylabels[ibeta],
                                         xlim=(1e-8, .3),
                                         ylim=(0,
                                               ylimslog[jet_type][ibeta]),
                                         showdate=False,
                                         title='Ungroomed '+jet_type
                                         +' radiator, '
                                         + r'fixed $\alpha_s$, '
                                         + r"$R=$" + str(radius),
                                         ratio_plot=False)
                axes[0].set_xscale('log')

                # Plotting analytic result
                pnts = np.logspace(-8.5, 0, 1000)
                axes[0].plot(pnts,
                             subRadAnalytic_fc_LL(pnts, beta,
                                                  jet_type=jet_type,
                                                  maxRadius=radius),
                             **style_dashed, label='Analytic',
                             color='cornflowerblue')
                for ieps, eps in enumerate(epsilons):
                    # Sampling
                    test_sampler = ungroomedSampler('log', radius=radius,
                                                    epsilon=eps)
                    test_sampler.generateSamples(NUM_SAMPLES)
                    samples = test_sampler.getSamples()
                    z = samples[:, 0]
                    theta = samples[:, 1]
                    obs = C_ungroomed(z, theta, beta, acc='LL')

                    # Weights, binned observables, and area
                    num_int.setBins(NUM_BINS, obs, 'log')
                    weights = radiatorWeight(z, theta, jet_type,
                                             fixedcoupling=True,
                                             acc='LL')
                    jacs = test_sampler.jacobians
                    area = test_sampler.area

                    num_int.setDensity(obs, weights * jacs, area)
                    num_int.integrate()

                    integral = num_int.integral
                    yerr = num_int.integralErr
                    x_vals = num_int.bins[:-1]

                    col = compcolors[(ieps, 'dark')]
                    ecol = compcolors[(ieps, 'dark')]

                    label = r'Log MC, $\epsilon$={:.0e}'.format(eps)
                    _, _, bars = axes[0].errorbar(x_vals, integral, yerr=yerr,
                                                  **style_yerr,
                                                  color=col, ecolor=ecol,
                                                  label=label)
                    bars = [b.set_alpha(.5) for b in bars]

                # Legend
                legend_yerr(axes[0])

                if SHOW_PLOTS:
                    plt.show()
                elif SAVE_PLOTS:
                    plt.savefig(pdffile, format='pdf')
            if SAVE_PLOTS:
                pdffile.close()

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_SubLinRadiator()
    test_SubLogRadiator()
