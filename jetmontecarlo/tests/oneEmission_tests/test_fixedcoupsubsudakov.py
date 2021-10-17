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
# Make this better:
from jetmontecarlo.tests.partonshower_tests.test_partonshower_angularities import *

# Parameters
SHOW_PLOTS = False
SAVE_PLOTS = True

X_LABEL_0 = r'$C_1^{(\beta)}$'
Y_LABEL_0 = r'$\Sigma\left(C_1^{(\beta)}\right)$'

X_LABELS = [r'$C_1^{(2)}$', r'$C_1^{(3)}$', r'$C_1^{(4)}$']
Y_LABELS = [r'$\Sigma\left(C_1^{(2)}\right)$',
            r'$\Sigma\left(C_1^{(3)}\right)$',
            r'$\Sigma\left(C_1^{(4)}\right)$']

betas = [2, 3, 4]

# ------------------------------------
# Lin Sampling Sudakov Exponent:
# ------------------------------------
def test_SubLinSudakov():
    # Basic setup
    num_samples = 1000
    num_bins = 100

    # Setting up integrator
    test_int = integrator()

    # Integrating to find a CDF
    test_int.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        if SAVE_PLOTS and not SHOW_PLOTS:
            filename = jet_type+"_fc_linsud_sub_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

        # Setting up plot
        _, axes = aestheticfig(xlabel=X_LABEL_0,
                               ylabel=X_LABEL_0,
                               ylim=(.7, 1.1),
                               xlim=(0, .5),
                               title='Subsequent '+jet_type
                               +' Sudakov factor, '
                               +r'fixed $\alpha_s$',
                               showdate=False,
                               ratio_plot=False)

        for ibeta, beta in enumerate(betas):
            # Plotting analytic result
            pnts = np.linspace(0, .5, 100)
            label = (r"$\beta=$" + str(beta))
            axes[0].plot(pnts,
                         np.exp(-subRadAnalytic_fc_LL(pnts, beta,
                                                      jet_type=jet_type)),
                         **style_dashed,
                         color=compcolors[(ibeta, 'light')],
                         label=label)

        # Labelling
        labelLines(axes[0].get_lines(), xvals=[.1, .2, .3])

        for ibeta, beta in enumerate(betas):
            # Subsequent Sampling:
            sub_sampler = ungroomedSampler('lin')
            sub_sampler.generateSamples(num_samples)
            samples = sub_sampler.getSamples()
            c_sub = samples[:, 0]
            # Since z_sub and c_sub have the same range of integration,
            # we can pretend that we are instead sampling over c_sub here

            # Setting up observables/integration variables:
            obs = c_sub
            #rather than C_ungroomed(z_sub, theta_sub, beta, acc='LL')

            # Setting up binning/integration region:
            test_int.setBins(num_bins, obs, 'lin')

            # Setting up weights/integrand:
            weights = subPDFAnalytic_fc_LL(c_sub, beta, jet_type=jet_type)
            # Rather than
            # weights = subsequentEmissionWeight(z_sub, theta_sub,
            #                                    beta, jet_type,
            #                                    fixedcoupling=True)


            # Setting up jacobians and integration region:
            jacs = np.array(sub_sampler.jacobians)
            area = np.array(sub_sampler.area)

            # Integrating:
            test_int.setDensity(obs, weights * jacs, area)
            test_int.integrate()

            integral = test_int.integral
            yerr = test_int.integralErr
            x_vals = test_int.bins[:-1]

            col = compcolors[(ibeta, 'dark')]
            ecol = compcolors[(ibeta, 'dark')]

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
            pdffile.close()

# ------------------------------------
# Log Sampling Sudakov Exponent:
# ------------------------------------
def test_SubLogSudakov():
    # Basic setup
    num_samples = 1000
    num_bins = 100

    epsilons = [1e-3, 1e-5, 1e-10]
    # Setting up integrator
    test_int = integrator()

    # Integrating to find a CDF
    test_int.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        if SAVE_PLOTS and not SHOW_PLOTS:
            filename = jet_type+"_fc_logsud_sub_test.pdf"
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

        for ibeta, beta in enumerate(betas):
            # Setting up plot
            _, axes = aestheticfig(xlabel=X_LABELS[ibeta],
                                   ylabel=Y_LABELS[ibeta],
                                   ylim=(0, 1),
                                   xlim=(1e-8, .5),
                                   title='Subsequent LL '+jet_type
                                   +' Sudakov factor,'
                                   +r'fixed $\alpha_s$',
                                   showdate=False,
                                   ratio_plot=True)
            axes[0].set_xscale('log')
            axes[1].set_xscale('log')

            # Plotting analytic result
            pnts = np.logspace(-8, np.log10(.5), 100)
            axes[0].plot(pnts,
                         np.exp(-subRadAnalytic_fc_LL(pnts, beta,
                                                      jet_type=jet_type)),
                         **style_dashed, color='dimgrey',
                         label='Analytic')
            axes[1].plot(pnts, np.ones(len(pnts)), **style_dashed,
                         color='dimgrey')

            for ieps, eps in enumerate(epsilons):
                # Subsequent Sampling
                sub_sampler = ungroomedSampler('log', epsilon=eps)
                sub_sampler.generateSamples(num_samples)
                samples = sub_sampler.getSamples()
                c_sub = samples[:, 0]
                # Since z_sub and c_sub have the same range of integration,
                # we can pretend that we are instead sampling over c_sub here

                obs = c_sub
                # Rather than:
                #C_groomed(z_sub, theta_sub, z_c, beta,
                #                z_pre=0., f=1., acc='LL')

                # Weights, binned observables, and area
                test_int.setBins(num_bins, obs, 'log')
                weights = subPDFAnalytic_fc_LL(c_sub, beta, jet_type=jet_type)
                # Rather than
                # weights = subsequentEmissionWeight(z_sub, theta_sub,
                #                                    beta, jet_type,
                #                                    fixedcoupling=True)

                jacs = np.array(sub_sampler.jacobians)
                area = np.array(sub_sampler.area)

                test_int.setDensity(obs, weights * jacs, area)
                test_int.integrate()

                integral = test_int.integral
                yerr = test_int.integralErr
                x_vals = test_int.bins[:-1]

                col = compcolors[(ieps, 'dark')]
                ecol = compcolors[(ieps, 'dark')]

                label = r'Log MC, $\epsilon$={:.0e}'.format(eps)
                _, _, bars = axes[0].errorbar(x_vals, integral,
                                              yerr=yerr,
                                              **style_yerr,
                                              color=col, ecolor=ecol,
                                              label=label)
                bars = [b.set_alpha(.5) for b in bars]

                analytic = np.exp(-subRadAnalytic_fc_LL(x_vals, beta,
                                                        jet_type=jet_type))
                analytic = [float(a) for a in analytic]

                _, _, bars = axes[1].errorbar(x_vals,
                                              integral/analytic,
                                              yerr=yerr/analytic,
                                              color=col, ecolor=ecol,
                                              **style_yerr, label=label)
                bars = [b.set_alpha(.5) for b in bars]

            legend_yerr(axes[0])

            if SHOW_PLOTS:
                plt.show()
            elif SAVE_PLOTS:
                plt.savefig(pdffile, format='pdf')
        if SAVE_PLOTS and not SHOW_PLOTS:
            pdffile.close()


#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    test_SubLinSudakov()
    test_SubLogSudakov()
