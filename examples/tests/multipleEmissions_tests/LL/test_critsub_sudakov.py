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
SHOW_PLOTS = False
SAVE_PLOTS = True

NUM_SAMPLES = int(1e3)
NUM_BINS = 100

epsilons = [1e-3, 1e-5, 1e-10]

XLABEL_0 = r'$C_1^{(\beta)}$'
YLABEL_0 = r'$\Sigma\left(C_1^{(\beta)}\right)$'

xlabels = [r'$C_1^{(2)}$', r'$C_1^{(3)}$',
           r'$C_1^{(4)}$']
ylabels = [r'$\Sigma\left(C_1^{(2)}\right)$',
           r'$\Sigma\left(C_1^{(3)}\right)$',
           r'$\Sigma\left(C_1^{(4)}\right)$',]

betas = [2, 3, 4]
z_cuts = [.05, .1, .2]


# ------------------------------------
# Lin Sampling Sudakov Exponent:
# ------------------------------------
def test_critsub_lin_sudakov():
    # Setting up integrator
    test_int = integrator()

    # Integrating to find a CDF
    test_int.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        if SAVE_PLOTS and not SHOW_PLOTS:
            filename = jet_type+"_critsub_fc_linsud_test_"\
                       +"{:.0e}.pdf".format(NUM_SAMPLES)
            pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

        for ibeta, beta in enumerate(betas):
            _, axes = aestheticfig(xlabel=xlabels[ibeta],
                                   ylabel=ylabels[ibeta],
                                   ylim=(.8, 1.025),
                                   xlim=(0, 1.),
                                   title='Crit + Sub'\
                                   +' '+jet_type
                                   +' Sudakov factor, '
                                   +r'fixed $\alpha_s$',
                                   showdate=False,
                                   ratio_plot=False)

            # Plotting the one emission analytic results
            pnts = np.linspace(0, .5, 100)

            for i, z_c in enumerate(z_cuts):
                label = (r"$z_c=$" + str(z_c))
                axes[0].plot(pnts,
                             critSudakov_fc_LL(pnts, z_c, beta,
                                               jet_type=jet_type),
                             **style_dashed,
                             color=compcolors[(i, 'light')],
                             label=label)
            # Labelling
            labelLines(axes[0].get_lines(), xvals=[.1, .2, .3])

            for i, z_c in enumerate(z_cuts):
                # Critical Sampling
                crit_sampler = criticalSampler('lin', zc=z_c)
                crit_sampler.generateSamples(NUM_SAMPLES)
                samples = crit_sampler.getSamples()
                z_crit = samples[:, 0]
                theta_crit = samples[:, 1]

                # Subsequent Sampling
                sub_sampler = ungroomedSampler('lin')
                sub_sampler.generateSamples(NUM_SAMPLES)
                samples = sub_sampler.getSamples()
                c_sub = samples[:, 0]
                # Since z_sub and c_sub have the same range of integration,
                # we can pretend that we are instead sampling over c_sub here

                obs = (C_groomed(z_crit, theta_crit, z_c, beta,
                                 z_pre=0., f=1., acc='LL')
                       + c_sub)

                # Weights, binned observables, and area
                test_int.setBins(NUM_BINS, obs, 'lin')

                weights = (
                    criticalEmissionWeight(z_crit, theta_crit,
                                           z_c, jet_type,
                                           fixedcoupling=True)
                    *
                    subPDFAnalytic_fc_LL(c_sub, beta, jet_type=jet_type)
                    )

                jacs = (np.array(crit_sampler.jacobians)
                        * np.array(sub_sampler.jacobians))
                area = (np.array(crit_sampler.area)
                        * np.array(sub_sampler.area))

                test_int.setDensity(obs, weights * jacs, area)
                test_int.integrate()

                integral = test_int.integral
                yerr = test_int.integralErr
                x_vals = test_int.bins[:-1]

                # Choosing color scheme
                col = compcolors[(i, 'dark')]
                ecol = compcolors[(i, 'dark')]

                _, _, bars = axes[0].errorbar(x_vals, integral, yerr=yerr,
                                              **style_yerr,
                                              color=col, ecolor=ecol)
                bars = [b.set_alpha(.5) for b in bars]

            # Legend
            legend_darklight(axes[0], errtype='yerr', twosigma=False,
                             lightlabel='Analytic, SE')
            if SHOW_PLOTS:
                plt.show()
            elif SAVE_PLOTS:
                plt.savefig(pdffile, format='pdf')

        if SAVE_PLOTS and not SHOW_PLOTS:
            pdffile.close()


# ------------------------------------
# Log Sampling Sudakov Exponent:
# ------------------------------------
def test_critsub_log_sudakov():
    # Setting up integrator
    test_int = integrator()

    # Integrating to find a CDF
    test_int.setLastBinBndCondition([1., 'minus'])

    for jet_type in ['quark', 'gluon']:
        for ibeta, beta in enumerate(betas):
            if SAVE_PLOTS and not SHOW_PLOTS:
                filename = jet_type+"_critsub_fc_logsud_test_"\
                           +str(beta)\
                           +"_{:.0e}.pdf".format(NUM_SAMPLES)
                pdffile = matplotlib.backends.backend_pdf.PdfPages(filename)

            for _, z_c in enumerate(z_cuts):
                _, axes = aestheticfig(xlabel=xlabels[ibeta],
                                       ylabel=ylabels[ibeta],
                                       ylim=(0., 1.025),
                                       xlim=(1e-8, 1.),
                                       title='Crit + Sub'\
                                       +' '+jet_type
                                       +' Sudakov factor, '
                                       +r'fixed $\alpha_s$',
                                       showdate=False,
                                       ratio_plot=True)

                axes[0].set_xscale('log')
                axes[1].set_xscale('log')

                # Plotting analytic result
                pnts = np.logspace(-8, np.log10(.5), 100)

                axes[0].plot(pnts, critSudakov_fc_LL(pnts, z_c, beta,
                                                     jet_type=jet_type),
                             **style_dashed, color='dimgrey',
                             label='Analytic, SE')

                axes[1].plot(pnts, np.ones(len(pnts)), **style_dashed,
                             color='dimgrey')

                for ieps, eps in enumerate(epsilons):
                    # Critical Sampling
                    crit_sampler = criticalSampler('log', zc=z_c,
                                                   epsilon=eps)
                    crit_sampler.generateSamples(NUM_SAMPLES)
                    samples = crit_sampler.getSamples()
                    z_crit = samples[:, 0]
                    theta_crit = samples[:, 1]

                    # Subsequent Sampling
                    sub_sampler = ungroomedSampler('log', epsilon=eps)
                    sub_sampler.generateSamples(NUM_SAMPLES)
                    samples = sub_sampler.getSamples()
                    c_sub = samples[:, 0]
                    # Since z_sub and c_sub have the same range of
                    # integration,we can pretend that we are instead
                    # sampling over c_sub here

                    obs = (C_groomed(z_crit, theta_crit, z_c, beta,
                                     z_pre=0., f=1., acc='LL')
                           + c_sub)

                    # Weights, binned observables, and area
                    test_int.setBins(NUM_BINS, obs, 'log')

                    weights = (
                        criticalEmissionWeight(z_crit, theta_crit,
                                               z_c, jet_type,
                                               fixedcoupling=True)
                        *
                        subPDFAnalytic_fc_LL(c_sub, beta,
                                             jet_type=jet_type)
                        )

                    jacs = (np.array(crit_sampler.jacobians)
                            * np.array(sub_sampler.jacobians))
                    area = (np.array(crit_sampler.area)
                            * np.array(sub_sampler.area))

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

                    analytic = critSudakov_fc_LL(x_vals, z_c, beta,
                                                 jet_type=jet_type)
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
    # Linear tests:
    test_critsub_lin_sudakov()

    # Logarithmic tests
    test_critsub_log_sudakov()
