from __future__ import absolute_import
import dill as pickle
from pathlib import Path

# Local utilities for comparison
from jetmontecarlo.utils.montecarlo_utils import getLinSample
from jetmontecarlo.numerics.observables import *
from examples.utils.plot_comparisons import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *
from jetmontecarlo.montecarlo.partonshower import *

# Parameters
from examples.params import *
from examples.filenames import *

from examples.sudakov_comparison.sudakov_utils import plot_label
from examples.sudakov_comparison.sudakov_utils import split_fn_num
from examples.sudakov_comparison.sudakov_utils import radiators

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]
Z_CUT_PLOT = [F_SOFT * zc for zc in Z_CUT_PLOT]


###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                 load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=F_SOFT, acc=OBS_ACC)

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_crit(beta=BETA, plot_approx=False):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUT_PLOT):
        if icol > 0:
            plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                               beta, icol=icol, jet_type='quark',
                               f_soft=F_SOFT,
                               label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol+1, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUT_PLOT):
        if i > 0:
            plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, icol=i)
            plot_shower_pdf_cdf(ps_correlations(beta)['rss_c1s_crit'][i],
                                axes_pdf, axes_cdf,
                                label='Parton Shower', colnum=i)

            # Narrowing in on jets with P_T between 3 and 3.5 TeV
            cond_floor = (3000 < np.array(pythia_data['raw'][plot_level]['pt'][beta]))
            cond_ceil = (np.array(pythia_data['raw'][plot_level]['pt'][beta]) < 3500)
            inds = np.where(cond_floor * cond_ceil)[0]

            # Getting substructure
            pythia_c2s = pythia_data['rss'][plot_level][params]['C1'][beta]
            pythia_c2s = np.array(pythia_c2s)[inds]
            # print(pythiadata['rss']['hadron'][z_cut][F_SOFT].keys())
            plot_pythia_pdf_cdf(pythia_c2s,
                                axes_pdf, axes_cdf,
                                label='Pythia', colnum=i)

    lightlabel='Analytic'

    # Saving plots/
    legend_darklight(axes_pdf[0], darklabel='pQCD',
                     lightlabel=lightlabel, errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='pQCD',
                     lightlabel=lightlabel, errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_showeronly_RSS_crit_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    """
    fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_showeronly_RSS_crit_'
                    +BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    """

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!")

###########################################
# Main:
###########################################
if __name__ == '__main__':
    for beta in [.5, 1, 2]:
        compare_crit(beta=beta, plot_approx=False)
