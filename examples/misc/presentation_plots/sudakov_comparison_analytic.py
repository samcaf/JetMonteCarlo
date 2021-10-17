# Local utilities for comparison
from examples.comparison_plots.comparison_plot_utils import *
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.jets.observables import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
# MC events
NUM_MC_EVENTS = int(1e7)
NUM_SHOWER_EVENTS = int(5e5)

# ------------------------------------
# Shower correlations
# ------------------------------------
sample_folder = Path("jetmontecarlo/utils/samples/shower_correlations/")
filename = 'shower_{:.0e}_c1_'.format(NUM_SHOWER_EVENTS)+str(BETA)
# filename += '_lowcutoff'
filename += '_LL_fewem.npz'

ps_correlations = np.load(sample_folder / filename)

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                 load=True):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    extra_label = '_fc_an_'

    # File path
    sample_folder = Path("jetmontecarlo/utils/samples/"
                         +"inverse_transform_samples")
    sample_file = ("theta_crits"
                   +"_zc"+str(z_cut)
                   +"_beta"+str(beta)
                   +"_{:.0e}".format(NUM_MC_EVENTS)
                   +extra_label
                   +"samples.npy")
    sample_file_path = sample_folder / sample_file

    print("    Loading critical samples with z_c="+str(z_cut)+"...")
    theta_crits = np.load(sample_file_path)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = 1./(z_crits * -np.log(2.*z_cut))

    obs = C_groomed(z_crits, theta_crits, z_cut, BETA,
                    z_pre=0., f=F_SOFT, acc=ACC)

    # Weights, binned observables, and area
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

def compare_crit():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                           beta=BETA, icol=icol,
                           label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_crit(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations['rss_c1s_crit'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)

    fig_pdf.savefig('single_em_analytic_plot_pdf.pdf',
                    format='pdf')

###########################################
# All Emissions
###########################################
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta = BETA, icol=0):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    extra_label = '_fc_an_'

    # File path
    sample_folder = Path("jetmontecarlo/utils/samples/"
                         +"inverse_transform_samples")
    sample_file_crit = ("theta_crits"
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    sample_file_path_crit = sample_folder / sample_file_crit
    print("    Loading critical samples with z_c="+str(z_cut)+"...")
    theta_crits = np.load(sample_file_path_crit)

    # File path
    sample_file_sub = ("c_subs_from_crits"
                       +"_zc"+str(z_cut)
                       +"_beta"+str(beta)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    sample_file_path_sub = sample_folder / sample_file_sub
    print("    Loading subsequent samples with beta="+str(beta)+
          " from crit samples with z_cut="+str(z_cut)+"...")
    c_subs = np.load(sample_file_path_sub)

    # File path
    sample_file_pre = ("z_pres_from_crits"
                       +"_zc"+str(z_cut)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    sample_file_path_pre = sample_folder / sample_file_pre

    print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...")
    z_pres = np.load(sample_file_path_pre)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = 1./(z_crits * -np.log(2.*z_cut))

    c_crits = C_groomed(z_crits, theta_crits, z_cut, BETA,
                        z_pre=z_pres, f=F_SOFT, acc=ACC)
    obs = np.maximum(c_crits, c_subs)

    # Weights, binned observables, and area
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

def compare_all():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('all', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut, icol=icol,
                           label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_all(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations['rss_c1s_all'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic (1 Emission)', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)

    fig_pdf.savefig('multiple_em_analytic_plot_pdf.pdf',
                    format='pdf')

###########################################
# Main:
###########################################
if __name__ == '__main__':
    compare_crit()
    compare_all()
