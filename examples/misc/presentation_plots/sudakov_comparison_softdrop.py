# Local utilities for comparison
from examples.comparison_plots.comparison_plot_utils import *
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.jets.observables import *

# Local analytics
from jetmontecarlo.analytics.soft_drop import *

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
filename += '_LL_fewem.npz'

ps_correlations = np.load(sample_folder / filename)

###########################################
# Critical Emission Only
###########################################
def plot_mc_softdrop(axes_pdf, axes_cdf, z_cut, beta=BETA,
                     beta_sd=0, icol=0):
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

    obs = z_crits * theta_crits**beta

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


def compare_softdrop():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('Soft Drop', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, beta=BETA, beta_sd=0,
                               jet_type=JET_TYPE,
                               icol=icol, label=r'$z_{\rm cut}=$'+str(z_cut),
                               acc='LL')

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_softdrop(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations['softdrop_c1s_crit'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)

    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)

    fig_pdf.savefig('single_em_softdrop_plot_pdf.pdf',
                    format='pdf')

###########################################
# Multiple Emissions
###########################################

def plot_mc_softdrop_me(axes_pdf, axes_cdf, z_cut, beta=BETA,
                        beta_sd=0, icol=0):
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

    print("    Loading subsequent samples with beta="+str(beta)
          +" and z_c="+str(z_cut)+"...")
    c_subs = np.load(sample_file_path_sub)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = 1./(z_crits * -np.log(2.*z_cut))

    obs = z_crits * theta_crits**beta + c_subs

    # Weights, binned observables, and area
    sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.8),
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

def compare_softdrop_me():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('Soft Drop', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, beta=BETA, beta_sd=0,
                               jet_type=JET_TYPE,
                               icol=icol, label=r'$z_{\rm cut}=$'+str(z_cut),
                               acc='Multiple Emissions')

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_softdrop_me(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations['softdrop_c1s_all'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)

    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic (Many Emissions)',
                     errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)

    fig_pdf.savefig('multiple_em_softdrop_plot_pdf.pdf',
                    format='pdf')
###########################################
# Main:
###########################################
if __name__ == '__main__':
    compare_softdrop()
    compare_softdrop_me()
