# Local utilities for comparison
from examples.generate_MC.partonshower import *
from examples.comparison_plots.comparison_plot_utils import *

# Local analytics
from jetmontecarlo.analytics.soft_drop import *

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
# MC events
NUM_MC_EVENTS = int(1e7)
LOAD_MC_EVENTS = True
SAVE_MC_EVENTS = True

# Choosing which emissions to plot
COMPARE_CRIT = False
COMPARE_SUB = False
COMPARE_CRIT_AND_SUB = False
COMPARE_PRE_AND_CRIT = False
COMPARE_ALL = False

if FIXED_COUPLING:
    extra_label = '_fc_an_'
else:
    extra_label = '_rc_an_'

###########################################
# Critical Emission Only
###########################################
def plot_mc_softdrop(axes_pdf, axes_cdf, z_cut, beta=BETA,
                     beta_sd=0, icol=0,
                     load=True):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

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

    if load:
        if sample_file_path.is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...")
            theta_crits = np.load(sample_file_path)
        else:
            load = False

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...")
        def cdf_crit(theta):
            if FIXED_COUPLING:
                return np.exp(-critRadAnalytic_fc_LL(theta, z_cut, JET_TYPE))
            else:
                return np.exp(-critRadAnalytic(theta, z_cut, JET_TYPE))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0,1])
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(sample_file_path, theta_crits)

    if FIXED_COUPLING:
        z_crits = np.array([getLinSample(z_cut, 1./2.)
                            for i in range(NUM_MC_EVENTS)])
        weights = 1./(z_crits * -np.log(2.*z_cut))
    else:
        pass

    obs = z_crits * theta_crits**beta

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


def compare_softdrop():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('Soft Drop', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, beta=BETA, beta_sd=0,
                               jet_type=JET_TYPE,
                               icol=icol, label=r'$z_c$='+str(z_cut),
                               acc='LL')

    x_vals_pdf = np.linspace(.1, .4, len(Z_CUTS))
    x_vals_cdf = np.linspace(.1, .4, len(Z_CUTS))
    if BIN_SPACE == 'log':
        x_vals_pdf = np.array([6e-3, 1e-2, 2.5*1e-2, 6e-2])
        x_vals_cdf = np.logspace(-7, -2, len(Z_CUTS))
    labelLines(axes_pdf[0].get_lines(), xvals=x_vals_pdf)
    labelLines(axes_cdf[0].get_lines(), xvals=x_vals_cdf)

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_softdrop(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        shower_correlations = getECFs_softdrop(JET_LIST, z_cut, BETA,
                                               beta_sd=0., acc='LL')
        plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)

    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')

    extra_label = '_ancdf'
    if BIN_SPACE == 'log':
        extra_label += '_{:.0e}cutoff'.format(EPSILON)
    extra_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_softdrop_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_softdrop_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')

###########################################
# Multiple Emissions
###########################################

def plot_mc_softdrop_me(axes_pdf, axes_cdf, z_cut, beta=BETA,
                        beta_sd=0, icol=0,
                        load=True):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

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

    if load:
        if sample_file_path_crit.is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...")
            theta_crits = np.load(sample_file_path_crit)
        else:
            load = False

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...")
        def cdf_crit(theta):
            if FIXED_COUPLING:
                return np.exp(-critRadAnalytic_fc_LL(theta, z_cut, JET_TYPE))
            else:
                return np.exp(-critRadAnalytic(theta, z_cut, JET_TYPE))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0,1])
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(sample_file_path_crit, theta_crits)

    # File path
    sample_file_sub = ("c_subs_from_crits"
                       +"_zc"+str(z_cut)
                       +"_beta"+str(beta)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    sample_file_path_sub = sample_folder / sample_file_sub

    if load:
        if sample_file_path_sub.is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...")
            c_subs = np.load(sample_file_path_sub)
        else:
            load = False

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...")
        c_subs = []
        for theta in theta_crits:
            def cdf_sub_conditional(c_sub):
                if FIXED_COUPLING:
                    return np.exp(-subRadAnalytic_fc_LL(c_sub, beta, JET_TYPE,
                                                        maxRadius=theta))
                else:
                    return np.exp(-subRadAnalytic(c_sub, beta, JET_TYPE,
                                                  maxRadius=theta))

            c_sub = samples_from_cdf(cdf_sub_conditional, 1,
                                     domain=[0,theta**beta/2.])[0]
            c_subs.append(c_sub)
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(sample_file_path_sub, c_subs)

    if FIXED_COUPLING:
        z_crits = np.array([getLinSample(z_cut, 1./2.)
                            for i in range(NUM_MC_EVENTS)])
        weights = 1./(z_crits * -np.log(2.*z_cut))
    else:
        pass

    obs = z_crits * theta_crits**beta + c_subs

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .8, NUM_BINS)
    if BIN_SPACE == 'log':
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
                               icol=icol, label=r'$z_c$='+str(z_cut),
                               acc='Multiple Emissions')

    x_vals_pdf = np.linspace(.1, .4, len(Z_CUTS))
    x_vals_cdf = np.linspace(.1, .4, len(Z_CUTS))
    if BIN_SPACE == 'log':
        x_vals_pdf = np.array([6e-3, 1e-2, 2.5*1e-2, 6e-2])
        x_vals_cdf = np.logspace(-7, -2, len(Z_CUTS))
    labelLines(axes_pdf[0].get_lines(), xvals=x_vals_pdf)
    labelLines(axes_cdf[0].get_lines(), xvals=x_vals_cdf)

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_softdrop(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        shower_correlations = getECFs_softdrop(JET_LIST, z_cut, BETA,
                                               beta_sd=0., acc='ME')
        plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)

    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')

    extra_label = '_ancdf'
    if BIN_SPACE == 'log':
        extra_label += '_{:.0e}cutoff'.format(EPSILON)
    extra_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_softdrop_me_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_softdrop_me_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')

###########################################
# Main:
###########################################
if __name__ == '__main__':
    # compare_softdrop()
    compare_softdrop_me()
