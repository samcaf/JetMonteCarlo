# Local utilities for comparison
from examples.comparison_plots.comparison_event_gen import *
from examples.utils.plot_comparisons import *

# Local analytics
from jetmontecarlo.analytics.radiators.running_coupling import *
from jetmontecarlo.analytics.radiators.fixedcoupling import *
from jetmontecarlo.analytics.sudakov_factors.fixedcoupling import *

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, rad, z_cut, beta = BETA, icol=0):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if BIN_SPACE == 'lin':
        x_vals = np.linspace(0., 1., NUM_MC_EVENTS)
    else:
        x_vals = np.logspace(np.log10(EPSILON), 0, NUM_MC_EVENTS)

    cdf_vals = np.exp(-rad(x_vals))

    theta_crits = inverse_transform_samples(cdf_vals, x_vals, NUM_MC_EVENTS)
    theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

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
                           label=r'$z_c$='+str(z_cut))

    x_vals_pdf = np.linspace(.1, .4, len(Z_CUTS))
    x_vals_cdf = np.linspace(.1, .4, len(Z_CUTS))
    if BIN_SPACE == 'log':
        x_vals_pdf = np.array([6e-3, 1e-2, 2.5*1e-2, 6e-2])
        x_vals_cdf = np.logspace(-7, -2, len(Z_CUTS))
    labelLines(axes_pdf[0].get_lines(), xvals=x_vals_pdf)
    labelLines(axes_cdf[0].get_lines(), xvals=x_vals_cdf)

    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_crit(axes_pdf, axes_cdf, CRIT_RADIATORS[i],
                     z_cut, BETA, icol=i)
        shower_correlations = get_ps_ECFs(JET_LIST, 'crit', z_cut, BETA)
        plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')

    extra_label = ''
    if BIN_SPACE == 'log':
        extra_label = '_{:.0e}cutoff'.format(EPSILON)
    extra_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_crit_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
###########################################
# Subsequent Emissions
###########################################
def plot_mc_sub(axes_pdf, axes_cdf, rad_sub, icol=0):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if BIN_SPACE == 'lin':
        x_vals = np.linspace(0., .5, NUM_MC_EVENTS)
    else:
        x_vals = np.logspace(np.log10(EPSILON), np.log10(.5),
                             NUM_MC_EVENTS)

    cdf_vals_sub = np.exp(-rad_sub(x_vals))

    c_subs = inverse_transform_samples(cdf_vals_sub, x_vals,
                                       NUM_MC_EVENTS)
    c_subs = np.where(np.isinf(c_subs), 0, c_subs)

    obs = c_subs

    # Weights, binned observables, and area
    sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                      NUM_BINS)
    sud_integrator.hasBins = True
    sud_integrator.setDensity(obs, np.ones(len(obs)), 1.)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr

    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_sub():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('ungroomed', ratio_plot=False)

    # Analytic plot
    for icol, beta in enumerate(BETAS):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut=0, beta=beta,
                           icol=icol, label=r'$\beta$='+str(beta))

    x_vals_pdf = np.linspace(.1, .4, len(BETAS))
    x_vals_cdf = np.linspace(.1, .4, len(BETAS))
    if BIN_SPACE == 'log':
        x_vals_pdf = np.array([1e-3, 2e-2, 6e-2])
        x_vals_cdf = np.logspace(-3, -1.5, len(BETAS))
    labelLines(axes_pdf[0].get_lines(), xvals=x_vals_pdf)
    labelLines(axes_cdf[0].get_lines(), xvals=x_vals_cdf)

    for i, beta in enumerate(BETAS):
        plot_mc_sub(axes_pdf, axes_cdf, SUB_RADIATORS[i], icol=i)
        shower_correlations = get_ps_ECFs(JET_LIST, 'sub', z_cut=0.,
                                          beta=beta, few_emissions=False)
        plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')

    extra_label = ''
    if BIN_SPACE == 'log':
        extra_label = '_{:.0e}cutoff'.format(EPSILON)
    extra_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_sub_only_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_sub_only_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')

###########################################
# Critical and Subsequent Emissions
###########################################
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, rad_crit, rad_sub,
                         z_cut, beta = BETA, icol=0):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if BIN_SPACE == 'lin':
        x_vals_crit = np.linspace(0., 1., NUM_MC_EVENTS)
        x_vals_sub = np.linspace(0., .5, NUM_MC_EVENTS)
    else:
        x_vals_crit = np.logspace(np.log10(EPSILON), 0, NUM_MC_EVENTS)
        x_vals_sub = np.logspace(np.log10(EPSILON), np.log10(.5),
                                 NUM_MC_EVENTS)

    cdf_vals_crit = np.exp(-rad_crit(x_vals_crit))
    cdf_vals_sub = np.exp(-rad_sub(x_vals_sub))

    theta_crits = inverse_transform_samples(cdf_vals_crit, x_vals_crit,
                                            NUM_MC_EVENTS)
    theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)
    # plt.figure()
    # plt.hist(np.log10(theta_crits), NUM_BINS)
    # plt.title("thetas")

    c_subs = inverse_transform_samples(cdf_vals_sub, x_vals_sub,
                                       NUM_MC_EVENTS)
    c_subs = np.where(np.isinf(c_subs), 0, c_subs)
    c_subs = c_subs * theta_crits**BETA

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = 1./(z_crits * -np.log(2.*z_cut))

    obs_old = C_groomed(z_crits, theta_crits, z_cut, BETA,
                        z_pre=0., f=F_SOFT, acc=ACC)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, BETA,
                        z_pre=0., f=F_SOFT, acc=ACC)
    obs = c_crits + c_subs #np.maximum(c_crits, c_subs)

    # Weights, binned observables, and area
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

def compare_crit_and_sub():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit and sub', ratio_plot=False)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut, icol=icol,
                           label=r'$z_c$='+str(z_cut))

    x_vals_pdf = np.linspace(.1, .4, len(Z_CUTS))
    x_vals_cdf = np.linspace(.1, .4, len(Z_CUTS))
    if BIN_SPACE == 'log':
        x_vals_pdf = np.array([6e-3, 1e-2, 2.5*1e-2, 6e-2])
        x_vals_cdf = np.logspace(-7, -2, len(Z_CUTS))
    labelLines(axes_pdf[0].get_lines(), xvals=x_vals_pdf)
    labelLines(axes_cdf[0].get_lines(), xvals=x_vals_cdf)

    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_crit_and_sub(axes_pdf, axes_cdf,
                             CRIT_RADIATORS[i], SUB_RADIATORS[0],
                             z_cut, BETA, icol=i)
        shower_correlations = get_ps_ECFs(JET_LIST, 'critsub', z_cut, BETA,
                                          few_emissions=True)
        plot_shower_pdf_cdf(shower_correlations, axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic (crit)', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic (crit)', errtype='yerr',
                     twosigma=False, extralabel='Shower')

    extra_label = ''
    if BIN_SPACE == 'log':
        extra_label = '_{:.0e}cutoff'.format(EPSILON)
    extra_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_crit_and_sub_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(extra_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_crit_and_sub_'+BIN_SPACE+'_cdf_comp'
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
    # For each value of epsilon we want to use as an integration cutoff:
    if COMPARE_CRIT:
        compare_crit()
    if COMPARE_SUB:
        compare_sub()
    if COMPARE_CRIT_AND_SUB:
        compare_crit_and_sub()
