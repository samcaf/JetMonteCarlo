# Local jet utilities
from jetmontecarlo.utils.montecarlo_utils import getLinSample
from jetmontecarlo.numerics.observables import *

# Local analytics
from jetmontecarlo.analytics.radiators.running_coupling import *
from jetmontecarlo.analytics.radiators.fixedcoupling import *
from jetmontecarlo.analytics.sudakov_factors.fixedcoupling import *
from jetmontecarlo.montecarlo.partonshower import *

# Parameters and plotting utilities
from examples.params import ALL_MONTECARLO_PARAMS,\
    RADIATOR_PARAMS, SHOWER_PARAMS
from examples.utils.plot_comparisons import *

from examples.sudakov_comparisons.sudakov_utils import Z_CUT_PLOT
from examples.sudakov_comparisons.sudakov_utils import plot_label

# =====================================
# Notes:
# =====================================
# =====================================
# To Do
# =====================================
# ------------------------------------
# Physics
# ------------------------------------
# Run with fixed coupling, LL, with 5e6 samples, 5e3 bins
# Run with running coupling, MLL, with 5e6

# =====================================
# Done:
# =====================================
# ------------------------------------
# Physics
# ------------------------------------
# Run with fixed coupling, LL, with 1e6 samples, 5e3 bins
    # * Subsequent emissions not behaving well -- not strictly increasing cdf
    #   interpolation function
# Run with fixed coupling, LL, with 5e6 samples, 5e3 bins
    # * Critical emission distributions don't look right...

# =====================================
# Definitions and Parameters
# =====================================
params          = ALL_MONTECARLO_PARAMS

radiator_params = RADIATOR_PARAMS
del radiator_params['z_cut']
del radiator_params['beta']

shower_params = SHOWER_PARAMS

# ---------------------------------
# Unpacking parameters
# ---------------------------------
jet_type = params['jet type']
fixed_coupling = params['fixed coupling']

num_mc_events = params['number of MC events']

num_rad_bins = params['number of bins']


# =====================================
# Critical Emission Only
# =====================================
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                 load=LOAD_MC_EVENTS, verbose=5):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=F_SOFT, acc=OBS_ACC)

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights

    if verbose > 1:
        arg = np.argmax(obs)
        print("zc: " + str(z_cut))
        print("obs_acc: " + OBS_ACC)
        print("maximum observable: " + str(obs[arg]))
        print("associated with\n    z = "+str(z_crits[arg])
              +"\n    theta = "+str(theta_crits[arg]))
        print('', flush=True)

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
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
        if not plot_approx:
            plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                               beta, icol=icol, jet_type='quark',
                               f_soft=F_SOFT,
                               label=r'$z_{\rm cut}=$'+str(z_cut))
        else:
            plot_crit_approx(axes_pdf, axes_cdf, z_cut,
                             beta, icol=icol,
                             label=r'$z_{\rm cut}=$'+str(z_cut),
                             multiple_em=False)

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...", flush=True)
    for i, z_cut in enumerate(Z_CUT_PLOT):
        plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, icol=i)
        plot_shower_pdf_cdf(ps_correlations(beta)['rss_c1s_crit'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)

    lightlabel='Analytic f.c. crit'
    if plot_approx:
        lightlabel = 'Approx. f.c. crit'

    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel=lightlabel, errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel=lightlabel, errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_crit_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_crit_'
                        +BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            num_shower_events,  num_mc_events)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

# =====================================
# Ungroomed Emissions
# =====================================
def plot_mc_ungroomed(axes_pdf, axes_cdf, beta, icol=0,
                load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    c_raws, c_raw_weights = get_c_raw(beta, load=True, save=True,
                                      rad_raw=radiators.get('ungroomed'))

    obs = c_raws
    weights = c_raw_weights

    # Weights, binned observables, and area
    sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                      NUM_BINS)
    sud_integrator.hasBins = True
    sud_integrator.setDensity(obs, weights, 1.)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr

    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_ungroomed():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('ungroomed', ratio_plot=False, ylim=ylim_3)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, beta in enumerate(BETAS):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut=0,
                           beta=beta, icol=icol, jet_type='quark',
                           f_soft=F_SOFT,
                           label=r'$\beta=$'+str(beta),
                           fixed_coupling=FIXED_COUPLING)

    leg1 = axes_pdf[0].legend(loc=(0.019,.35), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.35), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting ungroomed samples...", flush=True)
    for icol, beta in enumerate(BETAS):
        plot_mc_ungroomed(axes_pdf, axes_cdf, beta, icol=icol)
        plot_shower_pdf_cdf(ps_correlations(beta)['ungroomed_c1s'],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=icol)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic', errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_ungroomed_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_ungroomed_'
                        +BIN_SPACE+'_cdf_comp'
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            num_shower_events,  num_mc_events)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

# =====================================
# Critical and Subsequent Emissions
# =====================================
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                         load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
                         load=load, save=True, theta_crits=theta_crits,
                         rad_crit_sub=radiators.get('subsequent', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=0., f=F_SOFT, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights * c_sub_weights

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_crit_and_sub(beta=BETA):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit and sub', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUT_PLOT):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                           beta, icol=icol, jet_type='quark',
                           f_soft=F_SOFT,
                           label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical and subsequent samples...", flush=True)

    for i, z_cut in enumerate(Z_CUT_PLOT):
        plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta, icol=i)
        plot_shower_pdf_cdf(ps_correlations(beta)['rss_c1s_critsub'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic f.c. crit', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic f.c. crit', errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_crit_and_sub_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_crit_and_sub_'
                        +BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            num_shower_events,  num_mc_events)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

# =====================================
# Pre + Critical Emissions
# =====================================
def plot_mc_pre_and_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                         load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=True,
                        theta_crits=theta_crits,
                        rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = c_crits

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights * z_pre_weights

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_pre_and_crit(beta=BETA):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('pre and crit', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUT_PLOT):
        plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                           beta, icol=icol, jet_type='quark',
                           f_soft=F_SOFT,
                           label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical and pre-critical samples...", flush=True)

    for i, z_cut in enumerate(Z_CUT_PLOT):
        plot_mc_pre_and_crit(axes_pdf, axes_cdf, z_cut, beta, icol=i)
        plot_shower_pdf_cdf(ps_correlations(beta)['rss_c1s_precrit'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel='Analytic f.c. crit', errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel='Analytic f.c. crit', errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_pre_and_crit_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_pre_and_crit_'
                        +BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events,  num_mc_events)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

# =====================================
# All Emissions
# =====================================
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    theta_crits, theta_crit_weights, load = get_theta_crits(
                          z_cut, beta, load=load, save=True,
                          rad_crit=radiators.get('critical', None))

    c_subs, c_sub_weights, load = get_c_subs(z_cut, beta,
                         load=load, save=True, theta_crits=theta_crits,
                         rad_crit_sub=radiators.get('subsequent', None))

    z_pres, z_pre_weights, load = get_z_pres(z_cut, load=load, save=True,
                        theta_crits=theta_crits,
                        rad_pre=radiators.get('pre-critical', None))

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(num_mc_events)])

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

    weights = split_fn_num(z_crits, theta_crits, z_cut)
    weights *= theta_crit_weights * c_sub_weights * z_pre_weights

    # Weights, binned observables, and area
    if BIN_SPACE == 'lin':
        sud_integrator.bins = np.linspace(0, .5, NUM_BINS)
        sud_integrator.binspacing = 'lin'
    if BIN_SPACE == 'log':
        sud_integrator.bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5),
                                          NUM_BINS)
        sud_integrator.binspacing = 'log'
    sud_integrator.hasBins = True

    sud_integrator.setDensity(obs, weights, 1./2.-z_cut)
    sud_integrator.integrate()

    pdf = sud_integrator.density
    pdferr = sud_integrator.densityErr
    integral = sud_integrator.integral
    integralerr = sud_integrator.integralErr

    plot_mc_pdf(axes_pdf, pdf, pdferr, sud_integrator.bins, icol)
    plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

def compare_all(beta=BETA, plot_approx=False):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('all', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUT_PLOT):
        if not plot_approx:
            plot_crit_analytic(axes_pdf, axes_cdf, z_cut,
                               beta, icol=icol, jet_type='quark',
                               f_soft=F_SOFT,
                               label=r'$z_{\rm cut}=$'+str(z_cut))
        else:
            plot_crit_approx(axes_pdf, axes_cdf, z_cut,
                             beta, icol=icol,
                             label=r'$z_{\rm cut}=$'+str(z_cut),
                             multiple_em=True)

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting all emissions samples...", flush=True)

    for i, z_cut in enumerate(Z_CUT_PLOT):
        plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, icol=i)
        plot_shower_pdf_cdf(ps_correlations(beta)['rss_c1s_two'][i],
                            axes_pdf, axes_cdf,
                            label='Parton Shower', colnum=i)
    # Saving plots
    lightlabel = 'Analytic f.c. crit'
    if plot_approx:
        lightlabel = 'Approx. Many Em.'

    legend_darklight(axes_pdf[0], darklabel='Integration',
                     lightlabel=lightlabel, errtype='modstyle',
                     twosigma=False, extralabel='Shower')
    legend_darklight(axes_cdf[0], darklabel='Integration',
                     lightlabel=lightlabel, errtype='yerr',
                     twosigma=False, extralabel='Shower')
    axes_pdf[0].add_artist(leg1)
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_all_em_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        num_shower_events, num_mc_events)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(str(fig_folder) + '/' + JET_TYPE+'_RSS_all_em_'
                        +BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                          num_shower_events,  num_mc_events)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

# =====================================
# Main:
# =====================================
if __name__ == '__main__':
    # For each value of epsilon we want to use as an integration cutoff:
    if COMPARE_CRIT:
        compare_crit(plot_approx=False)
    if COMPARE_RAW:
        compare_ungroomed()
    if COMPARE_CRIT_AND_SUB:
        compare_crit_and_sub()
    if COMPARE_PRE_AND_CRIT:
        compare_pre_and_crit()
    if COMPARE_ALL:
        compare_all(plot_approx=False)
