from __future__ import absolute_import
import dill as pickle
from pathlib import Path

# Local utilities for comparison
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.jets.observables import *
from examples.comparison_plot_utils import *

# Local analytics
from jetmontecarlo.analytics.radiators import *
from jetmontecarlo.analytics.radiators_fixedcoupling import *
from jetmontecarlo.analytics.sudakovFactors_fixedcoupling import *
from jetmontecarlo.montecarlo.partonshower import *

# Parameters
from examples.params import *

###########################################
# Notes:
###########################################
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

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Parameters for plotting
# ------------------------------------
Z_CUT_PLOT = [.05, .1, .2]
F_SOFT = 1
Z_CUT_PLOT = [F_SOFT * zc for zc in Z_CUT_PLOT]

save_cdf = False

# ------------------------------------
# Comparison parameters
# ------------------------------------
# Choosing which emissions to plot
COMPARE_CRIT = True
COMPARE_SUB = False
COMPARE_CRIT_AND_SUB = True
COMPARE_PRE_AND_CRIT = True
COMPARE_ALL = True

if FIXED_COUPLING:
    extra_label = '_fc_num_'
    plot_label = '_fc_num_'+str(OBS_ACC)
else:
    extra_label = '_rc_num_'
    plot_label = '_rc_num_'+str(OBS_ACC)

plot_label += '_showerbeta'+str(SHOWER_BETA)
if F_SOFT:
    plot_label += '_f{}'.format(F_SOFT)

# ==========================================
# Loading Files
# ==========================================
# ------------------------------------
# Parton shower files
# ------------------------------------

# Correlation files
def ps_correlations(beta):
    # Getting filenames using proxy shower:
    shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                           shower_cutoff=SHOWER_CUTOFF,
                           shower_beta=SHOWER_BETA if FIXED_COUPLING else beta,
                           jet_type=JET_TYPE)
    shower.num_events = NUM_SHOWER_EVENTS
    ps_file = shower.correlation_path(int(beta), OBS_ACC, few_pres=True,
                                     f_soft=F_SOFT,
                                     angular_ordered=ANGULAR_ORDERING)
    ps_data = np.load(ps_file, allow_pickle=True)

    return ps_data

# ------------------------------------
# MC integration files:
# ------------------------------------
# Splitting function file path:
with open(splitfn_path, 'rb') as f:
    splitting_fns = pickle.load(f)
# Index of z_cut values in the splitting function file
def split_fn_num(z, theta, z_cut):
    return splitting_fns[INDEX_ZC[z_cut]](z, theta)

# Sample file paths:
sample_folder = Path("jetmontecarlo/utils/samples/inverse_transform_samples")

def crit_sample_file_path(z_cut, beta):
    beta=float(beta)
    crit_sample_file = ("theta_crits"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    return sample_folder / crit_sample_file
print(crit_sample_file_path(.1, 2))

def sub_sample_file_path(beta):
    beta=float(beta)
    sub_sample_file = ("c_subs"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                       +"_beta"+str(beta)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    return sample_folder / sub_sample_file

def crit_sub_sample_file_path(z_cut, beta):
    beta=float(beta)
    crit_sub_sample_file = ("c_subs_from_crits"
                            +"_obs"+str(OBS_ACC)
                            +"_splitfn"+str(SPLITFN_ACC)
                            +"_zc"+str(z_cut)
                            +"_beta"+str(beta)
                            +"_{:.0e}".format(NUM_MC_EVENTS)
                            +extra_label
                            +"samples.npy")
    return sample_folder / crit_sub_sample_file

def pre_sample_file_path(z_cut):
    pre_sample_file = ("z_pres_from_crits"
                       +"_obs"+str(OBS_ACC)
                       +"_splitfn"+str(SPLITFN_ACC)
                       +"_zc"+str(z_cut)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    return sample_folder / pre_sample_file

# ------------------------------------
# Loading Radiators:
# ------------------------------------
def load_radiators():
    print("Loading pickled radiator functions:")
    print("    Loading critical radiator from "
          +str(critrad_path)+"...", flush=True)
    if True in [COMPARE_CRIT, COMPARE_PRE_AND_CRIT,
                COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        with open(critrad_path, 'rb') as file:
            rad_crit_list = pickle.load(file)
        global rad_crit
        def rad_crit(theta, z_cut):
            return rad_crit_list[INDEX_ZC[z_cut]](theta)

    if COMPARE_SUB:
        print("    Loading subsequent/ungroomed radiator from "
              +str(subrad_path)+"...", flush=True)
        with open(subrad_path_path, 'rb') as file:
            rad_sub_list = pickle.load(file)
        global rad_sub
        def rad_sub(c_sub, beta):
            return rad_sub_list[INDEX_BETA[beta]](c_sub)

    if True in [COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        print("    Loading critical/subsequent radiator from "
              +str(subrad_path)+"...", flush=True)
        global rad_crit_sub
        with open(subrad_path, 'rb') as file:
            rad_crit_sub = pickle.load(file)[0]

    if True in [COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        print("    Loading pre-critical radiator from "
              +str(prerad_path)+"...", flush=True)
        with open(prerad_path, 'rb') as file:
            rad_pre_list = pickle.load(file)
        global rad_pre
        def rad_pre(z_pre, theta, z_cut):
            return rad_pre_list[INDEX_ZC[z_cut]](z_pre, theta)

if not(LOAD_MC_EVENTS):
    load_radiators()

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                 load=LOAD_MC_EVENTS, verbose=5):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=F_SOFT, acc=OBS_ACC)

    if verbose > 1:
        arg = np.argmax(obs)
        print("zc: " + str(z_cut))
        print("obs_acc: " + OBS_ACC)
        print("maximum observable: " + str(max(obs)))
        print("associated with\n    z = "+str(z_crits[arg])
              +"\n    theta = "+str(theta_crits[arg]))
        print('', flush=True)

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

    fig_pdf.savefig(JET_TYPE+'_RSS_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(JET_TYPE+'_RSS_crit_'+BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')
    
    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

###########################################
# Subsequent Emissions
###########################################
def plot_mc_sub(axes_pdf, axes_cdf, beta, icol=0,
                load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])
    if load:
        print(sub_sample_file_path(beta))
        if sub_sample_file_path(beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+"...",
                  flush=True)
            c_subs = np.load(sub_sample_file_path(beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        def cdf_sub(c_sub):
            return np.exp(-1.*rad_sub(c_sub, beta))

        c_subs = samples_from_cdf(cdf_sub, NUM_MC_EVENTS, domain=[0.,.5],
                                  verbose=3)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(sub_sample_file_path(beta), c_subs)

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

    print("Getting subsequent samples...", flush=True)
    for icol, beta in enumerate(BETAS):
        plot_mc_sub(axes_pdf, axes_cdf, beta, icol=icol)
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

    fig_pdf.savefig(JET_TYPE+'_ungroomed_'+BIN_SPACE+'_pdf_comp'
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(JET_TYPE+'_ungroomed_'+BIN_SPACE+'_cdf_comp'
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

###########################################
# Critical and Subsequent Emissions
###########################################
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                         load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...", flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...", flush=True)

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if load:
        if crit_sub_sample_file_path(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            c_subs = np.load(crit_sub_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def cdf_sub_conditional(c_sub):
                return np.exp(-1.*rad_crit_sub(c_sub, theta))

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
            else:
                c_sub = samples_from_cdf(cdf_sub_conditional, 1,
                                     domain=[0.,theta**beta/2.],
                                     verbose=3)[0]
            c_subs.append(c_sub)
            if (i+1)%(len(theta_crits)/10)==0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(crit_sub_sample_file_path(z_cut, beta), c_subs)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=0., f=F_SOFT, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

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

    fig_pdf.savefig(JET_TYPE+'_RSS_crit_and_sub_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(JET_TYPE+'_RSS_crit_and_sub_'+BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                            NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

###########################################
# Pre + Critical Emissions
###########################################
def plot_mc_pre_and_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                         load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if load:
        if pre_sample_file_path(z_cut).is_file():
            print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            z_pres = np.load(pre_sample_file_path(z_cut))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)

        z_pres = []

        for i, theta in enumerate(theta_crits):
            def cdf_pre_conditional(z_pre):
                return np.exp(-1.*rad_pre(z_pre, theta, z_cut))

            z_pre = samples_from_cdf(cdf_pre_conditional, 1,
                                     domain=[0.,z_cut], verbose=3)[0]
            z_pres.append(z_pre)
            if (i+1)%(len(theta_crits)/10)==0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        z_pres = np.array(z_pres)
        z_pres = np.where(np.isinf(z_pres), 0, z_pres)
        np.save(pre_sample_file_path(z_cut), z_pres)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = c_crits

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

    fig_pdf.savefig(JET_TYPE+'_RSS_pre_and_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(JET_TYPE+'_RSS_pre_and_crit_'+BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

###########################################
# All Emissions
###########################################
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)

        with open(critrad_path, 'rb') as file:
            rad_crit = pickle.load(file)[INDEX_ZC[z_cut]]

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if load:
        if crit_sub_sample_file_path(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            c_subs = np.load(crit_sub_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def cdf_sub_conditional(c_sub):
                return np.exp(-1.*rad_crit_sub(c_sub, theta))

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
            else:
                c_sub = samples_from_cdf(cdf_sub_conditional, 1,
                                     domain=[0.,theta**beta/2.],
                                     verbose=3)[0]
            c_subs.append(c_sub)
            if (i+1)%(len(theta_crits)/10) == 0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(crit_sub_sample_file_path(z_cut, beta), c_subs)

    if load:
        if pre_sample_file_path(z_cut).is_file():
            print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            z_pres = np.load(pre_sample_file_path(z_cut))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)

        z_pres = []

        for i, theta in enumerate(theta_crits):
            def cdf_pre_conditional(z_pre):
                return np.exp(-1.*rad_pre(z_pre, theta, z_cut))

            z_pre = samples_from_cdf(cdf_pre_conditional, 1,
                                     domain=[0,z_cut],
                                     verbose=3)[0]
            z_pres.append(z_pre)
            if (i+1)%(len(theta_crits)/10) == 0:
                print("        Generated "+str(i+1)+" events...",
                      flush=True)
        z_pres = np.array(z_pres)
        z_pres = np.where(np.isinf(z_pres), 0, z_pres)
        np.save(pre_sample_file_path(z_cut), z_pres)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=z_pres, f=F_SOFT, acc=OBS_ACC)
    obs = np.maximum(c_crits, c_subs)

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

    fig_pdf.savefig(JET_TYPE+'_RSS_all_em_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(beta)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    if save_cdf:
        fig_cdf.savefig(JET_TYPE+'_RSS_all_em_'+BIN_SPACE+'_cdf_comp'
                        +'_beta'+str(beta)
                        +'_{:.0e}showers_{:.0e}mc'.format(
                          NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                        +str(this_plot_label)
                        +'.pdf',
                        format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!", flush=True)

###########################################
# Main:
###########################################
if __name__ == '__main__':
    # For each value of epsilon we want to use as an integration cutoff:
    if COMPARE_CRIT:
        compare_crit(plot_approx=False)
    if COMPARE_SUB:
        compare_sub()
    if COMPARE_CRIT_AND_SUB:
        compare_crit_and_sub()
    if COMPARE_PRE_AND_CRIT:
        compare_pre_and_crit()
    if COMPARE_ALL:
        compare_all(plot_approx=False)
