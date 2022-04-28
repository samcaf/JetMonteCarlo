from __future__ import absolute_import
import dill as pickle
from pathlib import Path

from scipy.misc import derivative

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
# Definitions and Parameters
###########################################
# ------------------------------------
# Parameters for plotting
# ------------------------------------
F_SOFT_PLOT = [.5, 1]
F_SOFT_PLOT_IVS = [.5, 1, 'ivs']
F_SOFT_STR = ['1/2', '1']

save_cdf = False

#f_colors = {.5: OrangeShade3, 1: GreenShade2, 'ivs': PurpleShade2}
f_colors = {.5: 'goldenrod', 1: 'forestgreen', 'ivs': 'darkmagenta'}

plot_colors = {k: {
                'fc': adjust_lightness(f_colors[k], 1),
                'num': adjust_lightness(f_colors[k], .75),
                'shower': adjust_lightness(f_colors[k], .75),
                'pythia': adjust_lightness(f_colors[k], .5)
                } for k in f_colors.keys()}

def f_ivs(theta):
    return 1./2. - theta/(np.pi*R0)
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

if MULTIPLE_EMISSIONS:
    extra_label += 'ME_'
    plot_label += 'ME_'

plot_label += '_showerbeta'+str(SHOWER_BETA)
if F_SOFT:
    plot_label += '_f{}'.format(F_SOFT)

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


# ==========================================
# Loading Files
# ==========================================
# ------------------------------------
# Parton shower files
# ------------------------------------
# Correlation files
def ps_correlations(beta, f_soft, verbose=5):
    # Getting filenames using proxy shower:
    shower_beta = beta # SHOWER_BETA if FIXED_COUPLING else beta
    shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                           shower_cutoff=SHOWER_CUTOFF,
                           shower_beta=shower_beta,
                           jet_type=JET_TYPE)
    shower.num_events = NUM_SHOWER_EVENTS
    ps_file = shower.correlation_path(beta, OBS_ACC,
                                      few_pres=True, f_soft=f_soft,
                                      angular_ordered=ANGULAR_ORDERING,
                                      info=SHOWER_INFO)
    if verbose > 0:
        print("    Loading parton shower data from:", ps_file)
    try:
        ps_data = np.load(ps_file, allow_pickle=True)
    except FileNotFoundError:
        print("    Trying to load data from file:", ps_file)
        print("    Params given to parton shower:")
        print("        NUMBER OF EVENTS:", NUM_SHOWER_EVENTS)
        print("        FIXED COUPLING:", FIXED_COUPLING)
        print("        SHOWER_CUTOFF:", SHOWER_CUTOFF)
        print("        SHOWER_BETA:", shower_beta)
        print("        OBSERVABLE ACCURACY:", OBS_ACC)
        print("        BETA:", beta)
        print("        F_RSS:", f_soft)
        print("        JET_TYPE:", JET_TYPE)
        print("    (Few pre-critical emissions)")
    if verbose > 3:
        print("        ps_data.keys():", ps_data.keys())
        print("        len(ps_data['softdrop_c1s_crit']):",
              len(ps_data['softdrop_c1s_crit']))
        print("        len(ps_data['softdrop_c1s_two']):",
              len(ps_data['softdrop_c1s_two']))

        print("        len(ps_data['softdrop_c1s_all']):",
              len(ps_data['softdrop_c1s_all']))

    return ps_data

# Pythia Data
#pythiafile = open('pythiadata/groomed_pythia_obs.pkl', 'rb')
#pythiadata = pickle.load(pythiafile)
#pythiafile.close()
rss_data = {'partons': {}, 'hadrons': {}, 'charged': {}}
softdrop_data = {'partons': {}, 'hadrons': {}, 'charged': {}}
raw_data = {'partons': {}, 'hadrons': {}, 'charged': {}}

for level in ['partons', 'hadrons', 'charged']:
    # Raw
    raw_file = open('pythiadata/raw_Zq_pT3TeV_noUE_'+level+'.pkl', 'rb')
    this_raw = pickle.load(raw_file)
    raw_data[level] = this_raw
    raw_file.close()

    # Softdrop
    for i in range(6):
        softdrop_file = open('pythiadata/softdrop_Zq_pT3TeV_noUE_param'+str(i)+'_'+level+'.pkl', 'rb')
        this_softdrop = pickle.load(softdrop_file)
        softdrop_data[level][this_softdrop['params']] = this_softdrop
        softdrop_file.close()

    # RSS
    for i in range(9):
        rss_file = open('pythiadata/rss_Zq_pT3TeV_noUE_param'+str(i)+'_'+level+'.pkl', 'rb')
        this_rss = pickle.load(rss_file)
        rss_data[level][this_rss['params']] = this_rss
        rss_file.close()

pythia_data = {'raw': raw_data, 'softdrop': softdrop_data, 'rss': rss_data}


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
    beta=float(2)
    crit_sample_file = ("theta_crits"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    return sample_folder / crit_sample_file

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
    if True in [COMPARE_CRIT, COMPARE_PRE_AND_CRIT,
                COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        print("    Loading critical radiator from "
              +str(critrad_path)+"...", flush=True)
        with open(critrad_path, 'rb') as file:
            rad_crit_list = pickle.load(file)
        global rad_crit
        def rad_crit(theta, z_cut):
            return rad_crit_list[INDEX_ZC[z_cut]](theta)

        # Using scipy to get radiator (linear) derivatives
        # for multiple emissions computations.
        def deriv_rad_crit(theta, z_cut):
            this_rad = lambda t: rad_crit(t, z_cut)
            try:
                this_drad = derivative(this_rad, theta, dx=theta * 1e-5)
            except RuntimeWarning:
                print("theta:", theta)
                print("this_drad", derivative(this_rad, theta, dx=theta * 1e-5))
            return this_drad

        global cdf_crit
        def cdf_crit(theta, z_cut):
            me_factor = 1.
            if MULTIPLE_EMISSIONS:
                # Getting multiple emissions expression for the CDF
                me_factor = np.exp(euler_constant * theta
                                   * deriv_rad_crit(theta, z_cut))
            return me_factor * np.exp(-1.*rad_crit(theta, z_cut))



    if COMPARE_SUB:
        print("    Loading subsequent/ungroomed radiator from "
              +str(subrad_path)+"...", flush=True)
        with open(subrad_path_path, 'rb') as file:
            rad_sub_list = pickle.load(file)
        global rad_sub
        def rad_sub(c_sub, beta):
            return rad_sub_list[INDEX_BETA[beta]](c_sub)

        def deriv_rad_sub(c_sub, beta):
            this_rad = lambda c, beta: rad_sub(c, beta)
            this_drad = derivative(this_rad, csub, dx=csub * 1e-5)
            return this_drad

        global cdf_sub
        def cdf_sub(c_sub):
            me_factor = 1.
            if MULTIPLE_EMISSIONS:
                me_factor = np.exp(euler_constant * c_sub
                                   * deriv_rad_sub(c_sub))
            return me_factor * np.exp(-1.*rad_sub(c_sub))

    if True in [COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        print("    Loading critical/subsequent radiator from "
              +str(subrad_path)+"...", flush=True)
        global rad_crit_sub
        with open(subrad_path, 'rb') as file:
            rad_crit_sub_list = pickle.load(file)
        # DEBUG: Is this right? No z_cut dependence?
        print("LENGTH OF RAD CRIT SUB LIST:", len(rad_crit_sub_list))
        rad_crit_sub = rad_crit_sub_list[0]

        if MULTIPLE_EMISSIONS:
            def deriv_rad_crit_sub(c_sub, theta):
                this_rad = lambda c: rad_crit_sub(c, theta)
                this_drad = derivative(this_rad, c_sub, dx=c_sub * 1e-5)
                return this_drad

        global cdf_sub_conditional
        def cdf_sub_conditional(c_sub, theta):
            me_factor = 1.
            if MULTIPLE_EMISSIONS:
                me_factor = np.exp(euler_constant * c_sub
                                   * deriv_rad_crit_sub(c_sub, theta))
            return me_factor * np.exp(-1.*rad_crit_sub(c_sub, theta))




    if True in [COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        print("    Loading pre-critical radiator from "
              +str(prerad_path)+"...", flush=True)
        with open(prerad_path, 'rb') as file:
            rad_pre_list = pickle.load(file)
        global rad_pre
        def rad_pre(z_pre, theta, z_cut):
            return rad_pre_list[INDEX_ZC[z_cut]](z_pre, theta)
        
        def deriv_rad_pre(z_pre, theta, z_cut):
            this_rad = lambda z: rad_pre(z, theta, z_cut)
            this_drad = derivative(this_rad, z_pre, dx=z_pre * 1e-5)
            return this_drad

        global cdf_pre_conditional
        def cdf_pre_conditional(z_pre, theta, z_cut):
            me_factor = 1.
            if MULTIPLE_EMISSIONS:
                me_factor = np.exp(euler_constant * z_pre
                                   * deriv_rad_pre(z_pre, theta, z_cut)) 
            return me_factor * np.exp(-1.*rad_pre(z_pre, theta, z_cut))



if not(LOAD_MC_EVENTS):
    load_radiators()

###########################################
# Additional Plot Utils
###########################################
def plot_mc_banded(ax, ys, err, bins, label, col):
    if BIN_SPACE == 'lin':
        xs = (bins[:-1] + bins[1:])/2.
        xerr = (bins[1:] - bins[:-1])
    if BIN_SPACE == 'log':
        xs = np.sqrt(bins[:-1]*bins[1:])
        xerr = (xs - bins[:-1], bins[1:]-xs)
        ys = xs * ys * np.log(10) # dY / d log10 C
        err = xs * err * np.log(10) # delta( dY / d log10 C)

    line = ax.plot(xs, ys, ls='-', lw=2., color=col, label=label)
    # DEBUG: Some weird features in some plots, look like they could be
    #        misplaced bands
    band = draw_error_band(ax, xs, ys, err, color=col, alpha=.4)

    return line, band

def full_legend(ax, labels, loc='upper left'):
    ax.plot(-100, -100, **style_dashed, color=compcolors[(-1, 'medium')],
            label=labels[0])
    line, band = plot_mc_banded(ax, [-100,-100], [1,1], np.array([-100,-99,-98]),
                                label=labels[1], col=compcolors[(-1, 'dark')])
    ax.errorbar(-100., -100, yerr=1., xerr=1., **modstyle,
                color=compcolors[(-1, 'dark')],
                label=labels[2])
    ax.hist(np.arange(-100,-90), 5,
            histtype='step', lw=2, edgecolor=compcolors[(-1, 'dark')],
            label=labels[3])

    handles, _ = ax.get_legend_handles_labels()
    new_handles = [handles[0], handles[1], handles[3], handles[2]]

    ax.legend(new_handles, labels, loc=loc)

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                 load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            print("    Unable to find file "+str(crit_sample_file_path(z_cut, beta)),
                  flush=True)
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS,
                                       domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                    z_pre=0., f=f_soft, acc=OBS_ACC)

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

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf, 2.*pdferr,
                                      sud_integrator.bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral, integralerr,
                                      sud_integrator.bins,
                                      label=None, col=col)

    return pdfline, pdfband, cdfline, cdfband



###########################################
# All Emissions
###########################################
def plot_mc_all(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            print("    Unable to find file "+str(crit_sample_file_path(z_cut, beta)),
                  flush=True)
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS, 
                                       domain=[0.,1.],
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
            print("    Unable to find file "+str(crit_sub_sample_file_path(z_cut, beta)))
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_sub(c_sub):
                return cdf_sub_conditional(c_sub, theta)

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
            else:
                c_sub = samples_from_cdf(this_cdf_sub, 1,
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
            print("    Unable to find file "+str(pre_sample_file_path(z_cut, beta)))
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)

        z_pres = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_pre(z_pre):
                return cdf_pre_conditional(z_pre, theta, z_cut)

            z_pre = samples_from_cdf(this_cdf_pre, 1,
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
                        z_pre=z_pres, f=f_soft, acc=OBS_ACC)
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

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf, 2.*pdferr,
                                      sud_integrator.bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral, integralerr,
                                      sud_integrator.bins,
                                      label=None, col=col)

    return pdfline, pdfband, cdfline, cdfband

###########################################
# IVS
###########################################
def plot_mc_ivs(axes_pdf, axes_cdf, z_cut, beta, f_soft, col,
                emissions='crit',
                load=LOAD_INV_SAMPLES):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            print("    Unable to find file "+str(crit_sample_file_path(z_cut, beta)))
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS, 
                                       domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if load and emissions=='all':
        if crit_sub_sample_file_path(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            c_subs = np.load(crit_sub_sample_file_path(z_cut, beta))
        else:
            print("    Unable to find file "+str(crit_sub_sample_file_path(z_cut, beta)))
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if emissions=='all' and not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_sub(c_sub):
                return cdf_sub_conditional(c, theta)

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
            else:
                c_sub = samples_from_cdf(this_cdf_sub, 1,
                                     domain=[0.,theta**beta/2.],
                                     verbose=3)[0]
            c_subs.append(c_sub)
            if (i+1)%(len(theta_crits)/10) == 0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(crit_sub_sample_file_path(z_cut, beta), c_subs)

    if emissions=='all' and load:
        if pre_sample_file_path(z_cut).is_file():
            print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            z_pres = np.load(pre_sample_file_path(z_cut))
        else:
            print("    Unable to find file "+str(pre_sample_file_path(z_cut, beta)))
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if emissions=='all' and not load:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)

        z_pres = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_pre(z_pre):
                return cdf_pre_conditional(z_pre, theta, z_cut)

            z_pre = samples_from_cdf(this_cdf_pre, 1,
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

    if emissions=='crit':
        obs = C_groomed(z_crits, theta_crits, z_cut, beta,
                        z_pre=0, f=f_ivs(theta_crits),
                        acc=OBS_ACC)
    elif emissions=='all':
        c_crits = C_groomed(z_crits, theta_crits, z_cut, beta,
                            z_pre=z_pres, f=f_ivs(theta_crits),
                            acc=OBS_ACC)
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

    pdfline, pdfband = plot_mc_banded(axes_pdf[0], pdf, 2.*pdferr,
                                      sud_integrator.bins,
                                      label=None, col=col)
    cdfline, cdfband = plot_mc_banded(axes_cdf[0], integral, integralerr,
                                      sud_integrator.bins,
                                      label=None, col=col)

    return pdfline, pdfband, cdfline, cdfband


"""
###########################################
# Subsequent Emissions
###########################################
def plot_mc_sub(axes_pdf, axes_cdf, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])
    if LOAD_INV_SAMPLES:
        if sub_sample_file_path(beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+"...",
                  flush=True)
            c_subs = np.load(sub_sample_file_path(beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        def this_cdf_sub(c_sub):
            return rad_sub(c_sub, beta)

        c_subs = samples_from_cdf(this_cdf_sub, NUM_MC_EVENTS, domain=[0.,.5],
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


###########################################
# Critical and Subsequent Emissions
###########################################
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if LOAD_INV_SAMPLES:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...", flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making critical samples with z_c="+str(z_cut)+"...", flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS, 
                                       domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if LOAD_INV_SAMPLES:
        if crit_sub_sample_file_path(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            c_subs = np.load(crit_sub_sample_file_path(z_cut, beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()


    if not LOAD_INV_SAMPLES:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_sub(c_sub):
                return cdf_sub_conditional(c, theta)

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
            else:
                c_sub = samples_from_cdf(this_cdf_sub, 1,
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

###########################################
# Pre + Critical Emissions
###########################################
def plot_mc_pre_and_crit(axes_pdf, axes_cdf, z_cut, beta):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if LOAD_INV_SAMPLES:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)
        def this_cdf_crit(theta):
            return cdf_crit(theta, z_cut)

        theta_crits = samples_from_cdf(this_cdf_crit, NUM_MC_EVENTS, domain=[0.,1.],
                                       verbose=3)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if LOAD_INV_SAMPLES:
        if pre_sample_file_path(z_cut).is_file():
            print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            z_pres = np.load(pre_sample_file_path(z_cut))
        else:
            LOAD_INV_SAMPLES = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not LOAD_INV_SAMPLES:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)

        z_pres = []

        for i, theta in enumerate(theta_crits):
            def this_cdf_pre(z_pre):
                return cdf_pre_conditional(z, theta, z_cut)

            z_pre = samples_from_cdf(this_cdf_pre, 1,
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

"""
