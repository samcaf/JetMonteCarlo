import dill as pickle
from pathlib import Path

# Local utilities for comparison
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.jets.observables import *
from examples.comparison_plots.comparison_plot_utils import *

# Local analytics
from jetmontecarlo.analytics.soft_drop import *

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Physics inputs
# ------------------------------------
FIXED_COUPLING = False
OBS_ACC = 'LL' if FIXED_COUPLING else 'MLL'

ANGULAR_ORDERING = False

# Jet and grooming parameters
Z_CUTS = [.05, .1, .2]
Z_CUT = .1
BETA = 2
BETAS = [1, 2, 3, 4]
F_SOFT = 1.
JET_TYPE = 'quark'

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
BIN_SPACE = 'log'

# Shower info
NUM_SHOWER_EVENTS = int(5e5)
SHOWER_BETA = 2
SHOWER_CUTOFF = 1e-10 if FIXED_COUPLING else MU_NP

# MC events
NUM_MC_EVENTS = int(1e6)
LOAD_MC_EVENTS = True
SAVE_MC_EVENTS = True

NUM_RAD_EVENTS = int(1e6)
NUM_RAD_BINS = int(5e3)

NUM_SPLITFN_EVENTS = int(1e6)
NUM_SPLITFN_BINS = int(5e3)

# Choosing which emissions to plot
COMPARE_CRIT = True
COMPARE_CRIT_AND_SUB = True

if FIXED_COUPLING:
    extra_label = '_fc_num_'
    plot_label = '_fc_num_'+str(OBS_ACC)+'_'
else:
    extra_label = '_rc_num_'
    plot_label = '_rc_num_'+str(OBS_ACC)+'_'

# ------------------------------------
# Parton shower files
# ------------------------------------
ps_sample_folder = Path("jetmontecarlo/utils/samples/shower_correlations/")

def ps_correlations(beta):
    ps_file = 'shower_{:.0e}_c1_'.format(NUM_SHOWER_EVENTS)+str(beta)
    if ANGULAR_ORDERING:
        ps_file += '_angord'
    if FIXED_COUPLING and SHOWER_CUTOFF == 1e-20:
        ps_file += '_lowcutoff'
    elif not FIXED_COUPLING and SHOWER_CUTOFF == 1e-10:
        ps_file += '_lowcutoff'
    if SHOWER_BETA != 1.:
        ps_file += '_showerbeta'+str(SHOWER_BETA)
    if not FIXED_COUPLING and OBS_ACC=='MLL':
        ps_file += '_MLL_fewem.npz'
    elif not FIXED_COUPLING and OBS_ACC=='LL':
        ps_file += '_rc_LL_fewem.npz'
    else:
        ps_file += '_' + OBS_ACC
        # if few_emissions:
        ps_file += '_fewem.npz'
        # else:
        #     ps_file += '_manyem.npz'

    ps_data = np.load(ps_sample_folder / ps_file, allow_pickle=True)

    return ps_data

# ------------------------------------
# MC paths
# ------------------------------------
# File folders
rad_folder = Path("jetmontecarlo/utils/functions/radiators/")
sample_folder = Path("jetmontecarlo/utils/samples/"
                     +"inverse_transform_samples")

# Radiator paths:
rad_extension = ("_{:.0e}events".format(NUM_RAD_EVENTS)
                 +"_{:.0e}bins".format(NUM_RAD_BINS)
                 +".pkl")
splitfn_extension = ("_{:.0e}events".format(NUM_SPLITFN_EVENTS)
                     +"_{:.0e}bins".format(NUM_SPLITFN_BINS)
                     +".pkl")
if not FIXED_COUPLING:
    rad_extension = '_rc' + rad_extension
    splitfn_extension = '_rc' + splitfn_extension

crit_rad_file_path = rad_folder / ("crit_{}_rads".format(BIN_SPACE)
                                   + rad_extension)
sub_rad_file_path = rad_folder / ("sub_{}_rads".format(BIN_SPACE)
                                  + rad_extension)
crit_sub_rad_file_path = rad_folder / ("crit_sub_{}_rads".format(BIN_SPACE)
                                       + rad_extension)
pre_rad_file_path = rad_folder / ("pre_{}_rads".format(BIN_SPACE)
                                  + rad_extension)

splitfn_folder = Path("jetmontecarlo/utils/functions/splitting_fns/")
splitfn_file = 'split_fns' + splitfn_extension
splitfn_path = splitfn_folder / splitfn_file

with open(splitfn_path, 'rb') as f:
    splitting_fns = pickle.load(f)
# Index of z_cut values in the splitting function file
index_zc = {.05: 0, .1: 1, .2: 2}
def split_fn_num(z, theta, z_cut):
    return splitting_fns[index_zc[z_cut]](z, theta)

# Sample file paths:
def crit_sample_file_path(z_cut, beta):
    beta=float(beta)
    crit_sample_file = ("theta_crits"
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    return sample_folder / crit_sample_file

def crit_sub_sample_file_path(z_cut, beta):
    beta=float(beta)
    crit_sub_sample_file = ("c_subs_from_crits"
                            +"_zc"+str(z_cut)
                            +"_beta"+str(beta)
                            +"_{:.0e}".format(NUM_MC_EVENTS)
                            +extra_label
                            +"samples.npy")
    return sample_folder / crit_sub_sample_file

# ------------------------------------
# Loading Radiators:
# ------------------------------------
def load_radiators():
    print("Loading pickled radiator functions:")
    print("    Loading critical radiator...")
    index = {.05: 0, .1: 1, .2: 2}
    if True in [COMPARE_CRIT, COMPARE_CRIT_AND_SUB]:
        with open(crit_rad_file_path, 'rb') as file:
            rad_crit_list = pickle.load(file)
        global rad_crit
        def rad_crit(theta, z_cut):
            return rad_crit_list[index[z_cut]](theta)

    if COMPARE_CRIT_AND_SUB:
        print("    Loading critical/subsequent radiator...")
        global rad_crit_sub
        with open(crit_sub_rad_file_path, 'rb') as file:
            rad_crit_sub = pickle.load(file)[0]

if not(LOAD_MC_EVENTS):
    load_radiators()

###########################################
# Critical Emission Only
###########################################
def plot_mc_crit(axes_pdf, axes_cdf, z_cut, beta=BETA, icol=0,
                 load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...")
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...")

        index = {.05: 0, .1: 1, .2: 2}

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0,1])
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    if OBS_ACC == 'LL':
        obs = z_crits * theta_crits**beta
    elif OBS_ACC == 'MLL':
        obs = z_crits * (1.-z_crits) * theta_crits**beta

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

def compare_crit():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, BETA, beta_sd=0,
                               jet_type=JET_TYPE, acc='LL',
                               fixed_coupling=FIXED_COUPLING,
                               icol=icol, label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_crit(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations(BETA)['softdrop_c1s_crit'][i],
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

    fig_pdf.savefig(JET_TYPE+'_softdrop_crit_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_softdrop_crit_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    print("Plotting complete!")

###########################################
# Critical and Subsequent Emissions
###########################################
def plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, beta = BETA, icol=0,
                         load=LOAD_MC_EVENTS):
    sud_integrator = integrator()
    sud_integrator.setLastBinBndCondition([1., 'minus'])

    if load:
        if crit_sample_file_path(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...")
            theta_crits = np.load(crit_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...")

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits = samples_from_cdf(cdf_crit, NUM_MC_EVENTS, domain=[0,1])
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        np.save(crit_sample_file_path(z_cut, beta), theta_crits)

    if load:
        if crit_sub_sample_file_path(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...")
            c_subs = np.load(crit_sub_sample_file_path(z_cut, beta))
        else:
            load = False
            if LOAD_MC_EVENTS:
                load_radiators()

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...")
        c_subs = []

        for i, theta in enumerate(theta_crits):
            def cdf_sub_conditional(c_sub):
                return np.exp(-1.*rad_crit_sub(c_sub, theta))

            c_sub = samples_from_cdf(cdf_sub_conditional, 1,
                                     domain=[0,theta**beta/2.])[0]
            c_subs.append(c_sub)
            if (i+1)%(len(theta_crits)/10)==0:
                print("        Generated "+str(i+1)+" events...")
        c_subs = np.array(c_subs)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)
        np.save(crit_sub_sample_file_path(z_cut, beta), c_subs)

    z_crits = np.array([getLinSample(z_cut, 1./2.)
                        for i in range(NUM_MC_EVENTS)])
    weights = split_fn_num(z_crits, theta_crits, z_cut)

    if OBS_ACC == 'LL':
        c_crits = z_crits * theta_crits**beta
    elif OBS_ACC == 'MLL':
        c_crits = z_crits * (1.-z_crits) * theta_crits**beta
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

def compare_crit_and_sub():
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit and sub', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, BETA, beta_sd=0,
                               jet_type=JET_TYPE, acc='Multiple Emissions',
                               fixed_coupling=FIXED_COUPLING,
                               icol=icol, label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical and subsequent samples...")

    for i, z_cut in enumerate(Z_CUTS):
        plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
        plot_shower_pdf_cdf(ps_correlations(BETA)['softdrop_c1s_two'][i],
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
    axes_cdf[0].add_artist(leg2)

    this_plot_label = plot_label
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(JET_TYPE+'_softdrop_crit_and_sub_'+BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    fig_cdf.savefig(JET_TYPE+'_softdrop_crit_and_sub_'+BIN_SPACE+'_cdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    print("Plotting complete!")

###########################################
# Main:
###########################################
if __name__ == '__main__':
    # For each value of epsilon we want to use as an integration cutoff:
    if COMPARE_CRIT:
        compare_crit()
    if COMPARE_CRIT_AND_SUB:
        compare_crit_and_sub()
