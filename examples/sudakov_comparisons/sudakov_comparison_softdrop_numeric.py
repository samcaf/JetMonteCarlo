from __future__ import absolute_import
import dill as pickle

# Local utilities for comparison
from jetmontecarlo.utils.montecarlo_utils import *
from jetmontecarlo.jets.observables import *
from examples.comparison_plot_utils import *

# Parameters
from examples.sudakov_comparisons.sudakov_utils import ps_correlations
from examples.params import *

# Local analytics
from jetmontecarlo.analytics.soft_drop import *

from pathlib import Path

###########################################
# Definitions and Parameters
###########################################
# ------------------------------------
# Physics inputs
# ------------------------------------

# Jet and grooming parameters
SD_Z_CUTS = [.05, .1, .2]
BETA = 2
BETAS = [1, 2, 3, 4]
JET_TYPE = 'quark'

# ------------------------------------
# Monte Carlo parameters
# ------------------------------------
# MC events
NUM_MC_EVENTS = int(1e6)

NUM_RAD_EVENTS = int(1e6)
NUM_RAD_BINS = int(5e3)

NUM_SPLITFN_EVENTS = int(5e6)
NUM_SPLITFN_BINS = int(5e3)

# Choosing which emissions to plot
COMPARE_CRIT = True
COMPARE_CRIT_AND_SUB = False
COMPARE_PYTHIA = (SPLITFN_ACC == 'MLL')

########### Turnign this on for tests with fcll
COMPARE_PYTHIA = True

if FIXED_COUPLING:
    extra_label = '_fc_num_'
    plot_label = '_fc_num_'+str(OBS_ACC)+'_'
else:
    extra_label = '_rc_num_'
    plot_label = '_rc_num_'+str(OBS_ACC)+'_'


# ------------------------------------
# MC paths
# ------------------------------------
# File folders
rad_folder = Path("output/serialized_functions/radiators/")

# Radiator paths:
rad_extension = ("_{:.0e}events".format(NUM_RAD_EVENTS)
                 +"_{:.0e}bins".format(NUM_RAD_BINS)
                 +".pkl")
splitfn_extension = ("_{:.0e}events".format(NUM_SPLITFN_EVENTS)
                     +"_{:.0e}bins".format(NUM_SPLITFN_BINS)
                     +".pkl")
if not FIXED_COUPLING:
    rad_extension = '_rc' + rad_extension
    splitfn_extension = '_'+SPLITFN_ACC+'_log_rc' + splitfn_extension

crit_rad_file_path = rad_folder / ("crit_{}_rads".format(BIN_SPACE)
                                   + rad_extension)
sub_rad_file_path = rad_folder / ("sub_{}_rads".format(BIN_SPACE)
                                  + rad_extension)
crit_sub_rad_file_path = rad_folder / ("crit_sub_{}_rads".format(BIN_SPACE)
                                       + rad_extension)
pre_rad_file_path = rad_folder / ("pre_{}_rads".format(BIN_SPACE)
                                  + rad_extension)

# ------------------------------------
# MC integration files:
# ------------------------------------
# Splitting function file path:
print("splitfn_extension:", splitfn_extension)
with open(splitfn_path, 'rb') as f:
    splitting_fns = pickle.load(f)
# Index of z_cut values in the splitting function file
def split_fn_num(z, theta, z_cut):
    return splitting_fns[INDEX_ZC[z_cut]](z, theta)


# Sample file paths:
sample_folder = Path("output/montecarlo_samples/sudakov_functions")

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
            if VERBOSE > 5:
                print("  zcut:", z_cut)
                print("  INDEX_ZC[z_cut]:", INDEX_ZC[z_cut])
            return rad_crit_list[INDEX_ZC[z_cut]](theta)

    print("    Loading critical/subsequent radiator from "
          +str(subrad_path)+"...", flush=True)
    global rad_crit_sub
    with open(subrad_path, 'rb') as file:
        rad_crit_sub = pickle.load(file)[0]

if not(LOAD_MC_EVENTS):
    load_radiators()


# Pythia Data
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

pythia_data = {'raw': raw_data, 'softdrop': softdrop_data}

print("Params keys:", list(pythia_data['softdrop']['partons'].keys()), flush=True)


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
    # plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

    return pdf, pdferr

crit_pdfs, crit_pdferrs = [], []

def compare_crit(pdfs=None, pdferrs=None,
                 num_ems='crit', show_pythia_level=None):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(SD_Z_CUTS):
        plot_softdrop_analytic(axes_pdf, axes_cdf, BIN_SPACE,
                               z_cut, BETA, beta_sd=0,
                               jet_type=JET_TYPE, acc=OBS_ACC,
                               fixed_coupling=FIXED_COUPLING,
                               icol=icol, label=r'$z_{\rm cut}=$'+str(z_cut))

    leg1 = axes_pdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    leg2 = axes_cdf[0].legend(loc=(0.019,.445), prop={'size': 15})
    for icol, text in enumerate(leg1.get_texts()):
        text.set_color(compcolors[(icol, 'dark')])

    print("Getting critical samples...")
    for i, z_cut in enumerate(SD_Z_CUTS):
        if pdfs is not None:
            bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5), NUM_BINS)
            plot_mc_pdf(axes_pdf, pdfs[i], pdferrs[i], bins, icol=i)
        else:
            pdf, err = plot_mc_crit(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
            crit_pdfs.append(pdf)
            crit_pdferrs.append(err)

        if show_pythia_level is None:
            shower_correlations = ps_correlations(BETA, 1)['softdrop_c1s_'+num_ems][PS_INDEX_ZC[z_cut]]
        else:
            # Narrowing in on jets with P_T between 3 and 3.5 TeV
            inds = np.where(
               3000 < np.array(pythia_data['raw'][plot_level]['pt'][beta]) *
               np.array(pythia_data['raw'][plot_level]['pt'][beta]) < 3500
               )[0]

            # Getting substructure
            shower_correlations = pythia_data['softdrop'][show_pythia_level][(0.0, z_cut, 1.0)]['C1'][BETA][inds]
        plot_shower_pdf_cdf(shower_correlations,
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
    if show_pythia_level is not None:
        this_plot_label += 'PYTHIA'+show_pythia_level+'_'
    else:
        this_plot_label += ''+num_ems+'psEms'+'_'
    if BIN_SPACE == 'log':
        this_plot_label += '{:.0e}mccutoff_'.format(EPSILON)
    this_plot_label += '{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(fig_folder / JET_TYPE+'_softdrop_crit_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    # fig_cdf.savefig(fig_folder / JET_TYPE+'_softdrop_crit_'
    #                 +BIN_SPACE+'_cdf_comp'
    #                 +'_beta'+str(BETA)
    #                 +'_{:.0e}showers_{:.0e}mc'.format(
    #                     NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
    #                 +str(this_plot_label)
    #                 +'.pdf',
    #                 format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

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
    # plot_mc_cdf(axes_cdf, integral, integralerr, sud_integrator.bins, icol)

    return pdf, pdferr

crit_sub_pdfs, crit_sub_pdferrs = [], []

def compare_crit_and_sub(pdfs=None, pdferrs=None,
                         num_ems='two', show_pythia_level=None):
    # Preparing figures
    fig_pdf, axes_pdf, fig_cdf, axes_cdf = \
            get_axes('crit and sub', ratio_plot=False)
    if FIXED_COUPLING:
        axes_pdf[0].set_ylim(0, .2)

    # Analytic plot
    for icol, z_cut in enumerate(SD_Z_CUTS):
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

    for i, z_cut in enumerate(SD_Z_CUTS):
        if pdfs is not None:
            bins = np.logspace(np.log10(EPSILON)-1, np.log10(.5), NUM_BINS)
            plot_mc_pdf(axes_pdf, pdfs[i], pdferrs[i], bins, icol=i)
        else:
            pdf, err = plot_mc_crit_and_sub(axes_pdf, axes_cdf, z_cut, BETA, icol=i)
            crit_sub_pdfs.append(pdf)
            crit_sub_pdferrs.append(err)

        if show_pythia_level is None:
            shower_correlations = ps_correlations(BETA, 1)['softdrop_c1s_'+num_ems][PS_INDEX_ZC[z_cut]]
        else:
            # Narrowing in on jets with P_T between 3 and 3.5 TeV
            cond_floor = (3000 < np.array(pythia_data['raw'][plot_level]['pt'][beta]))
            cond_ceil = (np.array(pythia_data['raw'][plot_level]['pt'][beta]) < 3500)
            inds = np.where(cond_floor * cond_ceil)[0]

            # Getting substructure
            shower_correlations = pythia_data['softdrop'][show_pythia_level][(0.0, z_cut, 1.0)]['C1'][BETA]
            shower_correlations = np.array(shower_correlations)[inds]
        plot_shower_pdf_cdf(shower_correlations,
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
    if show_pythia_level is not None:
        this_plot_label += '_PYTHIA'+show_pythia_level
    else:
        this_plot_label += '_'+num_ems+'psEms'
    if BIN_SPACE == 'log':
        this_plot_label += '_{:.0e}cutoff'.format(EPSILON)
    this_plot_label += '_{:.0e}shower'.format(SHOWER_CUTOFF)

    fig_pdf.savefig(fig_folder / JET_TYPE+'_softdrop_crit_and_sub_'
                    +BIN_SPACE+'_pdf_comp'
                    +'_beta'+str(BETA)
                    +'_{:.0e}showers_{:.0e}mc'.format(
                        NUM_SHOWER_EVENTS, NUM_MC_EVENTS)
                    +str(this_plot_label)
                    +'.pdf',
                    format='pdf')
    # fig_cdf.savefig(fig_folder / JET_TYPE+'_softdrop_crit_and_sub_'
    #                 +BIN_SPACE+'_cdf_comp'
    #                 +'_beta'+str(BETA)
    #                 +'_{:.0e}showers_{:.0e}mc'.format(
    #                     NUM_SHOWER_EVENTS,  NUM_MC_EVENTS)
    #                 +str(this_plot_label)
    #                 +'.pdf',
    #                 format='pdf')

    plt.close(fig_pdf)
    plt.close(fig_cdf)

    print("Plotting complete!")

###########################################
# Main:
###########################################
if __name__ == '__main__':
    if COMPARE_CRIT:
        compare_crit()
        compare_crit(pdfs=crit_pdfs, pdferrs=crit_pdferrs,
                     num_ems='two')
        compare_crit(pdfs=crit_pdfs, pdferrs=crit_pdferrs,
                     num_ems='all')
        if COMPARE_PYTHIA:
            compare_crit(pdfs=crit_pdfs, pdferrs=crit_pdferrs,
                         show_pythia_level='partons')
            compare_crit(pdfs=crit_pdfs, pdferrs=crit_pdferrs,
                         show_pythia_level='hadrons')
            compare_crit(pdfs=crit_pdfs, pdferrs=crit_pdferrs,
                         show_pythia_level='charged')
    if COMPARE_CRIT_AND_SUB:
        compare_crit_and_sub()
        compare_crit_and_sub(pdfs=crit_sub_pdfs, pdferrs=crit_sub_pdferrs,
                             num_ems='crit')
        compare_crit_and_sub(pdfs=crit_sub_pdfs, pdferrs=crit_sub_pdferrs,
                             num_ems='all')
        if COMPARE_PYTHIA:
            compare_crit_and_sub(pdfs=crit_sub_pdfs, pdferrs=crit_sub_pdferrs,
                                 show_pythia_level='partons')
            compare_crit_and_sub(pdfs=crit_sub_pdfs, pdferrs=crit_sub_pdferrs,
                                 show_pythia_level='hadrons')
            compare_crit_and_sub(pdfs=crit_sub_pdfs, pdferrs=crit_sub_pdferrs,
                                show_pythia_level='charged')
