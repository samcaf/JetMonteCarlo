from pathlib import Path

import numpy as np
import dill as pickle

# Data cataloging
import yaml

# Timing
from jetmontecarlo.utils.time_utils import timing

# Help with MC sampling
from jetmontecarlo.utils.montecarlo_utils import samples_from_cdf
from jetmontecarlo.montecarlo.partonshower import parton_shower

# Parameters for use in default arguments for filenames
from examples.params import *

# ===================================
# Folders
# ===================================
# Figure folder
fig_folder = Path('output/figures/current/')

# Monte Carlo Samples
sample_folder = Path("output/montecarlo_samples/")
# Numerical Integrals generated by MC
numerical_integral_folder = Path("output/numerical_integrals/")
# Serialized Interpolating Functions generated from numerical integrals
function_folder = Path("output/serialized_functions/")

# Phase Space Samples
phasespace_sampler_folder = sample_folder / "phase_space"

# Parton Shower Data
partonshower_folder = sample_folder / "parton_showers"

# Radiators
radiator_discrete_folder = numerical_int_folder / "radiators"
radiator_function_folder = function_folder / "radiators"

# Splitting Functions
splitting_function_folder = function_folder / "splitting_functions"

# Sudakov Factors
sudakov_sample_folder = sample_folder / "sudakov_functions"
sudakov_discrete_folder = numerical_int_folder / "sudakov_functions"
sudakov_function_folder = function_folder / "sudakov_functions"

# DEBUG: For use with catalog
def get_folder(data_type, data_name):
    assert data_type in ['samples', 'integral', 'function'],\
        f"Invalid data_type {data_type} must be one of 'samples', 'integral', or 'function'."
    if data_type == 'samples':
        folder = sample_folder
    if data_type == 'integral':
        folder = numerical_integral_folder
    if data_type = 'function':
        folder = function_folder


# =====================================
# MC Filenames
# =====================================
# ---------------------------------
# Catalog file:
# ---------------------------------
# Stores the filenames associated with data for each set of parameters
# DEBUG: Best practice, but unused
catalog = Path('output/catalog.yaml')

def new_cataloged_filename(data_type, params):
    """Add a new entry to the catalog file."""
    folder = get_folder(data_type, data_name)
    with open(catalog, 'r') as file:
        catalog_dict = yaml.safe_load(file)
        catalog_dict[data_type].update(dict(params, filename))
    if catalog_dict:
        with open(catalog, 'w') as file:
            yaml.safe_dump(catalog_dict, file)
    return filename

def filename_from_catalog(data_type, params):
    with open(catalog, 'r') as file:
        catalog_dict = yaml.safe_load(file)
    filename = catalog_dict[data_type].get(params)
    if filename is not None:
        return filename
    else:
        raise ValueError(f"Could not find {data_type} with params {params} in catalog.")


# ---------------------------------
# Sampler files
# ---------------------------------
critfile = 'crit_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
prefile = 'pre_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)
subfile = 'sub_samplers_{}_{:.0e}.pkl'.format(BIN_SPACE, NUM_MC_EVENTS)

# ---------------------------------
# Radiator files
# ---------------------------------
# Fixed Coupling
critradfile = 'crit_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
preradfile = 'pre_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
subradfile = 'sub_obs{}_split{}_{}_rads_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

splitfn_file = 'split_fns_{}_{}_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 SPLITFN_ACC, BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

# Running Coupling
if not FIXED_COUPLING:
    # Radiator files
    critradfile = 'crit_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
    preradfile = 'pre_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)
    subradfile = 'sub_obs{}_split{}_{}_rads_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                 OBS_ACC, SPLITFN_ACC,
                                                 BIN_SPACE,
                                                 NUM_MC_EVENTS, NUM_RAD_BINS)

    splitfn_file = 'split_fns_{}_{}_rc_{:.0e}events_{:.0e}bins.pkl'.format(
                                                     SPLITFN_ACC, BIN_SPACE,
                                                     NUM_MC_EVENTS, NUM_RAD_BINS)

if not COMPARE_RAW:
    # If we are not comparing ungroomed emissions, we remember that
    # the subsequent emissions are actually linked to the critical
    # emissions by angular ordering
    subfile = 'crit_' + subfile
    subradfile = 'crit_' + subradfile


# ===================================
# Sampler Files
# ===================================
critfile_path = phasespace_sampler_folder / critfile
prefile_path = phasespace_sampler_folder / prefile
subfile_path = phasespace_sampler_folder / subfile


# ===================================
# Radiator Files
# ===================================
# ---------------------------------
# Numerical Integrals (discrete)
# ---------------------------------
critrad_int_path = radiator_discrete_folder / critradfile
prerad_int_path = radiator_discrete_folder / preradfile
subrad_int_path = radiator_discrete_folder / subradfile

# ---------------------------------
# Serialized Functions (continuous)
# ---------------------------------
critrad_path = radiator_function_folder / critradfile
prerad_path = radiator_function_folder / preradfile
subrad_path = radiator_function_folder / subradfile


# =====================================
# Splitting Function Files
# =====================================
# ---------------------------------
# Serialized Functions (continuous)
# ---------------------------------
splitfn_path = splitting_function_folder / splitfn_file


# =====================================
# Sudakov Function Files
# =====================================
if FIXED_COUPLING:
    extra_label = '_fc_num_'
else:
    extra_label = '_rc_num_'

if MULTIPLE_EMISSIONS:
    extra_label += 'ME_'


# Samples generated via inverse transform from Sudakov Functions
def sudakov_raw_sample_file(beta):
    """Return the path to the samples generated via inverse transform
    from the ungroomed/raw Sudakov function.
    """
    beta=float(beta)
    raw_sample_file = ("c_raws"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                       +"_beta"+str(beta)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    if VERBOSE > 2:
        print("  raw sample file path:",
              sudakov_sample_folder / raw_sample_file)
    return sudakov_sample_folder / raw_sample_file

def sudakov_raw_weight_file(beta):
    """Return the path to the weights generated via inverse transform
    from the ungroomed/raw Sudakov function.
    """
    beta=float(beta)
    raw_weight_file = ("c_raw_weights"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                       +"_beta"+str(beta)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    if VERBOSE > 2:
        print("  raw sample file path:",
              sudakov_sample_folder / raw_weight_file)
    return sudakov_sample_folder / raw_weight_file


def sudakov_crit_sample_file(z_cut, beta):
    """Return the path to the samples generated via inverse transform
    from the critical Sudakov function.
    """
    beta = float(beta)
    crit_sample_file = ("theta_crits"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    if VERBOSE > 2:
        print("  crit sample file path:", sudakov_sample_folder / crit_sample_file)
    return sudakov_sample_folder / crit_sample_file

def sudakov_crit_weight_file(z_cut, beta):
    """Return the path to the samples generated via inverse transform
    from the critical Sudakov function.
    """
    beta = float(beta)
    crit_weight_file = ("theta_crit_weights"
                        +"_obs"+str(OBS_ACC)
                        +"_splitfn"+str(SPLITFN_ACC)
                        +"_zc"+str(z_cut)
                        +"_beta"+str(beta)
                        +"_{:.0e}".format(NUM_MC_EVENTS)
                        +extra_label
                        +"samples.npy")
    if VERBOSE > 2:
        print("  crit weight file path:",
              sudakov_sample_folder / crit_weight_file)
    return sudakov_sample_folder / crit_weight_file


def sudakov_crit_sub_sample_file(z_cut, beta):
    beta=float(beta)
    crit_sub_sample_file = ("c_subs_from_crits"
                            +"_obs"+str(OBS_ACC)
                            +"_splitfn"+str(SPLITFN_ACC)
                            +"_zc"+str(z_cut)
                            +"_beta"+str(beta)
                            +"_{:.0e}".format(NUM_MC_EVENTS)
                            +extra_label
                            +"samples.npy")
    if VERBOSE > 2:
        print("  crit sub sample file path:", sudakov_sample_folder / crit_sub_sample_file)
    return sudakov_sample_folder / crit_sub_sample_file

def sudakov_crit_sub_weight_file(z_cut, beta):
    beta=float(beta)
    crit_sub_weight_file = ("c_sub_weights_from_crits"
                            +"_obs"+str(OBS_ACC)
                            +"_splitfn"+str(SPLITFN_ACC)
                            +"_zc"+str(z_cut)
                            +"_beta"+str(beta)
                            +"_{:.0e}".format(NUM_MC_EVENTS)
                            +extra_label
                            +"samples.npy")
    if VERBOSE > 2:
        print("  crit sub weiight file path:",
              sudakov_sample_folder / crit_sub_weight_file)
    return sudakov_sample_folder / crit_sub_weight_file


def sudakov_pre_sample_file(z_cut):
    pre_sample_file = ("z_pres_from_crits"
                       +"_obs"+str(OBS_ACC)
                       +"_splitfn"+str(SPLITFN_ACC)
                       +"_zc"+str(z_cut)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    if VERBOSE > 2:
        print("  pre sample file path:", sudakov_sample_folder / pre_sample_file)
    return sudakov_sample_folder / pre_sample_file

def sudakov_pre_weight_file(z_cut):
    pre_weight_file = ("z_pre_weights_from_crits"
                       +"_obs"+str(OBS_ACC)
                       +"_splitfn"+str(SPLITFN_ACC)
                       +"_zc"+str(z_cut)
                       +"_{:.0e}".format(NUM_MC_EVENTS)
                       +extra_label
                       +"samples.npy")
    if VERBOSE > 2:
        print("  pre sample weight path:",
              sudakov_sample_folder / pre_weight_file)
    return sudakov_sample_folder / pre_weight_file


# ####################################
# Utilities for Loading Files and Functions
# ####################################

# =====================================
# Sudakov Sample Files
# =====================================
# Loading or generating samples generated through sudakov functions

# ---------------------------------
# Ungroomed
# ---------------------------------
@timing
def get_c_raw(beta, load=True, save=True, rad_raw=None):
    if load:
        if sudakov_raw_sample_file(beta).is_file():
            print("    Loading ungroomed samples with beta="+str(beta)+"...",
                  flush=True)
            # Loading files and samples:
            c_raws = np.load(sudakov_raw_sample_file(beta),
                                allow_pickle=True, mmap_mode='c')
            c_raw_weights = np.load(sudakov_raw_sample_file(beta),
                                allow_pickle=True, mmap_mode='c')
        else:
            load = False
            if LOAD_MC_EVENTS:
                if rad_raw is None:
                    radiators = get_radiator_functions()
                    try:
                        rad_raw = radiators['ungroomed']
                    except KeyError:
                        print("Unable to find ungroomed radiators")

    if not load:
        print("    Making ungroomed samples with beta="+str(beta)+"...",
              flush=True)

        if rad_raw is None:
            radiators = get_radiator_functions()
            try:
                rad_raw = radiators['ungroomed']
            except KeyError:
                print("Unable to find ungroomed radiators")

        def cdf_raw(c_raw):
            return np.exp(-1.*rad_raw(c_raw, beta))

        c_raws, c_raw_weights = samples_from_cdf(cdf_raw, NUM_MC_EVENTS,
                                        domain=[0.,.5],
                                        # DEBUG: No backup CDF
                                        backup_cdf=None,
                                        verbose=3)
        c_raw_weights = np.where(np.isinf(c_raws), 0,
                                 c_raw_weights)
        c_raws = np.where(np.isinf(c_raws), 0, c_raws)

        # Save samples and weights
        if save:
            np.save(sudakov_raw_sample_file(beta), c_raws)
            np.save(sudakov_raw_weight_file(beta), c_raw_weights)

    return c_raws, c_raw_weights

# ---------------------------------
# Critical
# ---------------------------------
@timing
def get_theta_crits(z_cut, beta, load=True, save=True,
                    rad_crit=None):
    if load:
        if sudakov_crit_sample_file(z_cut, beta).is_file():
            print("    Loading critical samples with z_c="+str(z_cut)+"...",
                  flush=True)
            # Loading files and samples:
            theta_crits = np.load(sudakov_crit_sample_file(z_cut, beta),
                                    allow_pickle=True, mmap_mode='c')
            theta_crit_weights = np.load(sudakov_crit_weight_file(z_cut, beta),
                                    allow_pickle=True, mmap_mode='c')
        else:
            load = False
            if rad_crit is None:
                radiators = get_radiator_functions()
                try:
                    rad_crit = radiators['critical']
                except KeyError as e:
                    print("Unable to find critical radiator function")
                    raise e

    if not load:
        print("    Making critical samples with z_c="+str(z_cut)+"...",
              flush=True)

        if rad_crit is None:
            radiators = get_radiator_functions()
            try:
                rad_crit = radiators['critical']
            except KeyError as e:
                print("Unable to find critical radiator function")
                raise e

        def cdf_crit(theta):
            return np.exp(-1.*rad_crit(theta, z_cut))

        theta_crits, theta_crit_weights = samples_from_cdf(cdf_crit, NUM_MC_EVENTS,
                                                domain=[0.,1.],
                                                backup_cdf=None,
                                                verbose=3)
        theta_crit_weights = np.where(np.isinf(theta_crits), 0,
                                      theta_crit_weights)
        theta_crits = np.where(np.isinf(theta_crits), 0, theta_crits)

        # Save samples and weights
        if save:
            np.save(sudakov_crit_sample_file(z_cut, beta),
                    theta_crits)
            np.save(sudakov_crit_weight_file(z_cut, beta),
                    theta_crit_weights)
    return theta_crits, theta_crit_weights, load


# ---------------------------------
# Subsequent
# ---------------------------------
@timing
def get_c_subs(z_cut, beta, load=True, save=True,
               theta_crits=None, rad_crit_sub=None):
    if load:
        if sudakov_crit_sub_sample_file(z_cut, beta).is_file():
            print("    Loading subsequent samples with beta="+str(beta)+
                  " from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            # Loading files and samples:
            c_subs = np.load(sudakov_crit_sub_sample_file(z_cut, beta),
                            allow_pickle=True, mmap_mode='c')
            c_sub_weights = np.load(sudakov_crit_sub_weight_file(z_cut, beta),
                            allow_pickle=True, mmap_mode='c')
        else:
            load = False
            if rad_crit_sub is None:
                radiators = get_radiator_functions()
                try:
                    rad_crit_sub = radiators['subsequent']
                except KeyError as e:
                    print("Unable to find subsequent radiator function")
                    raise e

    if not load:
        print("    Making subsequent samples with beta="+str(beta)+"...",
              flush=True)
        assert theta_crits is not None, "theta_crits must be provided"\
            + " to generate subsequent samples."

        if rad_crit_sub is None:
            radiators = get_radiator_functions()
            try:
                rad_crit_sub = radiators['subsequent']
            except KeyError as e:
                print("Unable to find subsequent radiator function")
                raise e

        c_subs = []
        c_sub_weights = []

        for i, theta in enumerate(theta_crits):
            def cdf_sub_conditional(c_sub):
                return np.exp(-1.*rad_crit_sub(c_sub, theta))

            if theta**beta/2. < 1e-10:
                # Assigning to an underflow bin for small observable values
                c_sub = 1e-100
                c_sub_weight = 1.
            else:
                c_sub, c_sub_weight = samples_from_cdf(cdf_sub_conditional, 1,
                                                domain=[0.,theta**beta/2.],
                                                # DEBUG: No backup CDF
                                                backup_cdf=None,
                                                verbose=3)
                c_sub, c_sub_weight = c_sub[0], c_sub_weight[0]
            c_subs.append(c_sub)
            c_sub_weights.append(c_sub_weight)
            if (i+1)%(len(theta_crits)/10) == 0:
                print("        Generated "+str(i+1)+" events...", flush=True)
        c_subs = np.array(c_subs)
        c_sub_weights = np.array(c_sub_weights)
        c_sub_weights = np.where(np.isinf(c_subs), 0, c_sub_weights)
        c_subs = np.where(np.isinf(c_subs), 0, c_subs)

        # Save samples and weights
        if save:
            np.save(sudakov_crit_sub_sample_file(z_cut, beta),
                    c_subs)
            np.save(sudakov_crit_sub_sample_file(z_cut, beta),
                    c_sub_weights)

    return c_subs, c_sub_weights, load


# ---------------------------------
# Pre-Critical
# ---------------------------------
@timing
def get_z_pres(z_cut, load=True, save=True,
               theta_crits=None, rad_pre=None):
    if load:
        if sudakov_pre_sample_file(z_cut).is_file():
            print("    Loading pre-critical samples"
                  +" from crit samples with z_cut="+str(z_cut)+"...",
                  flush=True)
            # Loading files and samples:
            z_pres = np.load(sudakov_pre_sample_file(z_cut),
                            allow_pickle=True, mmap_mode='c')
            z_pre_weights = np.load(sudakov_pre_weight_file(z_cut),
                            allow_pickle=True, mmap_mode='c')
        else:
            load = False
            if rad_pre is None:
                radiators = get_radiator_functions()
                try:
                    rad_pre = radiators['pre-critical']
                except KeyError as e:
                    print("Unable to find pre-critical radiator function")
                    raise e

    if not load:
        print("    Making pre-critical samples"
              +" from crit samples with z_cut="+str(z_cut)+"...",
              flush=True)
        assert theta_crits is not None, "theta_crits must be provided"\
            + " to generate pre-critical samples."

        if rad_pre is None:
            radiators = get_radiator_functions()
            try:
                rad_pre = radiators['pre-critical']
            except KeyError as e:
                print("Unable to find pre-critical radiator function")
                raise e

        z_pres = []
        z_pre_weights = []

        for i, theta in enumerate(theta_crits):
            def cdf_pre_conditional(z_pre):
                return np.exp(-1.*rad_pre(z_pre, theta, z_cut))

            z_pre, z_pre_weight = samples_from_cdf(cdf_pre_conditional, 1,
                                                domain=[0,z_cut],
                                                # DEBUG: No backup CDF
                                                backup_cdf=None,
                                                force_monotone=True,
                                                verbose=10)
            try:
                z_pre = z_pre[0]
                z_pre_weight = z_pre_weight[0]
            # DEBUG: I think I have fixed this bug, but if it reappears,
            #        I want an extra message
            except IndexError as e:
                print(f"IndexError using: {z_pre=}, {z_pre_weight=}")
                raise e
            z_pres.append(z_pre)
            z_pre_weights.append(z_pre_weight)
            if (i+1)%(len(theta_crits)/10) == 0:
                print("        Generated "+str(i+1)+" events...",
                      flush=True)
        z_pres = np.array(z_pres)
        z_pre_weights = np.array(z_pre_weights)
        z_pre_weights = np.where(np.isinf(z_pres), 0, z_pre_weights)
        z_pres = np.where(np.isinf(z_pres), 0, z_pres)

        # Save samples and weights
        if save:
            np.save(sudakov_pre_weight_file(z_cut),
                    z_pres)
            np.save(sudakov_pre_weight_file(z_cut),
                    z_pre_weights)

    return z_pres, z_pre_weights, load


# =====================================
# Parton shower files
# =====================================
# Correlation files
@timing
def ps_correlations(beta, f_soft=1):
    # Getting filenames using proxy shower:
    shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                           shower_cutoff=SHOWER_CUTOFF,
                           shower_beta=SHOWER_BETA if FIXED_COUPLING else beta,
                           jet_type=JET_TYPE)
    shower.num_events = NUM_SHOWER_EVENTS
    ps_file = shower.correlation_path(beta, OBS_ACC,
                                      few_pres=True, f_soft=f_soft,
                                      angular_ordered=ANGULAR_ORDERING,
                                      info=SHOWER_INFO)

    if VERBOSE > 0:
        print("    Loading parton shower data from:", ps_file)
    try:
        ps_data = np.load(ps_file, allow_pickle=True,
                          mmap_mode='c')
    except FileNotFoundError as error:
        print("    Trying to load data from file:", ps_file)
        print("    File not found.\n\n")
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
        raise error

    return ps_data


# =====================================
# Function Files
# =====================================
@timing
def get_splitting_function():
    """Load the splitting function for the given params"""
    with open (splitfn_path, 'rb') as file:
        splitting_fns = pickle.load(file)

    def split_fn_num(z, theta, z_cut):
        return splitting_fns[INDEX_ZC[z_cut]](z, theta)

    return split_fn_num


@timing
def get_radiator_functions():
    # Setup
    radiators = {'info': {}}
    print("Loading pickled radiator functions:")

    # Critical Radiator
    print("    Loading critical radiator from "
          +str(critrad_path)+"...", flush=True)
    if True in [COMPARE_CRIT, COMPARE_PRE_AND_CRIT,
                COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        with open(critrad_path, 'rb') as file:
            rad_crit_list = pickle.load(file)
        def rad_crit(theta, z_cut):
            if VERBOSE > 5:
                print("  zcut:", z_cut)
                print("  INDEX_ZC[z_cut]:", INDEX_ZC[z_cut])
            return rad_crit_list[INDEX_ZC[z_cut]](theta)
        radiators['critical'] = rad_crit
        radiators['info']['critical'] = 'Radiator for the angle, theta, of '\
            + 'the critical emission, or the first emission to survive'\
            + 'grooming. Dependent on the parameter z_cut.\nSyntax:\n'\
            + '`rad_crit = radiators[\'critical\']\nrad_crit(theta, z_cut)`'

    # Critical/Subsequent Radiator
    if True in [COMPARE_CRIT_AND_SUB, COMPARE_ALL]:
        print("    Loading critical/subsequent radiator from "
              +str(subrad_path)+"...", flush=True)
        with open(subrad_path, 'rb') as file:
            rad_crit_sub = pickle.load(file)[0]
        radiators['subsequent'] = rad_crit_sub
        radiators['info']['subsequent'] = 'Radiator for the jet ECF, C,'\
            + ' of the emissions after the critical emission, given the '\
            + 'angle, theta, of the critical emission.'\
            + '\nSyntax:\n`rad_crit_sub = radiators[\'subsequent\']\n'\
            + 'rad_crit_sub(C, theta)`'

    # Ungroomed Radiator
    if COMPARE_RAW:
        print("    Loading subsequent/ungroomed radiator from "
              +str(subrad_path)+"...", flush=True)
        with open(subrad_path_path, 'rb') as file:
            rad_sub_list = pickle.load(file)
        def rad_raw(c_sub, beta):
            return rad_sub_list[INDEX_BETA[beta]](c_sub)
        radiators['ungroomed'] = rad_raw
        radiators['info']['ungroomed'] = 'Radiator for the jet ECF, C,'\
            + ' of the emissions of an ungroomed jet.'\
            + '\nSyntax:\n`rad_sub = radiators[\'subsequent\']\n'\
            + 'rad_raw(C)`'

    if True in [COMPARE_PRE_AND_CRIT, COMPARE_ALL]:
        print("    Loading precritical radiator from "
              +str(prerad_path)+"...", flush=True)
        with open(prerad_path, 'rb') as file:
            rad_pre_list = pickle.load(file)
        def rad_pre(z_pre, theta, z_cut):
            return rad_pre_list[INDEX_ZC[z_cut]](z_pre, theta)
        radiators['pre-critical'] = rad_pre
        radiators['info']['pre-critical'] = 'Radiator for the energy '\
            + 'fraction, z_pre, of the emissions before the critical'\
            + ' emission, given the angle theta, of the critical'\
            + ' emission. \nSyntax:\n`rad_pre = radiators[\'pre-critical\']\n'\
            + 'rad_pre(z_pre, theta, z_cut)`'

    return radiators

@timing
def get_pythia_data(include=['raw', 'softdrop', 'rss']):
    # Dictionary of Pythia data
    if 'raw' in include:
        raw_data = {'partons': {}, 'hadrons': {}, 'charged': {}}
    if 'softdrop' in include:
        softdrop_data = {'partons': {}, 'hadrons': {}, 'charged': {}}
    if 'rss' in include:
        rss_data = {'partons': {}, 'hadrons': {}, 'charged': {}}

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

    pythia_data = {}
    if 'raw' in include:
        pythia_data['raw'] = raw_data
    if 'softdrop' in include:
        pythia_data['softdrop'] = softdrop_data
    if 'rss' in include:
        pythia_data['rss'] = rss_data

    return pythia_data