# Loading parton shower class and utilities to find ECFs/correlations:
from jetmontecarlo.montecarlo.partonshower import parton_shower
from jetmontecarlo.utils.partonshower_utils import getECFs_ungroomed
from jetmontecarlo.utils.partonshower_utils import getECFs_softdrop
from jetmontecarlo.utils.partonshower_utils import getECFs_rss

# File/data management
from examples.data_management import save_new_data, load_data

# Parameters
from examples.params import BETAS, Z_CUTS, SHOWER_PARAMS

# DEBUG: remove star import, use SHOWER_PARAMS
# from examples.params import *
# from examples.filenames import *

# =====================================
# Flags and Parameter setup
# =====================================
# Saving/loading flags
SAVE_SHOWER_EVENTS = False
LOAD_SHOWER_EVENTS = False
SAVE_SHOWER_CORRELATIONS = True

# Setting up parton shower parameters
fixed_coupling = SHOWER_PARAMS['fixed coupling']
jet_type = SHOWER_PARAMS['jet type']

num_shower_events = SHOWER_PARAMS['number of shower events']
shower_cutoff = SHOWER_PARAMS['shower cutoff']

# Setting up grooming parameters
z_cuts = Z_CUTS
beta_mmdt = 0
f_softs = [1, .75, .5]


# =====================================
# Utilities for saving/loading correlations
# =====================================
def get_correlation_params(partonshower_params, groomer, **kwargs):
    # Setting up the dictionary of parameters for the correlations
    correlation_params = partonshower_params.copy()
    correlation_params['groomer'] = groomer

    # Defining required parameters for each groomer
    extra_keywords = set(kwargs.keys())
    if groomer == 'ungroomed':
        required_keywords = {'number of emissions'}
    elif groomer == 'softdrop':
        required_keywords = {'z_cut', 'beta_sd', 'number of emissions'}
    elif groomer == 'rss':
        required_keywords = {'z_cut', 'f_soft',
                             'emission type', 'number of emissions'}
    else:
        raise ValueError(f"Unrecognized groomer {groomer}.")

    # Making sure we provide the correct parameters for the groomer
    if not required_keywords.issubset(extra_keywords):
        raise ValueError(f"Missing required keywords for {groomer}"\
            " correlations: {required_keywords - extra_keywords}")

    # If everything went well, extend the correlation parameters
    # by the given keywords and return
    correlation_params.update(kwargs)
    return correlation_params

def save_correlations(shower, partonshower_params,
                      groomer, **kwargs):
    # Setting up parameters
    params = get_correlation_params(partonshower_params,
                                    groomer, **kwargs)

    # Finding correlations
    if groomer == 'ungroomed':
        correlations = getECFs_ungroomed(
                            jet_list=shower.jet_list,
                            beta=params['shower beta'],
                            obs_acc=params['observable accuracy'],
                            n_emissions=params['number of emissions'])
    elif groomer == 'softdrop':
        correlations = getECFs_softdrop(
                            jet_list=shower.jet_list,
                            z_cut=params['z_cut'],
                            beta=params['shower beta'],
                            beta_sd=params['beta_sd'],
                            obs_acc=params['observable accuracy'],
                            n_emissions=params['number of emissions'])
    elif groomer == 'rss':
        correlations = getECFs_rss(
                            jet_list=shower.jet_list,
                            z_cut=params['z_cut'],
                            beta=params['shower beta'],
                            f=params['f_soft'],
                            obs_acc=params['observable accuracy'],
                            emission_type=params['emission type'],
                            n_emissions=params['number of emissions'])

    # Saving correlations
    save_new_data(correlations,
                  'montecarlo samples', 'parton shower',
                  params, extension='.npz')

    # Cleaning up
    del correlations
    return


"""
####################################
# Example parton shower usage:
####################################
# =====================================
# Initializing the Shower:
# =====================================
# Showers are ordered by an angularity e_beta
# Arguments are:
#       * the accuracy of the coupling;
#       * the cutoff angularity, at which the shower stops;
#       * the value of beta for the angularity e_beta which orders the shower;
#       * the type of parton initiating the parton shower.
shower = parton_shower(fixed_coupling=FIXED_COUPLING,
                       shower_cutoff=SHOWER_CUTOFF,
                       shower_beta=SHOWER_BETA,
                       jet_type=JET_TYPE)

# =====================================
# Generating or Loading Events:
# =====================================
shower.gen_events(NUM_SHOWER_EVENTS)
shower.save_events()
#shower.load_events(NUM_SHOWER_EVENTS)

for beta in BETAS:
    # =====================================
    # Saving Jet Observables:
    # =====================================
    shower.save_correlations(beta, OBS_ACC, f_soft=1)
    shower.save_correlations(beta, OBS_ACC, f_soft=.75)
    shower.save_correlations(beta, OBS_ACC, f_soft=.5)
    print()
"""


if __name__ == '__main__':
    for beta in BETAS:
        shower = parton_shower(fixed_coupling=fixed_coupling,
                               shower_cutoff=shower_cutoff,
                               shower_beta=beta,
                               jet_type=jet_type)

        if LOAD_SHOWER_EVENTS:
            shower.load_events(num_shower_events)
        else:
            shower.gen_events(num_shower_events)
            if SAVE_SHOWER_EVENTS:
                shower.save_events()

        # Preparing to save or load correlations, depending on
        # values of the SAVE_SHOWER_CORRELATIONS flag
        def save_or_load_correlations(groomer, **kwargs):
            if SAVE_SHOWER_CORRELATIONS:
                save_correlations(parton_shower, SHOWER_PARAMS,
                                  groomer, **kwargs)
            else:
                params = SHOWER_PARAMS.copy()
                params.update(kwargs)
                # Ensure that loading data is possible
                load_data('montecarlo samples', 'parton shower',
                           params)

        # Getting correlations for several grooming algorithms at
        # different levels of accuracy (number of emissions)
        for n_emissions in [1, 2, 'all']:
            kwargs = {'number of emissions': n_emissions}

            # Ungroomed correlations
            save_or_load_correlations('ungroomed', **kwargs)

            # Groomed correlations
            for z_cut in z_cuts:
                kwargs.update({'z_cut': z_cut})

                # Softdrop correlations
                save_or_load_correlations('softdrop',
                    **dict(**kwargs, **{'beta_sd': beta_mmdt}))

                # RSS correlations
                for f_soft in f_softs:
                    for emission_type in ['crit', 'precrit',
                                          'critsub', 'precritsub']:
                        save_correlations('rss',
                            **dict(**kwargs, **{
                                      'f_soft': f_soft,
                                      'emission_type' : emission_type
                                   })
                        )
        # Cleaning up
        del shower
