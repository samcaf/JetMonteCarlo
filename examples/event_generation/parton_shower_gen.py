# Loading parton shower class and utilities to find ECFs/correlations:
from jetmontecarlo.montecarlo.partonshower import parton_shower
from jetmontecarlo.utils.partonshower_utils import getECFs_ungroomed
from jetmontecarlo.utils.partonshower_utils import getECFs_softdrop
from jetmontecarlo.utils.partonshower_utils import getECFs_rss

# File/data management
from file_management.data_management import save_new_data, load_data

# Parameters
from examples.params import tab, BETAS, Z_CUTS, SHOWER_PARAMS


# =====================================
# Flags and Parameter setup
# =====================================
params = SHOWER_PARAMS.copy()
del params['shower beta']

# Saving/loading flags
SAVE_SHOWER_EVENTS = False
LOAD_SHOWER_EVENTS = True
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
            f" correlations: {required_keywords - extra_keywords}")

    # If everything went well, extend the correlation parameters
    # by the given keywords and return
    correlation_params.update(kwargs)
    return correlation_params

def save_correlations(shower_instance, partonshower_params,
                      groomer, **kwargs):
    """Saves the ECFs/correlations associated with the jets
    produced by the given shower, with the given groomer and
    keyword arguments.
    """
    # Setting up parameters
    groomer_params = get_correlation_params(
                                partonshower_params,
                                groomer, **kwargs)

    # Finding correlations
    if groomer == 'ungroomed':
        correlations = getECFs_ungroomed(
                    jet_list=shower_instance.jet_list,
                    beta=groomer_params['shower beta'],
                    obs_acc=groomer_params['observable accuracy'],
                    n_emissions=groomer_params['number of emissions'])
    elif groomer == 'softdrop':
        correlations = getECFs_softdrop(
                    jet_list=shower_instance.jet_list,
                    z_cut=groomer_params['z_cut'],
                    beta=groomer_params['shower beta'],
                    beta_sd=groomer_params['beta_sd'],
                    obs_acc=groomer_params['observable accuracy'],
                    n_emissions=groomer_params['number of emissions'])
    elif groomer == 'rss':
        correlations = getECFs_rss(
                    jet_list=shower_instance.jet_list,
                    z_cut=groomer_params['z_cut'],
                    beta=groomer_params['shower beta'],
                    f=groomer_params['f_soft'],
                    obs_acc=groomer_params['observable accuracy'],
                    emission_type=groomer_params['emission type'],
                    n_emissions=groomer_params['number of emissions'])

    # Saving correlations
    save_new_data(correlations,
                  'montecarlo samples', 'parton shower',
                  groomer_params, extension='.npy')

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
        # Setting the parameters of the shower
        params['shower beta'] = beta

        # Generating the shower instance and getting events
        shower = parton_shower(fixed_coupling=fixed_coupling,
                               shower_cutoff=shower_cutoff,
                               shower_beta=beta,
                               jet_type=jet_type)
        if LOAD_SHOWER_EVENTS:
            shower.load_events(num_shower_events)
        else:
            shower.gen_events(num_shower_events)
            print()
            if SAVE_SHOWER_EVENTS:
                shower.save_events()

        # Preparing to save or load correlations
        def save_or_load_correlations(groomer, **kwargs):
            """Saves or loads correlations for the given groomer
            and keyword arguments, depending on the value of the
            SAVE_SHOWER_CORRELATIONS flag.
            """
            if SAVE_SHOWER_CORRELATIONS:
                save_correlations(shower, params,
                                  groomer, **kwargs)
            else:
                groomer_params = params.copy()
                groomer_params.update(kwargs)
                # Ensure that loading data is possible
                load_data('montecarlo samples', 'parton shower',
                           groomer_params)


        # Getting correlations for several grooming algorithms at
        # different levels of accuracy (or number of emissions)

        # ---------------------------------
        # Ungroomed Correlations
        # ---------------------------------
        print(tab+"Getting ungroomed correlations")
        for n_emissions in [1, 2, 'all']:
            print(tab+tab+f"with {n_emissions} emissions")

            kwargs = {'number of emissions': n_emissions}
            save_or_load_correlations('ungroomed', **kwargs)
        print()

        # ---------------------------------
        # Soft Drop Groomed Correlations
        # ---------------------------------
        print(tab+"Getting Soft Drop groomed correlations")
        for n_emissions in [1, 2, 'all']:
            print(tab+tab+f"with {n_emissions} emissions")

            kwargs = {'number of emissions': n_emissions}

            # Looping over grooming parameters
            for z_cut in z_cuts:
                print(tab+tab+tab+f"and {z_cut = }")
                kwargs.update({'z_cut': z_cut})

                # Saving or loading correlations
                save_or_load_correlations('softdrop',
                    **dict(**kwargs, **{'beta_sd': beta_mmdt}))
        print()

        # ---------------------------------
        # RSS Groomed Correlations
        # ---------------------------------
        print(tab+"Getting RSS groomed correlations")
        for n_emissions in [1, 2, 'all']:
            print(tab+tab+f"with {n_emissions} emissions")

            kwargs = {'number of emissions': n_emissions}

            # Looping over grooming parameters
            for z_cut in z_cuts:
                print(tab+tab+tab+f"and {z_cut = },")

                kwargs.update({'z_cut': z_cut})

                for f_soft in f_softs:
                    print(tab+tab+tab+tab+f"and {f_soft = }:")

                    # Looping over types of emissions
                    print(tab+tab+tab+tab+tab+"Considered:",
                          end=' ')
                    for emission_type in ['crit', 'precrit',
                                          'critsub', 'precritsub']:
                        print(f"{emission_type} emissions", end='; ')

                        # Saving or loading correlations
                        save_or_load_correlations('rss',
                            **dict(**kwargs, **{
                                      'f_soft': f_soft,
                                      'emission type' : emission_type
                                   })
                        )
                    print()
        print()
        # Cleaning up
        del shower
