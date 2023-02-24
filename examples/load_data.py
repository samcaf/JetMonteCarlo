from examples.data_management import load_data, load_and_interpolate

from examples.params import Z_CUTS, BETAS
from examples.params import RADIATOR_PARAMS
from examples.params import SPLITTINGFN_PARAMS

# Setting up default parameters
default_radiator_params = RADIATOR_PARAMS.copy()
del default_radiator_params['z_cut']
del default_radiator_params['beta']

default_splittingfn_params = SPLITTINGFN_PARAMS.copy()
del default_splittingfn_params['z_cut']


# =====================================
# Monte Carlo Samples
# =====================================
# ---------------------------------
# Loading Sudakov Inverse Transform Samples
# ---------------------------------
def load_sudakov_samples(sudakov_params=default_radiator_params,
                         z_cuts=Z_CUTS, betas=BETAS,
                         emissions=['critical', 'pre-critical',
                                    'subsequent']):
    """Load Sudakov inverse transform samples from a file.

    Parameters
    ----------
    sudakov_params : dict
        Dictionary of parameters.
    z_cuts : list
        List of z_cuts to load.
    betas : list
        List of betas to load.
    emissions : list
        List of emissions to load.

    Returns
    -------
    dict
        Dictionary of Sudakov inverse transform samples.
    """
    samples = {emission: {} for emission in emissions}

    if 'critical' in emissions:
        print("Loading critical Sudakov inverse transform samples\n")
        for z_cut in z_cuts:
            samples['critical'][z_cut] = {}
            for b in betas:
                samples['critical'][z_cut][b] = load_data(
                            'montecarlo samples',
                            'critical sudakov inverse transform',
                            params=dict(**sudakov_params,
                                        **{'z_cut': z_cut}))

    if 'pre-critical' in emissions:
        print("Loading pre-critical Sudakov inverse transform samples\n")
        for z_cut in z_cuts:
            samples['pre-critical'][z_cut] = load_data(
                            'montecarlo samples',
                            'pre-critical sudakov inverse transform',
                            params=dict(**sudakov_params,
                                        **{'z_cut': z_cut}))

    if 'subsequent' in emissions:
        print("Loading subsequent Sudakov inverse transform samples\n")
        for z_cut in z_cuts:
            samples['subsequent'][z_cut] = {}
            for b in betas:
                samples['subsequent'][z_cut][b] = load_data(
                            'montecarlo samples',
                            'subsequent sudakov inverse transform',
                            params=dict(**sudakov_params,
                                        **{'z_cut': z_cut,
                                           'beta': b}))

    return samples

# =====================================
# Functions
# =====================================
# ---------------------------------
# Loading Radiator Interpolators
# ---------------------------------
def load_radiators(radiator_params=default_radiator_params,
                   z_cuts=Z_CUTS, betas=BETAS,
                   emissions=['critical', 'pre-critical',
                              'subsequent']):
    """Load radiators from a file.

    Parameters
    ----------
    radiator_params : dict
        Dictionary of radiator parameters.
    z_cuts : list
        List of z_cuts to load.
    betas : list
        List of betas to load.
    emissions : list
        List of emissions to load.

    Returns
    -------
    dict
        Dictionary of radiators.
    """
    radiators = {emission: {} for emission in emissions}

    if 'critical' in emissions:
        print("Loading critical radiators\n")
        for z_cut in z_cuts:
            radiators['critical'][z_cut] = load_and_interpolate(
                            'critical radiator',
                            params=dict(**radiator_params,
                                        **{'z_cut': z_cut}),
                            monotonic=True, bounds=(1e-10, 1))

    if 'pre-critical' in emissions:
        print("Loading pre-critical radiators\n")
        for z_cut in z_cuts:
            radiators['pre-critical'][z_cut] = load_and_interpolate(
                            'pre-critical radiator',
                            params=dict(**radiator_params,
                                        **{'z_cut': z_cut}),
                            interpolation_method="RectangularGrid")

    if 'subsequent' in emissions:
        print("Loading subsequent radiators\n")
        for b in betas:
            radiators['subsequent'][b] = load_and_interpolate(
                            'subsequent radiator',
                            params=dict(**radiator_params,
                                        **{'beta': b}),
                            interpolation_method='Nearest')

    return radiators



# ---------------------------------
# Loading Normalized Splitting Function
# ---------------------------------
def load_splittingfns(splittingfn_params=default_splittingfn_params,
                     z_cuts=Z_CUTS):
    """Load splitting function from a file.

    Parameters
    ----------
    splittingfn_params : dict
        Dictionary of splitting function parameters.
    z_cuts : list
        List of z_cuts to load.

    Returns
    -------
    dict
        Dictionary of splitting functions.
    """
    splitting_functions = {}
    for z_cut in z_cuts:
        splitting_functions[z_cut] = load_data(
                        'serialized function',
                        'splitting function',
                        params=dict(**splittingfn_params,
                                    **{'z_cut': z_cut}))
    return splitting_functions
