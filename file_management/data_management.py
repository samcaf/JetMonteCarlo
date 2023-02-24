# File storage
import dill as pickle
import numpy as np

# Local file management
from file_management.catalog_utils import new_cataloged_filename
from file_management.catalog_utils import filename_from_catalog

# Interpolation functions
from jetmontecarlo.utils.interpolation import get_1d_interpolation
from jetmontecarlo.utils.interpolation import get_2d_interpolation


def save_new_data(data, data_type, data_source,
                  params, extension):
    """Saves the given data to a new file, and stores the filename
    together with the associated parameter in the file catalog.
    """
    # Creating a new filename for the given data parameters
    filename = new_cataloged_filename(data_type, data_source,
                                      params, extension)

    # If pickling, save bytes to the new .pkl file
    if extension == '.pkl':
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    # If using numpy, save arrays to the new .npz file
    elif extension == '.npz':
        if isinstance(data, dict):
            np.savez(filename, **data)
        else:
            np.save(filename, data)


def load_data(data_type, data_source, params):
    """Loads and returns data of the given data type,
    source, and with the given parameters.
    """
    # Finding the file/extension associated with the given params
    filename = filename_from_catalog(data_type, data_source,
                                     params)
    extension = "."+filename.split('.')[-1]

    # Loading the associated data
    if extension == '.pkl':
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    elif extension == '.npz':
        data = np.load(filename, allow_pickle=True,
                       mmap_mode='c')
    else:
        raise ValueError("Extension must be .pkl or .npz, not"\
                         +f" {extension}.")

    return data


def load_and_interpolate(data_source, params, **interp_kwargs):
    """Loads data of the given data source and parameters,
    and interpolates it to the given parameters.
    """
    # Expect bins unless dealing with subsequent radiators
    #   (whose domain has a weird 2 dimensional shape
    #    not amenable to binning)
    unbinned_data_sources = ['subsequent radiator']

    # Loading the data
    data = load_data('numerical integral', data_source, params)

    # Formatting the data
    bins     = data.get('bins')
    integral = data['integral']

    if bins is None:
        if data_source not in unbinned_data_sources:
            raise ValueError("No bins found for data source "\
                             +f"{data_source}.")
        # NOTE: A bit hacked/not general, but we know the
        #        shape of the subsequent radiator data
        dimensionality = 2
        xs = data.get('xs')
        ys = data.get('ys')
    else:
        dimensionality = bins.ndim
        if dimensionality == 2:
            xs, ys = bins[0], bins[1]

    # Interpolating the data into an interpolation function
    if dimensionality == 1:
        # Monotonic behavior by default in 1d
        if interp_kwargs.get('monotonic') is None:
            interp_kwargs['monotonic'] = True
        # After setting monotonicity args, get the interpolation
        interpolation = get_1d_interpolation(bins, integral,
                                             **interp_kwargs)

    elif dimensionality == 2:
        interpolation = get_2d_interpolation(xs, ys, integral,
                                             **interp_kwargs)
    else:
        raise ValueError('Dimensionality of data must be 1 or 2.')

    return interpolation
