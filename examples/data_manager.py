# File storage
import dill as pickle
import numpy as np

# Interpolation functions
from jetmontecarlo.utils.interpolation_function_utils import get_1d_interpolation
from jetmontecarlo.utils.interpolation_function_utils import
get_2d_interpolation

# Local file management
from examples.file_manager import new_cataloged_filename
from examples.file_manager import filename_from_catalog


def save_new_data(data, data_type, data_source,
                  params, extension):
    """Saves the given data to a new file, and stores the filename
    together with the associated parameter in the file catalog.
    """
    # Creating a new filename for the given data parameters
    filename = new_cataloged_filename(data_type, data_source,
                                      params, extension)

    # Save the data to the new file with the given extension
    with open(filename, 'w') as file:
        if extension == '.pkl':
            pickle.dump(data, file)
        elif extension == '.npy':
            if isinstance(data, dict):
                np.savez(file, **data)
            else:
                np.save(file, data)


def load_data(data_type, data_source, params):
    """Loads and returns data of the given data type,
    source, and with the given parameters.
    """
    # Finding the file/extension associated with the given params
    filename = filename_from_catalog(data_type, data_source,
                                     params)
    extension = filename.split('.')[-1]

    # Loading the associated data
    with open(filename, 'rb') as file:
        if extension == '.pkl':
            data = pickle.load(file)
        elif extension == '.npy':
            data = np.load(file, allow_pickle=True,
                           mmap_mode='c')

    return data


def load_and_interpolate(data_source, params, **kwargs):
    """Loads data of the given data source and parameters,
    and interpolates it to the given parameters.
    """
    # Loading the data
    data = load_data('numerical integral', data_source, params)

    # Formatting the data
    bins = data['bins']
    integral = data['integral']
    dimensionality = bins.ndim

    # Monotonic behavior by default
    if kwargs.get('monotonic') is None:
        kwargs['monotonic'] = True

    # Interpolating the data into an interpolation function
    if dimensionality == 1:
        interpolation = get_1d_interpolation(bins, integral, **kwargs)
    elif dimensionality == 2:
        interpolation = get_2d_interpolation(bins, integral, **kwargs)
        pass
    else:
        raise ValueError('Dimensionality of data must be 1 or 2.')

    return interpolation
