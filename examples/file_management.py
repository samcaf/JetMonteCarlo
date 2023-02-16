from pathlib import Path
import warnings

# Importing time to wait if I run into `ScannerError`s
import time

# Data cataloging
import uuid
import yaml

# ===================================
# Folders and Fundamental Files
# ===================================
# Figure folder
fig_folder = Path('output/figures/current/')

# Example output folder
example_output_folder = Path('output/examples')
example_storage       = Path('output/examples/current')
example_trashbin      = Path('output/examples/overwritten')
example_catalog       = example_output_folder / 'file_catalog.yaml'


# =====================================
# MC Filenames and Cataloging
# =====================================

# ---------------------------------
# Expected data types/sources:
# ---------------------------------
# Setup the recognized data types and sources
recognized_data_types = ['montecarlo samples',
                         'numerical integral',
                         'serialized function']

emission_types = ['ungroomed', 'critical', 'subsequent', 'pre-critical']
function_types = ['radiator', 'sudakov function']
all_functions = [emission+' '+function for emission in emission_types
                    for function in function_types]
all_functions.append('splitting function')

recognized_data_sources = {
    'montecarlo samples': [*(emission+' phase space'
                            for emission in emission_types),
                           'parton shower',
                           *(emission+' sudakov inverse transform'
                             for emission in emission_types)],
    'numerical integral': all_functions,
    'serialized function': all_functions
}


# Checks for recognized data types
def check_data_type(data_type, warn_only=True):
    """Check if the data type is recognized."""
    recognized_data_type = (data_type in recognized_data_types)
    if warn_only and not recognized_data_type:
        warnings.warn(f"Unrecognized data type: {data_type}")
        return
    assert recognized_data_type, f"Unrecognized data type: {data_type}"


def check_data_source(data_type, data_source, warn_only=True):
    """Check if the data source is recognized."""
    recognized_data_source = (data_source in
              recognized_data_sources.get(data_type))

    if warn_only and not recognized_data_source:
        warnings.warn(f"Unrecognized data source {data_source} from"
                      +f" data type {data_type}.")
        return
    assert recognized_data_source, "Unrecognized data source"\
        +f" {data_source} from data type {data_type}."

def check_params(params, warn_only=True):
    """Check if there are unexpected entries in the given params."""
    # We don't expect any `None`s in the values of params
    nones_in_values = "None" in params.values()\
        or None in params.values()

    if nones_in_values:
        if warn_only:
            warnings.warn("There are None values in the given params.")
            return
        assert False, "There are None values in the params."


# ---------------------------------
# Catalog file utilities:
# ---------------------------------
def unique_filename(data_type, data_source, extension='.pkl',
                    folder=example_storage):
    """Generate a unique filename for a given data type and source."""
    data_type = data_type.replace(' ', '-')
    data_source = data_source.replace(' ', '-')
    unique_id = str(uuid.uuid4())
    return folder / f"{data_type}_{data_source}_{unique_id}{extension}"


def dict_to_yaml_key(d, pair_separator=' : ',
                     item_separator=' | '):
    """Takes a dictionary and turns it into a string that
    can be used as a key in a yaml file.
    """
    yaml_key = item_separator.join([
                    pair_separator.join([str(key), str(value)])
                    for key, value in sorted(d.items())])
    return yaml_key


def new_cataloged_filename(data_type: str, data_source: str,
                           params: dict,
                           extension: str = '.pkl',
                           overwrite_mode: str = 'delete'):
    """Add a new entry to the example catalog file and returns
    the associated filename.
    """
    check_data_type(data_type)
    check_data_source(data_type, data_source)
    check_params(params)

    filename = unique_filename(data_type, data_source, extension)

    # Adding the filename to the catalog
    with open(example_catalog, 'r') as file:
        # Attempting to open the catalog file
        open_attempts = 0
        while open_attempts < 12:
            try:
                catalog_dict = yaml.safe_load(file)
                break
            except yaml.scanner.ScannerError as error:
                # Sometimes I run into `ScannerError`s when
                # I try to run multiple jobs at once.
                # I wonder if it's because two jobs are both trying
                # to load and modify the data in the catalog file
                print("\nRan into a ScannerError when attempting to"
                    f"load {params=}. Waiting before attempting again.")
                # Keep trying for 12 attempts/60s total
                open_attempts += 1
                time.sleep(5)

        # Setting up dict structure if it does not already exist
        if catalog_dict is None:
            # First time only
            catalog_dict = {}
        if catalog_dict.get(data_type) is None:
            catalog_dict[data_type] = {}
        if catalog_dict[data_type].get(data_source) is None:
            catalog_dict[data_type][data_source] = {}

        # Making a key for the yaml file
        yaml_key = dict_to_yaml_key(params)

        # Checking if the set of parameters already has an entry
        entry = catalog_dict[data_type][data_source].get(yaml_key)
        if entry is not None:
            file_path = Path(entry['filename'])
            if file_path.exists():
                print("Existing file with the given parameters found.")
                if overwrite_mode == 'delete':
                    print("Overwriting old file.")
                    Path.unlink(file_path)
                elif overwrite_mode == 'trash':
                    print(f"Moving old file to {example_trashbin}.")
                    Path.rename(filename,
                            example_trashbin / file_path.name)

        # Updating the dict with the given params and filenames
        params = dict({key: str(value)
                       for key, value in params.items()})
        catalog_dict[data_type][data_source][yaml_key] = params
        catalog_dict[data_type][data_source][yaml_key]\
            ['filename'] = str(filename)

    # Storing the updated catalog
    if catalog_dict:
        with open(example_catalog, 'w') as file:
            yaml.safe_dump(catalog_dict, file, width=float("inf"))
    return filename


def filename_from_catalog(data_type, data_source, params):
    """Retrieve a filename from the example catalog file."""
    check_data_type(data_type)
    check_data_source(data_type, data_source)
    check_params(params)

    # Open the catalog
    with open(example_catalog, 'r') as file:
        catalog_dict = yaml.safe_load(file)

    # Getting info for the given params from the catalog
    yaml_key = dict_to_yaml_key(params)
    try:
        catalog_info =  catalog_dict[data_type]\
                            [data_source].get(yaml_key)
    except KeyError as error:
        warnings.warn(f"Ran into a KeyError: {error}")
        catalog_info = None

    if catalog_info is not None:
        return catalog_info['filename']

    raise FileNotFoundError(f"\n
            Could not find data type {data_type} and"
            f" data source {data_source} with params"
            f" {params} in the catalog."
            "\nThe associated key in the catalog would be:\n"
            +dict_to_yaml_key(params))


# =====================================
# Testing
# =====================================
if __name__ == "__main__":
    test_type = 'montecarlo samples'
    test_source = 'sudakov inverse transform'
    test_params = {'info': 'A non-existent test file',
                   'key1': 'param1',
                   'key2': 'param2',
                   'key3': 3,
                   'key4': True}
    print("Using a dict of test parameters, with associated"
          +f" yaml key {dict_to_yaml_key(test_params)}:")

    new_filename = new_cataloged_filename(data_type=test_type,
                                          data_source=test_source,
                                          params=test_params)
    print(f"Testing file utilities, generated {new_filename = }")

    loaded_filename = filename_from_catalog(data_type=test_type,
                                            data_source=test_source,
                                            params=test_params)

    print(f"Testing file utilities, found {loaded_filename = }")
