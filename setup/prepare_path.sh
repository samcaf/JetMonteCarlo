#!/bin/bash

# ============================
# Script Flags
# ============================
# The following flags affect how scripts are executed.
# They should be changed if one is using the MIT supercloud
# computing cluster.
# They should also be changed depending on whether one is
# using MacOS or Linux.
supercloud_syntax=false
linux_or_macos="macos"

# Changing syntax based on operating system
case $linux_or_macos in
    "linux")
        cp setup/linux/set_params.sh setup/set_params.sh;;
        # sed -e 's/supercloud_syntax=.*/supercloud_syntax='$supercloud_syntax'/g' setup/set_params.sh;;
    "macos")
        cp setup/macos/set_params.sh setup/set_params.sh;;
        # sed -i '' -e 's/supercloud_syntax=.*/supercloud_syntax='$supercloud_syntax'/g' setup/set_params.sh;;
    *)
        echo "Error: linux_or_macos must be set to either 'linux' or 'macos'";
        exit 1;;
esac


# ============================
# Path preparation:
# ============================
# Should be run from the root folder /JetMonteCarlo

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the JetMonteCarlo directory to the PYTHONPATH
# Must be used in the directory /path/to/JetMonteCarlo/
path_append() {
    if [ -n "$2" ]; then
        case ":$(eval "echo \$$1"):" in
            *":$2:"*) :;;
            *) eval "export $1=\${$1:+\"\$$1:\"}$2" ;;
        esac
    else
        case ":$PATH:" in
            *":$1:"*) :;;
            *) export PATH="${PATH:+"$PATH:"}$1" ;;
        esac
    fi
}

path_append PYTHONPATH $PWD


# ============================
# Folders:
# ============================
# -------------------------
# Monte Carlo Samples/Output:
# -------------------------
# Creating folders for the output of Monte Carlo simulations;
# Using different folders for samples, integrals, and functions.

output_folder=output

# Making a folder for output figures
mkdir -p $output_folder/figures/current

# Creating folders for generated Monte Carlo samples
for data_type in phase_space radiators splitting_functions sudakov_functions parton_showers
do
    # Using different sub-folders for different data of interest
    mkdir -p $output_folder/montecarlo_samples/$data_type
done

# Creating folders for numerical integrals and serialized functions
for output_type in numerical_integrals serialized_functions
do
    for data_type in radiators splitting_functions sudakov_functions
    do
        # sub-folders only for data which are associated with integrals or functions
        mkdir -p $output_folder/$output_type/$data_type
    done
done

# -------------------------
# Misc.:
# -------------------------
# Creating a folder for log files
mkdir -p logs
