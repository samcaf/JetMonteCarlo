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

# Changing syntax based on operating system.
# In particular, the syntax of `sed` changes between linux and macos
case $linux_or_macos in
    "linux")
        cp setup/linux/set_params.sh setup/set_params.sh;;
    "macos")
        cp setup/macos/set_params.sh setup/set_params.sh;;
    *)
        echo "Error: linux_or_macos must be set to either 'linux' or 'macos'";
        exit 1;;
esac


# ============================
# Path preparation:
# ============================
# Note: Should be run from the root folder /JetMonteCarlo

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
# Output Folders:
# ============================
# Creating folders for the output of Monte Carlo simulations;
# Using different folders for samples, integrals, and functions.

output_folder=output

# Making a folder for output figures
mkdir -p $output_folder/figures/current

# Creating a folder for log files
mkdir -p $output_folder/logs

# Making a catalog file for the output of the examples
mkdir -p $output_folder/examples/current
touch $output_folder/examples/file_catalog.yaml
