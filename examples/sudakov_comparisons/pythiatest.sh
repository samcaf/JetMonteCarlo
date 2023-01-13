#!/bin/bash
#SBATCH --job-name pythiatest
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o pythiatest-%j.out
#SBATCH -e pythiatest-%j.err
#SBATCH --constraint=xeon-p8

###################################
# Preparation
###################################
# ============================
# Misc. Preparation
# ============================
# -------------------------
# Code Switches:
# -------------------------
# Switch describing whether events are generated or loaded

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b
# pip install --user pynverse


# ============================
# Path preparation:
# ============================

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

###################################
# Beginning to log workflow
###################################
printf "# ============================
# Date: "`date '+%F'`"-("`date '+%T'`")
# ============================"

python examples/params.py

python examples/sudakov_comparisons/sudakov_comparison_pythia_numeric.py
