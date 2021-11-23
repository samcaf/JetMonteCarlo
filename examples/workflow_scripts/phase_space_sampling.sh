#!/bin/bash
#SBATCH --job-name phase_space_sampling
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o phase_space_sampling-%j.out
#SBATCH -e phase_space_sampling-%j.err
#SBATCH --constraint=xeon-p8


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

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b
# pip install --user pynverse


###################################
# Beginning to log workflow
###################################
printf "# ============================
# Date: "`date '+%F'`"-("`date '+%T'`")
# ============================"

# ============================
# Setting MC parameters:
# ============================
sed -i "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int(1e7)/" examples/params.py

# -------------------------
# Deciding whether to produce or reuse samples
# -------------------------
# Set all True if using already generated samples, and False otherwise
sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = False/" examples/params.py
sed -i "s/SAVE_MC_EVENTS = .*/SAVE_MC_EVENTS = True/" examples/params.py

sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = False/" examples/params.py
sed -i "s/SAVE_MC_RADS = .*/SAVE_MC_RADS = False/" examples/params.py

sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = False/" examples/params.py
sed -i "s/SAVE_SPLITTING_FNS = .*/SAVE_SPLITTING_FNS = False/" examples/params.py

    printf "
    # ============================
    # Event generation:
    # ============================
    \n"
    # Generating phase space samples
    # for numerical integration in pQCD
    python examples/event_generation/phase_space_sampling.py
