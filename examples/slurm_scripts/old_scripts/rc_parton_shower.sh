#!/bin/bash
#SBATCH --job-name rc_parton_shower_generation
#SBATCH --exclusive
#SBATCH -c 16
#SBATCH --mem=0
#SBATCH -o logs/rc_parton_shower-%j.out
#SBATCH -e logs/rc_parton_shower-%j.err
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
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = 'quark'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
sed -i "s/NUM_SHOWER_EVENTS = .*/NUM_SHOWER_EVENTS = int(5e5)/" examples/params.py

# ============================
# Setting desired accuracy:
# ============================
# Fixed coupling:
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = False/" examples/params.py
sed -i "s/OBS_ACC = .*/OBS_ACC = 'LL'/" examples/params.py

# Cutoff for the angularity which orders the parton shower:
sed -i "s/SHOWER_CUTOFF = .*/SHOWER_CUTOFF = 1e-10/" examples/params.py

    printf "
    # ============================
    # Parton shower generation:
    # ============================
    \n"
    python examples/params.py
    # Generating shower samples and observables
    python examples/event_generation/parton_shower_gen.py

