#!/bin/bash
#SBATCH --job-name rc_ll_function_generation
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o logs/rc_ll_function_generation-%j.out
#SBATCH -e logs/rc_ll_function_generation-%j.err
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
python examples/params.py

# ============================
# Setting desired accuracy:
# ============================
# Fixed coupling:
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = False/" examples/params.py
# Accuracy for f.c. observables and splitting functions is LL by default

# Setting accuracy for r.c. observables and splitting functions
sed -i "s/OBS_ACC = .*/OBS_ACC = 'LL'/" examples/params.py
sed -i "s/SPLITFN_ACC = .*/SPLITFN_ACC = 'LL'/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = 'quark'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
sed -i "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int(5e6)/" examples/params.py
sed -i "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int(5e3)/" examples/params.py
sed -i "s/NUM_SPLITFN_BINS = .*/NUM_SPLITFN_BINS = int(5e3)/" examples/params.py

# -------------------------
# Deciding whether to produce or reuse samples
# -------------------------
# Set all True if using already generated samples, and False otherwise
sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = True/" examples/params.py
sed -i "s/SAVE_MC_EVENTS = .*/SAVE_MC_EVENTS = True/" examples/params.py

sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = False/" examples/params.py
sed -i "s/SAVE_MC_RADS = .*/SAVE_MC_RADS = True/" examples/params.py

sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = False/" examples/params.py
sed -i "s/SAVE_SPLITTING_FNS = .*/SAVE_SPLITTING_FNS = True/" examples/params.py

    printf "
    # ============================
    # Event generation:
    # ============================
    \n"
    # Generating phase space samples
    # for numerical integration in pQCD
    python examples/event_generation/phase_space_sampling.py
