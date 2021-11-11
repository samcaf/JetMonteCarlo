#!/bin/bash
#SBATCH --job-name piranha_fc_ll
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o logs/fc_ll-%j.out
#SBATCH -e logs/fc_ll-%j.err
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
gen_events=false
if $gen_events
    then load_events_str='False';
    else load_events_str='True';
fi

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b
# pip install --user pynverse

# -------------------------
# log file preparation:
# -------------------------
#LOG_FILE="./examples/logs/fc_ll_workflow.log"
#> $LOG_FILE
#exec >> $LOG_FILE 2>&1


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


###################################
# Beginning to log workflow
###################################
printf "# ============================
# Date: "`date '+%F'`"-("`date '+%T'`")
# ============================"

printf "
###################################
# Setting up files
###################################"
# ============================
# Setting desired accuracy:
# ============================
# Fixed coupling:
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = True/" examples/params.py
# Accuracy for f.c. observables and splitting functions is LL by default
sed -i "s/OBS_ACC = .*/OBS_ACC = 'LL'/" examples/params.py
sed -i "s/SPLITFN_ACC = .*/SPLITFN_ACC = 'LL'/" examples/params.py

# Cutoff for the angularity which orders the parton shower:
sed -i "s/SHOWER_CUTOFF = .*/SHOWER_CUTOFF = 1e-10/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = 'quark'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
# -------------------------
# Deciding whether to produce or reuse samples
# -------------------------
# Set all True if using already generated samples, and False otherwise
sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = False/" examples/params.py
#sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = "$load_events_str"/" examples/params.py
sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = "$load_events_str"/" examples/params.py
sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = "$load_events_str"/" examples/params.py

# -------------------------
# Number of events/bins:
# Requires type casting as an integer:
# 'int(num_events or num_bins)'
# -------------------------
# Number of events (MC and parton shower)
sed -i "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int(5e6)/" examples/params.py
sed -i "s/NUM_SHOWER_EVENTS = .*/NUM_SHOWER_EVENTS = int(5e5)/" examples/params.py

# Number of bins used to calculate radiators
# I've found that 5e6 MC events and 5e3 bins yield good results
sed -i "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int(5e3)/" examples/params.py
sed -i "s/NUM_SPLITFN_BINS = .*/NUM_SPLITFN_BINS = int(5e3)/" examples/params.py

python examples/params.py

printf "\n
###################################
# Running workflow:
###################################"
if $gen_events
    then
    printf "
    # ============================
    # Event generation:
    # ============================

    # -------------------------
    # Phase space samples
    # -------------------------
    \n"
    # Generating phase space samples
    # for numerical integration in pQCD
    python examples/event_generation/phase_space_sampling.py

    printf "
    # -------------------------
    # Parton shower samples:
    # -------------------------
    \n"
    # Generating shower samples and observables
    python examples/event_generation/parton_shower_gen.py
fi

printf "\n
# ============================
# Comparison:
# ============================
\n"
# Generate pQCD distributions of observables,
# compare to the results from parton shower
python examples/comparison_plots/sudakov_comparison_numeric.py

printf "\n\n\n"
