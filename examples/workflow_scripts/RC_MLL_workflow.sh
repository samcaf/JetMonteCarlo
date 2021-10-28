#!/bin/bash

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
# log file preparation:
# -------------------------
LOG_FILE="./examples/logs/example_workflow.log"
> $LOG_FILE
exec >> $LOG_FILE 2>&1

# -------------------------
# Cluster preparation:
# -------------------------
# Slurm options for running in cluster:
# SBATCH -o log_workflow-%j.out --exclusive



###################################
# Beginning to log workflow
###################################
printf "# ============================
Date: "`date '+%F'`"-("`date '+%T'`")
# ============================"

printf "
# -------------------------
# Setting up files
# -------------------------"
# ============================
# Setting desired accuracy:
# ============================
# Fixed coupling:
sed -i '' -e "s/FIXED_COUPLING = .*/FIXED_COUPLING = False/" examples/params.py

# Accuracy for observable and splittning functions:
awk '!/OBS_ACC = / || seen { print } /OBS_ACC = / && !seen { print "OBS_ACC = \047MLL\047"; seen = 1 }' examples/params.py  > tmp && mv tmp examples/params.py
awk '!/SPLITFN_ACC = / || seen { print } /SPLITFN_ACC = / && !seen { print "SPLITFN_ACC = \047MLL\047"; seen = 1 }' examples/params.py  > tmp && mv tmp examples/params.py

# Cutoff for the angularity which orders the parton shower:
sed -i '' -e "s/SHOWER_CUTOFF = .*/SHOWER_CUTOFF = 1e-10/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -i '' -e "s/JET_TYPE = .*/JET_TYPE = 'quark'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
# -------------------------
# Telling params.py to generate phase space samples, rather than load them
# -------------------------
sed -i '' -e "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = False/" examples/params.py

# -------------------------
# Number of events/bins:
# Requires type casting as an integer:
# 'int(num_events or num_bins)'
# -------------------------
# Number of events (MC and parton shower)
sed -i '' -e "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int(5e6)/" examples/params.py
sed -i '' -e "s/NUM_SHOWER_EVENTS = .*/NUM_SHOWER_EVENTS = int(5e5)/" examples/params.py

# Number of bins used to calculate radiators
# I've found that 5e6 MC events and 5e3 bins yield good results
sed -i '' -e "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int(5e3)/" examples/params.py
sed -i '' -e "s/NUM_SPLITFN_BINS = .*/NUM_SPLITFN_BINS = int(5e3)/" examples/params.py

printf "\n
###################################
# Running workflow:
###################################"
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

printf "\n
# ============================
# Comparison:
# ============================
\n"
# Generate pQCD distributions of observables,
# compare to the results from parton shower
python examples/comparison_plots/sudakov_comparison_numeric.py

printf "\n\n\n"
