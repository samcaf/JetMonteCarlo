#!/bin/bash
#SBATCH --job-name fc_ll_function_generation
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o logs/zlog-%j.out
#SBATCH -e logs/zlog-%j.err
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
# Parameters for plotting and generation
nsamples='1e6'
nbins='1e3'

# If load is true, will not generate
load_events='True'
load_fns='True'

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b

# Linking log files to more precisely named logs
logfile='fc_fngen'$nsamples'samples_'$nbins'bins'
ln -f logs/zlog-${SLURM_JOB_ID}.out logs/$logfile.out.${SLURM_JOB_ID}
ln -f logs/zlog-${SLURM_JOB_ID}.err logs/$logfile.err.${SLURM_JOB_ID}

# ============================
# Path preparation:
# ============================
# Should be run from the root folder /JetMonteCarlo

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the JetMonteCarlo directory to the PYTHONPATH
# Must be used in the directory /path/to/JetMonteCarlo/
chmod +x examples/prepare_path.sh
./examples/prepare_path.sh

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
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = True/" examples/params.py
# Accuracy for f.c. observables and splitting functions is LL by default
sed -i "s/OBS_ACC = .*/OBS_ACC = 'LL'/" examples/params.py
sed -i "s/SPLITFN_ACC = .*/SPLITFN_ACC = 'LL'/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = 'quark'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
sed -i "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int("$nsamples")/" examples/params.py
sed -i "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int("$nbins")/" examples/params.py
sed -i "s/NUM_SPLITFN_BINS = .*/NUM_SPLITFN_BINS = int("$nbins")/" examples/params.py

# -------------------------
# Deciding whether to produce or reuse samples
# -------------------------
# Set all True if using already generated samples, and False otherwise
sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = "$load_events"/" examples/params.py
sed -i "s/SAVE_MC_EVENTS = .*/SAVE_MC_EVENTS = True/" examples/params.py

sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = "$load_fns"/" examples/params.py
sed -i "s/SAVE_MC_RADS = .*/SAVE_MC_RADS = True/" examples/params.py

sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = "$load_fns"/" examples/params.py
sed -i "s/SAVE_SPLITTING_FNS = .*/SAVE_SPLITTING_FNS = True/" examples/params.py

    printf "
    # ============================
    # Event generation:
    # ============================
    \n"
    # Generating phase space samples
    # for numerical integration in pQCD
    python examples/event_generation/phase_space_sampling.py

    python examples/radiator_comparisons/radiator_comparison.py

# Remove duplicate log files:
rm logs/zlog-${SLURM_JOB_ID}.out
rm logs/zlog-${SLURM_JOB_ID}.err

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

