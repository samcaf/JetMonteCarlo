#!/bin/bash
#SBATCH --job-name sudakov_generation
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
# Physics parameters
fixedcoup='False'
jet_type='quark'

obsacc='MLL'
splitacc='MLL'

# Parameters for event and function generation
nsamples='5e6'
nbins='5e3'

# Code Switches:
load_events='True'
load_fns='False'

save_events='True'
save_rads='True'
save_splitfns='True'

make_rad_plot=false
make_sudakov=true

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b
# pip install --user pynverse

# Linking log files to more precisely named logs
logfile='sudakov_fixedcoup'$fixedcoup'_'$nsamples'samples_'$nbins'bins'
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
chmod +x examples/slurm_scripts/prepare_path.sh
./examples/slurm_scripts/prepare_path.sh


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
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = "$fixedcoup"/" examples/params.py
# Accuracy for f.c. observables and splitting functions is LL by default
sed -i "s/OBS_ACC = .*/OBS_ACC = '"$obsacc"'/" examples/params.py
sed -i "s/SPLITFN_ACC = .*/SPLITFN_ACC = '"$splitacc"'/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = '"$jet_type"'/" examples/params.py

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
sed -i "s/SAVE_MC_EVENTS = .*/SAVE_MC_EVENTS = "$save_events"/" examples/params.py

sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = "$load_fns"/" examples/params.py
sed -i "s/SAVE_MC_RADS = .*/SAVE_MC_RADS = "$save_rads"/" examples/params.py

sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = "$load_fns"/" examples/params.py
sed -i "s/SAVE_SPLITTING_FNS = .*/SAVE_SPLITTING_FNS = "$save_splitfns"/" examples/params.py


if $make_rad_plot
then
	printf "
# ============================
# Plotting:
# ============================
\n"
	python examples/radiator_comparisons/radiator_comparison.py
fi


if $make_sudakov
then
        printf "
# ============================
# Plotting:
# ============================
\n"
	python examples/sudakov_comparisons/sudakov_comparison_full.py
fi

# Remove duplicate log files:
rm logs/zlog-${SLURM_JOB_ID}.out
rm logs/zlog-${SLURM_JOB_ID}.err

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

