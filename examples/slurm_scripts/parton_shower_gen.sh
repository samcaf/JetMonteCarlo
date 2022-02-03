#!/bin/bash
#SBATCH --job-name parton_shower_generation
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
# Physics parameters
fixedcoup='False'
jet_type='quark'
obsacc='MLL'

# Parameters for event and function generation
nevents='5e4'
cutoff='1e-10'
# Sets the shower cutoff for running coupling.
# The usual choice is MU_NP. I also use 1e-8, 1e-10.

load_events='False'
save_events='False'
save_correlations='True'

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b

# Linking log files to more precisely named logs
logfile='shower_fc'$fixedcoup'_obs'$obsacc'_'$nevents'events_cutoff'$cutoff
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

# ============================
# Setting jet type:
# ============================
sed -i "s/JET_TYPE = .*/JET_TYPE = '"$jet_type"'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
sed -i "s/NUM_SHOWER_EVENTS = .*/NUM_SHOWER_EVENTS = int("$nevents")/" examples/params.py
sed -i "s/SHOWER_CUTOFF = .*/SHOWER_CUTOFF = "$cutoff"/" examples/params.py


sed -i "s/LOAD_SHOWER_EVENTS = .*/LOAD_SHOWER_EVENTS = "$load_events"/" examples/event_generation/parton_shower_gen.py
sed -i "s/SAVE_SHOWER_EVENTS = .*/SAVE_SHOWER_EVENTS = "$save_events"/" examples/event_generation/parton_shower_gen.py
sed -i "s/SAVE_SHOWER_CORRELATIONS = .*/SAVE_SHOWER_CORRELATIONS = "$save_correlations"/" examples/event_generation/parton_shower_gen.py

printf "
# ============================
# Parton shower generation:
# ============================
\n"
# Generating shower samples and observables
python examples/event_generation/parton_shower_gen.py


# Remove duplicate log files:
rm logs/zlog-${SLURM_JOB_ID}.out
rm logs/zlog-${SLURM_JOB_ID}.err

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"


