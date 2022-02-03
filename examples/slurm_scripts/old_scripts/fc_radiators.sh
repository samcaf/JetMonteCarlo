#!/bin/bash
#SBATCH --job-name fc_radiator_plot
#SBATCH --exclusive
#SBATCH -c 24
#SBATCH --mem=0
#SBATCH -o logs/zlog_fc_rad-%j.out
#SBATCH -e logs/zlog_fc_rad-%j.err
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
nsamples='1e5'
nbins='1e2'

# Switch describing whether events are generated or loaded
gen_events=true
if $gen_events
    then load_events_str='False';
    else load_events_str='True';
fi

gen_rads=true
if $gen_rads
    then load_rads_str='False';
    else load_rads_str='True';
fi

# -------------------------
# Supercloud Preparation:
# -------------------------
# Preparation for running in supercloud cluster:
module load anaconda/2021b

# Linking log files to more precisely named logs
logfile='radiator_fc_'$nsamples'samples_'$nbins'bins'
ln -f logs/zlog_fc_rad-${SLURM_JOB_ID}.out logs/$logfile.out.${SLURM_JOB_ID}
ln -f logs/zlog_fc_rad-${SLURM_JOB_ID}.err logs/$logfile.err.${SLURM_JOB_ID}

# ============================
# Path preparation:
# ============================
# Should be run from the root folder /JetMonteCarlo
chmod +x examples/workflow_scripts/prepare_path.sh
./examples/workflow_scripts/prepare_path.sh

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
sed -i "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = "$load_events_str"/" examples/params.py
sed -i "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = "$load_rads_str"/" examples/params.py
sed -i "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = "$load_rads_str"/" examples/params.py

# -------------------------
# Number of events/bins:
# Requires type casting as an integer:
# 'int(num_events or num_bins)'
# -------------------------
# Number of events (MC and parton shower)
sed -i "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int("$nsamples")/" examples/params.py

# Number of bins used to calculate radiators
# I've found that 5e6 MC events and 5e3 bins yield good results
sed -i "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int("$nbins")/" examples/params.py

python examples/params.py

printf "\n
###################################
# Running workflow:
###################################"
if $gen_rads
    then
    printf "
    # ============================
    # Event generation:
    # ============================
    \n"
    # Generating phase space samples
    # for numerical integration in pQCD
    python examples/event_generation/phase_space_sampling.py
fi

printf "\n
# ============================
# Comparison:
# ============================
\n"
python examples/radiator_comparisons/radiator_comparison.py

printf "\n\n\n"

# Remove duplicate log files:
rm logs/zlog_fc_rad-${SLURM_JOB_ID}.out
rm logs/zlog_fc_rad-${SLURM_JOB_ID}.err
