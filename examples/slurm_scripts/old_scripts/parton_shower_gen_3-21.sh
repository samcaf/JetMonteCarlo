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
# Determining whether we use the syntax required for sbatch on the MIT supercloud
supercloud_syntax=true


# ============================
# Default Parameters:
# ============================
# Physics parameters
fixedcoup='False'
jet_type='quark'
obsacc='LL'

# Parameters for event and function generation
nevents='5e5'

# Default shower cutoff
cutoff='1e-15'

# Extra parameters
logfile='shower_fc'$fixedcoup'_obs'$obsacc'_'$nevents'events_cutoff'$cutoff

load_events='False'
save_events='True'
save_correlations='True'


# ============================
# Getting input parameters
# ============================
# Transform long options to short ones
args=( )
for arg; do
    case "$arg" in
        --fc|--fixed_coupling)   args+=( -f ) ;;
        --type|--jet_type)       args+=( -j ) ;;
        --obs_acc)               args+=( -o ) ;;
	--nev|--nevents)         args+=( -n ) ;;
	--cutoff)		 args+=( -c ) ;;
	--logfile)		 args+=( -l ) ;;
	--load_events)	 	 args+=( -e ) ;;
	--save_events)		 args+=( -s ) ;;
	--save_correlations)	 args+=( -c ) ;;
	--help)		 	 args+=( -h ) ;;
        *)                       args+=( "$arg" ) ;;
    esac
done


# -------------------------
# Help message:
# -------------------------
usage() {
printf "Options for $0:
  # ================================================
  # Physics Flags:
  # ================================================
  [<--fixed_coupling|-f> <string: True|False>] :	Determines if the parton shower is evaluated with fixed coupling (default False); 		..
  [<--type|--jet_type|-j> <string: JET_TYPE>]: 		Determines type of jet to be used in parton shower (default quark);
  [<--obs_acc|-o> <string: LL|MLL>]: 			Determines accuracy of angularity ordering variable (default LL);
  [<--nevents|-n> <int>] : 				Number of parton shower events (default 5e5);

  [<--cutoff|-c> <string: SHOWER_CUTOFF>] :		Cutoff for the transverse momentum of the parton shower (default 1e-15);

  # -------------------------------------------
  # Note on running coupling shower cutoff:
  # -------------------------------------------
  # To match the MLL numerical calculations of RSS and Soft Drop jet substructure observables, I've found that a cutoff of 1e-10 is
  # insufficient. The default, 1e-15, appears to be sufficient but is time consuming.


  # ================================================
  # Flags for data generation
  # ================================================
  [<--logfile|-l> <string>] :				Log file for the parton shower procedure;
  [<--load_events|-e> <string: True|False>] : 		Determines whether existing events are loaded (default False);
  [<--save_events|-s> <string: True|False>] : 		Determines whether generated events are saved (default True);
  [<--save_correlations|-c> <string: True|False>] : 	Determines whether generated correlations are saved (default True);
  
  # ================================================
  # Help:
  # ================================================
  [<--help|-h> ] :					Get this help message!\n\n"
  1>&2; exit 1; }


# ============================
# Reading options
# ============================
while getopts "f:j:o:n:c:l:h" OPTION; do
    #echo "option : ${OPTION}"
    #echo "optarg : ${OPTARG}"
    case $OPTION in
    f) fixedcoup=${OPTARG};;
    j) jet_type=${OPTARG};;
    o) obsacc=${OPTARG};;
    n) nevents=${OPTARG};;
    c) cutoff=${OPTARG};;
    l) logfile=${OPTARG};;
    h) usage ;;
    esac
done


# ============================
# Path preparation:
# ============================

# -------------------------
# Supercloud Preparation:
# -------------------------
if [ supercloud_syntax = true ] ;
then
  # Preparation for running in supercloud cluster:
  module load anaconda/2021b

  # Linking slurm log files to more precisely named logs
  ln -f logs/zlog-${SLURM_JOB_ID}.out logs/$logfile.out.${SLURM_JOB_ID}
  ln -f logs/zlog-${SLURM_JOB_ID}.err logs/$logfile.err.${SLURM_JOB_ID}
else
  # Writing to log files without slurm
  exec 1>logs/$logfile.out
  exec 2>logs/$logfile.err
fi

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
# ============================\n\n"

printf 'Running '"$0"' with options:'"\n"
printf '%q ' "$@"
printf "\n\n"
set -- "${args[@]}"


# ============================
# Setting desired accuracy:
# ============================
sed -i "s/FIXED_COUPLING = .*/FIXED_COUPLING = "$fixedcoup"/" examples/params.py
sed -i "s/OBS_ACC = .*/OBS_ACC = '"$obsacc"'/" examples/params.py
sed -i "s/SPLITFN_ACC = .*/SPLITFN_ACC = '"$splitacc"'/" examples/params.py

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


# ============================
# Printing out useful output information
# ============================
python examples/params.py


printf "
# ============================
# Parton shower generation:
# ============================
\n"
# ============================
# Generating shower samples and observables
# ============================
python examples/event_generation/parton_shower_gen.py

if [ supercloud_syntax = true ] ;
then
  # Remove duplicate log files:
  rm logs/zlog-${SLURM_JOB_ID}.out
  rm logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

