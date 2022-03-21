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

# -------------------------
# Help message:
# -------------------------
usage() {
printf "###################################
# Parton Shower Generation:
###################################

Options for $0:
  # ============================
  # Setting Parameters:
  # ============================
  [<--set_params|-t> <int>] :				Set params:
  							0) Using the given inputs (see setup/set_params.sh);
							1) With a particular choice of parameters for a non-perturbative shower:
                                                          * Running coupling and MLL accuracy for observables;
                                                          * Shower cutoff of LAMBDA_QCD ~ 340 MeV.
							(see setup/set_np1.sh)
  # ============================
  # Other Options:
  # ============================
  [<--logfile|-l> ] :					Set log file;
  [<--help|-h> ] :                                      Get this help message!\n\n"
  1>&2; exit 1; }



# ============================
# Setting Parameters:
# ============================
# Transform long options to short ones
args=( )
for arg; do
    case "$arg" in
	--set_params)		 args+=( -t ) ;;
	--logfile)		 args+=( -l ) ;;
        --help|-h)               usage ;;
        *)                       args+=( "$arg" ) ;;
    esac
done

# Extra parameters
logfile='shower_logfile'

while getopts "t:l:" OPTION; do
    case $OPTION in
    t)
    case $OPTARG in
      0) chmod +x setup/set_params.sh; ./setup/set_params.sh $@ ;;
      1) chmod +x setup/set_params_NP1.sh; ./setup/set_params_NP1.sh ;;
    esac
    ;;
    l) logfile=${OPTARG};;
    esac
done


# ============================
# Path preparation:
# ============================

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the JetMonteCarlo directory to the PYTHONPATH
# Must be used in the directory /path/to/JetMonteCarlo/
chmod +x setup/prepare_path.sh
./setup/prepare_path.sh

# -------------------------
# Log File Preparation:
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

printf "
# ============================
# Parton shower generation:
# ============================
\n"
# ============================
# Generating shower samples and observables
# ============================
# python examples/event_generation/parton_shower_gen.py

if [ supercloud_syntax = true ] ;
then
  # Remove duplicate log files:
  rm logs/zlog-${SLURM_JOB_ID}.out
  rm logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

