#!/bin/bash
#SBATCH --job-name phasespace_generation
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o logs/zlog-%j.out
#SBATCH -e logs/zlog-%j.err
#SBATCH --constraint=xeon-p8

###################################
# Preparation
###################################
verbose=true

# -------------------------
# Help message:
# -------------------------
usage() {
chmod +x setup/set_params.sh; ./setup/set_params.sh --help
printf "###################################
# Phase Space Sample Generation:
###################################
Note: Haven't tested this recently, may need to set load_events=True in scripts/phasespace_generation.sh for certain situations.

Options for $0:
  # ============================
  # Setting Parameters:
  # ============================
  [<--set_params|-t> <int>] :                           Set params:
                                                        0) Using the given inputs (see setup/set_params.sh);
                                                        FCLL) Using inputs for fixed coupling perturbative calculations;
                                                        RCLL) Using inputs for running coupling perturbative calculations:
                                                          * Running coupling and LL accuracy for observables;
                                                          * Shower cutoff of 1e-15;
                                                        MU_NP) With a particular choice of parameters for a non-perturbative shower:
                                                          * Running coupling and MLL accuracy for observables;
                                                          * Shower cutoff of LAMBDA_QCD ~ 340 MeV.
                                                        LAMBDA) With a particular choice of parameters for a non-perturbative shower:
                                                          * Running coupling and MLL accuracy for observables;
                                                          * Shower cutoff of MU_NP ~ 1 GeV.

  # ============================
  # Other Options:
  # ============================
  [<--logfile|-l> ] :                                   Set log file;
  [<--help|-h> ] :                                      Get this help message!\n\n"
  1>&2; exit 1; }


# ============================
# Path preparation:
# ============================

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the JetMonteCarlo directory to the PYTHONPATH
# Must be used in the directory /path/to/JetMonteCarlo/
chmod +x setup/prepare_path.sh
source setup/prepare_path.sh


# ============================
# Setting Parameters:
# ============================
# Extra parameters
logfile='phasespace_logfile'

# Transform long options to short ones
args="$@"
for arg in "$@"; do
    shift
    case "$arg" in
        --set_params)            set -- "$@" "-t" ;;
        --logfile)               set -- "$@" "-l" ;;
        --help|-h)               usage ;;
        *)                       set -- "$@" "$arg" ;;
    esac
done

while getopts "t:l:v" OPTION; do
    case $OPTION in
    t)
        chmod +x setup/set_params.sh
        source setup/params_list.sh
        case $OPTARG in
            0)
                source setup/set_params.sh "$@" ;;
            TEST)
                source setup/set_params.sh "${_test_params[@]}" --load_events False ;;
            TESTMUNP)
                source setup/set_params.sh "${_test_params_munp[@]}" --load_events False ;;
            FCLL)
                source setup/set_params.sh "${_fc_ll_params[@]}" --load_events True ;;
            FCLLprime)
                source setup/set_params.sh "${_fcprime_ll_params[@]}" --load_events True ;;
            RCLL)
                source setup/set_params.sh "${_rc_ll_params[@]}" --load_events True ;;
            MU_NP)
                source setup/set_params.sh "${_munp_params[@]}" --load_events True ;;
            LAMBDA)
                source setup/set_params.sh "${_lambda_params[@]}" --load_events True ;;
            ME1)
                source setup/set_params.sh "${_me_munp_params[@]}" --load_events True ;;
            ME2)
                source setup/set_params.sh "${_me_lambda_params[@]}" --load_events True ;;
            *)
                echo "Unrecognized parameter type "$OPTARG"."; exit 1 ;;
            esac;;
    l) logfile=${OPTARG};;
    v) verbose=false ;;
    esac
done


# -------------------------
# Log File Preparation:
# -------------------------
if [ "$supercloud_syntax" = true ] ;
then
  # Loading python packages
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

printf 'Running scripts/phasespace_generation.sh ('"$0"') with options:'"\n"
printf '%q ' "$args"
printf "\n\n"

if [ "$verbose" = true ] ;
then
    python3 examples/params.py
fi

printf "
# ============================
# Generating Phase Space Samples:
# ============================
\n"
	python3 examples/event_generation/phase_space_sampling.py
printf "
# Complete!
"

if [ "$supercloud_syntax" = true ] ;
then
  # Remove duplicate log files:
  rm logs/zlog-${SLURM_JOB_ID}.out
  rm logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"
