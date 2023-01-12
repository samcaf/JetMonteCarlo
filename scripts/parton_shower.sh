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
verbose=true

# -------------------------
# Help message:
# -------------------------
usage() {
chmod +x setup/set_params.sh; ./setup/set_params.sh --help
printf "###################################
# Parton Shower Generation:
###################################

Options for $0:
  # ============================
  # Setting Parameters:
  # ============================
  [<--set_params|-t> <int>] :				Set params:
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
  [<--logfile|-l> ] :					Set log file;
  [<-v> ]:						Verbose switch to print out used parameters (default true);
  [<--help|-h> ] :                                      Get this help message!\n\n"
  1>&2; exit 1; }

# Extra parameters
logfile='shower_logfile'


# ============================
# Setting Parameters:
# ============================
# Transform long options to short ones
args="$@"
for arg in "$@"; do
    shift
    case "$arg" in
	--set_params)		 set -- "$@" "-t" ;;
	--logfile)		 set -- "$@" "-l" ;;
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
      0) ./setup/set_params.sh "$@" ;;
      FCLL) ./setup/set_params.sh "${_fc_ll_params[@]}" --load_events False ;;
      RCLL) ./setup/set_params.sh "${_rc_ll_params[@]}" --load_events False ;;
      MU_NP) ./setup/set_params.sh "${_munp_params[@]}" --load_events False ;;
      LAMBDA) ./setup/set_params.sh "${_lambda_params[@]}" --load_events False ;;
    esac
    ;;
    l) logfile=${OPTARG};;
    v) verbose=false;
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
source setup/prepare_path.sh

# -------------------------
# Log File Preparation:
# -------------------------
if [ "$supercloud_syntax" = true ] ;
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

printf 'Running scripts/parton_shower.sh ('"$0"') with options:'"\n"
printf '%q ' "$args"
printf "\n\n"

if [ "$verbose" = true ] ;
then
    python3 examples/params.py
fi

printf "
# ============================
# Parton shower generation:
# ============================
\n"
# ============================
# Generating shower samples and observables
# ============================
python3 examples/event_generation/parton_shower_gen.py

if [ "$supercloud_syntax" = true ] ;
then
  # Remove duplicate log files:
  rm logs/zlog-${SLURM_JOB_ID}.out
  rm logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"
