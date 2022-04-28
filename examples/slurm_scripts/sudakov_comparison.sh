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
# Determining whether we use the syntax required for sbatch on the MIT supercloud
supercloud_syntax=true

# -------------------------
# Help message:
# -------------------------
usage() {
chmod +x setup/set_params.sh; ./setup/set_params.sh --help
printf "###################################
# Sudakov Exponent Plotting:
###################################

Options for $0:
  # ============================
  # Setting Parameters:
  # ============================
  [<--set_params|-t> <int>] :                           Set params:
                                                        0) Using the given inputs (see setup/set_params.sh);
                                                        1) With a particular choice of parameters for a non-perturbative shower:
                                                          * Running coupling and MLL accuracy for observables;
                                                          * Shower cutoff of LAMBDA_QCD ~ 340 MeV.
                                                        (see setup/set_np1.sh)
  # ============================
  # Other Options:
  # ============================
  [<--make_rad_plot|d> ] :				Make radiator plots;
  [<--make_sudakov|k> ] :				Make Sudakov plots;
  [<--logfile|-l> ] :                                   Set log file;
  [<--help|-h> ] :                                      Get this help message!\n\n"
  1>&2; exit 1; }

# Extra parameters
logfile='sudakov_logfile'
make_rad_plot=false
make_sudakov=false


# ============================
# Setting Parameters:
# ============================
# Transform long options to short ones
args="$@"
for arg in "$@"; do
    shift
    case "$arg" in
        --set_params)            set -- "$@" "-t" ;;
	--make_rad_plot)         set -- "$@" "-k" ;;
	--dont_make_sudakov)     set -- "$@" "-d" ;;
        --logfile)               set -- "$@" "-l" ;;
        --help|-h)               usage ;;
        *)                       set -- "$@" "$arg" ;;
    esac
done

while getopts "t:l:kd" OPTION; do
    case $OPTION in
    t)
    case $OPTARG in
      0) chmod +x setup/set_params.sh; ./setup/set_params.sh $@ ;;
      1) chmod +x setup/set_params_NP1.sh; ./setup/set_params_NP1.sh ;;
    esac
    ;;
    l) logfile=${OPTARG};;
    k) make_rad_plot=true;;
    d) make_sudakov=false;;
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
printf '%q ' "$args"
printf "\n\n"

if [ verbose = true ] ;
then
    python3 examples/params.py
fi

if $make_rad_plot
then
	printf "
# ============================
# Plotting Radiators:
# ============================
\n"
	python examples/radiator_comparisons/radiator_comparison.py
printf "
# Complete!
"
fi


if $make_sudakov
then
        printf "
# ============================
# Plotting Sudakov Exponents:
# ============================
\n"

printf "# ----------------------------
# Old Script:
# ----------------------------
\n"
        python examples/sudakov_comparisons/sudakov_comparison_numeric.py



printf "# ----------------------------
# Soft Drop: 
# ----------------------------
\n"
	python examples/sudakov_comparisons/sudakov_comparison_softdrop_numeric.py


printf "

# ----------------------------
# Recursive Safe Subtraction:
# ----------------------------
\n"
	python examples/sudakov_comparisons/sudakov_comparison_full.py
printf "
# Complete!
"
fi

# Remove duplicate log files:
rm logs/zlog-${SLURM_JOB_ID}.out
rm logs/zlog-${SLURM_JOB_ID}.err

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

