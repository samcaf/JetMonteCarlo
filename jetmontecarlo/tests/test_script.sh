#!/bin/bash
#SBATCH --job-name sudakov_generation
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o logs/zlog-%j.out
#SBATCH -e logs/zlog-%j.err
#SBATCH --constraint=xeon-p8

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

logfile='test_logfile'

# Transform long options to short ones
args="$@"
for arg in "$@"; do
    shift
    case "$arg" in
        --set_params)            set -- "$@" "-t" ;;
        --logfile)               set -- "$@" "-l" ;;
        *)                       set -- "$@" "$arg" ;;
    esac
done

while getopts "t:l:" OPTION; do
    # echo "option | optarg : ${OPTION} | ${OPTARG}"
    case $OPTION in
    t)
        chmod +x setup/set_params.sh
        source setup/params_list.sh
        case $OPTARG in
            0)
                ./setup/set_params.sh "$@" ;;
            TEST)
                ./setup/set_params.sh "${_test_params[@]}" --load_events False ;;
            TESTMUNP)
                ./setup/set_params.sh "${_test_params_munp[@]}" --load_events False ;;
            FCLL)
                ./setup/set_params.sh "${_fc_ll_params[@]}" --load_events True ;;
            FCLLprime)
                ./setup/set_params.sh "${_fcprime_ll_params[@]}" --load_events True ;;
            RCLL)
                ./setup/set_params.sh "${_rc_ll_params[@]}" --load_events True ;;
            MU_NP)
                ./setup/set_params.sh "${_munp_params[@]}" --load_events True ;;
            LAMBDA)
                ./setup/set_params.sh "${_lambda_params[@]}" --load_events True ;;
            ME1)
                ./setup/set_params.sh "${_me_munp_params[@]}" --load_events True ;;
            ME2)
                ./setup/set_params.sh "${_me_lambda_params[@]}" --load_events True ;;
            *)
                echo "Unrecognized parameter type "$OPTARG"."; exit 1 ;;
        esac;;
    l) logfile=${OPTARG};;
    esac
done


# -------------------------
# Log File Preparation:
# -------------------------
if [ "$supercloud_syntax" = true ] ;
then
  # Linking slurm log files to more precisely named logs
  echo "Test: linking log files"
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

printf 'Running scripts/test_script.sh ('"$0"') with options:'"\n"
printf '%q ' "$args"
printf "\n\n"

echo "Testing the result of printing in python"
python3 -c """print('test: hello world')"""


if [ "$supercloud_syntax" = true ] ;
then
  echo "Test: removing duplicate log files"
  rm logs/zlog-${SLURM_JOB_ID}.out
  rm logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"
