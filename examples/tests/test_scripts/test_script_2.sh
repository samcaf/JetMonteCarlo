#!/bin/bash
#SBATCH --job-name test_script_2
#SBATCH --exclusive
#SBATCH -c 10
#SBATCH --mem=0
#SBATCH -o output/logs/zlog-%j.out
#SBATCH -e output/logs/zlog-%j.err
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

logfile='test_2_logfile'

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
  echo "Test: loading python modules"
  module load anaconda/2021b
  # Linking slurm log files to more precisely named logs
  echo "Test: linking log files"
  ln -f output/logs/zlog-${SLURM_JOB_ID}.out output/logs/$logfile.out.${SLURM_JOB_ID}
  ln -f output/logs/zlog-${SLURM_JOB_ID}.err output/logs/$logfile.err.${SLURM_JOB_ID}
else
  # Writing to log files without slurm
  exec 1>output/logs/$logfile.out
  exec 2>output/logs/$logfile.err
fi


###################################
# Beginning to log workflow
###################################
printf "# ============================
# Date: "`date '+%F'`"-("`date '+%T'`")
# ============================\n\n"

printf 'Running scripts/test_script_2.sh ('"$0"') with options:'"\n"
printf '%q ' "$args"
printf "\n\n"

echo "Testing the result of printing in python"
python3 -c """print('test: hello world')"""

echo "Running a test python program"
python3 examples/tests/test_scripts/sample_program.py

echo "Running params.py in python"
python3 examples/params.py

if [ "$supercloud_syntax" = true ] ;
then
  echo "Test: removing duplicate log files"
  rm output/logs/zlog-${SLURM_JOB_ID}.out
  rm output/logs/zlog-${SLURM_JOB_ID}.err
fi

printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"
