#!/bin/bash

# Preparing path variables, and determining whether we use
# the syntax required for sbatch on the MIT supercloud
source setup/prepare_path.sh

if [ supercloud_syntax = true ] ;
then
  # Preparation for running in supercloud cluster:
  module load anaconda/2021b
fi


# ============================
# Default Parameters:
# ============================
verbose=false

# Physics parameters
fixedcoup='False'
multiple_emissions='False'
jet_type='quark'

obsacc='MLL'
splitacc='MLL'

# Default shower cutoff for comparisons
shower_cutoff='1e-15'

# Parameters for MC event and function generation
nmcsamples='5e6'
nshowers='5e5'
nbins='5e3'

# Code Switches:
load_events='True'
load_fns='False'

save_mc='True'
save_showers='False'
save_rads='True'
save_splitfns='True'
save_correlations='True'


# ============================
# Getting input parameters
# ============================
# Transform long options to short ones

for arg in "$@"; do
    shift
    case "$arg" in
        --fc|--fixed_coupling)		set -- "$@" "-f" ;;
        --me|--multiple_emissions)	multiple_emissions='True' ;;
        --type|--jet_type)		set -- "$@" "-j" ;;
        --obs|--obs_acc)		set -- "$@" "-o" ;;
        --split|--split_acc)		set -- "$@" "-p" ;;
        --nmc|--nsamples)		set -- "$@" "-m" ;;
        --nshowers)			set -- "$@" "-n" ;;
        --nbins)			set -- "$@" "-b" ;;
        --cutoff)			set -- "$@" "-c" ;;
        --save_events|--save_mc)	set -- "$@" "-s" ;;
        --save_showers)			set -- "$@" "-w" ;;
        --save_rads)			set -- "$@" "-r" ;;
        --save_splitfns)		set -- "$@" "-v" ;;
        --save_correlations)		set -- "$@" "-i" ;;
        --load_events)			set -- "$@" "-e" ;;
        --load_fns)			set -- "$@" "-a" ;;
        --help)				set -- "$@" "-h" ;;
        *)				set -- "$@" "$arg" ;;
    esac
done

# -------------------------
# Help message:
# -------------------------
usage() {
printf "###################################
# Setting Parameters:
###################################

Options for $0:
  # ================================================
  # Physics Flags:
  # ================================================
  [<--fixed_coupling|-f> <string: True|False>] :        Determines if the parton shower is evaluated with fixed coupling (default False);               ..
  [<--me|--multiple_emissions> <string: True|False>] :  Turns on multiple emissions in perturbative QCD calculations (default False);
  [<--type|--jet_type|-j> <string: JET_TYPE>]:          Determines type of jet to be used in parton shower (default quark);
  [<--obs|obs_acc|-o> <string: LL|MLL>]:                Determines accuracy of angularity ordering variable (default LL);
  [<--split|split_acc|-p> <string: LL|MLL>]:            Determines accuracy of splitting function (default MLL, including non-singular pieces);

  [<--nmc|--nsamples|-m> <int>] :                       Number of MC samples/events used (default 5e6);
  [<--nshowers|-n> <int>] :  		                Number of parton shower events (default 5e5);
  [<--nbins|-b> <int>] :                                Number of bins used for plotting and function generation (default 5e3);

  [<--cutoff|-c> <string: SHOWER_CUTOFF>] :             Cutoff for the transverse momentum of the parton shower (default 1e-15);

  # -------------------------------------------
  # Note on running coupling shower cutoff:
  # -------------------------------------------
  # To match the MLL numerical calculations of RSS and Soft Drop jet substructure observables, I've found that a cutoff of 1e-10 is
  # insufficient. The default, 1e-15, appears to be sufficient but is time consuming.


  # ================================================
  # Flags for data generation
  # ================================================
  [<--save_events|save_mc|-s> <string: True|False>] :   Determines whether generated events are saved (default True);
  [<--load_events|-e> <string: True|False>] :           Determines whether existing events are loaded (default False);

  [<--save_rads|-r> <string: True|False>] :             Determines whether generated radiators are saved (default True);
  [<--save_splitfns|-v> <string: True|False>] :         Determines whether generated splitting functions are saved (default True);
  [<--load_fns|-a> <string: True|False>] :		Determines whether functions are loaded from existing files (default True);

  [<--save_showers|-w> <string: True|False>]:		Determines whether generated showers are saved (default False);
  [<--save_correlations|-i> <string: True|False>] :     Determines whether generated correlations are saved (default True);


  # ================================================
  # Help:
  # ================================================
  [<--help|-h> ] :                                      Get this help message!\n\n"
  1>&2; exit 1; }


# ============================
# Reading options
# ============================
while getopts "f:j:o:p:m:n:b:c:s:r:v:t:e:a:vh" OPTION; do
    #echo "option : ${OPTION}"
    #echo "optarg : ${OPTARG}"
    case $OPTION in
    # --------------------
    # Physics Flags:
    # --------------------
    f) fixedcoup=${OPTARG};;
    j) jet_type=${OPTARG};;
    o) obsacc=${OPTARG};;
    p) splitacc=${OPTARG};;
    # --------------------
    # Monte Carlo Flags:
    # --------------------
    m) nmcsamples=${OPTARG};;
    n) nshowers=${OPTARG};;
    b) nbins=${OPTARG};;
    c) shower_cutoff=${OPTARG};;
    # --------------------
    # Data Generation Flags:
    # --------------------
    s) save_mc=${OPTARG};;
    e) load_events=${OPTARG};;
    r) save_rads=${OPTARG};;
    v) save_splitfns=${OPTARG};;
    a) load_fns=${OPTARG};;
    w) save_showers=${OPTARG};;
    i) save_correlations=${OPTARG};;
    # --------------------
    # Misc:
    # --------------------
    v) verbose=true ;;
    h) usage ;;
    esac
done


###################################
# Setting Parameters:
###################################

# ============================
# Setting desired accuracy:
# ============================
# Fixed coupling:
sed -e "s/FIXED_COUPLING = .*/FIXED_COUPLING = "$fixedcoup"/" examples/params.py
# Multiple emissions:
sed -e "s/MULTIPLE_EMISSIONS = .*/MULTIPLE_EMISSIONS = "$multiple_emissions"/" examples/params.py
# Accuracy for f.c. observables and splitting functions is LL by default
sed -e "s/OBS_ACC = .*/OBS_ACC = '"$obsacc"'/" examples/params.py
sed -e "s/SPLITFN_ACC = .*/SPLITFN_ACC = '"$splitacc"'/" examples/params.py

# ============================
# Setting jet type:
# ============================
sed -e "s/JET_TYPE = .*/JET_TYPE = '"$jet_type"'/" examples/params.py

# ============================
# Setting MC parameters:
# ============================
# -----------------
# MC Integration:
# -----------------
sed -e "s/NUM_MC_EVENTS = .*/NUM_MC_EVENTS = int("$nmcsamples")/" examples/params.py
sed -e "s/NUM_RAD_BINS = .*/NUM_RAD_BINS = int("$nbins")/" examples/params.py
sed -e "s/NUM_SPLITFN_BINS = .*/NUM_SPLITFN_BINS = int("$nbins")/" examples/params.py

sed -e "s/LOAD_MC_EVENTS = .*/LOAD_MC_EVENTS = "$load_events"/" examples/params.py
sed -e "s/SAVE_MC_EVENTS = .*/SAVE_MC_EVENTS = "$save_mc"/" examples/params.py

sed -e "s/LOAD_MC_RADS = .*/LOAD_MC_RADS = "$load_fns"/" examples/params.py
sed -e "s/SAVE_MC_RADS = .*/SAVE_MC_RADS = "$save_rads"/" examples/params.py

sed -e "s/LOAD_SPLITTING_FNS = .*/LOAD_SPLITTING_FNS = "$load_fns"/" examples/params.py
sed -e "s/SAVE_SPLITTING_FNS = .*/SAVE_SPLITTING_FNS = "$save_splitfns"/" examples/params.py


# -----------------
# Parton Showers:
# -----------------
sed -e "s/NUM_SHOWER_EVENTS = .*/NUM_SHOWER_EVENTS = int("$nshowers")/" examples/params.py
sed -e "s/SHOWER_CUTOFF = .*/SHOWER_CUTOFF = "$shower_cutoff"/" examples/params.py

sed -e "s/LOAD_SHOWER_EVENTS = .*/LOAD_SHOWER_EVENTS = "$load_events"/" examples/event_generation/parton_shower_gen.py
sed -e "s/SAVE_SHOWER_EVENTS = .*/SAVE_SHOWER_EVENTS = "$save_showers"/" examples/event_generation/parton_shower_gen.py
sed -e "s/SAVE_SHOWER_CORRELATIONS = .*/SAVE_SHOWER_CORRELATIONS = "$save_correlations"/" examples/event_generation/parton_shower_gen.py

# ============================
# Printing out useful output information
# ============================
if [ "$verbose" = true ] ;
then
    python3 examples/params.py
fi
