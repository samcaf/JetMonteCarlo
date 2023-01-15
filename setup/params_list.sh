#!/bin/bash

# ==================
# Lists of Params:
# ==================
# Listing useful params for the bash function "setup/set_params.sh"

_test_params=( --fixed_coupling True --type quark --obs LL --split LL --nmc 1e4 --nshowers 5e5 --nbins 5e2 --cutoff 1e-15 )
_test_params_munp=( --fixed_coupling False --type quark --obs MLL --split MLL --nmc 1e4 --nshowers 5e5 --nbins 5e2 --cutoff 1e-15 )

_fc_ll_params=( --fixed_coupling True --type quark --obs LL --split LL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff 1e-15 )
_fcprime_ll_params=( --fixed_coupling True --type quark --obs MLL --split MLL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff 1e-15 )

_rc_ll_params=( --fixed_coupling False --type quark --obs LL --split LL --nmc 5e5 --nshowers 5e5 --nbins 5e2 --cutoff 1e-15 )

_munp_params=( --fixed_coupling False --type quark --obs MLL --split MLL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff MU_NP )

_lambda_params=( --fixed_coupling False --type quark --obs MLL --split MLL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff LAMBDA_QCD )

_me_munp_params=( --fixed_coupling False --multiple_emissions --type quark --obs MLL --split MLL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff MU_NP )
_me_lambda_params=( --fixed_coupling False --multiple_emissions --type quark --obs MLL --split MLL --nmc 5e6 --nshowers 5e5 --nbins 5e3 --cutoff LAMBDA_QCD )
