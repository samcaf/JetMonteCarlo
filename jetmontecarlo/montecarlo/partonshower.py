from __future__ import absolute_import
import dill as pickle
from pathlib import Path

from jetmontecarlo.utils.partonshower_utils import *

class parton_shower():
    # ------------------------------------
    # Event generation
    # ------------------------------------
    def gen_events(self, num_events):
        self.num_events = num_events
        self.jet_list = gen_jets(num_events, beta=self.shower_beta,
                                 radius=self.radius, jet_type=self.jet_type,
                                 acc='LL' if self.fixed_coupling else 'MLL',
                                 cutoff=self.shower_cutoff)

    # ------------------------------------
    # File paths
    # ------------------------------------
    def showerfile_path(self, info=''):
        """Sets up a path for loading or saving shower events."""
        info += '' if self.jet_type == 'quark' else '_'+self.jet_type

        shower_folder = Path("jetmontecarlo/utils/samples/parton_showers/")
        if self.fixed_coupling:
            showerfile = 'jet_list_shower_{:.0e}_fc_{:.0e}cutoff'.format(
                                        self.num_events, self.shower_cutoff)
        else:
            showerfile = 'jet_list_shower_{:.0e}_rc_{:.0e}cutoff'.format(
                                        self.num_events, self.shower_cutoff)
        if self.shower_beta != 1:
            showerfile = showerfile + '_beta'+str(self.shower_beta)

        if info != '':
            showerfile += '_' + info

        showerfile = showerfile + '.npz'

        return shower_folder / showerfile

    def correlation_path(self, beta, obs_acc, few_pres,
                    f_soft=1., angular_ordered=False,
                    info=''):
        info += '' if self.jet_type == 'quark' else '_'+self.jet_type

        # Preparing filename
        ps_sample_folder = Path("jetmontecarlo/utils/samples/shower_correlations/")
        ps_file = 'shower_{:.0e}_c1_'.format(self.num_events)+str(beta)

        ps_file = ps_file + '_f{}'.format(f_soft)

        # Angular ordering descriptor
        if angular_ordered:
            ps_file += '_angord'
            assert False, "No angular ordering yet!"

        # Cutoff descriptor
        if self.fixed_coupling and self.shower_cutoff == 1e-20:
            ps_file += '_lowcutoff'
        elif not self.fixed_coupling and self.shower_cutoff == 1e-10:
            ps_file += '_lowcutoff'

        # Evolution variable descriptor
        if self.shower_beta != 1.:
            ps_file += '_showerbeta'+str(self.shower_beta)

        # Correlation accuracy descriptor
        if not self.fixed_coupling and obs_acc=='MLL':
            ps_file += '_MLL_fewem'
        elif not self.fixed_coupling and obs_acc=='LL':
            ps_file += '_rc_LL_fewem'
        else:
            ps_file += '_' + obs_acc
            # if few_pres:
            ps_file += '_fewem'
            # else:
            #     ps_file += '_manypres.npz'

        if info != '':
            ps_file += '_' + info

        ps_file = ps_file + '.npz'

        return ps_sample_folder / ps_file

    # ------------------------------------
    # Helper functions
    # ------------------------------------
    def save_events(self, file_path=None, info=''):
        if file_path is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.jet_list, f)
            return

        file_path = self.showerfile_path(info=info)
        if self.verbose > 0:
            print("Saving {:.0e} parton shower events to {}...".format(
                                            self.num_events, str(file_path)),
                  flush=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.jet_list, f)
        if self.verbose > 0:
            print("Parton shower events saved!", flush=True)

    def load_events(self, num_events, info=''):
        self.num_events = num_events
        file_path = self.showerfile_path(info=info)
        if self.verbose > 0:
            print("Loading {:.0e} parton shower events from {}...".format(
                                            self.num_events, str(file_path)),
                  flush=True)
        with open(file_path, 'rb') as f:
            self.jet_list = pickle.load(f)
        if self.verbose > 0:
            print("Parton shower events loaded!", flush=True)

    def save_correlations(self, beta, obs_acc, few_pres=True, f_soft=1., info=''):
        if isinstance(beta, list):
            for b in beta:
                self.save_correlations(b, obs_acc, few_pres, f_soft=f_soft,
                                       info=info)
        else:
            file_path = self.correlation_path(beta, obs_acc, few_pres, f_soft=f_soft,
                                              info=info)
            if self.verbose > 0:
                print("Saving shower correlations to {}...".format(str(file_path)), flush=True)
            save_shower_correlations(self.jet_list,
                                     file_path,
                                     beta=beta, obs_acc=obs_acc,
                                     f_soft=f_soft,
                                     few_pres=few_pres,
                                     fixed_coupling=self.fixed_coupling,
                                     verbose=self.verbose)

        if self.verbose > 0:
            print("Shower correlations saved!", flush=True)

    # ------------------------------------
    # Init
    # ------------------------------------
    def __init__(self, fixed_coupling, shower_cutoff=None, shower_beta=1,
                 radius=1., jet_type='quark',
                 num_events=0, verbose=1):
        # Initializing shower parameters
        self.fixed_coupling = fixed_coupling
        self.shower_cutoff = shower_cutoff if shower_cutoff is not None\
                             else 1e-10 if fixed_coupling\
                             else MU_NP
        self.shower_beta = shower_beta
        self.verbose = verbose

        self.radius = radius
        self.jet_type = jet_type

        # Generating events
        if num_events > 0:
            self.gen_events(num_events)
