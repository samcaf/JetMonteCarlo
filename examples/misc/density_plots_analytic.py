import dill as pickle
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from jetmontecarlo.utils.partonshower_utils import *

# Calculation parameters
few_emissions = False

# Plotting parameters
log_scale = True

def get_jet_list(num_events, shower_cutoff=1e-10,
                 coupling='fc', angular_ordering=False):
    # Filename
    sample_folder = Path("jetmontecarlo/utils/samples/")
    jetfile = ('jet_list_shower_{:.0e}_'.format(num_events)
               +coupling+'_{:.0e}cutoff.pkl'.format(shower_cutoff))
    jetfile_path = sample_folder / jetfile

    # Loading events
    print("Loading parton shower events...")
    with open(jetfile_path, 'rb') as f:
        jet_list = pickle.load(f)
    print("Parton shower events loaded!")

    if angular_ordering:
        pass
        print("Angularly ordering jets...")
        # Code
        print("Angular ordering complete!")
    return jet_list

def plot_ecf_density_2d(jet_list, z_cut, beta, f=1., acc='LL',
                        show_histplot=False):
    ecfs_bare = getECFs_groomed(jet_list, z_cut=1e-15, beta=beta, f=0.,
                                acc=acc, emission_type='precritsub',
                                few_emissions=few_emissions)

    ecfs_sd = getECFs_softdrop(jet_list, z_cut=z_cut, beta=beta,
                                     beta_sd=0., acc=acc,
                                     few_emissions=few_emissions)

    ecfs_rss = getECFs_groomed(jet_list, z_cut=z_cut, beta=beta, f=f,
                               acc=acc, emission_type='precritsub',
                               few_emissions=few_emissions)

    extra_label = '_scatter'
    if show_histplot:
        extra_label = '_hist'

    # Both on same plot
    groomed_data_full = {'Ungroomed ECF': np.append(ecfs_bare, ecfs_bare),
                         'Groomed ECF': np.append(ecfs_sd, ecfs_rss),
                         'Groomer': np.append(np.repeat('Soft Drop',
                                                        len(ecfs_sd)),
                                              np.repeat('RSS',
                                                        len(ecfs_rss)))}

    groomed_dataframe_full = pd.DataFrame(data=groomed_data_full)

    g = sns.JointGrid('Ungroomed ECF', 'Groomed ECF',
                      groomed_dataframe_full,
                      xlim=[1e-3,.4],ylim=[1e-3,.4],
                      hue='Groomer')
    g.plot_marginals(sns.histplot, kde=True, log_scale=log_scale)
    if show_histplot == True:
        g.plot_joint(sns.histplot, log_scale=(log_scale, log_scale),
                     alpha=.7)
    else:
        g.plot_joint(sns.scatterplot, alpha=.5)
    plt.savefig('2d_density_partonshower_full_example'+extra_label+'.pdf',
                format='pdf')

    # Only Soft Drop
    groomed_data_sd = {'Ungroomed ECF': ecfs_bare,
                       'Groomed ECF': ecfs_sd}
    groomed_dataframe_sd = pd.DataFrame(data=groomed_data_sd)
    g = sns.JointGrid('Ungroomed ECF', 'Groomed ECF',
                      groomed_dataframe_sd,
                      xlim=[1e-3,.4],ylim=[1e-3,.4])
    g.plot_marginals(sns.histplot, kde=True, log_scale=log_scale)

    if show_histplot == True:
        g.plot_joint(sns.histplot, log_scale=(log_scale, log_scale),
                     alpha=.7)
    else:
        g.plot_joint(sns.scatterplot, alpha=.5)

    plt.savefig('2d_density_partonshower_sd_example'+extra_label+'.pdf',
                format='pdf')


    # Only RSS
    groomed_data_rss = {'Ungroomed ECF': ecfs_bare,
                        'Groomed ECF': ecfs_rss}
    groomed_dataframe_rss = pd.DataFrame(data=groomed_data_rss)
    g = sns.JointGrid('Ungroomed ECF', 'Groomed ECF',
                      groomed_dataframe_rss,
                      xlim=[1e-3,.4],ylim=[1e-3,.4])
    g.plot_marginals(sns.histplot, kde=True, log_scale=log_scale,
                     color='orange')

    if show_histplot == True:
        g.plot_joint(sns.histplot, log_scale=(log_scale, log_scale),
                     alpha=.7)
    else:
        g.plot_joint(sns.scatterplot, alpha=.5, color='orange')

    plt.savefig('2d_density_partonshower_rss_example'+extra_label+'.pdf',
                format='pdf')


if __name__ == '__main__':
    jet_list = get_jet_list(1e4)
    plot_ecf_density_2d(jet_list, z_cut=.1, beta=2., f=1., show_histplot=False)
    plot_ecf_density_2d(jet_list, z_cut=.1, beta=2., f=1., show_histplot=True)
