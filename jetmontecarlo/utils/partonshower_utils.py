import random
import numpy as np
from pathlib import Path

# Local utils:
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.analytics.QCD_utils import *

########################################
# Basic Methods and Classes:
########################################

def split_momentum(mom_vector, z, theta):
    """Splits the momentum vector mom_vector
    into two new momenta separated by an angle
    theta, of which the softer momentum has
    momentum fraction:
    |p_soft| = z |p_original|
    """
    assert mom_vector.dim() == 3, \
        "Currently only splitting 3D vectors!"
    assert mom_vector.mag() > 0, \
        "Cannot split zero momentum."

    q = Vector(mom_vector.vector * z)
    k = Vector(mom_vector.vector * (1-z))

    perp_vector = rand_perp_vector(mom_vector)

    q = q.rotate_around(perp_vector, theta/2)
    k = k.rotate_around(perp_vector, -theta/2)

    return q, k

def split_parton(parton, z, theta):
    """Splits a parton with momentum vector mom_vector
    into two new partons with momenta separated by an angle
    theta, of which the softer parton has momentum fraction:
    |p_soft| = z |p_original|
    """
    mom_vector = parton.momentum

    q, k = split_momentum(mom_vector, z, theta)

    daughter1 = Parton(q)
    daughter1.mother = parton
    daughter2 = Parton(k)
    daughter2.mother = parton

    return daughter1, daughter2

class Parton():
    """A class which encodes the information contained by a parton,
    including momentum and daughter partons due to splittings.
    Also includes methods to allow partons to split.
    """
    def __init__(self, momentum):
        # Initializing kinematic information
        self.momentum = momentum
        self.mass = None

        # Initializing information for branching
        self.mother = None
        self.daughter1 = None
        self.daughter2 = None

        # Initializing extra identifiers
        self.pid = None
        self.isFinalState = True

        # An additional identifier for more complicated manipulations
        self.type = None
        # For example, type could indicate an emission type pulled
        # 'precrit', 'crit', or 'sub'
        # to single out specific emissions which contribute to observables

    def split(self, z, theta):
        """A method which splits a parton into two
        daughter partons separated by angle theta,
        of which the softer parton has momentum fraction:
        |p_soft| = z |p_original|

        The daughter partons are then encoded into this
        parton as thisparton.daughter1 and thisparton.daughter2.
        """
        daughter1, daughter2 = split_parton(self, z, theta)
        self.daughter1 = daughter1
        self.daughter2 = daughter2
        self.isFinalState = False

class Jet:
    """A class which encodes a list of partons,
    designed for convenient use in showering algorithms.
    """
    def __init__(self, momentum, radius, partons=None):
        assert momentum.mag() > 0, "Unphysical transverse momentum."
        assert radius > 0, "Unphysical jet radius."

        self.momentum = momentum
        self.R = radius

        self.partons = []
        if partons is None:
            self.partons.append(Parton(momentum))
        else:
            for parton in partons:
                self.partons.append(parton)

        self.has_showered = False

    def combine_with(self, jet_to_add):
        """Considers the partons of this jet with another jet
        to produce a new jet object"""
        pass

########################################
# Angularity Based Showers:
########################################
# From https://arxiv.org/pdf/1307.1699.pdf#page=17&zoom=100,114,698

# ----------------------------------
# Obtaining splitting factors:
# ----------------------------------
# LL described at:
# https://arxiv.org/pdf/1307.1699.pdf#page=18&zoom=200,0,300%5D
# MLL described right below:
# https://arxiv.org/pdf/1307.1699.pdf#page=19&zoom=200,0,200%5D
# Additional vetos discussed in Equation 5.5

def angularity_split(ang_init, beta, jet_type,
                     acc='LL'):
    """A method which determines the angle and
    momentum fraction of a parton-level splitting.

    Parameters
    ----------
    ang_init : float
        An angularity associated with the parton to be split.
        This sets the scale for the splitting.
    beta : float
        The parameter beta which defines the angularity ang_init.
    jet_type : string
        The type of parton we are splitting: 'quark' or 'gluon'
    acc : string
        The accuracy of the parton shower:
        'LL':
            Leading log, fixed coupling, only singular
            pieces of the splitting function.
        'MLL' or 'ME':
            Modified leading log, running coupling, includes
            non-singular pieces of the splitting function.

    Returns ang_final, z, theta
    -------
    float, float, float
        The angularity of the splitting, the momentum fraction of
        the splitting, and the angle of the splitting.
    """
    assert(acc in ['LL', 'MLL', 'ME']), \
    "Invalid calculation accuracy."

    if acc == 'LL':
        alpha = alpha_fixed
    else:
        alpha = 1.5

    accept_emission = False
    while not accept_emission:
        r1, r2 = random.random(), random.random()

        ang_final = np.exp(-np.sqrt(np.log(2.*ang_init)**2.
                                    - np.pi*beta/(CR(jet_type)*alpha)
                                    * np.log(r1))) / 2.

        z = (2.*ang_final)**r2 / 2.
        theta = (2.*ang_final)**((1-r2) / beta)

        # If the accuracy is LL, use this emission
        if acc == 'LL':
            accept_emission = True

        # In the MLL case, we use the acceptance-rejection
        # method to generate events associated with running coupling.
        # Notice that alpha is O(1) for the MLL case. This is always
        # larger than alpha(freezing scale = 1 GeV) ~ 0.35

        if acc in ['MLL', 'ME']:
            pdf = alpha_s(z, theta) * splitting_bar(z, jet_type)
            cut = (pdf / (2.*CR(jet_type)*alpha/z))
            # Note: there is a typo in Eqn 5.4 of the resource I
            # cite above the angularity_split method which is
            # important here (a stray factor of 2 on the RHS),
            # but the conclusion in Equation 5.5, which we use
            # here, is correct

            # Nonsense!
            if cut > .6:
                print("beep boop")
            if cut > .8:
                print("close")
            if cut > .9:
                print("closer")
            if cut > .95:
                print("even closer")

            if cut > 1:
                raise ValueError("The pdf must be everywhere less than the"
                                 + "proposed pdf!")
            if random.random() < cut:
                # If we accept the emission, stop the algorithm here
                accept_emission = True
            else:
                # Otherwise, continue but reset the scale of the emission
                ang_init = ang_final

            # The above lines of code are the soul of the veto algorithm:
            # rather than generating from scratch, as you would
            # for the von Neumann acceptance-rejection algorithm,
            # you use this scale as the scale for the next emission.
            # This correctly takes into account the exponentiation of
            # multiple emissions, as described in the Pythia manual:
            # https://arxiv.org/pdf/hep-ph/0603175.pdf#page=66&zoom=150,0,240

    return ang_final, z, theta

# ----------------------------------
# Recursive Shower:
# ----------------------------------
def angularity_shower(parton, ang_init, beta, jet_type, jet,
                      acc='LL', split_soft=True, cutoff=MU_NP):
    """Starts with an initial parton and jet, and
    performs a recursive angularity shower, defined by
    the splitting algorithm angularity_split above.

    This procedure modifies the Jet object, which
    will subsequently contain all of the information
    about the parton shower.

    Parameters
    ----------
    parton : Parton
        The mother parton of the parton shower process.
    ang_init : float
        The angularity associated with the mother parton.
    beta : float
        The parameter beta which defines the angularity ang_init.
    jet_type : string
        The type of parton we are splitting: 'quark' or 'gluon'
    jet : Jet
        A Jet object containing the given mother parton,
        to encode/contain the parton shower.
    acc : string
        The accuracy of the parton shower:
        'LL':
            Leading log, fixed coupling, only singular
            pieces of the splitting function.
        'MLL' or 'ME':
            Modified leading log, running coupling, includes
            non-singular pieces of the splitting function.
    split_soft : bool
        A boolean which determines whether the softer
        parton of each splitting (the one with momentum
        fraction z) will itself be split.

    Returns
    -------
        None
    """

    if ang_init > cutoff:
        ang_final, z, theta = angularity_split(ang_init, beta,
                                               jet_type, acc)
        if ang_final is None:
            return

        parton.split(z, theta)
        d1, d2 = parton.daughter1, parton.daughter2
        d1.type = 'soft'
        d2.type = 'hard'

        jet.partons.append(d1)
        if split_soft:
            angularity_shower(d1, ang_final, beta, jet_type, jet,
                              acc, split_soft, cutoff)

        jet.partons.append(d2)
        angularity_shower(d2, ang_final, beta, jet_type, jet,
                          acc, split_soft, cutoff)


######################################################
# Producing sets of jets:
######################################################
# -----------------------------------------
# Basic jet production:
# -----------------------------------------
def gen_jets(num_events, beta=2.,
             jet_type='quark',
             radius=1., acc='LL',
             cutoff=MU_NP,
             verbose=1):
    """Generates a list of num_events jet_type jets
    through angularity based parton showering.
    """
    if verbose > 0:
        print("Generating {:.0e} shower events...".format(num_events))
    # Setting up for angularities
    jet_list = []

    for i in range(int(num_events)):
        if (i+1)%(int(num_events/10)) == 0 and verbose > 0:
            print("   Generated "+str(i+1)+" events")
        # Initializing a parton
        ang_init = radius**beta / 2.
        momentum = Vector([0, P_T, 0])

        # Performing parton shower to produce jet
        jet = Jet(momentum, radius, partons=None)
        mother = jet.partons[0]
        mother.type = 'mother'
        angularity_shower(mother, ang_init, beta, jet_type,
                          jet, acc=acc, split_soft=False,
                          cutoff=cutoff)
        jet.has_showered = True

        # Adding jet to list
        jet_list.append(jet)

    return jet_list

########################################
# Ordering Jets:
########################################
def angular_ordered(jet):
    #pass # Incomplete!
    """Takes in a jet event and outputs the same event, ordered by angle
    rather than, say, angularity.

    This is designed to work in particular on parton showers at single
    emission accuracy, in the sense that when a parton splits, only the
    harder of the 2 daughters undergoes further splitting.

    Here's the idea:
    1. Loop through pairs of emissions which share a mother parton.
    2. Find the angle between each pair. Record the pair and angle
    3. Sort the parton pairs by angle.
    4. Flatten the array of parton pairs.

    In this way, any loop through the partons of the jet will
    """
    if isinstance(jet, list):
        return np.array([angular_ordered(j) for j in jet])

    ordered_partons = []
    angles = []

    for parton in jet.partons:
        if parton.mother is None:
            # If this is the first parton in the jet
            ordered_partons.append([parton])
            angles.append(jet.R)
        if parton.daughter1 is not None and parton.daughter2 is not None:
            ordered_partons.append([parton.daughter1, parton.daughter2])
            angles.append(angle(parton.daughter1.momentum,
                                parton.daughter2.momentum))
    # Taking care of nans. Nans mess with the ordering procedure below
    angles = np.nan_to_num(angles)

    # Sort the partons by decreasing angle of emission
    ordered_partons = np.hstack([parton for _, parton in
                                 sorted(zip(angles, ordered_partons),
                                        key=lambda pair: pair[0],
                                        reverse=True)])

    ordered_angles = np.hstack([angle for angle, _ in
                                 sorted(zip(angles, ordered_partons),
                                        key=lambda pair: pair[0],
                                        reverse=True)])

    ordered_jet = Jet(jet.momentum, jet.R, partons=ordered_partons)
    ordered_jet.has_showered = True
    return ordered_jet

########################################
# Observables:
########################################
# -----------------------------------------
# Angularity calculation:
# -----------------------------------------
# Remember that at leading logarithmic accuracy, the generalized
# jet correlation functions are simply angularities!
def getangs(jet_list, beta=2., acc='LL', emission_type=None,
            verbose=1):
    """Retrieves a list of angularities from a list of jets."""
    # Setting up a list with angularities for each event
    angs = []

    for i, jet in enumerate(jet_list):
        # Setting up a list with angularities for each parton
        parton_angs = []
        for mother in jet.partons:
            daughter1 = mother.daughter1
            daughter2 = mother.daughter2
            if daughter1 is None or daughter2 is None:
                continue

            theta = angle(daughter1.momentum, daughter2.momentum)
            z = min(daughter1.momentum.mag()/mother.momentum.mag(),
                    daughter2.momentum.mag()/mother.momentum.mag())

            if emission_type in [daughter1.type, daughter2.type, None]:
                parton_angs.append(z * theta**beta)

        if len(parton_angs) == 0:
            # If there is no angularity found (for example, if there is
            # no emission of the required type), add this event to the
            # "zero bin" by giving it an infinitesimal value
            angularity = 1e-50

        # The LL result is dominated by a single emission
        elif acc in ['LL', 'MLL']:
            angularity = max(parton_angs)

        # The ME result sums many emissions
        # (note: the definition is nebulous, but usually does not
        # allow the softer partons of a splitting to split themselves)
        # For now, we get this by summing the two strongest emissions.
        elif acc == 'ME':
            ang1 = max(parton_angs)
            parton_angs.pop(parton_angs.index(ang1))
            ang2 = max(parton_angs)
            angularity = sum(ang1+ang2)

        if angularity >= 0:
            angs.append(angularity)

        # Additional verbose messages
        if (i+1)%(int(len(jet_list)/5)) == 0 and verbose > 0:
            print("       Generated "+str(i+1)+" ungroomed correlations.")

    return angs

# -----------------------------------------
# ECF Calculations:
# -----------------------------------------
def getECFs_ungroomed(jet_list, beta=2.,
                      obs_acc='LL', n_emissions=1,
                      verbose=1):
    """Retrieves a list of ungroomed jet energy correlation functions from a
    list of ungroomed jets.
    """
    all_ecfs = []

    # Preparing calculation of critical angularity
    def ecf_fn(z, theta):
        if obs_acc == 'LL':
            return z * theta**beta
        return z * (1.-z) * theta**beta

    for i, jet in enumerate(jet_list):
        cs = [1e-100]

        for mother in jet.partons:
            # LL calculation of angularity.
            # Note in particular that this does not take into account
            # splitting of the softer partons at each branch.
            # Such a study would take more effort, but fortunately is
            # beyond the accuracy we would like to achieve here.
            daughter1 = mother.daughter1
            daughter2 = mother.daughter2
            if daughter1 is None or daughter2 is None:
                continue

            theta = angle(daughter1.momentum, daughter2.momentum)
            z = min(daughter1.momentum.mag()/mother.momentum.mag(),
                    daughter2.momentum.mag()/mother.momentum.mag())

            cs.append(ecf_fn(z, theta))

        if n_emissions == 'all':
            ecf = sum(cs)

        else:
            ecf_list = []
            for i in range(int(n_emissions)):
                ecfi = max(cs)
                ecf_list.append(ecfi)
                cs.pop(cs.index(ecfi))
                if len(cs) == 0: break
            ecf = sum(ecf_list)

        if ecf >= 0:
            all_ecfs.append(ecf)

        # Additional verbose messages
        if (i+1)%(int(len(jet_list)/5)) == 0 and verbose > 0:
            print("       Generated "+str(i+1)+" "+str(n_emissions)
                  +" emissions ungroomed correlations.")

    return all_ecfs

def getECFs_rss(jet_list, z_cut, beta=2., f=1.,
                obs_acc='LL', emission_type='precritsub',
                n_emissions=1, few_emissions=False,
                verbose=1):
    """Retrieves a list of groomed jet energy correlation functions from a
    list of *ungroomed* jets.
    In particular, ungroomed jets do not record "types" of emissions, so in
    this method, we groom away pre-critical emissions to disccover the
    contributions from critical and subsequent emissions.

    Note in particular that this does not take into account
    splitting of the softer partons at each branch.
    Such a study would take more effort, but fortunately is
    beyond the accuracy we would like to achieve here.

    Parameters
    ----------
    jet_list : list
        List of jets to groom.
    z_cut : float
        Grooming parameter which determines the strength of the RSS procedure.
        Less than 1/2
    beta : float
        The value of beta associated with the ECF observable $C_1^{(\beta)}$.
    f : float
        Parameter which determines the energy fraction f of grooming which
        removes energy from the softer branch at each step.
        f = 1 means that no energy is removed from the hard branch at each stage
        of the grooming.
        Between 0 and 1.
    obs_acc : str
        Accuracy at which we calculate the observable $C_1^{(\beta)}$.
        'LL' or 'MLL'.
    emission_type : str
        Emissions considered in the calculation of the ECF.
        'crit', 'precrit', 'sub', or 'precritsub'
    n_emissions : int
        Number of emissions used to calculate ECF.
        Example: n_emissions = 1 calculates the contributionn from the leading
                 emission only
    few_emissions : bool
        .
    verbose : int
        Determines the verbosity of the messages returned by the code.

    Returns
    -------
    list
        List of floats corresponding to groomed jet ECFs for each jet in
        the jet_list
    """
    if z_cut == 0:
        return getECFs_ungroomed(jet_list, beta=beta, obs_acc=obs_acc,
                                 n_emissions=n_emissions, verbose=verbose)

    all_ecfs = []

    # Preparing calculation of critical angularity
    def ang_crit(z, theta, z_cut):
        if obs_acc == 'LL':
            return (z - f*z_cut) * theta**beta * (1. - (1.-f)*z_cut)
        return (z - f*z_cut) * theta**beta * (1. - z - (1.-f)*z_cut)

    for i, jet in enumerate(jet_list):
        # Setting up a z_cut/grooming parameter for the event:
        this_zcut = z_cut

        # Introducing scale factors for f!=1;
        # the harder branch gets groomed in this scenario, and subsequent
        # momenta will be scaled down as a result.
        scale_factor = 1.

        # Setting up for different emission types:
        c_crit = 1e-100
        theta_crit = 1e-100
        use_precrit = True
        z_pres = [1e-100]
        c_subs = [1e-100]

        for mother in jet.partons:
            daughter1 = mother.daughter1
            daughter2 = mother.daughter2
            if daughter1 is None or daughter2 is None:
                continue

            theta = angle(daughter1.momentum, daughter2.momentum)
            z = min(daughter1.momentum.mag()/mother.momentum.mag(),
                    daughter2.momentum.mag()/mother.momentum.mag())

            # Could also use the `true' momentum fraction:
            # ---------------
            # z = min(daughter1.momentum.mag()/jet.momentum.mag(),
            #         daughter2.momentum.mag()/jet.momentum.mag())
            # ---------------
            # This is an effect which contributes at higher accuracy;
            # to compare to lower order theory we will use the uncommented
            # code above

            # Finding the remaining z_cut available for a critical emission
            if few_emissions:
                cutoff_zcut = this_zcut - max(z_pres)
            else:
                cutoff_zcut = this_zcut - sum(z_pres)

            # If the emission is soft enough to be groomed (pre-critical):
            if 'precrit' in emission_type and use_precrit and z < cutoff_zcut:
                z_pres.append(z)

            # If the emission is the first to survive grooming (critical):
            elif 'crit' in emission_type and z >= cutoff_zcut > 0.:
                c_crit = ang_crit(z, theta, cutoff_zcut)

                theta_crit = theta
                this_zcut = 0.
                use_precrit = False

            # If we have already had a critical emission:
            elif 'sub' in emission_type and this_zcut == 0.:
                c_subs.append(ang_crit(z, theta, z_cut=0.))

        c_list = c_subs
        c_list.append(c_crit)

        if n_emissions == 'all':
            ecf = sum(c_list)

        else:
            ecf_list = []
            for i in range(int(n_emissions)):
                ecfi = max(c_list)
                ecf_list.append(ecfi)
                c_list.pop(c_list.index(ecfi))
                if len(c_list) == 0: break
            ecf = sum(ecf_list)

        if ecf >= 0:
            all_ecfs.append(ecf)

        # Additional verbose messages
        if (i+1)%(int(len(jet_list)/5)) == 0 and verbose > 0:
            print("       Generated "+str(i+1)+" "+str(obs_acc)
                  +" ("+str(n_emissions)
                  +" emissions) RSS correlations.")

    return all_ecfs

def getECFs_softdrop(jet_list, z_cut, beta=2., beta_sd=0.,
                     obs_acc='LL', n_emissions=1,
                     verbose=1, reproduce_approximations=True):
    """Retrieves a list of soft drop groomed jet energy correlation
    functions from a list of *ungroomed* jets.
    Note in particular that this does not take into account
    splitting of the softer partons at each branch.
    Such a study would take more effort, but fortunately is
    beyond the accuracy we would like to achieve here.

    Parameters
    ----------
    jet_list : list
        List of jets to groom.
    z_cut : float
        Grooming parameter which determines the strength of the RSS procedure.
        Less than 1/2
    beta : float
        The value of beta associated with the ECF observable $C_1^{(\beta)}$.
    beta_sd : float
        Parameter which determines which angles of radiation Soft Drop prefers
        to remove.
    obs_acc : str
        Accuracy at which we calculate the observable $C_1^{(\beta)}$.
        'LL' or 'MLL'.
    n_emissions : int
        Number of emissions used to calculate ECF.
        Example: n_emissions = 1 calculates the contributionn from the leading
                 emission only
    verbose : int
        Determines the verbosity of the messages returned by the code.
    reproduce_approximations : bool
        Determines whether or not the parton shower calculation reproduces
        some of the approximations used in the analytic calculation.
        Detailed in code comments below.

    Returns
    -------
    list
        List of floats corresponding to groomed jet ECFs for each jet in
        the jet_list
    """
    all_ecfs = []
    # Preparing calculation of critical angularity
    def ecf_val(z, theta):
        if obs_acc == 'LL':
            return z * theta**beta
        return z*(1.-z) * theta**beta

    for i, jet in enumerate(jet_list):
        # Setting up for different emission types
        soft_drop = True
        c_list = []

        for mother in jet.partons:
            # LL calculation of angularity.
            # Note in particular that this does not take into account
            # splitting of the softer partons at each branch.
            # Such a study would take more effort, but fortunately is
            # beyond the accuracy we would like to achieve here.
            daughter1 = mother.daughter1
            daughter2 = mother.daughter2
            if daughter1 is None or daughter2 is None:
                continue

            theta = angle(daughter1.momentum, daughter2.momentum)
            z = min(daughter1.momentum.mag()/mother.momentum.mag(),
                    daughter2.momentum.mag()/mother.momentum.mag())

            if not soft_drop and z > z_cut*theta**beta_sd:
                c_list.append(ecf_val(z, theta))
            if soft_drop == True and z > z_cut*theta**beta_sd:
                # The first emission corresponds to c_list[0]
                soft_drop = False
                c_list.append(ecf_val(z, theta))

        c_list.append(1e-100); c_list.append(1e-100)

        if n_emissions == 'all':
            ecf = sum(c_list)

        elif n_emissions == 1:
            if reproduce_approximations:
                # The first emission to satisfy Soft Drop is the leading
                # emission to the ECF, to MLL accuracy
                ecf = c_list[0]
            else:
                ecf = max(c_list)

        else:
            ecf_list = []
            if reproduce_approximations:
                # At <= MLL accuracy, the critical emission is dominant
                ecf_list.append(c_list[0])

                # Reduce the number of remaining emissions to consider
                c_list.pop(0)
                n_emissions = n_emissions-1

            for i in range(int(n_emissions)):
                ecfi = max(c_list)
                ecf_list.append(ecfi)
                c_list.pop(c_list.index(ecfi))
                if len(c_list) == 0: break
            ecf = sum(ecf_list)

        if ecf >= 0:
            all_ecfs.append(ecf)

        # Additional verbose messages
        if (i+1)%(int(len(jet_list)/5)) == 0 and verbose > 0:
            print("       Generated "+str(i+1)+" "+str(n_emissions)
                  +" emission Soft Drop correlations.")

    return all_ecfs

def getECFs(jet_list, groomer=None, **params)
    if groomer is None:
        return getECFs_ungroomed(jet_list, **params)
    elif groomer == 'RSS':
        return getECFs_rss(jet_list, **params)
    elif groomer in ['SoftDrop', 'SD', 'Soft Drop']:
        return getECFs_softdrop(jet_list, **params)
    else:
        raise ValueError('Invalid groomer type '+str(groomer)+'.')

def save_shower_correlations(jet_list, file_path,
                             z_cuts=[.05, .1, .2], beta=2.,
                             beta_sd=0., f_soft=1.,
                             obs_acc='LL', few_emissions=True,
                             fixed_coupling=True,
                             verbose=1):
    # Setting up lists of observables
    rss_c1s_crit, rss_c1s_critsub, rss_c1s_precrit = [], [], []
    softdrop_c1s_crit, softdrop_c1s_two, softdrop_c1s_three = [], [], []
    rss_c1s_precritsub, rss_c1s_two, rss_c1s_three = [], [], []
    rss_c1s_all, softdrop_c1s_all = [], []

    n_emissions_list = [1, 2, 'all']

    if verbose > 1:
        print("    beta for shower correlations: " +str(beta))

    ungroomed_c1s = []
    for n in n_emissions_list:
        ungroomed_c1s.append([getECFs_ungroomed(jet_list, beta=beta,
                                                obs_acc=obs_acc, n_emissions=n,
                                                verbose=verbose)])

    for z_cut in z_cuts:
        # RSS
        params = {'jet_list' : jet_list,
                  'z_cut' : z_cut,
                  'beta' : beta,
                  'f' : f_soft,
                  'obs_acc' : obs_acc,
                  'few_emissions' : few_emissions,
                  'verbose' : verbose}
        crit_c1s = getECFs_rss(**params, emission_type='crit')
        precrit_c1s = getECFs_rss(**params, emission_type='precrit')
        critsub_c1s = getECFs_rss(**params, emission_type='critsub')

        precritsub_c1s = []
        for n in n_emissions_list:
            precritsub_c1s.append([getECFs_rss(**params, emission_type='precritsub',
                                               n_emissions=n)])
        rss_c1s_crit.append(crit_c1s)
        rss_c1s_precrit.append(precrit_c1s)
        rss_c1s_critsub.append(critsub_c1s)
        rss_c1s_precritsub.append(precritsub_c1s[0])
        rss_c1s_two.append(precritsub_c1s[1])
        # rss_c1s_three.append(precritsub_c1s[2])
        rss_c1s_all.append(precritsub_c1s[2])

    for z_cut in z_cuts:
        # Soft Drop
        params = {'jet_list' : jet_list,
                  'z_cut' : z_cut,
                  'beta' : beta,
                  'beta_sd' : 0.,
                  'obs_acc' : obs_acc,
                  'verbose' : verbose}

        c1s_sd = []
        for n in n_emissions_list:
            c1s_sd.append([getECFs_softdrop(**params, n_emissions=n)])

        softdrop_c1s_crit.append(c1s_sd[0])
        softdrop_c1s_two.append(c1s_sd[1])
        # softdrop_c1s_three.append(c1s_sd[2])
        softdrop_c1s_all.append(c1s_sd[2])

    np.savez(file_path,
             ungroomed_c1s = np.asarray(ungroomed_c1s[0]),
             ungroomed_c1s_two = np.asarray(ungroomed_c1s[1]),
             # ungroomed_c1s_three = np.asarray(ungroomed_c1s[2]),
             ungroomed_c1s_all = np.asarray(ungroomed_c1s[2]),
             rss_c1s_crit = np.asarray(rss_c1s_crit),
             rss_c1s_critsub = np.asarray(rss_c1s_critsub),
             rss_c1s_precrit = np.asarray(rss_c1s_precrit),
             rss_c1s_precritsub = np.asarray(rss_c1s_precritsub),
             rss_c1s_two = np.asarray(rss_c1s_two),
             rss_c1s_three = np.asarray(rss_c1s_three),
             rss_c1s_all = np.asarray(rss_c1s_all),
             softdrop_c1s_crit = np.asarray(softdrop_c1s_crit),
             softdrop_c1s_two = np.asarray(softdrop_c1s_two),
             softdrop_c1s_three = np.asarray(softdrop_c1s_three),
             softdrop_c1s_all = np.asarray(softdrop_c1s_all)
             )