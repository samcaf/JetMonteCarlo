import random
import numpy as np

# Local utils:
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.analytics.QCD_utils import *
from jetmontecarlo.utils.partonshower_utils import *


###########################################
# Single emission utilities:
###########################################

# The following code attempts to use the veto method to produce
# critical emissions in RSS groomed jets at LL or MLL accuracy,
# at the level of accuracy of single emissions!

# Here, "LL" really means "fixed coupling", and "MLL" means
# "running coupling". However, we leave "LL" and "MLL" for
# easy compatibility with other parts of the code

def angularity_split_precrit(ang_init, z_cut, beta,
                             jet_type, acc='LL'):
    """A method which determines the angle and
    momentum fraction of a parton-level splitting.

    Parameters
    ----------
    ang_init : float
        An angularity associated with the parton to be split.
        This sets the scale for the splitting.
    z_cut : float
        A grooming parameter which determines the scale of the critical
        emission at single emission accuracy.
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
    assert(acc in ['LL', 'MLL']), \
    "Invalid calculation accuracy."

    if acc == 'LL':
        alpha = alpha_fixed
    else:
        alpha = 2.5

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
        # Notice that alpha is 1. for the MLL case. This is always
        # larger than alpha(freezing scale = 1 GeV) ~ 0.35
        if acc == 'MLL':
            pdf = alpha_s(z, theta) * splitting_bar(z, jet_type)
            cut = (pdf / (2.*CR(jet_type)*alpha/z))

            # Note: there is a typo in Eqn 5.4 of the resource I
            # cite above the angularity_split method which is
            # important here (a stray factor of 2 on the RHS),
            # but the conclusion in Equation 5.5, which we use
            # here, is correct

            if random.random() < cut:
                accept_emission = True
            # These next two lines of code are the soul of the veto
            # algorithm. Rather than generating from scratch, as you
            # would for the von Neumann acceptance-rejection algorithm,
            # you use this scale as the scale for the next emission.
            # This correctly takes into account the exponentiation of
            # multiple emissions, as described in the Pythia manual:
            # https://arxiv.org/pdf/hep-ph/0603175.pdf#page=66&zoom=150,0,240
            else:
                ang_init = ang_final

        if z > z_cut:
            accept_emission = True
        else:
            accept_emission = False
            z_cut = z_cut-z
            ang_init = ang_final
    return ang_final, z, theta, z_cut

def angularity_shower_groomed_veto_precrit(parton, ang_init,
                                           z_cut, beta, f,
                                           jet_type, jet,
                                           acc='LL'):
    """Starts with an initial parton and jet, and
    performs a recursive angularity shower, defined by
    the splitting algorithm angularity_split above.

    This procedure modifies the Jet object, which
    will subsequently contain all of the information
    about the parton shower.

    Note:
    Here, we won't even give an option to split the softer
    parton. That would require keeping closer track of
    z_cut, and this extra effort won't contribute at LL or
    even MLL/multiple emissions!

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
    assert f >= 1/2, "For now, I have only considered f >= 1/2."

    types = [parton.type for parton in jet.partons]

    # If we want to explore what happens when we have no non-perturbative
    # cutoff, and continue until we have at least one ungroomed emission,
    types = [parton.type for parton in jet.partons]
    continue_shower = 'crit' not in types

    if continue_shower:
        ang_final, z, theta, z_cut = angularity_split_precrit(ang_init,
                                                              f*z_cut, beta,
                                                              jet_type, acc)
        if ang_final is None:
            return
        # Splitting
        parton.split(z, theta)
        d1, d2 = parton.daughter1, parton.daughter2

        mag1 = d1.momentum.mag()
        mag2 = d2.momentum.mag()

        # If the softer parton is too soft, we haven't correctly
        # generated a critical emission.
        assert mag1 >= f*P_T*z_cut,\
        "Emission is not hard enough to be critical!"

        # Grooming the momentum of the softer parton
        cut_soft = f*z_cut
        d1.momentum.vector = d1.momentum.vector * (1. - cut_soft*P_T/mag1)
        # This ensures |p^groomed|/PT = z_soft^gr = z_soft - f*z_cut
        d1.type = 'crit'
        jet.partons.append(d1)

        # In the case z_soft > f*z_cut, we use up all of the grooming!
        z_cut = 0.

        # Grooming the harder parton
        cut_hard = (1.-f)*cut_soft / f

        d2.momentum.vector = d2.momentum.vector * (1.-cut_hard*P_T/mag2)
        # Again, this ensures |p^groomed|/PT = z_hard^gr = z_hard - cut_hard

        d2.type = 'core'
        jet.partons.append(d2)

        # At this point, we do not need to keep going with a recursive
        # shower algorithm. We only want to consider the precritical
        # and critial emissions.

# -----------------------------------------
# One Emission/Critical Parton Shower:
# -----------------------------------------
def gen_jets_groomed_precrit(num_events, z_cut, beta, f,
                             jet_type='quark',
                             radius=1., acc='LL'):
    """Generates a list of num_events jet_type jets
    through angularity based parton showering.
    """
    # Setting up for angularities
    jet_list = []

    for _ in range(int(num_events)):
        # Initializing a parton
        ang_init = radius**beta / 2.
        momentum = Vector([0, P_T, 0])

        # Performing parton shower to produce jet
        jet = Jet(momentum, radius, partons=None)
        mother = jet.partons[0]
        mother.type = 'mother'
        angularity_shower_groomed_veto_precrit(mother, ang_init,
                                               z_cut, beta, f,
                                               jet_type, jet,
                                               acc=acc)
        jet.has_showered = True

        jet_list.append(jet)

    return jet_list
