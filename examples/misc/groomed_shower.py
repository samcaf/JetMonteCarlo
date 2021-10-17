from jetmontecarlo.utils.partonshower_utils import *

# ----------------------------------
# Recursive Groomed Shower:
# ----------------------------------
def angularity_shower_groomed(parton, ang_init,
                              z_cut, beta, f,
                              jet_type, jet,
                              acc='LL',
                              cutoff=MU_NP):
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
    if cutoff == 'crit':
        types = [parton.type for parton in jet.partons]
        continue_shower = 'crit' not in types

    # If the shower is cut off by a non-perturbative scale,
    # set by default to MU_NP,
    else:
        continue_shower = ang_init > cutoff

    if continue_shower:
        ang_final, z, theta = angularity_split(ang_init, beta,
                                               jet_type, acc)
        if ang_final is None:
            return
        # Splitting
        parton.split(z, theta)
        d1, d2 = parton.daughter1, parton.daughter2

        mag1 = d1.momentum.mag()
        mag2 = d2.momentum.mag()

        # If there is no grooming to be done at all:
        if z_cut == 0:
            cut_soft = 0
            d1.type = 'sub'
            jet.partons.append(d1)

        # If the softer parton is not groomed away:
        elif mag1 >= f*P_T*z_cut:
            # Grooming the momentum of the softer parton
            cut_soft = f*z_cut
            d1.momentum.vector = d1.momentum.vector * (1. - cut_soft*P_T/mag1)
            # This ensures |p^groomed|/PT = z_soft^gr = z_soft - f*z_cut
            d1.type = 'crit'
            jet.partons.append(d1)

            # In the case z_soft > f*z_cut, we use up all of the grooming!
            z_cut = 0.

        # If the softer parton is entirely groomed away:
        else:
            # z_soft^groomed = z_soft - cut_soft = 0
            cut_soft = mag1/P_T
            d1.momentum.vector = d1.momentum.vector * 0.
            d1.type = 'pre'
            jet.partons.append(d1)

            # In the case z_soft < f*z_cut, we use only some of the grooming!
            z_cut = z_cut - mag1/(P_T * f)

        # Grooming the harder parton
        cut_hard = (1.-f)*cut_soft / f

        d2.momentum.vector = d2.momentum.vector * (1.-cut_hard*P_T/mag2)
        # Again, this ensures |p^groomed|/PT = z_hard^gr = z_hard - cut_hard

        d2.type = 'core'
        jet.partons.append(d2)
        angularity_shower_groomed(d2, ang_final,
                                  z_cut, beta, f,
                                  jet_type, jet, acc,
                                  cutoff=cutoff)



# -----------------------------------------
# Groomed Parton Shower:
# -----------------------------------------
def gen_jets_groomed(num_events, z_cut, beta, f,
                     jet_type='quark',
                     radius=1., acc='LL',
                     cutoff='NP'):
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
        angularity_shower_groomed(mother, ang_init,
                                  z_cut, beta, f,
                                  jet_type, jet, acc=acc,
                                  cutoff=cutoff)
        jet.has_showered = True

        jet_list.append(jet)

    return jet_list
