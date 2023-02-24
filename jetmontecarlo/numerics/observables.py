import numpy as np

# Local imports
from jetmontecarlo.utils.vector_utils import angle

#######################################
# Observables used in MC integration:
#######################################
# ---------------------------------------------------
# Two-Point Generalized Energy Correlation Functions:
# ---------------------------------------------------
# See, for example:
# https://arxiv.org/pdf/1402.2657.pdf#page=4&zoom=200,0,300

def C_groomed(z_crit, theta_crit, z_cut, beta,
              z_pre=0., f=1., acc='LL', verbose=0):
    """The contribution of a groomed, critical emission
    to a generalized energy correlation function (ECF).

    Parameters
    ----------
    z_crit : float
        The energy fraction of the critical emission.
    theta_crit : float
        The angle of the critical emission.
    z_cut : float
        The energy fraction which is groomed by the grooming
        procedure.
    beta : float
        The paramter beta which controls the influence
        of the angle of an emission on the ECF
    z_pre : float
        The energy fraction of any pre-critical emissions,
        which may have `eaten up` some of the grooming
        parameter z_cut as in the recursive safe subtraction
        grooming procedure.
        Default is 0.
    f : float
        The fraction of the grooming parameter that grooms
        the harder branch of the emission.
        Default is 1.
    acc  : str
        The accuracy to which we calculate the observable.
        Default is leading logarithmic accuracy ('LL').

    Returns
    -------
    float
        A generalized ECF associated with the groomed
        emission or emissions.
        (critical by default, or critical + pre-critical).
    """
    if isinstance(f, list) or isinstance(f, (np.ndarray, np.generic)):
        assert all(0 <= f_i <= 1 for f_i in f), "f must be between 0 and 1!"
    else:
        assert 0 <= f <= 1, "f must be between 0 and 1!"
    # assert beta > 1, "beta must be greater than 1!"
    assert acc in ['LL', 'MLL'], "Accuracy must be LL or MLL!"

    # Groomed value of z_cut:
    z_cut_g = z_cut-z_pre

    # LL accuracy in the observable:
    if acc == 'LL':
        z_crit = np.minimum(z_crit, 1.-z_crit)
        return ((z_crit - f*z_cut_g) * (1.-(1.-f)*z_cut_g)
                * theta_crit**beta) / (1.-z_cut)**2.
    # MLL accuracy in the observable:
    z_crit = np.minimum(z_crit, 1.-z_crit)
    C = ((z_crit - f*z_cut_g) * (1. - z_crit - (1.-f)*z_cut_g)
         * theta_crit**beta) / (1.-z_cut)**2.
    if verbose > 0:
        print("    zcrit: "+str(z_crit))
        print("    z_crit - f*z_c: "+str(z_crit - f*z_cut_g))
        print("    1. - z_crit - (1.-f)*z_c: "
              +str(1. - z_crit - (1.-f)*z_cut_g))
        print("    theta_crit**beta: "+str(theta_crit**beta))
        print("    C : " + str(C))
    return C


def C_ungroomed(z, theta, beta, acc='LL'):
    """The contribution of an ungroomed emission
    to a generalized energy correlation function (ECF).

    Parameters
    ----------
    z : float
        The energy fraction of the ungroomed emission.
    theta : float
        The angle of the ungroomed emission.
    acc  : str
        The accuracy to which we calculate the observable.

    Returns
    -------
    float
        A generalized ECF associated with the ungroomed
        emission.
    """
    # assert beta > 1, "beta must be greater than 1!"
    assert acc in ['LL', 'MLL'], "Accuracy must be LL or MLL"

    # LL accuracy in the observable:
    if acc == 'LL':
        return z * theta**beta

    # MLL accuracy in the observable:
    return z * (1.-z) * theta**beta

def C_ungroomed_max(beta, radius, acc):
    if acc == 'LL':
        return radius**beta/2.
    return radius**beta/4.


#######################################
# Observables used in parton showers:
#######################################
def jet_C_ungroomed(jet, beta):
    """Finds the ungroomed jet generalized energy
    correlation function (GECF) given a jet produced by
    a parton shower.

    Parameters
    ----------
    jet : Jet
        A Jet class, which describes a collection of partons
        produced by a parton shower.
    beta : float
        The parameter beta of the GECF
    acc : string
        The accuracy to which we calculate the ungroomed correlator.
        'LL' or 'MLL'.

    Returns
    -------
    float
        A float corresponding to an ungroomed jet GECF
    """
    finalpartons = [parton for parton in jet.partons
                    if parton.isFinalState]
    P = jet.momentum.mag()
    C = 0
    for i, partoni in enumerate(finalpartons):
        for j, partonj in enumerate(finalpartons):
            if not i == j:
                momi = partoni.momentum
                momj = partonj.momentum
                z_i = momi.mag() / P
                z_j = momj.mag() / P
                theta = angle(momi, momj)

                if(np.isnan(z_i) or np.isnan(z_j) or np.isnan(theta)):
                    print(partoni.momentum.vector)
                    print(partonj.momentum.vector)

                C += z_i*z_j * theta**beta

    return C


def jet_C_groomed(jet, beta):
    """Finds the generalized energy correlation function (GECF)
    given a jet produced by a parton shower.
    Manually produces the _groomed_ correlation function by
    grooming the energy fractions of the jet using a grooming
    parameter z_c.

    Parameters
    ----------
    jet : Jet
        A Jet class, which describes a collection of partons
        produced by a parton shower.
    beta : float
        The parameter beta of the GECF
    acc : string
        The accuracy to which we calculate the ungroomed correlator.
        'LL' or 'MLL'.

    Returns
    -------
    float
        A float corresponding to an ungroomed jet GECF
    """
    pass
    finalpartons = [parton for parton in jet.partons
                    if parton.isFinalState]
    P = jet.momentum.mag()
    C = 0
    for i, partoni in enumerate(finalpartons):
        for j, partonj in enumerate(finalpartons):
            if not i == j:
                momi = partoni.momentum
                momj = partonj.momentum
                z_i = momi.mag() / P
                z_j = momj.mag() / P
                theta = angle(momi, momj)

                if(np.isnan(z_i) or np.isnan(z_j) or np.isnan(theta)):
                    print("Found a NaN value!")
                    print(partoni.momentum.vector)
                    print(partonj.momentum.vector)

                C += z_i*z_j * theta**beta

    return C
