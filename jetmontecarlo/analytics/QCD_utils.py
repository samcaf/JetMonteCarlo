import numpy as np
from mpmath import polylog, hyp2f1

# Jets are quark or gluon jets:
valid_jet_types = ['quark', 'gluon']
def check_jet_type(jet_type):
    """Verifies that the given jet type is either
    'quark' or 'gluon'

    Parameters
    ----------
    jet_type : string
        Description of the type of jet: 'quark' or 'gluon'
    """
    assert jet_type in valid_jet_types, \
        str(jet_type) + "is not a supported jet type."

# ---------------------------------------------------
# Factors from QCD Dynamics
# ---------------------------------------------------
# Group theory factors
CF = 8./6.
CA = 3.
TF = 1./2.
N_F = 5.

def CR(jet_type):
    """Casimir of the representation of the given jet_type."""
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return CF
    #elif jet_type == 'gluon':
    return CA

# QCD Beta function coefficient with N_F flavors
beta_0 = (33 - 2*N_F)/(12 * np.pi)

# Hard-Collinear factors
B_Q = -3./4.
B_G = -11./12. + N_F/(6*CA)

def b_i_sing(jet_type):
    """Factors which encode the non-singular piece of quark
    or gluon splitting functions. In particular,
    b_i = int dz_0^1 p_i^(non-singular)[z]
    """
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return B_Q
    #elif jet_type == 'gluon':
    return B_G

# Mass scales
M_Z = 91.19 # GeV
P_T = 3000. # GeV

# Euler constant, emerges for multiple emissions
euler_constant = 0.5772156649

# ---------------------------------------------------
# Strong Coupling
# ---------------------------------------------------
def alpha1loop(mu):
    """1-loop strong force coupling. Argument in GeV."""
    return .12/(1 + 2*.12*beta_0*np.log(mu/M_Z))

def alpha_s(z, theta):
    """Non-perturbative value of alpha_S, using the scale
    determined by parameters z, theta of a splitting
    """
    return alpha1loop(np.maximum(P_T*z*theta, 1.))

# ---------------------------------------------------
# Splitting functions
# ---------------------------------------------------
def q_splitting(z):
    """Returns quark splitting function"""
    return CF * ((1 + (1 - z)**2.)/z)
def g_splitting(z):
    """Returns gluon splitting function"""
    return CA * (
        2.*(1.-z)/z + z*(1.-z)
        + (TF*N_F)*(z**2.+(1.-z)**2.)/CA
        )

def splitting(z, jet_type):
    """Returns the splitting function of a given type."""
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return q_splitting(z)
    #elif jet_type=='gluon':
    return g_splitting(z)

def q_splitting_bar(z):
    """Returns quark 'reduced' splitting function"""
    return q_splitting(z) + q_splitting(1-z)
def g_splitting_bar(z):
    """Returns gluon 'reduced' splitting function"""
    return g_splitting(z) + g_splitting(1-z)

def splitting_bar(z, jet_type):
    """Returns the reduced splitting function
    for a given jet_type
    """
    check_jet_type(jet_type)
    if jet_type == 'quark':
        return q_splitting_bar(z)
    #elif jet_type == 'gluon':
    return g_splitting_bar(z)

# Final, most general form of the splitting function:
def splittingFn(z, jet_type, accuracy):
    """Returns the overall splitting function
    for a given jet_type and accuracy.

    Parameters
    ----------
    z : float
        Energy fraction at which we evaluate the splitting function
    jet_type : string
        Description of the type of jet: 'quark' or 'gluon'
    accuracy : string
        Description of the accuracy of the calculation:
        'LL': only the singular piece of the splitting function
        or
        'MLL': the full reduced splitting function

    Returns
    -------
    float
        The splitting function of a jet_type jet
        at energy fraction z for the desired accuracy.
    """
    assert accuracy in ['LL', 'LL_nonsing', 'MLL'], "Invalid accuracy."
    # At LL, use the singular pieces only
    if accuracy == 'LL':
        return 2.*CR(jet_type)/z
    # Here, we include non-singular pieces, but not reduced splitting functions
    elif accuracy == 'LL_nonsing':
        return splitting(z, jet_type)
    # otherwise, use reduced splitting functions
    return splitting_bar(z, jet_type)

# ---------------------------------------------------
# Jet Parameters
# ---------------------------------------------------
R0 = 1.
alpha_fixed = alpha1loop(P_T*R0)

MU_NP = 1./(P_T*R0) # Unitless non-perturbative scale

# ---------------------------------------------------
# Other useful function definitions
# ---------------------------------------------------
def W(x):
    """W(x) = x log(x). This was used, for example, in the appendices
    of Soft Drop, and is a very useful function for shortening the
    MLL/running coupling integral expressions
    """
    return x * np.log(x)

def Lambda(x, alpha):
    """Another useful function definition from Soft Drop for
    running coupling calculations"""
    return 2.*alpha*beta_0*np.log(x)

def b_q_bar(z_c):
    """Result of integrating from z_c to 1/2 of the
    hard collinear part of the quark splitting function"""
    b_q_zc = CF * (-3. + 6. * z_c + 4.* np.log(2. - 2.*z_c))/2.
    return b_q_zc

def b_g_bar(z_c):
    """Result of integrating from z_c to 1/2 of the
     hard collinear part of the gluon splitting function"""
    b_g_zc = (
        (-1.+2.*z_c)*(-4.*N_F*TF*(1. + (-1. + z_c)*z_c)
                      + CA * (11. + 2.*(-1. + z_c)*z_c)) / 6.
        + 2.*CA*np.log(2. - 2.*z_c))
    return b_g_zc

# Setting up polylog for np-friendly use:
polylog_vec = np.frompyfunc(polylog, 2, 1)
hyp2f1_vec = np.frompyfunc(hyp2f1, 4, 1)
