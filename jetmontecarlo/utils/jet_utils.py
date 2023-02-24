from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class JetAlgorithm(ABC):
    """Class for keeping track of jet algorithms."""
    radius: float
    description: str


def alg_to_string(jet_alg, latex=True):
    """
    Returns a string naming the jet algorithm with the
    given index.

    Parameters
    ----------
        jet_alg : int
            Integer indicating the FastJet index associated with the jet
            algorithm.

    Returns
    -------
        str
            A (LaTeX-compatible, if requested) string naming the algorithm.
    """
    # - - - - - - - - - - - - - - - - -
    # pp algorithms
    # - - - - - - - - - - - - - - - - -
    if jet_alg in [0, "0", "kt"]:
        if latex:
            return r'$k_T$'
        else:
            return 'kt'
    elif jet_alg in [1, "1", "ca", "cambridge-aachen"]:
        if latex:
            return r'C/A'
        else:
            return 'ca'
    elif jet_alg in [2, "2", "akt", "anti-kt", "antikt"]:
        if latex:
            return r'anti-$k_T$'
        else:
            return 'akt'
    # - - - - - - - - - - - - - - - - -
    # ee algorithms
    # - - - - - - - - - - - - - - - - -
    if jet_alg in ["ee_0", "ee_kt"]:
        if latex:
            return r'$k_T^{(e^+ e^-)}$'
        else:
            return 'ee_kt'
    elif jet_alg in ["ee_1", "ee_ca", "ee_cambridge-aachen"]:
        if latex:
            return r'C/A$\?^{(e^+ e^-)}$'
        else:
            return 'ee_ca'
    elif jet_alg in ["ee_2", "ee_akt", "ee_anti-kt", "ee_antikt"]:
        if latex:
            return r'anti-$k_T^{(e^+ e^-)}$'
        else:
            return 'ee_akt'
    else:
        raise AssertionError(f"Invalid jet algorithm {jet_alg}")


