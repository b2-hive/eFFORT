import numpy as np


class PDG:
    """
    Lazy interface to PDG.
    TODO: Do something smart.
    """
    m_Bzero = 5.27963  # http://pdg.lbl.gov/2018/listings/rpp2018-list-B-zero.pdf
    m_Bplus = 5.27932  # http://pdg.lbl.gov/2018/listings/rpp2018-list-B-plus-minus.pdf
    m_Dzero = 1.86483  # http://pdg.lbl.gov/2018/listings/rpp2018-list-D-zero.pdf
    m_Dplus = 1.86965  # http://pdg.lbl.gov/2018/listings/rpp2018-list-D-plus-minus.pdf
    m_Dstarzero = 2.00685  # http://pdg.lbl.gov/2018/listings/rpp2018-list-D-star-2007-zero.pdf
    m_Dstarplus = 2.01026  # http://pdg.lbl.gov/2018/listings/rpp2018-list-D-star-2010-plus-minus.pdf
    G_F = 1.1663787e-5  # http://pdg.lbl.gov/2018/reviews/rpp2018-rev-phys-constants.pdf
    
    # following from http://pdg.lbl.gov/2019/tables/rpp2019-sum-mesons.pdf
    m_Piplus = 0.13957061
    m_Pizero = 0.1349770
    m_Eta = 0.547862
    m_Etap = 0.95778

    m_Omega = 0.78265
    m_Rho = 0.77526

def w(q2: float, m_parent: float, m_daughter: float) -> float:
    """
    Calculates the recoil variable w, which runs from 1 (zero recoil) to the maximum value (depends on daughter).
    :param q2: Momentum transfer to the lepton-neutrino system.
    :param m_parent: Mass of the parent meson, e.g. the B meson.
    :param m_daughter: Mass of the daughter meson, e.g. the D or D* meson.
    :return:
    """
    return (m_parent ** 2 + m_daughter ** 2 - q2) / (2 * m_parent * m_daughter)


def z_var(w):
    """
    BGL expansion parameter.
    :param w: Recoil variable w.
    :return:
    """
    term1 = np.sqrt(w + 1)
    term2 = np.sqrt(2)
    return (term1 - term2) / (term1 + term2)


def BGL_form_factor(z, p, phi, a: list):
    """
    Calculates the BGL form factor.
    :param z: BGL expansion parameter.
    :param p: The Blaschke factors containing explicit poles in the q2 region.
    :param phi: Outer functions.
    :param a: Expansion coefficients.
    :return:
    """
    return 1 / (p(z) * phi(z)) * sum([a_i * z ** n for n, a_i in enumerate(a)])
