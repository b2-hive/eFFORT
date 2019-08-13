import numpy as np
from eFFORT.utility import BGL_form_factor, z_var, PDG
import abc
import scipy.integrate


class BToDLNu(abc.ABC):
    """
    A class containing functions specific to the differential decay rate of the B to D transitions with the BCL/BGL
    parametrization. If not states otherwise, the numerical values and variable/function definitions are taken from:
    https://arxiv.org/abs/1510.03657

    Zero mass approximation for the lepton is implicit.
    The Blaschke factors do not explicitly appear, because they are 1.
    """

    def __init__(self, m_B: float, m_D: float, V_cb: float, eta_EW: float = 1.0066) -> None:
        # Some of the following can be inherited from a parent class / initialized from a super constructor in the
        # future.
        self.m_B = m_B
        self.m_D = m_D
        self.V_cb = V_cb
        self.eta_EW = eta_EW
        self.G_F = PDG.G_F

        # Variables which are often used and can be computed once
        self.r = self.m_D / self.m_B

        self._gamma_int = self._Gamma()

    @abc.abstractmethod
    def G(self, w: float) -> float:
        pass

    def dGamma_dw(self, w):
        # For easier variable handling in the equations
        m_B = self.m_B
        m_D = self.m_D

        return self.G_F ** 2 * m_D ** 3 / 48 / np.pi ** 3 * (m_B + m_D) ** 2 * (w ** 2 - 1) ** (
                3 / 2) * self.eta_EW ** 2 * self.V_cb ** 2 * self.G(w)

    def _Gamma(self):
        w_min = 1
        w_max = (self.m_B ** 2 + self.m_D ** 2) / (2 * self.m_B * self.m_D)
        return scipy.integrate.quad(self.dGamma_dw, w_min, w_max)[0]

    def Gamma(self):
        return self._gamma_int


class BToDLNuCLN(BToDLNu):
    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066, cln_g1=None, cln_rho2=None):
        # CLN specifics, default is given by the Belle Evtgen parameters
        self.G_1 = 1.074 if cln_g1 is None else cln_g1
        self.rho2 = 1.15 if cln_rho2 is None else cln_rho2

        super(BToDLNuCLN, self).__init__(m_B, m_Dstar, V_cb, eta_EW)

    def G(self, w: float) -> float:
        """
        Squared Caprini, Lellouch and Neubert (CLN) form factor.
        :param w:
        :return:
        """
        rho2 = self.rho2
        z = z_var(w)
        return (self.G_1 * (1 - 8 * rho2 * z + (51 * rho2 - 10) * z ** 2 - (252 * rho2 - 84) * z ** 3)) ** 2


class BToDLNuBelle(BToDLNu):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066, cln_g1=None, cln_rho2=None):
        super(BToDLNuBelle, self).__init__(m_B, m_Dstar, V_cb, eta_EW)

        # CLN specifics, default is given by the Belle Evtgen parameters
        self.G_1 = 1.074 if cln_g1 is None else cln_g1
        self.rho2 = 1.15 if cln_rho2 is None else cln_rho2

    def G(self, w: float) -> float:
        """
        Squared Caprini, Lellouch and Neubert (CLN) form factor.
        :param z:
        :return:
        """
        rho2 = self.rho2
        z = z_var(w)
        return (self.G_1 * (1 - 8 * rho2 * z + (51 * rho2 - 10) * z ** 2 - (252 * rho2 - 84) * z ** 3)) ** 2


class BToDLNuBGL(BToDLNu):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066, bgl_fplus_coefficients=None):
        # BGL specifics
        self.expansion_coefficients = [] if bgl_fplus_coefficients is None else bgl_fplus_coefficients

        super(BToDLNuBGL, self).__init__(m_B, m_Dstar, V_cb, eta_EW)

    def phi_plus(self, z: float) -> float:
        """
        Outer function for the f+ form factor.
        :param z: BGL expansion parameter.
        :return:
        """
        r = self.r
        return 1.1213 * (1 + z) ** 2 * (1 - z) ** (1 / 2) * ((1 + r) * (1 - z) + 2 * np.sqrt(r) * (1 + z)) ** -5

    def G(self, w: float) -> float:
        """
        Squared Boyd, Grinstein and Lebed (BGL) form factor.
        :param w: Recoil variable w.
        :return:
        """
        return 4 * self.r / (1 + self.r) ** 2 * BGL_form_factor(z_var(w), lambda x: 1, self.phi_plus,
                                                                self.expansion_coefficients) ** 2

    def fplus(self, w: float) -> float:
        return BGL_form_factor(z_var(w), lambda x: 1, self.phi_plus, self.expansion_coefficients) ** 2


if __name__ == '__main__':
    bToD_glattauer_bgl = BToDLNuBGL(PDG.m_Bplus, PDG.m_Dzero, V_cb=40.83e-3,
                                    bgl_fplus_coefficients=[0.0126, -0.094, 0.34, -0.1])
    bToD_glattauer_cln = BToDLNuCLN(PDG.m_Bplus, PDG.m_Dzero, V_cb=39.86e-3, cln_g1=1.0541, cln_rho2=1.09)

    print("CLN total rate: {}".format(bToD_glattauer_cln.Gamma()))
    print("BGL total rate: {}".format(bToD_glattauer_bgl.Gamma()))
