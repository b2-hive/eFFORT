import abc

import numpy as np
import scipy.integrate
import uncertainties

from eFFORT.utility import PDG


class BToPLNu:

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066) -> None:
        self.m_B = m_B
        self.m_P = m_P
        self.m_L = m_L
        self._V_ub = V_ub
        self.eta_EW = eta_EW

        # self.N0 = PDG.G_F ** 2 / (192. * np.pi ** 3 * self.m_B ** 3)  # Save some computing time.

        self.tplus = (self.m_B + self.m_P) ** 2
        self.tminus = (self.m_B - self.m_P) ** 2
        self.tzero = self.tplus * (1 - (1 - self.tminus / self.tplus) ** 0.5)

        self.q2min = self.m_L ** 2
        self.q2min += 1e-3  # numerical stability
        self.q2max = self.m_B ** 2 + self.m_P ** 2 - 2 * self.m_B * self.m_P
        self.q2max -= 1e-3  # numerical stability

        self.gamma = None

    @property
    def V_ub(self):
        return self._V_ub

    @V_ub.setter
    def V_ub(self, V_ub):
        self._V_ub = V_ub
        self.gamma = None  # Clear cache of normalization integral when updating Vub

    def pion_momentum(self, q2):
        return np.sqrt(
            ((self.m_B ** 2 + self.m_P ** 2 - q2) / (2 * self.m_B)) ** 2 - self.m_P ** 2
        )

    def z(self, q2):
        return (np.sqrt(self.tplus - q2) - np.sqrt(self.tplus - self.tzero)) / \
               (np.sqrt(self.tplus - q2) + np.sqrt(self.tplus - self.tzero))

    def N0(self, q2):
        return (PDG.G_F ** 2 * self._V_ub ** 2 * q2) / (256 * np.pi ** 3 * self.m_B ** 2)

    @abc.abstractmethod
    def fzero(self, q2):
        return 0

    @abc.abstractmethod
    def fplus(self, q2):
        return 0

    def H0(self, q2):
        return 2 * self.m_B * self.pion_momentum(q2) / np.sqrt(q2) * self.fplus(q2)

    def Ht(self, q2):
        return (self.m_B ** 2 - self.m_P ** 2) / np.sqrt(q2) * self.fzero(q2)

    def dGamma_dq2(self, q2):
        return (8. / 3. * self.N0(q2) * self.pion_momentum(q2)) * (1 - self.m_L ** 2 / q2) ** 2 * \
               (self.H0(q2) ** 2 * (1 + self.m_L ** 2 / 2. / q2) + 3. / 2. * self.m_L ** 2 / q2 * self.Ht(q2) ** 2)

    def dGamma_dw(self, w):
        # For easier variable handling in the equations
        m_B = self.m_B
        m_P = self.m_P

        return PDG.G_F**2 * m_P**3 / 48 / np.pi**3 * (m_B + m_P) *2 * (w**2 - 1)**(3/2) * self.eta_EW ** 2 * self.V_ub ** 2 * self.fplus**2

    def Gamma(self):
        return scipy.integrate.quad(lambda x: uncertainties.nominal_value(self.dGamma_dq2(x)),
                                    self.q2min, self.q2max, epsabs=1e-20)[0]


class BToPLNuBCL(BToPLNu):

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToPLNuBCL, self).__init__(m_B, m_P, m_L, V_ub, eta_EW)
        self._coefficients = None
        self.mBstar = 5.325

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients
        self.gamma = None  # Clear cache of normalization integral when updating coefficients

    def fzero(self, q2):
        N = 4
        return sum([b * self.z(q2) ** n for n, b in enumerate(self._coefficients[N:])])

    def fplus(self, q2):
        N = 4
        return 1 / (1 - q2 / self.mBstar ** 2) * sum(
            [b * (self.z(q2) ** n - (-1) ** (n - N) * n / N * self.z(q2) ** N) for n, b in
             enumerate(self._coefficients[:N])]
        )


class BToPLNuEvtGenBelle(BToPLNu):

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToPLNuEvtGenBelle, self).__init__(m_B, m_P, m_L, V_ub, eta_EW)
        self.parameters = [
            0.261, -2.03, 1.293,  # fplus
            0.261, -0.27, -0.752,  # fzero
        ]

    def fplus(self, q2):
        pars = self.parameters[0:3]
        return pars[0] / (1 + pars[1] * (q2 / self.m_B ** 2) + pars[2] * (q2 / self.m_B ** 2) ** 2)


class BToEtaLNuLCSR_BZ(BToPLNu):

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToEtaLNuLCSR_BZ, self).__init__(m_B, m_P, m_L, V_ub, eta_EW)
        self.parameters = [
        # Ball-Zwicky calculation 2007  JHEP. 0708:025
            0.231, # fzero
            0.851, # alpha
            0.411,  # r
            5.33   # mB*
       
        ]

    def fplus(self, q2):
        pars = self.parameters[0:4]
        return pars[0] * ( 1./(1.- (q2/pars[3]**2)) + (pars[2] * q2 / pars[3]** 2)/((1.- q2/pars[3] ** 2)*(1.- (pars[1] *q2)/self.m_B ** 2)))
    
    
    
class BToEtaLNuLCSR_DM(BToPLNu):

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToEtaLNuLCSR_DM, self).__init__(m_B, m_P, m_L, V_ub, eta_EW)
        self.parameters = [
        # G. Duplancic, B. Melic calculation 2015 https://arxiv.org/abs/1508.05287  JHEP 1511 (2015) 138
            0.168, # fzero
            0.462, # alpha
            5.3252   # mB*
       
        ]

    def fplus(self, q2):
        pars = self.parameters[0:3]
        return pars[0] / ((1 - q2/pars[2] **2)*(1- pars[1] * q2 /pars[2] ** 2))
