import abc
import numpy as np
from eFFORT.utility import PDG
import scipy.integrate
import uncertainties

from eFFORT.BRhoLepNuRateExp import getDiffRatedq2


class BToPiLNu:

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

    def Gamma(self):
        return scipy.integrate.quad(lambda x: self.dGamma_dq2(x).nominal_value, self.q2min, self.q2max, epsabs=1e-20)

    # def cosTheta(self, q2, El):
    #     return ((self.m_B**2 - self.m_P**2 + q2) * (q2 + self.m_L**2) - (4*q2*self.m_B*El)) / \
    #            (2*self.m_B*self.pion_momentum(q2) * (q2 - self.m_L**2))
    #
    # def dcosTheta_dEl(self, q2):
    #     return 2 * q2 / (self.pion_momentum(q2) * (q2 - self.m_L**2))
    #
    # def ddGamma_dq2dEl(self, q2, El):
    #     # This term already has approximations
    #     return self.N0(q2) * self.pion_momentum(q2) * (2 * (1 - self.cosTheta(q2, El)**2) * self.H0(q2)**2) * self.dcosTheta_dEl(q2)


class BToPiLNuBCL(BToPiLNu):

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToPiLNuBCL, self).__init__(m_B, m_V, m_L, V_ub, eta_EW)
        self._coefficients = None

        # correlation_matrix = np.array([
        #     [1, -0.870, -0.400, 0.453, 0.428, -0.175, -0.201, -0.119, -0.009],
        #     [0, 1, 0.14, -0.455, -0.342, 0.224, 0.174, 0.047, -0.033],
        #     [0, 0, 1, -0.789, -0.874, -0.068, 0.142, 0.025, -0.007],
        #     [0, 0, 0, 1, 0.879, -0.051, -0.253, 0.098, 0.234],
        #     [0, 0, 0, 0, 1, 0.076, 0.038, 0.018, -0.200],
        #     [0, 0, 0, 0, 0, 1, -0.043, -0.604, -0.388],
        #     [0, 0, 0, 0, 0, 0, 1, -0.408, -0.758],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 0.457],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # ])
        # correlation_matrix = correlation_matrix + correlation_matrix.T - np.diag(correlation_matrix.diagonal())
        # (v_ub, fp0, fp1, fp2, fp3, f00, f01, f02, f03) = uncertainties.correlated_values_norm([
        #     (3.72e-3, 0.16e-3),
        #     (0.419, 0.013),
        #     (-0.495, 0.054),
        #     (-0.43, 0.13),
        #     (0.22, 0.31),
        #     (0.510, 0.019),
        #     (-1.700, 0.082),
        #     (1.53, 0.19),
        #     (4.52, 0.83),
        # ], correlation_matrix)
        # self.fplus_par = [fp0, fp1, fp2, fp3]
        # self.fzero_par = [f00, f01, f02, f03]

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients
        self.gamma = None  # Clear cache of normalization integral when updating coefficients

    def fzero(self, q2):
        N = 4
        return sum([b * self.z(q2) ** n for n, b in enumerate(self.coefficients[N:])])

    def fplus(self, q2):
        m_Bstar = 5.325
        N = 4
        return 1 / (1 - q2 / m_Bstar ** 2) * sum(
            [b * (self.z(q2) ** n - (-1) ** (n - N) * n / N * self.z(q2) ** N) for n, b in
             enumerate(self.coefficients[:N])])
