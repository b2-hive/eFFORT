import numpy as np
import scipy.integrate
from eFFORT.utility import BGL_form_factor, z_var, PDG
import functools
import operator
import abc


class BToDstar:
    """
    A class containing functions specific to the differential decay rate of the B to Dstar transitions with the BCL/BGL
    parametrization. If not states otherwise, the numerical values and variable/function definitions are taken from:
    https://arxiv.org/abs/1702.01521v2 and ...

    Zero mass approximation for the lepton is implicit.
    The Blaschke factors do not explicitly appear, because they are 1.
    """

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066) -> None:
        # Some of the following can be inherited from a parent class / initialized from a super constructor in the
        # future.
        self.m_B = m_B
        self.m_Dstar = m_Dstar
        self.V_cb = V_cb
        self.eta_EW = eta_EW
        self.G_F = PDG.G_F

        # Variables which are often used and can be computed once
        self.r = self.m_Dstar / self.m_B
        self.rprime = 2 * np.sqrt(self.m_B * self.m_Dstar) / (self.m_B + self.m_Dstar)

    @functools.lru_cache(maxsize=2 ** 10)
    def A0(self, w):
        raise RuntimeError("Not implemented. But also not required for light leptons.")

    @functools.lru_cache(maxsize=2 ** 10)
    def A1(self, w):
        return (w + 1) / 2 * self.rprime * self.h_A1(w)

    @functools.lru_cache(maxsize=2 ** 10)
    def A2(self, w):
        return self.R2(w) / self.rprime * self.h_A1(w)

    @functools.lru_cache(maxsize=2 ** 10)
    def V(self, w):
        return self.R1(w) / self.rprime * self.h_A1(w)

    @functools.lru_cache(maxsize=2 ** 10)
    def Hplus(self, w):
        return (self.m_B + self.m_Dstar) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_Dstar) * self.m_Dstar * (
                w ** 2 - 1) ** 0.5 * self.V(w)

    @functools.lru_cache(maxsize=2 ** 10)
    def Hminus(self, w):
        return (self.m_B + self.m_Dstar) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_Dstar) * self.m_Dstar * (
                w ** 2 - 1) ** 0.5 * self.V(w)

    @functools.lru_cache(maxsize=2 ** 10)
    def Hzero(self, w):
        m_B = self.m_B
        m_D = self.m_Dstar
        q2 = (m_B ** 2 + m_D ** 2 - 2 * w * m_B * m_D)
        return 1 / (2 * m_D * q2 ** 0.5) * ((m_B ** 2 - m_D ** 2 - q2) * (m_B + m_D) * self.A1(w)
                                            - 4 * m_B ** 2 * m_D ** 2 * (w ** 2 - 1) / (m_B + m_D) * self.A2(w))

    @abc.abstractmethod
    def h_A1(self, w: float) -> float:
        pass

    @abc.abstractmethod
    def R0(self, w: float) -> float:
        raise RuntimeError("Not implemented. But also not required for light leptons.")

    @abc.abstractmethod
    def R1(self, w: float) -> float:
        pass

    @abc.abstractmethod
    def R2(self, w: float) -> float:
        pass

    def dGamma_dw_dcosLepton_dcosNeutrino_dChi(self, w, cos_l, cos_nu, chi):
        sin_l = (1 - cos_l ** 2) ** 0.5
        sin_nu = (1 - cos_nu ** 2) ** 0.5

        Hplus = self.Hplus(w)
        Hminus = self.Hminus(w)
        Hzero = self.Hzero(w)

        return 6 * self.m_B * self.m_Dstar ** 2 / 8 / (4 * np.pi) ** 4 * (w ** 2 - 1) ** 0.5 * (
                1 - 2 * w * self.r + self.r ** 2) * self.G_F ** 2 * self.V_cb ** 2 * (
                       (1 - cos_l) ** 2 * sin_nu ** 2 * Hplus ** 2
                       + (1 + cos_l) ** 2 * sin_nu ** 2 * Hminus ** 2
                       + 4 * sin_l ** 2 * cos_nu ** 2 * Hzero ** 2
                       - 2 * sin_l ** 2 * sin_nu ** 2 * np.cos(2 * chi) * Hplus * Hminus
                       - 4 * sin_l * (1 - cos_l) * sin_nu * cos_nu * np.cos(chi) * Hplus * Hzero
                       + 4 * sin_l * (1 + cos_l) * sin_nu * cos_nu * np.cos(chi) * Hminus * Hzero
               )

    def dGamma_dw(self, w):
        return scipy.integrate.nquad(
            lambda cosl, cosnu, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi),
            [[-1, 1], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dcosLepton(self, cosl):
        return scipy.integrate.nquad(
            lambda w, cosnu, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi),
            [[1, w_max], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dcosNeutrino(self, cosnu):
        return scipy.integrate.nquad(
            lambda w, cosl, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi),
            [[1, w_max], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dchi(self, chi):
        return scipy.integrate.nquad(
            lambda w, cosl, cosnu: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi),
            [[1, w_max], [-1, 1], [-1, 1]]
        )[0]

    @functools.lru_cache(maxsize=1)
    def Gamma(self):
        w_min = 1
        w_max = (self.m_B ** 2 + self.m_Dstar ** 2) / (2 * self.m_B * self.m_Dstar)
        return scipy.integrate.nquad(
            self.dGamma_dw_dcosLepton_dcosNeutrino_dChi,
            [[w_min, w_max], [-1, 1], [-1, 1], [0, 2 * np.pi]]
        )[0]


class BToDstarCLN(BToDstar):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066):
        super(BToDstarCLN, self).__init__(m_B, m_Dstar, V_cb, eta_EW)

        # CLN specifics, default is given by values in https://arxiv.org/abs/1702.01521v2
        self.h_A1_1 = 0.906
        self.rho2 = 1.03
        self.R1_1 = 1.38
        self.R2_1 = 0.97

    @functools.lru_cache(maxsize=2 ** 10)
    def h_A1(self, w):
        rho2 = self.rho2
        _z = z_var(w)
        return self.h_A1_1 * (1 - 8 * rho2 * _z + (53 * rho2 - 15) * _z ** 2 - (231 * rho2 - 91) * _z ** 3)

    @functools.lru_cache(maxsize=2 ** 10)
    def R1(self, w):
        return self.R1_1 - 0.12 * (w - 1) + 0.05 * (w - 1) ** 2

    @functools.lru_cache(maxsize=2 ** 10)
    def R2(self, w):
        return self.R2_1 + 0.11 * (w - 1) - 0.06 * (w - 1) ** 2


class BToDstarBGL(BToDstar):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066):
        super(BToDstarBGL, self).__init__(m_B, m_Dstar, V_cb, eta_EW)

        # BGL specifics, default is given in arXiv:1703.08170v2
        self.chiT_plus33 = 5.28e-4  # GeV^-2
        self.chiT_minus33 = 3.07e-4  # GeV^-2
        self.n_i = 2.6  # effective number of light quarks
        self.axialvector_poles = [6.730, 6.736, 7.135, 7.142]
        self.vector_poles = [6.337, 6.899, 7.012, 7.280]
        # Coefficients from Florian
        self.eta_ew_Vcb = self.eta_EW * self.V_cb
        self.expansion_coefficients_a = [3.79139e-04 / self.eta_ew_Vcb, 2.69537e-02 / self.eta_ew_Vcb]  # FF g
        self.expansion_coefficients_b = [5.49846e-04 / self.eta_ew_Vcb, -2.04028e-03 / self.eta_ew_Vcb]  # FF f
        self.expansion_coefficients_c = [
            ((self.m_B - self.m_Dstar) * self.phi_F1(0) / self.phi_f(0)) * self.expansion_coefficients_b[0],
            -4.32818e-04 / self.eta_ew_Vcb, 5.35029e-03 / self.eta_ew_Vcb]  # FF F1

        assert sum([a ** 2 for a in self.expansion_coefficients_a]) <= 1, "Unitarity bound violated."
        assert sum([b ** 2 + c ** 2 for b, c in zip(self.expansion_coefficients_b,
                                                    self.expansion_coefficients_c)]) <= 1, "Unitarity bound violated."

    @functools.lru_cache(maxsize=2 ** 10)
    def h_A1(self, w):
        z = z_var(w)
        return self.f(z) / (self.m_B * self.m_Dstar) ** 0.5 / (1 + w)

    @functools.lru_cache(maxsize=2 ** 10)
    def R1(self, w):
        z = z_var(w)
        return (w + 1) * self.m_B * self.m_Dstar * self.g(z) / self.f(z)

    @functools.lru_cache(maxsize=2 ** 10)
    def R2(self, w):
        z = z_var(w)
        return (w - self.r) / (w - 1) - self.F1(z) / self.m_B / (w - 1) / self.f(z)

    @functools.lru_cache(maxsize=2 ** 10)
    def g(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.vector_poles), self.phi_g,
                               self.expansion_coefficients_a)

    @functools.lru_cache(maxsize=2 ** 10)
    def f(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_f,
                               self.expansion_coefficients_b)

    @functools.lru_cache(maxsize=2 ** 10)
    def F1(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_F1,
                               self.expansion_coefficients_c)

    def blaschke_factor(self, z, poles):
        return functools.reduce(operator.mul, [(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in poles])

    def z_p(self, m_pole):
        m_B = self.m_B
        m_D = self.m_Dstar
        term1 = ((m_B + m_D) ** 2 - m_pole ** 2) ** 0.5
        term2 = ((m_B + m_D) ** 2 - (m_B - m_D) ** 2) ** 0.5
        return (term1 - term2) / (term1 + term2)

    @functools.lru_cache(maxsize=2 ** 10)
    def phi_g(self, z):
        r = self.r
        return (256 * self.n_i / 3 / np.pi / self.chiT_plus33) ** 0.5 \
               * r ** 2 * (1 + z) ** 2 * (1 - z) ** -0.5 / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4

    @functools.lru_cache(maxsize=2 ** 10)
    def phi_f(self, z):
        r = self.r
        return 1 / self.m_B ** 2 * (16 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (3. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4

    @functools.lru_cache(maxsize=2 ** 10)
    def phi_F1(self, z):
        r = self.r
        return 1 / self.m_B ** 3 * (8 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (5. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 5


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from eFFORT.plotting import Tango, init_thesis_plot_style

    init_thesis_plot_style()

    bToDstar_CLN = BToDstarCLN(PDG.m_Bplus, PDG.m_Dstarzero, 37.4e-3)
    bToDstar_BGL = BToDstarBGL(PDG.m_Bplus, PDG.m_Dstarzero, 41.6558e-3)

    w_min = 1
    w_max = (bToDstar_CLN.m_B ** 2 + bToDstar_CLN.m_Dstar ** 2) / (2 * bToDstar_CLN.m_B * bToDstar_CLN.m_Dstar)

    w_range = np.linspace(w_min + 1e-7, w_max - 1e-7, endpoint=True)
    cosl_range = np.linspace(-1, 1, endpoint=True)
    cosnu_range = np.linspace(-1, 1, endpoint=True)
    chi_range = np.linspace(0, 2 * np.pi, endpoint=True)

    plt.plot(w_range, [bToDstar_CLN.dGamma_dw(x) * 1e15 for x in w_range],
             color=Tango.slate, ls='solid', lw=2, label='CLN arXiv:1702.01521v2')
    plt.plot(w_range, [bToDstar_BGL.dGamma_dw(x) * 1e15 for x in w_range],
             color=Tango.orange, ls='dotted', lw=2, label='BGL arXiv:1703.08170v2')
    plt.xlabel(r'$w$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}w \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(w_min, w_max)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dw.png')
    plt.show()
    plt.close()

    plt.plot(chi_range, [bToDstar_CLN.dGamma_dchi(x) * 1e15 for x in chi_range],
             color=Tango.slate, ls='solid', lw=2, label='CLN arXiv:1702.01521v2')
    plt.plot(chi_range, [bToDstar_BGL.dGamma_dchi(x) * 1e15 for x in chi_range],
             color=Tango.orange, ls='dotted', lw=2, label='BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\chi \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 6)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dchi.png')
    plt.show()
    plt.close()

    plt.plot(cosl_range, [bToDstar_CLN.dGamma_dcosLepton(x) * 1e15 for x in cosl_range],
             color=Tango.slate, ls='solid', lw=2, label='CLN arXiv:1702.01521v2')
    plt.plot(cosl_range, [bToDstar_BGL.dGamma_dcosLepton(x) * 1e15 for x in cosl_range],
             color=Tango.orange, ls='dotted', lw=2, label='BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\cos\theta_l$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\cos\theta_l \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(-1, 1)
    plt.ylim(0, 23)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dcosl.png')
    plt.show()
    plt.close()

    plt.plot(cosnu_range, [bToDstar_CLN.dGamma_dcosNeutrino(x) * 1e15 for x in cosnu_range],
             color=Tango.slate, ls='solid', lw=2, label='CLN arXiv:1702.01521v2')
    plt.plot(cosnu_range, [bToDstar_BGL.dGamma_dcosNeutrino(x) * 1e15 for x in cosnu_range],
             color=Tango.orange, ls='dotted', lw=2, label='BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\cos\theta_\nu$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\cos\theta_\nu \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(-1, 1)
    plt.ylim(0, 21)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dcosnu.png')
    plt.show()
    plt.close()

    print("CLN total rate: {}".format(bToDstar_CLN.Gamma()))
    print("BGL total rate: {}".format(bToDstar_BGL.Gamma()))
