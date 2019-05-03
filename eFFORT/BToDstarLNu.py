import numpy as np
import scipy.integrate
from eFFORT.utility import BGL_form_factor, z_var, PDG
import functools
import abc


class BToDstarLNu:
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

        self.w_min = 1
        self.w_max = (m_B ** 2 + m_Dstar ** 2) / (2 * m_B * m_Dstar)

        # Variables which are often used and can be computed once
        self.r = self.m_Dstar / self.m_B
        self.rprime = 2 * np.sqrt(self.m_B * self.m_Dstar) / (self.m_B + self.m_Dstar)

        self._gamma_int = {
            22: None,
            111: None,
            211: None,
        }

    def A0(self, w):
        raise RuntimeError("Not implemented. But also not required for light leptons.")

    def A1(self, w):
        return (w + 1) / 2 * self.rprime * self.h_A1(w)

    def A2(self, w):
        return self.R2(w) / self.rprime * self.h_A1(w)

    def V(self, w):
        return self.R1(w) / self.rprime * self.h_A1(w)

    def Hplus(self, w):
        return (self.m_B + self.m_Dstar) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_Dstar) * self.m_Dstar * (
                w ** 2 - 1) ** 0.5 * self.V(w)

    def Hminus(self, w):
        return (self.m_B + self.m_Dstar) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_Dstar) * self.m_Dstar * (
                w ** 2 - 1) ** 0.5 * self.V(w)

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

    def dGamma_dw_dcosLepton_dcosNeutrino_dChi(self, w, cos_l, cos_nu, chi, pdg):
        return np.where(
            np.abs(pdg) == 22,
            self.dGamma_dw_dcosLepton_dcosNeutrino_dChi_gamma(w, cos_l, cos_nu, chi),
            self.dGamma_dw_dcosLepton_dcosNeutrino_dChi_pion(w, cos_l, cos_nu, chi)
        )

    def dGamma_dw_dcosLepton_dcosNeutrino_dChi_pion(self, w, cos_l, cos_nu, chi):
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

    def dGamma_dw_dcosLepton_dcosNeutrino_dChi_gamma(self, w, cos_l, cos_nu, chi):
        sin_l = (1 - cos_l ** 2) ** 0.5
        sin_nu = (1 - cos_nu ** 2) ** 0.5

        Hplus = self.Hplus(w)
        Hminus = self.Hminus(w)
        Hzero = self.Hzero(w)

        return 6 * self.m_B * self.m_Dstar ** 2 / 8 / (4 * np.pi) ** 4 * (w ** 2 - 1) ** 0.5 * (
                1 - 2 * w * self.r + self.r ** 2) * self.G_F ** 2 * self.V_cb ** 2 * (
                       (1 - cos_l) ** 2 * (1 + cos_nu ** 2) * Hplus ** 2
                       + (1 + cos_l) ** 2 * (1 + cos_nu ** 2) * Hminus ** 2
                       + 4 * sin_l ** 2 * sin_nu ** 2 * Hzero ** 2
                       - 2 * sin_l ** 2 * (-1) * sin_nu ** 2 * np.cos(2 * chi) * Hplus * Hminus
                       - 4 * sin_l * (1 - cos_l) * (-1) * sin_nu * cos_nu * np.cos(chi) * Hplus * Hzero
                       + 4 * sin_l * (1 + cos_l) * (-1) * sin_nu * cos_nu * np.cos(chi) * Hminus * Hzero
               )

    def dGamma_dw(self, w, pdg):
        return scipy.integrate.nquad(
            lambda cosl, cosnu, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi, pdg),
            [[-1, 1], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dcosLepton(self, cosl, pdg):
        return scipy.integrate.nquad(
            lambda w, cosnu, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi, pdg),
            [[self.w_min, self.w_max], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dcosNeutrino(self, cosnu, pdg):
        return scipy.integrate.nquad(
            lambda w, cosl, chi: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi, pdg),
            [[self.w_min, self.w_max], [-1, 1], [0, 2 * np.pi]]
        )[0]

    def dGamma_dchi(self, chi, pdg):
        return scipy.integrate.nquad(
            lambda w, cosl, cosnu: self.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w, cosl, cosnu, chi, pdg),
            [[self.w_min, self.w_max], [-1, 1], [-1, 1]]
        )[0]

    def Gamma(self, pdg):
        gamma = np.full(len(pdg), np.nan)
        for pdg_code in self._gamma_int.keys():
            gamma[abs(pdg) == pdg_code] = self._gamma_int[pdg_code]
        assert not np.isnan(gamma).any()
        return gamma

    def _Gamma(self, pdg):
        w_min = 1
        w_max = (self.m_B ** 2 + self.m_Dstar ** 2) / (2 * self.m_B * self.m_Dstar)
        return scipy.integrate.nquad(
            self.dGamma_dw_dcosLepton_dcosNeutrino_dChi,
            [[w_min, w_max], [-1, 1], [-1, 1], [0, 2 * np.pi]],
            args=(pdg,)
        )[0]

    def get_gammas(self):
        return self._gamma_int

    @staticmethod
    def check_precomputed_gammas_dict(gammas_dict):
        if not isinstance(gammas_dict, dict):
            raise ValueError(
                f"The parameter cached_gammas must be a dictionary containing precomputed integral values.\n"
                f"The provided cached_gammas was of the type {type(gammas_dict)}."
            )
        if not len(gammas_dict) == 3 or not all(k in gammas_dict.keys() for k in [22, 111, 211]):
            raise KeyError(
                f"The provided cached_gammas dictionary must contain values for the three keys 22, 111 and 211.\n"
                f"It contained the {len(gammas_dict)} keys {list(gammas_dict.keys())}."
            )
        if not all(isinstance(v, float) for v in gammas_dict.values()):
            raise ValueError(f"The provided cached_gamas dictionary must contain floats as values.")


class BToDstarLNuCLN(BToDstarLNu):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066, cached_gammas=None):
        super().__init__(m_B, m_Dstar, V_cb, eta_EW)

        # CLN specifics, default is given by values in https://arxiv.org/abs/1702.01521v2
        self.h_A1_1 = 0.906
        self.rho2 = 1.03
        self.R1_1 = 1.38
        self.R2_1 = 0.97

        if cached_gammas is None:
            self._gamma_int[22] = self._Gamma(22)
            self._gamma_int[111] = self._Gamma(111)
            self._gamma_int[211] = self._gamma_int[111]
        else:
            self.check_precomputed_gammas_dict(cached_gammas)
            self._gamma_int = cached_gammas

    def h_A1(self, w):
        rho2 = self.rho2
        _z = z_var(w)
        return self.h_A1_1 * (1 - 8 * rho2 * _z + (53 * rho2 - 15) * _z ** 2 - (231 * rho2 - 91) * _z ** 3)

    def R1(self, w):
        return self.R1_1 - 0.12 * (w - 1) + 0.05 * (w - 1) ** 2

    def R2(self, w):
        return self.R2_1 + 0.11 * (w - 1) - 0.06 * (w - 1) ** 2


class BToDstarLNuBGL(BToDstarLNu):

    def __init__(self, m_B: float, m_Dstar: float, V_cb: float, eta_EW: float = 1.0066, cached_gammas=None,
                 exp_coeff=(3.79139e-04, 2.69537e-02, 5.49846e-04, -2.04028e-03, -4.32818e-04, 5.35029e-03)):
        super().__init__(m_B, m_Dstar, V_cb, eta_EW)

        # BGL specifics, default is given in arXiv:1703.08170v2
        self.chiT_plus33 = 5.28e-4  # GeV^-2
        self.chiT_minus33 = 3.07e-4  # GeV^-2
        self.n_i = 2.6  # effective number of light quarks
        self.axialvector_poles = [6.730, 6.736, 7.135, 7.142]
        self.vector_poles = [6.337, 6.899, 7.012, 7.280]
        # Coefficients from Florian
        self.eta_ew_Vcb = self.eta_EW * self.V_cb
        self.expansion_coefficients_a = [exp_coeff[0] / self.eta_ew_Vcb, exp_coeff[1] / self.eta_ew_Vcb]  # FF g
        self.expansion_coefficients_b = [exp_coeff[2] / self.eta_ew_Vcb, exp_coeff[3] / self.eta_ew_Vcb]  # FF f
        self.expansion_coefficients_c = [
            ((self.m_B - self.m_Dstar) * self.phi_F1(0) / self.phi_f(0)) * self.expansion_coefficients_b[0],
            exp_coeff[4] / self.eta_ew_Vcb, exp_coeff[5] / self.eta_ew_Vcb]  # FF F1

        assert sum([a ** 2 for a in self.expansion_coefficients_a]) <= 1, "Unitarity bound violated."
        assert sum([b ** 2 + c ** 2 for b, c in zip(self.expansion_coefficients_b,
                                                    self.expansion_coefficients_c)]) <= 1, "Unitarity bound violated."
        if cached_gammas is None:
            self._gamma_int[22] = self._Gamma(22)
            self._gamma_int[111] = self._Gamma(111)
            self._gamma_int[211] = self._gamma_int[111]
        else:
            self.check_precomputed_gammas_dict(cached_gammas)
            self._gamma_int = cached_gammas

    def h_A1(self, w):
        z = z_var(w)
        return self.f(z) / (self.m_B * self.m_Dstar) ** 0.5 / (1 + w)

    def R1(self, w):
        z = z_var(w)
        return (w + 1) * self.m_B * self.m_Dstar * self.g(z) / self.f(z)

    def R2(self, w):
        z = z_var(w)
        return (w - self.r) / (w - 1) - self.F1(z) / self.m_B / (w - 1) / self.f(z)

    def g(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.vector_poles), self.phi_g,
                               self.expansion_coefficients_a)

    def f(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_f,
                               self.expansion_coefficients_b)

    def F1(self, z):
        return BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_F1,
                               self.expansion_coefficients_c)

    def blaschke_factor(self, z, poles):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in poles])

    @functools.lru_cache(2 ** 10)
    def z_p(self, m_pole):
        m_B = self.m_B
        m_D = self.m_Dstar
        term1 = ((m_B + m_D) ** 2 - m_pole ** 2) ** 0.5
        term2 = ((m_B + m_D) ** 2 - (m_B - m_D) ** 2) ** 0.5
        return (term1 - term2) / (term1 + term2)

    def phi_g(self, z):
        r = self.r
        return (256 * self.n_i / 3 / np.pi / self.chiT_plus33) ** 0.5 \
               * r ** 2 * (1 + z) ** 2 * (1 - z) ** -0.5 / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4

    def phi_f(self, z):
        r = self.r
        return 1 / self.m_B ** 2 * (16 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (3. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4

    def phi_F1(self, z):
        r = self.r
        return 1 / self.m_B ** 3 * (8 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (5. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 5


if __name__ == '__main__':
    bToDstar_CLN = BToDstarLNuCLN(PDG.m_Bplus, PDG.m_Dstarzero, 37.4e-3)
    bToDstar_BGL = BToDstarLNuBGL(PDG.m_Bplus, PDG.m_Dstarzero, 41.6558e-3)

    w_min = 1
    w_max = (bToDstar_CLN.m_B ** 2 + bToDstar_CLN.m_Dstar ** 2) / (2 * bToDstar_CLN.m_B * bToDstar_CLN.m_Dstar)

    w_range = np.linspace(w_min + 1e-7, w_max - 1e-7, endpoint=True)
    cosl_range = np.linspace(-1, 1, endpoint=True)
    cosnu_range = np.linspace(-1, 1, endpoint=True)
    chi_range = np.linspace(0, 2 * np.pi, endpoint=True)
    pdg_codes = np.random.choice([22, 111, 211], len(w_range))

    # Example call with numpy arrays
    print(bToDstar_BGL.dGamma_dw_dcosLepton_dcosNeutrino_dChi(w_range, cosl_range, cosnu_range, chi_range, pdg_codes))

    # print("CLN total rate: {}".format(bToDstar_CLN.Gamma()))
    # print("BGL total rate: {}".format(bToDstar_BGL.Gamma()))
