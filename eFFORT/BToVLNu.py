import abc
import numpy as np
from eFFORT.utility import PDG
import scipy.integrate


class BToVLNu:
    """
    A class containing functions specific to the differential decay rate of the B to Dstar transitions with the BCL/BGL
    parametrization. If not states otherwise, the numerical values and variable/function definitions are taken from:
    https://arxiv.org/abs/1702.01521v2 and ...

    Zero mass approximation for the lepton is implicit.
    The Blaschke factors do not explicitly appear, because they are 1.
    """

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066) -> None:
        self.m_B = m_B
        self.m_V = m_V
        self.m_L = m_L
        self.V_ub = V_ub
        self.eta_EW = eta_EW

        self.N0 = PDG.G_F ** 2 / (192. * np.pi ** 3 * self.m_B ** 3)  # Save some computing time.

        self.tplus = (self.m_B + self.m_V) ** 2
        self.tminus = (self.m_B - self.m_V) ** 2
        self.tzero = self.tplus * (1 - (1 - self.tminus / self.tplus)**0.5)

        self.q2min = self.m_L ** 2
        self.q2min += 1e-3  # numerical stability
        self.q2max = self.m_B ** 2 + self.m_V ** 2 - 2 * self.m_B * self.m_V
        self.q2max -= 1e-3  # numerical stability

        self.gamma = None

    def kaellen(self, q2):
        return ((self.m_B + self.m_V) ** 2 - q2) * ((self.m_B - self.m_V) ** 2 - q2)

    @abc.abstractmethod
    def A0(self, q2):
        raise RuntimeError("Not implemented. But also not required for light leptons.")

    @abc.abstractmethod
    def A1(self, q2):
        return 0

    @abc.abstractmethod
    def A12(self, q2):
        return 0

    @abc.abstractmethod
    def V(self, q2):
        return 0

    @staticmethod
    def blaschke_pole(q2, m_pole):
        return (1 - q2 / m_pole ** 2) ** -1

    def z(self, q2):
        return ((self.tplus - q2) ** 0.5 - (self.tplus - self.tzero) ** 0.5) / (
                (self.tplus - q2) ** 0.5 + (self.tplus - self.tzero) ** 0.5)

    def Hplus(self, q2):
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_V) + (self.m_B + self.m_V) * self.A1(q2)

    def Hminus(self, q2):
        return self.kaellen(q2)**0.5 * self.V(q2) / (self.m_B + self.m_V) - (self.m_B + self.m_V) * self.A1(q2)

    def Hzero(self, q2):
        return 8 * self.m_B * self.m_V / q2 ** 0.5 * self.A12(q2)

    def dGamma_dq2(self, q2):
        return self.N0 * self.V_ub ** 2 * self.kaellen(q2)**0.5 * q2 * (
                self.Hplus(q2) ** 2 + self.Hminus(q2) ** 2 + self.Hzero(q2) ** 2)

    def deltaGamma_deltaq2(self, lower, upper):
        return scipy.integrate.quad(self.dGamma_dq2, lower, upper)[0]

    def Gamma(self):
        if self.gamma is None:
            self.gamma = self.deltaGamma_deltaq2(self.q2min, self.q2max)
        return self.gamma


class BToVLNuBCL(BToVLNu):

    def __init__(self, m_B: float, m_V: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToVLNuBCL, self).__init__(m_B, m_V, m_L, V_ub, eta_EW)
        self.coefficients = None

    def set_bcl_coefficients(self, coefficients):
        self.coefficients = coefficients
        self.gamma = None  # Clear cache of normalization integral when updating coefficients

    def form_factor(self, q2, m_pole, coefficients):
        return BToVLNu.blaschke_pole(q2, m_pole) * sum(
            [par * (self.z(q2) - self.z(0)) ** k for k, par in enumerate(coefficients)])

    def A1(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[2], self.coefficients[3], self.coefficients[4]]

        return self.form_factor(q2, m_pole, coefficients)

    def A12(self, q2):
        m_pole = 5.724
        coefficients = [self.coefficients[5], self.coefficients[6], self.coefficients[7]]

        return self.form_factor(q2, m_pole, coefficients)

    def V(self, q2):
        m_pole = 5.325
        coefficients = [self.coefficients[8], self.coefficients[9], self.coefficients[10]]

        return self.form_factor(q2, m_pole, coefficients)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from uncertainties import correlated_values
    from tabulate import tabulate
    from eFFORT.plotting import init_thesis_plot_style, plot_with_errorband
    init_thesis_plot_style()

    coefficient_labels = [
        r'$\alpha_1^{A_0}$', r'$\alpha_2^{A_0}$',
        r'$\alpha_0^{A_1}$', r'$\alpha_1^{A_1}$', r'$\alpha_2^{A_1}$',
        r'$\alpha_0^{A_{12}}$', r'$\alpha_1^{A_{12}}$', r'$\alpha_2^{A_{12}}$',
        r'$\alpha_0^{V}$', r'$\alpha_1^{V}$', r'$\alpha_2^{V}$',
        r'$\alpha_1^{T_1}$', r'$\alpha_2^{T_1}$',
        r'$\alpha_0^{T_2}$', r'$\alpha_1^{T_2}$', r'$\alpha_2^{T_2}$',
        r'$\alpha_0^{T_{23}}$', r'$\alpha_1^{T_{23}}$', r'$\alpha_2^{T_{23}}$',
    ]

    lcsr_Brho = np.load('inputs/Brho_LCSR_central.npy')
    lcsr_Brho_cov = np.load('inputs/Brho_LCSR_covariance.npy')
    lcsr_Bomega = np.load('inputs/Bomega_LCSR_central.npy')
    lcsr_Bomega_cov = np.load('inputs/Bomega_LCSR_covariance.npy')

    plt.imshow(lcsr_Brho_cov / np.outer(lcsr_Brho_cov.diagonal()**0.5, lcsr_Brho_cov.diagonal()**0.5),
               vmin=-1, vmax=1, cmap='seismic')
    plt.xticks(range(len(coefficient_labels)), coefficient_labels, rotation='vertical', fontsize=10)
    plt.yticks(range(len(coefficient_labels)), coefficient_labels, rotation='horizontal', fontsize=10)
    plt.colorbar()
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.savefig('lcsr_Brho_correlation.pdf')
    plt.show()
    plt.close()

    with open('lcsr_Brho_correlation.tex', 'w') as f:
        f.write(tabulate(lcsr_Brho_cov / np.outer(lcsr_Brho_cov.diagonal()**0.5, lcsr_Brho_cov.diagonal()**0.5),
                tablefmt='latex_raw', floatfmt='.2f'))

    plt.imshow(lcsr_Bomega_cov / np.outer(lcsr_Bomega_cov.diagonal()**0.5, lcsr_Bomega_cov.diagonal()**0.5),
               vmin=-1, vmax=1, cmap='seismic')
    plt.xticks(range(len(coefficient_labels)), coefficient_labels, rotation='vertical', fontsize=10)
    plt.yticks(range(len(coefficient_labels)), coefficient_labels, rotation='horizontal', fontsize=10)
    plt.colorbar()
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.savefig('lcsr_Bomega_correlation.pdf')
    plt.show()
    plt.close()

    with open('lcsr_Bomega_correlation.tex', 'w') as f:
        f.write(tabulate(lcsr_Bomega_cov / np.outer(lcsr_Bomega_cov.diagonal()**0.5, lcsr_Bomega_cov.diagonal()**0.5),
                tablefmt='latex_raw', floatfmt='.2f'))

    lcsr_Brho_correlated = correlated_values(lcsr_Brho, lcsr_Brho_cov)
    lcsr_Bomega_correlated = correlated_values(lcsr_Bomega, lcsr_Bomega_cov)

    bcl_rho = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.775, m_L=0, V_ub=3.72e-3)
    bcl_rho.set_bcl_coefficients(lcsr_Brho_correlated)

    bcl_omega = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.782, m_L=0, V_ub=3.72e-3)
    bcl_omega.set_bcl_coefficients(lcsr_Bomega_correlated)

    q2range = np.linspace(bcl_rho.q2min, bcl_rho.q2max)

    plot_with_errorband(q2range, bcl_rho.A1(q2range), label=r'$A_{1}$', ls='-')
    plot_with_errorband(q2range, bcl_rho.A12(q2range), label=r'$A_{12}$', ls='--')
    plot_with_errorband(q2range, bcl_rho.V(q2range), label=r'$V$', ls=':')
    plt.ylim(0, 2.3)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$F_i(q^2)$')
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Brho_FF.pdf')
    plt.show()
    plt.close()

    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.A1(q2range)], label=r'$A_{1}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.A12(q2range)], label=r'$A_{12}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_rho.V(q2range)], label=r'$V$', ls='-')
    plt.ylim(0, 0.25)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$\sigma F_i(q^2)$ / $F_i(q^2)$')
    plt.title(r'$B\rightarrow \rho l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Brho_FF_relative_error.pdf')
    plt.show()
    plt.close()

    plot_with_errorband(q2range, bcl_omega.A1(q2range), label=r'$A_{1}$', ls='-')
    plot_with_errorband(q2range, bcl_omega.A12(q2range), label=r'$A_{12}$', ls='--')
    plot_with_errorband(q2range, bcl_omega.V(q2range), label=r'$V$', ls=':')
    plt.ylim(0, 2.3)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$F_i(q^2)$')
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Bomega_FF.pdf')
    plt.show()
    plt.close()

    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.A1(q2range)], label=r'$A_{1}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.A12(q2range)], label=r'$A_{12}$', ls='-')
    plt.plot(q2range, [y.std_dev / y.nominal_value for y in bcl_omega.V(q2range)], label=r'$V$', ls='-')
    plt.ylim(0, 0.25)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$\sigma F_i(q^2)$ / $F_i(q^2)$')
    plt.title(r'$B\rightarrow \omega (\rightarrow 3\pi) l \nu$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_Bomega_FF_relative_error.pdf')
    plt.show()
    plt.close()



    bcl_rho = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.775, m_L=0, V_ub=3.72e-3)
    bcl_rho.set_bcl_coefficients(lcsr_Brho)

    bcl_omega = BToVLNuBCL(m_B=PDG.m_Bzero, m_V=0.782, m_L=0, V_ub=3.72e-3)
    bcl_omega.set_bcl_coefficients(lcsr_Bomega)

    # tauBzero = 1.520e-12 * 1. / 6.582119e-16 / 1e-9
    # tauBplus = 1.638e-12 * 1. / 6.582119e-16 / 1e-9
    #
    # plt.plot(q2range, tauBzero * bcl_rho.dGamma_dq2(q2range) * 1e6, label='B0 rho')
    # plt.plot(q2range, tauBplus / 2 * bcl_rho.dGamma_dq2(q2range) * 1e6, label='B+ rho', ls='--')
    #
    # plt.plot(q2range, tauBzero * bcl_omega.dGamma_dq2(q2range) * 1e6, label='B0 omega', ls=':')
    # plt.plot(q2range, tauBplus / 2 * bcl_omega.dGamma_dq2(q2range) * 1e6, label='B+ omega', ls='-.')
    # plt.ylim(0, None)
    # plt.xlabel(r'$q^2$ / (GeV$^2$)')
    # plt.ylabel(r'$\mathrm{d}\mathcal{B} / \mathrm{d}q^2$')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    plt.plot(q2range, bcl_rho.dGamma_dq2(q2range) / bcl_rho.Gamma(),
             label=r'$B \rightarrow \rho l \nu$')
    plt.plot(q2range, bcl_omega.dGamma_dq2(q2range) / bcl_omega.Gamma(),
             label=r'$B \rightarrow \omega(\rightarrow 3\pi) l \nu$', ls='--')
    plt.ylim(0, None)
    plt.xlabel(r'$q^2$ / (GeV$^2$)')
    plt.ylabel(r'$1/ \Gamma$ $\mathrm{d}\Gamma / \mathrm{d}q^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lcsr_normalized_rates.pdf')
    plt.show()
    plt.close()

    with open('lcsr_central_values.tex', 'w') as f:
        f.write(tabulate(zip(coefficient_labels,
                             lcsr_Brho, lcsr_Brho_cov.diagonal()**0.5,
                             lcsr_Bomega, lcsr_Bomega_cov.diagonal()**0.5),
                         [r'',
                          r'$B\rightarrow\rho$', r'$\sigma(B\rightarrow\rho)$',
                          r'$B\rightarrow\omega$', r'$\sigma(B\rightarrow\omega)$'],
                         tablefmt='latex_raw', floatfmt='.2f'))
