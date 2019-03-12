from eFFORT.utility import PDG
from eFFORT.BToDLNu import BToDLNuCLN, BToDLNuBGL
from eFFORT.BToDstarLNu import BToDstarLNuCLN, BToDstarLNuBGL
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from eFFORT.plotting import Tango, init_thesis_plot_style

    init_thesis_plot_style()

    # B -> D l nu

    bToD_evtgen = BToDLNuCLN(PDG.m_Bplus, PDG.m_Dzero, 41.1e-3)  # FIXME: Put proper Vcb which was used in Belle's Evtgen
    bToD_glattauer_bgl = BToDLNuBGL(PDG.m_Bplus, PDG.m_Dzero, V_cb=40.83e-3,
                                    bgl_fplus_coefficients=[0.0126, -0.094, 0.34, -0.1])
    bToD_glattauer_cln = BToDLNuCLN(PDG.m_Bplus, PDG.m_Dzero, V_cb=39.86e-3, cln_g1=1.0541, cln_rho2=1.09)

    w_min = 1
    w_max = (bToD_evtgen.m_B ** 2 + bToD_evtgen.m_D ** 2) / (2 * bToD_evtgen.m_B * bToD_evtgen.m_D)

    w_range = np.linspace(w_min, w_max, endpoint=True)

    plt.plot(w_range, bToD_evtgen.dGamma_dw(w_range) * 1e15,
             color=Tango.slate, ls='solid', lw=2, label='CLN Belle Evtgen')
    plt.plot(w_range, bToD_glattauer_cln.dGamma_dw(w_range) * 1e15,
             color=Tango.sky_blue, ls='dashed', lw=2, label='CLN arXiv:1510.03657v3')
    plt.plot(w_range, bToD_glattauer_bgl.dGamma_dw(w_range) * 1e15,
             color=Tango.orange, ls='dotted', lw=2, label='BGL arXiv:1510.03657v3')
    plt.xlabel(r'$w$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}w \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(w_min, w_max)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig('BToD_dGamma.png')
    plt.show()
    plt.close()

    # B -> D* l nu

    bToDstar_CLN = BToDstarLNuCLN(PDG.m_Bplus, PDG.m_Dstarzero, 37.4e-3)
    bToDstar_BGL = BToDstarLNuBGL(PDG.m_Bplus, PDG.m_Dstarzero, 41.6558e-3)

    w_min = 1
    w_max = (bToDstar_CLN.m_B ** 2 + bToDstar_CLN.m_Dstar ** 2) / (2 * bToDstar_CLN.m_B * bToDstar_CLN.m_Dstar)

    w_range = np.linspace(w_min + 1e-7, w_max - 1e-7, endpoint=True)
    cosl_range = np.linspace(-1, 1, endpoint=True)
    cosnu_range = np.linspace(-1, 1, endpoint=True)
    chi_range = np.linspace(0, 2 * np.pi, endpoint=True)
    pdg_codes = np.random.choice([22, 111, 211], len(w_range))

    plt.plot(w_range, [bToDstar_CLN.dGamma_dw(x, 211) * 1e15 for x in w_range],
             color=Tango.sky_blue, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ CLN arXiv:1702.01521v2')
    plt.plot(w_range, [bToDstar_BGL.dGamma_dw(x, 211) * 1e15 for x in w_range],
             color=Tango.orange, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ BGL arXiv:1703.08170v2')
    plt.plot(w_range, [bToDstar_CLN.dGamma_dw(x, 22) * 1e15 for x in w_range],
             color=Tango.scarlet_red, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ CLN arXiv:1702.01521v2')
    plt.plot(w_range, [bToDstar_BGL.dGamma_dw(x, 22) * 1e15 for x in w_range],
             color=Tango.chameleon, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ BGL arXiv:1703.08170v2')
    plt.xlabel(r'$w$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}w \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(w_min, w_max)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dw.png')
    plt.show()
    plt.close()

    plt.plot(chi_range, [bToDstar_CLN.dGamma_dchi(x, 211) * 1e15 for x in chi_range],
             color=Tango.sky_blue, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ CLN arXiv:1702.01521v2')
    plt.plot(chi_range, [bToDstar_BGL.dGamma_dchi(x, 211) * 1e15 for x in chi_range],
             color=Tango.orange, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ BGL arXiv:1703.08170v2')
    plt.plot(chi_range, [bToDstar_CLN.dGamma_dchi(x, 22) * 1e15 for x in chi_range],
             color=Tango.scarlet_red, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ CLN arXiv:1702.01521v2')
    plt.plot(chi_range, [bToDstar_BGL.dGamma_dchi(x, 22) * 1e15 for x in chi_range],
             color=Tango.chameleon, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\chi \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(0, 2 * np.pi)
    # plt.ylim(0, 6)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dchi.png')
    plt.show()
    plt.close()

    plt.plot(cosl_range, [bToDstar_CLN.dGamma_dcosLepton(x, 211) * 1e15 for x in cosl_range],
             color=Tango.sky_blue, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ CLN arXiv:1702.01521v2')
    plt.plot(cosl_range, [bToDstar_BGL.dGamma_dcosLepton(x, 211) * 1e15 for x in cosl_range],
             color=Tango.orange, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ BGL arXiv:1703.08170v2')
    plt.plot(cosl_range, [bToDstar_CLN.dGamma_dcosLepton(x, 22) * 1e15 for x in cosl_range],
             color=Tango.scarlet_red, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ CLN arXiv:1702.01521v2')
    plt.plot(cosl_range, [bToDstar_BGL.dGamma_dcosLepton(x, 22) * 1e15 for x in cosl_range],
             color=Tango.chameleon, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\cos\theta_l$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\cos\theta_l \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(-1, 1)
    # plt.ylim(0, 23)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dcosl.png')
    plt.show()
    plt.close()

    plt.plot(cosnu_range, [bToDstar_CLN.dGamma_dcosNeutrino(x, 211) * 1e15 for x in cosnu_range],
             color=Tango.sky_blue, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ CLN arXiv:1702.01521v2')
    plt.plot(cosnu_range, [bToDstar_BGL.dGamma_dcosNeutrino(x, 211) * 1e15 for x in cosnu_range],
             color=Tango.orange, ls='solid', lw=2, label=r'$D^* \rightarrow D \pi$ BGL arXiv:1703.08170v2')
    plt.plot(cosnu_range, [bToDstar_CLN.dGamma_dcosNeutrino(x, 22) * 1e15 for x in cosnu_range],
             color=Tango.scarlet_red, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ CLN arXiv:1702.01521v2')
    plt.plot(cosnu_range, [bToDstar_BGL.dGamma_dcosNeutrino(x, 22) * 1e15 for x in cosnu_range],
             color=Tango.chameleon, ls='dotted', lw=2, label=r'$D^* \rightarrow D \gamma$ BGL arXiv:1703.08170v2')
    plt.xlabel(r'$\cos\theta_\nu$')
    plt.ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\cos\theta_\nu \cdot 10^{-15}$')
    plt.title(r'$B \rightarrow D^* l \nu$')
    plt.legend(prop={'size': 12})
    plt.xlim(-1, 1)
    # plt.ylim(0, 21)
    plt.tight_layout()
    plt.savefig('BToDstar_dGamma_dcosnu.png')
    plt.show()
    plt.close()
