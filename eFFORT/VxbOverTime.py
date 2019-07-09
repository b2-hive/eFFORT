import matplotlib.pyplot as plt
import numpy
import uncertainties
from uncertainties import ufloat
from wg1template.histogram_plots import create_solo_figure, add_descriptions_to_plot
from wg1template.plot_style import TangoColors
from wg1template.plot_utilities import export
from wg1template.point_plots import DataPoints, DataVariable, DataPointsPlot

"""The values are taken from from the PDG reviews of the corresponding year."""

v_ub_incl = {
    2002: ufloat(4.11, (0.25 ** 2 + 0.78 ** 2) ** 0.5) * 1e-3,
    2004: ufloat(4.68, 0.85) * 1e-3,
    2006: ufloat(4.40, (0.20 ** 2 + 0.27 ** 2) ** 0.5) * 1e-3,
    2008: ufloat(4.12, 0.43) * 1e-3,
    2010: ufloat(4.27, 0.38) * 1e-3,
    2012: ufloat(4.41, (0.15 ** 2 + 0.15 ** 2 + 0.16 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2014: ufloat(4.41, (0.15 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2015: ufloat(4.49, (0.16 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2016: ufloat(4.49, (0.16 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2017: ufloat(4.49, (0.15 ** 2 + 0.165 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2018: ufloat(4.49, (0.15 ** 2 + 0.165 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
    2019: ufloat(4.49, (0.15 ** 2 + 0.165 ** 2 + 0.17 ** 2) ** 0.5) * 1e-3,  # Up/down error symmetrized
}

v_ub_excl = {
    2002: ufloat(3.25, (0.32 ** 2 + 0.64 ** 2) ** 0.5) * 1e-3,
    2004: ufloat(3.326, 0.59) * 1e-3,
    2006: ufloat(3.84, (0.67 + 0.49) / 2) * 1e-3,  # Up/down error symmetrized
    2008: ufloat(3.5, (0.6 + 0.5) / 2) * 1e-3,  # Up/down error symmetrized
    2010: ufloat(3.38, 0.36) * 1e-3,
    2012: ufloat(3.23, 0.31) * 1e-3,
    2014: ufloat(3.28, 0.29) * 1e-3,
    2015: ufloat(3.72, 0.19) * 1e-3,
    2016: ufloat(3.72, 0.19) * 1e-3,
    2017: ufloat(3.70, (0.10 ** 2 + 0.12 ** 2) ** 0.5) * 1e-3,
    2018: ufloat(3.70, (0.10 ** 2 + 0.12 ** 2) ** 0.5) * 1e-3,
    2019: ufloat(3.70, (0.10 ** 2 + 0.12 ** 2) ** 0.5) * 1e-3,
}

v_ub_avg = {
    2002: ufloat(3.6, 0.7) * 1e-3,
    2004: ufloat(3.67, 0.47) * 1e-3,
    2006: ufloat(4.31, 0.30) * 1e-3,
    2008: ufloat(3.95, 0.35) * 1e-3,
    2010: ufloat(3.89, 0.44) * 1e-3,
    2012: ufloat(4.15, 0.49) * 1e-3,
    2014: ufloat(4.13, 0.49) * 1e-3,
    2015: ufloat(4.09, 0.39) * 1e-3,
    2016: ufloat(4.09, 0.39) * 1e-3,
    2017: ufloat(3.94, 0.36) * 1e-3,
    2018: ufloat(3.94, 0.36) * 1e-3,
    2019: ufloat(3.94, 0.36) * 1e-3,
}

v_ub_munu = {
    2019: ufloat(4.37, (0.82 + 1.01) / 2) * 1e-3
}

v_ub_taunu = {
    2015: ufloat(4.2, 0.45) * 1e-3
}

v_cb_incl = {
    2002: ufloat(40.4, (0.5 ** 2 + 0.5 ** 2 + 0.8 ** 2) ** 0.5) * 1e-3,
    2004: ufloat(41.0, (0.5 ** 2 + 0.5 ** 2 + 0.8 ** 2) ** 0.5) * 1e-3,
    2006: ufloat(41.7, 0.7) * 1e-3,
    2008: ufloat(41.6, 0.6) * 1e-3,
    2010: ufloat(41.5, 0.7) * 1e-3,
    2012: ufloat(41.9, 0.7) * 1e-3,
    2014: ufloat(42.2, 0.7) * 1e-3,
    2015: ufloat(42.2, 0.8) * 1e-3,
    2016: ufloat(42.2, 0.8) * 1e-3,
    2017: ufloat(42.2, 0.8) * 1e-3,
    2018: ufloat(42.2, 0.8) * 1e-3,
    # 2019: ufloat(42.2, 0.8) * 1e-3,
}

v_cb_excl = {
    2002: ufloat(42.1, (1.1 ** 2 + 1.9 ** 2) ** 0.5) * 1e-3,
    # B->D l nu: 2002: ufloat(41.3, (4.0 ** 2 + 2.9 ** 2) ** 0.5),
    2004: ufloat(42.0, (1.1 ** 2 + 1.9 ** 2) ** 0.5) * 1e-3,
    2006: ufloat(40.9, 1.8) * 1e-3,
    2008: ufloat(38.6, 1.3) * 1e-3,
    2010: ufloat(38.7, 1.1) * 1e-3,
    2012: ufloat(39.6, 0.9) * 1e-3,
    2014: ufloat(39.5, 0.8) * 1e-3,
    2015: ufloat(39.2, 0.7) * 1e-3,
    2016: ufloat(39.2, 0.7) * 1e-3,
    2017: ufloat(41.9, 2.0) * 1e-3,
    2018: ufloat(41.9, 2.0) * 1e-3,
    # 2019: ufloat(41.9, 2.0) * 1e-3,
}

v_cb_avg = {
    2002: ufloat(41.2, 2.0) * 1e-3,
    2004: ufloat(41.3, 1.5) * 1e-3,
    2006: ufloat(41.6, 0.6) * 1e-3,
    2008: ufloat(41.2, 1.1) * 1e-3,
    2010: ufloat(40.6, 1.3) * 1e-3,
    2012: ufloat(40.9, 1.1) * 1e-3,
    2014: ufloat(41.1, 1.3) * 1e-3,
    2015: ufloat(40.5, 1.5) * 1e-3,
    2016: ufloat(40.5, 1.5) * 1e-3,
    2017: ufloat(42.2, 0.8) * 1e-3,
    2018: ufloat(42.2, 0.8) * 1e-3,
    # 2019: ufloat(42.2, 0.8) * 1e-3,
}

v_xb_ratio = {
    2015: ufloat(0.083, (0.004 ** 2 + 0.004 ** 2) ** 0.5)  # Nature Physics 11, 743–747 (2015)
}

v_cb_phil = {
    2019: ufloat(38.3, (0.3 ** 2 + 0.7 ** 2 + 0.6 ** 2) ** 0.5) * 1e-3  # BGL 1809.03290v2.pdf
}


def generate_plot_points(yearly_data, jitter=0.):
    return DataPoints(
        x_values=numpy.array([x - 2000 + jitter for x in yearly_data.keys()]),
        y_values=numpy.array([uncertainties.nominal_value(x) * 1e3 for x in yearly_data.values()]),
        x_errors=None,
        y_errors=numpy.array([uncertainties.std_dev(x) * 1e3 for x in yearly_data.values()])
    )


if __name__ == '__main__':
    V_ub_exclusive = generate_plot_points(v_ub_excl, 0.1)
    V_ub_inclusive = generate_plot_points(v_ub_incl, -0.1)
    V_ub_average = generate_plot_points(v_ub_avg)

    V_cb_exclusive = generate_plot_points(v_cb_excl, 0.1)
    V_cb_inclusive = generate_plot_points(v_cb_incl, -0.1)
    V_cb_average = generate_plot_points(v_cb_avg)

    V_ub_lhcb = DataPoints(
        x_values=2015 - 2000,
        y_values=uncertainties.nominal_value(v_xb_ratio[2015] * v_cb_avg[2018]) * 1e3,
        x_errors=None,
        y_errors=uncertainties.std_dev(v_xb_ratio[2015] * v_cb_avg[2018]) * 1e3,
    )

    V_cb_lhcb = DataPoints(
        x_values=2015 - 2000,
        y_values=uncertainties.nominal_value(v_ub_avg[2018] / v_xb_ratio[2015]) * 1e3,
        x_errors=None,
        y_errors=uncertainties.std_dev(v_ub_avg[2018] / v_xb_ratio[2015]) * 1e3,
    )

    V_cb_phil = generate_plot_points(v_cb_phil)

    V_ub_munu = generate_plot_points(v_ub_munu)
    V_ub_taunu = generate_plot_points(v_ub_taunu)

    form_factors = DataVariable(r"", r"", r'$|V_\mathrm{ub}| \times 10^3$', "")

    dp = DataPointsPlot(form_factors)
    dp.add_component(r"$V_\mathrm{ub}$ Exclusive", V_ub_exclusive, color=TangoColors.orange, marker='.')
    dp.add_component(r"$V_\mathrm{ub}$ Inclusive", V_ub_inclusive, color=TangoColors.sky_blue, marker='.')
    dp.add_component(r"$\Lambda_b \rightarrow p\mu\nu$ (1504.01568)", V_ub_lhcb, color=TangoColors.plum, marker='d')
    dp.add_component(r"$B\rightarrow \mu \nu$ Preliminary", V_ub_munu, color=TangoColors.scarlet_red, marker='d')
    dp.add_component(r"$B\rightarrow \tau \nu$", V_ub_taunu, color=TangoColors.chameleon, marker='d')

    fig, ax = create_solo_figure(figsize=(5, 3), dpi=100)

    ax.fill_between([-1, V_ub_exclusive.x_values[-1] + 3],
                    V_ub_exclusive.y_values[-1] + V_ub_exclusive.y_errors[-1],
                    V_ub_exclusive.y_values[-1] - V_ub_exclusive.y_errors[-1],
                    color=TangoColors.orange, alpha=0.3
                    )

    ax.fill_between([-1, V_ub_inclusive.x_values[-1] + 3],
                    V_ub_inclusive.y_values[-1] + V_ub_inclusive.y_errors[-1],
                    V_ub_inclusive.y_values[-1] - V_ub_inclusive.y_errors[-1],
                    color=TangoColors.sky_blue, alpha=0.3
                    )

    ax.axhline((0.00360 + 0.00017) * 1e3, color=TangoColors.slate, ls='--', lw=1)
    ax.axhline((0.00360 - 0.00011) * 1e3, color=TangoColors.slate, ls='--', lw=1)

    ax.fill_between([-1, V_ub_inclusive.x_values[-1] + 3],
                    (0.00360 + 0.00017) * 1e3,
                    (0.00360 - 0.00011) * 1e3,
                    facecolor="none", hatch="///", edgecolor=TangoColors.slate, lw=0,
                    label='CKM Unitarity'
                    )

    dp.plot_on(ax, draw_legend=True, legend_kwargs={'ncol': 2, }, legend_inside=True)

    add_descriptions_to_plot(
        ax,
        experiment=r'',
        luminosity=r'',
        additional_info=''
    )
    ax.set_ylim(2.5, 7.1)
    ax.set_xlim(1, V_ub_exclusive.x_values[-1] + 1)
    ax.set_xticks(V_ub_exclusive.x_values)
    ax.set_xticklabels((V_ub_exclusive.x_values + 2000).astype(int), rotation=-45)
    plt.show()
    export(fig, 'vub-over-time', '.')
    plt.close()

    form_factors = DataVariable(r"", r"", r'$|V_\mathrm{cb}| \times 10^3$', "")

    dp = DataPointsPlot(form_factors)
    dp.add_component(r"$V_\mathrm{cb}$ Exclusive", V_cb_exclusive, color=TangoColors.orange, marker='.')
    dp.add_component(r"$V_\mathrm{cb}$ Inclusive", V_cb_inclusive, color=TangoColors.sky_blue, marker='.')
    # dp.add_component(r"$\Lambda_b \rightarrow p\mu\nu$ (1504.01568)", V_cb_lhcb, color=TangoColors.plum, marker='d')
    dp.add_component(r"$B\rightarrow D^*\ell\nu$ (1809.03290)", V_cb_phil, color=TangoColors.scarlet_red, marker='*')

    fig, ax = create_solo_figure(figsize=(5, 3), dpi=100)

    ax.fill_between([-1, V_cb_exclusive.x_values[-1] + 3],
                    V_cb_exclusive.y_values[-1] + V_cb_exclusive.y_errors[-1],
                    V_cb_exclusive.y_values[-1] - V_cb_exclusive.y_errors[-1],
                    color=TangoColors.orange, alpha=0.3
                    )

    ax.fill_between([-1, V_cb_inclusive.x_values[-1] + 3],
                    V_cb_inclusive.y_values[-1] + V_cb_inclusive.y_errors[-1],
                    V_cb_inclusive.y_values[-1] - V_cb_inclusive.y_errors[-1],
                    color=TangoColors.sky_blue, alpha=0.3
                    )

    ax.axhline((0.04250 + 0.00036) * 1e3, color=TangoColors.slate, ls='--', lw=1)
    ax.axhline((0.04250 - 0.00116) * 1e3, color=TangoColors.slate, ls='--', lw=1)

    ax.fill_between([-1, V_ub_inclusive.x_values[-1] + 3],
                    (0.04250 + 0.00036) * 1e3,
                    (0.04250 - 0.00116) * 1e3,
                    facecolor="none", hatch="///", edgecolor=TangoColors.slate, lw=0,
                    label='CKM Unitarity'
                    )

    dp.plot_on(ax, draw_legend=True, legend_kwargs={'ncol': 2}, legend_inside=True)

    add_descriptions_to_plot(
        ax,
        experiment=r'',
        luminosity=r'',
        additional_info=''
    )
    ax.set_ylim(36, 52)
    ax.set_xlim(1, V_cb_exclusive.x_values[-1] + 2)
    ax.set_xticks([*V_cb_exclusive.x_values, 19])
    ax.set_xticklabels(([*(V_cb_exclusive.x_values + 2000).astype(int), 2019]), rotation=-45)
    plt.show()
    export(fig, 'vcb-over-time', '.')
    plt.close()
