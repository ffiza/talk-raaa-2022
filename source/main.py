import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import pandas as pd
from settings import Settings
from scipy.signal import savgol_filter


class Analysis:
    """
    A class to make the plots included in the talk.
    """
    def __init__(self):
        # Initialize configuration class.
        self.settings = Settings()

        # Initialize data frames. This is a dictionary for each galaxy, where
        # values are data frames that contain all relevant information of each
        # galaxy. The keys to the dictionary are galaxy labels like
        # "Au6" or "Au6_rerun" for the rerun version.
        self.dfs = {}

        # Load data into self.dfs.
        self._make_dataframes()

    def _make_dataframes(self) -> None:
        """
        This method loads the disc_to_total data.
        """
        for galaxy in self.settings.galaxies:
            if galaxy in self.settings.galaxies:
                self._make_original_dataframe(galaxy, False)
            if galaxy in self.settings.reruns:
                self._make_original_dataframe(galaxy, True)

    def _make_original_dataframe(self, galaxy: int, rerun: bool) -> None:
        """
        This method makes a data frame with all relevant information for a
        given galaxy.

        :param galaxy: The galaxy number to make data frame.
        :param rerun: A boolean indicating if the galaxy is a rerun version
                      (True) or not (False).
        """
        if rerun:
            rerun_str = '_rerun'
            n_snaps = 252
        else:
            rerun_str = ''
            n_snaps = 128
        data_path = f'../data/au{galaxy}{rerun_str}/'

        df = pd.DataFrame()

        # Add snapshot properties.
        with open('../data/simulation_data.json', 'r') as f:
            data = json.load(f)
            if rerun:
                df['Time'] = np.array(data['Rerun']['Time_Gyr'])
                df['Redshift'] = np.array(data['Rerun']['Redshift'])
                df['ExpFactor'] = np.array(data['Rerun']['ExpansionFactor'])
            else:
                df['Time'] = np.array(data['Original']['Time_Gyr'])
                df['Redshift'] = np.array(data['Original']['Redshift'])
                df['ExpFactor'] = np.array(data['Original']['ExpansionFactor'])

        # Add disc-to-total.
        disc_to_total = np.loadtxt(f'{data_path}disc_to_total.csv')
        df['DiscToTotal'] = disc_to_total

        # Add alignment.
        alignment = np.loadtxt(f'{data_path}alignment.csv')
        df['CosBeta'] = alignment

        # Add disc SFR.
        sfr = np.loadtxt(f'{data_path}sfr.csv')
        df['SFR'] = sfr

        # Add the disc sizes to the data frame.
        disc_radius = np.loadtxt(f'{data_path}disc_radius.csv')
        disc_height = np.loadtxt(f'{data_path}disc_height.csv')
        df['DiscRadius'] = disc_radius
        df['DiscHeight'] = disc_height

        # Load present-day virial masses only for the original simulations.
        if galaxy in self.settings.galaxies:
            virial_masses = np.loadtxt('../data/virial_mass.csv')
            virial_mass = np.nan * np.ones(n_snaps)
            virial_mass[-1] = virial_masses[galaxy]
            df['VirialMass'] = virial_mass

        # Load inflow and outflow rates only for the reruns.
        if rerun:
            data = np.loadtxt(f'{data_path}accretion_tracers_disc_spacing.csv')
            df['InflowRate'] = data[:, 0]
            df['OutflowRate'] = data[:, 1]

        # Add this data frame to the class dictionary.
        self.dfs[f'Au{galaxy}{rerun_str}'] = df

    def plot_virial_mass_dist(self) -> None:
        """
        Creates a plot of the distribution of the present-day virial masses for
        all the original simulations.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(0.8, 2)
        ax.set_xticks([1, 1.2, 1.4, 1.6, 1.8])
        ax.set_ylim(0, 10)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel(r'$M_{200}$ [$10^{12} \, \mathrm{M}_\odot$]')

        virial_masses = [self.dfs[f'Au{i}']['VirialMass'].iloc[-1] for i in
                         self.settings.galaxies]
        ax.hist(np.array(virial_masses)/100, bins=10, histtype='stepfilled',
                edgecolor='tab:blue',
                facecolor=mcolors.to_rgba('tab:blue', 0.4))

        xerr = np.array([0.22, 0.25]).reshape(2, 1)
        ax.errorbar(1.15, 8.5, xerr=xerr, fmt='s-', elinewidth=1, capsize=3,
                    capthick=1, markersize=2, color='#073A59')
        ax.text(1.15, 9.25, r'\textsc{vía láctea}', ha='center', va='bottom',
                fontsize=6, color='#073A59')
        ax.text(1.15, 8.75, '(Carlesi et al. 2022)', ha='center', va='bottom',
                fontsize=4, color='#073A59')

        plt.savefig('../images/virial_mass_dist.png')
        plt.close(fig)

    def plot_disc_to_total_dist(self) -> None:
        """
        Creates a plot of the distribution of the present-day D/T mass ratios
        for all the original simulations.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylim(0, 13)
        # ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel(r'D/T')

        disc_to_totals = [self.dfs[f'Au{i}']['DiscToTotal'].iloc[-1]
                          for i in self.settings.galaxies]
        ax.hist(disc_to_totals, bins=10, histtype='stepfilled',
                edgecolor='tab:blue',
                facecolor=mcolors.to_rgba('tab:blue', 0.4),
                range=(0, 1))

        ax.text(0.15, 1.25, r'\textsc{Au29}', ha='center', va='bottom',
                fontsize=6, color='#073A59')
        ax.text(0.35, 1.25, r'\textsc{Au4}', ha='center', va='bottom',
                fontsize=6, color='#073A59')
        ax.text(0.85, 5.25, r'\textsc{Au18}', ha='center', va='bottom',
                fontsize=6, color='#073A59')

        plt.savefig('../images/disc_to_total_dist.png')
        plt.close(fig)

    def plot_disc_to_total_evolution(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the disc-to-total mass ratio for
        a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(0, 14)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel(r'Tiempo [Gyr]')
        ax.set_ylabel(r'D/T')

        for galaxy in galaxies:
            ax.plot(self.dfs[galaxy]['Time'],
                    savgol_filter(self.dfs[galaxy]['DiscToTotal'], 5, 1),
                    'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)

        self.settings.add_redshift(ax)
        ax.legend(loc='upper center', framealpha=0, fontsize=6, ncol=3)
        plt.savefig('../images/disc_to_total_evolution.png')
        plt.close(fig)

    def plot_alignment_dist(self) -> None:
        """
        Creates a plot of the distribution of the present-day cosine of beta
        for all the original simulations.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_xlabel(r'$\cos \beta$')

        alignments = [self.dfs[f'Au{i}']['CosBeta'].iloc[-1]
                      for i in self.settings.galaxies]
        ax.hist(alignments, bins=10, histtype='stepfilled',
                edgecolor='tab:blue',
                facecolor=mcolors.to_rgba('tab:blue', 0.4),
                range=(-1, 1))

        ax.text(-0.3, 1.25, r'\textsc{Au30}', ha='center', va='bottom',
                fontsize=6, color='#073A59')

        plt.savefig('../images/alignment_dist.png')
        plt.close(fig)

    def plot_alignment_evolution(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the cosine of beta for
        a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(0, 14)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel(r'Tiempo [Gyr]')
        ax.set_ylabel(r'$\cos \beta$')

        for galaxy in galaxies:
            ax.plot(self.dfs[galaxy]['Time'],
                    savgol_filter(self.dfs[galaxy]['CosBeta'], 5, 1),
                    'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower left', framealpha=0, fontsize=6)
        plt.savefig('../images/alignment_evolution.png')
        plt.close(fig)

    def plot_disc_size_evolution(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the radius and height of the
        stellar disc for a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        fig = plt.figure(figsize=(4*0.75, 3*1.25))
        gs = fig.add_gridspec(ncols=1, nrows=2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)

        for ax in axs:
            ax.tick_params(which='both', direction="in")
            ax.set_axisbelow(True)
            ax.grid(True, linestyle='--', lw=.5)
            ax.set_xlim(0, 14)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.set_xlabel(r'Tiempo [Gyr]')

        axs[0].set_ylim(0, 40)
        axs[1].set_ylim(0, 4)

        axs[0].set_ylabel(r'$R_\mathrm{d}$ [kpc]')
        axs[1].set_ylabel(r'$h_\mathrm{d}$ [kpc]')
        axs[1].set_yticks([0, 1, 2, 3])

        for galaxy in galaxies:
            axs[0].plot(self.dfs[galaxy]['Time'],
                        self.dfs[galaxy]['ExpFactor']
                        * self.dfs[galaxy]['DiscRadius'],
                        'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)
            axs[1].plot(self.dfs[galaxy]['Time'],
                        self.dfs[galaxy]['ExpFactor']
                        * self.dfs[galaxy]['DiscHeight'],
                        'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)

        self.settings.add_redshift(axs[0])
        axs[0].legend(loc='upper left', framealpha=0, fontsize=6)
        plt.savefig('../images/disc_size_evolution.png')
        plt.close(fig)

    def plot_inflows(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the inflow rate
        for a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        # Check if rerun is indicated in the labels.
        for galaxy in galaxies:
            if '_rerun' not in galaxy:
                raise Exception("_rerun not found in labels.")

        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(0, 14)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(0.1, 300)
        ax.set_yscale('log')
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels([0.1, 1, 10, 100])
        ax.set_xlabel(r'Tiempo [Gyr]')
        ax.set_ylabel(r'$\dot{M}_\mathrm{in}$ '
                      + r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')

        for galaxy in galaxies:
            is_finite = np.isfinite(self.dfs[galaxy]['InflowRate'])
            ax.plot(self.dfs[galaxy]['Time'][is_finite],
                    self.dfs[galaxy]['InflowRate'][is_finite],
                    'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower left', framealpha=0, fontsize=6)
        plt.savefig('../images/inflow_rate.png')
        plt.close(fig)

    def plot_outflows(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the outflow rate
        for a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        # Check if rerun is indicated in the labels.
        for galaxy in galaxies:
            if '_rerun' not in galaxy:
                raise Exception("_rerun not found in labels.")

        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=1)
        ax = gs.subplots(sharex=True, sharey=True)

        ax.tick_params(which='both', direction="in")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', lw=.5)
        ax.set_xlim(0, 14)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(0.1, 300)
        ax.set_yscale('log')
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels([0.1, 1, 10, 100])
        ax.set_xlabel(r'Tiempo [Gyr]')
        ax.set_ylabel(r'$\dot{M}_\mathrm{in}$ '
                      + r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')

        for galaxy in galaxies:
            is_finite = np.isfinite(self.dfs[galaxy]['InflowRate'])
            ax.plot(self.dfs[galaxy]['Time'][is_finite],
                    self.dfs[galaxy]['OutflowRate'][is_finite],
                    'o-', lw=0.75, ms=3, mec='White', mew=0.5, label=galaxy)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower left', framealpha=0, fontsize=6)
        plt.savefig('../images/outflow_rate.png')
        plt.close(fig)


if __name__ == '__main__':
    # Initialize class
    analysis = Analysis()

    # Configure plots
    analysis.settings.config_plots()

    # Make plots
    analysis.plot_virial_mass_dist()
    # analysis.plot_disc_to_total_dist()
    # analysis.plot_disc_to_total_evolution(galaxies=['Au4', 'Au18', 'Au29'])
    # analysis.plot_alignment_dist()
    # analysis.plot_alignment_evolution(galaxies=['Au6', 'Au19', 'Au30'])
    # analysis.plot_disc_size_evolution(galaxies=['Au6', 'Au8', 'Au30'])
    analysis.plot_inflows(galaxies=['Au6_rerun', 'Au13_rerun', 'Au28_rerun'])
    analysis.plot_outflows(galaxies=['Au6_rerun', 'Au13_rerun', 'Au28_rerun'])
