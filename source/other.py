import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import pandas as pd
from settings import Settings
from scipy.signal import savgol_filter


class Analysis:
    """
    A class to make the plots included in the proceeding published with
    the talk.
    """
    def __init__(self):
        # Initialize configuration class.
        self.settings = Settings()
        self.groups = json.load(open("data/groups.json"))

        self.galaxies = [f"Au{i}" for i in range(1, 31)]
        self.reruns = [
            f"Au{i}_rerun" for i in [5, 6, 9, 13, 17, 23, 24, 26, 28]]

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
                self._make_galaxy_dataframe(galaxy, False)
            if galaxy in self.settings.reruns:
                self._make_galaxy_dataframe(galaxy, True)

    def _make_galaxy_dataframe(self, galaxy: int, rerun: bool) -> None:
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
        data_path = f'data/au{galaxy}{rerun_str}/'

        df = pd.DataFrame()

        # Add snapshot properties.
        with open('data/simulation_data.json', 'r') as f:
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
        if not rerun:
            virial_masses = np.loadtxt('data/virial_mass.csv')
            virial_mass = np.nan * np.ones(n_snaps)
            virial_mass[-1] = virial_masses[galaxy]
            df['VirialMass'] = virial_mass

        # Load net accretion rates.
        if not rerun:
            net_rate = np.loadtxt(f'{data_path}accretion_cells_disc.csv')
            df['NetRate'] = net_rate

        # Load inflow and outflow rates only for the reruns.
        if rerun:
            data = np.loadtxt(f'{data_path}accretion_tracers_disc_spacing.csv')
            df['InflowRate'] = data[:, 0]
            df['OutflowRate'] = data[:, 1]

        # Add this data frame to the class dictionary.
        self.dfs[f'Au{galaxy}{rerun_str}'] = df

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
        plt.savefig('images/proceeding/disc_to_total_evolution.png')
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
        plt.savefig('images/proceeding/alignment_evolution.png')
        plt.close(fig)

    def plot_disc_size_evolution(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the radius and height of the
        stellar disc for a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """

        fig = plt.figure(figsize=(3, 3.75))
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
                        'o-', lw=0.75, ms=3, mec='White',
                        mew=0.5, label=galaxy)
            axs[1].plot(self.dfs[galaxy]['Time'],
                        self.dfs[galaxy]['ExpFactor']
                        * self.dfs[galaxy]['DiscHeight'],
                        'o-', lw=0.75, ms=3, mec='White', mew=0.5,
                        label=galaxy)

        self.settings.add_redshift(axs[0])
        axs[0].legend(loc='upper left', framealpha=0, fontsize=6)
        plt.savefig('images/proceeding/disc_size_evolution.png')
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
        ax.set_xlabel(r'Time [Gyr]')
        ax.set_ylabel(r'$\dot{M}_\mathrm{in}$ '
                      + r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')

        i = 0
        for galaxy in self.reruns:
            is_finite = np.isfinite(self.dfs[galaxy]['InflowRate'])
            if galaxy in galaxies:
                color = list(mcolors.TABLEAU_COLORS)[i]
                i += 1
                label = galaxy.split('_')[0]
                zorder = 10
            else:
                color = 'silver'
                label = None
                zorder = 9

            ax.plot(self.dfs[galaxy]['Time'][is_finite],
                    self.dfs[galaxy]['InflowRate'][is_finite],
                    '-', lw=1, color=color, zorder=zorder, label=label)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower right', framealpha=0, fontsize=6)
        plt.savefig('images/proceeding/inflow_rate.pdf')
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
        ax.set_xlabel(r'Time [Gyr]')
        ax.set_ylabel(r'$\dot{M}_\mathrm{out}$ '
                      + r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')

        i = 0
        for galaxy in self.reruns:
            is_finite = np.isfinite(self.dfs[galaxy]['OutflowRate'])
            if galaxy in galaxies:
                color = list(mcolors.TABLEAU_COLORS)[i]
                i += 1
                label = galaxy.split('_')[0]
                zorder = 10
            else:
                color = 'silver'
                label = None
                zorder = 9

            ax.plot(self.dfs[galaxy]['Time'][is_finite],
                    self.dfs[galaxy]['OutflowRate'][is_finite],
                    '-', lw=1, color=color, zorder=zorder, label=label)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower right', framealpha=0, fontsize=6)
        plt.savefig('images/proceeding/outflow_rate.pdf')
        plt.close(fig)

    def plot_net_accretion(self, galaxies: list) -> None:
        """
        Creates a plot of the evolution of the net accretion rate
        for a given list of galaxies.

        :param galaxies: A list of galaxy labels to include in the plot.
        """
        # Check if rerun is indicated in the labels.
        for galaxy in galaxies:
            if '_rerun' in galaxy:
                raise Exception("_rerun found in labels. This function only"
                                "runs on the original simulations.")

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
        ax.set_xlabel(r'Time [Gyr]')
        ax.set_ylabel(r'$\dot{M}_\mathrm{net}$ '
                      + r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')

        colors = ["tab:red", "tab:blue", "tab:purple"]

        i = 0
        for galaxy in self.galaxies:
            is_finite = np.isfinite(self.dfs[galaxy]['NetRate'])
            is_positive = self.dfs[galaxy]['NetRate'] > 0.1
            if galaxy in galaxies:
                color = colors[i]
                i += 1
                label = galaxy.split('_')[0]
                zorder = 10
            else:
                color = 'silver'
                label = None
                zorder = 9

            ax.plot(self.dfs[galaxy]['Time'][is_finite & is_positive][:-2],
                    savgol_filter(
                        self.dfs[galaxy]['NetRate'][is_finite & is_positive],
                        5,
                        1)[:-2],
                    '-', lw=1, color=color, label=label, zorder=zorder)

        self.settings.add_redshift(ax)
        ax.legend(loc='lower right', framealpha=0, fontsize=6)
        plt.savefig('images/proceeding/net_rate.pdf')
        plt.close(fig)


if __name__ == '__main__':
    # Initialize class
    analysis = Analysis()

    # Configure plots
    analysis.settings.config_plots()

    # Make plots
    analysis.plot_disc_to_total_evolution(galaxies=['Au4', 'Au18', 'Au29'])
    analysis.plot_alignment_evolution(galaxies=['Au6', 'Au19', 'Au30'])
    analysis.plot_disc_size_evolution(galaxies=['Au6', 'Au8', 'Au30'])
    analysis.plot_inflows(galaxies=['Au6_rerun', 'Au13_rerun', 'Au28_rerun'])
    analysis.plot_outflows(galaxies=['Au6_rerun', 'Au13_rerun', 'Au28_rerun'])
    analysis.plot_net_accretion(galaxies=['Au4', 'Au6', 'Au10'])
