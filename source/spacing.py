import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import pandas as pd
from settings import Settings
from scipy.signal import savgol_filter


def make_galaxy_dataframe(galaxy: int, rerun: bool) -> None:

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

    # Load net accretion rates
    if not rerun:
        net_rate = np.loadtxt(f'{data_path}accretion_cells_disc.csv')
        df['NetRate'] = net_rate

    # Load inflow and outflow rates only for the reruns
    if rerun:
        data = np.loadtxt(f'{data_path}accretion_tracers_disc_spacing.csv')
        df['InflowRate'] = data[:, 0]
        df['OutflowRate'] = data[:, 1]

    return df


def plot_inflows(df: pd.DataFrame) -> None:
    settings = Settings()

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

    is_finite = np.isfinite(df['InflowRate'])
    ax.plot(df['Time'][is_finite],
            df['InflowRate'][is_finite],
            '-', lw=1, color="tab:blue", label="Original Rate")

    is_finite = np.isfinite(df["CenteredInflowRate_N3"])
    ax.plot(df['Time'][is_finite],
            df['CenteredInflowRate_N3'][is_finite],
            '-', lw=1, color="tab:red", label="Centered, $N=3$")

    is_finite = np.isfinite(df["CenteredInflowRate_N5"])
    ax.plot(df['Time'][is_finite],
            df['CenteredInflowRate_N5'][is_finite],
            '-', lw=1, color="tab:purple", label="Centered, $N=5$")

    is_finite = np.isfinite(df["CenteredInflowRate_N7"])
    ax.plot(df['Time'][is_finite],
            df['CenteredInflowRate_N7'][is_finite],
            '-', lw=1, color="tab:green", label="Centered, $N=7$")

    settings.add_redshift(ax)

    ax.legend(loc='lower right', framealpha=0, fontsize=8)
    plt.savefig('images/proceeding/referee_plot.png')
    plt.close(fig)



if __name__ == '__main__':
    settings = Settings()
    settings.config_plots()

    df = make_galaxy_dataframe(galaxy=6, rerun=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    dt = np.diff(df["Time"].to_numpy())
    dt = np.append(np.nan, dt)
    df["DeltaTime_Gyr"] = dt

    inflow_mass = df["InflowRate"] * df["DeltaTime_Gyr"] * 1E9
    df["InflowMass_Msun"] = inflow_mass

    centered_inflow_rate = np.nan * np.ones(len(df))
    for i in range(len(centered_inflow_rate)):
        if i == 0 or i == 1 or i == len(centered_inflow_rate) - 1:
            pass
        else:
            delta_time = df["Time"].iloc[i + 1] - df["Time"].iloc[i - 1]
            delta_mass = df["InflowMass_Msun"].iloc[i] \
                + df["InflowMass_Msun"].iloc[i + 1]
            inflow_rate = delta_mass / delta_time / 1E9
            centered_inflow_rate[i] = inflow_rate
    df["CenteredInflowRate_N3"] = centered_inflow_rate

    centered_inflow_rate = np.nan * np.ones(len(df))
    for i in range(len(centered_inflow_rate)):
        length = len(centered_inflow_rate)
        if i in [0, 1, 2, length - 1, length - 2]:
            pass
        else:
            delta_time = df["Time"].iloc[i + 2] - df["Time"].iloc[i - 2]
            delta_mass = df["InflowMass_Msun"].iloc[i] \
                + df["InflowMass_Msun"].iloc[i + 1] \
                + df["InflowMass_Msun"].iloc[i + 2] \
                + df["InflowMass_Msun"].iloc[i - 1]
            inflow_rate = delta_mass / delta_time / 1E9
            centered_inflow_rate[i] = inflow_rate
    df["CenteredInflowRate_N5"] = centered_inflow_rate

    centered_inflow_rate = np.nan * np.ones(len(df))
    for i in range(len(centered_inflow_rate)):
        length = len(centered_inflow_rate)
        if i in [0, 1, 2, 3, length - 1, length - 2, length - 3]:
            pass
        else:
            delta_time = df["Time"].iloc[i + 3] - df["Time"].iloc[i - 3]
            delta_mass = df["InflowMass_Msun"].iloc[i] \
                + df["InflowMass_Msun"].iloc[i + 1] \
                + df["InflowMass_Msun"].iloc[i + 2] \
                + df["InflowMass_Msun"].iloc[i - 1] \
                + df["InflowMass_Msun"].iloc[i - 2] \
                + df["InflowMass_Msun"].iloc[i + 3]
            inflow_rate = delta_mass / delta_time / 1E9
            centered_inflow_rate[i] = inflow_rate
    df["CenteredInflowRate_N7"] = centered_inflow_rate

    plot_inflows(df)
