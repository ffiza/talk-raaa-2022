import matplotlib.pyplot as plt
from cosmology import Cosmology


class Settings:
    """
    A class to manage global configurations and variables.
    """
    def __init__(self):
        # Galaxy labels
        self.galaxies = [i for i in range(1, 31)]
        self.reruns = [5, 6, 9, 13, 17, 23, 24, 26, 28]
        self.labels = [f'Au{i}' for i in range(1, 31)]

        # Plot configurations
        self.dpi = 800
        self.figsize = (4*0.75, 3*0.75)
        self.fontsize = 8
        self.label_fontsize = 10
        self.pad_inches = 0.02
        self.grid_linewidth = 0.5

        # Cosmology
        self.cosmology = Cosmology()

    def config_plots(self) -> None:
        """
        Configure plot style.
        """
        params = {'figure.dpi':         self.dpi,
                  'figure.figsize':     self.figsize,
                  'text.usetex':        True,
                  'font.size':          self.fontsize,
                  'font.family':        'serif',
                  'axes.labelsize':     self.label_fontsize,
                  'xtick.top':          'on',
                  'xtick.minor.bottom': 'on',
                  'xtick.minor.top':    'on',
                  'xtick.direction':    'in',
                  'ytick.right':        'on',
                  'ytick.minor.left':   'on',
                  'ytick.minor.right':  'on',
                  'ytick.direction':    'in',
                  'savefig.dpi':        self.dpi,
                  'savefig.bbox':       'tight',
                  'savefig.pad_inches': self.pad_inches,
                  'axes.axisbelow':     True,
                  'axes.grid':          True,
                  'grid.linestyle':     '--',
                  'grid.linewidth':     self.grid_linewidth}
        plt.rcParams.update(params)

    def add_redshift(self, ax: plt.Axes) -> None:
        """
        Add ticks and labels of the redshift for the current ax.

        :param ax: An instance of the plt.Axes class to add the redshift.
        """
        ax2 = ax.twiny()
        ax2.tick_params(which='both', direction="in")
        ax2.grid(False)
        ax2_label_values = [0.1, 0.5, 1, 2, 3, 10]
        ax2_ticklabels = ['0.1', '0.5', '1', '2', '3', '10']
        ax2_ticks = [self.cosmology.cosmology.age(float(item)).value
                     for item in ax2_label_values]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax2_ticks)
        if ax.get_subplotspec().is_first_row():
            ax2.set_xticklabels(ax2_ticklabels)
            ax2.set_xlabel('$z$')
        else:
            ax2.set_xticklabels([])
