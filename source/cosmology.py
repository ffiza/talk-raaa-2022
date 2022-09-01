from astropy.cosmology import FlatLambdaCDM


class Cosmology:
    """
    A class to manage settings relative to cosmological parameters.
    """
    def __init__(self):
        self.h = 0.6777
        self.hubble_const = 67.77  # km/s/Mpc
        self.omega0 = 0.307
        self.omega_baryon = 0.048
        self.omega_lambda = 0.693

        self.cosmology = FlatLambdaCDM(H0=self.hubble_const, Om0=self.omega0)
        self.present_time = self.cosmology.age(0).value  # Gyr
