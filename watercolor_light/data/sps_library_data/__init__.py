from typing import Optional
import numpy as np
from pathlib import Path


LIBRARY_FLUX_FILE = Path(__file__).parent / "ssp_spec_flux_lines.npy"
LIBRARY_WAVE_FILE = Path(__file__).parent / "ssp_spec_wave.npy"
LIBRARY_AGE_FILE = Path(__file__).parent / "log_age.npy"
LIBRARY_METAL_FILE = Path(__file__).parent / "zlegend.npy"
Z_SOLAR_PADOVA = 0.019  # Solar metallicity value in Padova


class SPSLibraryData:
    def __init__(
        self,
        flux_file: str = LIBRARY_FLUX_FILE,
        wave_file: str = LIBRARY_WAVE_FILE,
        age_file: str = LIBRARY_AGE_FILE,
        metal_file: str = LIBRARY_METAL_FILE,
        Z_solar: np.float32 = Z_SOLAR_PADOVA,
    ):
        self._Z_solar = Z_solar
        self.flux: np.ndarray = _load_fsps_flux(flux_file)
        self.wave: np.ndarray = _load_fsps_wave(wave_file)
        self.age: np.ndarray = _load_fsps_age(age_file)
        self.metallicity: np.ndarray = _load_fsps_metallicity(metal_file, Z_solar)


def _load_fsps_flux(flux_file) -> tuple:  # Fluxes, wavelengths
    spec_flux = np.load(flux_file)
    return spec_flux


def _load_fsps_wave(wave_file):
    return np.load(wave_file)


def _load_fsps_age(age_file):
    log_age_gyr = np.load(age_file) - 9
    age_fsps_gyr = 10**log_age_gyr
    return age_fsps_gyr


def _load_fsps_metallicity(metal_file, z_solar):
    Z_legend = np.load(metal_file)
    Z_padova_fsps = Z_legend / z_solar
    return Z_padova_fsps

    # @property
    # def flux(self):
    #     if self._spectral_library_flux is None:
    #         self._spectral_library_flux = self._load_fsps_flux()
    #     return self._spectral_library_flux

    # @property
    # def wave(self):
    #     if self._spectral_library_wave is None:
    #         self._spectral_library_wave = self._load_fsps_wave()
    #     return self._spectral_library_wave

    # @property
    # def age(self):
    #     if self._spectral_library_age is None:
    #         self._spectral_library_age = self._load_fsps_age()
    #     return self._spectral_library_age

    # @property
    # def metallicity(self):
    #     if self._spectral_library_metal is None:
    #         self._spectral_library_metal = self._load_fsps_metallicity()
    #     return self._spectral_library_metal

    # def load_fsps_age(
    #     age_fileIn: str = LIBRARY_AGE_FILE,  # Input age file of the stellar spectra library
    # ) -> np.array:  # Age in Gyr

    #     # log_age_gyr = np.load(os.path.join(dirIn, "log_age.npy")) - 9
    #     log_age_gyr = np.load(age_fileIn) - 9
    #     age_fsps_gyr = 10**log_age_gyr
    #     ## (age is in 1/H0 units)
    #     return age_fsps_gyr

    # def load_fsps_metallicity(
    #     metal_fileIn: str = LIBRARY_METAL_FILE,  # Input metallicity file of the stellar spectra library
    #     Z_solar: np.float32 = Z_SOLAR_PADOVA,  # Solar metallicity in Padova
    # ) -> np.array:  # Metallicity values in Z/Z_solar units
    #     Z_legend = np.load(metal_fileIn)
    #     Z_padova_fsps = Z_legend / Z_solar
    #     return Z_padova_fsps

    # def load_fsps_age_metallicity(
    #     age_fileIn: str = LIBRARY_AGE_FILE,  # Input metallicity file of the stellar spectra library
    #     metal_fileIn: str = LIBRARY_METAL_FILE,  # Input age file of the stellar spectra library
    #     Z_solar: np.float32 = Z_SOLAR_PADOVA,  # Solar metallicity
    # ) -> tuple:  # Age in Gyr, Metallicity in Z/Z_sun
    #     age_fsps_gyr = load_fsps_age(age_fileIn)
    #     metallicity = load_fsps_metallicity(metal_fileIn, Z_solar)
    #     return age_fsps_gyr, metallicity
