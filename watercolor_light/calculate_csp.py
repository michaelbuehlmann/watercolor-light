import numpy as np
from .data import SPSLibraryData
from scipy.interpolate import RegularGridInterpolator


def total_luminosity(
    spec_flux_ssp: np.ndarray, spec_wave: np.array  # SSP SEDs  # SED Wavelength
) -> np.float32:  # Luminosity
    flux_proxy = np.trapz(spec_flux_ssp, spec_wave, axis=1)
    return flux_proxy


def spec_ssp_lookup(
    age_stars: np.ndarray = None,  # Age of the HACC stellar particle
    metal_stars: np.ndarray = None,  # Metallicity of the stellar particle
    age_fsps: np.ndarray = None,  # Ages in SPS library
    z_padova_fsps: np.ndarray = None,  # Metallicities in SPS library
    spec_flux: np.array = None,  # Stellar library SEDs
    *,
    interpolation_method: str = "linear",
) -> np.ndarray:  # SSP Luminosity (Lsun/A)
    """Interpolation for finding the closest SPS entry to HACC SSP

    Arguments:
    ----------
    age_stars : np.ndarray
        Age of the HACC stellar particles in Gyr
    metal_stars : np.float32
        Metallicity of the HACC stellar particles in Zsun units
    age_fsps : np.ndarray
        Ages in SPS library in Gyr
    z_padova_fsps : np.ndarray
        Metallicities in SPS library in Zsun units
    spec_flux : np.array
        Stellar library SEDs in Lsun/A

    Returns:
    --------
    np.ndarray
        SSP flux in Lsun/A/Msun
    """
    interpolator = RegularGridInterpolator(
        (z_padova_fsps, age_fsps),
        spec_flux,
        bounds_error=False,
        fill_value=None,  # None will extrapolate beyond bounds
        method=interpolation_method,
    )
    spec_flux_stars = interpolator(np.array([metal_stars, age_stars]).T)

    return spec_flux_stars


def calc_stellar_luminosity(
    stellar_age: float,
    stellar_metallicity: float,
    sps_library: SPSLibraryData,
):
    """Calculate the luminosity per Msun of a single stellar population (SSP) with given
    age and metallicity.

    Arguments:
    ----------
    stellar_age : float
        Age of the SSP in Gyr
    stellar_metallicity : float
        Metallicity of the SSP in Zsun units
    stellar_mass : float
        Mass of the SSP in Msun
    sps_library : SPSLibraryData
        Stellar population synthesis library data

    Returns:
    --------
    np.ndarray
        SSP luminosity in Lsun/A/Msun
    """
    spec_flux_ssp = spec_ssp_lookup(
        stellar_age,
        stellar_metallicity,
        sps_library.age,
        sps_library.metallicity,
        sps_library.flux,
    )
    return spec_flux_ssp


def calc_galaxy_luminosity(
    stellar_ages: np.ndarray,  # in Gyr
    stellar_metallicities: np.ndarray,  # absolute metallicity
    stellar_masses: np.ndarray,  # in Msun/h
    sps_library: SPSLibraryData,
) -> np.ndarray:
    """Calculate the flux of a single stellar population (SSP) with given age, metallicity, and mass.

    Arguments:
    ----------
    stellar_age : float
        Age of the SSP in Gyr
    stellar_metallicity : float
        Metallicity of the SSP in Zsun units
    stellar_mass : float
        Mass of the SSP in Msun
    sps_library : SPSLibraryData
        Stellar population synthesis library data

    Returns:
    --------
    np.ndarray
        SSP flux in Lsun/A
    """
    luminosities = spec_ssp_lookup(
        stellar_ages,
        stellar_metallicities,
        sps_library.age,
        sps_library.metallicity,
        sps_library.flux,
    )
    luminosities *= stellar_masses[:, np.newaxis]
    combined_luminosity = np.sum(luminosities, axis=0)
    return combined_luminosity


def luminosity_to_flux(luminosity: np.ndarray) -> np.ndarray:
    """Convert luminosity to flux, assuming 10pc distance

    Arguments:
    ----------
    luminosity : np.ndarray
        Luminosity in Lsun/A

    Returns:
    --------
    np.ndarray
        Flux in Lsun/A/pc^2
    """
    flux = luminosity / (4 * np.pi * 10**2)
    return flux
