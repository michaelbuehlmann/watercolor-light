import numpy as np
from scipy.interpolate import RegularGridInterpolator


def spec_ssp_lookup_nearest(
    age_hydro_i: np.float32 = None,  # Age of the HACC stellar particle
    metal_hydro_i: np.float32 = None,  # Metallicity of the stellar particle
    mass_hydro_i: np.float32 = None,  # Mass of the stellar particle
    age_fsps: np.float32 = None,  # Ages in SPS library
    z_padova_fsps: np.float32 = None,  # Metallicities in SPS library
    spec_flux: np.array = None,  # Stellar library SEDs
    spec_wave: np.array = None,  # Stellar library wavelengths
) -> tuple:  # SSP Wavelength (A), SSP Luminosity (Lsun/A)
    """
    Lookup table for finding the closest SPS entry to HACC SSP
    """
    # https://ned.ipac.caltech.edu/level5/Sept14/Conroy/Conroy2.html

    age_index = np.argmin(np.abs(age_fsps - age_hydro_i))
    met_index = np.argmin(np.abs(z_padova_fsps - metal_hydro_i))
    # print(age_index, met_index)
    # spec_flux_i = 1e10*mass_hydro_i*spec_flux[met_index, age_index]  ## Not sure where 1e10 factor is from. Will ignore for now
    spec_flux_i = mass_hydro_i * spec_flux[met_index, age_index]

    return spec_wave, spec_flux_i


def spec_ssp_lookup(
    age_hydro_i: np.float32,  # Age of the HACC stellar particle
    metal_hydro_i: np.float32,  # Metallicity of the stellar particle
    mass_hydro_i: np.float32,  # Mass of the stellar particle
    age_fsps: np.ndarray,  # Ages in SPS library
    z_padova_fsps: np.ndarray,  # Metallicities in SPS library
    spec_flux: np.array,  # Stellar library SEDs
) -> np.ndarray:  # SSP Luminosity (Lsun/A)
    """Interpolation for finding the closest SPS entry to HACC SSP

    Arguments:
    ----------
    age_hydro_i : np.float32
        Age of the HACC stellar particle in Gyr
    metal_hydro_i : np.float32
        Metallicity of the HACC stellar particle in Zsun units
    mass_hydro_i : np.float32
        Mass of the HACC stellar particle in Msun
    age_fsps : np.ndarray
        Ages in SPS library in Gyr
    z_padova_fsps : np.ndarray
        Metallicities in SPS library in Zsun units
    spec_flux : np.array
        Stellar library SEDs in Lsun/A

    Returns:
    --------
    np.ndarray
        SSP flux in Lsun/A
    """
    # https://ned.ipac.caltech.edu/level5/Sept14/Conroy/Conroy2.html

    age_diff = np.abs(age_fsps - age_hydro_i)
    metal_diff = np.abs(z_padova_fsps - metal_hydro_i)

    age_index = np.argsort(age_diff)
    met_index = np.argsort(metal_diff)

    age_min = np.min([age_index[1], age_index[0]])
    met_min = np.min([met_index[1], met_index[0]])

    age_diff_grid = np.abs(age_fsps[age_min + 1] - age_fsps[age_min])
    metal_diff_grid = np.abs(z_padova_fsps[met_min + 1] - z_padova_fsps[met_min])
    diff_denom = age_diff_grid * metal_diff_grid

    age_0 = np.abs(age_fsps[age_min] - age_hydro_i)
    met_0 = np.abs(z_padova_fsps[met_min] - metal_hydro_i)

    spec_flux_00 = spec_flux[met_min, age_min]
    spec_flux_01 = spec_flux[met_min, age_min + 1]
    spec_flux_10 = spec_flux[met_min + 1, age_min]
    spec_flux_11 = spec_flux[met_min + 1, age_min + 1]

    w00 = (1 - age_0) * (1 - met_0)
    w01 = (1 - age_0) * met_0
    w10 = age_0 * (1 - met_0)
    w11 = age_0 * met_0

    w00 = w00 / diff_denom
    w01 = w01 / diff_denom
    w10 = w10 / diff_denom
    w11 = w11 / diff_denom

    spec_flux_i = (
        (w00 * spec_flux_00)
        + (w01 * spec_flux_01)
        + (w10 * spec_flux_10)
        + (w11 * spec_flux_11)
    )

    spec_flux_i_mass_weighted = mass_hydro_i * spec_flux_i
    ## Not sure where 1e10 factor was from. Will ignore for now

    return spec_flux_i_mass_weighted


def spec_ssp_lookup_fast(
    age_stars: np.ndarray = None,  # Age of the HACC stellar particle
    metal_stars: np.ndarray = None,  # Metallicity of the stellar particle
    mass_stars: np.ndarray = None,  # Mass of the stellar particle
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
    mass_stars : np.ndarray
        Mass of the HACC stellar particles in Msun
    age_fsps : np.ndarray
        Ages in SPS library in Gyr
    z_padova_fsps : np.ndarray
        Metallicities in SPS library in Zsun units
    spec_flux : np.array
        Stellar library SEDs in Lsun/A

    Returns:
    --------
    np.ndarray
        SSP flux in Lsun/A
    """
    # https://ned.ipac.caltech.edu/level5/Sept14/Conroy/Conroy2.html

    interpolator = RegularGridInterpolator(
        (z_padova_fsps, age_fsps),
        spec_flux,
        bounds_error=False,
        fill_value=None,  # None will extrapolate beyond bounds
        method=interpolation_method,
    )
    spec_flux_stars = interpolator(np.array([metal_stars, age_stars]).T)
    spec_flux_stars *= mass_stars[:, np.newaxis]

    return spec_flux_stars
