import numpy as np
from .ssp_interpolation import spec_ssp_lookup, spec_ssp_lookup_fast
from .data import SPSLibraryData


def total_luminosity(
    spec_flux_ssp: np.ndarray, spec_wave: np.array  # SSP SEDs  # SED Wavelength
) -> np.float32:  # Luminosity
    flux_proxy = np.trapz(spec_flux_ssp, spec_wave, axis=1)
    return flux_proxy


def calc_stellar_flux(
    stellar_age: float,
    stellar_metallicity: float,
    stellar_mass: float,
    sps_library: SPSLibraryData,
):
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
    spec_flux_ssp = spec_ssp_lookup(
        stellar_age,
        stellar_metallicity,
        stellar_mass,
        sps_library.age,
        sps_library.metallicity,
        sps_library.flux,
    )
    return spec_flux_ssp


def calc_galaxy_flux(
    stellar_ages: np.ndarray,  # in Gyr
    stellar_metallicities: np.ndarray,  # absolute metallicity
    stellar_masses: np.ndarray,  # in Msun/h
    sps_library: SPSLibraryData,
    *,
    method="fast"
) -> (
    tuple
):  # SED wavelength (A), individual SSP SEDs (Lsun/A), CSP SED (Lsun/A), Luminosity, Galaxy stellar mass (Msun)

    if method == "fast":
        fluxes = spec_ssp_lookup_fast(
            stellar_ages,
            stellar_metallicities,
            stellar_masses,
            sps_library.age,
            sps_library.metallicity,
            sps_library.flux,
        )
        total_flux = np.sum(fluxes, axis=0)
    elif method == "individual":
        nstars = len(stellar_ages)
        nspec = len(sps_library.wave)
        total_flux = np.zeros(nspec, dtype=np.float64)

        for i in range(nstars):
            total_flux += spec_ssp_lookup(
                stellar_ages[i],
                stellar_metallicities[i],
                stellar_masses[i],
                sps_library.age,
                sps_library.metallicity,
                sps_library.flux,
            )
    else:
        raise ValueError("method must be either 'fast' or 'individual'")

    luminosty = total_luminosity(total_flux[np.newaxis, :], sps_library.wave)

    return total_flux, luminosty
