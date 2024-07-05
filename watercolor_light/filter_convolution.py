import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class Photometry:
    """Photometry results

    Attributes:
    -----------
    appmag_ext : np.ndarray
        Apparent magnitudes
    band_fluxes : np.ndarray
        Fluxes in mJy
    """

    band_magnitudes: np.ndarray
    band_fluxes: np.ndarray


def photometry_from_spectra(
    sed_um_wave: np.array = None,  # SED wavelengths (in microns)
    sed_mJy_flux: np.array = None,  # SED fluxes (in mJy)
    bandpass_wavs: np.array = None,  # Bandpass wavelenths
    bandpass_vals: np.array = None,  # Bandspass values
    interp_kind: str = "linear",  # Interpolation type
) -> tuple:  # Fluxes, Apparent magnitudes, Band fluxes
    """Calculate photometry from spectra

    Arguments:
    ----------
    sed_um_wave : np.array
        SED wavelengths (in microns)
    sed_mJy_flux : np.array
        SED fluxes (in mJy)
    bandpass_wavs : np.array
        Bandpass wavelenths
    bandpass_vals : np.array
        Bandspass values (transmission)
    interp_kind : str
        Interpolation type passed to scipy.interpolate.interp1d

    Returns:
    --------
    Photometry
        Photometry results, including apparent magnitudes and band fluxes in mJy
    """
    nbands = len(bandpass_wavs)
    sed_interp = interp1d(
        sed_um_wave, sed_mJy_flux, kind=interp_kind, bounds_error=False, fill_value=0.0
    )

    band_fluxes = np.zeros(nbands, dtype=np.float32)

    for b, bandpass_wav in enumerate(bandpass_wavs):
        # fluxes in mJy
        band_fluxes[b] = np.dot(bandpass_vals[b], sed_interp(bandpass_wav))

    flux = 1e3 * band_fluxes  # uJy
    appmag_ext = -2.5 * np.log10(flux) + 23.9

    return Photometry(appmag_ext, band_fluxes)
