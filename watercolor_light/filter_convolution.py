import speclite.filters
import numpy as np
import astropy.units as u
from typing import Union

filters = {"sdss": speclite.filters.load_filters("sdss2010-*")}


def photometry_from_flux(
    wave: np.ndarray,
    flux: np.ndarray,
    filter: Union[str,  speclite.filters.FilterSequence],
):
    """Calculate the photometry of a given flux in a given filter.

    Arguments:
    ----------
    wave : np.ndarray
        Wavelength in Angstrom. Shape (n_wave,)
    flux : np.ndarray
        Flux in Lsun/A/pc^2. Shape (nwave,) or (nobjects, nwave). If 2d, axis=0
        corresponds to the objects, and axis=1 to the wavelengths.
    filter_name : str
        Name of the filter

    Returns:
    --------
    dict[str, np.ndarray]
        Photometry in AB mag
    """
    if isinstance(filter, str):
        if filter not in filters:
            raise ValueError(
                f"Filter {filter} not available. Available filters are: {list(filters.keys())}"
            )
        filter = filters[filter]
    else:
        assert isinstance(filter, speclite.filters.FilterSequence)
    flux = flux * u.Lsun / u.Angstrom / u.pc**2
    wave = wave * u.Angstrom
    astropy_mags = filter.get_ab_magnitudes(flux, wave)
    return {k: m.data for k, m in astropy_mags.items()}
