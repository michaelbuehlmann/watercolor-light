import pickle
from pathlib import Path
import numpy as np
from dataclasses import dataclass

FILTERS_LSST = Path(__file__).parent / "LSST.pickle"
FILTERS_SPHEREX = Path(__file__).parent / "SPHEREx.pickle"
FILTERS_COSMOS = Path(__file__).parent / "COSMOS.pickle"
FILTERS_WISE = Path(__file__).parent / "WISE.pickle"


def _clip_bandpass_values(
    bandpass_wavs: np.float32 = None,  # Bandpass wavelengths
    bandpass_vals: np.float32 = None,  # Bandpasses
) -> tuple:  # Clipped bandpass wavelengths, clipped bandpass values

    all_clip_bandpass_wav, all_clip_bandpass_vals = [], []

    for b in range(len(bandpass_wavs)):
        nonz_bandpass_val = bandpass_vals[b] > 0
        clip_bandpass_wav = bandpass_wavs[b][nonz_bandpass_val]
        clip_bandpass_vals = bandpass_vals[b][nonz_bandpass_val]
        all_clip_bandpass_wav.append(clip_bandpass_wav)
        all_clip_bandpass_vals.append(clip_bandpass_vals)

    return all_clip_bandpass_wav, all_clip_bandpass_vals


@dataclass(frozen=True)
class Filter:
    central_wavelengths: np.ndarray
    bandpass_wavelengths: list[np.ndarray]
    bandpass_values: list[np.ndarray]
    bandpass_names: list[str]

    @classmethod
    def from_pickle(cls, path: str, clip_support: bool = True):
        with open(path, "rb") as f:
            central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = (
                pickle.load(f)
            )

        if clip_support:
            bandpass_wavs, bandpass_vals = _clip_bandpass_values(
                bandpass_wavs, bandpass_vals
            )

        s = np.argsort(central_wavelengths)
        return cls(
            central_wavelengths=np.array(central_wavelengths)[s],
            bandpass_wavelengths=list([np.array(bandpass_wavs[_s]) for _s in s]),
            bandpass_values=list([np.array(bandpass_vals[_s]) for _s in s]),
            bandpass_names=list([bandpass_names[_s] for _s in s]),
        )


class Filters:
    def __init__(self, clip_support: bool = True):
        self._clip_support = clip_support
        self._filters: dict[str, Filter] = {}
        self._filters["LSST"] = Filter.from_pickle(FILTERS_LSST, clip_support)
        self._filters["SPHEREx"] = Filter.from_pickle(FILTERS_SPHEREX, clip_support)
        self._filters["COSMOS"] = Filter.from_pickle(FILTERS_COSMOS, clip_support)
        self._filters["WISE"] = Filter.from_pickle(FILTERS_WISE, clip_support)

    def __getitem__(self, key) -> Filter:
        return self._filters[key]

    def __setitem__(self, key, filter) -> None:
        assert isinstance(filter, Filter)
        self._filters[key] = filter

    def add(self, name, filter) -> None:
        if isinstance(filter, str):
            filter = Filter.from_pickle(filter, self._clip_support)
        self._filters[name] = filter
