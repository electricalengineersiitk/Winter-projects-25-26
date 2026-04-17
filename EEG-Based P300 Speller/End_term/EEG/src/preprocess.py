"""
preprocess.py
=============
EEG preprocessing pipeline: filtering, re-referencing, ICA artifact
removal, bad channel interpolation, and epoching.
"""

import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Raw signal cleaning
# ---------------------------------------------------------------------------

def load_raw(filepath: str, preload: bool = True) -> mne.io.BaseRaw:
    """Load a raw EEG file supported by MNE (FIF, EDF, BDF, GDF…)."""
    raw = mne.io.read_raw(filepath, preload=preload)
    log.info("Loaded: %s  |  %d ch  |  %.1f s  |  %.0f Hz",
             filepath, len(raw.ch_names), raw.times[-1], raw.info["sfreq"])
    return raw


def apply_bandpass(raw: mne.io.BaseRaw,
                   l_freq: float = 0.1,
                   h_freq: float = 30.0) -> mne.io.BaseRaw:
    """Bandpass filter in-place (default: 0.1–30 Hz for P300)."""
    log.info("Bandpass filter: %.2f – %.1f Hz", l_freq, h_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               fir_window="hamming", verbose=False)
    return raw


def apply_notch(raw: mne.io.BaseRaw,
                freqs=(50.0,)) -> mne.io.BaseRaw:
    """Notch filter to remove power-line noise (50 Hz in India)."""
    log.info("Notch filter at %s Hz", freqs)
    raw.notch_filter(freqs=list(freqs), verbose=False)
    return raw


def set_reference(raw: mne.io.BaseRaw,
                  ref: str = "average") -> mne.io.BaseRaw:
    """Re-reference EEG channels."""
    log.info("Setting reference: %s", ref)
    raw.set_eeg_reference(ref_channels=ref, verbose=False)
    return raw


def interpolate_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Interpolate marked bad channels using spherical splines."""
    if raw.info["bads"]:
        log.info("Interpolating bad channels: %s", raw.info["bads"])
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        log.info("No bad channels to interpolate.")
    return raw


def run_ica(raw: mne.io.BaseRaw,
            n_components: int = 20,
            random_state: int = 42,
            max_iter: int = 800) -> mne.io.BaseRaw:
    """
    Fit ICA and automatically mark/remove eye-blink (EOG) components.
    Returns cleaned raw (in-place).
    """
    log.info("Running ICA with %d components …", n_components)
    ica = ICA(n_components=n_components,
              method="fastica",
              random_state=random_state,
              max_iter=max_iter)
    ica.fit(raw, verbose=False)

    # Try to find EOG components automatically
    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
    if eog_indices:
        log.info("ICA: marking EOG components %s", eog_indices)
        ica.exclude = eog_indices
    else:
        log.info("ICA: no EOG components found automatically.")

    ica.apply(raw, verbose=False)
    return raw


def preprocess_raw(filepath: str,
                   l_freq: float = 0.1,
                   h_freq: float = 30.0,
                   notch_freq: float = 50.0,
                   reference: str = "average",
                   run_ica_flag: bool = True,
                   n_ica_components: int = 20) -> mne.io.BaseRaw:
    """
    Full preprocessing pipeline (convenience wrapper).

    Parameters
    ----------
    filepath       : path to raw EEG file
    l_freq, h_freq : bandpass limits
    notch_freq     : power-line frequency (50 Hz for India)
    reference      : EEG reference scheme
    run_ica_flag   : whether to apply ICA artefact rejection
    n_ica_components: number of ICA components

    Returns
    -------
    Cleaned MNE Raw object
    """
    raw = load_raw(filepath)
    raw = apply_bandpass(raw, l_freq=l_freq, h_freq=h_freq)
    raw = apply_notch(raw, freqs=(notch_freq,))
    raw = set_reference(raw, ref=reference)
    raw = interpolate_bad_channels(raw)
    if run_ica_flag:
        raw = run_ica(raw, n_components=n_ica_components)
    log.info("Preprocessing complete.")
    return raw


# ---------------------------------------------------------------------------
# 2. Epoching
# ---------------------------------------------------------------------------

def make_epochs(raw: mne.io.BaseRaw,
                tmin: float = -0.2,
                tmax: float = 0.8,
                baseline: tuple = (-0.2, 0.0),
                event_id: dict | None = None,
                decim: int = 1) -> mne.Epochs:
    """
    Extract stimulus-locked epochs.

    Parameters
    ----------
    raw      : preprocessed Raw
    tmin     : epoch start relative to event (s)
    tmax     : epoch end relative to event (s)
    baseline : baseline correction window (s)
    event_id : dict mapping event label → int code (None = auto-detect)
    decim    : temporal decimation factor (speeds up processing)

    Returns
    -------
    MNE Epochs object
    """
    events, detected_event_id = mne.events_from_annotations(raw, verbose=False)
    if event_id is None:
        event_id = detected_event_id

    log.info("Detected events: %s", detected_event_id)
    log.info("Creating epochs: [%.2f, %.2f] s, baseline %s, decim=%d",
             tmin, tmax, baseline, decim)

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        decim=decim,
        verbose=False
    )
    log.info("Epochs created: %s", epochs)
    return epochs


def auto_reject_epochs(epochs: mne.Epochs) -> mne.Epochs:
    """
    Automatic bad-epoch rejection using the autoreject library.
    Falls back gracefully if autoreject is not installed.
    """
    try:
        from autoreject import AutoReject
        log.info("Running AutoReject …")
        ar = AutoReject(random_state=42, verbose=False)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        n_dropped = reject_log.bad_epochs.sum()
        log.info("AutoReject: dropped %d / %d epochs", n_dropped, len(epochs))
        return epochs_clean
    except ImportError:
        log.warning("autoreject not installed; skipping automatic rejection.")
        return epochs
