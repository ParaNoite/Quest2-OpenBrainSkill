"""Frequency-domain filters for brain signal preprocessing."""
from __future__ import annotations

from typing import List, Union

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt


def bandpass_filter(
    data: np.ndarray,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Args:
        data: Signal array of shape ``(n_channels, n_samples)`` or
            ``(n_epochs, n_channels, n_samples)``.
        l_freq: Lower passband edge (Hz).
        h_freq: Upper passband edge (Hz).
        sfreq: Sampling frequency (Hz).
        order: Filter order.

    Returns:
        Filtered signal array with the same shape as *data*.

    Raises:
        ValueError: If *l_freq* >= *h_freq*, or if either frequency is
            outside the valid Nyquist range.
    """
    nyq = sfreq / 2.0

    if l_freq <= 0:
        raise ValueError(f"l_freq must be > 0; got {l_freq}.")
    if l_freq >= nyq:
        raise ValueError(
            f"l_freq ({l_freq} Hz) must be < Nyquist frequency ({nyq} Hz)."
        )
    # Clamp h_freq to just below Nyquist
    h_freq = min(h_freq, nyq * 0.999)
    if l_freq >= h_freq:
        raise ValueError(f"l_freq ({l_freq}) must be < h_freq ({h_freq}).")

    low = l_freq / nyq
    high = h_freq / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def lowpass_filter(
    data: np.ndarray,
    h_freq: float,
    sfreq: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Args:
        data: Signal array.
        h_freq: Upper cutoff frequency (Hz).
        sfreq: Sampling frequency (Hz).
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = sfreq / 2.0
    # Clamp h_freq to just below Nyquist
    h_freq = min(h_freq, nyq * 0.999)
    sos = butter(order, h_freq / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def highpass_filter(
    data: np.ndarray,
    l_freq: float,
    sfreq: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Args:
        data: Signal array.
        l_freq: Lower cutoff frequency (Hz).
        sfreq: Sampling frequency (Hz).
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = sfreq / 2.0
    if l_freq <= 0:
        raise ValueError(f"l_freq must be > 0; got {l_freq}.")
    sos = butter(order, l_freq / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    freqs: Union[float, List[float]],
    sfreq: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply one or more notch (band-stop) filters.

    Args:
        data: Signal array of shape ``(n_channels, n_samples)`` or
            ``(n_epochs, n_channels, n_samples)``.
        freqs: Frequency or list of frequencies to notch (Hz).
        sfreq: Sampling frequency (Hz).
        quality_factor: Q factor of the notch filter (higher = narrower notch).

    Returns:
        Filtered signal array.
    """
    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    result = data.copy()
    nyq = sfreq / 2.0
    for f in freqs:
        if f <= 0 or f >= nyq:
            continue  # Skip invalid frequencies silently
        b, a = iirnotch(f, quality_factor, fs=sfreq)
        # Convert to sos for numerical stability
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)
        result = sosfiltfilt(sos, result, axis=-1)

    return result
