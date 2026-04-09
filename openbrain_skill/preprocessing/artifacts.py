"""Artifact detection and removal for brain signal preprocessing."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Re-reference signals to the common average of all channels.

    Subtracts the mean across channels from each channel at each time
    point, reducing common-mode noise and volume-conduction artefacts.

    Args:
        data: Signal array of shape ``(n_channels, n_samples)`` or
            ``(n_epochs, n_channels, n_samples)``.

    Returns:
        Re-referenced array with the same shape.
    """
    # Subtract channel-mean along the channel axis (axis -2)
    return data - data.mean(axis=-2, keepdims=True)


def threshold_artifact_rejection(
    data: np.ndarray,
    threshold: float = 100.0,
    replacement: str = "zero",
) -> np.ndarray:
    """Reject epochs or time points exceeding an amplitude threshold.

    For epoched data ``(n_epochs, n_channels, n_samples)``, the entire
    epoch is zeroed (or NaN-filled) when any channel in that epoch
    exceeds *threshold*.

    For continuous data ``(n_channels, n_samples)``, individual samples
    exceeding the threshold are replaced.

    Args:
        data: Signal array (2-D or 3-D).
        threshold: Peak absolute amplitude threshold (in signal units).
        replacement: ``"zero"`` to replace with zeros, ``"nan"`` to replace
            with NaN.

    Returns:
        Array with the same shape; artefact-contaminated data replaced.
    """
    fill = 0.0 if replacement == "zero" else float("nan")
    result = data.copy().astype(float)

    if result.ndim == 3:
        # Epoched: (n_epochs, n_channels, n_samples)
        bad_epochs = np.any(np.abs(result) > threshold, axis=(1, 2))
        result[bad_epochs] = fill
    elif result.ndim == 2:
        # Continuous: (n_channels, n_samples)
        bad_samples = np.any(np.abs(result) > threshold, axis=0)
        result[:, bad_samples] = fill
    else:
        raise ValueError(f"Expected 2-D or 3-D data; got {result.ndim}-D.")

    return result


def apply_ica_simple(
    data: np.ndarray,
    n_components: Optional[int] = None,
    artifact_indices: Optional[list] = None,
) -> np.ndarray:
    """Apply a simplified PCA/ICA-like artifact removal via SVD.

    This is a lightweight approximation of ICA intended for demonstration
    and teaching purposes. It uses SVD to decompose the signal and zeros
    out the specified components before reconstructing.

    For production use, employ MNE-Python's full ICA pipeline.

    Args:
        data: 2-D signal array ``(n_channels, n_samples)``.
        n_components: Number of components to retain. Defaults to
            ``min(n_channels, 20)``.
        artifact_indices: Component indices to remove (zero-based).
            Defaults to ``[0]`` (first component, often captures large
            artifacts such as EOG blinks).

    Returns:
        Reconstructed signal array with the same shape as *data*.

    Raises:
        ValueError: If *data* is not 2-D.
    """
    if data.ndim != 2:
        raise ValueError(
            f"apply_ica_simple expects 2-D data (n_channels, n_samples); "
            f"got {data.ndim}-D."
        )

    n_channels = data.shape[0]
    n_components = n_components or min(n_channels, 20)
    artifact_indices = artifact_indices or [0]

    # Centre data
    mean = data.mean(axis=1, keepdims=True)
    data_centred = data - mean

    # Thin SVD
    U, s, Vt = np.linalg.svd(data_centred, full_matrices=False)

    # Zero out artifact components
    s_clean = s.copy()
    for idx in artifact_indices:
        if idx < len(s_clean):
            s_clean[idx] = 0.0

    # Reconstruct
    reconstructed = (U * s_clean) @ Vt + mean

    return reconstructed


def motion_correction_tddr(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Temporal Derivative Distribution Repair (TDDR) for fNIRS motion artifacts.

    TDDR is a robust motion artifact correction method that operates on
    the temporal derivatives of the signal.

    Reference: Fishburn et al. (2019). Temporal Derivative Distribution
    Repair (TDDR): A Motion Correction Method for fNIRS.

    Args:
        data: Signal array ``(n_channels, n_samples)``.
        sfreq: Sampling frequency (Hz).

    Returns:
        Motion-corrected signal with the same shape.
    """
    result = np.zeros_like(data)

    for ch in range(data.shape[0]):
        signal = data[ch].copy()

        # Compute temporal derivative
        deriv = np.diff(signal, prepend=signal[0])

        # Robust estimate of derivative distribution (median absolute deviation)
        mad = np.median(np.abs(deriv - np.median(deriv)))
        sigma = mad / 0.6745  # convert MAD to std estimate

        if sigma < 1e-10:
            result[ch] = signal
            continue

        # Tukey biweight weights
        k = 4.685
        u = deriv / (k * sigma)
        weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)

        # Weighted reconstruction via cumulative sum of weighted derivative
        corrected_deriv = deriv * weights
        result[ch] = np.cumsum(corrected_deriv) + signal[0]

    return result
