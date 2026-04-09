"""Epoching and segmentation utilities."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def epoch_data(
    data: np.ndarray,
    events: np.ndarray,
    tmin: float,
    tmax: float,
    sfreq: float,
    event_codes: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Epoch continuous data around event onsets.

    Args:
        data: Continuous signal ``(n_channels, n_samples)``.
        events: Event array ``(n_events, 3)`` with columns
            ``[sample_index, 0, event_code]``.
        tmin: Epoch start time relative to event (seconds; may be negative).
        tmax: Epoch end time relative to event (seconds).
        sfreq: Sampling frequency (Hz).
        event_codes: Optional list of event codes to include. If *None*,
            all codes are included.

    Returns:
        Tuple of:
        - **epochs**: Array of shape ``(n_epochs, n_channels, n_epoch_samples)``
        - **codes**: 1-D array of event codes for each epoch.
    """
    if data.ndim != 2:
        raise ValueError(f"epoch_data expects 2-D data; got {data.ndim}-D.")
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("events must be a (n_events, 3) array.")

    n_channels, n_samples = data.shape
    onset_samples = int(round(tmin * sfreq))
    offset_samples = int(round(tmax * sfreq))
    epoch_len = offset_samples - onset_samples

    epochs_list = []
    codes_list = []

    for sample_idx, _, code in events:
        if event_codes is not None and code not in event_codes:
            continue
        start = int(sample_idx) + onset_samples
        end = start + epoch_len
        if start < 0 or end > n_samples:
            continue
        epochs_list.append(data[:, start:end])
        codes_list.append(int(code))

    if not epochs_list:
        return (
            np.empty((0, n_channels, epoch_len), dtype=data.dtype),
            np.empty(0, dtype=int),
        )

    return np.stack(epochs_list, axis=0), np.array(codes_list, dtype=int)


def baseline_correct(
    epochs: np.ndarray,
    baseline: Tuple[float, float],
    sfreq: float,
    tmin: float,
) -> np.ndarray:
    """Apply baseline correction by subtracting the mean of a pre-stimulus interval.

    Args:
        epochs: Epoched data ``(n_epochs, n_channels, n_samples)``.
        baseline: Tuple ``(start, end)`` in seconds relative to epoch onset
            (same reference as *tmin*). Use ``(None, 0)`` for onset to zero.
        sfreq: Sampling frequency (Hz).
        tmin: Epoch start time (seconds) used to convert baseline to samples.

    Returns:
        Baseline-corrected epochs with the same shape.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"baseline_correct expects 3-D epoched data; got {epochs.ndim}-D."
        )

    bl_start, bl_end = baseline
    if bl_start is None:
        bl_start = tmin

    bl_start_sample = int(round((bl_start - tmin) * sfreq))
    bl_end_sample = int(round((bl_end - tmin) * sfreq))
    bl_start_sample = max(0, bl_start_sample)
    bl_end_sample = max(bl_start_sample + 1, bl_end_sample)

    baseline_mean = epochs[:, :, bl_start_sample:bl_end_sample].mean(
        axis=-1, keepdims=True
    )
    return epochs - baseline_mean


def segment_into_windows(
    data: np.ndarray,
    window_duration: float,
    overlap: float,
    sfreq: float,
) -> np.ndarray:
    """Segment a continuous signal into overlapping fixed-length windows.

    Args:
        data: Continuous signal ``(n_channels, n_samples)`` or
            ``(n_samples,)`` for a single channel.
        window_duration: Window length in seconds.
        overlap: Fraction of window overlap ``[0, 1)``.
        sfreq: Sampling frequency (Hz).

    Returns:
        Array of shape ``(n_windows, n_channels, win_samples)`` (or
        ``(n_windows, win_samples)`` for single-channel input).

    Raises:
        ValueError: If *overlap* is not in ``[0, 1)``.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}.")

    win_samples = int(window_duration * sfreq)
    step = int(win_samples * (1.0 - overlap))
    step = max(1, step)

    if data.ndim == 1:
        n_samples = data.shape[0]
        starts = range(0, n_samples - win_samples + 1, step)
        windows = np.stack([data[s : s + win_samples] for s in starts])
        return windows

    n_channels, n_samples = data.shape
    starts = range(0, n_samples - win_samples + 1, step)
    windows = np.stack([data[:, s : s + win_samples] for s in starts])
    return windows
