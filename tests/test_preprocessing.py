"""Tests for preprocessing functions."""
import numpy as np
import pytest

from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter, lowpass_filter
from openbrain_skill.preprocessing.artifacts import (
    common_average_reference,
    threshold_artifact_rejection,
    apply_ica_simple,
)
from openbrain_skill.preprocessing.epoching import (
    epoch_data,
    baseline_correct,
    segment_into_windows,
)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape_preserved(self):
        data = np.random.randn(8, 512)
        result = bandpass_filter(data, l_freq=1.0, h_freq=40.0, sfreq=128.0)
        assert result.shape == data.shape

    def test_high_frequency_removed(self):
        """A 60 Hz signal should be attenuated by a 1–40 Hz bandpass."""
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        signal_60hz = np.sin(2 * np.pi * 60 * t)
        data = np.stack([signal_60hz])  # (1, 1024)
        filtered = bandpass_filter(data, l_freq=1.0, h_freq=40.0, sfreq=sfreq)
        # Power of filtered 60 Hz should be much less than original
        assert filtered.std() < signal_60hz.std() * 0.5

    def test_alpha_band_preserved(self):
        """A 10 Hz signal should pass through a 1–40 Hz bandpass."""
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        alpha = np.sin(2 * np.pi * 10 * t)
        data = np.stack([alpha])
        filtered = bandpass_filter(data, l_freq=1.0, h_freq=40.0, sfreq=sfreq)
        assert filtered.std() > alpha.std() * 0.5

    def test_invalid_l_freq_raises(self):
        with pytest.raises(ValueError):
            bandpass_filter(np.ones((2, 100)), l_freq=0.0, h_freq=10.0, sfreq=64.0)

    def test_h_freq_clamped_to_nyquist(self):
        """h_freq above Nyquist should be silently clamped (not raise)."""
        data = np.random.randn(2, 256)
        # sfreq=64 → Nyquist=32; h_freq=40 would be above Nyquist → clamped
        result = bandpass_filter(data, l_freq=1.0, h_freq=40.0, sfreq=64.0)
        assert result.shape == data.shape

    def test_l_freq_gte_h_freq_raises(self):
        with pytest.raises(ValueError):
            bandpass_filter(np.ones((2, 100)), l_freq=30.0, h_freq=10.0, sfreq=128.0)

    def test_3d_input(self):
        data = np.random.randn(10, 4, 256)
        result = bandpass_filter(data, l_freq=1.0, h_freq=20.0, sfreq=64.0)
        assert result.shape == data.shape


class TestNotchFilter:
    def test_output_shape_preserved(self):
        data = np.random.randn(4, 512)
        result = notch_filter(data, freqs=[50.0], sfreq=256.0)
        assert result.shape == data.shape

    def test_notch_attenuates_target_frequency(self):
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        noise = np.sin(2 * np.pi * 50 * t)
        data = np.stack([noise])
        filtered = notch_filter(data, freqs=[50.0], sfreq=sfreq)
        assert filtered.std() < noise.std() * 0.5

    def test_multiple_notch_freqs(self):
        data = np.random.randn(4, 512)
        result = notch_filter(data, freqs=[50.0, 100.0], sfreq=256.0)
        assert result.shape == data.shape

    def test_scalar_freq(self):
        data = np.random.randn(4, 512)
        result = notch_filter(data, freqs=50.0, sfreq=256.0)
        assert result.shape == data.shape

    def test_invalid_freq_skipped(self):
        data = np.random.randn(4, 512)
        # 0 Hz and Nyquist should be skipped gracefully
        result = notch_filter(data, freqs=[0.0, 128.0], sfreq=256.0)
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# Artifact removal
# ---------------------------------------------------------------------------

class TestCommonAverageReference:
    def test_output_shape(self):
        data = np.random.randn(8, 256)
        result = common_average_reference(data)
        assert result.shape == data.shape

    def test_channel_mean_is_zero(self):
        data = np.random.randn(8, 256)
        result = common_average_reference(data)
        # After CAR, mean across channels at each time point should be ~0
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)

    def test_3d_input(self):
        data = np.random.randn(10, 8, 256)
        result = common_average_reference(data)
        assert result.shape == data.shape
        # Mean across channels (axis 1) should be ~0
        np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-10)


class TestThresholdArtifactRejection:
    def test_output_shape_preserved(self):
        data = np.random.randn(4, 256)
        result = threshold_artifact_rejection(data, threshold=100.0)
        assert result.shape == data.shape

    def test_large_values_zeroed(self):
        data = np.zeros((4, 256))
        data[0, 10] = 200.0  # spike
        result = threshold_artifact_rejection(data, threshold=100.0)
        assert result[0, 10] == 0.0

    def test_epoched_bad_epoch_zeroed(self):
        data = np.zeros((5, 4, 256))
        data[2, 0, 50] = 500.0  # spike in epoch 2
        result = threshold_artifact_rejection(data, threshold=100.0)
        assert result[2].sum() == 0.0
        assert result[0].sum() == 0.0  # good epoch untouched

    def test_nan_replacement(self):
        data = np.zeros((4, 256))
        data[0, 10] = 200.0
        result = threshold_artifact_rejection(data, threshold=100.0, replacement="nan")
        assert np.isnan(result[0, 10])

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            threshold_artifact_rejection(np.zeros(100), threshold=1.0)


class TestApplyICASimple:
    def test_output_shape(self):
        data = np.random.randn(8, 512)
        result = apply_ica_simple(data)
        assert result.shape == data.shape

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            apply_ica_simple(np.zeros((4, 4, 100)))

    def test_artifact_component_removed(self):
        """Removing the dominant component should reduce overall variance."""
        data = np.random.randn(8, 512)
        # Add a very strong artifact on component 0 (dominant SVD component)
        data[0] += 1000.0
        result = apply_ica_simple(data, artifact_indices=[0])
        # Variance of first channel should be reduced
        assert result[0].var() < data[0].var()


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------

class TestEpochData:
    def setup_method(self):
        self.sfreq = 100.0
        self.n_channels = 4
        self.n_samples = 1000
        self.data = np.random.randn(self.n_channels, self.n_samples)
        # Two events at samples 200 and 600, codes 1 and 2
        self.events = np.array([[200, 0, 1], [600, 0, 2]])

    def test_basic_epoching(self):
        epochs, codes = epoch_data(
            self.data, self.events, tmin=0.0, tmax=1.0, sfreq=self.sfreq
        )
        assert epochs.shape == (2, self.n_channels, 100)
        np.testing.assert_array_equal(codes, [1, 2])

    def test_filter_by_event_code(self):
        epochs, codes = epoch_data(
            self.data, self.events, tmin=0.0, tmax=1.0,
            sfreq=self.sfreq, event_codes=[1]
        )
        assert epochs.shape[0] == 1
        assert codes[0] == 1

    def test_negative_tmin(self):
        events = np.array([[300, 0, 1]])
        epochs, codes = epoch_data(
            self.data, events, tmin=-0.5, tmax=0.5, sfreq=self.sfreq
        )
        assert epochs.shape == (1, self.n_channels, 100)

    def test_out_of_bounds_event_skipped(self):
        events = np.array([[5, 0, 1]])  # onset near start → can't go back 0.5 s
        epochs, codes = epoch_data(
            self.data, events, tmin=-1.0, tmax=1.0, sfreq=self.sfreq
        )
        assert epochs.shape[0] == 0

    def test_wrong_data_dim_raises(self):
        with pytest.raises(ValueError):
            epoch_data(np.zeros((4, 4, 100)), self.events, 0.0, 1.0, self.sfreq)

    def test_wrong_events_shape_raises(self):
        with pytest.raises(ValueError):
            epoch_data(self.data, np.array([1, 2, 3]), 0.0, 1.0, self.sfreq)


class TestBaselineCorrect:
    def test_output_shape(self):
        epochs = np.random.randn(10, 4, 200)
        result = baseline_correct(epochs, baseline=(-0.2, 0.0), sfreq=100.0, tmin=-0.2)
        assert result.shape == epochs.shape

    def test_baseline_mean_is_zero(self):
        epochs = np.ones((5, 4, 200)) * 5.0
        result = baseline_correct(epochs, baseline=(0.0, 0.2), sfreq=100.0, tmin=0.0)
        # After baseline correction the first 20 samples should be ~0
        np.testing.assert_allclose(result[:, :, :20].mean(axis=-1), 0.0, atol=1e-10)

    def test_non_3d_raises(self):
        with pytest.raises(ValueError):
            baseline_correct(np.zeros((4, 200)), (0, 0.1), 100.0, 0.0)


class TestSegmentIntoWindows:
    def test_basic_segmentation(self):
        data = np.random.randn(4, 1000)
        windows = segment_into_windows(data, window_duration=1.0, overlap=0.0, sfreq=100.0)
        assert windows.shape == (10, 4, 100)

    def test_overlapping_windows(self):
        data = np.random.randn(4, 1000)
        windows = segment_into_windows(data, window_duration=1.0, overlap=0.5, sfreq=100.0)
        # With 50% overlap on 1000 samples, 100-sample windows → ~19 windows
        assert windows.shape[0] > 10
        assert windows.shape[1] == 4
        assert windows.shape[2] == 100

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            segment_into_windows(np.zeros((4, 100)), 1.0, 1.5, 10.0)

    def test_1d_input(self):
        data = np.random.randn(1000)
        windows = segment_into_windows(data, window_duration=1.0, overlap=0.0, sfreq=100.0)
        assert windows.shape == (10, 100)
