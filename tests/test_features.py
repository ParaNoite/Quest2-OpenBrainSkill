"""Tests for feature extractors."""
import numpy as np
import pytest

from openbrain_skill.features.spectral import (
    BandPowerExtractor,
    PSDExtractor,
    SpectralEntropyExtractor,
    SNRExtractor,
)
from openbrain_skill.features.temporal import (
    MeanExtractor,
    SlopeExtractor,
    HjorthExtractor,
    ERPAmplitudeExtractor,
    ERPLatencyExtractor,
)
from openbrain_skill.features.spatial import CSPExtractor


SFREQ = 128.0
N_CHANNELS = 8
N_SAMPLES = 512
N_EPOCHS = 10


def make_data():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (N_CHANNELS, N_SAMPLES))


def make_epochs():
    rng = np.random.default_rng(1)
    return rng.normal(0, 1, (N_EPOCHS, N_CHANNELS, N_SAMPLES))


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

class TestBandPowerExtractor:
    def test_output_shape_2d(self):
        extractor = BandPowerExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_data())
        # (n_channels, n_bands)
        n_bands = len(extractor.bands)
        assert feats.shape == (N_CHANNELS, n_bands)

    def test_output_shape_3d(self):
        extractor = BandPowerExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_epochs())
        n_bands = len(extractor.bands)
        assert feats.shape == (N_EPOCHS, N_CHANNELS, n_bands)

    def test_custom_bands(self):
        bands = {"mu": (8, 12), "beta": (13, 30)}
        extractor = BandPowerExtractor(sfreq=SFREQ, bands=bands)
        feats = extractor.extract(make_data())
        assert feats.shape == (N_CHANNELS, 2)

    def test_band_power_nonnegative(self):
        extractor = BandPowerExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_data())
        assert np.all(feats >= 0)

    def test_feature_names(self):
        extractor = BandPowerExtractor(sfreq=SFREQ)
        names = extractor.feature_names
        assert len(names) == len(extractor.bands)
        assert all("band_power" in n for n in names)

    def test_invalid_ndim_raises(self):
        extractor = BandPowerExtractor(sfreq=SFREQ)
        with pytest.raises(ValueError):
            extractor.extract(np.zeros(N_SAMPLES))


class TestPSDExtractor:
    def test_output_shape_2d(self):
        extractor = PSDExtractor(sfreq=SFREQ)
        psd = extractor.extract(make_data())
        assert psd.ndim == 2
        assert psd.shape[0] == N_CHANNELS

    def test_output_shape_3d(self):
        extractor = PSDExtractor(sfreq=SFREQ)
        psd = extractor.extract(make_epochs())
        assert psd.ndim == 3
        assert psd.shape[:2] == (N_EPOCHS, N_CHANNELS)

    def test_psd_nonnegative(self):
        extractor = PSDExtractor(sfreq=SFREQ)
        psd = extractor.extract(make_data())
        assert np.all(psd >= 0)

    def test_get_frequencies(self):
        extractor = PSDExtractor(sfreq=SFREQ)
        freqs = extractor.get_frequencies(N_SAMPLES)
        assert freqs[0] == 0.0
        assert freqs[-1] <= SFREQ / 2


class TestSpectralEntropyExtractor:
    def test_output_shape_2d(self):
        extractor = SpectralEntropyExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_data())
        assert feats.shape == (N_CHANNELS, 1)

    def test_output_shape_3d(self):
        extractor = SpectralEntropyExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_epochs())
        assert feats.shape == (N_EPOCHS, N_CHANNELS, 1)

    def test_entropy_nonnegative(self):
        extractor = SpectralEntropyExtractor(sfreq=SFREQ)
        feats = extractor.extract(make_data())
        assert np.all(feats >= 0)

    def test_pure_tone_lower_entropy(self):
        """Pure sine should have lower spectral entropy than white noise."""
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        sine = np.sin(2 * np.pi * 10 * t)
        noise = np.random.default_rng(2).normal(0, 1, 1024)

        extractor = SpectralEntropyExtractor(sfreq=sfreq)
        h_sine = extractor.extract(sine.reshape(1, -1))[0, 0]
        h_noise = extractor.extract(noise.reshape(1, -1))[0, 0]
        assert h_sine < h_noise


class TestSNRExtractor:
    def test_output_shape(self):
        freqs = [7.5, 10.0]
        extractor = SNRExtractor(sfreq=SFREQ, target_frequencies=freqs)
        feats = extractor.extract(make_data())
        assert feats.shape == (N_CHANNELS, 2)

    def test_ssvep_signal_has_high_snr(self):
        """A pure sinusoid at the target frequency should have SNR > 1."""
        sfreq = 256.0
        target = 10.0
        t = np.arange(2048) / sfreq
        signal = np.sin(2 * np.pi * target * t)
        data = signal.reshape(1, -1)  # (1 channel, n_samples)

        extractor = SNRExtractor(sfreq=sfreq, target_frequencies=[target])
        snr = extractor.extract(data)
        assert snr[0, 0] > 1.0


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

class TestMeanExtractor:
    def test_output_shape_2d(self):
        feats = MeanExtractor().extract(make_data())
        assert feats.shape == (N_CHANNELS, 1)

    def test_output_shape_3d(self):
        feats = MeanExtractor().extract(make_epochs())
        assert feats.shape == (N_EPOCHS, N_CHANNELS, 1)

    def test_correct_mean(self):
        data = np.ones((4, 100)) * 3.0
        feats = MeanExtractor().extract(data)
        np.testing.assert_allclose(feats, 3.0)


class TestSlopeExtractor:
    def test_output_shape(self):
        feats = SlopeExtractor().extract(make_data())
        assert feats.shape == (N_CHANNELS, 1)

    def test_increasing_signal_positive_slope(self):
        data = np.arange(100, dtype=float).reshape(1, -1)
        feats = SlopeExtractor().extract(data)
        assert feats[0, 0] > 0

    def test_flat_signal_zero_slope(self):
        data = np.ones((4, 100)) * 5.0
        feats = SlopeExtractor().extract(data)
        np.testing.assert_allclose(feats, 0.0, atol=1e-10)


class TestHjorthExtractor:
    def test_output_shape_2d(self):
        feats = HjorthExtractor().extract(make_data())
        assert feats.shape == (N_CHANNELS, 3)

    def test_output_shape_3d(self):
        feats = HjorthExtractor().extract(make_epochs())
        assert feats.shape == (N_EPOCHS, N_CHANNELS, 3)

    def test_activity_equals_variance(self):
        data = make_data()
        feats = HjorthExtractor().extract(data)
        np.testing.assert_allclose(feats[:, 0], data.var(axis=-1), rtol=1e-5)

    def test_feature_names(self):
        names = HjorthExtractor().feature_names
        assert len(names) == 3
        assert "activity" in names[0]


class TestERPAmplitudeExtractor:
    def test_output_shape(self):
        extractor = ERPAmplitudeExtractor(window=(0.3, 0.5), sfreq=100.0, tmin=0.0)
        epochs = make_epochs()  # (10, 8, 512) with sfreq=128 → use as-is
        feats = extractor.extract(epochs)
        assert feats.shape == (N_EPOCHS, N_CHANNELS, 1)

    def test_constant_signal(self):
        extractor = ERPAmplitudeExtractor(window=(0.0, 0.5), sfreq=100.0, tmin=0.0)
        epochs = np.ones((5, 4, 100)) * 2.5
        feats = extractor.extract(epochs)
        np.testing.assert_allclose(feats, 2.5)

    def test_non_3d_raises(self):
        extractor = ERPAmplitudeExtractor(window=(0.3, 0.5), sfreq=100.0)
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((4, 100)))


class TestERPLatencyExtractor:
    def test_output_shape(self):
        extractor = ERPLatencyExtractor(window=(0.0, 0.5), sfreq=100.0, tmin=0.0)
        epochs = make_epochs()
        feats = extractor.extract(epochs)
        assert feats.shape == (N_EPOCHS, N_CHANNELS, 1)

    def test_non_3d_raises(self):
        extractor = ERPLatencyExtractor(window=(0.0, 0.5), sfreq=100.0)
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((4, 100)))


# ---------------------------------------------------------------------------
# Spatial features (CSP)
# ---------------------------------------------------------------------------

class TestCSPExtractor:
    def test_invalid_n_components(self):
        with pytest.raises(ValueError):
            CSPExtractor(n_components=1)

    def test_fit_and_extract(self):
        epochs = make_epochs()
        labels = np.array([1, 2] * (N_EPOCHS // 2))
        csp = CSPExtractor(n_components=4)
        csp.fit(epochs, labels)
        feats = csp.extract(epochs)
        assert feats.shape == (N_EPOCHS, 4)

    def test_fit_wrong_n_classes_raises(self):
        epochs = make_epochs()
        labels = np.ones(N_EPOCHS, dtype=int)  # only one class
        csp = CSPExtractor(n_components=4)
        with pytest.raises(ValueError, match="2 classes"):
            csp.fit(epochs, labels)

    def test_extract_without_fit_uses_fallback(self):
        """Unfitted CSP should use identity fallback and not raise."""
        epochs = make_epochs()
        csp = CSPExtractor(n_components=4)
        feats = csp.extract(epochs)
        assert feats.shape == (N_EPOCHS, 4)

    def test_feature_names(self):
        csp = CSPExtractor(n_components=6)
        names = csp.feature_names
        assert len(names) == 6
        assert all("csp" in n for n in names)
