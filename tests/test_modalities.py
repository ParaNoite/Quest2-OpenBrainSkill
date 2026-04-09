"""Tests for modality handlers."""
import numpy as np
import pytest

from openbrain_skill.modalities.eeg import EEGModality
from openbrain_skill.modalities.fnirs import FNIRSModality
from openbrain_skill.modalities.ecog import ECoGModality


class TestEEGModality:
    def setup_method(self):
        self.mod = EEGModality(sfreq=256.0, n_channels=8)

    def test_generate_synthetic_data_shape(self):
        data = self.mod.generate_synthetic_data(duration=2.0, seed=42)
        assert data.shape == (8, int(2.0 * 256))

    def test_generate_synthetic_data_dtype(self):
        data = self.mod.generate_synthetic_data(seed=0)
        assert data.dtype == np.float64

    def test_generate_with_trials(self):
        data = self.mod.generate_synthetic_data(duration=2.0, n_trials=5, seed=0)
        assert data.ndim == 3
        assert data.shape[0] == 5
        assert data.shape[1] == 8

    def test_validate_data_ok(self):
        data = np.zeros((8, 256))
        self.mod.validate_data(data)  # should not raise

    def test_validate_data_wrong_channels_raises(self):
        data = np.zeros((4, 256))
        with pytest.raises(ValueError, match="channels"):
            self.mod.validate_data(data)

    def test_validate_data_3d_ok(self):
        data = np.zeros((10, 8, 256))
        self.mod.validate_data(data)  # should not raise

    def test_validate_data_1d_raises(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            self.mod.validate_data(np.zeros(256))

    def test_default_channel_names_length(self):
        names = self.mod.default_channel_names
        assert len(names) == 8

    def test_repr(self):
        assert "EEGModality" in repr(self.mod)


class TestFNIRSModality:
    def setup_method(self):
        self.mod = FNIRSModality(sfreq=10.0, n_channels=10)

    def test_generate_synthetic_data_shape(self):
        data = self.mod.generate_synthetic_data(duration=30.0, seed=1)
        assert data.shape == (10, int(30.0 * 10))

    def test_generate_with_trials(self):
        data = self.mod.generate_synthetic_data(duration=10.0, n_trials=3, seed=2)
        assert data.ndim == 3
        assert data.shape[0] == 3

    def test_validate_data_ok(self):
        data = np.zeros((10, 100))
        self.mod.validate_data(data)

    def test_validate_data_wrong_channels_raises(self):
        with pytest.raises(ValueError, match="channels"):
            self.mod.validate_data(np.zeros((5, 100)))

    def test_default_channel_names(self):
        names = self.mod.default_channel_names
        assert len(names) == 10
        assert "HbO" in names[0]


class TestECoGModality:
    def setup_method(self):
        self.mod = ECoGModality(sfreq=1000.0, n_channels=16)

    def test_generate_synthetic_data_shape(self):
        data = self.mod.generate_synthetic_data(duration=2.0, seed=7)
        assert data.shape == (16, 2000)

    def test_validate_data_ok(self):
        data = np.zeros((16, 1000))
        self.mod.validate_data(data)

    def test_validate_data_wrong_channels_raises(self):
        with pytest.raises(ValueError, match="channels"):
            self.mod.validate_data(np.zeros((8, 1000)))

    def test_channel_names_grid_format(self):
        names = self.mod.default_channel_names
        assert len(names) == 16
        assert names[0].startswith("G")
