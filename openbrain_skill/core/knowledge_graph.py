"""Knowledge graph for modality–paradigm best-practice recommendations."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default knowledge entries
# ---------------------------------------------------------------------------

_DEFAULT_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # EEG
    # -----------------------------------------------------------------------
    "eeg:motor_imagery": {
        "modality": "eeg",
        "paradigm": "motor_imagery",
        "description": (
            "Motor imagery (MI) EEG records voluntary imagination of limb movements "
            "without actual execution. Discriminative patterns emerge in mu (8-12 Hz) "
            "and beta (13-30 Hz) bands as event-related desynchronisation/synchronisation."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.5, "h_freq": 40.0},
                "rationale": "Remove DC drift and high-frequency noise; preserve mu/beta.",
            },
            {
                "step": "notch_filter",
                "params": {"freqs": [50, 60]},
                "rationale": "Suppress power-line interference.",
            },
            {
                "step": "artifact_removal",
                "method": "ica",
                "rationale": "Remove ocular and muscular artifacts via ICA.",
            },
            {
                "step": "epoching",
                "params": {"tmin": -0.5, "tmax": 4.0, "baseline": [-0.5, 0]},
                "rationale": "Epoch around cue onset; baseline-correct pre-stimulus.",
            },
            {
                "step": "common_average_reference",
                "rationale": "Reduce volume-conduction artifacts common in EEG.",
            },
        ],
        "features": [
            {
                "name": "band_power",
                "bands": {"mu": [8, 12], "beta": [13, 30]},
                "rationale": "ERD/ERS in mu and beta reflects motor imagery.",
            },
            {
                "name": "csp",
                "n_components": 6,
                "rationale": "Common Spatial Patterns maximise variance differences between classes.",
            },
            {
                "name": "psd",
                "method": "welch",
                "rationale": "Power spectral density for spectral characterisation.",
            },
        ],
        "recommended_tools": ["MNE-Python", "EEGLAB", "BCI-toolkit"],
        "references": [
            "Pfurtscheller & Neuper (2001). Motor imagery and direct brain-computer communication.",
            "Blankertz et al. (2008). The BCI competition 2008 – Graz data sets A and B.",
        ],
    },
    "eeg:p300": {
        "modality": "eeg",
        "paradigm": "p300",
        "description": (
            "P300 is an event-related potential (ERP) peaking ~300 ms after an "
            "infrequent, task-relevant stimulus. It is widely used in speller BCIs."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.1, "h_freq": 30.0},
                "rationale": "Preserve ERP morphology; P300 is a slow cortical potential.",
            },
            {
                "step": "notch_filter",
                "params": {"freqs": [50, 60]},
                "rationale": "Suppress power-line interference.",
            },
            {
                "step": "artifact_removal",
                "method": "threshold",
                "rationale": "Reject epochs with amplitude exceeding ±100 µV.",
            },
            {
                "step": "epoching",
                "params": {"tmin": -0.2, "tmax": 1.0, "baseline": [-0.2, 0]},
                "rationale": "Capture full P300 deflection; pre-stimulus baseline.",
            },
            {
                "step": "averaging",
                "rationale": "Average multiple ERP repetitions to improve SNR.",
            },
        ],
        "features": [
            {
                "name": "erp_amplitude",
                "window": [0.25, 0.45],
                "rationale": "Mean amplitude in P300 latency window.",
            },
            {
                "name": "erp_latency",
                "rationale": "Peak latency of P300 component.",
            },
            {
                "name": "xdawn",
                "n_components": 3,
                "rationale": "xDAWN spatial filtering maximises target/non-target discriminability.",
            },
        ],
        "recommended_tools": ["MNE-Python", "ERPLAB"],
        "references": [
            "Farwell & Donchin (1988). Talking off the top of your head.",
            "Rivet et al. (2009). xDAWN algorithm to enhance evoked potentials.",
        ],
    },
    "eeg:ssvep": {
        "modality": "eeg",
        "paradigm": "ssvep",
        "description": (
            "Steady-state visual evoked potentials (SSVEP) are oscillatory responses "
            "to flickering stimuli. The response appears at the stimulus frequency and "
            "harmonics, mainly over occipital electrodes."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 1.0, "h_freq": 50.0},
                "rationale": "Preserve SSVEP frequency band up to second harmonic.",
            },
            {
                "step": "notch_filter",
                "params": {"freqs": [50, 60]},
                "rationale": "Remove power-line; avoid overlap with stimulus frequencies.",
            },
            {
                "step": "epoching",
                "params": {"tmin": 0.5, "tmax": 5.0, "baseline": None},
                "rationale": "Discard initial transient; use full steady-state window.",
            },
        ],
        "features": [
            {
                "name": "psd",
                "method": "welch",
                "rationale": "Identify spectral peaks at stimulus and harmonic frequencies.",
            },
            {
                "name": "snr_spectrum",
                "rationale": "SNR relative to neighbouring frequencies for SSVEP detection.",
            },
            {
                "name": "cca",
                "rationale": "Canonical Correlation Analysis between EEG and sinusoidal references.",
            },
        ],
        "recommended_tools": ["MNE-Python", "SSVEP-BCI"],
        "references": [
            "Norbert Birbaumer et al. (1999). Brain-computer interface research.",
            "Chen et al. (2015). Filter bank canonical correlation analysis for SSVEP-BCI.",
        ],
    },
    "eeg:resting_state": {
        "modality": "eeg",
        "paradigm": "resting_state",
        "description": (
            "Resting-state EEG captures spontaneous brain activity without explicit tasks. "
            "Used to characterise individual alpha peak, connectivity, and mental state."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 1.0, "h_freq": 45.0},
                "rationale": "Preserve classical frequency bands (delta to gamma).",
            },
            {
                "step": "artifact_removal",
                "method": "ica",
                "rationale": "Remove cardiac, ocular, and muscular artifacts.",
            },
            {
                "step": "segmentation",
                "params": {"duration": 2.0, "overlap": 0.5},
                "rationale": "Split continuous signal into overlapping windows.",
            },
        ],
        "features": [
            {
                "name": "band_power",
                "bands": {
                    "delta": [1, 4],
                    "theta": [4, 8],
                    "alpha": [8, 12],
                    "beta": [13, 30],
                    "gamma": [30, 45],
                },
                "rationale": "Absolute and relative band power quantify mental state.",
            },
            {
                "name": "spectral_entropy",
                "rationale": "Measures spectral complexity; reduced in pathological states.",
            },
            {
                "name": "hjorth_parameters",
                "rationale": "Activity, Mobility, Complexity from temporal domain.",
            },
        ],
        "recommended_tools": ["MNE-Python", "EEGLAB"],
        "references": [
            "Klimesch (1999). EEG alpha and theta oscillations reflect cognitive and memory performance.",
        ],
    },
    # -----------------------------------------------------------------------
    # fNIRS
    # -----------------------------------------------------------------------
    "fnirs:motor_imagery": {
        "modality": "fnirs",
        "paradigm": "motor_imagery",
        "description": (
            "fNIRS motor imagery measures haemodynamic responses (HbO/HbR) in motor cortex "
            "during imagined movement. Responses are slow (peak ~6-8 s)."
        ),
        "preprocessing": [
            {
                "step": "motion_correction",
                "method": "tddr",
                "rationale": "Temporal Derivative Distribution Repair (TDDR) for motion artifacts.",
            },
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.01, "h_freq": 0.1},
                "rationale": "Isolate haemodynamic response; remove cardiac/respiratory.",
            },
            {
                "step": "beer_lambert_law",
                "rationale": "Convert raw optical intensities to HbO and HbR concentrations.",
            },
            {
                "step": "epoching",
                "params": {"tmin": -2.0, "tmax": 20.0, "baseline": [-2, 0]},
                "rationale": "Capture full HRF; pre-stimulus baseline correction.",
            },
        ],
        "features": [
            {
                "name": "hrf_amplitude",
                "channel": "HbO",
                "window": [4, 12],
                "rationale": "Peak HbO amplitude reflects cortical activation.",
            },
            {
                "name": "mean_hbo",
                "rationale": "Mean HbO during activation window.",
            },
            {
                "name": "slope",
                "rationale": "Linear slope of HbO during task indicates response speed.",
            },
        ],
        "recommended_tools": ["MNE-Python", "Homer3", "NIRS-SPM"],
        "references": [
            "Strangman et al. (2002). A quantitative comparison of simultaneous BOLD fMRI and NIRS recordings.",
        ],
    },
    "fnirs:resting_state": {
        "modality": "fnirs",
        "paradigm": "resting_state",
        "description": (
            "Resting-state fNIRS captures spontaneous haemodynamic fluctuations and "
            "functional connectivity between brain regions."
        ),
        "preprocessing": [
            {
                "step": "motion_correction",
                "method": "wavelet",
                "rationale": "Wavelet-based motion artifact correction.",
            },
            {
                "step": "beer_lambert_law",
                "rationale": "Convert optical intensities to HbO/HbR concentrations.",
            },
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.01, "h_freq": 0.1},
                "rationale": "Focus on low-frequency haemodynamic fluctuations.",
            },
        ],
        "features": [
            {
                "name": "functional_connectivity",
                "method": "pearson",
                "rationale": "Pearson correlation between channel pairs for FC networks.",
            },
            {
                "name": "psd",
                "method": "welch",
                "rationale": "Power spectrum of haemodynamic fluctuations.",
            },
        ],
        "recommended_tools": ["MNE-Python", "Homer3"],
        "references": [],
    },
    # -----------------------------------------------------------------------
    # ECoG
    # -----------------------------------------------------------------------
    "ecog:motor_imagery": {
        "modality": "ecog",
        "paradigm": "motor_imagery",
        "description": (
            "Electrocorticography (ECoG) provides high-resolution direct cortical recordings. "
            "High-gamma (70-150 Hz) power is the primary feature for motor decoding."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.5, "h_freq": 200.0},
                "rationale": "Preserve high-gamma; ECoG supports broader bandwidth than EEG.",
            },
            {
                "step": "notch_filter",
                "params": {"freqs": [50, 100, 150, 60, 120, 180]},
                "rationale": "Remove line noise and harmonics from ECoG recordings.",
            },
            {
                "step": "common_average_reference",
                "rationale": "CAR suppresses common-mode noise in ECoG.",
            },
            {
                "step": "epoching",
                "params": {"tmin": -1.0, "tmax": 4.0, "baseline": [-1, 0]},
                "rationale": "Epoch around movement/imagery cue.",
            },
        ],
        "features": [
            {
                "name": "high_gamma_power",
                "band": [70, 150],
                "rationale": "Broadband high-gamma is the most reliable ECoG motor feature.",
            },
            {
                "name": "band_power",
                "bands": {"mu": [8, 12], "beta": [13, 30], "high_gamma": [70, 150]},
                "rationale": "Multi-band power analysis.",
            },
            {
                "name": "phase_amplitude_coupling",
                "rationale": "PAC between theta phase and high-gamma amplitude encodes cognitive load.",
            },
        ],
        "recommended_tools": ["MNE-Python", "FieldTrip", "BrainDecoder"],
        "references": [
            "Shenoy et al. (2013). Cortical control of arm movements: a dynamical systems perspective.",
        ],
    },
    "ecog:resting_state": {
        "modality": "ecog",
        "paradigm": "resting_state",
        "description": (
            "Resting-state ECoG characterises intrinsic cortical dynamics and spectral "
            "properties with spatial precision unavailable to scalp EEG."
        ),
        "preprocessing": [
            {
                "step": "bandpass_filter",
                "params": {"l_freq": 0.5, "h_freq": 200.0},
                "rationale": "Preserve full ECoG bandwidth.",
            },
            {
                "step": "common_average_reference",
                "rationale": "Reference to common average across electrodes.",
            },
            {
                "step": "segmentation",
                "params": {"duration": 1.0, "overlap": 0.5},
                "rationale": "Short windows for non-stationary ECoG signals.",
            },
        ],
        "features": [
            {
                "name": "band_power",
                "bands": {
                    "delta": [1, 4],
                    "theta": [4, 8],
                    "alpha": [8, 12],
                    "beta": [13, 30],
                    "low_gamma": [30, 70],
                    "high_gamma": [70, 150],
                },
                "rationale": "Full spectral characterisation of ECoG.",
            },
            {
                "name": "spectral_exponent",
                "rationale": "1/f slope reflects cortical excitability.",
            },
        ],
        "recommended_tools": ["MNE-Python", "FieldTrip"],
        "references": [],
    },
}


class KnowledgeGraph:
    """Knowledge graph of preprocessing and feature extraction best practices.

    Entries are keyed by ``"<modality>:<paradigm>"`` and contain
    preprocessing steps, feature recommendations, references, and tool
    suggestions for that combination.

    Args:
        entries: Optional additional entries to merge into the built-in graph.
    """

    def __init__(self, entries: Optional[Dict[str, Any]] = None) -> None:
        self._graph: Dict[str, Any] = dict(_DEFAULT_KNOWLEDGE)
        if entries:
            self._graph.update(entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, modality: str, paradigm: str) -> Optional[Dict[str, Any]]:
        """Return the knowledge entry for a modality–paradigm pair.

        Args:
            modality: Signal modality (e.g. ``"eeg"``).
            paradigm: Experimental paradigm (e.g. ``"motor_imagery"``).

        Returns:
            Dict with preprocessing and feature recommendations, or *None*
            if the combination is not in the graph.
        """
        key = f"{modality.lower()}:{paradigm.lower()}"
        return self._graph.get(key)

    def get_preprocessing_steps(
        self, modality: str, paradigm: str
    ) -> List[Dict[str, Any]]:
        """Return recommended preprocessing steps for a modality–paradigm pair.

        Args:
            modality: Signal modality.
            paradigm: Experimental paradigm.

        Returns:
            List of preprocessing step dicts (may be empty).
        """
        entry = self.query(modality, paradigm)
        if entry is None:
            return []
        return entry.get("preprocessing", [])

    def get_features(self, modality: str, paradigm: str) -> List[Dict[str, Any]]:
        """Return recommended features for a modality–paradigm pair.

        Args:
            modality: Signal modality.
            paradigm: Experimental paradigm.

        Returns:
            List of feature dicts (may be empty).
        """
        entry = self.query(modality, paradigm)
        if entry is None:
            return []
        return entry.get("features", [])

    def get_description(self, modality: str, paradigm: str) -> str:
        """Return the description for a modality–paradigm pair.

        Args:
            modality: Signal modality.
            paradigm: Experimental paradigm.

        Returns:
            Description string, or empty string if not found.
        """
        entry = self.query(modality, paradigm)
        if entry is None:
            return ""
        return entry.get("description", "")

    def get_references(self, modality: str, paradigm: str) -> List[str]:
        """Return literature references for a modality–paradigm pair.

        Args:
            modality: Signal modality.
            paradigm: Experimental paradigm.

        Returns:
            List of reference strings (may be empty).
        """
        entry = self.query(modality, paradigm)
        if entry is None:
            return []
        return entry.get("references", [])

    def list_combinations(self) -> List[str]:
        """Return all registered modality–paradigm combination keys."""
        return list(self._graph.keys())

    def add_entry(self, modality: str, paradigm: str, entry: Dict[str, Any]) -> None:
        """Add or replace a knowledge entry.

        Args:
            modality: Signal modality name.
            paradigm: Experimental paradigm name.
            entry: Knowledge entry dict.
        """
        key = f"{modality.lower()}:{paradigm.lower()}"
        self._graph[key] = entry

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the entire graph as a plain dict."""
        return dict(self._graph)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the graph to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the knowledge graph.
        """
        return json.dumps(self._graph, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        """Create a KnowledgeGraph from a JSON string.

        Args:
            json_str: JSON string of knowledge entries.

        Returns:
            KnowledgeGraph with the deserialized entries merged in.
        """
        data = json.loads(json_str)
        return cls(entries=data)
