# Quest2-OpenBrainSkill — OpenBrainSkill Framework

**OpenBrainSkill** is a multi-modal brain signal processing skill framework that provides a unified, teachable entry point for EEG, fNIRS, and ECoG signal preprocessing and feature extraction.

---

## Overview

Brain signal preprocessing (EEG, fNIRS, ECoG) is highly modality- and paradigm-dependent. Existing tools (MNE-Python, EEGLAB, FieldTrip) each cover different parts of the pipeline but lack a unified, extensible interface. **OpenBrainSkill** solves this by:

- Building a **knowledge graph** of preprocessing + feature extraction best practices per modality/paradigm combination.
- Providing an **automatic mode** (input data + paradigm → feature matrix + report).
- Providing a **teaching mode** (step-by-step explanations + reusable code templates).
- Being **dynamically extensible** to new modalities or paradigms.
- Producing **structured documentation** and **modular, reusable code**.

---

## Supported Modalities and Paradigms

| Modality | Paradigm         | Key Features                          |
|----------|------------------|---------------------------------------|
| EEG      | Motor Imagery    | Band power (mu/beta), CSP, PSD        |
| EEG      | P300             | ERP amplitude & latency               |
| EEG      | SSVEP            | PSD, SNR at stimulus frequencies      |
| EEG      | Resting State    | Band power, spectral entropy, Hjorth  |
| fNIRS    | Motor Imagery    | Mean HbO, slope (HRF analysis)        |
| fNIRS    | Resting State    | PSD, functional connectivity          |
| ECoG     | Motor Imagery    | High-gamma power, band power          |
| ECoG     | Resting State    | Broadband power, spectral exponent    |

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Auto Mode

```python
from openbrain_skill import BrainSkill

skill = BrainSkill(modality="eeg", paradigm="motor_imagery", sfreq=256.0, n_channels=64)
result = skill.run_auto()

print("Feature matrix shape:", result["feature_matrix"].shape)
print("Features:", result["feature_names"][:5])
```

### Teaching Mode

```python
from openbrain_skill import BrainSkill, TeachingMode

skill = BrainSkill(modality="eeg", paradigm="ssvep", sfreq=256.0, n_channels=8)
mode = TeachingMode(skill=skill)
result = mode.run()

for step in result["steps"]:
    print(f"Step: {step['name']}")
    print(f"  {step['description']}")
    print(f"  Code:\n{step['code_template']}")
```

### CLI

```bash
openbrain-skill list
openbrain-skill run --modality eeg --paradigm p300 --sfreq 256 --n-channels 64
openbrain-skill info --modality ecog --paradigm motor_imagery
openbrain-skill run --modality fnirs --paradigm resting_state --output json
```

---

## Architecture

```
openbrain_skill/
├── core/
│   ├── pipeline.py         # Ordered preprocessing pipeline
│   ├── skill.py            # Top-level BrainSkill orchestrator
│   ├── knowledge_graph.py  # Best-practice knowledge graph
│   └── registry.py         # Dynamic modality/paradigm registry
├── modalities/             # EEG, fNIRS, ECoG handlers
├── paradigms/              # Motor Imagery, P300, SSVEP, Resting State
├── preprocessing/          # Filters, artifact removal, epoching
├── features/               # Spectral, temporal, spatial extractors
├── modes/                  # Auto mode, Teaching mode
├── report/                 # Report generation (text + JSON)
└── cli.py                  # Command-line interface
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT License
