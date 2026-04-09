"""Example: EEG Motor Imagery — Auto mode pipeline.

Demonstrates how to run the full preprocessing + feature extraction pipeline
for Motor Imagery EEG using OpenBrainSkill in automatic mode.

Usage:
    python examples/eeg_motor_imagery_auto.py
"""

import numpy as np
from openbrain_skill import BrainSkill
from openbrain_skill.report.generator import ReportGenerator

def main():
    print("OpenBrainSkill — EEG Motor Imagery Example (Auto Mode)")
    print("=" * 60)

    # 1. Create a BrainSkill for EEG Motor Imagery at 256 Hz, 32 channels
    skill = BrainSkill(
        modality="eeg",
        paradigm="motor_imagery",
        sfreq=256.0,
        n_channels=32,
    )

    print(f"\nModality: {skill.modality.upper()}")
    print(f"Paradigm: {skill.paradigm.replace('_', ' ').title()}")
    print(f"\nParadigm description:")
    print(f"  {skill.describe()[:200]}...")

    # 2. Generate synthetic EEG data
    print("\n[1] Generating synthetic EEG data (10 s)...")
    data = skill.modality_handler.generate_synthetic_data(duration=10.0, seed=42)
    print(f"    Data shape: {data.shape}  (channels × samples)")

    # 3. Generate synthetic events
    events = skill.paradigm_handler.generate_events(
        n_samples=data.shape[-1], sfreq=skill.sfreq, seed=42
    )
    print(f"    Events: {len(events)} trials")

    # 4. Run the full pipeline in auto mode
    print("\n[2] Running auto pipeline...")
    result = skill.run_auto(data=data, events=events, verbose=True)

    # 5. Inspect outputs
    print(f"\n[3] Results")
    print(f"    Feature matrix shape: {result['feature_matrix'].shape}")
    print(f"    Feature names (first 10): {result['feature_names'][:10]}")

    # 6. Print report
    gen = ReportGenerator()
    print("\n" + gen.to_text(result["report"]))

    # 7. Show recommended features from knowledge graph
    print("Recommended features from Knowledge Graph:")
    for feat in skill.get_recommended_features():
        print(f"  • {feat['name']}: {feat.get('rationale', '')}")


if __name__ == "__main__":
    main()
