"""Example: fNIRS Resting State — Auto mode pipeline.

Demonstrates the OpenBrainSkill framework for fNIRS resting-state data.

Usage:
    python examples/fnirs_resting_state_auto.py
"""

import numpy as np
from openbrain_skill import BrainSkill
from openbrain_skill.report.generator import ReportGenerator


def main():
    print("OpenBrainSkill — fNIRS Resting State Example (Auto Mode)")
    print("=" * 60)

    # 1. Create a BrainSkill for fNIRS Resting State
    skill = BrainSkill(
        modality="fnirs",
        paradigm="resting_state",
        sfreq=10.0,       # Typical fNIRS sampling rate
        n_channels=20,
    )

    print(f"\nModality: {skill.modality.upper()}")
    print(f"Paradigm: {skill.paradigm.replace('_', ' ').title()}")
    print(f"\nParadigm description:")
    print(f"  {skill.describe()[:200]}...")

    # 2. Show knowledge graph recommendations
    print("\nKnowledge Graph — Preprocessing recommendations:")
    for step in skill.get_preprocessing_steps():
        name = step.get("step", step.get("name", "?"))
        rationale = step.get("rationale", "")
        print(f"  • {name}: {rationale}")

    # 3. Generate synthetic fNIRS data (2 minutes of recording)
    print("\n[1] Generating synthetic fNIRS HbO data (120 s)...")
    data = skill.modality_handler.generate_synthetic_data(duration=120.0, seed=0)
    print(f"    Data shape: {data.shape}  (channels × samples)")

    # 4. Run auto pipeline
    print("\n[2] Running auto pipeline...")
    result = skill.run_auto(data=data, verbose=True)

    print(f"\n[3] Feature matrix shape: {result['feature_matrix'].shape}")
    print(f"    Feature names: {result['feature_names']}")

    gen = ReportGenerator()
    print("\n" + gen.to_text(result["report"]))


if __name__ == "__main__":
    main()
