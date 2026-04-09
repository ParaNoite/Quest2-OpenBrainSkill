"""Example: Teaching Mode Tutorial.

Demonstrates OpenBrainSkill's teaching mode for EEG SSVEP analysis.
Teaching mode provides step-by-step explanations and code templates.

Usage:
    python examples/teaching_mode_tutorial.py
"""

from openbrain_skill import BrainSkill, TeachingMode


def main():
    print("OpenBrainSkill — Teaching Mode Tutorial")
    print("=" * 60)
    print(
        "This example demonstrates the Teaching Mode, which provides:\n"
        "  • Step-by-step explanations of the preprocessing pipeline\n"
        "  • Reusable Python code templates for each step\n"
        "  • Knowledge graph descriptions and literature references\n"
    )

    # Create skill for EEG SSVEP
    skill = BrainSkill(
        modality="eeg",
        paradigm="ssvep",
        sfreq=256.0,
        n_channels=8,
    )

    # Run in teaching mode (generates synthetic data automatically)
    mode = TeachingMode(skill=skill)
    result = mode.run()

    # Display collected code template
    print("\n" + "=" * 70)
    print("  GENERATED CODE TEMPLATE")
    print("=" * 70)
    print(result["code_templates"])

    # Display references
    refs = result["references"]
    if refs:
        print("\n" + "=" * 70)
        print("  KEY REFERENCES")
        print("=" * 70)
        for i, ref in enumerate(refs, 1):
            print(f"  [{i}] {ref}")

    print(f"\n[Feature matrix shape] {result['feature_matrix'].shape}")
    print(f"[Feature names]        {result['feature_names'][:5]}...")

    print("\n✅ Teaching mode complete!")


if __name__ == "__main__":
    main()
