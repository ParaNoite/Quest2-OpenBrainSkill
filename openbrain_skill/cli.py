"""Command-line interface for OpenBrainSkill."""
from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openbrain-skill",
        description=(
            "OpenBrainSkill: Multi-modal brain signal processing framework.\n\n"
            "Supports EEG, fNIRS, and ECoG signals with automatic and teaching modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # ------------------------------------------------------------------ run
    run_p = sub.add_parser("run", help="Run the pipeline in auto or teaching mode.")
    run_p.add_argument(
        "--modality",
        required=True,
        choices=["eeg", "fnirs", "ecog"],
        help="Signal modality.",
    )
    run_p.add_argument(
        "--paradigm",
        required=True,
        choices=["motor_imagery", "p300", "ssvep", "resting_state"],
        help="Experimental paradigm.",
    )
    run_p.add_argument(
        "--mode",
        choices=["auto", "teaching"],
        default="auto",
        help="Operating mode (default: auto).",
    )
    run_p.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (uses modality default if omitted).",
    )
    run_p.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="Number of channels (uses modality default if omitted).",
    )
    run_p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format for the report (default: text).",
    )
    run_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )

    # ---------------------------------------------------------- list-combinations
    list_p = sub.add_parser(
        "list", help="List available modality/paradigm combinations."
    )
    list_p.add_argument(
        "--json", action="store_true", help="Output as JSON."
    )

    # ------------------------------------------------------------------ info
    info_p = sub.add_parser(
        "info", help="Show knowledge graph entry for a modality/paradigm pair."
    )
    info_p.add_argument("--modality", required=True, help="Signal modality.")
    info_p.add_argument("--paradigm", required=True, help="Experimental paradigm.")
    info_p.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser


def _default_sfreq(modality: str) -> float:
    defaults = {"eeg": 256.0, "fnirs": 10.0, "ecog": 1000.0}
    return defaults.get(modality, 256.0)


def _default_n_channels(modality: str) -> int:
    defaults = {"eeg": 64, "fnirs": 20, "ecog": 128}
    return defaults.get(modality, 64)


def _cmd_run(args: argparse.Namespace) -> int:
    from openbrain_skill.core.skill import BrainSkill
    from openbrain_skill.report.generator import ReportGenerator

    sfreq = args.sfreq or _default_sfreq(args.modality)
    n_channels = args.n_channels or _default_n_channels(args.modality)

    try:
        skill = BrainSkill(
            modality=args.modality,
            paradigm=args.paradigm,
            sfreq=sfreq,
            n_channels=n_channels,
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    verbose = not args.quiet

    if args.mode == "auto":
        result = skill.run_auto(verbose=verbose)
    else:
        result = skill.run_teaching()

    gen = ReportGenerator()
    report = result["report"]

    if args.output == "json":
        print(gen.to_json(report))
    else:
        print(gen.to_text(report))

    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from openbrain_skill.core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    combos = kg.list_combinations()

    if getattr(args, "json", False):
        print(json.dumps(combos, indent=2))
    else:
        print("Available modality/paradigm combinations:")
        for combo in combos:
            modality, paradigm = combo.split(":", 1)
            desc = kg.get_description(modality, paradigm)
            short_desc = desc[:80] + "..." if len(desc) > 80 else desc
            print(f"  {combo:<30s}  {short_desc}")

    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from openbrain_skill.core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    entry = kg.query(args.modality, args.paradigm)

    if entry is None:
        print(
            f"No knowledge entry for {args.modality}/{args.paradigm}.",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(entry, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"  {args.modality.upper()} / {args.paradigm.replace('_', ' ').title()}")
        print(f"{'='*60}")
        print(f"\nDescription:\n  {entry.get('description', '')}")

        print("\nPreprocessing steps:")
        for step in entry.get("preprocessing", []):
            name = step.get("step", step.get("name", "?"))
            rationale = step.get("rationale", "")
            print(f"  • {name}: {rationale}")

        print("\nRecommended features:")
        for feat in entry.get("features", []):
            print(f"  • {feat.get('name', '?')}: {feat.get('rationale', '')}")

        tools = entry.get("recommended_tools", [])
        if tools:
            print(f"\nRecommended tools: {', '.join(tools)}")

        refs = entry.get("references", [])
        if refs:
            print("\nReferences:")
            for i, ref in enumerate(refs, 1):
                print(f"  [{i}] {ref}")

        print()

    return 0


def main(argv=None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "run": _cmd_run,
        "list": _cmd_list,
        "info": _cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
