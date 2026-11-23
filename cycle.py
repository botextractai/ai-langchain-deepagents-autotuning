"""
One hill-climbing cycle, end to end, with a person approving each change.

Runs a before batch, asks the analyst for one AGENTS.md change, applies it to a
copy if you approve, runs the after batch, compares the two, and only updates
workspace/memories/AGENTS.md if you approve that too.
"""

import argparse
import difflib
import shutil
import sys

import deepagent_helpers as dh
from hill_climbing import (
    DEFAULT_ANALYST_MODEL,
    apply_change,
    compare,
    format_comparison,
    propose_harness_change,
)
from run_history import OUTPUT_ROOT, generate_run_history


def confirm(question: str) -> bool:
    return input(f"\n{question} [y/N] ").strip().lower() in ("y", "yes")


def step(title: str) -> None:
    print(f"\n=== {title} ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hill-climbing cycle.")
    parser.add_argument("--before", required=True,
                        help="baseline batch label; reused if it already exists")
    parser.add_argument("--after", required=True,
                        help="label for the batch run with the improved instructions")
    parser.add_argument("--model", default=dh.DEFAULT_MODEL,
                        help="model for the review agent and its grader")
    parser.add_argument("--analyst-model", default=DEFAULT_ANALYST_MODEL)
    args = parser.parse_args()

    baseline_path = dh.AGENTS_BASELINE
    before_dir = OUTPUT_ROOT / args.before
    after_dir = OUTPUT_ROOT / args.after
    candidate_path = OUTPUT_ROOT / f"{args.after}_AGENTS.md"

    if args.before == args.after:
        parser.error("--before and --after must be different labels")
    if after_dir.exists():
        parser.error(f"batch '{args.after}' already exists in {OUTPUT_ROOT}; "
                     "choose a new --after label or delete that folder")

    baseline = baseline_path.read_text(encoding="utf-8")

    if (before_dir / "summary.json").exists():
        step(f"Step 1: reusing existing batch '{args.before}'")
        # The proposal is applied to the current baseline, so it has to have
        # been derived from runs that used that same baseline.
        if (before_dir / "AGENTS.md").read_text(encoding="utf-8") != baseline:
            sys.exit(f"Batch '{args.before}' was run with different instructions than "
                     f"{baseline_path}. Use a new --before label to run a fresh baseline.")
        print("It was run with the current AGENTS.md, so no new run is needed.")
    else:
        step(f"Step 1: running the baseline batch '{args.before}'")
        generate_run_history(label=args.before, model=args.model)

    step("Step 2: asking the analyst for a change")
    change, proposal = propose_harness_change(args.before, model=args.analyst_model)
    print(proposal)
    print(f"Saved as {OUTPUT_ROOT / f'proposal_{args.before}.md'} "
          f"and {OUTPUT_ROOT / f'proposal_{args.before}.json'}.")

    if not confirm("Apply this change?"):
        print("Not applied. AGENTS.md is unchanged.")
        return

    step("Step 3: applying the change to a copy of AGENTS.md")
    try:
        improved = apply_change(change, baseline)
    except ValueError as e:
        sys.exit(f"Could not apply the change automatically: {e}")
    candidate_path.write_text(improved, encoding="utf-8")
    sys.stdout.writelines(difflib.unified_diff(
        baseline.splitlines(keepends=True), improved.splitlines(keepends=True),
        str(baseline_path), str(candidate_path)))
    print(f"\nSaved as {candidate_path}.")

    step(f"Step 4: running the batch '{args.after}' with the new instructions")
    generate_run_history(label=args.after, model=args.model, agents_md=candidate_path)

    step("Step 5: comparing before and after")
    result = compare(args.before, args.after)
    print(format_comparison(result, args.before, args.after))

    if result["verdict"] not in ("success", "partial"):
        print(f"\nThe change didn't help, so AGENTS.md is unchanged. "
              f"The tried version is in {candidate_path}.")
        return

    if confirm(f"Make the new instructions the baseline ({baseline_path})?"):
        shutil.copyfile(candidate_path, baseline_path)
        print(f"Updated {baseline_path}.")
        if result["verdict"] == "partial":
            print(f"For another cycle: python cycle.py --before {args.after} --after <new label>")
    else:
        print(f"AGENTS.md is unchanged. The tried version is in {candidate_path}.")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()
