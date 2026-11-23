"""
Read a batch of real runs and propose an improvement to the harness.

This is the analysis half of the hill climbing loop. `run_history.py` produces
the evidence; this reads it and says what should change.

The harness has four parts that can be improved: the prompt, the tools, the
grader, and the configuration. This analysis is scoped to the prompt, because
that is what the current evidence implicates. `HarnessChange.target` still names
all four, so a proposal that points elsewhere is expressible rather than
silently reshaped into a prompt edit.

Two properties worth keeping if this is extended.

The agent reads the traces and cannot write them. An agent able to edit the
evidence it is reasoning from is not analyzing anything.

The agent proposes and does not apply. It writes `/proposal.md` and returns a
structured change; `apply_change` performs the edit in plain code. That keeps the
edit reviewable, keeps it deterministic, and means a proposal about tools or
configuration can be read and rejected rather than half-applied.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import tiktoken
from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    create_deep_agent,
)
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from pydantic import BaseModel, Field

import deepagent_helpers as dh
from run_history import OUTPUT_ROOT, criterion_labels

# The hill climber does the only real reasoning in the loop: reading several
# conversations and working out what recurs. The history batches are routine, so
# they run on a cheaper model. See run_history.generate_run_history.
DEFAULT_ANALYST_MODEL = "openai:gpt-6.1-sol"

PROPOSAL_PATH = "/proposal.md"

# Per million tokens, short context, for the cost figure in `compare`.
PRICES = {
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-6.1-terra": (1.00, 5.00),
}


class HarnessChange(BaseModel):
    """One proposed improvement, structured so it can be applied or rejected."""

    target: Literal["prompt", "tool", "grader", "configuration"] = Field(
        description="Which part of the harness should change."
    )
    issue: str = Field(
        description="What goes wrong, stated as a property of the agent rather "
                    "than of one document."
    )
    recurrence: str = Field(
        description="How many of the runs show it, and which ones."
    )
    evidence: List[str] = Field(
        description="Short quotations from the runs that show the issue. Quote, "
                    "do not summarise."
    )
    section: str = Field(
        description="The exact heading in AGENTS.md the new text belongs under, "
                    "for example '### Ambiguities'. Empty if target is not "
                    "'prompt'."
    )
    change: str = Field(
        description="The literal text to add, ready to insert with no editing. "
                    "Not a description of what to add."
    )
    rejected: List[str] = Field(
        default_factory=list,
        description="Anything noticed but not acted on, and why. A single "
                    "occurrence across the batch is not a pattern.",
    )


ANALYST_PROMPT = f"""You are improving a Requirements Review Agent by reading its \
production runs.

Each run is one requirement document, reviewed by the agent and then graded \
against a rubric. If the grader rejected an attempt it injected its feedback into \
the conversation and the agent tried again. So a run records what the agent was \
told at the start, what it produced, what it had to be told afterwards, and what \
it then fixed.

Your job is to find what the agent has to be corrected on *every time*, and to \
propose the change to its instructions that would stop that correction being \
necessary.

Start with `/summary.json`, which gives the first-attempt pass rate per criterion \
across the batch. That tells you which criteria to investigate. Then read into \
the individual runs (`/run_*.json`) for evidence, and read `/AGENTS.md` to see \
what the agent was actually told, so your addition fits the file and does not \
repeat something already there.

Discipline that matters:

- A criterion failing in most runs is a property of the agent. A problem \
appearing in one run out of five is not a pattern, and you must not act on it. \
Note it under `rejected` instead.
- Quote your evidence from the runs. Do not paraphrase it.
- Your `change` must be the literal text to add to `/AGENTS.md`, ready to insert \
without editing. If the agent needs to produce a specific labelled line, give the \
exact label.
- Do not restate instructions the file already contains.

Write your reasoning to `{PROPOSAL_PATH}` as markdown, then return the structured \
change. You cannot modify anything else, and you should not try."""


def build_hill_climbing_agent(
    batch_dir: Path,
    model: str = DEFAULT_ANALYST_MODEL,
    max_turns: int = dh.MAX_TURNS,
):
    """An agent that can read one batch of runs and write one proposal.

    Rooted at the batch directory, so `/summary.json` and `/AGENTS.md` are that
    batch's files. Other batches and earlier proposals are out of reach: a later
    batch's AGENTS.md may already contain the fix, and an analyst that can read
    it would copy the answer rather than derive it from the evidence.
    """
    backend = FilesystemBackend(root_dir=str(batch_dir), virtual_mode=True)

    # First match wins, and anything unmatched is allowed, so the allow for the
    # proposal has to come before the blanket deny.
    permissions = [
        FilesystemPermission(operations=["write"], paths=[PROPOSAL_PATH], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
    ]

    return create_deep_agent(
        model=dh.resolve_model(model),
        system_prompt=ANALYST_PROMPT,
        backend=backend,
        middleware=[
            FilesystemMiddleware(
                backend=backend,
                tools=["ls", "read_file", "glob", "grep", "write_file"],
                permissions=permissions,
            ),
            ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="end"),
        ],
        response_format=HarnessChange,
    )


def propose_harness_change(
    label: str,
    model: str = DEFAULT_ANALYST_MODEL,
    history_root: Path = OUTPUT_ROOT,
) -> Tuple[HarnessChange, str]:
    """Read one batch and return the proposed change plus the written reasoning."""
    history_root = Path(history_root)
    batch_dir = history_root / label
    if not (batch_dir / "summary.json").exists():
        raise FileNotFoundError(
            f"No batch '{label}' in {history_root}. Run `python run_history.py "
            f"--label {label}` first."
        )

    # A proposal left over from an earlier analysis would otherwise be read back
    # as this one's if the agent ends without writing a new file.
    proposal_file = batch_dir / PROPOSAL_PATH.lstrip("/")
    proposal_file.unlink(missing_ok=True)

    agent = build_hill_climbing_agent(batch_dir, model)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content":
                           "Read the run history and propose one improvement to the "
                           "agent's instructions."}]},
            config={
                "run_name": f"hill climbing analysis / {label}",
                "tags": ["loop-engineering", "hill-climbing", label],
            },
        )
    except Exception as e:
        dh.abort_on_timeout(
            e,
            f"analysing batch '{label}'",
            "python -c \"from pathlib import Path; "
            f"[p.unlink(missing_ok=True) for p in ["
            f"Path('run_history/{label}/proposal.md'), "
            f"Path('run_history/proposal_{label}.md'), "
            f"Path('run_history/proposal_{label}.json')]]\"",
        )
        raise
    change = result["structured_response"]

    proposal = proposal_file.read_text(encoding="utf-8") if proposal_file.exists() else ""

    # Persist both halves keyed by the batch they came from, so the proposal can
    # be reviewed and applied later without paying for the analysis again.
    # The batch's own proposal.md stays as the agent's scratch file.
    (history_root / f"proposal_{label}.md").write_text(proposal, encoding="utf-8")
    (history_root / f"proposal_{label}.json").write_text(
        json.dumps(change.model_dump(), indent=2), encoding="utf-8"
    )
    return change, proposal


def load_proposal(
    label: str, history_root: Path = OUTPUT_ROOT
) -> Tuple[HarnessChange, str]:
    """The saved analysis for a batch, as written by `propose_harness_change`."""
    history_root = Path(history_root)
    change = HarnessChange(
        **json.loads((history_root / f"proposal_{label}.json").read_text(encoding="utf-8"))
    )
    return change, (history_root / f"proposal_{label}.md").read_text(encoding="utf-8")


def apply_change(change: HarnessChange, agents_md: str) -> str:
    """Insert the proposed text at the end of the section it names.

    Plain string work on purpose. The agent decided what should change; deciding
    where it lands should not also be a model call.
    """
    if change.target != "prompt":
        raise ValueError(
            f"target is '{change.target}', which is not applied automatically. "
            "Read the proposal and decide."
        )

    lines = agents_md.splitlines()
    heading = change.section.strip()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == heading)
    except StopIteration:
        raise ValueError(f"section {heading!r} not found in AGENTS.md")

    # The section ends at the next heading of the same depth or shallower.
    depth = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        if stripped.startswith("#"):
            if len(lines[i]) - len(lines[i].lstrip("#")) <= depth:
                end = i
                break

    body = lines[start:end]
    while body and not body[-1].strip():
        body.pop()

    addition = change.change.strip().splitlines()
    out = "\n".join(lines[:start] + body + [""] + addition + [""] + lines[end:])
    return out if out.endswith("\n") else out + "\n"


# ---------------------------------------------------------------------------
# Measuring whether it worked
# ---------------------------------------------------------------------------

def load_summary(label: str, history_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    return json.loads((Path(history_root) / label / "summary.json").read_text(encoding="utf-8"))


def load_runs(label: str, history_root: Path = OUTPUT_ROOT) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(history_root) / label).glob("run_*.json"))]


def usage(label: str, history_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    """Approximate tokens and cost for a batch, from the saved conversations.

    Every model call sends the history so far, so input is the running total
    rather than the message length.
    """
    enc = tiktoken.get_encoding("o200k_base")
    count = lambda s: len(enc.encode(s))

    runs = load_runs(label, history_root)
    tin = tout = attempts = 0
    for r in runs:
        base = count(r["instructions"]) + 120
        hist = 0
        for m in r["messages"]:
            size = count(m["content"]) + sum(
                count(json.dumps(c.get("args") or {})) for c in m["tool_calls"]
            )
            if m["type"] == "ai":
                tin += base + hist
                tout += size
            hist += size
        attempts += len(r["grading"])

    model = runs[0]["model"] if runs else ""
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return {
        "label": label,
        "model": model,
        "input_tokens": tin,
        "output_tokens": tout,
        "attempts": attempts,
        "cost_usd": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
    }


def compare(before: str, after: str, history_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    """Did the change work? One dict, and `verdict` is the answer."""
    sb, sa = load_summary(before, history_root), load_summary(after, history_root)
    ub, ua = usage(before, history_root), usage(after, history_root)
    labels = criterion_labels()

    moved, regressed = [], []
    for name in labels:
        b = sb["first_attempt"][name]["passed"]
        a = sa["first_attempt"][name]["passed"]
        if a > b:
            moved.append(name)
        elif a < b:
            regressed.append(name)

    if regressed:
        verdict = "regression"
    elif sa["revisions_needed"] == 0:
        verdict = "success"
    elif moved:
        verdict = "partial"
    else:
        verdict = "no effect"

    return {
        "verdict": verdict,
        "criteria": {n: (sb["first_attempt"][n]["passed"], sa["first_attempt"][n]["passed"])
                     for n in labels},
        "runs": sb["runs"],
        "revisions_needed": (sb["revisions_needed"], sa["revisions_needed"]),
        "attempts": (ub["attempts"], ua["attempts"]),
        "input_tokens": (ub["input_tokens"], ua["input_tokens"]),
        "output_tokens": (ub["output_tokens"], ua["output_tokens"]),
        "cost_usd": (ub["cost_usd"], ua["cost_usd"]),
        "improved": moved,
        "regressed": regressed,
    }


def format_comparison(c: Dict[str, Any], before: str = "v1", after: str = "v2") -> str:
    n = c["runs"]
    w = 22
    lines = [f"{'':<{w}}{before:>10}{after:>10}", ""]
    for name, (b, a) in c["criteria"].items():
        lines.append(f"{name:<{w}}{f'{b}/{n}':>10}{f'{a}/{n}':>10}")
    lines.append("")
    rb, ra = c["revisions_needed"]
    lines.append(f"{'revisions needed':<{w}}{f'{rb}/{n}':>10}{f'{ra}/{n}':>10}")
    for key, label in (("attempts", "attempts total"),
                       ("input_tokens", "input tokens"),
                       ("output_tokens", "output tokens")):
        b, a = c[key]
        lines.append(f"{label:<{w}}{b:>10,}{a:>10,}")
    b, a = c["cost_usd"]
    lines.append(f"{'cost, usd':<{w}}{b:>10.4f}{a:>10.4f}")
    lines.append("")
    lines.append(f"verdict: {c['verdict'].upper()}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Propose a harness improvement.")
    parser.add_argument("--label", help="which batch to analyze")
    parser.add_argument("--model", default=DEFAULT_ANALYST_MODEL)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="skip the analysis and compare two batches")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.compare:
        if args.label:
            parser.error("--label is not used with --compare")
        print(format_comparison(compare(*args.compare), *args.compare))
    else:
        if not args.label:
            parser.error("--label is required unless --compare is used")
        change, proposal = propose_harness_change(args.label, args.model)
        print(json.dumps(change.model_dump(), indent=2))
        print()
        print(proposal)
