"""
Run the Requirements Review Agent over a set of documents and record everything.

This is the evidence half of the hill climbing loop. One batch of runs, all
produced by one version of the agent's instructions, with enough detail kept
that something reading them afterwards can work out what the agent is
systematically bad at.

Each run is one document and every attempt the grader demanded on it. It saves
the full conversation, every grading pass (results, explanation, per-criterion
pass or fail, and `gap` text), the analysis file at each pass and at the end,
and the AGENTS.md the run was given. The first attempt's results are the signal
for "is this systemic?"; later attempts show what the agent had to be corrected
on.
"""

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import deepagent_helpers as dh

DOCS_DIR = Path("requirement_docs")
OUTPUT_ROOT = Path("run_history")
WORKSPACE_ROOT = Path(dh.WORKSPACE_DIR) / "run_history"

# How many times the grader may send the work back. A run is one document and
# however many attempts the grader demands, so this bounds the run rather than
# describing it. Each attempt is recorded with the file it produced and the
# verdict on it.
RUBRIC_ITERATIONS = 3


@dataclass
class RunRecord:
    """One document, reviewed once and graded, with everything kept."""

    run_id: int
    doc_name: str
    label: str
    model: str
    requirement_doc: str
    instructions: str                    # the AGENTS.md this run was given
    messages: List[Dict[str, Any]]       # the whole conversation, untruncated
    grading: List[Dict[str, Any]]        # one entry per grading pass
    analysis_markdown: str               # the file at the end
    status: str
    started_at: str

    @property
    def first_pass(self) -> Optional[Dict[str, Any]]:
        return self.grading[0] if self.grading else None

    @property
    def graded(self) -> bool:
        """Whether this run actually received a verdict.

        A grader can fail: `RubricMiddleware` catches the error, records a pass
        with `result="grader_error"` and no criteria, and the run continues. That
        run carries no evidence either way and must not be counted as a pass.
        """
        p = self.first_pass
        return bool(p and p.get("criteria"))

    @property
    def first_pass_failures(self) -> List[int]:
        """Indexes of the criteria that failed on the agent's first attempt.

        This is the signal. A criterion that fails here in most runs is a
        property of the agent, not of one document.

        Only meaningful when `graded` is true. An ungraded run returns an empty
        list, which means "no evidence", not "nothing failed".

        Indexes, not names: the grader paraphrases the criterion text differently
        from run to run, sometimes dropping the leading label, sometimes swapping
        quote characters. Position is stable because the criteria come back in
        rubric order.
        """
        p = self.first_pass
        if not p:
            return []
        return [i for i, c in enumerate(p["criteria"]) if not c.get("passed")]


def _serialize_message(message: Any) -> Dict[str, Any]:
    """Keep everything a diagnosis might want, including full tool call args."""
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return {
        "type": getattr(message, "type", None),
        "content": content,
        "tool_calls": [
            {"name": c.get("name"), "args": c.get("args")}
            for c in (getattr(message, "tool_calls", None) or [])
        ],
        "tool_call_id": getattr(message, "tool_call_id", None),
        "status": getattr(message, "status", None),
    }


def run_one(
    run_id: int,
    doc_path: Path,
    label: str,
    model: str,
    agents_md: Path,
    rubric: str,
    workspace_root: Path,
) -> RunRecord:
    """Review one document and record the whole run."""
    requirement_doc = doc_path.read_text(encoding="utf-8")
    # Scratch work lives outside the saved batch. The analyst is rooted at
    # run_history/<label>, so it can read the evidence but not these files.
    workspace = Path(workspace_root) / label / f"ws_{run_id:02d}"
    shutil.rmtree(workspace, ignore_errors=True)

    # Seed this run's workspace from the instructions under test, so every run in
    # a label is produced by the same version and runs never see each other.
    target = workspace / dh.AGENTS_PATH.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(agents_md, target)

    grading: List[Dict[str, Any]] = []

    def record(evaluation) -> None:
        analysis = workspace / dh.ANALYSIS_PATH.lstrip("/")
        grading.append({
            "iteration": evaluation["iteration"],
            "result": str(evaluation["result"]),
            "explanation": evaluation.get("explanation", ""),
            "criteria": [dict(c) for c in evaluation.get("criteria", [])],
            # The file exactly as this pass saw it.
            "analysis_markdown": analysis.read_text(encoding="utf-8") if analysis.exists() else "",
        })

    # Seeded at invocation time: the agent is built with no knowledge of any
    # particular document, then told which ticket to fetch.
    dh.seed_fake_jira(dh.DEFAULT_TICKET_ID, requirement_doc)

    agent = dh.build_review_agent(
        model=model,
        max_turns=dh.MAX_TURNS,
        simulated_failures=dh.SIMULATED_JIRA_FAILURES,
        workspace_dir=str(workspace),
        rubric_iterations=RUBRIC_ITERATIONS,
        on_evaluation=record,
    )

    started = datetime.now(timezone.utc).isoformat()
    config = {
        # The checkpointer requires a thread_id.
        "configurable": {"thread_id": f"{label}-{run_id}-{uuid4().hex[:8]}"},
        "run_name": f"run_history {label} / {doc_path.stem}",
        "tags": ["loop-engineering", "run-history", label],
    }
    result = agent.invoke(
        {"messages": [{"role": "user", "content":
                       f"The requirement document is tracked in Jira as ticket "
                       f"{dh.DEFAULT_TICKET_ID}. Fetch it to begin."}],
         "rubric": rubric},
        config=config,
    )

    status = ""
    try:
        status = str(agent.get_state(config).values.get("_rubric_status") or "")
    except Exception:
        pass
    if not status and grading:
        status = grading[-1]["result"]

    analysis = workspace / dh.ANALYSIS_PATH.lstrip("/")
    return RunRecord(
        run_id=run_id,
        doc_name=doc_path.stem,
        label=label,
        model=model,
        requirement_doc=requirement_doc,
        instructions=target.read_text(encoding="utf-8"),
        messages=[_serialize_message(m) for m in result["messages"]],
        grading=grading,
        analysis_markdown=analysis.read_text(encoding="utf-8") if analysis.exists() else "",
        status=status,
        started_at=started,
    )


def generate_run_history(
    label: str,
    model: str = dh.DEFAULT_MODEL,
    docs_dir: Path = DOCS_DIR,
    agents_md: Optional[Path] = None,
    rubric: str = dh.DEFAULT_RUBRIC,
    output_root: Path = OUTPUT_ROOT,
    workspace_root: Path = WORKSPACE_ROOT,
    verbose: bool = True,
) -> List[RunRecord]:
    """Review every document once, under one version of the instructions.

    `label` names this batch, and is the directory the results land in. Run it
    once before a change to the instructions and once after, then compare the two
    summaries.

    `verbose` prints a progress line as each run finishes and the pass/fail grid
    at the end.
    """
    agents_md = Path(agents_md) if agents_md else dh.AGENTS_BASELINE
    docs = sorted(Path(docs_dir).glob("*.md"))
    if not docs:
        raise FileNotFoundError(f"No .md documents in {docs_dir}")

    out_dir = Path(output_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    # Record which instructions produced this batch. The source may already be
    # that file, when re-running a batch against instructions saved beside it.
    snapshot = out_dir / "AGENTS.md"
    if Path(agents_md).resolve() != snapshot.resolve():
        shutil.copyfile(agents_md, snapshot)

    history: List[RunRecord] = []
    for i, doc in enumerate(docs, 1):
        if verbose:
            print(f"[{i}/{len(docs)}] {doc.stem} ...", end=" ", flush=True)
        try:
            record = run_one(i, doc, label, model, agents_md, rubric, workspace_root)
        except Exception as e:
            dh.abort_on_timeout(
                e,
                f"reviewing {doc.stem} ({i}/{len(docs)}) in batch '{label}'",
                "python -c \"import shutil; "
                f"shutil.rmtree('run_history/{label}', ignore_errors=True); "
                f"shutil.rmtree('workspace/run_history/{label}', ignore_errors=True)\"",
            )
            raise
        history.append(record)
        path = out_dir / f"run_{i:02d}_{doc.stem}.json"
        path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        if verbose:
            failed = record.first_pass_failures
            print(f"{record.status}, first attempt failed: "
                  f"{len(failed)} of {len(record.first_pass["criteria"]) if record.first_pass else 0}")

    (out_dir / "summary.json").write_text(json.dumps(summarize(history), indent=2), encoding="utf-8")
    if verbose:
        print()
        print(format_summary(history))
    return history


def criterion_labels(rubric: str = dh.DEFAULT_RUBRIC) -> List[str]:
    """The rubric's own criterion names, in order.

    Taken from the rubric rather than from what the grader echoed back, because
    the grader's phrasing varies between runs.
    """
    return [line.strip().lstrip("-").split(":")[0].strip()
            for line in rubric.splitlines() if line.strip().startswith("-")]


def summarize(history: List[RunRecord],
              rubric: str = dh.DEFAULT_RUBRIC) -> Dict[str, Any]:
    """First-attempt pass rate per criterion, which is the systemic-failure signal."""
    labels = criterion_labels(rubric)
    graded = [r for r in history if r.graded]
    per_criterion = {}
    for i, label in enumerate(labels):
        passed = sum(1 for r in graded if i not in r.first_pass_failures)
        per_criterion[label] = {
            "passed": passed,
            "total": len(graded),
            "pass_rate": round(passed / len(graded), 3) if graded else None,
        }
    return {
        "label": history[0].label if history else "",
        "model": history[0].model if history else "",
        "runs": len(history),
        # Denominators are graded runs. A batch with errors is a weaker result
        # than a clean one and has to say so rather than average them away.
        "graded": len(graded),
        "grader_errors": len(history) - len(graded),
        "first_attempt": per_criterion,
        "revisions_needed": sum(1 for r in graded if r.first_pass_failures),
        "documents": [r.doc_name for r in history],
    }


def format_summary(history: List[RunRecord],
                   rubric: str = dh.DEFAULT_RUBRIC) -> str:
    """The grid: one row per document, one column per criterion, first attempt."""
    labels = criterion_labels(rubric)
    lines = ["first attempt, per document:", ""]
    lines.append("  " + f"{'document':<28}" + "".join(f"{l:<16}" for l in labels))
    for r in history:
        if r.graded:
            cells = ["FAIL" if i in r.first_pass_failures else "pass"
                     for i in range(len(labels))]
        else:
            cells = ["not graded"] * len(labels)
        lines.append("  " + f"{r.doc_name:<28}" + "".join(f"{c:<16}" for c in cells))
    s = summarize(history, rubric)
    rates = s["first_attempt"]
    lines.append("  " + f"{'pass rate':<28}"
                 + "".join(f"{str(rates[l]['passed']) + '/' + str(rates[l]['total']):<16}"
                           for l in labels))
    lines.append("")
    lines.append(f"  documents needing a revision: "
                 f"{s['revisions_needed']}/{s['graded']}")
    if s["grader_errors"]:
        lines.append(f"  grader errors, excluded from the rates above: "
                     f"{s['grader_errors']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--label", required=True,
                        help="names this batch, and the output directory")
    parser.add_argument("--model", default=dh.DEFAULT_MODEL)
    parser.add_argument("--docs", default=str(DOCS_DIR))
    parser.add_argument("--agents-md", default=None,
                        help=f"instructions to test (default: {dh.AGENTS_BASELINE})")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    generate_run_history(
        label=args.label,
        model=args.model,
        docs_dir=Path(args.docs),
        agents_md=Path(args.agents_md) if args.agents_md else None,
    )
