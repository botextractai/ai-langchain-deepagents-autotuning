"""
The Requirements Review Agent, built on the Deep Agents harness.

`build_review_agent` builds the agent. With `rubric_iterations > 0` it also adds
a grader that can send the work back (the verification loop). `run_history.py`
runs it over a batch of documents and records the results.

The loop, the tool execution, and the message plumbing all come from
`create_deep_agent`. What is defined here is the part specific to this agent:
one custom tool, the AGENTS.md file it reads, the rubric, and the grader prompt.

Two things to know before reading further.

**The analysis is a file.** The agent writes `/analysis.md` with the harness
filesystem tools, and that file is the deliverable the grader reads.

**The instructions are a file.** `workspace/memories/AGENTS.md` is loaded into context at
the start of every run, so reading it tells you exactly what the agent was told.
It separates Procedure, which is fixed, from Review guidance, which is the half
that gets revised. `REVIEW_SYSTEM_PROMPT` does nothing but point at it.
"""

import json
import sys
import warnings
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    RubricMiddleware,
    create_deep_agent,
)
from deepagents.backends import FilesystemBackend
from deepagents.middleware.rubric import GRADER_SYSTEM_PROMPT
from dotenv import load_dotenv
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai:gpt-6.1-terra"

ANALYSIS_PATH = "/analysis.md"
AGENTS_PATH = "/memories/AGENTS.md"

# The agent's virtual filesystem. A real directory means you can open
# workspace/analysis.md and watch it fill in stage by stage, and can read the
# AGENTS.md file the agent was given. `virtual_mode=True` confines it here,
# so `/analysis.md` means `workspace/analysis.md` and paths above it are
# unreachable. That matters, because this repository holds a real .env.
WORKSPACE_DIR = "workspace"

# The one tracked copy of AGENTS.md, and the file to edit. It lives inside the
# default workspace rather than beside this module because `memory=` takes a
# path and reads it from the agent's own filesystem root, so the instructions
# have to sit under a root the agent can see. Everything else in `workspace/` is
# scratch and gitignored; this file is not.
#
# Not derived from WORKSPACE_DIR on purpose. `build_review_agent` takes a
# workspace per call and run_history gives each run its own, so a derived constant
# would read as "this run's instructions" while pointing somewhere else.
AGENTS_BASELINE = Path("workspace/memories/AGENTS.md")

# Model turns allowed in one run. Enough for the six review stages, grader
# revisions, simulated Jira retries, and a closing message.
MAX_TURNS = 40

# How many times the Jira fetch fails before succeeding. Deterministic rather
# than random so a run reproduces. 0 by default, a clean fetch; set it higher
# to exercise the agent's retry behaviour.
SIMULATED_JIRA_FAILURES = 0

_JIRA_SIMULATED_ERRORS = [
    "Error fetching ticket from Jira: 503 Service Unavailable. "
    "The Jira API is temporarily unreachable, please retry.",
    "Error fetching ticket from Jira: request timed out after 10s. Please retry.",
]

# The ticket ID used in every run's opening message. One constant, shared by
# whoever seeds the fake Jira backend and whoever tells the agent where to
# look, so the two can never drift apart.
DEFAULT_TICKET_ID = "AUTH-142"

# ---------------------------------------------------------------------------
# The rubric the verification loop grades against
# ---------------------------------------------------------------------------
#
# RubricMiddleware reference, including which of its arguments are supported:
# https://docs.langchain.com/oss/python/deepagents/rubric
#
# Three criteria, each with its threshold written into the criterion itself.
# The grader returns pass or fail per criterion, so "what does done mean" lives
# in this text rather than in a numeric threshold in Python.
#
# The hill-climbing analyst works from these too: a criterion that fails on most
# first attempts is what it diagnoses, and the change it proposes to AGENTS.md
# is the guidance that makes that criterion pass.
DEFAULT_RUBRIC = """\
- completeness: every clause of the requirement document is addressed somewhere \
in the analysis
- clarity: every bullet in the Ambiguities section has an indented "Resolved by:" \
line naming the specific fact or decision that would settle it
- depth: every bullet in the Ambiguities section has an indented "Edge case:" \
line naming a concrete production failure that ambiguity could cause"""

# Passing `system_prompt` replaces RubricMiddleware's default. Build on the
# installed default instead of copying it, so Deep Agents upgrades automatically
# carry their current verdict definitions and untrusted-transcript safeguards.
GRADER_PROMPT = GRADER_SYSTEM_PROMPT + """

## Additional instructions for this rubric

`<transcript>` is this agent's full conversation, every tool call included. \
That is what RubricMiddleware always hands you, but it is not the deliverable: \
the deliverable is a file the agent wrote, /analysis.md. The transcript can \
make a criterion look satisfied just from how the conversation reads, so do \
not judge from it. Call read_file on /analysis.md before judging anything, \
and judge only what that file actually contains.

Before passing a criterion, find the text in the file that satisfies it and \
quote it verbatim, with its line number. If you cannot quote it, it is not \
there, and the criterion fails. Do not pass a criterion because the file looks \
like the sort of file that would satisfy it.

Apply each criterion literally. Where a criterion names a specific labelled \
line, only that exact label satisfies it: a line with a different label is a \
different line, however similar its content. Where a criterion asks for \
coverage, check every item rather than sampling."""


# ---------------------------------------------------------------------------
# The one custom tool
# ---------------------------------------------------------------------------

# A minimal fake Jira backend: ticket ID -> document text. This stands in for
# the real external system the fetch tool queries. It comes pre-populated
# from jira_tickets.json, a handful of tickets including DEFAULT_TICKET_ID,
# so a run against the default ticket needs no setup first. Building the
# agent never touches this store; only `fetch_requirement` (called by the
# agent, during a run) and `seed_fake_jira` (for adding or overriding a
# ticket on the fly) do.
_FAKE_JIRA: Dict[str, str] = json.loads(
    (Path(__file__).resolve().parent / "jira_tickets.json").read_text(encoding="utf-8")
)


def seed_fake_jira(ticket_id: str, requirement_doc: str) -> None:
    """Register a document under `ticket_id` in the fake Jira backend,
    adding a new ticket or overriding one already in jira_tickets.json.

    Call this right before invoking the agent against that ticket, the same
    way the real document would already exist in Jira before anyone asked an
    agent to go fetch it. The agent itself is never built with a document; it
    only ever knows how to look one up by ID.
    """
    _FAKE_JIRA[ticket_id] = requirement_doc


def make_fetch_tool(simulated_failures: int = SIMULATED_JIRA_FAILURES):
    """Build the Jira fetch tool for one run.

    The attempt counter lives in this closure, so a fresh agent per run starts
    over from the first simulated failure. The document itself does not: the
    tool resolves it from the fake Jira backend by ticket ID when it is
    actually called, the same way a real Jira client would look one up.
    """
    attempts = {"n": 0}

    @tool
    def fetch_requirement(ticket_id: str) -> str:
        """Fetch the requirement document from Jira by ticket ID. Always call
        this first. This call can fail transiently (a busy or unreachable Jira
        instance). If it returns an error, call it again with the same ticket
        ID rather than giving up."""
        attempts["n"] += 1
        if attempts["n"] <= simulated_failures:
            return _JIRA_SIMULATED_ERRORS[(attempts["n"] - 1) % len(_JIRA_SIMULATED_ERRORS)]
        if ticket_id not in _FAKE_JIRA:
            return f"Error fetching ticket from Jira: no ticket '{ticket_id}' found."
        return _FAKE_JIRA[ticket_id]

    return fetch_requirement


# ---------------------------------------------------------------------------
# The system prompt only points at AGENTS.md, which holds everything.
# ---------------------------------------------------------------------------

# Agent instructions are in AGENTS.md which injected into the system prompt
# at startup, so instructions placed there carry the same weight as 
# instructions placed here.
REVIEW_SYSTEM_PROMPT = f"""Your instructions are in `{AGENTS_PATH}`, which is \
already in your context. Follow the Procedure section exactly, in order, and \
apply the Review guidance section as you go. Do not ask for instructions you \
already have."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Fail a hung OpenAI request instead of waiting forever. One agent turn can
# involve several model calls; this limit is per request, not per document.
REQUEST_TIMEOUT_SECONDS = 180


def resolve_model(model: str):
    """Build the chat model, with a request timeout and Responses API for GPT-6."""
    if not model.startswith("openai:"):
        return model
    kwargs = {"timeout": REQUEST_TIMEOUT_SECONDS}
    # GPT-6 tool calling requires the Responses API.
    if model.startswith("openai:gpt-6"):
        kwargs["use_responses_api"] = True
    return init_chat_model(model, **kwargs)


def is_request_timeout(exc: BaseException) -> bool:
    """True if this exception, or one of its causes, is a request timeout."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "timeout" in type(current).__name__.lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def abort_on_timeout(exc: BaseException, what: str, cleanup: str) -> None:
    """Print recovery steps for a timed-out OpenAI request, then exit."""
    if not is_request_timeout(exc):
        return
    print(
        f"\nOpenAI request timed out after {REQUEST_TIMEOUT_SECONDS}s while {what}.\n"
        f"Delete the incomplete output and try again:\n"
        f"  {cleanup}\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def require_agents_file(workspace_dir: str = WORKSPACE_DIR) -> Path:
    """Fail loudly if the agent's instructions are missing from its workspace.

    `memory=` ignores a missing file without warning, so the agent would run on
    the system prompt alone and produce a plausible bad review.
    """
    target = Path(workspace_dir) / AGENTS_PATH.lstrip("/")
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing, so the agent would run with no review "
            f"guidance. The tracked copy is {AGENTS_BASELINE}; check it was "
            "committed, or pass the instructions under test into this workspace."
        )
    return target


def build_review_agent(
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    simulated_failures: int = SIMULATED_JIRA_FAILURES,
    workspace_dir: str = WORKSPACE_DIR,
    rubric_iterations: int = 0,
    grader_model: Optional[str] = None,
    on_evaluation=None,
):
    """The whole agent: one custom tool, three file tools, AGENTS.md, one cap.

    Building the agent does not take a requirement document. It only wires up
    a `fetch_requirement` tool that knows how to look a ticket up; which
    ticket, and what document sits behind it, is decided later, at run time
    (see `seed_fake_jira`). The same built agent could review any ticket you
    seed it with.

    With `rubric_iterations > 0` a `RubricMiddleware` is added, which turns the
    execution loop into a verification loop: when the agent stops, a grader
    reads the finished file and can send it back for another pass. It only
    activates when a `rubric` is passed on invocation state, so including it
    changes nothing until a rubric is supplied. `on_evaluation` is called after
    each grading pass, before any revision.

    This agent never edits its own instructions. It has `edit_file` and
    `AGENTS.md` sits in the same writable filesystem, so writes under
    `/memories/` are denied outright, via `permissions=`. Changes to the
    instructions are proposed and applied from outside the agent.
    """
    # virtual_mode confines the agent to this directory: it sees workspace_dir
    # as "/", and paths above it are unreachable.
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)

    require_agents_file(workspace_dir)

    # The agent reads its instructions and never writes them. It has edit_file
    # and AGENTS.md is in the same filesystem, so deny writes there rather than
    # relying on the instructions telling it not to.

    permissions = [
        FilesystemPermission(
            operations=["write"], paths=[f"{PurePosixPath(AGENTS_PATH).parent}/**"], mode="deny"
        )
    ]

    middleware = [ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="end")]

    if rubric_iterations:
        # RubricMiddleware is flagged beta and warns on construction. suppress the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            middleware.append(
                RubricMiddleware(
                    model=resolve_model(grader_model or model),
                    max_iterations=rubric_iterations,
                    # The grader needs read_file, or it grades the conversation
                    # rather than the file the agent actually wrote. `tools` is
                    # the documented way to give it one.
                    tools=FilesystemMiddleware(
                        backend=backend, tools=["read_file"]
                    ).tools,
                    system_prompt=GRADER_PROMPT,
                    on_evaluation=on_evaluation,
                )
            )

    return create_deep_agent(
        model=resolve_model(model),
        tools=[make_fetch_tool(simulated_failures)],
        system_prompt=REVIEW_SYSTEM_PROMPT,
        # Loaded into the system prompt at the start of every run.
        memory=[AGENTS_PATH],
        backend=backend,
        permissions=permissions,
        middleware=middleware,
        # Not required for the rubric loop: it revises correctly without one,
        # measured. A checkpointer is what would carry a rubric into a
        # follow-up invocation, which nothing here does, so this only
        # matches the shape the deepagents docs show.
        checkpointer=InMemorySaver() if rubric_iterations else None,
    )
