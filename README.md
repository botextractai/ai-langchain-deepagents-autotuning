# Autotune LangChain Deep Agent prompts from past runs

An AI agent that **learns from its own past runs** and gets better at its job, using a hill-climbing loop built on LangChain Deep Agents.

## Purpose

The agent in this project reviews five software requirement documents and writes a structured analysis. A grader checks each analysis against a rubric. When a draft fails, the grader explains what's wrong and the agent revises it. This is the **verification loop**.

The verification loop has a limit: it improves **one run at a time**. Give the same agent a new document tomorrow and it starts from scratch. If it made a mistake yesterday, it makes the same mistake again and has to be corrected again, which costs time and tokens on every document.

This project closes one more loop, the **hill-climbing (or learning) loop**:

1. Run the agent on a batch of different documents and keep the full record of every run.
2. Look across all those runs and ask: _what does this agent keep getting wrong?_
3. If the same correction shows up across completely different documents, the cause is in the agent's instructions, not bad luck. An **analyst agent** reads the runs and proposes a fix to the instruction file, `AGENTS.md`.
4. A human reviews the proposal. If approved, the change is applied.
5. Run the same batch again and check that the agent actually got better.

In the saved example, the agent failed the **clarity** and **depth** criteria on its first attempt for every document, for the same reason: `AGENTS.md` never told it to include the `Resolved by:` and `Edge case:` lines the grader wanted. The analyst spotted this and proposed adding one short paragraph to `AGENTS.md`. After that change, all five software requirement documents passed every criterion on the first attempt, with no retries needed.

![Hill-climbing diagram](https://github.com/user-attachments/assets/77c32ca4-03e0-4156-8068-565a8d329c2f "Hill-climbing diagram")

Read the diagram from top to bottom. The five red requirement documents are the starting point; `AGENTS.md` supplies the instructions used to process each one. The **requirements review agent** (blue) does the work, while the **analyst agent** (orange) learns from the completed runs. The **yellow** boxes are the two decisions only you can make. Nothing changes `AGENTS.md` until you say yes to both. Answering **No** at either decision ends the cycle without changing it; a result with no measured gain also discards the copy. The step numbers match the steps below.

## How it works

### The three loops

| Loop                   | What it does                                                                                                        | Where it lives                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Agent loop**         | The review agent fetches a requirement document from a fake Jira and writes `/analysis.md`                          | `deepagent_helpers.build_review_agent`                                                 |
| **Verification loop**  | A grader checks the analysis against the rubric and sends it back with feedback until it passes (up to 3 revisions) | `RubricMiddleware` with `DEFAULT_RUBRIC` and `GRADER_PROMPT` in `deepagent_helpers.py` |
| **Hill-climbing loop** | An analyst agent reads many completed runs, finds recurring failures, and proposes one change to `AGENTS.md`        | `hill_climbing.py`                                                                     |

There is no real Jira connection, API key, or Jira network call. Before each run, `run_history.py` reads a document from `requirement_docs/` and stores it in an in-memory fake ticket store under `AUTH-142`. The agent then calls `fetch_requirement`, which returns that text. `jira_tickets.json` is an example of what that store can look like; during the walkthrough the seeded document overrides it. Simulated fetch failures are off by default.

The rubric has three criteria:

- **completeness:** every clause of the requirement document is addressed.
- **clarity:** every ambiguity has an indented `Resolved by:` line naming what would settle it.
- **depth:** every ambiguity has an indented `Edge case:` line naming a concrete production failure.

Two production models are used. The review agent and its grader run on `openai:gpt-6.1-terra`. The analyst runs on the stronger `openai:gpt-6.1-sol`, because finding patterns across many runs is the hardest reasoning in the project. OpenAI GPT-6 requires the Responses API for tool calling, so the code explicitly enables it when resolving either model.

The project-specific grader instructions are appended to Deep Agents' `GRADER_SYSTEM_PROMPT` at runtime. They therefore retain the installed library's current grading rules without keeping a copied version in this repository.

The design keeps a human in control. The review agent **cannot edit** its own instructions. The analyst **can only read** the batch it's analysing and **can only write** its proposal, so it can't peek at other batches that may already contain a fix. Models propose changes, a person approves them, and plain Python code applies them.

### What is a hill-climbing loop?

Picture climbing a hill in fog. You can't see the top, so you take one small step, check whether you're higher than before, and keep the step only if you are.

- **Where you stand** is the current version of `AGENTS.md`.
- **Your height** is how often the agent's first draft passes the grader, measured on the same batch of documents every time.
- **A step** is one small, evidence-backed edit to `AGENTS.md`.
- **Keep or undo:** re-run the batch. Keep the edit if the scores went up, and discard it if they didn't.

A single edit that doesn't help is simply discarded, and you try a different one. The loop is finished when either:

- **Nothing is left to fix.** Every criterion passes on the first attempt. The saved example gets there in one step.
- **You reach a local peak.** Several different edits in a row bring no improvement, even though some criteria still fail. Small instruction changes have stopped helping. Going further would need a bigger change, such as a different tool, grader or model, or a rewrite of `AGENTS.md`.

The loop only improves what the rubric measures. If the rubric misses something important, the agent won't get better at it.

## Setup

**Requires Python 3.12+.** On Apple macOS and Linux the launcher may be `python3` instead of `python`.

```
python -m pip install -r requirements.txt
```

### Required API key for this example

You need an OpenAI API key for this example. [Get your OpenAI API key here](https://platform.openai.com/login). Insert the OpenAI API key into the `.env.example` file and then rename this file to just `.env` (remove the ".example" ending).

## Quick look: the saved example (free)

The `run_history/` folder contains a completed cycle: `v1` (before the change), `v2` (after), and the analyst's proposal (`proposal_v1.md`). To compare them without calling OpenAI:

```
python hill_climbing.py --compare v1 v2
```

This is also a good check that your installation works. You should see:

```
                              v1        v2

completeness                 5/5       5/5
clarity                      0/5       5/5
depth                        0/5       5/5

revisions needed             5/5       0/5
...
verdict: SUCCESS
```

## Run one improvement cycle

Run every command **from the repo root**, because the code uses relative paths like `workspace/` and `run_history/`.

```
python cycle.py --before v1_live --after v2_live
```

This runs the whole cycle and stops twice to ask you a question:

1. Runs the baseline batch `v1_live`. If that batch already exists, it's reused instead.
2. Asks the analyst for a change and prints its proposal.
3. **Asks "Apply this change?"** If you answer no, it stops and nothing has changed. If yes, it applies the change to a copy of `AGENTS.md` and shows exactly what was added.
4. Runs the same five documents again with the new instructions, as batch `v2_live`.
5. Prints the before-and-after comparison and a verdict.
6. **If the change helped, asks "Make the new instructions the baseline?"** Only a yes changes `workspace/memories/AGENTS.md`.

A full cycle run calls OpenAI and costs money. In testing, a full cycle took over 10 minutes. Both step 1 and step 4 took about 5 minutes each.

Use new labels such as `v1_live` and `v2_live`, not `v1` or `v2`, which are the saved example. The script refuses to overwrite an existing `--after` batch.

To run another cycle after adopting a change, move the labels up one:

```
python cycle.py --before v2_live --after v3_live
```

Each command runs one cycle. It doesn't keep going on its own until results stop improving, because every cycle costs money and needs your approval. You decide whether to run the next one.

## The steps in detail

`cycle.py` runs these steps for you. This section explains what each one does and what to look at. You can also run the steps one at a time with the commands shown.

### Step 1: Run the baseline

```
python run_history.py --label v1_live
```

**What it does:** The review agent, with its grader, reviews each of the five documents in `requirement_docs/` using the current instructions in `workspace/memories/AGENTS.md`. This is the baseline you'll try to beat.

**What you get:** A progress line per document, then a grid showing which criteria passed or failed on the **first attempt**:

```
[1/5] 01_user_authentication ... satisfied, first attempt failed: 2 of 3
...
  document                    completeness    clarity         depth
  01_user_authentication      pass            FAIL            FAIL
  ...
  pass rate                   5/5             0/5             0/5

  documents needing a revision: 5/5
```

`satisfied` means the document passed in the end, after any revisions. `first attempt failed: 2 of 3` means two of the three criteria failed before the grader sent it back. The first attempt is what matters here: a criterion that fails on most documents points to a gap in the instructions, not bad luck.

The evidence is saved to `run_history/v1_live/`:

- `run_NN_<document>.json`: the full conversation, including the grader's feedback on each attempt
- `summary.json`: first-attempt pass rates per criterion
- `AGENTS.md`: a copy of the instructions this batch used

Per-run scratch files are kept separately in `workspace/run_history/v1_live/ws_NN/`. This keeps them outside the analyst's readable batch.

### Step 2: Ask the analyst for a fix

```
python hill_climbing.py --label v1_live
```

**What it does:** The analyst agent reads `summary.json`, the individual runs, and the current `AGENTS.md`. It only sees `run_history/v1_live/`, not other batches or earlier proposals. It looks for problems that recur across documents, ignores one-off failures, and proposes **one** change to the instructions. Its working copy is saved as `run_history/v1_live/proposal.md`.

**What you get:** Two files in `run_history/`, also printed to the screen:

- `proposal_v1_live.md`: a report for you to review. It contains a summary of the findings, the evidence quoted from the grader's feedback, the relevant part of the current `AGENTS.md`, the proposed text, and observations the analyst chose not to act on.
- `proposal_v1_live.json`: the same change in a structured form that code can apply:

| Field        | Meaning                                                           |
| ------------ | ----------------------------------------------------------------- |
| `target`     | What should change: `prompt`, `tool`, `grader` or `configuration` |
| `issue`      | The recurring problem the analyst found                           |
| `recurrence` | How many runs showed it, and which ones                           |
| `evidence`   | Direct quotes from the grader's feedback                          |
| `section`    | The heading in `AGENTS.md` the new text belongs under             |
| `change`     | The exact text to add                                             |
| `rejected`   | Problems noticed but not acted on, and why                        |

### Step 3: Review and apply the change

**Read `run_history/proposal_v1_live.md` first.** You are the human in the loop. Check that the evidence really supports the issue, and that the proposed text makes sense. If you don't agree, stop here.

If you approve, apply the change to a copy of the instructions:

```
python -c "from pathlib import Path; from hill_climbing import load_proposal, apply_change; change, _ = load_proposal('v1_live'); Path('run_history/v2_live_AGENTS.md').write_text(apply_change(change, Path('workspace/memories/AGENTS.md').read_text(encoding='utf-8')), encoding='utf-8')"
```

**What it does:** Inserts the proposed text at the end of the named section and saves the result as `run_history/v2_live_AGENTS.md`. It prints nothing if it works. The original `AGENTS.md` is left untouched.

It stops with an error in two cases, and `cycle.py` stops with the same message. Both mean you apply the change by hand instead:

- `target is '...'`: the proposal is about a tool, the grader or the configuration, not the prompt.
- `section '...' not found in AGENTS.md`: the analyst named a heading that doesn't exist. Fix the `section` value in `proposal_v1_live.json` and run the command again.

`cycle.py` shows you the added lines itself. When running step by step, you can see exactly what changed with this command (needs Git; install it separately if `git` is not already on your PATH, for example from [git-scm.com](https://git-scm.com/downloads)):

```
git diff --no-index workspace/memories/AGENTS.md run_history/v2_live_AGENTS.md
```

Added lines start with `+`. Git finishes with exit code 1 when the files differ; that is normal. On Windows it may also warn about line endings (`LF will be replaced by CRLF`).

### Step 4: Re-run with the new instructions

```
python run_history.py --label v2_live --agents-md run_history/v2_live_AGENTS.md
```

**What it does:** Reviews the **same five documents** again, this time with the updated instructions. Results are saved to `run_history/v2_live/`.

### Step 5: Compare before and after

```
python hill_climbing.py --compare v1_live v2_live
```

**What you get:** A table with first-attempt pass counts per criterion, how many documents needed a revision, and estimated tokens and cost, followed by a verdict:

| Verdict      | Meaning                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------- |
| `success`    | Nothing got worse, and no document needed a revision                                          |
| `partial`    | At least one criterion improved and none got worse, but some documents still needed revisions |
| `no effect`  | No criterion improved                                                                         |
| `regression` | At least one criterion got worse                                                              |

LLM output varies from run to run, so your numbers won't exactly match the saved example. In testing, the live cycle gave the same result as the saved one: clarity and depth went from 0/5 to 5/5, and revisions from 5/5 to 0/5.

The cost figure is an estimate of the review agent's conversation only. It leaves out the grader's and the analyst's model calls, so the real spend is higher. Check your OpenAI usage page for the actual cost. Models other than the two built-in ones show a cost of 0.

### Step 6: Keep or discard

`cycle.py` asks you this question. When running by hand:

**`success` or `partial`: keep the change** by making it the new baseline:

```
python -c "import shutil; shutil.copyfile('run_history/v2_live_AGENTS.md', 'workspace/memories/AGENTS.md')"
```

If you got `partial`, start another cycle with every label moved up one. With `cycle.py`, that's `python cycle.py --before v2_live --after v3_live`. By hand, analyse `v2_live`, change the step 3 command to use `load_proposal('v2_live')` and save to `run_history/v3_live_AGENTS.md`, run the batch as `v3_live`, then compare `v2_live v3_live`.

**`no effect` or `regression`: discard the change.** Your `AGENTS.md` was never modified, so just delete the copy and try a different change:

```
python -c "import shutil; from pathlib import Path; root = Path('run_history'); [shutil.rmtree(p, ignore_errors=True) for p in root.glob('*_live') if p.is_dir()]; shutil.rmtree('workspace/run_history', ignore_errors=True); [p.unlink(missing_ok=True) for p in list(root.glob('*_live_AGENTS.md')) + list(root.glob('proposal_*_live.*'))]"
```

## Cleaning up

Each cycle leaves its batches and proposals in `run_history/`. To remove every walkthrough batch (`v1_live`, `v2_live`, `v3_live`, …), their proposals, candidate `AGENTS.md` copies, and scratch files:

```
python -c "import shutil; from pathlib import Path; root = Path('run_history'); [shutil.rmtree(p, ignore_errors=True) for p in root.glob('*_live') if p.is_dir()]; shutil.rmtree('workspace/run_history', ignore_errors=True); [p.unlink(missing_ok=True) for p in list(root.glob('*_live_AGENTS.md')) + list(root.glob('proposal_*_live.*'))]"
```

Don't delete `run_history/v1`, `run_history/v2` or `proposal_v1.*`. They're the saved example.

## Options

- `cycle.py` accepts `--model` for the review agent and its grader, and `--analyst-model` for the analyst.
- `run_history.py` accepts `--model` for the review agent and its grader, for example `python run_history.py --label v1_strong --model openai:gpt-6.1-sol`. It also accepts `--docs <folder>` to review your own requirement documents instead of the five in `requirement_docs/`.
- `hill_climbing.py` accepts `--model` for the analyst.
