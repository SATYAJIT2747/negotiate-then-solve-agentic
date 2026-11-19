# Negotiate, Then Solve

**Conflict-aware university timetabling with LLM agents and a constraint solver.**

University timetables keep changing after they are published. A teacher is away at a conference, a lab is
needed for an extra session, two people want the same room at the same hour. These requests arrive as free
text ("I'm away Tuesday and Wednesday next week, can my classes move?"), and when two of them cannot both be
met, there is no principled way to decide who gives way.

This project lets an LLM read and negotiate such requests **while a CP-SAT solver guarantees that nothing
invalid is ever offered or published**. The repository contains two complete versions of the system:

| Folder | Approach | In one line |
|---|---|---|
| [`pipeline/`](pipeline/) | **Fixed pipeline** | Every request goes through the same stages: read into a fixed set of intents → policy check → solver repair → grounded negotiation → human approval |
| [`agentic/`](agentic/) | **Tool-calling agent** | One LLM loop handles the whole request through ~30 typed tools (look up, draft, check with the solver, ask, negotiate, submit), behind a commit gate that re-validates every change |

Both folders are self-contained projects (each has its own `pyproject.toml`, tests, data, notebooks, web app
and a detailed README). `agentic/` also contains the full pipeline code, which it uses as its baseline.

---

## Contents

1. [The problem](#1-the-problem)
2. [The core idea: solver-grounded negotiation](#2-the-core-idea-solver-grounded-negotiation)
3. [Version 1: the fixed pipeline](#3-version-1-the-fixed-pipeline-pipeline)
4. [Version 2: the agentic system](#4-version-2-the-agentic-system-agentic)
5. [Data: how it was created](#5-data-how-it-was-created)
6. [Fine-tuning: LoRA adapters on Qwen3.5](#6-fine-tuning-lora-adapters-on-qwen35)
7. [The web app](#7-the-web-app)
8. [Evaluation and results](#8-evaluation-and-results)
9. [Running it](#9-running-it)
10. [Repository layout](#10-repository-layout)
11. [Ongoing work](#11-ongoing-work-independent-study)
12. [Credits](#12-credits)

---

## 1. The problem

- A department timetable (faculty, groups, rooms, 5 days × 8 slots) is built once, then changes **every week**.
- Change requests come in natural language, from people with different authority (a teacher may change only
  their own classes; the timetable office may change anything).
- Many requests are **infeasible** on their own: meeting one breaks someone else's constraint.
- Plain LLMs cannot be trusted here. In our tests an LLM-only scheduler produced invalid timetables 80% of
  the time, and options invented by an LLM were infeasible about half the time.

**Research question.** Can solver-derived conflict sets constrain an LLM negotiator so that it resolves
infeasible timetable requests with fewer negotiation rounds and lower disruption than LLM-invented options,
even when those invented options are filtered by the same solver?

## 2. The core idea: solver-grounded negotiation

When a request is infeasible:

1. **MUS** (minimal unsatisfiable subset): the solver finds exactly which constraints clash.
2. **MCS** (minimal correction sets): the solver lists the smallest sets of constraints that could be relaxed
   to resolve the clash. These are the only concessions that make sense.
3. **CP-SAT re-solve**: each concession is turned into a concrete, valid timetable (minimal-change repair).
4. **LLM negotiator**: offers **only these verified options** to the people involved, ranked by priority
   tiers and a **concession ledger** (who has given way before), with explanations grounded in the solver's
   output (no private reasons leaked).
5. **Human approval**: the timetable office approves before anything is published.

The LLM handles language and dialogue. The solver handles correctness. The LLM never writes the timetable.

## 3. Version 1: the fixed pipeline (`pipeline/`)

A layered architecture (L0–L6), each layer a separate module:

| Layer | What it does | Where |
|---|---|---|
| L0–L1 Channels & store | Requests arrive from a portal, email or messaging; one central request store with a lifecycle state machine | `src/pipeline/` |
| L2 Reading (System One / Two) | **System One:** a fast calibrated classifier (TF-IDF + linear SVM) routes easy requests. **System Two:** a fine-tuned LLM reads the request into a typed **intent language** (17 intent kinds: absences, moves, limits like "at most two days a week", relations like "the practical after the lectures", choices like "Tuesday, failing that Thursday"). A deterministic compiler turns intents into solver constraints, and checks repair or question bad readings. | `src/language/` |
| L3 Policy agent (RAG) | Retrieves the relevant rules from the handbook and circulars (BM25, dense embeddings, or hybrid by reciprocal rank fusion) and returns `allowed` / `needs_approval` / `forbidden` with cited rule IDs | `src/agents/policy.py` |
| L4 Solver | CP-SAT model with guarded constraints (`a_k ⇒ C_k`), full build, minimal-change repair, lexicographic priority tiers; MUS by deletion-based shrinking, MCS enumerated cheapest-first with solver-verified witnesses | `src/core/` |
| L5 Interpretation & negotiation | Resolution ladder, ranked grounded options, grounded explanations, concession ledger, reading stakeholders' replies as tool calls | `src/agents/` |
| L6 Approval & publishing | Coordinator approves; versioned timetables are published | `src/pipeline/`, `src/semester/` |

**Strength:** predictable, fast for the request types it knows, every step is measurable.
**Limit:** a request has to fit one of the fixed intents. Anything outside that set cannot be expressed.

## 4. Version 2: the agentic system (`agentic/`)

A rebuild around **one loop** (`agentic/src/nts/`): the model reads the department's state through tools,
makes changes in a private **draft**, asks the solver to check them, talks to people if needed, and submits.

**The rule everything rests on:** only `Store.commit` changes the state, and it re-checks every change with
the solver's validator whatever the agent claimed. So the CP-SAT guarantee still holds, and every model
mistake becomes an error the model sees (and a training example), never a bad timetable.

**Tools, by scope**

| Scope | Tools |
|---|---|
| read | `find`, `describe`, `classes_of`, `free_hours`, `who_is_free`, `free_rooms`, `changes` |
| draft | `constrain`, `move_class`, `retract`, `list_draft` |
| evaluate | `check_draft`: a repair solve of every affected timetable; on a clash, the clashing rules in words, with their owners |
| talk | `ask_requester`, `make_offer` (ask a rule's owner to waive it for the blocked weeks), `message_party` |
| policy | `rules`: search the handbook and cite a rule |
| end | `submit` (only a draft checked at the current version), `decline`, `escalate` |

**Packages:** `kernel` (solver and validator), `state` (append-only hash-chained event log, immutable world
versions, drafts, commit gate), `tools`, `people` (simulated stakeholders), `harness` (turn / token / time
budgets, context trimming, pause and resume while waiting for a reply), `telemetry` (spans per episode, turn,
model call, tool call), `data` (training data rebuilt from telemetry), `eval` (scenario families), `api`
(FastAPI with server-sent events).

Also built on top: **fairness** (the concession ledger is the event history itself; the agent asks whoever has
given way least first), **regulations** search, **semester rebuild** (re-solve the whole semester from every
standing rule and wish), and **club room bookings**.

**Strength:** general. Swaps, make-up classes and multi-part requests need no new intent type.
**Limit:** slower (many model turns and solver calls per request), and weaker on complex compositional
requests. These two are the focus of the [ongoing work](#11-ongoing-work-independent-study).

## 5. Data: how it was created

All data is synthetic or team-authored, so the ground truth is known by construction.

| Dataset | How it was built | Size |
|---|---|---|
| Department instance | Seeded generator: a CSE department with 30 faculty, 10 groups, 20 rooms, 5 days × 8 slots | 202 sessions |
| A. Negotiation | Planted conflicts of 7 kinds, hidden stakeholder flexibility, a solver **oracle** for the best achievable outcome | 60 scenarios |
| B. Requests / intents | Generated from known targets, then paraphrased by an LLM in varied registers (a rule-based check rejects paraphrases that change a day, number, name or room), plus ambiguous, typo and adversarial variants; 24 held-out test suites | 600 requests; 4,363 intent training rows (v12) |
| C. Policy (RAG) | A 32-rule handbook + 4 circulars; queries labelled with gold rule IDs | 106 + 40 paraphrased |
| D. Replies & safety | Stakeholder replies labelled with the correct tool call; prompt injections and authority violations | 100 replies; 48 safety cases |
| Agent trajectories | 19 request families generated from the department's own names; episodes run with a strong model, kept only when the **solver-checked outcome** is correct, exported from telemetry as SFT and DPO pairs | 38 scenarios per seed; v7: ~28 MB SFT + 13 MB DPO |

Agent training data comes out of the system itself: every episode is recorded as telemetry spans, labelled
by the scenario's verdict (or a human approval), and exported. Failed attempts become the "rejected" side
of DPO pairs. One labelling rule lives in code, so labels don't drift between data versions (firmness-label
agreement rose from 86.6% to 96.7%).

## 6. Fine-tuning: LoRA adapters on Qwen3.5

- **Base model:** Qwen3.5-4B (a 2B version was also trained; 4B won on action accuracy, 99.3% vs 96.5%).
- **Method:** LoRA (rank 16, alpha 16) with PEFT and TRL's `SFTTrainer`; DPO for the agent. Trained on
  Kaggle GPUs (notebooks in `notebooks/`, run queue and plans in `kaggle/`).
- **One base, one adapter per task group** (`adapters.toml`): `nts-intents` (reading requests),
  `nts-policy` (policy verdicts), `nts-negotiation` (replies and explanations), plus the agent adapters
  (`nts-agent-v1` … `v6-dpo`).
- **Serving:** exported to GGUF (Q5_K_M / Q8_0) and served by `llama-server` (llama.cpp), with LoRA chosen
  per request. Every LLM answer is cached by model, prompt, schema and temperature, so re-scoring an
  experiment costs nothing.
- Every LLM component also has a **deterministic fallback** (rule parser, template explanations, scripted
  simulators), so the whole system runs and is tested without any model.

## 7. The web app

A role-based web app (FastAPI + React + TypeScript + Tailwind) for teachers, the timetable office and clubs:

- **Request chat**: send a request in plain words and watch it being read, checked and negotiated
  (in `agentic/`, the chat is the agent loop and its turns stream live)
- **Inbox & approvals**: answer the system's questions and offers; the office approves changes
- **Timetable** views per teacher, group and room, with versions
- **Policy documents**: upload handbooks and circulars for the RAG policy check
- **Semester build & preferences**: rebuild the semester timetable from course offerings and plain-word wishes
- **Clubs**: room bookings
- **Fairness**: the concession ledger, i.e. who has given way and how often
- **Transparency, observability, experiments**: telemetry, traces and evaluation results in the browser
- **Human study**: rating interface for the explanation study

## 8. Evaluation and results

### Main experiment: solver-grounded negotiation

60 scenarios × 3 seeds = 180 paired runs, with four families of simulated stakeholders. "Ours" uses the
fine-tuned **4B** local model. The baselines use the much larger **Gemini 3.5 Flash-Lite**, so they are not
handicapped on language.

| System | n | Correct outcome | Agreement | Rounds | Oracle gap ↓ | Invalid options |
|---|---|---|---|---|---|---|
| B1 LLM-only | 20 | 0.0% | 0.0% | – | – | 80% of timetables |
| B2 Solver-only | 60 | 31.7% | 0.0% | – | 0.131 | – |
| B3 LLM options, unverified | 180 | 83.3% | 66.7% | 1.47 | 0.443 | 51.0% |
| B4 LLM options, solver-filtered | 180 | 75.6% | 51.1% | 1.48 | 0.628 | 5.7 filtered / scenario |
| **Ours (grounded)** | 180 | **89.4%** | **78.9%** | **1.38** | **0.041** | **0** |

McNemar test on correct outcome: Ours vs B4 p = 4.6×10⁻⁶, vs B3 p = 0.013. **No run of ours ever offered an
infeasible timetable.** Every miss (19/180) was a safe escalation. B2 shows that negotiation is needed: a
solver alone, imposing the best repair, is right only 31.7% of the time.

| Hypothesis | Result |
|---|---|
| H1 invented options waste effort | ✅ 51% of B3's options infeasible; B4 needs ~4 generation calls per scenario (rounds n.s.) |
| H2 grounded beats filtered | ✅ 89.4% vs 75.6% correct; oracle gap 0.04 vs 0.63 (rounds n.s.) |
| H3 grounded explanations are faithful and private | 🟡 0 vs 12 private-reason leaks; faithfulness 0.98 (human validation pending) |
| H4 ledger spreads burden more fairly | ❌ not yet: Gini 0.241 vs 0.224; one conflict per scenario gives the ledger little to act on |

Ablations (flat weights, free-text explanations, regex reply handler) keep resolution at 90.0%. They change
what stakeholders see and how the burden spreads.

### Reading requests (fine-tuned vs prompted)

| Reader (151 test requests) | Compile exact match | Atom F1 | Action accuracy |
|---|---|---|---|
| Gemini 3.5 Flash-Lite (prompted) | 92.4% | 0.977 | 99.3% |
| Qwen3.5-2B + LoRA | **97.8%** | **0.996** | 96.5% |
| Qwen3.5-4B + LoRA | **97.8%** | **0.996** | **99.3%** |

Intent reader, outcome match / silent wrong (%):

| Reader | test | test-typo | compositional | relations |
|---|---|---|---|---|
| Rule parser | 51.6 / 21.1 | 38.6 / 24.9 | 50.0 / 28.6 | 0.0 / 5.0 |
| v11 | 95.4 / 2.4 | 94.2 / 3.1 | 78.6 / 14.3 | 45.0 / 0 |
| **v12** | **98.1 / 1.9** | **97.6 / 1.9** | **85.7 / 0.0** | **77.5 / 0** |

### Policy check (RAG)

Hybrid retrieval recall@5 = 0.93.

| Model | Allow/deny accuracy | Deny recall (cited) | Make-up obligation recall |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | 98% | 0.80 | 0.21 |
| Qwen3.5-4B + LoRA | **99%** | **0.90** | **1.00** |

### Reading replies and safety

| Reply handler (100 replies) | Tool accuracy | Argument accuracy | Unsafe actions |
|---|---|---|---|
| Qwen3.5-4B base (prompted) | 65% | 100% | 0 |
| Gemini 3.5 Flash-Lite | **100%** | 85.5% | 0 |
| Qwen3.5-4B + LoRA | **100%** | **100%** | 0 |

All 48 safety cases (prompt injections, authority violations) were handled safely. None changed the
published timetable.

### The agentic coordinator

| Agent | Default (19 families) | Held-out | Stress | Compositional |
|---|---|---|---|---|
| Qwen3.5-4B base (prompted) | 66.7% | – | – | – |
| Gemini 3.5 Flash-Lite | 85.1% | – | 90.0% | – |
| nts-agent-v3 (SFT + DPO) | 98.2% | 95.0% | 59.4% | – |
| nts-agent-v4 (SFT) | **100%** | **100%** | **100%** | 42.9% |

A fine-tuned 4B agent beats a much larger prompted model on the standard families, but **compositional
requests remain the open problem**.

Full write-ups: `agentic/report/mid-submission.pdf`, the proposal in `agentic/proposal/`, and the design
notes in `agentic/docs/`.

## 9. Running it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and Node.js (for the web app). Run these inside
`pipeline/` or `agentic/`:

```sh
uv sync                              # install dependencies
uv run pytest -m "not slow"          # fast tests (< 1 min)
uv run python -m evaluation.demo     # generate a department, solve it, run a lab-contention scenario
uv run nts-web                       # web app on http://127.0.0.1:8000 (offline rule-based parser, no model needed)
uv run nts-web --parser local        # with the fine-tuned model behind llama-server
uv run nts-web --parser gemini       # with Gemini (GEMINI_API_KEY in .env)
```

Agentic system only (inside `agentic/`, with `PYTHONPATH=src`):

```sh
python -m nts init --store runs/agentic/demo.jsonl
python -m nts ask "I have a conference next week on Tuesday and Wednesday" --as "Dr. Iyer" --week 5 --show
python -m nts serve --store runs/agentic/demo.jsonl --model gemini --simulate --port 8100
python -m nts eval                   # scenario families
```

Each folder's own README covers local model serving (llama.cpp, LoRA per request), API keys and every
evaluation script in detail.

## 10. Repository layout

```
pipeline/            Version 1: the fixed pipeline
  src/core/          timetabling model: schemas, generator, CP-SAT solver, validator, MUS/MCS
  src/language/      LLM clients, System One / Two, intent reader and compiler, paraphrasing
  src/agents/        policy agent (RAG), negotiation, priorities, concession ledger, explanations
  src/pipeline/      channels, request store, orchestrator
  src/semester/      semester timetables, preferences, versions, club bookings
  src/training/      multi-task training data for the local model
  src/evaluation/    benchmark, baselines B1–B4, ablations, safety set, LLM judge, statistics
  src/web/           JSON API for the web app
  frontend/          React + TypeScript + Tailwind app
  notebooks/         Kaggle fine-tuning notebooks
  data/              handbook, request corpora, policy documents, training data
  tests/             pytest suite

agentic/             Version 2: everything above, plus the agent
  src/nts/           kernel, state, tools, people, harness, models, telemetry, data, agents, eval, api
  data/agent-v7/     agent SFT and DPO data
  kaggle/            training queue, plans, evaluation bundles
  docs/              design notes (agentic.md, improvement plan, labelling rules)
  report/, proposal/ project report and proposal
```

## 11. Ongoing work (independent study)

This project continues as an independent study (Jan–Apr 2027), built on the agentic version:

- **Speed:** profile where an agent request spends its time and cut it, with fewer model turns, fewer
  and faster solver calls (incremental repair, caching), and better model serving.
- **Complex queries:** move beyond the pipeline's fixed intents to multi-part, conditional and
  semester-wide requests, and close the compositional gap.
- **Fairness and memory** across a whole semester of requests, plus stronger and more consistent evaluation.

## 12. Credits

Originally built as the LMA Group Project (2026) by **Team Farali Chevdo**: Satyajit Priyadarshi,
Shubhadeep Mandal, Achyutananda Sahoo and Rakesh C. This repository is the starting point for Satyajit
Priyadarshi's independent study extending the system. The original team repository is
[Seronic2001/negotiate-then-solve](https://github.com/Seronic2001/negotiate-then-solve).
