# Causal Fine-Tuning Engine

Research-oriented fine-tuning workspace for GLUE-style sequence classification with three implementation tracks in the codebase:

- standard LoRA training wired into the main runtime
- an optional causal LoRA orchestration pipeline wired behind a settings flag
- legacy or experimental Laplace and QLoRA components that exist as library modules but are not currently used by `main.py`

**Status**: Project implementation complete through Phase 7. Documentation refreshed from the current `src/` tree on March 24, 2026.

## Project Governance

Delivery standards for architecture boundaries, UV-managed execution, testing,
memory-safe async behavior, and documentation synchronization are defined in
`.specify/memory/constitution.md`. Planning and implementation artifacts are
expected to satisfy the constitution gates before work is considered complete.

---

## What This Repository Does

The repository exposes **two entry points** for fine-tuning and evaluation:

### `main.py` — scripted single-task entry point

Loads a GLUE dataset, tokenizes sentence pairs, wraps a Hugging Face classification model with LoRA adapters, and trains with `transformers.Trainer`.

If `EXECUTE_CAUSAL_ENGINE=true`, switches from plain LoRA training to a causal orchestration path that:

1. identifies causal paths in the adapted model
2. allocates a causal sampling budget
3. starts an asynchronous weight sampler backed by a double buffer
4. applies sampled weights during training at a fixed interval
5. reports diagnostics, budget utilization, warm-up state, and marginal-likelihood estimates

### `finetunning.ipynb` — interactive benchmark notebook

A production-ready Jupyter notebook that benchmarks **three fine-tuning methods** across **three GLUE tasks** (MRPC, SST-2, QNLI) in a single session:

| Method | Description |
|--------|-------------|
| **LoRA Baseline** | Standard cross-entropy training on LoRA adapters |
| **Laplace-LoRA** | MAP training followed by K-FAC Hessian posterior for calibrated uncertainty |
| **Causal FT** | Do-calculus reweighted training for structural invariance |

The notebook is fault-tolerant by design: it prefers Google Drive when running on Colab, saves step-based checkpoints, resumes from the latest checkpoint on reconnect, and snapshots results to disk after every completed task.

---

## Current Runtime Scope

### Actively used by `main.py`

- `src/finetuner/data_loader.py`
- `src/finetuner/lora_engine.py`
- `src/finetuner/causal_engine.py`
- `src/finetuner/causal_training_orchestrator.py`
- `src/settings/settings.py`
- `src/utils/causal_sampler.py`
- `src/utils/logger.py`
- `src/utils/metrics.py`

### Actively used by `finetunning.ipynb` (additionally)

- `src/finetuner/laplace_engine.py` — Laplace-LoRA benchmark method
- `src/utils/memory_manager.py` — `MemoryOptimizer.log_resource_usage` called after each experiment
- `src/utils/multiprocessing.py` — `configure_spawn_context` called at notebook startup

### Used transitively by the causal orchestration path

- `src/finetuner/qlora_engine.py` *(not wired into either entry point; present as a library module)*
- `src/utils/math_utils.py`
- `src/utils/async_sampler.py`
- `src/utils/training_integrator.py`
- `src/utils/hf_manager.py`

These modules are part of the active causal execution path through `CausalTrainingOrchestrator` and are covered by the test suite.

---

## Source Tree Guide

### `src/finetuner/`

- `data_loader.py`: GLUE sentence-pair data loading and tokenizer application.
- `lora_engine.py`: wraps a base transformer with PEFT LoRA adapters.
- `laplace_engine.py`: Laplace approximation around a LoRA-adapted model.
- `qlora_engine.py`: 4-bit QLoRA preparation helper using bitsandbytes.
- `causal_engine.py`: causal-path discovery, budget allocation, warm-up, and marginal-likelihood validation.
- `causal_training_orchestrator.py`: top-level coordinator for causal training, async sampling, callback-based weight application, and diagnostics.

### `src/utils/`

- `logger.py`: global application logger with console and file handlers.
- `hf_manager.py`: singleton Hugging Face dataset manager with login and in-memory task caching.
- `math_utils.py`: causal and Laplace math primitives used by the analytical layers.
- `metrics.py`: GLUE metric evaluation, NLL, ECE, and natural indirect effect computation.
- `memory_manager.py`: RAM/VRAM reporting, cleanup, and double-buffer helpers.
- `multiprocessing.py`: `DoubleBuffer` implementation for latest-value inter-process handoff.
- `async_sampler.py`: background sampling process with legacy random and causal-aware modes.
- `causal_sampler.py`: causal-budget-aware weight generation for LoRA parameters.
- `training_integrator.py`: continuous weight application and causal budget monitoring.
- `__init__.py`: convenience exports for the causal utility layer.

### `src/settings/`

- `settings.py`: environment-backed runtime settings plus `CausalTrainingConfig` for orchestrator behavior.

---

## Architecture Summary

```text
main.py
  -> Settings / env configuration
  -> GLUEDataLoader
  -> FineTuningEngine
  -> Trainer
  -> optional CausalTrainingOrchestrator

CausalTrainingOrchestrator
  -> CausalMonteCLoRAEngine
  -> CausalWeightSampler
  -> BackgroundSampler
  -> DoubleBuffer
  -> ContinuousWeightApplier
  -> TrainingBudgetMonitor
  -> WeightApplicationCallback
```

The codebase follows a composition model:

- configuration lives in `settings.py`
- model adaptation lives in `finetuner/`
- operational support and mathematical helpers live in `utils/`
- orchestration composes components rather than subclassing them

---

## Environment Requirements

### Python and package manager

- Python 3.12+
- `uv`

### Required environment variables

- `HF_TOKEN`: required because `src/utils/hf_manager.py` logs into Hugging Face when the singleton dataset manager is created

### Common optional environment variables

**Used by `main.py` and `finetunning.ipynb`**:

- `MODEL_ID`
- `TASK_NAME`
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `OUTPUT_DIR`
- `LOGGING_DIR`
- `LORA_RANK`
- `LORA_ALPHA`
- `LORA_DROPOUT`
- `EXECUTE_CAUSAL_ENGINE`
- `EXECUTE_LAPLACE`
- `EXECUTE_QLORA`
- `TOTAL_CAUSAL_BUDGET`
- `ASYNC_MAX_STEPS`
- `APPLY_INTERVAL`

**Used by `finetunning.ipynb` only** (multi-task benchmark extras):

- `GRADIENT_ACCUMULATION_STEPS` — gradient accumulation steps per optimizer update
- `SAVE_STEPS` — checkpoint save frequency in steps (default: 100)
- `EVAL_STEPS` — evaluation frequency in steps (default: 100)
- `LOGGING_STEPS` — log frequency in steps (default: 25)
- `MAX_STEPS` — hard step limit overriding epoch count (-1 = unlimited)
- `RESUME_POLICY` — checkpoint resume strategy (`latest`, `auto`, `true`, or `false`)
- `SEED` — random seed for reproducibility (default: 42)

---

## Setup With UV

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync
```

If the project is already configured, the shorter workflow is:

```powershell
.\.venv\Scripts\Activate.ps1
uv sync
```

---

## Running The Project

Both entry points are settings-driven rather than argparse-based.

### `main.py` — Standard LoRA run

```powershell
$env:HF_TOKEN = "<your-token>"
$env:MODEL_ID = "bert-base-uncased"
$env:TASK_NAME = "mrpc"
$env:EXECUTE_CAUSAL_ENGINE = "false"
uv run python main.py
```

### `main.py` — Causal orchestration run

```powershell
$env:HF_TOKEN = "<your-token>"
$env:MODEL_ID = "bert-base-uncased"
$env:TASK_NAME = "mrpc"
$env:EXECUTE_CAUSAL_ENGINE = "true"
uv run python main.py
```

What changes when causal mode is enabled:

- `CausalMonteCLoRAEngine` is created
- `CausalWeightSampler` generates causal-budget-scaled weights
- `CausalTrainingOrchestrator.prepare()` initializes the async pipeline
- training proceeds through a callback that periodically applies weights to the live model
- diagnostics are logged after training, including async sampler health and callback-visible errors

**Note**: the causal config values in `main.py` (`sample_budget`, `total_causal_budget`, `async_max_steps`, `apply_interval`) are currently hardcoded at 1000 / 100 / 10. The notebook reads these from env vars (`TOTAL_CAUSAL_BUDGET`, `ASYNC_MAX_STEPS`, `APPLY_INTERVAL`).

### `finetunning.ipynb` — Interactive benchmark

Open the notebook in Jupyter or VS Code and run cells in order:

1. Cell 2 (Bootstrap) — installs `uv` on Colab, no-op locally
2. Cell 3 (Environment detection) — resolves `base_dir`, `output_dir`, etc.
3. Cell 4 (Project install) — reads `pyproject.toml`; on Colab installs editable, on local shows installed packages
4. Cell 5 (Runtime env) — sets all env vars from Python dict; set `HF_TOKEN` here
5. Cell 6 (Imports) — imports the full `src/` surface
6. Cell 7 (Config) — builds `LoRAExperimentConfig` instances for mrpc, sst2, qnli
7. Cell 8 (Data helpers) — wraps `GLUEDataLoader` for notebook use
8. Cell 10 (Utilities) — runtime-state persistence, checkpoint helpers, `build_training_args`
9. **Cells 12–13** — LoRA Baseline training and results
10. **Cells 15–16** — Laplace-LoRA training and results
11. **Cells 18–19** — Causal FT training and results
12. Cell 21 — Benchmark results table and bar charts

---

## Testing And Validation

### Full test suite

```powershell
uv run python -m pytest tests/ -q --tb=no
```

### Focused suites

```powershell
uv run python -m pytest tests/test_causal_engine.py -v
uv run python -m pytest tests/test_causal_sampler.py -v
uv run python -m pytest tests/test_training_integrator.py -v
uv run python -m pytest tests/test_causal_training_orchestrator.py -v
uv run python -m pytest tests/test_causal_flow_e2e.py -v
uv run python -m pytest tests/test_cross_cutting_validation.py -v
uv run python -m pytest tests/test_finetunning.py -v
```

### Linting and syntax checks

```powershell
uv run ruff check src tests
uv run python -m compileall src
```

---

## Known Implementation Constraints

These notes come directly from reviewing the current `src/` code.

### Data loader assumptions

`GLUEDataLoader` now uses a Strategy-based task resolver for `mrpc`, `sst2`, and `qnli`, so schema handling is centralized in `src/finetuner/data_loader.py` and shared by both `main.py` and the notebook workflow.

### Runtime mode selection

`main.py` uses environment-backed settings, not argparse flags. Earlier docs described CLI flags, but that is not how the current runtime works.

### Legacy modules

`laplace_engine.py` and `qlora_engine.py` are part of the codebase, but they are not exercised by `main.py` today.

### Hugging Face login behavior

`hf_manager.py` creates a singleton client at import time and calls `login(token=settings.hf_token)`. That means `HF_TOKEN` must be available before modules importing the manager are loaded.

### Causal path sensitivity

`CausalMonteCLoRAEngine._compute_causal_sensitivity()` is currently a simplified gradient-magnitude proxy rather than a full intervention-based causal estimator.

### Device-aware causal execution

The causal path now supports mixed deployment conditions safely:

- training tensors are moved to the model device during causal-path discovery, warm-up, and marginal-likelihood validation
- if CUDA is available, `settings.device` resolves to `cuda`; otherwise the runtime stays on `mps` or `cpu`
- sampled weights remain on CPU across the multiprocessing boundary and are moved onto the target training device only at application time

### Multiprocessing safety and error capture

The async sampling path now uses an explicit `spawn` multiprocessing context, a spawn-safe `DoubleBuffer`, and parent-visible sampler health checks.

- worker failures are surfaced to the parent process through `BackgroundSampler.raise_if_failed()`
- `WeightApplicationCallback` checks sampler health during training and records callback-visible errors
- `CausalTrainingOrchestrator.get_diagnostics()` now includes `async_sampler_status` and `callback_error`

### Math utilities

`math_utils.py` contains the core causal and Laplace helper primitives used across the project. After the strict Phase 6 cleanup pass, duplicate helper definitions were removed and the module is lint-clean.

### Validation snapshot

Current validation snapshot on March 31, 2026 (governance/documentation-alignment re-audit):

- `uv run ruff check src tests` — clean
- `uv run pytest -q` — 129 tests passing
- Constitution v1.1.1 — all 11 documented gaps resolved (H2 code fix applied)
- Notebook imports — 100% aligned with `src/`
<<<<<<< HEAD

> Note: the snapshot date refers to the `src/` and test suite state. The notebook was reviewed and documentation updated on March 24, 2026.
=======
>>>>>>> governance/documentation-alignment

---

## Recommended Reading

- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md): entry point for the repo docs set
- [SOURCE_CODE_DOCUMENTATION.md](SOURCE_CODE_DOCUMENTATION.md): module-by-module reference for `src/`
- [MULTIPROCESSING_DUMMIES_GUIDE.md](MULTIPROCESSING_DUMMIES_GUIDE.md): beginner-friendly guide to multiprocessing, traceability issues, and safe fixes in this repo
- [specs/feature_causal_lora/tasks.md](specs/feature_causal_lora/tasks.md): implementation status and task history
- [PHASE_6_COMPLETION.md](PHASE_6_COMPLETION.md): final polish-phase summary
- [CONSISTENCY_REPORT.md](CONSISTENCY_REPORT.md): documentation/code alignment audit results
- [NEW_FEATURE_SDD_GUIDE.md](NEW_FEATURE_SDD_GUIDE.md): step-by-step guide for adding a new feature using Spec-Driven Development

---

## Documentation Policy In This Repo

The current source of truth for implemented behavior is:

1. `src/`
2. `README.md`
3. `SOURCE_CODE_DOCUMENTATION.md`
4. `specs/feature_causal_lora/tasks.md`
5. `CONSISTENCY_REPORT.md` — alignment audit with gap inventory and score

<<<<<<< HEAD
<<<<<<< HEAD
After any change to `src/` or canonical docs, re-run `/speckit.analyze` and update
`CONSISTENCY_REPORT.md`. Alignment score must remain ≥ 90% (Principle VI).

=======
>>>>>>> governance/documentation-alignment
=======
>>>>>>> governance/documentation-alignment
Older planning and milestone documents remain useful as historical records, but they may describe intermediate project states.

---

<<<<<<< HEAD
<<<<<<< HEAD
## Creating a New Feature (Spec-Driven Development Workflow)

This project uses **Spec-Driven Development (SDD)** via the `speckit` agent suite. Every new feature must pass through the following stages before implementation begins.

### Prerequisites

- You are on a dedicated feature branch.
- `uv sync` has been run and the virtual environment is active.
- Governance branch reviewed: `specs/feature_causal_lora/plan_documentation_alignment.md`.

---

### Stage 1 — Branch

```powershell
git checkout -b feature/<your-feature-name>
```

Convention: `feature/verb-noun` (e.g. `feature/add-qnli-strategy`).

---

### Stage 2 — Specify

Open GitHub Copilot Chat and run:

```
/speckit.specify
```

Describe the feature in plain language. The agent creates:

```
specs/feature_<your-feature-name>/spec.md
```

The spec captures: objective, scope, functional requirements, acceptance criteria,
and user stories. It is the single source-of-truth for *what* you are building.

---

### Stage 3 — Clarify (optional but recommended)

```
/speckit.clarify
```

The agent asks up to 5 targeted questions about underspecified areas, then encodes
the answers back into `spec.md`. Run this before planning.

---

### Stage 4 — Plan

```
/speckit.plan
```

The agent reads `spec.md` and produces:

```
specs/feature_<your-feature-name>/
├── plan.md            ← design decisions, phased tasks, constitution check
├── research.md        ← background analysis
├── data-model.md      ← class/data contracts
├── quickstart.md      ← fast-start usage guide
└── contracts/         ← interface contracts (optional)
```

**Constitution gate** runs automatically inside `plan.md`. All 8 principles must
pass before Phase 1 outputs are approved.

---

### Stage 5 — Generate Tasks

```
/speckit.tasks
```

Produces a dependency-ordered `tasks.md` with every task mapped to a user story,
an implementation path, and a validation command. No task may reference a symbol
not present in `src/`.

---

### Stage 6 — Analyze Artifacts

```
/speckit.analyze
```

Cross-checks `spec.md`, `plan.md`, and `tasks.md` for:

- API signatures that diverge from `src/` (Principle VI)
- Missing OOP/SOLID patterns (Principle VIII)
- Undocumented tests or missing artifacts (Principle V)

Resolve all CRITICAL and HIGH findings before proceeding.

---

### Stage 7 — Implement

```
/speckit.implement
```

Executes each task in `tasks.md` in order. Follow these rules during implementation:

| Rule | Detail |
|------|--------|
| One class per file | `src/finetuner/` and `src/utils/` follow single-responsibility |
| Protocol-first | Define a `Protocol` before the concrete class |
| Dependency injection | Engines receive collaborators via constructor |
| Settings via `Settings` | Use `src/settings/settings.py`; never hardcode |
| All commands via `uv` | `uv run pytest`, `uv add`, `uv sync` |

---

### Stage 8 — Test

```powershell
uv run pytest                                 # full suite
uv run pytest tests/test_<your_module>.py -v  # targeted
```

New behavior **must** have tests. Documentation-only changes may skip tests (Principle III).

---

### Stage 9 — Update Docs & Re-analyze

Update `README.md` → `Source Tree Guide` and `SOURCE_CODE_DOCUMENTATION.md` with
any new classes or modules.

Then run `/speckit.analyze` once more to confirm 0 CRITICAL gaps and alignment
score ≥ 90%.

---

### Stage 10 — Update Constitution (if governance changed)

```
/speckit.constitution
```

Only needed if the feature introduces a new global contract (e.g. a new required
pattern, a new entry point, a new hardware constraint).

---

### Stage 11 — Commit & Merge

```powershell
git add src/ tests/ specs/ README.md SOURCE_CODE_DOCUMENTATION.md
git commit -m "feat(<scope>): <what it does>"
```

Before merging, verify:

- [ ] `uv run pytest` passes
- [ ] `CONSISTENCY_REPORT.md` updated (or alignment check re-run)
- [ ] `README.md` Source Tree Guide reflects new modules
- [ ] All spec artifacts reference actual `src/` symbols
- [ ] No `pip install` or bare `python` calls in new code or docs

---

### Quick Reference Card

| Stage | Command | Output |
|-------|---------|--------|
| Specify | `/speckit.specify` | `spec.md` |
| Clarify | `/speckit.clarify` | Updated `spec.md` |
| Plan | `/speckit.plan` | `plan.md`, `research.md`, `data-model.md` |
| Tasks | `/speckit.tasks` | `tasks.md` |
| Analyze | `/speckit.analyze` | Gap report |
| Implement | `/speckit.implement` | Code in `src/`, tests in `tests/` |
| Checklist | `/speckit.checklist` | Feature-specific QA checklist |
=======
=======
>>>>>>> governance/documentation-alignment
## Governance

This project follows a constitution-governed workflow. The constitution lives in [.specify/memory/constitution.md](.specify/memory/constitution.md) (v1.1.1).

Key rules:
- `plan.md` and `tasks.md` must never contradict `src/` (Principle VI)
- Notebook results must be reproducible from the current `src/` (Principle VII)
<<<<<<< HEAD
- All code uses strict OOP/SOLID/DI and always runs under `uv` (Principle VIII)
>>>>>>> governance/documentation-alignment
=======
- All code uses strict OOP/SOLID/DI and always runs under `uv` (Principle VIII)
>>>>>>> governance/documentation-alignment
