# Causal Fine-Tuning Engine

Research-oriented fine-tuning workspace for GLUE-style sequence classification with three implementation tracks in the codebase:

- standard LoRA training wired into the main runtime
- an optional causal LoRA orchestration pipeline wired behind a settings flag
- legacy or experimental Laplace and QLoRA components that exist as library modules but are not currently used by `main.py`

**Status**: Project implementation complete through Phase 7. Documentation refreshed from the current `src/` tree on March 20, 2026.

---

## What This Repository Does

At runtime, the project loads a GLUE dataset, tokenizes sentence pairs, wraps a Hugging Face classification model with LoRA adapters, and trains it with `transformers.Trainer`.

If `EXECUTE_CAUSAL_ENGINE=true`, the runtime switches from plain LoRA training to a causal orchestration path that:

1. identifies causal paths in the adapted model
2. allocates a causal sampling budget
3. starts an asynchronous weight sampler backed by a double buffer
4. applies sampled weights during training at a fixed interval
5. reports diagnostics, budget utilization, warm-up state, and marginal-likelihood estimates

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

### Present in `src/` and used transitively by the causal path

- `src/finetuner/laplace_engine.py`
- `src/finetuner/qlora_engine.py`
- `src/utils/math_utils.py`
- `src/utils/async_sampler.py`
- `src/utils/training_integrator.py`
- `src/utils/memory_manager.py`
- `src/utils/multiprocessing.py`
- `src/utils/hf_manager.py`

These modules are not imported directly by `main.py`, but they are part of the active causal execution path through `CausalTrainingOrchestrator` and are covered by the test suite.

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

- `MODEL_ID`
- `TASK_NAME`
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `OUTPUT_DIR`
- `EXECUTE_CAUSAL_ENGINE`
- `EXECUTE_LAPLACE`
- `EXECUTE_QLORA`

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

The current entry point is `main.py` and it is driven by settings rather than a real CLI parser.

### Standard LoRA run

```powershell
$env:HF_TOKEN = "<your-token>"
$env:MODEL_ID = "bert-base-uncased"
$env:TASK_NAME = "mrpc"
$env:EXECUTE_CAUSAL_ENGINE = "false"
uv run python main.py
```

### Causal orchestration run

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

Older planning and milestone documents remain useful as historical records, but they may describe intermediate project states.

---

## Governance

This project follows a constitution-governed workflow. The constitution lives in [.specify/memory/constitution.md](.specify/memory/constitution.md) (v1.1.1).

Key rules:
- `plan.md` and `tasks.md` must never contradict `src/` (Principle VI)
- Notebook results must be reproducible from the current `src/` (Principle VII)
- All code uses strict OOP/SOLID/DI and always runs under `uv` (Principle VIII)