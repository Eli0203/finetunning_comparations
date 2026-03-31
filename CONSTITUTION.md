# Project Constitution

**Repository**: `finetunning`  
**Last Modified**: 2026-03-31  
**Version**: 1.1.0  
**Maintainer**: Senior ML Engineer

---

## 1. Purpose

This document defines the engineering constraints, architectural principles, and compliance rules that govern every change merged into this repository. All contributors and automated agents **must** verify compliance before committing.

---

## 2. Hardware & Resource Constraints (HARD LIMITS)

### 2.1 Memory Ceiling — 14 GB RAM

| Rule | Detail |
|------|--------|
| **Max resident memory** | ≤ 14 GB across all processes at any time |
| **Dataset loading** | Use `datasets.Dataset` lazy loaders or `torch.utils.data.DataLoader` generators — never load a full split into a Python list |
| **Model loading** | `bert-base-uncased` (~440 MB) + LoRA adapters (~5–10 MB). Target total < 2 GB for training run |
| **CUDA cache** | Call `torch.cuda.empty_cache()` after each eval/test pass |
| **QLoRA exception** | QLoRA (4-bit NF4) requires a CUDA GPU with ≥ 8 GB VRAM; still must fit within the 14 GB ceiling when combined with host RAM. See §2.3 |

**Prohibited patterns**:
- Loading a full model in full (FP32) precision when a quantised or LoRA alternative exists
- `dataset.to_pandas()` or similar full-materialisation calls on GLUE splits
- Spawning multiple training runs in the same process without releasing models between them
- Initializing large `float32` tensors without pre-allocation memory checks against the 14 GB combined cap

### 2.2 Compute — Device Agnosticism

All fine-tuning and inference code **must** be device-agnostic.

```python
# Canonical device detection — replicate everywhere device is needed
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

`src/settings/settings.py` exposes this via `Settings.device` (a `@property`). All engine and orchestrator classes must accept `device` via dependency injection rather than hard-coding `"cuda"`.

### 2.3 QLoRA Exception

`src/finetuner/qlora_engine.py` is the **only** module that depends on CUDA explicitly.

| Requirement | Detail |
|-------------|--------|
| CUDA GPU | ≥ 8 GB VRAM (NF4 4-bit quantisation) |
| `bitsandbytes` | `>=0.43.1` — the Windows-compatible build is distributed via PyPI as of 0.43.0; no Linux-only lock |
| Guard | Caller must check `torch.cuda.is_available()` before constructing `QLoraEngine` |
| Quantisation flag | `load_in_4bit=True` is mandatory in `src/finetuner/qlora_engine.py` |
| Fallback policy | CPU fallback is prohibited for QLoRA execution paths |
| RAM budget | Quantised BERT-base ≈ 150 MB on GPU, leaving ≥ 12 GB for host RAM |

### 2.4 Engine Load Policies (MANDATORY)

| Engine | Requirement |
|--------|-------------|
| `src/finetuner/qlora_engine.py` | Strictly enforce `load_in_4bit=True`; never permit CPU fallback path |
| `src/finetuner/lora_engine.py` | Implement and use dynamic `device_map` resolution for CPU/GPU contexts |

### 2.5 Float32 Initialization Restriction

No process may initialize `float32` tensors whose total combined RAM footprint exceeds 14 GB.

Implementation requirement:
- Any tensor allocation path that can approach the ceiling must estimate bytes before allocation (`numel * element_size`) and short-circuit with a controlled error when the combined process budget would be exceeded.

---

## 3. Cross-Platform Compatibility

| Rule | Implementation |
|------|---------------|
| **Path handling** | Always use `pathlib.Path` or `os.path` — never string concatenation with `/` or `\\` |
| **Scripts** | Shell scripts live under `scripts/` and have both `.sh` and `.ps1` variants |
| **Line endings** | `.gitattributes` enforces `text=auto` — CRLF on Windows checkout, LF in the repo |
| **`bitsandbytes`** | `>=0.43.1` ships Windows CUDA wheels; no OS lock in `pyproject.toml` |
| **Environment variables** | Loaded from `.env` via `pydantic-settings`; no hard platform assumptions |

---

## 4. Architectural Principles

### 4.1 SOLID Compliance

| Principle | Application |
|-----------|-------------|
| **SRP** | Each engine class implements exactly one concern: `FineTuningEngine` (LoRA hooks), `LaplaceLoRAEngine` (curvature), `CausalMonteCLoRAEngine` (causal analysis), `CausalTrainingOrchestrator` (coordination) |
| **OCP** | GLUE task loading extended via `GLUETaskStrategy` protocol — add a new task by registering a new `BaseGLUEStrategy` subclass without touching existing code |
| **LSP** | Strategies are interchangeable through `GLUETaskStrategy` protocol |
| **ISP** | `GLUEConfig` protocol exposes only `task_name`, `batch_size`, `max_seq_length` — callers depend on nothing else |
| **DIP** | All engines injected at construction; no `import`-time singletons except `settings` |

### 4.2 Causal Training Stack

```
CausalTrainingOrchestrator
  ├── CausalMonteCLoRAEngine   (causal path discovery, budget allocation)
  ├── CausalWeightSampler      (budget-aware weight tensors, spawn-safe)
  ├── BackgroundSampler        (multiprocessing subprocess manager)
  ├── DoubleBuffer             (O(1) lock-free weight exchange)
  ├── ContinuousWeightApplier  (rate-limited application during training)
  ├── TrainingBudgetMonitor    (utilisation tracking)
  └── WeightApplicationCallback (HF Trainer integration)
```

Multiprocessing constraints:
- Explicit `spawn` context only — no `fork` (CUDA incompatibility on Linux with fork)
- CPU-only tensors across process boundaries — CUDA tensors never pickled
- Parent-visible worker error capture via `BackgroundSampler.raise_if_failed()`

### 4.3 Settings & Secrets

- `HF_TOKEN` and all credentials loaded exclusively from environment variables or Colab Secrets
- **Never** hardcode tokens, passwords, or API keys in source files or notebooks
- `src/settings/settings.py` is the single settings authority; notebooks set `os.environ` before importing `src.*`

### 4.4 Python Memory-Efficiency Mandate

All Python code in this repository must be implemented with high memory efficiency by default.

Required practices:
- Prefer streaming, iterators, generators, and chunked processing over full in-memory materialisation.
- Reuse buffers/tensors where safe; avoid duplicate copies of large arrays, tensors, or datasets.
- Release references promptly and clean temporary structures after use.
- Choose the lowest precision and representation that satisfies correctness and stability constraints.

---

## 5. Dependency Policy

### 5.1 Core dependencies (`pyproject.toml`)

| Package | Constraint | Reason |
|---------|-----------|--------|
| `torch` | `>=2.3.0` (CPU wheel via pytorch-cpu index) | NumPy 2.0 header alignment |
| `transformers` | `>=4.41.0` | NumPy 2.0 compat |
| `peft` | `>=0.11.0` | Modern API |
| `bitsandbytes` | `>=0.43.1` | Windows wheels available; QLoRA only |
| `numpy` | `>=2.0.0` | Explicit 2.0 to align C-extension headers |
| `datasets` | `>=2.19.0` | PyArrow 16+ integration |

### 5.2 Adding a new dependency

Before adding any package:
1. Verify it ships universal wheels (no OS-specific binaries) via PyPI
2. Estimate peak memory contribution — reject if combined training peak > 14 GB
3. Run `uv run ruff check src tests` — ruff must remain clean
4. Run `uv run pytest tests/ -x -q` — all tests must pass

---

## 6. Git Commit Protocol

Every commit message **must** confirm resource compliance using the format:

```
<type>(<scope>): <description> (verified <14GB RAM & Win/Lin compat) - YYYY-MM-DD
```

### Commit types

| Type | Use when |
|------|----------|
| `feat` | New feature or experiment |
| `fix` | Bug fix |
| `refactor` | Internal restructuring, no behaviour change |
| `docs` | Documentation only |
| `test` | Test additions or corrections |
| `chore` | Dependency bumps, tooling, CI |

### Examples

```
feat(causal): add warmup phase to CausalTrainingOrchestrator (verified <14GB RAM & Win/Lin compat) - 2026-03-31
fix(laplace): accumulate_curvature hook cleanup on exception (verified <14GB RAM & Win/Lin compat) - 2026-03-31
docs(notebook): Colab quick start + experiment cell alignment (verified <14GB RAM & Win/Lin compat) - 2026-03-31
```

### Prohibited commit messages

- Single-word messages (`DOCUMENTATION`, `fix`, `update`)
- Messages with no compliance confirmation
- Force-pushes to `main` or `freeze-base` without a PR

---

## 7. Violation Response Protocol

If a proposed change violates any rule in this document:

| Violation | Required action |
|-----------|----------------|
| Would exceed 14 GB RAM | Refuse; suggest quantised/sharded alternative |
| OS-specific dependency | Refuse; find cross-platform wheel or implement guard |
| Hardcoded secret | Reject immediately; redirect to env var / Colab Secrets |
| Non-compliant commit message | Rewrite before committing |
| Model > 7B in full precision | Refuse; suggest 4-bit QLoRA or PEFT adapter approach |

---

## 8. Validation Snapshot

**Date**: 2026-03-31  
**Test suite**: 133 tests passing (`uv run pytest tests/ -x -q`)  
**Lint**: Clean (`uv run ruff check src tests`)  
**Peak training RAM** (MRPC, bert-base-uncased + LoRA r=8): ~1.8 GB  
**Notebook status**: Colab-safe, `src/`-integrated, no hardcoded tokens  
