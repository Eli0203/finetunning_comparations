# New Feature Guide — Spec-Driven Development (SDD)

**Applies to**: `c:\tutoriales\fineTunning`  
**Constitution**: `.specify/memory/constitution.md` v1.1.1  
**Toolchain**: `uv` (required for all commands — see Principle VIII)  
**Last updated**: 2026-03-31

This guide explains how to add a new feature to this codebase from idea to merged code,
following the Spec-Driven Development (SDD) workflow enforced by the project constitution.
Every step corresponds to a constitution check gate or a `speckit` agent command.

---

## Overview of the SDD Workflow

```
Idea → Spec → Plan → Tasks → Implement → Verify → Merge
  ↑                                          ↓
  └──────── Constitution Check ──────────────┘
```

Each stage produces a canonical artifact in `specs/<feature-name>/` that the next stage consumes.

---

## Step 0 — Pre-flight Checks

Before writing a single line, verify the environment and confirm the constitution applies.

```powershell
# 1. Confirm you are on a fresh feature branch
git checkout -b <###-short-feature-name>

# 2. Sync dependencies
uv sync

# 3. Run the test suite — it must be fully green before new work begins
uv run python -m pytest tests/ -q --tb=no

# 4. Lint the codebase — zero warnings required
uv run ruff check src tests
```

If any test fails or `ruff` reports issues, fix them before proceeding.  
**Never start a new feature on a broken baseline.**

---

## Step 1 — Write the Feature Specification (`/speckit.specify`)

Create `specs/<feature-name>/spec.md` from the feature description.

### What goes in the spec

| Section | Content |
|---------|---------|
| User Stories | Priority-ordered user journeys, each independently testable |
| Acceptance Scenarios | Given/When/Then examples that define done |
| Requirements | Functional and non-functional constraints |
| Out of Scope | Explicit exclusions to prevent scope creep |
| Constitution Alignment | How the feature satisfies Principles I–VIII |

### Rules from the constitution

- **Principle VI**: Every class name, method signature, and module path you write in the spec
  MUST match the actual names you intend to create in `src/`. Confirm against existing
  `src/` code before finalising.
- **Principle VII**: If the feature produces new metrics or model outputs, the spec must state
  they will be demonstrable via `finetunning.ipynb` using only `src/` imports.
- **Principle VIII**: The spec must declare that all new classes follow SRP, that
  dependencies are constructor-injected, and that all commands use `uv`.

### Concrete example — spec excerpt

```markdown
## Requirements
- New `AdaptiveBudgetAllocator` class in `src/utils/adaptive_budget.py`
- Constructor: `AdaptiveBudgetAllocator(causal_engine, decay_rate: float = 0.9)`
- Public method: `reallocate(step: int, current_metrics: Dict[str, float]) -> Dict[str, int]`
- Integrates with `CausalTrainingOrchestrator` via constructor injection

## Constitution Alignment
- Principle VI: module path and class name above are confirmed absent in current src/
- Principle VIII: `AdaptiveBudgetAllocator` has SRP (budget reallocation only);
  `CausalMonteCLoRAEngine` will be injected, not instantiated inside the new class
```

### Agent shortcut

```
/speckit.specify <your feature description in natural language>
```

The agent writes the full `spec.md` file to `specs/<feature-name>/spec.md`.

---

## Step 2 — Plan the Implementation (`/speckit.plan`)

Generate `specs/<feature-name>/plan.md` from the spec.

The plan must pass the **Constitution Check** gate before you start coding:

```markdown
## Constitution Check (Required Gate)

- [ ] Modular OOP boundaries preserved in src/ with clear responsibilities
- [ ] All execution/testing commands are UV-managed
- [ ] Behavior changes have planned test evidence (unit + integration)
- [ ] Memory and async execution risks are bounded for constrained hardware
- [ ] Documentation sync scope identified (README, SOURCE_CODE_DOCUMENTATION.md)
- [ ] plan.md cross-checked against src/: no documented module path or signature
      contradicts an existing src/ file
- [ ] Notebook reproducibility planned if new metrics are introduced
- [ ] All new classes enforce SRP and use constructor-based DI
- [ ] No bare `python` or `pip` commands appear anywhere in the plan
```

Every failing gate item blocks implementation. Document exceptions with rationale.

### What the plan includes

- **Phase 0**: Research (dependency audit, existing API surface review)
- **Phase 1**: Design (data model, contracts, quickstart demo)
- **Phase 2**: Task generation (delegates to `/speckit.tasks`)
- **Phase 3**: Implementation (see Step 3)
- **Phase 4**: Documentation sync

### Agent shortcut

```
/speckit.plan
```

---

## Step 3 — Generate Actionable Tasks (`/speckit.tasks`)

Create `specs/<feature-name>/tasks.md` from the plan.

### Task numbering convention

Tasks are numbered sequentially from the last task in the existing
`specs/feature_causal_lora/tasks.md` to avoid ID collisions, or use a
separate file for independent features (e.g., `specs/feature-adaptive-budget/tasks.md`).

### Task anatomy

Each task must have:

```markdown
### T###: [Short Title]

**Phase**: [4a / 4b / 5 / etc.]
**Acceptance Criteria**:
- [ ] Class name and module path match the spec exactly
- [ ] All public methods are type-annotated
- [ ] Unit tests exist in tests/test_<module>.py
- [ ] Docstring follows Google style
- [ ] `uv run ruff check src tests` passes
- [ ] `uv run pytest tests/test_<module>.py -q` passes
```

### Agent shortcut

```
/speckit.tasks
```

---

## Step 4 — Implement

Work through tasks in dependency order. For each task:

### 4a — Create the class (OOP template)

```python
# src/utils/your_new_module.py
"""
One-line module description (what it does, not how).
"""
from __future__ import annotations

from typing import Dict, Protocol

from src.utils.logger import logger
# Import only protocols/ABCs at module level; inject concretes via constructor


class YourProtocol(Protocol):
    """Describe the contract this new class depends on."""
    def some_method(self) -> Dict[str, float]: ...


class YourNewClass:
    """
    Single responsibility: [state the one thing this class does].

    Does NOT: [explicitly forbid responsibilities that might creep in].
    """

    def __init__(
        self,
        dependency: YourProtocol,  # Injected, not instantiated here
        config_param: float = 0.9,
    ) -> None:
        self._dependency = dependency
        self._config_param = config_param
        logger.info(f"YourNewClass initialised (config_param={config_param})")

    def do_the_thing(self, step: int) -> Dict[str, float]:
        """
        What this method does.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Dict mapping path names to updated values.
        """
        ...
```

### 4b — Wire into the orchestrator (DI pattern)

New classes are always passed into `CausalTrainingOrchestrator` (or whichever
coordinator is appropriate) via its constructor — never imported and instantiated
inside the orchestrator body.

```python
# main.py / test setup
your_component = YourNewClass(dependency=some_existing_object, config_param=0.9)

orchestrator = CausalTrainingOrchestrator(
    ...existing_args...,
    your_component=your_component,   # <-- injected
)
```

### 4c — Write tests

All tests belong in `tests/test_<module_name>.py`.

```powershell
# Run after each task
uv run pytest tests/test_your_new_module.py -v
```

Minimum test structure:

| Group | What to test |
|-------|-------------|
| Initialization | Valid params, invalid params raise ValueError |
| Core method | Happy path, boundary values, empty input |
| Integration | Works together with the injected dependency (use a mock or minimal stub) |
| Edge cases | Device mismatches, empty data, extreme values |

### 4d — Lint after every file

```powershell
uv run ruff check src tests
```

Fix all issues before moving to the next task.

---

## Step 5 — Verify

After all tasks are complete, run the full suite:

```powershell
# Full test suite
uv run python -m pytest tests/ -q --tb=short

# Lint
uv run ruff check src tests

# Import smoke test
uv run python -c "from src.utils.your_new_module import YourNewClass; print('import OK')"
```

### Notebook verification (Principle VII)

If the feature produces new metrics or model behavior:

1. Add a short cell to `finetunning.ipynb` importing only from `src/`.
2. Execute the cell end-to-end:
   ```powershell
   # Use the Jupyter kernel backed by the uv environment
   uv run jupyter nbconvert --to notebook --execute finetunning.ipynb --output finetunning.ipynb
   ```
3. The cell must execute without errors.

---

## Step 6 — Documentation Sync

Before closing any task, update:

| Document | What to update |
|----------|---------------|
| `README.md` | Add module to "Source Tree Guide" if it is a new file |
| `SOURCE_CODE_DOCUMENTATION.md` | Add class-level entry with constructor signature |
| `CONSISTENCY_REPORT.md` | Re-run the audit; date-stamp the new entry |
| `specs/<feature-name>/tasks.md` | Mark each task `[x]` as it is completed |

### Constitution compliance check

```markdown
- [ ] All new classes have a single, stated responsibility
- [ ] No concrete class is instantiated inside a high-level coordinator (DI only)
- [ ] All public methods have type annotations and Google-style docstrings
- [ ] No bare `python` or `pip` in any new docs or scripts
- [ ] CONSISTENCY_REPORT.md updated and alignment score ≥ 90%
```

---

## Step 7 — Merge

```powershell
# Stage only production and test files (not generated artifacts)
git add src/ tests/ specs/<feature-name>/ README.md CONSISTENCY_REPORT.md

git commit -m "feat(<feature-name>): <summary>

- Classes added: <list>
- Tests: X passing
- Constitution gates: all passed
- Alignment score: 100%"

git push origin <###-short-feature-name>
```

Open a pull request targeting `main`. The PR description must include:

- Summary of new classes and their responsibilities
- Test count before and after
- Constitution Check result (all items ✅)
- `uv run ruff check src tests` output (must be clean)

---

## Quick Reference

### Speckit command sequence

| Step | Command | Output artifact |
|------|---------|----------------|
| 1 | `/speckit.specify <description>` | `specs/<feature>/spec.md` |
| 2 | `/speckit.plan` | `specs/<feature>/plan.md`, `research.md`, `data-model.md`, `quickstart.md` |
| 3 | `/speckit.tasks` | `specs/<feature>/tasks.md` |
| 4a-4d | Implement tasks manually | `src/`, `tests/` |
| 5 | `/speckit.analyze` | Gap report, constitution compliance summary |
| (optional) | `/speckit.checklist` | Feature-specific checklist |

### Key source module locations

| What | Where |
|------|-------|
| Settings / env config | `src/settings/settings.py` |
| Data loading strategies | `src/finetuner/data_loader.py` |
| LoRA model adaptation | `src/finetuner/lora_engine.py` (use `lora_rank` property, not `_config.r`) |
| Causal path engine | `src/finetuner/causal_engine.py` |
| Causal training coordinator | `src/finetuner/causal_training_orchestrator.py` |
| Weight sampling | `src/utils/causal_sampler.py` |
| Async background sampling | `src/utils/async_sampler.py` |
| Inter-process buffer | `src/utils/multiprocessing.py` (`DoubleBuffer`) |
| Rate-limited weight application | `src/utils/training_integrator.py` |
| Math primitives | `src/utils/math_utils.py` |
| GLUE metrics + NIE | `src/utils/metrics.py` |
| Logger | `src/utils/logger.py` |

### Design pattern cheat-sheet

| Pattern | When to use | Example in this repo |
|---------|------------|---------------------|
| **Strategy** | Swap-able algorithms (e.g. new task loaders) | `GLUETaskStrategy` / `MRPCStrategy` |
| **Dependency Injection** | Decouple coordinator from concrete classes | `CausalTrainingOrchestrator(causal_engine=...)` |
| **Protocol** | Define contracts without inheritance | `GLUEConfig`, `GLUETaskStrategy` |
| **Template Method** | Fixed sequence, variable steps | `CausalTrainingOrchestrator.prepare()` |
| **Composition** | Reuse without deep hierarchies | Orchestrator holds references to all components |
| **Factory** | Create variants without `if/else` in callers | Planned: Sampler factory for causal vs. random |

---

## Common Pitfalls (Avoid These)

| Anti-pattern | Why it violates the constitution | Correct approach |
|-------------|----------------------------------|-----------------|
| `from src.finetuner.lora_engine import FineTuningEngine` inside an orchestrator body | Violates DIP (Principle VIII) | Receive `lora_engine` via `__init__(self, lora_engine: EngineProtocol)` |
| `import subprocess; subprocess.run(['pip', 'install', ...])` | Violates Principle II and VIII | Add to `pyproject.toml` and run `uv add <package>` |
| `engine._config.r` from outside `FineTuningEngine` | Violates encapsulation (OCP/LSP) | Use `engine.lora_rank` (public property) |
| Notebook cell that defines a function duplicating `src/` logic | Violates Principle VII | Promote the function to `src/` and import it |
| Marking a task complete without a passing test | Violates Principle III | Add test before closing task |
| Using `python main.py` in docs | Violates Principle VIII | Always `uv run python main.py` |
