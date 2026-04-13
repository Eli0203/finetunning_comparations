from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/causal_finetuning_demo.ipynb")


def _notebook_text() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_uses_isolated_output_roots():
    text = _notebook_text()
    assert "lora_only" in text
    assert "laplace_lora" in text
    assert "causal_lora" in text


def test_notebook_removes_strict_resume_heuristic():
    text = _notebook_text()
    assert "_should_enable_strict_resume" not in text


def test_notebook_routes_execution_through_uv():
    text = _notebook_text()
    assert '\\"uv\\",' in text
    assert '\\"run\\",' in text
    assert '\\"python\\",' in text
    assert '\\"import json, main; print(json.dumps(main.main()))\\"' in text


def test_notebook_uses_explicit_utf8_for_subprocess_and_dotenv():
    text = _notebook_text()
    assert 'dotenv_values(str(ENV_PATH), encoding=\\"utf-8\\")' in text
    assert 'encoding=\\"utf-8\\"' in text
    assert 'errors=\\"replace\\"' in text


def test_notebook_uses_trial_setup_and_cleanup_hooks():
    text = _notebook_text()
    assert "def _prepare_trial_state(seed: int = 42)" in text
    assert "def _cleanup_trial_state()" in text
    assert "_prepare_trial_state()" in text
    assert "_cleanup_trial_state()" in text
    assert "torch.manual_seed(seed)" in text
    assert "torch.cuda.manual_seed_all(seed)" in text
    assert "torch.cuda.empty_cache()" in text


def test_notebook_preserves_resume_policy_on_retry():
    text = _notebook_text()
    assert "RESUME_POLICY" in text
    assert "BASE_ENV" in text
    assert "retry_updates" in text
    assert "retry_updates[\"RESUME_POLICY\"] = \"false\"" not in text


def test_notebook_avoids_global_env_mutation_for_trials():
    text = _notebook_text()
    assert "def _build_trial_env(env_updates" in text
    assert "child_env = os.environ.copy()" in text
    assert 'os.environ[key] = str(value)' not in text
    assert 'set_key(str(ENV_PATH), key, str(value))' not in text


def test_notebook_prints_per_epoch_and_run_details_for_each_experiment():
    text = _notebook_text()
    assert "Per-Epoch Metrics:" in text
    assert "def _print_run_details(run_result" in text
    assert "Run details | suite=" in text
    assert "output_dir" in text
    assert "model_dir" in text
    assert "runtime_seconds" in text
    assert "resume_checkpoint" in text


def test_notebook_isolates_eval_and_log_directories_per_trial():
    text = _notebook_text()
    assert 'log_output_dir = (LOG_ROOT / suite_cfg[\\"EXPERIMENT_TYPE\\"] / task_name).resolve()' in text
    assert '\\"LOGGING_DIR\\": str(log_output_dir)' in text
    assert 'eval_root = run_dir /' in text
    assert '_eval_tmp' in text
    assert 'suite_name.lower().replace(' in text
    assert 'output_dir=str(eval_output_dir.resolve())' in text
    assert 'output_eval_tmp' not in text
