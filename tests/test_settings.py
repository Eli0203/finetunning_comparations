"""
Tests for src/settings/settings.py Phase 6 (Orchestrator Integration) field validation.

Coverage:
- T061: Temperature field validators (causal_softmax_temp_init >= final, final > 0.1)
- T062: MAX_RAM_THRESHOLD_GB enforcement in Settings
- T063: DEFAULT_QUANTIZATION options and validation
"""

import pytest
from pydantic import ValidationError
from src.settings import (
    CausalLoraSettings,
    LaplaceLoraSettings,
    LoraSettings,
    Settings,
    SettingsFactory,
)


class TestTemperatureFieldValidation:
    """T061: Test causal temperature field constraints."""

    def test_temperature_init_gte_final_valid(self):
        """Valid: temp_init > temp_final."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            causal_softmax_temp_initial=2.0,
            causal_softmax_temp_final=0.5,
        )
        assert settings.causal_softmax_temp_initial == 2.0
        assert settings.causal_softmax_temp_final == 0.5

    def test_temperature_init_equal_final_valid(self):
        """Valid: temp_init == temp_final (edge case)."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            causal_softmax_temp_initial=1.0,
            causal_softmax_temp_final=1.0,
        )
        assert settings.causal_softmax_temp_initial == 1.0
        assert settings.causal_softmax_temp_final == 1.0

    def test_temperature_init_lt_final_invalid(self):
        """Invalid: temp_init < temp_final."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                causal_softmax_temp_initial=0.5,
                causal_softmax_temp_final=2.0,
            )
        assert "causal_softmax_temp_initial" in str(exc_info.value)
        assert "must be >=" in str(exc_info.value)

    def test_temperature_final_below_threshold_invalid(self):
        """Invalid: temp_final <= 0.1."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                causal_softmax_temp_initial=2.0,
                causal_softmax_temp_final=0.1,
            )
        assert "causal_softmax_temp_final" in str(exc_info.value)
        assert "must be > 0.1" in str(exc_info.value)

    def test_temperature_final_just_above_threshold_valid(self):
        """Valid: temp_final > 0.1 (edge case)."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            causal_softmax_temp_initial=2.0,
            causal_softmax_temp_final=0.10001,
        )
        assert settings.causal_softmax_temp_final == pytest.approx(0.10001)

    def test_temperature_both_positive_required(self):
        """Invalid: either temperature field <= 0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                causal_softmax_temp_initial=-1.0,
                causal_softmax_temp_final=0.5,
            )
        # Should fail on gt > 0 constraint
        assert "greater than 0" in str(exc_info.value)


class TestMaxRamThresholdValidation:
    """T062: Test MAX_RAM_THRESHOLD_GB field and enforcement."""

    def test_max_ram_default_value(self):
        """Default MAX_RAM_THRESHOLD_GB = 9.0 GB."""
        settings = Settings(_env_file=None, hf_token="test_token")
        assert settings.max_ram_threshold_gb == 9.0

    def test_max_ram_custom_valid_value(self):
        """Valid custom MAX_RAM_THRESHOLD_GB."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            max_ram_threshold_gb=5.0,
        )
        assert settings.max_ram_threshold_gb == 5.0

    def test_max_ram_below_10_gb_valid(self):
        """Valid: MAX_RAM_THRESHOLD_GB < 10 GB."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            max_ram_threshold_gb=9.99,
        )
        assert settings.max_ram_threshold_gb == 9.99

    def test_max_ram_equal_10_gb_invalid(self):
        """Invalid: MAX_RAM_THRESHOLD_GB >= 10."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                max_ram_threshold_gb=10.0,
            )
        assert "less than 10" in str(exc_info.value)

    def test_max_ram_above_10_gb_invalid(self):
        """Invalid: MAX_RAM_THRESHOLD_GB > 10."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                max_ram_threshold_gb=12.0,
            )
        assert "less than 10" in str(exc_info.value)

    def test_max_ram_zero_or_negative_invalid(self):
        """Invalid: MAX_RAM_THRESHOLD_GB <= 0."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                max_ram_threshold_gb=0.0,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_max_ram_threshold_from_env(self):
        """MAX_RAM_THRESHOLD_GB can be loaded from env variable."""
        import os
        os.environ["MAX_RAM_THRESHOLD_GB"] = "7.5"
        try:
            settings = Settings(_env_file=None, hf_token="test_token")
            assert settings.max_ram_threshold_gb == 7.5
        finally:
            del os.environ["MAX_RAM_THRESHOLD_GB"]


class TestDefaultQuantizationValidation:
    """T063: Test DEFAULT_QUANTIZATION field and Low-VRAM logic."""

    def test_default_quantization_default_value(self):
        """Default DEFAULT_QUANTIZATION = 'auto'."""
        settings = Settings(_env_file=None, hf_token="test_token")
        assert settings.default_quantization == "auto"

    def test_default_quantization_auto_valid(self):
        """Valid: DEFAULT_QUANTIZATION = 'auto'."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            default_quantization="auto",
        )
        assert settings.default_quantization == "auto"

    def test_default_quantization_fp16_valid(self):
        """Valid: DEFAULT_QUANTIZATION = 'fp16'."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            default_quantization="fp16",
        )
        assert settings.default_quantization == "fp16"

    def test_default_quantization_nf4_forced_valid(self):
        """Valid: DEFAULT_QUANTIZATION = 'nf4_forced'."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            default_quantization="nf4_forced",
        )
        assert settings.default_quantization == "nf4_forced"

    def test_default_quantization_invalid_value(self):
        """Invalid: DEFAULT_QUANTIZATION with unknown value."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, 
                hf_token="test_token",
                default_quantization="invalid_mode",
            )
        assert "default_quantization" in str(exc_info.value)
        assert "auto" in str(exc_info.value)

    def test_default_quantization_from_env(self):
        """DEFAULT_QUANTIZATION can be loaded from env variable."""
        import os
        os.environ["DEFAULT_QUANTIZATION"] = "nf4_forced"
        try:
            settings = Settings(_env_file=None, hf_token="test_token")
            assert settings.default_quantization == "nf4_forced"
        finally:
            del os.environ["DEFAULT_QUANTIZATION"]

    def test_quantization_strategy_coverage(self):
        """All three quantization strategies are valid."""
        strategies = ["auto", "fp16", "nf4_forced"]
        for strategy in strategies:
            settings = Settings(_env_file=None, 
                hf_token="test_token",
                default_quantization=strategy,
            )
            assert settings.default_quantization == strategy


class TestInterventionalWeightsField:
    """Test enable_interventional_weights field (Phase 3 + Phase 6)."""

    def test_enable_interventional_weights_default(self):
        """Default enable_interventional_weights = False."""
        settings = Settings(_env_file=None, hf_token="test_token")
        assert settings.enable_interventional_weights is False

    def test_enable_interventional_weights_true(self):
        """Enable interventional weights via setting."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            enable_interventional_weights=True,
        )
        assert settings.enable_interventional_weights is True

    def test_enable_interventional_weights_from_env(self):
        """enable_interventional_weights can be loaded from env."""
        import os
        os.environ["ENABLE_INTERVENTIONAL_WEIGHTS"] = "true"
        try:
            settings = Settings(_env_file=None, hf_token="test_token")
            assert settings.enable_interventional_weights is True
        finally:
            del os.environ["ENABLE_INTERVENTIONAL_WEIGHTS"]


class TestPhase6SettingsIntegration:
    """Integration tests for Phase 6 Settings as a complete unit."""

    def test_all_phase6_fields_together(self):
        """Test all Phase 6 fields initialized together."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            causal_softmax_temp_initial=2.0,
            causal_softmax_temp_final=0.5,
            max_ram_threshold_gb=8.0,
            default_quantization="auto",
            enable_interventional_weights=True,
        )
        assert settings.causal_softmax_temp_initial == 2.0
        assert settings.causal_softmax_temp_final == 0.5
        assert settings.max_ram_threshold_gb == 8.0
        assert settings.default_quantization == "auto"
        assert settings.enable_interventional_weights is True

    def test_phase6_with_causal_engine_flags(self):
        """Phase 6 settings integrate with causal engine flags."""
        settings = Settings(_env_file=None, 
            hf_token="test_token",
            execute_causal_engine=True,
            causal_sampler_mode="mixture_of_gaussians",
            enable_pg_pos=True,
            kfac_correlation=True,
            enable_interventional_weights=True,
            max_ram_threshold_gb=9.0,
            default_quantization="auto",
        )
        assert settings.execute_causal_engine is True
        assert settings.causal_sampler_mode == "mixture_of_gaussians"
        assert settings.enable_pg_pos is True
        assert settings.kfac_correlation is True
        assert settings.enable_interventional_weights is True


class TestSettingsFactory:
    """Phase 3 US1: fresh settings per run via factory."""

    def test_settings_factory_creates_fresh_instances(self):
        """T016: Two factory calls should return equal values but distinct objects."""
        first = SettingsFactory.create_settings(
            override_values={"hf_token": "test_token", "task_name": "mrpc"}
        )
        second = SettingsFactory.create_settings(
            override_values={"hf_token": "test_token", "task_name": "mrpc"}
        )

        assert first is not second
        assert first.task_name == second.task_name == "mrpc"
        assert first.hf_token == second.hf_token == "test_token"

    def test_settings_factory_reloads_env_file(self, env_file_fixture):
        """T017: Factory should re-read updated env file content on each call."""
        env_file_fixture.write_text("HF_TOKEN=token_a\nTASK_NAME=mrpc\n")
        first = SettingsFactory.create_settings(env_file=env_file_fixture)

        env_file_fixture.write_text("HF_TOKEN=token_b\nTASK_NAME=sst2\n")
        second = SettingsFactory.create_settings(env_file=env_file_fixture)

        assert first is not second
        assert first.hf_token == "token_a"
        assert second.hf_token == "token_b"
        assert first.task_name == "mrpc"
        assert second.task_name == "sst2"

    def test_settings_fresh_per_run_isolation(self):
        """T018: Sequential runs with conflicting engine flags remain isolated."""
        run_a = SettingsFactory.create_settings(
            override_values={
                "hf_token": "test_token",
                "execute_causal_engine": False,
                "execute_laplace": True,
            }
        )
        run_b = SettingsFactory.create_settings(
            override_values={
                "hf_token": "test_token",
                "execute_causal_engine": True,
                "execute_laplace": False,
            }
        )

        assert run_a.execute_causal_engine is False
        assert run_a.execute_laplace is True
        assert run_b.execute_causal_engine is True
        assert run_b.execute_laplace is False


    # ============================================================================
    # Phase 2 (Foundation) - T016-T018: SettingsFactory Tests (Failing-First TDD)
    # ============================================================================

    class TestSettingsFactoryFreshInstances:
        """T016: SettingsFactory creates fresh instances without caching."""
    
        def test_settings_factory_creates_fresh_instances(self):
            """T016: Two factory calls return different instances with same values."""
            from src.settings import SettingsFactory
            from pathlib import Path
            import tempfile
        
            # Create a temporary .env file with test values
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
                f.write("HF_TOKEN=test_token\n")
                f.write("MODEL_ID=bert-base-uncased\n")
                f.write("TASK_NAME=mrpc\n")
                env_file = Path(f.name)
        
            try:
                # Create two instances from the same .env file
                config1 = SettingsFactory.create_settings(env_file=env_file)
                config2 = SettingsFactory.create_settings(env_file=env_file)
            
                # Different objects with same values
                assert config1 is not config2, "Factory should return different instances"
                assert config1.model_id == config2.model_id, "Values should be identical"
                assert config1.task_name == config2.task_name, "Values should be identical"
            finally:
                import os
                env_file.unlink()
    
        def test_settings_factory_override_values(self):
            """T016: SettingsFactory.create_settings() respects override_values parameter."""
            from src.settings import SettingsFactory
            from pathlib import Path
            import tempfile
        
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
                f.write("HF_TOKEN=test_token\n")
                f.write("MODEL_ID=bert-base-uncased\n")
                env_file = Path(f.name)
        
            try:
                # Create with overrides
                config = SettingsFactory.create_settings(
                    env_file=env_file,
                    override_values={"model_id": "gpt2"}
                )
            
                # Override should take effect
                assert config.model_id == "gpt2", "Override value should be applied"
            finally:
                env_file.unlink()


    class TestSettingsFactoryEnvFileReload:
        """T017: SettingsFactory detects .env file changes between calls."""
    
        def test_settings_factory_reloads_env_file(self):
            """T017: Change .env on disk; second factory call reads new values."""
            from src.settings import SettingsFactory
            from pathlib import Path
            import tempfile
            import time
        
            with tempfile.TemporaryDirectory() as tmpdir:
                env_file = Path(tmpdir) / ".env"
            
                # Write initial values
                env_file.write_text("HF_TOKEN=test_token\n")
                env_file.write_text("HF_TOKEN=test_token\nMODEL_ID=bert-base-uncased\n")
            
                # First load
                config1 = SettingsFactory.create_settings(env_file=env_file)
                assert config1.model_id == "bert-base-uncased"
            
                # Change the .env file
                time.sleep(0.1)  # Ensure file timestamp changes
                env_file.write_text("HF_TOKEN=test_token\nMODEL_ID=roberta-base\n")
            
                # Second load should see new values
                config2 = SettingsFactory.create_settings(env_file=env_file)
                assert config2.model_id == "roberta-base", "Factory should reload updated .env"
                assert config1.model_id == "bert-base-uncased", "Original config should be unchanged"


    class TestSettingsFactoryIsolation:
        """T018: Fresh settings per run; no cross-run configuration bleeding."""
    
        def test_settings_fresh_per_run_isolation(self):
            """T018: Two runs with different engine flags are completely isolated."""
            from src.settings import SettingsFactory
            from pathlib import Path
            import tempfile
            import os
        
            # Save original env
            original_causal = os.environ.get("EXECUTE_CAUSAL_ENGINE")
        
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    env1 = Path(tmpdir) / ".env1"
                    env2 = Path(tmpdir) / ".env2"
                
                    # Run A: Causal disabled
                    env1.write_text("HF_TOKEN=token_a\nEXECUTE_CAUSAL_ENGINE=false\n")
                    run_a = SettingsFactory.create_settings(env_file=env1)
                
                    # Run B: Causal enabled
                    env2.write_text("HF_TOKEN=token_b\nEXECUTE_CAUSAL_ENGINE=true\n")
                    run_b = SettingsFactory.create_settings(env_file=env2)
                
                    # Verify complete isolation
                    assert run_a.execute_causal_engine is False, "Run A should have causal disabled"
                    assert run_b.execute_causal_engine is True, "Run B should have causal enabled"
                    assert run_a.hf_token == "token_a", "Run A should have its own token"
                    assert run_b.hf_token == "token_b", "Run B should have its own token"
            finally:
                # Restore original env
                if original_causal is not None:
                    os.environ["EXECUTE_CAUSAL_ENGINE"] = original_causal
                elif "EXECUTE_CAUSAL_ENGINE" in os.environ:
                    del os.environ["EXECUTE_CAUSAL_ENGINE"]


class TestSettingsFactoryExperimentDispatch:
    """Phase 3B: experiment_type dispatch and algorithm guardrails."""

    def test_factory_dispatches_lora_subclass(self):
        settings = SettingsFactory.create_settings(
            override_values={
                "hf_token": "token",
                "experiment_type": "lora",
                "execute_causal_engine": False,
                "execute_laplace": False,
            }
        )
        assert isinstance(settings, LoraSettings)
        assert settings.experiment_type == "lora"

    def test_factory_dispatches_laplace_lora_subclass(self):
        settings = SettingsFactory.create_settings(
            override_values={
                "hf_token": "token",
                "experiment_type": "laplace_lora",
                "execute_causal_engine": False,
                "execute_laplace": True,
            }
        )
        assert isinstance(settings, LaplaceLoraSettings)
        assert settings.experiment_type == "laplace_lora"

    def test_factory_dispatches_causal_lora_subclass(self):
        settings = SettingsFactory.create_settings(
            override_values={
                "hf_token": "token",
                "experiment_type": "causal_lora",
                "execute_causal_engine": True,
                "execute_laplace": False,
            }
        )
        assert isinstance(settings, CausalLoraSettings)
        assert settings.experiment_type == "causal_lora"

    def test_missing_experiment_type_defaults_to_lora_with_warning(self):
        with pytest.warns(UserWarning, match="defaulting to 'lora'"):
            settings = SettingsFactory.create_settings(
                override_values={
                    "hf_token": "token",
                    "execute_causal_engine": False,
                    "execute_laplace": False,
                }
            )
        assert isinstance(settings, LoraSettings)
        assert settings.experiment_type == "lora"

    def test_strict_mode_requires_experiment_type(self):
        with pytest.raises(ValueError, match="experiment_type is required"):
            SettingsFactory.create_settings(
                override_values={
                    "hf_token": "token",
                    "execute_causal_engine": False,
                    "execute_laplace": False,
                },
                strict_experiment_type=True,
            )

    def test_algorithm_mismatch_fails_fast(self):
        with pytest.raises(ValueError, match="Algorithm mismatch"):
            SettingsFactory.create_settings(
                override_values={
                    "hf_token": "token",
                    "experiment_type": "causal_lora",
                    "execute_causal_engine": False,
                    "execute_laplace": False,
                }
            )

    def test_canonical_artifact_root_builder(self):
        settings = SettingsFactory.create_settings(
            override_values={
                "hf_token": "token",
                "experiment_type": "laplace_lora",
                "task_name": "sst2",
                "run_id": "run-123",
                "execute_causal_engine": False,
                "execute_laplace": True,
            }
        )

        root = str(settings.canonical_artifact_root).replace("\\", "/")
        checkpoints = str(settings.checkpoints_dir).replace("\\", "/")
        metrics = str(settings.metrics_dir).replace("\\", "/")
        logs = str(settings.logs_dir).replace("\\", "/")

        assert root.endswith("output/laplace_lora/sst2/run-123")
        assert checkpoints.endswith("output/laplace_lora/sst2/run-123/checkpoints")
        assert metrics.endswith("output/laplace_lora/sst2/run-123/metrics")
        assert logs.endswith("output/laplace_lora/sst2/run-123/logs")

