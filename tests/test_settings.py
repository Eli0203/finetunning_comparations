"""
Tests for src/settings/settings.py Phase 6 (Orchestrator Integration) field validation.

Coverage:
- T061: Temperature field validators (causal_softmax_temp_init >= final, final > 0.1)
- T062: MAX_RAM_THRESHOLD_GB enforcement in Settings
- T063: DEFAULT_QUANTIZATION options and validation
"""

import pytest
from pydantic import ValidationError
from src.settings.settings import Settings


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

