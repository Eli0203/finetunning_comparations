"""Settings factory for creating fresh instances with experiment-specific dispatch.

Single responsibility: Factory pattern for creating and reloading settings
with environment file loading, experiment_type resolution, and algorithm
validation without caching.
"""

from pathlib import Path
from typing import Any, Dict, Type
import warnings
from dotenv import dotenv_values

from .base import Settings
from .experiment_settings import (
    ExperimentType,
    BaseRunSettings,
    LoraSettings,
    LaplaceLoraSettings,
    CausalLoraSettings,
)


class SettingsFactory:
    """Factory for creating fresh Settings instances without caching.
    
    Enforces single responsibility: decoupled from storage, validation occurs
    in Settings models, experiment dispatch determined by experiment_type field.
    """

    _SETTINGS_BY_EXPERIMENT: Dict[str, Type[BaseRunSettings]] = {
        "lora": LoraSettings,
        "laplace_lora": LaplaceLoraSettings,
        "causal_lora": CausalLoraSettings,
    }

    @staticmethod
    def _normalize_env_values(raw_values: Dict[str, Any]) -> Dict[str, Any]:
        """Map env-style keys to Settings field names."""
        normalized: Dict[str, Any] = {}
        alias_to_field: Dict[str, str] = {}

        for field_name, field_info in Settings.model_fields.items():
            alias = field_info.validation_alias
            if isinstance(alias, str):
                alias_to_field[alias] = field_name

        for key, value in raw_values.items():
            if value is None:
                continue
            normalized[alias_to_field.get(key, key.lower())] = value

        return normalized

    @staticmethod
    def _load_env_file(env_file: Path) -> Dict[str, Any]:
        """Load values from an env file and normalize them to Settings fields."""
        env_values = dotenv_values(env_file)
        return SettingsFactory._normalize_env_values(dict(env_values))

    @staticmethod
    def _resolve_experiment_type(
        init_kwargs: Dict[str, Any],
        strict_experiment_type: bool,
    ) -> ExperimentType:
        """Resolve experiment_type with migration compatibility and strict behavior."""

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off", ""}:
                    return False
            return bool(value)

        resolved = init_kwargs.get("experiment_type")

        if resolved is None:
            if strict_experiment_type:
                raise ValueError("experiment_type is required when strict_experiment_type=True")

            causal_enabled = _coerce_bool(init_kwargs.get("execute_causal_engine", False))
            laplace_enabled = _coerce_bool(init_kwargs.get("execute_laplace", False))

            if causal_enabled and laplace_enabled:
                raise ValueError(
                    "Cannot infer experiment_type when both execute_causal_engine and execute_laplace are True"
                )

            if causal_enabled:
                init_kwargs["experiment_type"] = "causal_lora"
                warnings.warn(
                    "Missing experiment_type; inferred 'causal_lora' from algorithm flags for migration compatibility.",
                    UserWarning,
                    stacklevel=3,
                )
                return "causal_lora"

            if laplace_enabled:
                init_kwargs["experiment_type"] = "laplace_lora"
                warnings.warn(
                    "Missing experiment_type; inferred 'laplace_lora' from algorithm flags for migration compatibility.",
                    UserWarning,
                    stacklevel=3,
                )
                return "laplace_lora"

            warnings.warn(
                "Missing experiment_type; defaulting to 'lora' for migration compatibility.",
                UserWarning,
                stacklevel=3,
            )
            init_kwargs["experiment_type"] = "lora"
            return "lora"

        if resolved not in SettingsFactory._SETTINGS_BY_EXPERIMENT:
            raise ValueError("experiment_type must be one of: lora, laplace_lora, causal_lora")
        return resolved

    @staticmethod
    def _validate_algorithm_match(settings: BaseRunSettings) -> None:
        """Fail-fast guard for experiment_type to algorithm mismatch."""
        expected_flags = {
            "lora": (False, False),
            "laplace_lora": (False, True),
            "causal_lora": (True, False),
        }
        expected_causal, expected_laplace = expected_flags[settings.experiment_type]
        if (
            settings.execute_causal_engine != expected_causal
            or settings.execute_laplace != expected_laplace
        ):
            raise ValueError(
                f"Algorithm mismatch for experiment_type='{settings.experiment_type}': "
                f"expected execute_causal_engine={expected_causal}, "
                f"execute_laplace={expected_laplace}"
            )

    @staticmethod
    def create_settings(
        env_file: Path | None = None,
        override_values: Dict[str, Any] | None = None,
        strict_experiment_type: bool = False,
    ) -> BaseRunSettings:
        """Create a fresh experiment settings instance from env file and overrides."""
        init_kwargs: Dict[str, Any] = {"_env_file": None}
        if env_file is not None:
            env_path = Path(env_file)
            if not env_path.exists():
                raise FileNotFoundError(f"Environment file not found: {env_path}")
            init_kwargs.update(SettingsFactory._load_env_file(env_path))
        if override_values:
            init_kwargs.update(override_values)

        experiment_type = SettingsFactory._resolve_experiment_type(
            init_kwargs,
            strict_experiment_type,
        )
        settings_cls = SettingsFactory._SETTINGS_BY_EXPERIMENT[experiment_type]
        settings = settings_cls(**init_kwargs)
        SettingsFactory._validate_algorithm_match(settings)
        return settings

    @staticmethod
    def create_settings_from_env(
        env_file: Path,
        strict_experiment_type: bool = False,
    ) -> BaseRunSettings:
        """Convenience method to create settings from a specific env file."""
        return SettingsFactory.create_settings(
            env_file=env_file,
            strict_experiment_type=strict_experiment_type,
        )

    @staticmethod
    def reload_settings(
        current: Settings,
        env_file: Path | None = None,
        strict_experiment_type: bool = False,
    ) -> BaseRunSettings:
        """Return a new settings instance without mutating the current one."""
        current_values = current.model_dump()
        if env_file is not None:
            return SettingsFactory.create_settings(
                env_file=env_file,
                strict_experiment_type=strict_experiment_type,
            )
        return SettingsFactory.create_settings(
            override_values=current_values,
            strict_experiment_type=strict_experiment_type,
        )
