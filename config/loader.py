"""Path resolution for bootstrap: YAML config > env var > function default."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

CONFIG_FILENAME = "harness.yaml"
ENV_SKILLS_DIR = "AGENT_SKILLS_DIR"
ENV_MCP_SERVER = "AGENT_MCP_SERVER"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_AGENT_MODEL = "AGENT_MODEL"

@dataclass(frozen=True)
class ResolvedPaths:
    """Final, resolved paths after applying file > env > default precedence."""

    skills_dir: Path
    mcp_server_script: Path


@dataclass(frozen=True)
class ResolvedLLM:
    """Final, resolved LLM settings after applying file > env > default precedence."""

    api_key: Optional[str]
    base_url: Optional[str]
    model: Optional[str]


def _resolve_one(
    *,
    yaml_value: Optional[str],
    env_var: str,
    default: str,
) -> Path:
    """Apply precedence: yaml > env > default. Resolve to absolute Path."""
    raw = yaml_value
    if raw is None or not str(raw).strip():
        raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        raw = default
    return Path(str(raw).strip()).resolve()

def resolve_paths(
    *,
    default_skills_dir: str,
    default_mcp_server_script: str,
) -> ResolvedPaths:
    """Resolve both paths from YAML > env > default. Auto-creates skills_dir. Exits on config errors."""
    yaml_cfg = _load_yaml_config(Path(CONFIG_FILENAME))

    paths_section = yaml_cfg.get("paths")
    if paths_section is None:
        paths_section = {}
    if not isinstance(paths_section, dict):
        print(
            f"[config] {CONFIG_FILENAME} 'paths' must be a mapping, got {type(paths_section).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    skills_dir = _resolve_one(
        yaml_value=paths_section.get("skills_dir"),
        env_var=ENV_SKILLS_DIR,
        default=default_skills_dir,
    )
    mcp_server_script = _resolve_one(
        yaml_value=paths_section.get("mcp_server_script"),
        env_var=ENV_MCP_SERVER,
        default=default_mcp_server_script,
    )

    if not mcp_server_script.is_file():
        print(
            f"[config] mcp_server_script not found: {mcp_server_script}",
            file=sys.stderr,
        )
        sys.exit(1)

    skills_dir.mkdir(parents=True, exist_ok=True)

    return ResolvedPaths(skills_dir=skills_dir, mcp_server_script=mcp_server_script)


def _resolve_llm_one(
    *,
    yaml_value: Optional[str],
    env_var: str,
    default: Optional[str],
) -> Optional[str]:
    """Apply precedence: yaml > env > default. Then interpolate ${VAR}."""
    raw = yaml_value
    if raw is None or not str(raw).strip():
        raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        raw = default
    if raw is None:
        return None
    return _interpolate_env(str(raw).strip())


def resolve_llm(
    *,
    default_api_key: Optional[str],
    default_base_url: Optional[str],
    default_model: Optional[str],
) -> ResolvedLLM:
    """Resolve LLM settings from YAML > env > default. Interpolation in Task 4."""
    yaml_cfg = _load_yaml_config(Path(CONFIG_FILENAME))

    llm_section = yaml_cfg.get("llm")
    if llm_section is None:
        llm_section = {}
    if not isinstance(llm_section, dict):
        print(
            f"[config] {CONFIG_FILENAME} 'llm' must be a mapping, got {type(llm_section).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    for field_name in ("api_key", "base_url", "model"):
        value = llm_section.get(field_name)
        if value is not None and not isinstance(value, str):
            print(
                f"[config] {CONFIG_FILENAME} 'llm.{field_name}' must be a string or null, got {type(value).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

    api_key = _resolve_llm_one(
        yaml_value=llm_section.get("api_key"),
        env_var=ENV_OPENAI_API_KEY,
        default=default_api_key,
    )
    base_url = _resolve_llm_one(
        yaml_value=llm_section.get("base_url"),
        env_var=ENV_OPENAI_BASE_URL,
        default=default_base_url,
    )
    model = _resolve_llm_one(
        yaml_value=llm_section.get("model"),
        env_var=ENV_AGENT_MODEL,
        default=default_model,
    )

    return ResolvedLLM(api_key=api_key, base_url=base_url, model=model)


def _load_yaml_config(path: Path) -> dict:
    """Read YAML file from CWD. Returns {} if absent. Exits on parse error."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[config] failed to parse {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data if isinstance(data, dict) else {}


_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_ANY_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]*)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} placeholders with os.environ values.

    - Valid name match + env set → replace with env value.
    - Valid name match + env unset → stderr warning, keep ${VAR} literal.
    - Invalid name (e.g. ${1bad}, ${bad-name}) → sys.exit(1).
    - Non-recursive: result is not re-interpolated.
    """
    for match in _ANY_PLACEHOLDER_PATTERN.finditer(value):
        candidate = match.group(1)
        if not _ENV_VAR_NAME_PATTERN.match(candidate):
            print(
                f"[config] invalid env var name in interpolation: ${{{candidate}}}",
                file=sys.stderr,
            )
            sys.exit(1)

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        env_value = os.environ.get(name)
        if env_value is None:
            print(
                f"[config] env var not set: ${{{name}}} (used in {CONFIG_FILENAME})",
                file=sys.stderr,
            )
            return match.group(0)
        return env_value

    return _PLACEHOLDER_PATTERN.sub(replace, value)