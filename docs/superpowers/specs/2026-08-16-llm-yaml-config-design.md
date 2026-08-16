# LLM YAML Configuration

**Date:** 2026-08-16
**Status:** Approved (brainstorming complete, awaiting implementation plan)
**Related:** Extends `2026-08-16-skill-mcp-path-config-design.md` — adds a new section to the existing `harness.yaml` schema.

## Purpose

Allow `api_key`, `base_url`, and `model` for the LLM client to be configured via `harness.yaml` in addition to environment variables and function defaults. Supports `${ENV_VAR}` interpolation for safer secret handling.

## Scope

**In scope:** Three LLM settings — `api_key`, `base_url`, `model`.

**Out of scope:** `timeout`, `temperature`, `max_tokens`, or any other `LLMClient` parameter. These remain hardcoded in `bootstrap.py`.

## Precedence (highest to lowest)

1. `./harness.yaml` — YAML config file in current working directory
2. Environment variable — `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `AGENT_MODEL`
3. Function default — passed into `bootstrap()` (and currently hardcoded in `bootstrap.py`)

Same precedence as the existing path configuration. No CLI flags.

## Configuration File

The existing `./harness.yaml` schema is extended with a new top-level `llm:` section:

```yaml
paths:
  skills_dir: ./my_skills
  mcp_server_script: ./tools/my_mcp.py

llm:
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_BASE_URL}
  model: ${AGENT_MODEL}
```

All three `llm:` fields are optional. Missing field → fall through to next layer.

## Interpolation

YAML string values support `${ENV_VAR}` interpolation for any of the three fields.

**Rules:**

- Regex: `\$\{([A-Z_][A-Z0-9_]*)\}` — env var names must be uppercase letters, digits, and underscores, starting with a letter or underscore.
- Match found → replace with `os.environ[name]`.
- Match found but env var not set → **stderr warning** (no exit), keep `${VAR}` literal in value.
- Invalid env var name (e.g., `${1bad}`, `${bad-name}`, `${}`) → **exit 1** with `[config] invalid env var name in interpolation: ${<name>}`.
- Non-recursive — a value that becomes `${OTHER}` after interpolation is not re-interpolated.
- Multiple interpolations in one value supported: `"prefix-${VAR1}-middle-${VAR2}-suffix"`.

## Null / Empty Handling

- Field value `null` (YAML) → treated as unset, fall through.
- Field value empty string `""` → treated as unset, fall through.
- Env var value empty string → treated as unset, fall through.

## Module Structure

**Modified file:** `config/loader.py`

Add to existing module:

```python
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_AGENT_MODEL = "AGENT_MODEL"


@dataclass(frozen=True)
class ResolvedLLM:
    """Final, resolved LLM settings after applying file > env > default precedence."""

    api_key: Optional[str]
    base_url: Optional[str]
    model: Optional[str]


def resolve_llm(
    *,
    default_api_key: Optional[str],
    default_base_url: Optional[str],
    default_model: Optional[str],
) -> ResolvedLLM:
    """Resolve LLM settings from YAML > env > default, with env interpolation."""


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} placeholders with os.environ values. Warn on missing env."""
```

`resolve_llm()` reuses `_load_yaml_config()` for YAML loading.

**Modified file:** `config/bootstrap.py`

Add `resolve_llm` import and call after `resolve_paths()`:

```python
from config.loader import resolve_llm, ResolvedLLM

# After resolve_paths() call (around current line 36):
resolved_llm = resolve_llm(
    default_api_key=None,
    default_base_url="https://api.minimaxi.com/v1",
    default_model=model or "MiniMax-M3",
)

# Replace LLMClient(...) construction (current lines 60-65):
llm = LLMClient(
    api_key=resolved_llm.api_key,
    base_url=resolved_llm.base_url,
    model=resolved_llm.model,
    timeout=120.0,
)
```

**Unchanged:** `llm_client.py`, `main.py`, `tests/__init__.py`, all other files.

## Data Flow

```
main.py
  └─ bootstrap()
       ├─ resolve_paths(...) → ResolvedPaths
       │     └─ YAML paths: > AGENT_SKILLS_DIR / AGENT_MCP_SERVER > defaults
       │
       ├─ resolve_llm(default_api_key=None,
       │              default_base_url="https://api.minimaxi.com/v1",
       │              default_model=model or "MiniMax-M3")
       │     └─ ResolvedLLM
       │           ├─ YAML llm: > env > defaults
       │           └─ _interpolate_env() on each string value
       │
       └─ LLMClient(api_key=resolved_llm.api_key,
                    base_url=resolved_llm.base_url,
                    model=resolved_llm.model,
                    timeout=120.0)
```

## Error Handling

All hard errors print to stderr with `[config]` prefix, then `sys.exit(1)`.

| Condition | Message |
|---|---|
| `llm` exists but is not a dict | `[config] harness.yaml 'llm' must be a mapping, got <type>` |
| `llm.<field>` value is not string or null | `[config] harness.yaml 'llm.<field>' must be a string or null, got <type>` |
| `${...}` contains invalid env var name | `[config] invalid env var name in interpolation: ${<name>}` |

Soft warning (no exit):
| Condition | Message |
|---|---|
| `${VAR}` referenced but `VAR` not in environment | `[config] env var not set: ${VAR} (used in harness.yaml)` — keeps `${VAR}` literal in value |

Unknown fields under `llm:` are silently ignored (forward compatibility). Missing `harness.yaml` is silent. Empty/null fields fall through silently.

## Testing

**Extended file:** `tests/test_config_loader.py`

| Test | Verifies |
|---|---|
| `test_resolve_llm_defaults_only` | No YAML, no env → returns defaults |
| `test_resolve_llm_env_only` | Env set, no YAML → env wins |
| `test_resolve_llm_yaml_only` | YAML set, no env → YAML wins |
| `test_resolve_llm_yaml_overrides_env` | Both set → YAML wins |
| `test_resolve_llm_partial_yaml_falls_through` | YAML sets only api_key; others fall through |
| `test_resolve_llm_null_field_falls_through` | `api_key: null` → falls through to env |
| `test_resolve_llm_interpolation_resolved` | `${VAR}` with env set → replaced |
| `test_resolve_llm_interpolation_missing_env_warns` | `${MISSING}` with env unset → warning, literal kept |
| `test_resolve_llm_invalid_env_name_exits` | `${1bad}` or `${bad-name}` → exit 1 |
| `test_resolve_llm_top_level_not_dict_exits` | `llm: [list]` → exit 1 |
| `test_resolve_llm_field_not_string_exits` | `api_key: [list]` → exit 1 |
| `test_resolve_llm_unknown_field_silently_ignored` | `llm.apikey:` typo → ignored |

Uses pytest fixtures: `tmp_path`, `monkeypatch.setenv` / `delenv`, `capsys`, `monkeypatch.chdir`.

## Backwards Compatibility

- `bootstrap()` public signature unchanged (still has `model: Optional[str]` parameter, now used only as a default for `resolve_llm`).
- Existing path config still works (new `llm:` section is independent).
- No `harness.yaml` file → all three LLM settings come from env vars + bootstrap defaults (current behavior preserved).
- `OPENAI_API_KEY` env var still works as before.

## Out of Scope / Future Work

- `timeout`, `temperature`, `max_tokens`, and other `LLMClient` parameters.
- Recursive `${VAR}` interpolation.
- Different precedence for sensitive fields (env-first for api_key).
- Project-level config override (e.g., `~/.harness/config.yaml` global default).