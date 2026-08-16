# Skill / MCP Path Configuration

**Date:** 2026-08-16
**Status:** Approved (brainstorming complete, awaiting implementation plan)

## Purpose

Allow `skills_dir` and `mcp_server_script` to be configured via a YAML file in the project root, with environment variable fallback, in addition to the existing function-level defaults. This makes per-project customization possible without editing source code.

## Scope

**In scope:** External configuration for `skills_dir` and `mcp_server_script` only.

**Out of scope:** `workspace`, `model`, `store_path`, `max_steps`, `api_key`, `base_url`, or any other `bootstrap()` parameter. These remain hardcoded defaults or environment variables (e.g., `AGENT_WORKSPACE`, `AGENT_MODEL`, `OPENAI_API_KEY`).

## Precedence (highest to lowest)

1. `./harness.yaml` — YAML config file in current working directory
2. Environment variable — `AGENT_SKILLS_DIR` / `AGENT_MCP_SERVER`
3. Function default — `skills_dir="./skills"` (hardcoded in `main.py`); `mcp_server_script` defaults to `tools/coding_tools_server.py` relative to project root (computed inside `bootstrap()`)

No CLI flags.

## Configuration File

**Filename:** `./harness.yaml` (relative to CWD at process start)

**Schema:**

```yaml
paths:
  skills_dir: ./my_skills              # optional
  mcp_server_script: ./tools/my_mcp.py # optional
```

**Rules:**

- Top-level `paths:` section. Future path-type settings belong here.
- Every field optional. Missing field → fall through to next layer.
- Field value `null`, empty string, or whitespace-only → treated as unset, fall through.
- Top-level `paths:` absent, or file is empty dict → all fields fall through, no error.
- Unknown field names (e.g., `paths.skill_dir`) → silently ignored (forward compatibility).

**Example `harness.yaml`:**

```yaml
paths:
  skills_dir: ./my_skills
  mcp_server_script: ./tools/my_mcp.py
```

## Environment Variables

| Variable | Maps to |
|---|---|
| `AGENT_SKILLS_DIR` | `paths.skills_dir` |
| `AGENT_MCP_SERVER` | `paths.mcp_server_script` |

Empty string → treated as unset, fall through.

Naming convention follows existing project convention (`AGENT_WORKSPACE`, `AGENT_MODEL`).

## Path Resolution

| Input | Behavior |
|---|---|
| Relative path in YAML/env | Resolved relative to CWD via `pathlib.Path(p).resolve()` |
| Absolute path | `.resolve()` normalizes (Windows drive letter case, `.`/`..`, symlinks) |
| `mcp_server_script` resolved but file missing | Startup error, exit 1 |
| `skills_dir` directory missing | Auto-created with `Path.mkdir(parents=True, exist_ok=True)` |

`.resolve()` is `pathlib` built-in; cross-platform by design. `~` is NOT auto-expanded (matches pathlib default).

## Module Structure

**New file:** `config/loader.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ResolvedPaths:
    skills_dir: Path
    mcp_server_script: Path

def resolve_paths(
    *,
    default_skills_dir: str | Path,
    default_mcp_server_script: str | Path,
) -> ResolvedPaths:
    """Resolve skills_dir and mcp_server_script via file > env > default."""

def _load_yaml_config(path: Path) -> dict:
    """Read YAML file; raise on parse error."""

def _coerce_path(value, *, env_var: str, field: str) -> Optional[Path]:
    """Validate and resolve a path string; return None if empty/null."""
```

**Modified file:** `config/bootstrap.py`

Insert a call to `resolve_paths()` between line 41 (MCP discovery print) and line 42 (`skill_mgr = ProgressiveSkillManager(skills_dir=skills_dir)`). Replace the existing `skills_dir` and `mcp_server_script` parameter values with the resolved ones. Keep `bootstrap()`'s public signature unchanged.

**Unchanged:** `main.py`, `tools/coding_tools_server.py`, all other files.

## Data Flow

```
main.py
  └─ bootstrap(workspace, skills_dir="./skills", mcp_server_script=None, ...)
       └─ resolve_paths(default_skills_dir="./skills",
                        default_mcp_server_script="<path-to-coding_tools_server>")
            ├─ read ./harness.yaml if exists
            ├─ for each field: yaml_value ?? env_var ?? default
            ├─ .resolve() all paths
            ├─ validate mcp_server_script exists (else exit 1)
            └─ mkdir skills_dir if missing
       └─ ProgressiveSkillManager(skills_dir=resolved.skills_dir)
       └─ create_pool_and_discover(server_script=resolved.mcp_server_script)
```

## Error Handling

All errors print to stderr with `[config]` prefix, then `sys.exit(1)`.

| Condition | Message |
|---|---|
| YAML parse error in `./harness.yaml` | `[config] failed to parse ./harness.yaml: <yaml error>` |
| `paths` exists but is not a dict | `[config] ./harness.yaml 'paths' must be a mapping, got <type>` |
| `mcp_server_script` resolved but file missing | `[config] mcp_server_script not found: <abs path>` |

Missing `./harness.yaml` → silent. Empty/null fields → fall through silently. Unknown fields → silently ignored.

`mkdir` failure for `skills_dir` → propagate the `OSError` (consistent with existing `bootstrap()` workspace handling).

## Testing

**New file:** `tests/test_config_loader.py`

| Test | Verifies |
|---|---|
| `test_yaml_only` | YAML present, no env → values come from YAML |
| `test_env_only` | No YAML, env set → values come from env |
| `test_default_only` | No YAML, no env → values are defaults |
| `test_yaml_overrides_env` | Both set → YAML wins |
| `test_relative_path_resolved` | YAML `./foo` → absolute path under CWD |
| `test_missing_mcp_script_exits` | YAML points to nonexistent file → `SystemExit`, stderr message |
| `test_malformed_yaml_exits` | Garbage YAML → `SystemExit`, stderr message |
| `test_null_field_falls_through` | YAML field explicit `null` → falls to env |
| `test_skills_dir_auto_created` | Nonexistent skills_dir → `mkdir` runs |
| `test_empty_file_ok` | Empty `harness.yaml` → no error, all defaults |

Uses `pytest` fixtures: `tmp_path` (YAML file), `monkeypatch.setenv` (env vars), `capsys` (stderr capture), `monkeypatch.chdir` (CWD control).

## Backwards Compatibility

- `bootstrap()` public signature unchanged.
- Existing `main.py` call site unchanged.
- Existing `./skills` and `tools/coding_tools_server.py` defaults still apply when no config file or env var is set.
- No behavior change for users who don't create `./harness.yaml`.

## Out of Scope / Future Work

- CLI flags (user opted out).
- `~/.harness/config.yaml` global config layer (user opted for project-only).
- Validation schema beyond existence checks (e.g., type checks per field).
- Other bootstrap parameters (workspace, model, store_path, etc.).