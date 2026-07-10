# Implementation Plan: Extensible Backend SDK for openai_http

**Branch**: `002-extensible-backend-sdk` | **Date**: 2026-05-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-extensible-backend-sdk/spec.md`

## Summary

Refactor `openai_http` from a standalone CLI application into an installable Python library with a clean public API. The core deliverable is `BackendBase` — an abstract base class that developers subclass to implement custom inference backends — and `run_server()` — a blocking entry point that starts the OpenAI-compatible HTTP server backed by the custom backend. The library also provides optional lifecycle hooks (`setup`/`teardown`), fail-fast backend validation at startup, a convenience logging function, and 501 responses for unimplemented optional endpoints. All changes preserve backward compatibility with the existing `python -m openai_http` CLI workflow.

## Technical Context

**Language/Version**: Python >=3.12 (per `pyproject.toml`)

**Primary Dependencies**: FastAPI, uvicorn[standard], pydantic>=2.5, pydantic-settings (all existing)

**Storage**: N/A (no new storage requirements; backend-specific storage is developer responsibility)

**Testing**: pytest, httpx (async client fixture for endpoint tests), openai SDK (compatibility tests)

**Target Platform**: Any platform Python 3.12+ supports (Linux server primary, macOS/Windows for development)

**Project Type**: Library (pip-installable package with optional CLI entry point)

**Performance Goals**: N/A (performance is backend-dependent; the library adds negligible overhead beyond HTTP routing)

**Constraints**: Max 5 top-level public names (SC-002). No side-effects at import time (US-1 scenario 3).

**Scale/Scope**: ~5 files modified, 1 new file (`backends/base.py` rewritten, `__init__.py` rewritten, `app.py` modified, `errors.py` extended, optional router changes for 501 handling)

## Project Structure

### Documentation (this feature)

```text
specs/002-extensible-backend-sdk/
├── plan.md              # This file
├── research.md          # Research decisions (R-001 through R-008)
├── data-model.md        # Entity definitions (BackendBase, ServerConfig)
├── quickstart.md        # Developer quick-start guide
├── contracts/
│   └── python-api.md    # Public API contract (BackendBase, run_server, setup_logging)
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Tasks (created by /speckit.tasks)
```

## Implementation Tasks (by phase)

### Phase 1 — Setup

No project-level setup is required; the package, dev venv (`uv`), and test infrastructure already exist.

- [x] T001 Verify Python 3.12 venv is active and `uv run pytest tests/ -v` baseline passes on current code before any modifications
- [x] T002 Verify `python -m openai_http` starts the mock CLI server on port 8000 (baseline backward-compat check)

### Phase 2 — Foundational (blocking prerequisites)

These tasks define the core `BackendBase` contract and error types that every subsequent user story depends on. Must complete before US1/US2/US3 work can begin.

- [x] T003 Rewrite `BackendBase` as ABC in openai_http/backends/base.py
- [x] T004 Update openai_http/backends/__init__.py to re-export `BackendBase`
- [x] T005 [P] Update openai_http/backends/mock_backend.py to inherit from `BackendBase`
- [x] T006 [P] Add `NotImplementedOpenAIError` to openai_http/errors.py
