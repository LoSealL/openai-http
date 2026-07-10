# Python Public API Contract: openai_http

**Feature**: `002-extensible-backend-sdk`
**Version**: 0.2.0
**Date**: 2026-05-27

## Top-Level Namespace

```python
import openai_http

# Public names (enforced via __all__):
openai_http.BackendBase       # Abstract base class for custom backends
openai_http.run_server        # Blocking server entry point
openai_http.setup_logging     # Convenience function for logging
openai_http.__version__       # Package version string
```

---

## BackendBase

See `openai_http/backends/base.py` for the full implementation. Key points:

- 4 required abstract methods: `generate`, `generate_stream`, `list_models`, `get_model`
- 2 optional methods (default raise `NotImplementedError`): `embed`, `generate_tool_calls`
- 2 lifecycle hooks (default no-op): `setup`, `teardown`

---

## Error Behavior Summary

| Scenario | HTTP Status | Error Type | Response Body |
| -------- | ----------- | ---------- | ------------- |
| Backend method raises unhandled exception | 500 | `server_error` | `{"error": {"message": "...", "type": "server_error", "param": null, "code": "internal_error"}}` |
| Optional backend method not implemented | 501 | `not_implemented_error` | `{"error": {"message": "...", "type": "not_implemented_error", "param": null, "code": null}}` |
| Backend setup() raises exception | N/A (server won't start) | RuntimeError | Traceback printed to stderr, process exits non-zero |
