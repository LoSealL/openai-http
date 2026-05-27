# Data Model: OpenAI-Compatible HTTP API Service

**Feature**: `001-openai-http-api`
**Date**: 2026-05-27

## Entity Relationship Overview

```
┌──────────────────┐     ┌──────────────────┐
│  Model           │     │  Backend         │
│──────────────────│     │──────────────────│
│ id               │────▶│ name             │
│ created          │     │ device           │
│ owned_by         │     │ model_path       │
│ object           │     │ tokenizer        │
└──────────────────┘     └──────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ CompletionReq    │     │ CompletionResp   │
│──────────────────│     │──────────────────│
│ model            │     │ id               │
│ messages/prompt  │     │ object           │
│ temperature      │     │ created          │
│ max_tokens       │     │ model            │
│ top_p            │     │ choices[]        │
│ n                │     │ usage            │
│ stream           │     │ system_fingerprint│
│ stop             │     └──────────────────┘
│ tools            │
│ tool_choice      │
│ response_format  │
│ stream_options   │
└──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│  FileObject      │     │ FineTuningJob    │
│──────────────────│     │──────────────────│
│ id               │     │ id               │
│ object           │     │ object           │
│ bytes            │     │ model            │
│ created_at       │     │ created_at       │
│ filename         │     │ finished_at      │
│ purpose          │     │ fine_tuned_model │
│ status           │     │ status           │
│ storage_path     │     │ trained_tokens   │
└──────────────────┘     │ training_file    │
         │               │ hyperparameters  │
         │               └──────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│  BatchJob        │     │  BatchRequestFile│
│──────────────────│     │──────────────────│
│ id               │     │ file_id          │
│ object           │     └──────────────────┘
│ input_file_id    │
│ output_file_id   │
│ status           │
│ created_at       │
│ completed_at     │
│ request_counts   │
└──────────────────┘
```

## Entities

### Model

Represents a loaded model in the service registry.

| Field      | Type    | Description                        | Validation                   |
| ---------- | ------- | ---------------------------------- | ---------------------------- |
| id         | string  | Unique model identifier            | Non-empty, alphanumeric+dash |
| object     | string  | Fixed: `"model"`                   | Literal                      |
| created    | integer | Unix timestamp of creation         | Positive integer             |
| owned_by   | string  | Organization owning the model      | Non-empty string             |

**Lifecycle**: Created when backend loads a model. Removed when backend unloads. No persistence (rebuilt on startup from config).

**Uniqueness**: `id` must be unique across all registered models.

### ChatMessage

A single message in a chat conversation.

| Field   | Type               | Description                              |
| ------- | ------------------ | ---------------------------------------- |
| role    | string             | One of: `system`, `user`, `assistant`, `tool` |
| content | string or null     | Message text (null for some tool messages) |
| name    | string (optional)  | Sender name for multi-agent contexts     |
| tool_calls | array (optional) | Tool calls made by the assistant         |
| tool_call_id | string (optional) | Links a tool message to its tool call   |

### ChatCompletionRequest

| Field             | Type               | Default | Description                                    |
| ----------------- | ------------------ | ------- | ---------------------------------------------- |
| model             | string             | (required) | Target model ID                             |
| messages          | ChatMessage[]      | (required) | Conversation history                        |
| temperature       | float              | 1.0     | Sampling temperature [0.0, 2.0]               |
| top_p             | float              | 1.0     | Nucleus sampling [0.0, 1.0]                   |
| n                 | integer            | 1       | Number of completions [1, 128]                 |
| stream            | boolean            | false   | Enable SSE streaming                          |
| stop              | string \| string[] | null    | Stop sequences (max 4)                        |
| max_tokens        | integer \| null    | null    | Max generation tokens [1, 131072]             |
| presence_penalty  | float              | 0.0     | [-2.0, 2.0]                                   |
| frequency_penalty | float              | 0.0     | [-2.0, 2.0]                                   |
| logit_bias        | object \| null     | null    | Token ID → bias mapping                        |
| logprobs          | boolean            | false   | Include token log probabilities               |
| top_logprobs      | integer \| null    | null    | Number of top logprobs [0, 20]                |
| response_format   | object \| null     | null    | `{"type": "json_object"}` or `{"type": "text"}` |
| seed              | integer \| null    | null    | Reproducibility seed                          |
| tools             | Tool[] \| null     | null    | Function definitions for tool calling          |
| tool_choice       | string \| object   | "auto"  | "auto", "none", "required", or specific tool   |
| user              | string \| null     | null    | End-user identifier                            |
| stream_options    | object \| null     | null    | `{"include_usage": true}`                      |

### ChatCompletionResponse

| Field              | Type              | Description                                 |
| ------------------ | ----------------- | ------------------------------------------- |
| id                 | string            | `chatcmpl-{random}`                         |
| object             | string            | `"chat.completion"`                         |
| created            | integer           | Unix timestamp                              |
| model              | string            | Model ID used                               |
| choices            | Choice[]          | Array of completions (length = n)           |
| usage              | UsageInfo         | Token counts                                |
| system_fingerprint | string            | Service fingerprint                         |
| service_tier       | string \| null    | Processing tier                             |

**Choice**:
| Field         | Type          | Description                                  |
| ------------- | ------------- | -------------------------------------------- |
| index         | integer       | Zero-based index                             |
| message       | ChatMessage   | Assistant response message                   |
| logprobs      | object \| null | Token log probabilities (if requested)       |
| finish_reason | string \| null | "stop", "length", "tool_calls", "content_filter" |

### CompletionRequest (Legacy)

| Field             | Type                       | Description                     |
| ----------------- | -------------------------- | ------------------------------- |
| model             | string                     | Target model ID                 |
| prompt            | string \| string[] \| int[] | Input prompt(s)                |
| suffix            | string \| null             | Suffix for FIM (fill-in-middle) |
| max_tokens        | integer                    | Default 16, [1, 131072]         |
| temperature       | float                      | [0.0, 2.0]                      |
| top_p             | float                      | [0.0, 1.0]                      |
| n                 | integer                    | [1, 128]                        |
| stream            | boolean                    | SSE streaming                   |
| logprobs          | integer \| null            | [0, 5]                          |
| echo              | boolean                    | Include prompt in completion    |
| stop              | string \| string[] \| null | Stop sequences                  |
| presence_penalty  | float                      | [-2.0, 2.0]                     |
| frequency_penalty | float                      | [-2.0, 2.0]                     |
| best_of           | integer                    | [1, 20]                         |
| logit_bias        | object \| null             | Token bias map                  |
| user              | string \| null             | User identifier                 |

### CompletionResponse

| Field              | Type          | Description                    |
| ------------------ | ------------- | ------------------------------ |
| id                 | string        | `cmpl-{random}`               |
| object             | string        | `"text_completion"`           |
| created            | integer       | Unix timestamp                |
| model              | string        | Model ID used                 |
| choices            | TextChoice[]  | Array of completions          |
| usage              | UsageInfo     | Token counts                  |
| system_fingerprint | string        | Service fingerprint           |

**TextChoice**:
| Field         | Type           | Description                         |
| ------------- | -------------- | ----------------------------------- |
| text          | string         | Generated text                      |
| index         | integer        | Zero-based index                    |
| logprobs      | object \| null | Token log probabilities             |
| finish_reason | string \| null | "stop", "length", "content_filter"  |

### StreamingChunk (ChatCompletionChunk)

| Field              | Type          | Description                   |
| ------------------ | ------------- | ----------------------------- |
| id                 | string        | Same as request's response ID |
| object             | string        | `"chat.completion.chunk"`    |
| created            | integer       | Unix timestamp               |
| model              | string        | Model ID                     |
| system_fingerprint | string        | Service fingerprint          |
| choices            | ChunkChoice[] | Delta choices                |

**ChunkChoice**:
| Field         | Type          | Description                         |
| ------------- | ------------- | ----------------------------------- |
| index         | integer       | Zero-based index                    |
| delta         | object        | `{"content": "..."}`, `{"role": "..."}`, `{"tool_calls": [...]}` |
| logprobs      | object \| null | Token log probabilities            |
| finish_reason | string \| null | Set on final chunk                |

### EmbeddingRequest

| Field            | Type                                              | Description                   |
| ---------------- | ------------------------------------------------- | ----------------------------- |
| input            | string \| string[] \| int[] \| int[][]            | Text(s) or token array(s)     |
| model            | string                                            | Embedding model ID            |
| encoding_format  | string                                            | "float" or "base64"           |
| dimensions       | integer \| null                                   | Truncated embedding dimension |
| user             | string \| null                                    | User identifier               |

### EmbeddingResponse

| Field  | Type          | Description          |
| ------ | ------------- | -------------------- |
| object | string        | `"list"`            |
| data   | Embedding[]   | Embedding objects   |
| model  | string        | Model ID used        |
| usage  | UsageInfo     | Token counts        |

**Embedding**:
| Field     | Type              | Description                           |
| --------- | ----------------- | ------------------------------------- |
| object    | string            | `"embedding"`                        |
| index     | integer           | Input index (matches input order)     |
| embedding | float[] \| string | Numeric array or base64-encoded string |

### UsageInfo

| Field             | Type    | Description                    |
| ----------------- | ------- | ------------------------------ |
| prompt_tokens     | integer | Number of prompt tokens        |
| completion_tokens | integer | Number of completion tokens (0 for embeddings) |
| total_tokens      | integer | prompt_tokens + completion_tokens |

### FileObject

| Field       | Type    | Description                          |
| ----------- | ------- | ------------------------------------ |
| id          | string  | `file-{random}`                      |
| object      | string  | `"file"`                             |
| bytes       | integer | File size in bytes                   |
| created_at  | integer | Unix timestamp                      |
| filename    | string  | Original filename                   |
| purpose     | string  | "fine-tune", "assistants", "batch"   |
| status      | string  | "uploaded", "processed", "error"     |
| status_details | string \| null | Error details if status="error" |

**Storage**: Local filesystem path derived from `id`. Metadata stored in SQLite.

**State transitions**: `uploaded` → `processed` (validation passes) or `uploaded` → `error` (validation fails)

### FineTuningJob

| Field            | Type                   | Description                              |
| ---------------- | ---------------------- | ---------------------------------------- |
| id               | string                 | `ftjob-{random}`                         |
| object           | string                 | `"fine_tuning.job"`                     |
| created_at       | integer                | Unix timestamp                          |
| finished_at      | integer \| null        | When job completed/failed              |
| model            | string                 | Base model ID                           |
| fine_tuned_model | string \| null         | Resulting model ID                      |
| organization_id  | string                 | Organization                            |
| status           | string                 | "queued", "running", "succeeded", "failed", "cancelled" |
| hyperparameters  | object                 | `{"n_epochs": int, "batch_size": int, "learning_rate_multiplier": float}` |
| training_file    | string                 | File ID of training data                |
| validation_file  | string \| null         | File ID of validation data              |
| result_files     | string[]               | Output file IDs                         |
| trained_tokens   | integer \| null        | Total tokens processed                  |
| error            | object \| null         | `{"code": "...", "message": "..."}`    |
| seed             | integer                | Reproducibility seed                    |
| method           | object \| null         | Training method details                 |

**State transitions**:
```
queued → running → succeeded
                  → failed
       → cancelled (from queued or running)
```

### BatchJob

| Field           | Type             | Description                           |
| --------------- | ---------------- | ------------------------------------- |
| id              | string           | `batch_{random}`                      |
| object          | string           | `"batch"`                             |
| endpoint        | string           | Target endpoint (e.g., "/v1/chat/completions") |
| input_file_id   | string           | Uploaded JSONL file ID                |
| output_file_id  | string \| null   | Generated results file ID             |
| error_file_id   | string \| null   | Error details file ID                 |
| status          | string           | "validating", "in_progress", "completed", "failed", "cancelled", "cancelling", "expired", "finalizing" |
| created_at      | integer          | Unix timestamp                        |
| completed_at    | integer \| null  | When processing finished              |
| expires_at      | integer          | 24h expiration window                   |
| request_counts  | object \| null   | `{"total": int, "completed": int, "failed": int}` |
| metadata        | object \| null   | User-provided key-value metadata      |

### Tool / FunctionDefinition

| Field              | Type   | Description                        |
| ------------------ | ------ | ---------------------------------- |
| type               | string | `"function"`                       |
| function.name      | string | Function name                      |
| function.description| string| Description of function            |
| function.parameters | object | JSON Schema for arguments         |
| function.strict     | bool  | Enable strict schema validation    |

### ToolCall

| Field               | Type   | Description                        |
| ------------------- | ------ | ---------------------------------- |
| id                  | string | Unique call identifier              |
| type                | string | `"function"`                       |
| function.name       | string | Called function name                |
| function.arguments  | string | JSON-encoded arguments string       |

### ErrorResponse

| Field          | Type             | Description                              |
| -------------- | ---------------- | ---------------------------------------- |
| error.message  | string           | Human-readable error description         |
| error.type     | string           | "invalid_request_error", "authentication_error", "not_found_error", "rate_limit_error", "server_error" |
| error.param    | string \| null   | Parameter name that caused the error     |
| error.code     | string \| null   | Machine-readable error code              |

### HealthStatus

| Field          | Type             | Description                              |
| -------------- | ---------------- | ---------------------------------------- |
| status         | string           | "ready" or "not_ready"                   |
| models         | ModelInfo[]      | List of loaded models with status        |
| backend_type   | string           | Active backend type (e.g., "transformers") |
| uptime_seconds | float            | Service uptime                           |
