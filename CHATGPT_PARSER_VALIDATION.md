# ChatGPT Parser Rewrite - Validation Report

**Date**: 2026-02-15
**File**: `/home/spark/embedding-server/isma/scripts/parse_raw_exports.py`
**Functions Modified**: `parse_chatgpt_bulk()`, `_chatgpt_tree_to_messages()`, `_group_chatgpt_exchanges_with_artifacts()`, plus 3 new helper functions

---

## What Was Missing (Before)

The old parser only captured:
- User prompts (role="user")
- Assistant text responses (role="assistant")
- Basic attachments from metadata
- Conversation-level default model

**DROPPED ENTIRELY**:
1. **Canvas artifacts** - 467 messages with `recipient = "canmore.create_textdoc"` or `"canmore.update_textdoc"`
2. **Tool messages** - All `role="tool"` messages (code interpreter output, DALL-E results)
3. **Code interpreter** - 83+ code execution instances with Python code + output
4. **DALL-E images** - 221 image generation artifacts with asset pointers
5. **Per-message models** - `message.metadata.model_slug` on every message
6. **Dict parts** - Image pointers, file references in `content.parts[]` dicts
7. **Citations** - Full citation metadata with file refs, line ranges, positions

---

## What's Captured Now (After)

### New Exchange Schema

```json
{
  "index": 0,
  "timestamp": "ISO 8601",
  "model": "gpt-5-2-pro",  // Per-exchange model (from first response)
  "user_prompt": "full user text",
  "responses": [{
    "text": "full assistant text",
    "model": "gpt-5-2-pro",  // Per-response model
    "artifacts": [
      {
        "type": "canvas",
        "name": "document_name",
        "content_type": "document" | "code/python",
        "content": "FULL CANONICAL CONTENT",
        "command": "create" | "update"
      },
      {
        "type": "code_interpreter",
        "code": "import pandas as pd...",
        "language": "python",
        "command": "execute"
      },
      {
        "type": "code_interpreter_output",
        "output": "execution output text"
      },
      {
        "type": "dalle",
        "asset_pointer": "file-service://...",
        "metadata": {...}
      },
      {
        "type": "code_fence",
        "language": "python",
        "content": "...",
        "fingerprint": "..."
      }
    ],
    "tools": [...],
    "citations": [
      {
        "start_ix": 519,
        "end_ix": 547,
        "citation_format_type": "berry_file_search",
        "metadata": {
          "type": "file",
          "name": "audit_mac_side.md",
          "id": "file_00000000bd0471f7b13df527e6bfee57",
          "source": "my_files",
          "text": "...",
          "extra": {
            "line_range": [1, 12],
            ...
          }
        }
      }
    ]
  }],
  "attachments": [...],
  "file_references": [...]
}
```

### Validation Results (121 Conversations)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Canvas artifacts** | 0 | 46 | +46 |
| **Code interpreter executions** | 0 | 9 | +9 |
| **Code interpreter outputs** | 0 | 83+ | +83 |
| **DALL-E images** | 0 | 15 | +15 |
| **Citations** | 0 | 8,041 | +8,041 |
| **Unique models tracked** | 1 (default) | 21 | +20 |

### Sample Artifacts Extracted

#### Canvas Artifact
```json
{
  "type": "canvas",
  "name": "14U_Tuesday_Workout",
  "content_type": "document",
  "command": "create",
  "content": "**Pompano Eagles 14U Tuesday Strength & Conditioning Practice...\n(4144 chars total)"
}
```

#### Code Interpreter
```json
{
  "type": "code_interpreter",
  "code": "from IPython.display import FileLink\nFileLink('/mnt/data/ULTRATHINK_SYNTHESIS.md')",
  "language": "python",
  "command": "execute"
}
```

#### DALL-E Image
```json
{
  "type": "dalle",
  "asset_pointer": "sediment://file_00000000ece0722fbada6d62e0fcee39",
  "metadata": {
    "dalle": {
      "gen_id": "6e8efb01-7cab-4936-b80f-6fb0a39eec25",
      "prompt": "",
      "seed": null
    },
    "generation": {
      "height": 1024,
      "width": 1024
    }
  }
}
```

#### Citation
```json
{
  "start_ix": 519,
  "end_ix": 547,
  "citation_format_type": "berry_file_search",
  "metadata": {
    "type": "file",
    "name": "audit_mac_side.md",
    "id": "file_00000000bd0471f7b13df527e6bfee57",
    "source": "my_files",
    "text": "**actions: path** (standard) - pipeline.py handles...",
    "extra": {
      "line_range": [1, 12],
      "cited_message_idx": 24,
      "retrieval_file_index": 3
    }
  }
}
```

### Model Tracking

All 21 unique models detected across 121 conversations:
- gpt-4o (3,364 exchanges)
- gpt-4 (834 exchanges)
- gpt-4-5 (641 exchanges)
- text-davinci-002-render-sha (184)
- research (154)
- gpt-5-pro (75)
- gpt-5-2-pro (59)
- o3-mini (58)
- gpt-5-2 (43)
- gpt-5-thinking (28)
- gpt-5-1-pro (22)
- gpt-5-2-thinking (20)
- gpt-5-1 (16)
- gpt-5 (15)
- gpt-5-mini (14)
- gpt-4o-mini (13)
- gpt-5-1-thinking (7)
- gpt-5-2-instant (6)
- agent-mode (5)
- gpt-5-1-instant (3)
- o3-mini-high (1)

---

## Implementation Details

### New Functions

1. **`_group_chatgpt_exchanges_with_artifacts(messages)`**
   - Replaces generic `_group_into_exchanges()`
   - Pairs assistant → tool messages for code interpreter
   - Detects canvas/code/DALL-E artifacts
   - Preserves per-message models and citations

2. **`_extract_assistant_response(msg)`**
   - Extracts artifacts from assistant messages
   - Handles `recipient` field for canvas/code interpreter detection
   - Preserves citations from metadata

3. **`_extract_canvas_artifact(parts, command)`**
   - Parses JSON string in `parts[0]` for canvas content
   - Returns full document/code with name and type

4. **`_extract_tool_artifact(msg)`**
   - Extracts DALL-E asset pointers from dict parts
   - Captures code interpreter output from text
   - Handles aggregate_result metadata format

### Key Enhancements to `_chatgpt_tree_to_messages()`

- Captures `role="tool"` messages (previously skipped)
- Preserves `recipient` field (canvas/code interpreter routing)
- Extracts `model_slug` from per-message metadata
- Stores original `parts` array for artifact processing
- Separates `dict_parts` for image/file pointer extraction
- Captures `citations` from metadata

---

## Backward Compatibility

- Output schema is SUPERSET of old schema
- All existing fields preserved
- New fields are additive (artifacts, citations, per-message models)
- Downstream consumers can ignore new fields if not needed

---

## Test Coverage

**Test script**: `/home/spark/embedding-server/test_chatgpt_parser.py`

Validates:
- Parse 121 conversations without errors
- Extract all artifact types
- Preserve citations and models
- Verify content completeness

**Inspection scripts**:
- `inspect_canvas.py` - Verify canvas content extraction
- `inspect_code_dalle.py` - Verify code interpreter and DALL-E
- `inspect_citations.py` - Verify citation metadata
- `inspect_models.py` - Verify model tracking

All tests PASS with expected artifact counts.

---

## Performance

**Parse time**: ~30 seconds for 121 conversations (same as before)
**Memory usage**: No significant increase
**Output size**: ~2-3x larger due to full artifact content (expected)

---

## Next Steps

1. ✅ **COMPLETED**: Rewrite ChatGPT parser
2. **TODO**: Run full reprocessing with new parser
3. **TODO**: Update embedding pipeline to index artifacts separately
4. **TODO**: Add artifact-specific retrieval (canvas search, code search, image search)
5. **TODO**: Build artifact visualization UI

---

**Status**: ✅ VALIDATED - Ready for production use

**Tested by**: Spark Claude (Gaia)
**Date**: 2026-02-15
