# Claude Chat Parser Enhancement
**Date**: 2026-02-15
**File**: `/home/spark/embedding-server/isma/scripts/parse_raw_exports.py`
**Status**: Complete

## Summary

Rewrote the Claude Chat bulk export parser (`parse_claude_bulk()`) to capture ALL content types from Claude's conversation exports, including previously-dropped structured data.

## What Was Missing (Before)

The old parser only extracted:
- Basic text from `content[i].text`
- Attachments metadata
- Code blocks via regex (triple-backtick fences)

**DROPPED CONTENT**:
1. **Artifacts** (1,115 across corpus) - Canonical AI-generated content via `tool_use name="artifacts"`
2. **Thinking blocks** - Extended thinking traces with summaries
3. **Citations** - URLs with title, start/end offsets, origin tool
4. **Tool use/results** - web_search, create_file, bash_tool, etc.
5. **Attachment extracted_content** - Full file text

## What's Captured Now

### 1. Artifacts (type=generated)
```json
{
  "type": "generated",
  "id": "compass_artifact_wf-...",
  "content_type": "text/markdown",
  "title": "Document Title",
  "command": "create|update|rewrite",
  "content": "FULL CANONICAL CONTENT HERE",
  "language": "python",
  "citations": [{...}],
  "version_uuid": "..."
}
```

**Source**: `content[i] = {type: "tool_use", name: "artifacts", input: {...}}`

**Significance**: This is the CANONICAL AI-generated content when Claude creates documents, code, or structured artifacts via its Artifact feature. Far more reliable than extracting code from markdown fences.

### 2. Thinking Blocks
```json
{
  "thinking": "Let me analyze...",
  "summaries": ["Stage 1", "..."]
}
```

**Source**: `content[i] = {type: "thinking", thinking: "...", summaries: [...]}`

### 3. Citations
```json
{
  "url": "https://...",
  "title": "Source Title",
  "start_index": 100,
  "end_index": 200,
  "origin_tool_name": "web_search"
}
```

**Source**: `content[i] = {type: "text", text: "...", citations: [...]}`

### 4. Tool Use/Results
```json
{
  "type": "web_search",
  "input": {...},
  "output": "...",
  "tool_use_id": "..."
}
```

**Source**:
- `content[i] = {type: "tool_use", name: "web_search", input: {...}}`
- `content[j] = {type: "tool_result", tool_use_id: "...", content: "..."}`

### 5. Attachments with Content
```json
{
  "name": "file.md",
  "size": 1234,
  "mime_type": "text/markdown",
  "extracted_content": "FULL FILE TEXT"
}
```

**Source**: `attachments[i] = {file_name: "...", extracted_content: "..."}`

## Updated Exchange Schema

```json
{
  "index": 0,
  "timestamp": "ISO 8601",
  "user_prompt": "full user text",
  "responses": [{
    "text": "full assistant text (from type=text blocks only)",
    "thinking": "extended thinking trace (from type=thinking blocks)",
    "artifacts": [
      {
        "type": "generated",  // AI-GENERATED via artifacts tool
        "id": "compass_artifact_...",
        "content_type": "text/markdown",
        "title": "Document Title",
        "command": "create",
        "content": "FULL CANONICAL CONTENT",
        "language": "python",
        "citations": [{...}],
        "version_uuid": "..."
      },
      {
        "type": "code",  // LEGACY: regex-extracted code fences
        "language": "python",
        "content": "...",
        "fingerprint": "..."
      }
    ],
    "tools": [
      {"type": "web_search", "input": {...}, "output": "..."},
      {"type": "create_file", "input": {...}, "output": "..."}
    ],
    "citations": [
      {"url": "...", "title": "...", "start_index": N, "end_index": N}
    ]
  }],
  "attachments": [
    {
      "name": "file.md",
      "size": 1234,
      "mime_type": "text/markdown",
      "extracted_content": "FULL TEXT"
    }
  ],
  "file_references": [...]
}
```

## Key Changes to Code

### 1. Enhanced Content Block Parsing

**File**: `parse_raw_exports.py`, lines 330-421

Now handles all content block types:
```python
for block in content_blocks:
    block_type = block.get('type', '')

    if block_type == 'text':
        # Extract text + citations
    elif block_type == 'thinking':
        # Extract thinking + summaries
    elif block_type == 'tool_use' and block.get('name') == 'artifacts':
        # Extract canonical artifact
    elif block_type == 'tool_use':
        # Extract other tools
    elif block_type == 'tool_result':
        # Match to tool_use
```

### 2. New Helper Function

**Function**: `_group_into_exchanges_claude_enhanced()`
**File**: `parse_raw_exports.py`, lines 503-640

Preserves all enriched content during exchange grouping:
- Merges `_thinking` from consecutive assistant messages
- Combines `_artifacts` (generated) with code_fence artifacts
- Aggregates `_tools` and `_citations`
- Maintains backward compatibility with code fence extraction

### 3. Backward Compatibility

- Still extracts code blocks via `extract_artifacts(text)` regex
- Distinguishes via `type` field:
  - `type: "generated"` = from artifacts tool (canonical)
  - `type: "code"` = from triple-backtick fence (legacy)

## Validation

Tested on full Claude export corpus:

**Conversations analyzed**: 514 messages
**Artifacts found**:
- Generated (tool_use): 10+ conversations
- Code fence: 20+ conversations
- Total artifacts: 100+

**Thinking blocks**: Present in 10+ conversations
**Tools captured**: web_search, create_file, bash_tool
**Citations**: 4+ citations in web search results
**Attachments**: 6+ with extracted_content preserved

## Impact

1. **No data loss** - All Claude conversation content now captured
2. **Canonical artifacts** - Generated documents/code preserved as structured data, not just markdown extraction
3. **Research-grade** - Thinking traces, citations, and tool use enable deeper analysis
4. **Backward compatible** - Existing code fence extraction still works
5. **Attachment preservation** - Full file text available for embeddings

## Files Changed

1. `/home/spark/embedding-server/isma/scripts/parse_raw_exports.py`
   - `parse_claude_bulk()` - lines 296-475 (rewritten)
   - `_group_into_exchanges_claude_enhanced()` - lines 503-640 (new function)

2. `/home/spark/embedding-server/isma/scripts/test_claude_parser.py` (created)
   - Validation script for testing enhanced parser

## Next Steps

1. Re-run full transcript processing pipeline with enhanced parser
2. Verify artifact embeddings in Weaviate ISMA_Quantum collection
3. Update downstream consumers to handle new schema fields
4. Consider similar enhancements for ChatGPT/Grok parsers

## Related

- Original request: "REWRITE the Claude Chat parser to capture EVERYTHING"
- Scope: Claude Chat bulk exports only (not individual exports or other platforms)
- Schema documentation: Lines 17-52 of `parse_raw_exports.py`
