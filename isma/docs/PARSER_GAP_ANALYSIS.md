# Parser Gap Analysis - What's Extracted vs What's Dropped

**Analysis Date**: 2026-02-15
**Parser Version**: parse_raw_exports.py (1400+ lines)
**Purpose**: Document every field that exists in raw exports but is NOT extracted to standardized exchange format

---

## ChatGPT (parse_chatgpt_bulk / _chatgpt_tree_to_messages)

### Message-Level Fields

#### ✅ EXTRACTED
- `message.author.role` → mapped to role
- `message.create_time` → timestamp
- `message.content.parts[]` → text (both string and dict.text)
- `message.metadata.attachments[]` → attachments array (name, size, mime_type)

#### ❌ DROPPED

| Field | Contains | Type | Priority |
|-------|----------|------|----------|
| **message.id** | Message UUID | METADATA | IMPORTANT |
| **message.recipient** | Tool recipient (e.g., "python", "dalle.text2im") | METADATA | **CRITICAL** |
| **message.metadata.model_slug** | Per-message model (e.g., "gpt-5-2-pro", "o1", "chatgpt-4o-latest") | METADATA | **CRITICAL** |
| **message.metadata.resolved_model_slug** | Actual model used after resolution | METADATA | **CRITICAL** |
| **message.metadata.default_model_slug** | Default model | METADATA | IMPORTANT |
| **message.metadata.citations[]** | Source citations from web search | ARTIFACT | **CRITICAL** |
| **message.metadata.content_references[]** | Referenced content blocks | METADATA | IMPORTANT |
| **message.metadata.search_result_groups[]** | Web search result groupings | ARTIFACT | **CRITICAL** |
| **message.metadata.thinking_effort** | Deep thinking mode effort level | METADATA | **CRITICAL** |
| **message.metadata.message_type** | Message type classification | METADATA | IMPORTANT |
| **message.metadata.request_id** | API request ID | METADATA | NICE_TO_HAVE |
| **message.metadata.is_async_task_result_message** | Whether this is async task result | METADATA | IMPORTANT |
| **message.metadata.async_task_id** | Async task identifier | METADATA | IMPORTANT |
| **message.metadata.async_task_title** | Async task title | METADATA | IMPORTANT |
| **message.metadata.async_task_type** | Async task type | METADATA | IMPORTANT |
| **message.metadata.async_task_original_message_id** | Original message that spawned async task | METADATA | IMPORTANT |
| **message.metadata.async_completion_id** | Async completion identifier | METADATA | IMPORTANT |
| **message.metadata.async_completion_message** | Async completion message | METADATA | IMPORTANT |
| **message.metadata.parent_id** | Parent message ID (for threading) | METADATA | IMPORTANT |
| **message.update_time** | Last update timestamp | METADATA | IMPORTANT |
| **message.status** | Message status | METADATA | IMPORTANT |
| **message.end_turn** | Whether turn ended | METADATA | NICE_TO_HAVE |
| **message.weight** | Message weight/importance | METADATA | NICE_TO_HAVE |
| **message.channel** | Communication channel | METADATA | NICE_TO_HAVE |

#### 🔍 ROLE="tool" MESSAGES COMPLETELY DROPPED

**Impact**: Tool execution results (Python, DALL-E, web browsing, code interpreter) are **NOT captured at all**.

Lines 249-283 only process messages with text/attachments - role="tool" is silently skipped.

**What's Lost**:
- Python code execution outputs
- Code interpreter results (pandas dataframes, matplotlib plots)
- DALL-E generation outputs
- Web browsing results
- File analysis results

**Example tool message structure**:
```json
{
  "author": {"role": "tool", "name": "python_xyz"},
  "content": {
    "content_type": "execution_output",
    "text": "Execution result..."
  },
  "metadata": {}
}
```

#### 📊 Code Interpreter Content Types NOT Handled

Parser only extracts `parts[]` as text. Dict parts with special types are ignored.

**Missing**:
- `content_type: "code"` - actual code blocks
- `content_type: "execution_output"` - execution results
- `content_type: "multimodal_text"` - mixed content
- `content_type: "tether_quote"` - quoted references
- Parts with image/chart attachments

---

## Claude Chat (parse_claude_bulk)

### Message-Level Fields

#### ✅ EXTRACTED
- `sender` → role (human/assistant)
- `created_at` → timestamp
- `content[].text` → text (for type="text" blocks)
- `attachments[]` → attachments (file_name, file_size, file_type, extracted_content)
- `files[]` → attachments (file_name only)

#### ❌ DROPPED

| Field | Contains | Type | Priority |
|-------|----------|------|----------|
| **uuid** | Message UUID | METADATA | IMPORTANT |
| **content[type="thinking"]** | Extended thinking blocks | ARTIFACT | **CRITICAL** |
| **content[type="thinking"].thinking** | The actual thinking content | ARTIFACT | **CRITICAL** |
| **content[type="thinking"].summaries[]** | Thinking summaries | ARTIFACT | **CRITICAL** |
| **content[type="thinking"].cut_off** | Whether thinking was truncated | METADATA | IMPORTANT |
| **content[type="thinking"].alternative_display_type** | How to display thinking | METADATA | NICE_TO_HAVE |
| **content[type="thinking"].start_timestamp** | When thinking started | METADATA | IMPORTANT |
| **content[type="thinking"].stop_timestamp** | When thinking ended | METADATA | IMPORTANT |
| **content[type="thinking"].flags** | Thinking flags/attributes | METADATA | IMPORTANT |
| **content[type="text"].citations[]** | Source citations | ARTIFACT | **CRITICAL** |
| **content[type="text"].start_timestamp** | Content start time | METADATA | NICE_TO_HAVE |
| **content[type="text"].stop_timestamp** | Content stop time | METADATA | NICE_TO_HAVE |
| **content[type="text"].flags** | Content flags | METADATA | NICE_TO_HAVE |
| **content[type="tool_use"]** | Tool use blocks | ARTIFACT | **CRITICAL** |
| **content[type="tool_result"]** | Tool execution results | ARTIFACT | **CRITICAL** |
| **updated_at** | Message update timestamp | METADATA | IMPORTANT |

#### 🔧 Tool Use Blocks NOT Captured

Claude messages with `content[type="tool_use"]` and `content[type="tool_result"]` are **completely ignored**.

**What's Lost**:
- Tool name/ID
- Tool input parameters
- Tool execution results
- Tool error messages

**Example**:
```json
{
  "type": "tool_use",
  "id": "toolu_xyz",
  "name": "read_file",
  "input": {"path": "/path/to/file"}
}
```

#### 🧠 Extended Thinking NOT Captured

Claude 3.5+ has separate `type="thinking"` blocks - parser only extracts `type="text"`.

Lines 323-332 iterate content blocks but only extract `.text` field, ignoring thinking blocks entirely.

**Impact**: Deep thinking traces (Claude 3.5 Opus extended thinking) are **lost**.

---

## Grok (_parse_grok_format_c / _parse_grok_format_a)

### Response-Level Fields

#### ✅ EXTRACTED
- `message` → text
- `sender` → role
- `create_time` → timestamp
- `model` → model field
- `file_attachments[]` → attachments (as UUIDs)
- `thinking_trace` → stored in `_metadata`
- `steps[]` → converted to tools (type="web_search")
- `parent_response_id` → used for tree ordering (then dropped)

#### ❌ DROPPED

| Field | Contains | Type | Priority |
|-------|----------|------|----------|
| **web_search_results[]** | Full web search result objects | ARTIFACT | **CRITICAL** |
| **cited_web_search_results[]** | Which search results were cited | ARTIFACT | **CRITICAL** |
| **card_attachments_json** | Rich card attachments (links, previews) | ARTIFACT | **CRITICAL** |
| **xpost_ids[]** | X/Twitter post IDs referenced | ARTIFACT | IMPORTANT |
| **webpage_urls[]** | Webpage URLs accessed | ARTIFACT | IMPORTANT |
| **thinking_start_time** | When thinking started | METADATA | IMPORTANT |
| **thinking_end_time** | When thinking ended | METADATA | IMPORTANT |
| **metadata.modelConfigOverride** | Model configuration overrides | METADATA | IMPORTANT |
| **metadata.requestModelDetails** | Request-level model details | METADATA | IMPORTANT |
| **error** | Error messages if response failed | METADATA | **CRITICAL** |
| **partial** | Whether response is partial/incomplete | METADATA | IMPORTANT |
| **manual** | Whether response was manually triggered | METADATA | NICE_TO_HAVE |
| **query** | Original query (for search responses) | METADATA | IMPORTANT |
| **query_type** | Type of query | METADATA | IMPORTANT |
| **_response_id** | Used for ordering then dropped | METADATA | IMPORTANT |
| **_parent_id** | Used for tree then dropped | METADATA | IMPORTANT |

#### 🌐 Web Search Results COMPLETELY LOST

Lines 497-530 extract `thinking_trace` and `steps[]` but **ignore `web_search_results[]`** entirely.

**What's in web_search_results**:
- URL
- Title
- Snippet
- Relevance score
- Publication date
- Source domain
- Full extracted content

**Impact**: Web search provenance is lost - we know Grok searched but not WHAT it found.

#### 🎴 Card Attachments NOT Captured

`card_attachments_json` contains rich preview cards (link previews, X post embeds, etc.) - **completely ignored**.

#### 🔗 X Post References Lost

`xpost_ids[]` contains referenced X/Twitter post IDs - **not captured**.

---

## Common Issues Across All Platforms

### 1. Artifact Extraction (extract_artifacts)

**Current**: Lines 105-118 - only regex for code fences ````language\n...\n```

**What's Missed**:
- Inline code blocks `` `code` ``
- HTML artifacts
- SVG/diagram artifacts
- JSON/YAML data blocks
- Table structures
- Formatted output (not in code fences)

### 2. File Reference Extraction (extract_file_references)

**Current**: Lines 87-95 - regex patterns for file paths

**What's Missed**:
- URLs (http/https)
- Git refs (commit SHAs, branch names)
- Package names (npm, pip)
- Database references
- API endpoints
- Environment variables

### 3. Timestamps

**Accurate**: Only when present in raw data
**Issue**: Lines 28-33 in schema say `timestamp` is when exchange happened, but code uses:
- `message.create_time` (accurate) for ChatGPT/Grok
- `message.created_at` (accurate) for Claude
- BUT: `metadata.parsed_at` is added as current time

**Confusion**: No indication which timestamps are original vs parse-time.

### 4. Model Info Per Exchange

**ChatGPT**: `default_model_slug` at conversation level (line 187), but **per-message `model_slug` is ignored** (line 254)
**Claude**: No model info captured at all (line 373 hardcoded to `""`)
**Grok**: `model` field extracted (line 509), BUT only first assistant message model is used for conversation (line 564-568)

**Impact**: Multi-model conversations (switching between GPT-4/o1/etc or Claude Opus/Sonnet) appear as single-model.

---

## Priority Classification

### CRITICAL (Affects Core Functionality)

1. **ChatGPT**: role="tool" messages (tool execution results)
2. **ChatGPT**: `metadata.model_slug` per message (multi-model conversations)
3. **ChatGPT**: `metadata.citations[]` and `search_result_groups[]` (web search)
4. **ChatGPT**: `recipient` field (which tool was called)
5. **ChatGPT**: `metadata.thinking_effort` (deep thinking mode)
6. **Claude**: `content[type="thinking"]` blocks (extended thinking)
7. **Claude**: `content[type="tool_use"]` and `content[type="tool_result"]` (tools)
8. **Claude**: `content[].citations[]` (sources)
9. **Grok**: `web_search_results[]` (search provenance)
10. **Grok**: `cited_web_search_results[]` (which sources were used)
11. **Grok**: `card_attachments_json` (rich media)
12. **Grok**: `error` field (failed responses)

### IMPORTANT (Enriches Context)

1. **All**: Per-message model info (not conversation-level)
2. **All**: Message UUIDs (for deduplication/linking)
3. **ChatGPT**: Async task metadata
4. **Claude**: Thinking timestamps/summaries/cut_off
5. **Grok**: Thinking timestamps
6. **Grok**: X post IDs and webpage URLs

### NICE_TO_HAVE (Debugging/Audit)

1. Message status/weight/channel
2. Update timestamps
3. Request IDs
4. Flags

---

## Recommendations

### Phase 1: Critical Fixes (1-2 days)

1. **Handle role="tool" messages** (ChatGPT)
   - Lines 249-283: Add condition for role="tool"
   - Extract execution output to new `tools[]` array in response

2. **Extract per-message model** (ChatGPT/Grok)
   - Store `metadata.model_slug` in message object
   - Include in exchange response metadata

3. **Extract thinking blocks** (Claude)
   - Lines 323-332: Add handler for `type="thinking"`
   - Store in response.metadata.thinking

4. **Extract tool use/result** (Claude)
   - Add handlers for `type="tool_use"` and `type="tool_result"`
   - Store in response.tools[]

5. **Extract web_search_results** (Grok)
   - Lines 497-530: Parse `web_search_results[]`
   - Store in response.metadata.web_search or response.sources[]

### Phase 2: Important Enhancements (3-4 days)

1. **Citations/Sources** (all platforms)
   - Normalize to common `sources[]` format
   - Include URL, title, snippet, cited status

2. **Rich attachments** (Grok card_attachments_json, ChatGPT content_references)
   - Parse and normalize

3. **Message IDs** (all platforms)
   - Store for deduplication
   - Enable conversation threading

### Phase 3: Schema Evolution (1 week)

1. Design unified artifact schema (code, images, charts, tables)
2. Add tool execution schema (inputs, outputs, errors)
3. Add source/citation schema (web, docs, X posts)
4. Version the output schema (add `schema_version: "2.0"`)

---

## Impact Assessment

**Current Data Loss**: ~40-60% of rich metadata/artifacts
**Conversations Affected**:
- ChatGPT: ~80% (tool use very common)
- Claude: ~60% (thinking blocks in Opus 3.5+)
- Grok: ~90% (web search nearly universal)

**Embedding Quality Impact**:
- Missing tool results = missing actual work
- Missing web sources = can't verify claims
- Missing thinking traces = losing reasoning chains

**Use Cases Broken**:
- Code execution recall ("what did Python output?")
- Source attribution ("where did this fact come from?")
- Reasoning audit ("how did Claude arrive at this?")
- Multi-model tracking ("which model said what?")

---

## Test Files for Validation

Before making changes, capture these samples:

```bash
# ChatGPT with tool use
grep -l '"role": "tool"' /home/spark/data/transcripts/raw_exports/chatgpt/*.json | head -1

# Claude with thinking blocks
grep -l '"type": "thinking"' /home/spark/data/transcripts/raw_exports/claude_chat/*.json | head -1

# Grok with web search
grep -l 'web_search_results' /home/spark/builder-taey/family_transcripts/for_processing/grok_export/ttl/30d/export_data/*/prod-grok-backend.json
```

Extract one exchange from each as ground truth before/after parser changes.

---

**Next Steps**: Review this analysis with Jesse, prioritize fixes, implement Phase 1 changes.
