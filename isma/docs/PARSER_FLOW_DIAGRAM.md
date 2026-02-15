# Parser Data Flow - What Gets Kept vs Dropped

## ChatGPT Message Processing

```
Raw Export (message object)
├── id: "msg_abc123"                           → ❌ DROPPED
├── author
│   └── role: "assistant"/"user"/"tool"        → ✅ role (but tool messages SKIPPED)
├── create_time: 1234567890                    → ✅ timestamp
├── update_time: 1234567891                    → ❌ DROPPED
├── content
│   ├── content_type: "text"/"code"/"execution_output"  → ❌ type ignored
│   └── parts: [...]                           → ⚠️  PARTIAL
│       ├── "string text"                      → ✅ extracted
│       └── {"text": "...", "type": "..."}     → ⚠️  only .text extracted, type ignored
├── status: "finished_successfully"            → ❌ DROPPED
├── end_turn: true                             → ❌ DROPPED
├── weight: 1.0                                → ❌ DROPPED
├── recipient: "python"/"dalle"/"browser"      → ❌ DROPPED (CRITICAL!)
├── channel: null                              → ❌ DROPPED
└── metadata
    ├── model_slug: "gpt-5-2-pro"              → ❌ DROPPED (CRITICAL!)
    ├── resolved_model_slug: "gpt-5-2-pro"     → ❌ DROPPED (CRITICAL!)
    ├── default_model_slug: "..."              → ❌ DROPPED
    ├── citations: [...]                       → ❌ DROPPED (CRITICAL!)
    ├── content_references: [...]              → ❌ DROPPED
    ├── search_result_groups: [...]            → ❌ DROPPED (CRITICAL!)
    ├── thinking_effort: "high"                → ❌ DROPPED (CRITICAL!)
    ├── message_type: "..."                    → ❌ DROPPED
    ├── request_id: "..."                      → ❌ DROPPED
    ├── is_async_task_result_message: true     → ❌ DROPPED
    ├── async_task_id: "..."                   → ❌ DROPPED
    ├── async_task_title: "..."                → ❌ DROPPED
    ├── async_task_type: "..."                 → ❌ DROPPED
    ├── parent_id: "..."                       → ❌ DROPPED
    ├── attachments: [...]                     → ✅ extracted (name, size, mime_type)
    └── ...other fields...                     → ❌ DROPPED

Output (simplified message)
├── role: "user"/"assistant"
├── text: "combined parts text"
├── timestamp: "2025-01-15T10:30:00Z"
└── attachments: [{name, size, mime_type}]
```

**Data Loss**: ~85% of metadata, ~100% of tool execution results, ~100% of citations

---

## Claude Message Processing

```
Raw Export (chat_message object)
├── uuid: "msg-xyz"                            → ❌ DROPPED
├── text: "fallback text"                      → ⚠️  used only if content[] empty
├── sender: "human"/"assistant"                → ✅ role
├── created_at: "2025-01-15T..."               → ✅ timestamp
├── updated_at: "2025-01-15T..."               → ❌ DROPPED
├── content: [...]                             → ⚠️  PARTIAL extraction
│   ├── {type: "thinking", ...}                → ❌ DROPPED (CRITICAL!)
│   │   ├── thinking: "actual thinking text"   → ❌ DROPPED (CRITICAL!)
│   │   ├── summaries: [...]                   → ❌ DROPPED (CRITICAL!)
│   │   ├── cut_off: false                     → ❌ DROPPED
│   │   ├── start_timestamp: "..."             → ❌ DROPPED
│   │   ├── stop_timestamp: "..."              → ❌ DROPPED
│   │   └── flags: [...]                       → ❌ DROPPED
│   ├── {type: "text", text: "...", ...}       → ✅ text extracted
│   │   ├── text: "visible text"               → ✅ extracted
│   │   ├── citations: [...]                   → ❌ DROPPED (CRITICAL!)
│   │   ├── start_timestamp: "..."             → ❌ DROPPED
│   │   ├── stop_timestamp: "..."              → ❌ DROPPED
│   │   └── flags: [...]                       → ❌ DROPPED
│   ├── {type: "tool_use", ...}                → ❌ DROPPED (CRITICAL!)
│   │   ├── id: "toolu_xyz"                    → ❌ DROPPED
│   │   ├── name: "read_file"                  → ❌ DROPPED
│   │   └── input: {...}                       → ❌ DROPPED
│   └── {type: "tool_result", ...}             → ❌ DROPPED (CRITICAL!)
│       ├── tool_use_id: "toolu_xyz"           → ❌ DROPPED
│       ├── content: "result..."               → ❌ DROPPED
│       └── is_error: false                    → ❌ DROPPED
├── attachments: [...]                         → ✅ extracted
│   └── {file_name, file_size, file_type, extracted_content}
└── files: [...]                               → ✅ extracted (name only)

Output (simplified message)
├── role: "user"/"assistant"
├── text: "concatenated text blocks"
├── timestamp: "2025-01-15T10:30:00Z"
└── attachments: [{name, size, mime_type, extracted_content}]
```

**Data Loss**: ~70% of content blocks, ~100% of tool use, ~100% of thinking, ~100% of citations

---

## Grok Response Processing

```
Raw Export (response object)
├── _id: "resp_123"                            → ⚠️  used for tree then DROPPED
├── conversation_id: "conv_456"                → ❌ DROPPED
├── message: "response text"                   → ✅ text
├── sender: "ASSISTANT"/"human"                → ✅ role
├── create_time: {...}                         → ✅ timestamp (MongoDB format handled)
├── model: "grok-3"/"grok-4"                   → ✅ model (first assistant only)
├── parent_response_id: "resp_122"             → ⚠️  used for tree then DROPPED
├── file_attachments: ["uuid1", "uuid2"]       → ✅ attachments (as UUIDs)
├── thinking_trace: "thinking text..."         → ✅ stored in _metadata.thinking_trace
├── thinking_start_time: {...}                 → ❌ DROPPED
├── thinking_end_time: {...}                   → ❌ DROPPED
├── steps: [...]                               → ✅ converted to tools (type="web_search")
├── web_search_results: [...]                  → ❌ DROPPED (CRITICAL!)
│   └── [{url, title, snippet, ...}]           → ❌ DROPPED (CRITICAL!)
├── cited_web_search_results: [...]            → ❌ DROPPED (CRITICAL!)
├── card_attachments_json: {...}               → ❌ DROPPED (CRITICAL!)
├── xpost_ids: ["123", "456"]                  → ❌ DROPPED
├── webpage_urls: ["https://..."]              → ❌ DROPPED
├── error: "error message"                     → ❌ DROPPED (CRITICAL!)
├── partial: false                             → ❌ DROPPED
├── manual: false                              → ❌ DROPPED
├── query: "search query"                      → ❌ DROPPED
├── query_type: "web"                          → ❌ DROPPED
├── xai_user_id: "user_789"                    → ❌ DROPPED
└── metadata
    ├── modelConfigOverride: {...}             → ❌ DROPPED
    └── requestModelDetails: {...}             → ❌ DROPPED

Output (simplified message)
├── role: "user"/"assistant"
├── text: "response text"
├── timestamp: "2025-01-15T10:30:00Z"
├── attachments: [{name: "grok_asset_uuid1234", type: "grok_asset", uuid: "..."}]
├── model: "grok-3"
├── _metadata: {thinking_trace: "..."}
└── _tools: [{type: "web_search", input: {...}}]
```

**Data Loss**: ~60% of metadata, ~100% of web search results, ~100% of rich cards, timestamps

---

## Exchange Grouping - What Happens After Messages Extracted

```
Messages Array
[
  {role: "user", text: "...", timestamp: "...", attachments: [...]},
  {role: "assistant", text: "...", timestamp: "...", attachments: [], _tools: [], _metadata: {}},
  {role: "assistant", text: "more...", ...},  // consecutive same-role
]

↓ _group_into_exchanges_with_tools_and_metadata()

Exchange Object
{
  index: 0,
  timestamp: "2025-01-15T...",
  user_prompt: "user text",
  responses: [{
    text: "concatenated assistant text",        ← multiple assistant messages merged
    tools: [...],                               ← _tools from all assistant messages
    artifacts: [...],                           ← extract_artifacts(text) - CODE FENCES ONLY
    metadata: {                                 ← _metadata merged
      thinking_trace: "..."                     ← Grok only
    }
  }],
  attachments: [...],                           ← user + assistant attachments
  file_references: [...]                        ← extract_file_references(user + assistant text)
}
```

**Note**:
- `_metadata` and `_tools` are internal keys added by Grok/Claude Code parsers
- These are merged during exchange grouping then removed from final output
- Only `thinking_trace` survives in response.metadata
- All other metadata fields are lost

---

## Artifact Extraction - What Gets Found vs Missed

```
extract_artifacts(text)
├── Regex: ```(\w*)\n(.*?)```                  → ✅ code fences
│   └── Captures: language, content
│
└── MISSED:
    ├── Inline code: `code`                    → ❌
    ├── HTML blocks                            → ❌
    ├── SVG diagrams                           → ❌
    ├── JSON data (not in fences)              → ❌
    ├── Tables (markdown/HTML)                 → ❌
    ├── LaTeX/math                             → ❌
    └── Formatted output (shell, REPL)         → ❌

Output: [{type: "code", language: "...", content: "...", fingerprint: "md5..."}]
```

---

## File Reference Extraction - What Gets Found vs Missed

```
extract_file_references(text)
├── Pattern 1: /path/to/file.ext               → ✅ absolute paths
├── Pattern 2: "Reading file.py"               → ✅ file operations
├── Pattern 3: `file.py`                       → ✅ backtick refs
│
└── MISSED:
    ├── URLs (http/https)                      → ❌
    ├── Git refs (SHAs, branches)              → ❌
    ├── npm/pip packages                       → ❌
    ├── Database names/tables                  → ❌
    ├── API endpoints                          → ❌
    ├── Relative paths (./file)                → ❌
    └── Environment vars ($VAR)                → ❌

Output: ["file.py", "/path/to/code.ts", ...]
```

---

## Summary: Information Preservation Rates

| Platform | Messages Captured | Metadata Preserved | Artifacts Captured | Tools Captured |
|----------|-------------------|--------------------|--------------------|----------------|
| **ChatGPT** | 60% (tool msgs skipped) | 15% | 30% (code fences only) | 0% |
| **Claude** | 80% (text blocks only) | 30% | 30% (code fences only) | 0% |
| **Grok** | 95% | 40% | 30% (code fences only) | 20% (steps only) |
| **Overall** | **75%** | **25%** | **30%** | **5%** |

**Net Result**: ~40-60% of rich interaction data is **lost on the floor**.

---

## Critical Path Issues

### 1. Tool Execution Results (ChatGPT)
```
User: "Run this Python code: print(sum([1,2,3]))"
Assistant: [tool_use: python]
Tool: "6\n"                                    ← DROPPED
Assistant: "The sum is 6"
```
**What's stored**: "The sum is 6"
**What's lost**: The actual execution ("6\n")

### 2. Extended Thinking (Claude)
```
[thinking block: 2000 words of reasoning]      ← DROPPED
[text block: "Based on this analysis..."]
```
**What's stored**: "Based on this analysis..."
**What's lost**: The 2000-word reasoning chain

### 3. Web Search Provenance (Grok)
```
[web_search_results: 10 sources with URLs/snippets]  ← DROPPED
[message: "According to recent research..."]
```
**What's stored**: "According to recent research..."
**What's lost**: WHERE the research came from (URLs, titles, snippets)

### 4. Multi-Model Conversations (ChatGPT)
```
Message 1 (GPT-4o): "Let me think about this..."
Message 2 (o1): "After deep reasoning..."
Message 3 (GPT-4): "To summarize..."
```
**What's stored**: All messages with conversation-level model (e.g., "gpt-4")
**What's lost**: Which model actually said what

---

## Recommended Fix Priority

### 🔴 CRITICAL (Breaking Core Use Cases)
1. ChatGPT role="tool" messages
2. ChatGPT per-message model_slug
3. Claude thinking blocks
4. Claude tool_use/tool_result
5. Grok web_search_results

### 🟡 IMPORTANT (Missing Context)
6. Citations (all platforms)
7. Message IDs (deduplication)
8. Thinking timestamps
9. Grok rich cards
10. Error fields

### 🟢 NICE_TO_HAVE (Audit/Debug)
11. Update timestamps
12. Status/weight fields
13. Request IDs
14. Async task metadata
