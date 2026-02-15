# Parser Quick Fix Guide - Implementation Checklist

**For**: Implementing Phase 1 critical fixes
**Time**: 1-2 days
**Files**: `/home/spark/embedding-server/isma/scripts/parse_raw_exports.py`

---

## Fix #1: ChatGPT Tool Messages (Lines 249-283)

### Current Code
```python
def _chatgpt_tree_to_messages(mapping: Dict) -> List[Dict]:
    # ...
    if msg:
        author = msg.get('author', {})
        role = author.get('role', '')
        # ...
        if text or attachments:  # Skip empty nodes
            ordered.append({
                "role": role,
                "text": text,
                "timestamp": unix_to_iso(create_time),
                "attachments": attachments,
            })
```

### Problem
- `role="tool"` messages have no text/attachments → skipped
- Tool execution results lost

### Fix
```python
def _chatgpt_tree_to_messages(mapping: Dict) -> List[Dict]:
    # ...
    if msg:
        author = msg.get('author', {})
        role = author.get('role', '')
        # ...

        # NEW: Extract tool metadata
        tool_name = author.get('name', '')  # e.g., "python_abc123"
        recipient = msg.get('recipient', '')  # e.g., "python", "dalle"

        if text or attachments or role == 'tool':  # ← CHANGED
            msg_obj = {
                "role": role,
                "text": text,
                "timestamp": unix_to_iso(create_time),
                "attachments": attachments,
            }

            # NEW: Add tool info
            if role == 'tool':
                msg_obj['_tool_name'] = tool_name
                msg_obj['_tool_output'] = text
            if recipient and recipient != 'all':
                msg_obj['_tool_recipient'] = recipient

            ordered.append(msg_obj)
```

### Test
```bash
grep -l '"role": "tool"' /home/spark/data/transcripts/raw_exports/chatgpt/conversations.json
# Extract one conversation, verify tool messages appear in output
```

---

## Fix #2: ChatGPT Per-Message Model (Lines 250-254)

### Current Code
```python
if msg:
    author = msg.get('author', {})
    role = author.get('role', '')
    content = msg.get('content', {})
    parts = content.get('parts', [])
    meta = msg.get('metadata', {})  # ← metadata retrieved but not used
```

### Problem
- `meta['model_slug']` exists but not extracted
- Multi-model conversations appear as single-model

### Fix
```python
if msg:
    author = msg.get('author', {})
    role = author.get('role', '')
    content = msg.get('content', {})
    parts = content.get('parts', [])
    meta = msg.get('metadata', {})

    # NEW: Extract model info
    model_slug = meta.get('model_slug', '')
    resolved_model = meta.get('resolved_model_slug', '')

    # ... (text extraction) ...

    msg_obj = {
        "role": role,
        "text": text,
        "timestamp": unix_to_iso(create_time),
        "attachments": attachments,
    }

    # NEW: Add model info
    if model_slug:
        msg_obj['_model'] = model_slug
    if resolved_model and resolved_model != model_slug:
        msg_obj['_resolved_model'] = resolved_model

    ordered.append(msg_obj)
```

### Test
```bash
# Find conversation with o1 messages
grep -A5 '"model_slug": "o1"' /home/spark/data/transcripts/raw_exports/chatgpt/conversations.json
```

---

## Fix #3: ChatGPT Citations (Lines 250-254)

### Current Code
```python
meta = msg.get('metadata', {})
# citations extracted but not stored
```

### Problem
- `meta['citations']` and `meta['search_result_groups']` ignored
- Web search provenance lost

### Fix
```python
meta = msg.get('metadata', {})

# NEW: Extract citations and search results
citations = meta.get('citations', [])
search_groups = meta.get('search_result_groups', [])

# ... (text extraction) ...

msg_obj = {
    "role": role,
    "text": text,
    "timestamp": unix_to_iso(create_time),
    "attachments": attachments,
}

# NEW: Add sources
sources = []
for cit in citations:
    if isinstance(cit, dict):
        sources.append({
            "type": "citation",
            "url": cit.get('url', ''),
            "title": cit.get('metadata', {}).get('title', ''),
        })

for group in search_groups:
    for result in group.get('results', []):
        if isinstance(result, dict):
            sources.append({
                "type": "search_result",
                "url": result.get('url', ''),
                "title": result.get('title', ''),
                "snippet": result.get('snippet', ''),
                "publication_date": result.get('publication_date', ''),
            })

if sources:
    msg_obj['_sources'] = sources

ordered.append(msg_obj)
```

### Test
```bash
grep -l 'search_result_groups' /home/spark/data/transcripts/raw_exports/chatgpt/conversations.json
```

---

## Fix #4: Claude Thinking Blocks (Lines 323-332)

### Current Code
```python
content_blocks = msg.get('content', [])
text_parts = []
for block in content_blocks:
    if isinstance(block, dict):
        t = block.get('text', '')  # ← only extracts text blocks
        if t:
            text_parts.append(t)
```

### Problem
- `type="thinking"` blocks skipped
- Reasoning traces lost

### Fix
```python
content_blocks = msg.get('content', [])
text_parts = []
thinking_parts = []  # NEW
tool_uses = []       # NEW
tool_results = []    # NEW

for block in content_blocks:
    if isinstance(block, dict):
        block_type = block.get('type', '')

        if block_type == 'text':
            t = block.get('text', '')
            if t:
                text_parts.append(t)

        elif block_type == 'thinking':  # NEW
            thinking = block.get('thinking', '')
            if thinking:
                thinking_parts.append({
                    "trace": thinking,
                    "summaries": block.get('summaries', []),
                    "start_time": block.get('start_timestamp', ''),
                    "stop_time": block.get('stop_timestamp', ''),
                    "cut_off": block.get('cut_off', False),
                })

        elif block_type == 'tool_use':  # NEW
            tool_uses.append({
                "id": block.get('id', ''),
                "name": block.get('name', ''),
                "input": block.get('input', {}),
            })

        elif block_type == 'tool_result':  # NEW
            tool_results.append({
                "tool_use_id": block.get('tool_use_id', ''),
                "content": block.get('content', ''),
                "is_error": block.get('is_error', False),
            })

# ... (message creation) ...

msg_obj = {
    "role": role,
    "text": text,
    "timestamp": timestamp,
    "attachments": attachments,
}

# NEW: Add thinking/tools
if thinking_parts:
    msg_obj['_thinking'] = thinking_parts
if tool_uses:
    msg_obj['_tool_uses'] = tool_uses
if tool_results:
    msg_obj['_tool_results'] = tool_results

messages.append(msg_obj)
```

### Test
```bash
grep -l '"type": "thinking"' /home/spark/data/transcripts/raw_exports/claude_chat/conversations.json
```

---

## Fix #5: Claude Citations (Lines 323-332)

### Current Code
```python
if block_type == 'text':
    t = block.get('text', '')
    if t:
        text_parts.append(t)
    # citations ignored
```

### Problem
- `block.citations[]` not extracted
- Document references lost

### Fix
```python
citations = []  # NEW (at message level)

for block in content_blocks:
    if isinstance(block, dict):
        block_type = block.get('type', '')

        if block_type == 'text':
            t = block.get('text', '')
            if t:
                text_parts.append(t)

            # NEW: Extract citations from text block
            block_citations = block.get('citations', [])
            for cit in block_citations:
                if isinstance(cit, dict):
                    citations.append({
                        "type": "document",
                        "document_id": cit.get('document_id', ''),
                        "document_name": cit.get('document_name', ''),
                        "page_number": cit.get('page_number'),
                        "quote": cit.get('quote', ''),
                    })

# ... (message creation) ...

if citations:
    msg_obj['_sources'] = citations
```

---

## Fix #6: Grok Web Search Results (Lines 497-530)

### Current Code
```python
# Extract thinking trace
thinking = resp.get('thinking_trace', '')

# Extract web search steps
steps = resp.get('steps', [])

# Build message object
msg = {
    "role": role,
    "text": text.strip() if text else '',
    "timestamp": ts,
    "attachments": attachments,
    "model": model,
    "_response_id": resp_id,
    "_parent_id": resp.get('parent_response_id'),
}

# Store metadata and tools
metadata = {}
tools = []

if thinking:
    metadata['thinking_trace'] = thinking

if steps:
    for step in steps:
        if isinstance(step, dict):
            tools.append({
                "type": "web_search",
                "input": step,
            })

# web_search_results ignored
```

### Problem
- `resp['web_search_results']` not extracted
- Search provenance lost

### Fix
```python
# Extract thinking trace
thinking = resp.get('thinking_trace', '')
thinking_start = resp.get('thinking_start_time')
thinking_end = resp.get('thinking_end_time')

# Extract web search steps
steps = resp.get('steps', [])

# NEW: Extract web search results
web_results = resp.get('web_search_results', [])
cited_results = resp.get('cited_web_search_results', [])

# Build message object
msg = {
    "role": role,
    "text": text.strip() if text else '',
    "timestamp": ts,
    "attachments": attachments,
    "model": model,
    "_response_id": resp_id,
    "_parent_id": resp.get('parent_response_id'),
}

# Store metadata and tools
metadata = {}
tools = []
sources = []  # NEW

if thinking:
    metadata['thinking_trace'] = thinking
    if thinking_start:
        metadata['thinking_start'] = ensure_iso(thinking_start)
    if thinking_end:
        metadata['thinking_end'] = ensure_iso(thinking_end)

if steps:
    for step in steps:
        if isinstance(step, dict):
            tools.append({
                "type": "web_search",
                "input": step,
            })

# NEW: Extract web search results
for idx, result in enumerate(web_results):
    if isinstance(result, dict):
        sources.append({
            "type": "web_search",
            "url": result.get('url', ''),
            "title": result.get('title', ''),
            "snippet": result.get('snippet', ''),
            "domain": result.get('domain', ''),
            "relevance_score": result.get('relevance_score'),
            "publication_date": result.get('publication_date', ''),
            "cited": idx in cited_results,
        })

msg['_metadata'] = metadata
msg['_tools'] = tools
if sources:
    msg['_sources'] = sources  # NEW
```

### Test
```bash
grep -l 'web_search_results' /home/spark/builder-taey/family_transcripts/for_processing/grok_export/ttl/30d/export_data/*/prod-grok-backend.json
```

---

## Fix #7: Grok Card Attachments (Lines 497-530)

### Current Code
```python
# card_attachments_json ignored
```

### Problem
- Rich media cards not extracted
- X posts, link previews lost

### Fix
```python
# NEW: Extract card attachments
card_json = resp.get('card_attachments_json', {})
xpost_ids = resp.get('xpost_ids', [])
webpage_urls = resp.get('webpage_urls', [])

# ... (in message building) ...

# NEW: Parse card attachments
cards = []
if isinstance(card_json, dict):
    for card in card_json.get('cards', []):
        if isinstance(card, dict):
            card_type = card.get('type', '')
            if card_type == 'x_post':
                cards.append({
                    "type": "x_post",
                    "post_id": card.get('post_id', ''),
                    "author": card.get('author', ''),
                    "text": card.get('text', ''),
                    "likes": card.get('likes', 0),
                    "retweets": card.get('retweets', 0),
                    "posted_at": card.get('posted_at', ''),
                })
            elif card_type == 'link_preview':
                cards.append({
                    "type": "link_preview",
                    "url": card.get('url', ''),
                    "title": card.get('title', ''),
                    "description": card.get('description', ''),
                    "image_url": card.get('image_url', ''),
                    "author": card.get('author', ''),
                    "publication": card.get('publication', ''),
                })

if cards:
    msg['_cards'] = cards
if xpost_ids:
    metadata['xpost_ids'] = xpost_ids
if webpage_urls:
    metadata['webpage_urls'] = webpage_urls
```

### Test
```bash
grep -l 'card_attachments_json' /home/spark/builder-taey/family_transcripts/for_processing/grok_export/ttl/30d/export_data/*/prod-grok-backend.json
```

---

## Exchange Grouping Update (Lines 1061-1319)

### Current Code
```python
def _group_into_exchanges_with_tools_and_metadata(messages: List[Dict], platform: str):
    # ... grouping logic ...

    response_obj = {
        "text": resp_text,
        "tools": all_tools,
        "artifacts": extract_artifacts(resp_text),
    }
    if all_metadata:
        response_obj['metadata'] = all_metadata
```

### Problem
- `_sources`, `_thinking`, `_tool_uses`, `_cards` not merged
- New fields lost during grouping

### Fix
```python
def _group_into_exchanges_with_tools_and_metadata(messages: List[Dict], platform: str):
    # ... grouping logic ...

    resp_texts = []
    all_tools = []
    all_metadata = {}
    all_sources = []        # NEW
    all_thinking = []       # NEW
    all_tool_uses = []      # NEW
    all_tool_results = []   # NEW
    all_cards = []          # NEW

    for r in current_responses:
        if r.get('text'):
            resp_texts.append(r['text'])
        all_tools.extend(r.get('_tools', []))
        all_sources.extend(r.get('_sources', []))        # NEW
        all_thinking.extend(r.get('_thinking', []))      # NEW
        all_tool_uses.extend(r.get('_tool_uses', []))    # NEW
        all_tool_results.extend(r.get('_tool_results', []))  # NEW
        all_cards.extend(r.get('_cards', []))            # NEW

        # Merge metadata
        meta = r.get('_metadata', {})
        for k, v in meta.items():
            # ... existing merge logic ...

    response_obj = {
        "text": resp_text,
        "tools": all_tools,
        "artifacts": extract_artifacts(resp_text),
    }
    if all_metadata:
        response_obj['metadata'] = all_metadata
    if all_sources:          # NEW
        response_obj['sources'] = all_sources
    if all_thinking:         # NEW
        response_obj['thinking'] = all_thinking
    if all_tool_uses:        # NEW
        response_obj['tool_uses'] = all_tool_uses
    if all_tool_results:     # NEW
        response_obj['tool_results'] = all_tool_results
    if all_cards:            # NEW
        response_obj['cards'] = all_cards
```

---

## Testing Checklist

After implementing fixes:

- [ ] ChatGPT tool messages appear in output
- [ ] ChatGPT per-message models captured
- [ ] ChatGPT citations/search results in sources[]
- [ ] Claude thinking blocks in response.thinking[]
- [ ] Claude tool_use/tool_result in response.tool_uses[]
- [ ] Claude citations in response.sources[]
- [ ] Grok web_search_results in response.sources[]
- [ ] Grok card_attachments in response.cards[]
- [ ] Exchange grouping preserves all new fields
- [ ] Existing fields unchanged (backward compatible)

---

## Validation Script

```python
#!/usr/bin/env python3
"""Validate parser fixes - check new fields are populated."""

import json
from pathlib import Path

def validate_parsed_output(filepath: str):
    with open(filepath) as f:
        conv = json.load(f)

    issues = []
    stats = {
        "tool_messages": 0,
        "per_message_models": 0,
        "citations": 0,
        "thinking_blocks": 0,
        "tool_uses": 0,
        "web_sources": 0,
        "cards": 0,
    }

    for ex in conv.get('exchanges', []):
        for resp in ex.get('responses', []):
            # Check new fields
            if resp.get('tools'):
                stats['tool_messages'] += len(resp['tools'])
            if resp.get('sources'):
                stats['citations'] += len(resp['sources'])
                stats['web_sources'] += len([s for s in resp['sources'] if s.get('type') == 'web_search'])
            if resp.get('thinking'):
                stats['thinking_blocks'] += len(resp['thinking'])
            if resp.get('tool_uses'):
                stats['tool_uses'] += len(resp['tool_uses'])
            if resp.get('cards'):
                stats['cards'] += len(resp['cards'])

    print(f"Conversation: {conv.get('title', 'Untitled')}")
    print(f"  Tool messages: {stats['tool_messages']}")
    print(f"  Citations: {stats['citations']}")
    print(f"  Web sources: {stats['web_sources']}")
    print(f"  Thinking blocks: {stats['thinking_blocks']}")
    print(f"  Tool uses: {stats['tool_uses']}")
    print(f"  Cards: {stats['cards']}")
    print()

if __name__ == '__main__':
    import sys
    validate_parsed_output(sys.argv[1])
```

---

## Rollout Plan

1. **Implement fixes** in `parse_raw_exports.py`
2. **Test on 10 conversations** (2 per platform type)
3. **Validate output** with script above
4. **Compare before/after** (ensure existing fields unchanged)
5. **Full reprocessing** (all raw exports)
6. **Measure improvement** (query recall tests)

---

**Estimated Time**: 8-16 hours for implementation + testing
**Risk**: Low (additive changes, backward compatible)
**Impact**: 40-60% more data preserved
