#!/usr/bin/env python3
"""
ISMA Raw Export Parser - All Platforms

Parses raw platform exports (bulk and individual) into standardized exchange JSON.
Each platform has a dedicated parser. Output is the normalized format consumed
by process_transcripts.py.

Supported formats:
  1. ChatGPT bulk export (conversations.json with tree-based mapping)
  2. Claude Chat bulk export (conversations.json with chat_messages array)
  3. Grok bulk export (responses[] or conversation.messages[])
  4. Gemini individual exports (simple messages[])
  5. Individual taeys-hands exports (already {title, url, timestamp, messages[]})
  6. Claude Code JSONL transcripts

Output schema per conversation:
{
  "platform": "chatgpt|claude_chat|gemini|grok|perplexity|claude_code",
  "conversation_id": "uuid or generated hash",
  "title": "conversation title",
  "model": "model slug if available",
  "source_file": "original export filename",
  "created_at": "ISO 8601 timestamp",
  "updated_at": "ISO 8601 timestamp",
  "exchanges": [
    {
      "index": 0,
      "timestamp": "ISO 8601",
      "user_prompt": "full user text",
      "responses": [
        {
          "text": "full assistant text",
          "tools": [{"type": "tool_name", "input": {...}, "output": "..."}],
          "artifacts": [{"type": "code", "language": "python", "content": "..."}]
        }
      ],
      "attachments": [
        {"name": "file.md", "size": 1234, "mime_type": "text/markdown",
         "extracted_content": "..." }
      ],
      "file_references": ["file.py", "/path/to/code.ts"]
    }
  ],
  "metadata": {
    "total_exchanges": 25,
    "total_attachments": 3,
    "total_file_references": 12,
    "parser": "parse_raw_exports.py",
    "parsed_at": "ISO 8601"
  }
}

Usage:
    python3 parse_raw_exports.py --input /path/to/export --output /path/to/output/
    python3 parse_raw_exports.py --scan /home/spark/data/transcripts/raw_exports/
    python3 parse_raw_exports.py --check  # Preview only
"""

import json
import hashlib
import re
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


# =============================================================================
# FILE REFERENCE EXTRACTION
# =============================================================================

# Patterns to find file references in message text
FILE_REF_PATTERNS = [
    # Explicit file paths
    re.compile(r'(?:^|\s)((?:/[\w./-]+)+\.(?:py|js|ts|md|json|yaml|yml|sh|txt|css|html))\b'),
    # File operations: "Reading file.py", "Looking at code.ts"
    re.compile(r'(?:Reading|Looking at|Checking|Opening|Examining|Reviewing|Editing|Creating|Writing)\s+([\w./-]+\.(?:py|js|ts|md|json|yaml|yml|sh|txt))', re.IGNORECASE),
    # Backtick file refs: `file.py`
    re.compile(r'`([\w./-]+\.(?:py|js|ts|md|json|yaml|yml|sh|txt))`'),
]


def extract_file_references(text: str) -> List[str]:
    """Extract file path/name references from message text."""
    refs = set()
    for pattern in FILE_REF_PATTERNS:
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if ref and len(ref) > 2 and not ref.startswith('http'):
                refs.add(ref)
    return sorted(refs)


# =============================================================================
# ARTIFACT EXTRACTION
# =============================================================================

CODE_FENCE_RE = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)


def extract_artifacts(text: str) -> List[Dict]:
    """Extract code blocks and artifacts from response text."""
    artifacts = []
    for match in CODE_FENCE_RE.finditer(text):
        lang = match.group(1) or 'unknown'
        code = match.group(2).strip()
        if len(code) > 20:  # Skip tiny fragments
            artifacts.append({
                "type": "code",
                "language": lang,
                "content": code,
                "fingerprint": hashlib.md5(code.encode()).hexdigest()[:16],
            })
    return artifacts


# =============================================================================
# TIMESTAMP HELPERS
# =============================================================================

def unix_to_iso(ts) -> str:
    """Convert UNIX timestamp (float or int) to ISO 8601."""
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def ensure_iso(ts) -> str:
    """Ensure timestamp is ISO 8601 string."""
    if not ts:
        return ""
    if isinstance(ts, (int, float)):
        return unix_to_iso(ts)
    return str(ts)


def make_id(platform: str, *parts) -> str:
    """Deterministic conversation ID from parts."""
    content = f"{platform}|{'|'.join(str(p) for p in parts)}"
    return hashlib.sha256(content.encode()).hexdigest()[:24]


# =============================================================================
# CHATGPT PARSER - Tree-based mapping structure
# =============================================================================

def parse_chatgpt_bulk(filepath: str) -> List[Dict]:
    """Parse ChatGPT bulk export conversations.json."""
    with open(filepath) as f:
        conversations = json.load(f)

    results = []
    for conv in conversations:
        title = conv.get('title', 'Untitled')
        conv_id = conv.get('conversation_id', make_id('chatgpt', title))
        model = conv.get('default_model_slug', '')
        created = unix_to_iso(conv.get('create_time'))
        updated = unix_to_iso(conv.get('update_time'))
        mapping = conv.get('mapping', {})

        # BFS traversal of tree to get messages in order
        messages = _chatgpt_tree_to_messages(mapping)

        # Group into exchanges (user prompt -> assistant responses)
        exchanges = _group_into_exchanges(messages, 'chatgpt')

        all_attachments = sum(len(ex.get('attachments', [])) for ex in exchanges)
        all_refs = sum(len(ex.get('file_references', [])) for ex in exchanges)

        results.append({
            "platform": "chatgpt",
            "conversation_id": conv_id,
            "title": title,
            "model": model,
            "source_file": Path(filepath).name,
            "created_at": created,
            "updated_at": updated,
            "exchanges": exchanges,
            "metadata": {
                "total_exchanges": len(exchanges),
                "total_attachments": all_attachments,
                "total_file_references": all_refs,
                "parser": "parse_raw_exports.py::chatgpt_bulk",
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return results


def _chatgpt_tree_to_messages(mapping: Dict) -> List[Dict]:
    """BFS traversal of ChatGPT mapping tree to ordered messages."""
    # Find root node (no parent or parent not in mapping)
    root_id = None
    for nid, node in mapping.items():
        parent = node.get('parent')
        if parent is None or parent not in mapping:
            root_id = nid
            break

    if not root_id:
        return []

    # BFS from root, following children
    ordered = []
    queue = [root_id]
    visited = set()

    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)

        node = mapping.get(nid, {})
        msg = node.get('message')

        if msg:
            author = msg.get('author', {})
            role = author.get('role', '')
            content = msg.get('content', {})
            parts = content.get('parts', [])
            meta = msg.get('metadata', {})
            create_time = msg.get('create_time')

            # Extract text from parts (can be strings or dicts)
            text_parts = []
            for part in parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    t = part.get('text', '')
                    if t:
                        text_parts.append(t)
            text = '\n'.join(text_parts).strip()

            # Extract attachments from metadata
            attachments = []
            for att in meta.get('attachments', []):
                attachments.append({
                    "name": att.get('name', ''),
                    "size": att.get('size', 0),
                    "mime_type": att.get('mime_type', ''),
                })

            if text or attachments:  # Skip empty nodes
                ordered.append({
                    "role": role,
                    "text": text,
                    "timestamp": unix_to_iso(create_time),
                    "attachments": attachments,
                })

        # Add children to queue
        children = node.get('children', [])
        queue.extend(children)

    return ordered


# =============================================================================
# CLAUDE CHAT PARSER - chat_messages array
# =============================================================================

def parse_claude_bulk(filepath: str) -> List[Dict]:
    """Parse Claude Chat bulk export conversations.json."""
    with open(filepath) as f:
        conversations = json.load(f)

    results = []
    for conv in conversations:
        conv_id = conv.get('uuid', make_id('claude_chat', conv.get('name', '')))
        title = conv.get('name', 'Untitled')
        created = ensure_iso(conv.get('created_at', ''))
        updated = ensure_iso(conv.get('updated_at', ''))

        chat_messages = conv.get('chat_messages', [])

        # Convert to our message format
        messages = []
        for msg in chat_messages:
            sender = msg.get('sender', '')
            # Map sender to role
            if sender == 'human':
                role = 'user'
            elif sender == 'assistant':
                role = 'assistant'
            else:
                role = sender

            # Extract text from content array
            content_blocks = msg.get('content', [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict):
                    t = block.get('text', '')
                    if t:
                        text_parts.append(t)
                elif isinstance(block, str):
                    text_parts.append(block)
            text = '\n'.join(text_parts).strip()

            # If content array is empty, fall back to text field
            if not text:
                text = msg.get('text', '').strip()

            # Extract attachments
            attachments = []
            for att in msg.get('attachments', []):
                attachments.append({
                    "name": att.get('file_name', ''),
                    "size": att.get('file_size', 0),
                    "mime_type": att.get('file_type', ''),
                    "extracted_content": att.get('extracted_content', ''),
                })
            # Also check files[] array
            for f_info in msg.get('files', []):
                if isinstance(f_info, dict):
                    name = f_info.get('file_name', '')
                    if name and not any(a['name'] == name for a in attachments):
                        attachments.append({"name": name, "size": 0, "mime_type": ""})

            timestamp = ensure_iso(msg.get('created_at', ''))

            if text or attachments:
                messages.append({
                    "role": role,
                    "text": text,
                    "timestamp": timestamp,
                    "attachments": attachments,
                })

        exchanges = _group_into_exchanges(messages, 'claude_chat')

        all_attachments = sum(len(ex.get('attachments', [])) for ex in exchanges)
        all_refs = sum(len(ex.get('file_references', [])) for ex in exchanges)

        results.append({
            "platform": "claude_chat",
            "conversation_id": conv_id,
            "title": title,
            "model": "",
            "source_file": Path(filepath).name,
            "created_at": created,
            "updated_at": updated,
            "exchanges": exchanges,
            "metadata": {
                "total_exchanges": len(exchanges),
                "total_attachments": all_attachments,
                "total_file_references": all_refs,
                "parser": "parse_raw_exports.py::claude_bulk",
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return results


# =============================================================================
# GROK PARSER - Two formats: responses[] and conversation.messages[]
# =============================================================================

def parse_grok_bulk(filepath: str) -> List[Dict]:
    """Parse Grok bulk export (prod-grok-backend.json or similar)."""
    with open(filepath) as f:
        data = json.load(f)

    # Grok can be a single conversation or array
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        conversations = [data]
    else:
        return []

    results = []
    for conv_data in conversations:
        # Format A: responses[] with conversation metadata
        if 'responses' in conv_data:
            result = _parse_grok_format_a(conv_data, filepath)
            if result:
                results.append(result)
        # Format B: conversation.messages[]
        elif 'conversation' in conv_data:
            inner = conv_data['conversation']
            if 'messages' in inner:
                result = _parse_grok_format_b(conv_data, filepath)
                if result:
                    results.append(result)

    return results


def _parse_grok_format_a(data: Dict, filepath: str) -> Optional[Dict]:
    """Grok Format A: responses[] array."""
    conv_meta = data.get('conversation', {})
    conv_id = conv_meta.get('conversation_id', make_id('grok', filepath))
    title = conv_meta.get('title', 'Untitled')
    created = unix_to_iso(conv_meta.get('create_time'))
    updated = unix_to_iso(conv_meta.get('modify_time'))

    messages = []
    for resp_wrapper in data.get('responses', []):
        resp = resp_wrapper.get('response', resp_wrapper)
        text = resp.get('message', '')
        sender = resp.get('sender', '')
        ts = unix_to_iso(resp.get('create_time'))

        role = 'user' if sender.lower() == 'human' else 'assistant'

        # Handle sub-messages (Grok delimiter)
        sub_msgs = text.split('\n\n---\n\n') if '\n\n---\n\n' in text else [text]
        for sub in sub_msgs:
            sub = sub.strip()
            if sub:
                messages.append({
                    "role": role,
                    "text": sub,
                    "timestamp": ts,
                    "attachments": [],
                })

    exchanges = _group_into_exchanges(messages, 'grok')
    return _build_result('grok', conv_id, title, '', filepath, created, updated, exchanges)


def _parse_grok_format_b(data: Dict, filepath: str) -> Optional[Dict]:
    """Grok Format B: conversation.messages[] array."""
    inner = data.get('conversation', data)
    title = inner.get('title', 'Untitled')
    conv_id = inner.get('id', make_id('grok', title, filepath))
    msgs = inner.get('messages', [])

    messages = []
    for msg in msgs:
        role = msg.get('role', 'user')
        text = msg.get('content', '')
        ts = ensure_iso(msg.get('timestamp', ''))
        messages.append({
            "role": role,
            "text": text.strip(),
            "timestamp": ts,
            "attachments": [],
        })

    exchanges = _group_into_exchanges(messages, 'grok')
    return _build_result('grok', conv_id, title, '', filepath, '', '', exchanges)


# =============================================================================
# INDIVIDUAL EXPORT PARSER - {title, url, timestamp, messages[]}
# =============================================================================

def parse_individual_export(filepath: str) -> List[Dict]:
    """Parse individual taeys-hands export (any platform)."""
    with open(filepath) as f:
        data = json.load(f)

    title = data.get('title', 'Untitled')
    url = data.get('url', '')
    timestamp = ensure_iso(data.get('timestamp', ''))
    msgs = data.get('messages', [])

    # Detect platform from filename prefix or platform field
    fname = Path(filepath).name
    platform = data.get('platform', '')
    if not platform:
        if fname.startswith('ChatGPT-'):
            platform = 'chatgpt'
        elif fname.startswith('Claude-'):
            platform = 'claude_chat'
        elif fname.startswith('Gemini-'):
            platform = 'gemini'
        elif fname.startswith('Grok-'):
            platform = 'grok'
        elif fname.startswith('Perplexity-'):
            platform = 'perplexity'
        else:
            platform = 'unknown'

    # Generate conversation ID from URL or title
    conv_id = make_id(platform, url or title)

    messages = []
    for msg in msgs:
        role = msg.get('role', 'user')
        # Gemini has html/markdown fields
        text = msg.get('content', '') or msg.get('markdown', '') or msg.get('text', '')
        ts = ensure_iso(msg.get('timestamp', ''))
        messages.append({
            "role": role,
            "text": text.strip() if text else '',
            "timestamp": ts,
            "attachments": [],
        })

    exchanges = _group_into_exchanges(messages, platform)
    return [_build_result(platform, conv_id, title, '', filepath, timestamp, timestamp, exchanges)]


# =============================================================================
# GEMINI PARSER - Simple messages[] with optional HTML
# =============================================================================

def parse_gemini_individual(filepath: str) -> List[Dict]:
    """Parse Gemini export - just delegates to individual parser."""
    return parse_individual_export(filepath)


# =============================================================================
# CLAUDE CODE JSONL PARSER
# =============================================================================

def parse_claude_code_jsonl(filepath: str) -> List[Dict]:
    """Parse Claude Code JSONL transcript."""
    lines = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not lines:
        return []

    session_id = ''
    messages = []

    for entry in lines:
        etype = entry.get('type', '')
        is_meta = entry.get('isMeta', False)
        msg = entry.get('message', {})
        ts = ensure_iso(entry.get('timestamp', ''))

        if not session_id:
            session_id = entry.get('sessionId', '')

        if etype == 'user' and not is_meta:
            content = msg.get('content', '')
            # Skip tool results
            if isinstance(content, list):
                # Check if it's a tool_result
                if any(isinstance(c, dict) and c.get('type') == 'tool_result' for c in content):
                    continue
                text_parts = []
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict) and c.get('type') == 'text':
                        text_parts.append(c.get('text', ''))
                content = '\n'.join(text_parts)
            elif isinstance(content, dict):
                if content.get('type') == 'tool_result':
                    continue
                content = content.get('text', str(content))

            if content and content.strip():
                messages.append({
                    "role": "user",
                    "text": content.strip(),
                    "timestamp": ts,
                    "attachments": [],
                })

        elif etype == 'assistant':
            content = msg.get('content', '')
            tools = []
            text_parts = []

            if isinstance(content, list):
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict):
                        ctype = c.get('type', '')
                        if ctype == 'text':
                            text_parts.append(c.get('text', ''))
                        elif ctype == 'tool_use':
                            tools.append({
                                "type": c.get('name', 'unknown'),
                                "input": c.get('input', {}),
                            })
            elif isinstance(content, str):
                text_parts.append(content)

            text = '\n'.join(text_parts).strip()
            if text or tools:
                messages.append({
                    "role": "assistant",
                    "text": text,
                    "timestamp": ts,
                    "attachments": [],
                    "_tools": tools,
                })

    conv_id = session_id or make_id('claude_code', filepath)
    title = Path(filepath).stem

    exchanges = _group_into_exchanges_with_tools(messages, 'claude_code')
    return [_build_result('claude_code', conv_id, title, '', filepath, '', '', exchanges)]


# =============================================================================
# EXCHANGE GROUPING - Core logic
# =============================================================================

def _group_into_exchanges(messages: List[Dict], platform: str) -> List[Dict]:
    """Group sequential messages into user->assistant exchanges."""
    exchanges = []
    current_user = None
    current_responses = []
    current_attachments = []
    current_timestamp = ''
    idx = 0

    for msg in messages:
        role = msg.get('role', '')

        if role == 'user':
            # Save previous exchange
            if current_user is not None and current_responses:
                resp_text = '\n'.join(r['text'] for r in current_responses if r.get('text'))
                file_refs = extract_file_references(current_user + '\n' + resp_text)
                exchanges.append({
                    "index": idx,
                    "timestamp": current_timestamp,
                    "user_prompt": current_user,
                    "responses": [{
                        "text": resp_text,
                        "tools": [],
                        "artifacts": extract_artifacts(resp_text),
                    }],
                    "attachments": current_attachments,
                    "file_references": file_refs,
                })
                idx += 1

            current_user = msg.get('text', '')
            current_responses = []
            current_attachments = msg.get('attachments', [])
            current_timestamp = msg.get('timestamp', '')

        elif role == 'assistant':
            current_responses.append(msg)

        elif role == 'system':
            pass  # Skip system messages

    # Save last exchange
    if current_user is not None and current_responses:
        resp_text = '\n'.join(r['text'] for r in current_responses if r.get('text'))
        file_refs = extract_file_references(current_user + '\n' + resp_text)
        exchanges.append({
            "index": idx,
            "timestamp": current_timestamp,
            "user_prompt": current_user,
            "responses": [{
                "text": resp_text,
                "tools": [],
                "artifacts": extract_artifacts(resp_text),
            }],
            "attachments": current_attachments,
            "file_references": file_refs,
        })

    return exchanges


def _group_into_exchanges_with_tools(messages: List[Dict], platform: str) -> List[Dict]:
    """Group messages with tool extraction (for Claude Code)."""
    exchanges = []
    current_user = None
    current_responses = []
    current_attachments = []
    current_timestamp = ''
    idx = 0

    for msg in messages:
        role = msg.get('role', '')

        if role == 'user':
            if current_user is not None and current_responses:
                resp_texts = []
                all_tools = []
                for r in current_responses:
                    if r.get('text'):
                        resp_texts.append(r['text'])
                    all_tools.extend(r.get('_tools', []))

                resp_text = '\n'.join(resp_texts)
                file_refs = extract_file_references(current_user + '\n' + resp_text)

                # Extract file refs from tool inputs too
                for tool in all_tools:
                    inp = tool.get('input', {})
                    fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
                    if fp:
                        refs = extract_file_references(fp)
                        file_refs.extend(refs)
                file_refs = sorted(set(file_refs))

                exchanges.append({
                    "index": idx,
                    "timestamp": current_timestamp,
                    "user_prompt": current_user,
                    "responses": [{
                        "text": resp_text,
                        "tools": all_tools,
                        "artifacts": extract_artifacts(resp_text),
                    }],
                    "attachments": current_attachments,
                    "file_references": file_refs,
                })
                idx += 1

            current_user = msg.get('text', '')
            current_responses = []
            current_attachments = msg.get('attachments', [])
            current_timestamp = msg.get('timestamp', '')

        elif role == 'assistant':
            current_responses.append(msg)

    # Save last exchange
    if current_user is not None and current_responses:
        resp_texts = []
        all_tools = []
        for r in current_responses:
            if r.get('text'):
                resp_texts.append(r['text'])
            all_tools.extend(r.get('_tools', []))
        resp_text = '\n'.join(resp_texts)
        file_refs = extract_file_references(current_user + '\n' + resp_text)
        for tool in all_tools:
            inp = tool.get('input', {})
            fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
            if fp:
                file_refs.extend(extract_file_references(fp))
        file_refs = sorted(set(file_refs))

        exchanges.append({
            "index": idx,
            "timestamp": current_timestamp,
            "user_prompt": current_user,
            "responses": [{
                "text": resp_text,
                "tools": all_tools,
                "artifacts": extract_artifacts(resp_text),
            }],
            "attachments": current_attachments,
            "file_references": file_refs,
        })

    return exchanges


# =============================================================================
# HELPERS
# =============================================================================

def _build_result(platform, conv_id, title, model, filepath, created, updated, exchanges):
    all_attachments = sum(len(ex.get('attachments', [])) for ex in exchanges)
    all_refs = sum(len(ex.get('file_references', [])) for ex in exchanges)
    return {
        "platform": platform,
        "conversation_id": conv_id,
        "title": title,
        "model": model,
        "source_file": Path(filepath).name,
        "created_at": created,
        "updated_at": updated,
        "exchanges": exchanges,
        "metadata": {
            "total_exchanges": len(exchanges),
            "total_attachments": all_attachments,
            "total_file_references": all_refs,
            "parser": f"parse_raw_exports.py::{platform}",
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        }
    }


# =============================================================================
# AUTO-DETECT AND PARSE
# =============================================================================

def detect_and_parse(filepath: str) -> List[Dict]:
    """Auto-detect format and parse."""
    fp = Path(filepath)
    fname = fp.name.lower()

    # JSONL = Claude Code
    if fp.suffix == '.jsonl':
        return parse_claude_code_jsonl(filepath)

    # Must be JSON
    if fp.suffix != '.json':
        return []

    # Peek at structure
    with open(filepath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"  SKIP (invalid JSON): {filepath}", flush=True)
            return []

    # Array of conversations
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict):
            # ChatGPT: has 'mapping' key
            if 'mapping' in first:
                return parse_chatgpt_bulk(filepath)
            # Claude: has 'chat_messages' key
            if 'chat_messages' in first:
                return parse_claude_bulk(filepath)
            # Grok array
            if 'responses' in first or 'conversation' in first:
                return parse_grok_bulk(filepath)

    # Single conversation object
    if isinstance(data, dict):
        # Individual export: has 'messages' and 'title'
        if 'messages' in data and 'title' in data:
            return parse_individual_export(filepath)
        # Grok single conversation
        if 'responses' in data or 'conversation' in data:
            return parse_grok_bulk(filepath)

    print(f"  SKIP (unknown format): {filepath}", flush=True)
    return []


# =============================================================================
# MAIN
# =============================================================================

def scan_directory(input_dir: str, output_dir: str, check_only: bool = False):
    """Scan directory for exports and parse them all."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not check_only:
        output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON/JSONL files
    files = sorted(input_path.rglob('*.json')) + sorted(input_path.rglob('*.jsonl'))

    print(f"Found {len(files)} files in {input_dir}", flush=True)

    total_convos = 0
    total_exchanges = 0
    total_attachments = 0
    total_refs = 0
    by_platform = defaultdict(int)

    for fp in files:
        conversations = detect_and_parse(str(fp))
        if not conversations:
            continue

        for conv in conversations:
            platform = conv['platform']
            n_ex = conv['metadata']['total_exchanges']
            n_att = conv['metadata']['total_attachments']
            n_ref = conv['metadata']['total_file_references']

            total_convos += 1
            total_exchanges += n_ex
            total_attachments += n_att
            total_refs += n_ref
            by_platform[platform] += 1

            if check_only:
                print(f"  [{platform}] {conv['title'][:60]}: "
                      f"{n_ex} exchanges, {n_att} attachments, {n_ref} file refs",
                      flush=True)
            else:
                # Write each conversation as separate JSON file
                safe_id = conv['conversation_id'][:16]
                out_file = output_path / platform / f"{safe_id}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w') as f:
                    json.dump(conv, f, indent=2, ensure_ascii=False)

    print(f"\n{'CHECK' if check_only else 'PARSED'}:", flush=True)
    print(f"  Conversations: {total_convos}", flush=True)
    print(f"  Exchanges: {total_exchanges}", flush=True)
    print(f"  Attachments: {total_attachments}", flush=True)
    print(f"  File references: {total_refs}", flush=True)
    print(f"  By platform:", flush=True)
    for p, c in sorted(by_platform.items()):
        print(f"    {p}: {c}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ISMA Raw Export Parser")
    parser.add_argument("--input", help="Single file to parse")
    parser.add_argument("--scan", help="Directory to scan recursively")
    parser.add_argument("--output", default="/home/spark/data/transcripts/parsed",
                        help="Output directory")
    parser.add_argument("--check", action="store_true", help="Preview only")
    args = parser.parse_args()

    if args.input:
        results = detect_and_parse(args.input)
        if args.check:
            for r in results:
                print(json.dumps({
                    "platform": r["platform"],
                    "title": r["title"],
                    "exchanges": r["metadata"]["total_exchanges"],
                    "attachments": r["metadata"]["total_attachments"],
                    "file_refs": r["metadata"]["total_file_references"],
                }, indent=2))
        else:
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            for r in results:
                safe_id = r['conversation_id'][:16]
                out_file = out / r['platform'] / f"{safe_id}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
                print(f"Wrote {out_file}")

    elif args.scan:
        scan_directory(args.scan, args.output, check_only=args.check)
    else:
        parser.print_help()
