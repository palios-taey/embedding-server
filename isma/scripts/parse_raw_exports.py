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


def mongo_timestamp_to_iso(ts_obj) -> str:
    """Convert MongoDB timestamp format to ISO 8601.

    Format: {"$date": {"$numberLong": "1762748309259"}}
    The numberLong is milliseconds since epoch.
    """
    if not ts_obj:
        return ""
    if isinstance(ts_obj, dict):
        if '$date' in ts_obj:
            date_obj = ts_obj['$date']
            if isinstance(date_obj, dict) and '$numberLong' in date_obj:
                # Convert milliseconds to seconds
                ms = int(date_obj['$numberLong'])
                return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    # Fallback to regular ISO or unix timestamp
    return ensure_iso(ts_obj)


def ensure_iso(ts) -> str:
    """Ensure timestamp is ISO 8601 string."""
    if not ts:
        return ""
    # Handle MongoDB format
    if isinstance(ts, dict) and '$date' in ts:
        return mongo_timestamp_to_iso(ts)
    # Handle unix timestamp
    if isinstance(ts, (int, float)):
        return unix_to_iso(ts)
    # Already ISO string
    return str(ts)


def make_id(platform: str, *parts) -> str:
    """Deterministic conversation ID from parts."""
    content = f"{platform}|{'|'.join(str(p) for p in parts)}"
    return hashlib.sha256(content.encode()).hexdigest()[:24]


# =============================================================================
# CHATGPT PARSER - Tree-based mapping structure
# =============================================================================

def parse_chatgpt_bulk(filepath: str) -> List[Dict]:
    """Parse ChatGPT bulk export conversations.json with full artifact support.

    Captures:
    - Canvas artifacts (canmore.create_textdoc / canmore.update_textdoc)
    - Code interpreter (role="tool", python recipient)
    - DALL-E results (role="tool" with image asset pointers)
    - Per-message model slugs
    - Dict parts (image/file references)
    - Citations from metadata
    """
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

        # Group into exchanges with artifact pairing
        exchanges = _group_chatgpt_exchanges_with_artifacts(messages)

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
    """BFS traversal of ChatGPT mapping tree to ordered messages.

    Extracts ALL message types:
    - role="user" (user prompts)
    - role="assistant" (ChatGPT responses)
    - role="tool" (code interpreter output, DALL-E results, browser results)
    - recipient="python" (code interpreter input)
    - recipient="canmore.create_textdoc" (canvas creation)
    - recipient="canmore.update_textdoc" (canvas updates)

    Preserves:
    - Per-message model_slug
    - Dict parts (image pointers, file references)
    - Citations from metadata
    - Full content parts array for downstream processing
    """
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
            recipient = msg.get('recipient', 'all')  # Canvas/code interpreter target

            # Per-message model (overrides conversation default)
            model_slug = meta.get('model_slug', '')

            # Extract text from parts (can be strings or dicts)
            text_parts = []
            dict_parts = []  # Store dict parts for downstream artifact processing

            for part in parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    # Store dict part for artifact processing
                    dict_parts.append(part)

                    # Try to extract text if available
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

            # Extract citations
            citations = meta.get('citations', [])

            # Build message object
            # Include even if text is empty - role="tool" messages often have no text
            # but contain crucial output in metadata
            msg_obj = {
                "role": role,
                "text": text,
                "timestamp": unix_to_iso(create_time),
                "attachments": attachments,
                "model": model_slug,
                "recipient": recipient,
                "parts": parts,  # Preserve original parts for artifact extraction
                "dict_parts": dict_parts,
                "citations": citations,
                "metadata": meta,  # Preserve full metadata for artifact processing
            }

            ordered.append(msg_obj)

        # Add children to queue
        children = node.get('children', [])
        queue.extend(children)

    return ordered


def _group_chatgpt_exchanges_with_artifacts(messages: List[Dict]) -> List[Dict]:
    """Group ChatGPT messages into exchanges with artifact extraction.

    Pairs messages to extract artifacts:
    1. assistant + canmore.create_textdoc → Canvas artifact (create)
    2. assistant + canmore.update_textdoc → Canvas artifact (update)
    3. assistant + python → tool + Code interpreter artifact
    4. tool messages → Code interpreter output, DALL-E results

    Exchange schema:
    {
      "index": 0,
      "timestamp": "ISO 8601",
      "model": "gpt-5-2-pro",  # Per-exchange model
      "user_prompt": "full user text",
      "responses": [{
        "text": "full assistant text",
        "model": "gpt-5-2-pro",  # Per-response model
        "artifacts": [...],
        "tools": [...],
        "citations": [...]
      }],
      "attachments": [...],
      "file_references": [...]
    }
    """
    exchanges = []
    idx = 0
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg.get('role', '')

        if role == 'user':
            # Start new exchange
            user_text = msg.get('text', '')
            user_ts = msg.get('timestamp', '')
            user_attachments = msg.get('attachments', [])
            user_model = msg.get('model', '')

            # Collect all assistant/tool responses until next user message
            responses = []
            i += 1

            while i < len(messages) and messages[i].get('role') != 'user':
                resp_msg = messages[i]
                resp_role = resp_msg.get('role', '')
                recipient = resp_msg.get('recipient', 'all')

                if resp_role == 'assistant':
                    # Build response object
                    response_obj = _extract_assistant_response(resp_msg)
                    responses.append(response_obj)

                elif resp_role == 'tool':
                    # Tool output - pair with previous assistant if it was code interpreter
                    if responses and recipient == 'all':
                        # This is output from previous tool call
                        tool_artifact = _extract_tool_artifact(resp_msg)
                        if tool_artifact:
                            # Add to last response's artifacts
                            responses[-1]['artifacts'].append(tool_artifact)
                    else:
                        # Orphaned tool message - create standalone response
                        response_obj = {
                            "text": resp_msg.get('text', ''),
                            "model": resp_msg.get('model', ''),
                            "artifacts": [_extract_tool_artifact(resp_msg)] if _extract_tool_artifact(resp_msg) else [],
                            "tools": [],
                            "citations": [],
                        }
                        responses.append(response_obj)

                i += 1

            # Build exchange
            if responses or user_text:
                # Merge all response texts
                merged_text = '\n\n'.join(r['text'] for r in responses if r.get('text'))

                # Merge all artifacts
                all_artifacts = []
                all_tools = []
                all_citations = []
                response_model = ''

                for r in responses:
                    all_artifacts.extend(r.get('artifacts', []))
                    all_tools.extend(r.get('tools', []))
                    all_citations.extend(r.get('citations', []))
                    if not response_model and r.get('model'):
                        response_model = r['model']

                # Extract file references
                file_refs = extract_file_references(user_text + '\n' + merged_text)

                # Add code fence artifacts
                code_artifacts = extract_artifacts(merged_text)
                all_artifacts.extend(code_artifacts)

                exchange = {
                    "index": idx,
                    "timestamp": user_ts,
                    "model": response_model or user_model,
                    "user_prompt": user_text,
                    "responses": [{
                        "text": merged_text,
                        "model": response_model,
                        "artifacts": all_artifacts,
                        "tools": all_tools,
                        "citations": all_citations,
                    }],
                    "attachments": user_attachments,
                    "file_references": file_refs,
                }

                exchanges.append(exchange)
                idx += 1
        else:
            # Orphaned assistant/tool at start - skip or handle
            i += 1

    return exchanges


def _extract_assistant_response(msg: Dict) -> Dict:
    """Extract response object from assistant message.

    Handles:
    - Canvas artifacts (canmore recipients)
    - Code interpreter (python recipient)
    - Regular text responses
    - Citations
    """
    text = msg.get('text', '')
    model = msg.get('model', '')
    recipient = msg.get('recipient', 'all')
    citations = msg.get('citations', [])
    parts = msg.get('parts', [])
    metadata = msg.get('metadata', {})

    artifacts = []
    tools = []

    # Canvas artifact detection
    if recipient == 'canmore.create_textdoc':
        # Initial canvas creation
        canvas_artifact = _extract_canvas_artifact(parts, 'create')
        if canvas_artifact:
            artifacts.append(canvas_artifact)

    elif recipient == 'canmore.update_textdoc':
        # Canvas update
        canvas_artifact = _extract_canvas_artifact(parts, 'update')
        if canvas_artifact:
            artifacts.append(canvas_artifact)

    elif recipient == 'python':
        # Code interpreter input
        # Extract code from parts or metadata
        code = ''
        if isinstance(text, str) and text.strip():
            code = text

        # Check metadata for aggregate_result
        agg = metadata.get('aggregate_result', {})
        if agg.get('code'):
            code = agg['code']

        if code:
            artifacts.append({
                "type": "code_interpreter",
                "code": code,
                "language": "python",
                "command": "execute",
            })

    return {
        "text": text,
        "model": model,
        "artifacts": artifacts,
        "tools": tools,
        "citations": citations,
    }


def _extract_canvas_artifact(parts: List, command: str) -> Optional[Dict]:
    """Extract canvas artifact from parts.

    Canvas format:
    parts[0] = JSON string: {"name": "doc_name", "type": "document"|"code/python", "content": "FULL CONTENT"}
    """
    if not parts:
        return None

    first_part = parts[0]

    # Canvas content is in a JSON string
    if isinstance(first_part, str):
        try:
            canvas_data = json.loads(first_part)
            if isinstance(canvas_data, dict) and 'content' in canvas_data:
                return {
                    "type": "canvas",
                    "name": canvas_data.get('name', 'untitled'),
                    "content_type": canvas_data.get('type', 'document'),
                    "content": canvas_data.get('content', ''),
                    "command": command,
                }
        except json.JSONDecodeError:
            pass

    return None


def _extract_tool_artifact(msg: Dict) -> Optional[Dict]:
    """Extract artifact from tool message.

    Tool messages contain:
    - Code interpreter output (text in parts)
    - DALL-E results (asset_pointer in dict parts)
    - Browser results
    """
    parts = msg.get('parts', [])
    metadata = msg.get('metadata', {})

    # Check for DALL-E asset pointer
    for part in parts:
        if isinstance(part, dict):
            content_type = part.get('content_type', '')
            if content_type == 'image_asset_pointer':
                return {
                    "type": "dalle",
                    "asset_pointer": part.get('asset_pointer', ''),
                    "metadata": part.get('metadata', {}),
                }

    # Check for code interpreter output
    text = msg.get('text', '')
    if text:
        # This is code execution output
        return {
            "type": "code_interpreter_output",
            "output": text,
        }

    # Check metadata for aggregate result
    agg = metadata.get('aggregate_result', {})
    if agg:
        messages = agg.get('messages', [])
        if messages:
            output_parts = []
            for m in messages:
                if isinstance(m, dict):
                    output_parts.append(m.get('text', str(m)))
                else:
                    output_parts.append(str(m))

            return {
                "type": "code_interpreter_output",
                "output": '\n'.join(output_parts),
            }

    return None


# =============================================================================
# CLAUDE CHAT PARSER - chat_messages array
# =============================================================================

def parse_claude_bulk(filepath: str) -> List[Dict]:
    """Parse Claude Chat bulk export conversations.json.

    Captures:
    - type=text blocks with inline citations
    - type=thinking blocks (extended thinking)
    - type=tool_use with name=artifacts (AI-generated canonical content)
    - type=tool_use/tool_result for other tools (web_search, create_file, bash_tool, etc.)
    - Attachments with extracted_content
    """
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

            # Extract all content types from content array
            content_blocks = msg.get('content', [])
            text_parts = []
            thinking_parts = []
            artifacts = []
            tools = []
            citations = []

            for block in content_blocks:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue

                if not isinstance(block, dict):
                    continue

                block_type = block.get('type', '')

                # type=text - main response text with optional citations
                if block_type == 'text':
                    t = block.get('text', '')
                    if t:
                        text_parts.append(t)
                    # Extract citations from text blocks
                    block_citations = block.get('citations', [])
                    if block_citations:
                        for cit in block_citations:
                            citations.append({
                                "url": cit.get('url', ''),
                                "title": cit.get('title', ''),
                                "start_index": cit.get('start_index', 0),
                                "end_index": cit.get('end_index', 0),
                                "origin_tool_name": cit.get('origin_tool_name', ''),
                            })

                # type=thinking - extended thinking blocks
                elif block_type == 'thinking':
                    thinking = block.get('thinking', '')
                    if thinking:
                        thinking_parts.append(thinking)
                    # Also preserve summaries if present
                    summaries = block.get('summaries', [])
                    if summaries:
                        thinking_parts.append('\n'.join(f"[Summary] {s}" for s in summaries))

                # type=tool_use with name=artifacts - CANONICAL AI-GENERATED CONTENT
                elif block_type == 'tool_use' and block.get('name') == 'artifacts':
                    tool_input = block.get('input', {})
                    artifacts.append({
                        "type": "generated",  # Distinguish from code_fence
                        "id": tool_input.get('id', ''),
                        "content_type": tool_input.get('type', ''),
                        "title": tool_input.get('title', ''),
                        "command": tool_input.get('command', ''),  # create/update/rewrite
                        "content": tool_input.get('content', ''),  # THE CANONICAL CONTENT
                        "language": tool_input.get('language'),
                        "citations": tool_input.get('md_citations', []),
                        "version_uuid": tool_input.get('version_uuid', ''),
                    })

                # type=tool_use - other tools (web_search, create_file, bash_tool, etc.)
                elif block_type == 'tool_use':
                    tool_name = block.get('name', 'unknown')
                    tool_input = block.get('input', {})
                    tools.append({
                        "type": tool_name,
                        "input": tool_input,
                        "tool_use_id": block.get('id', ''),
                    })

                # type=tool_result - tool outputs
                elif block_type == 'tool_result':
                    tool_use_id = block.get('tool_use_id', '')
                    # Find matching tool_use to attach output
                    for tool in tools:
                        if tool.get('tool_use_id') == tool_use_id:
                            # Content can be string or array of blocks
                            result_content = block.get('content', '')
                            if isinstance(result_content, list):
                                result_parts = []
                                for rc in result_content:
                                    if isinstance(rc, str):
                                        result_parts.append(rc)
                                    elif isinstance(rc, dict) and rc.get('type') == 'text':
                                        result_parts.append(rc.get('text', ''))
                                result_content = '\n'.join(result_parts)
                            tool['output'] = result_content
                            break

                # type=token_budget, type=flag - minor types, just note them
                elif block_type in ('token_budget', 'flag'):
                    pass  # Skip for now

            # Combine text parts
            text = '\n'.join(text_parts).strip()

            # If content array is empty, fall back to text field
            if not text:
                text = msg.get('text', '').strip()

            # Combine thinking parts
            thinking = '\n\n'.join(thinking_parts).strip() if thinking_parts else ''

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
                        attachments.append({
                            "name": name,
                            "size": 0,
                            "mime_type": "",
                            "extracted_content": "",
                        })

            timestamp = ensure_iso(msg.get('created_at', ''))

            # Store all extracted data in message
            if text or attachments or thinking or artifacts or tools:
                msg_obj = {
                    "role": role,
                    "text": text,
                    "timestamp": timestamp,
                    "attachments": attachments,
                }
                # Store metadata fields for later grouping
                if thinking:
                    msg_obj['_thinking'] = thinking
                if artifacts:
                    msg_obj['_artifacts'] = artifacts
                if tools:
                    msg_obj['_tools'] = tools
                if citations:
                    msg_obj['_citations'] = citations

                messages.append(msg_obj)

        # Group into exchanges with enhanced content extraction
        exchanges = _group_into_exchanges_claude_enhanced(messages, 'claude_chat')

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


def _group_into_exchanges_claude_enhanced(messages: List[Dict], platform: str) -> List[Dict]:
    """Group Claude messages into exchanges, preserving thinking/artifacts/tools/citations.

    This extends the standard grouping to handle:
    - _thinking: Extended thinking blocks
    - _artifacts: AI-generated canonical content from artifacts tool
    - _tools: Other tool uses (web_search, create_file, etc.)
    - _citations: Inline citations with URL/title/offsets

    Also preserves code_fence artifacts from regex extraction.
    """
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
                # Merge all assistant response data
                resp_texts = []
                all_thinking = []
                all_artifacts = []
                all_tools = []
                all_citations = []

                for r in current_responses:
                    if r.get('text'):
                        resp_texts.append(r['text'])
                    if r.get('_thinking'):
                        all_thinking.append(r['_thinking'])
                    if r.get('_artifacts'):
                        all_artifacts.extend(r['_artifacts'])
                    if r.get('_tools'):
                        all_tools.extend(r['_tools'])
                    if r.get('_citations'):
                        all_citations.extend(r['_citations'])

                resp_text = '\n'.join(resp_texts)
                thinking = '\n\n---\n\n'.join(all_thinking) if all_thinking else ''

                # Also extract code_fence artifacts from text (backwards compat)
                code_fence_artifacts = extract_artifacts(resp_text)

                # Combine generated artifacts and code_fence artifacts
                combined_artifacts = all_artifacts + code_fence_artifacts

                # Build response object
                response_obj = {
                    "text": resp_text,
                    "tools": all_tools,
                    "artifacts": combined_artifacts,
                }
                if thinking:
                    response_obj['thinking'] = thinking
                if all_citations:
                    response_obj['citations'] = all_citations

                # Extract file references
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
                    "responses": [response_obj],
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
        resp_texts = []
        all_thinking = []
        all_artifacts = []
        all_tools = []
        all_citations = []

        for r in current_responses:
            if r.get('text'):
                resp_texts.append(r['text'])
            if r.get('_thinking'):
                all_thinking.append(r['_thinking'])
            if r.get('_artifacts'):
                all_artifacts.extend(r['_artifacts'])
            if r.get('_tools'):
                all_tools.extend(r['_tools'])
            if r.get('_citations'):
                all_citations.extend(r['_citations'])

        resp_text = '\n'.join(resp_texts)
        thinking = '\n\n---\n\n'.join(all_thinking) if all_thinking else ''

        code_fence_artifacts = extract_artifacts(resp_text)
        combined_artifacts = all_artifacts + code_fence_artifacts

        response_obj = {
            "text": resp_text,
            "tools": all_tools,
            "artifacts": combined_artifacts,
        }
        if thinking:
            response_obj['thinking'] = thinking
        if all_citations:
            response_obj['citations'] = all_citations

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
            "responses": [response_obj],
            "attachments": current_attachments,
            "file_references": file_refs,
        })

    return exchanges


# =============================================================================
# GROK PARSER - Three formats:
#   A: responses[] with conversation metadata (conversations_old files)
#   B: conversation.messages[] array (alternative individual format)
#   C: {"conversations": [...]} master bulk export (prod-grok-backend.json)
#
# Enhanced to capture ALL data:
#   - Asset file content (read from prod-mc-asset-server/{uuid}/content)
#   - Steps tagged_text, web_search_results, rag_results, tool_usage_results
#   - card_attachments_json (X/Twitter citations)
#   - xpost_ids (X post references)
#   - web_search_results (top-level and cited)
#   - thinking timing (start/end timestamps)
#   - metadata.requestModelDetails
# =============================================================================


def parse_xai_grok_bulk(filepath: str) -> List[Dict]:
    """Parse xAI Grok bulk export (conversations.json with chat_messages + account).

    Format: [{uuid, name, summary, created_at, updated_at, account, chat_messages: [
        {uuid, text, content: [{type, text, citations}], sender, created_at, ...}
    ]}]
    """
    with open(filepath) as f:
        conversations = json.load(f)

    results = []
    for conv in conversations:
        conv_id = conv.get('uuid', make_id('grok', conv.get('name', '')))
        title = conv.get('name', 'Untitled')
        created = ensure_iso(conv.get('created_at', ''))
        updated = ensure_iso(conv.get('updated_at', ''))

        chat_messages = conv.get('chat_messages', [])
        if not chat_messages:
            continue

        # Build message list
        messages = []
        for msg in chat_messages:
            sender = msg.get('sender', '')
            if sender == 'human':
                role = 'user'
            elif sender == 'assistant':
                role = 'assistant'
            else:
                role = sender

            # xAI Grok has text at message level AND in content blocks
            # Prefer message-level text (complete), fall back to content blocks
            text = msg.get('text', '').strip()
            if not text:
                # Assemble from content blocks
                for block in msg.get('content', []):
                    if isinstance(block, dict) and block.get('type') == 'text':
                        t = block.get('text', '')
                        if t:
                            text += t

            # Extract citations from content blocks
            citations = []
            for block in msg.get('content', []):
                if isinstance(block, dict):
                    for cit in block.get('citations', []):
                        if isinstance(cit, dict) and cit.get('url'):
                            citations.append({
                                "url": cit.get('url', ''),
                                "title": cit.get('title', ''),
                            })

            # Extract code artifacts from text
            artifacts = extract_artifacts(text) if role == 'assistant' else []
            file_refs = extract_file_references(text)

            msg_obj = {
                'role': role,
                'text': text,
                'timestamp': ensure_iso(msg.get('created_at', '')),
            }
            if artifacts:
                msg_obj['_artifacts'] = artifacts
            if citations:
                msg_obj['_citations'] = citations

            messages.append(msg_obj)

        # Group into exchanges (standard user/assistant pairing)
        exchanges = []
        current_user = None
        current_responses = []
        current_timestamp = ''
        idx = 0

        for msg in messages:
            role = msg.get('role', '')
            if role == 'user':
                if current_user is not None and current_responses:
                    exchanges.append(_build_exchange(
                        idx, current_timestamp, current_user,
                        current_responses, 'grok'))
                    idx += 1
                current_user = msg
                current_responses = []
                current_timestamp = msg.get('timestamp', '')
            elif role == 'assistant':
                current_responses.append(msg)

        # Don't forget last exchange
        if current_user is not None and current_responses:
            exchanges.append(_build_exchange(
                idx, current_timestamp, current_user,
                current_responses, 'grok'))

        if not exchanges:
            continue

        all_refs = sum(len(ex.get('file_references', [])) for ex in exchanges)

        results.append({
            "platform": "grok",
            "conversation_id": conv_id,
            "title": title,
            "model": "",
            "source_file": Path(filepath).name,
            "created_at": created,
            "updated_at": updated,
            "exchanges": exchanges,
            "metadata": {
                "total_exchanges": len(exchanges),
                "total_attachments": 0,
                "total_file_references": all_refs,
                "parser": "parse_raw_exports.py::xai_grok_bulk",
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return results


def _build_exchange(idx, timestamp, user_msg, assistant_msgs, platform):
    """Build a standard exchange dict from user + assistant messages."""
    user_text = user_msg.get('text', '')
    file_refs = extract_file_references(user_text)

    responses = []
    for amsg in assistant_msgs:
        resp = {"text": amsg.get('text', '')}
        if amsg.get('_artifacts'):
            resp["artifacts"] = amsg['_artifacts']
        if amsg.get('_citations'):
            resp["citations"] = amsg['_citations']
        if amsg.get('_thinking'):
            resp["thinking"] = amsg['_thinking']
        if amsg.get('_tools'):
            resp["tools"] = amsg['_tools']
        responses.append(resp)
        file_refs.extend(extract_file_references(amsg.get('text', '')))

    return {
        "index": idx,
        "timestamp": timestamp,
        "user_prompt": user_text,
        "responses": responses,
        "attachments": [],
        "file_references": list(set(file_refs)),
    }


def parse_grok_bulk(filepath: str) -> List[Dict]:
    """Parse Grok bulk export (prod-grok-backend.json or individual files)."""
    with open(filepath) as f:
        data = json.load(f)

    # Determine asset directory (if it exists)
    asset_dir = None
    file_path = Path(filepath)
    # Check for prod-mc-asset-server in parent directories
    potential_asset_dir = file_path.parent / 'prod-mc-asset-server'
    if potential_asset_dir.exists() and potential_asset_dir.is_dir():
        asset_dir = potential_asset_dir

    # Format C: Master bulk export with {"conversations": [...]}
    if isinstance(data, dict) and 'conversations' in data:
        results = []
        for conv_wrapper in data['conversations']:
            result = _parse_grok_format_c(conv_wrapper, filepath, asset_dir)
            if result:
                results.append(result)
        return results

    # Legacy formats: single conversation or array
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
            result = _parse_grok_format_a(conv_data, filepath, asset_dir)
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


def _read_asset_content(asset_dir: Optional[Path], uuid: str) -> Optional[str]:
    """Read asset file content from prod-mc-asset-server/{uuid}/content."""
    if not asset_dir or not uuid:
        return None

    content_path = asset_dir / uuid / 'content'
    if not content_path.exists():
        return None

    try:
        with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return None


def _parse_card_attachments(cards_json_list: List[str]) -> List[Dict]:
    """Parse card_attachments_json (list of JSON strings) into structured data."""
    citations = []
    for card_str in cards_json_list:
        try:
            card = json.loads(card_str) if isinstance(card_str, str) else card_str
            if isinstance(card, dict):
                citations.append({
                    "id": card.get('id', ''),
                    "type": card.get('cardType', ''),
                    "url": card.get('url', ''),
                })
        except json.JSONDecodeError:
            continue
    return citations


def _calculate_thinking_duration(start_time, end_time) -> Optional[int]:
    """Calculate thinking duration in milliseconds from MongoDB timestamps."""
    if not start_time or not end_time:
        return None

    try:
        # Extract milliseconds from MongoDB format
        if isinstance(start_time, dict) and '$date' in start_time:
            start_ms = int(start_time['$date']['$numberLong'])
        elif isinstance(start_time, (int, float)):
            start_ms = int(start_time)
        else:
            return None

        if isinstance(end_time, dict) and '$date' in end_time:
            end_ms = int(end_time['$date']['$numberLong'])
        elif isinstance(end_time, (int, float)):
            end_ms = int(end_time)
        else:
            return None

        return end_ms - start_ms
    except (KeyError, ValueError, TypeError):
        return None


def _parse_grok_format_c(conv_wrapper: Dict, filepath: str, asset_dir: Optional[Path]) -> Optional[Dict]:
    """Grok Format C: Master bulk export - FULL extraction.

    Captures:
    - Asset file content (read from disk)
    - Steps: tagged_text, web_search_results, rag_results, tool_usage_results
    - card_attachments_json (parsed as citations)
    - xpost_ids
    - web_search_results (top-level and cited)
    - thinking timing
    - metadata.requestModelDetails
    """
    conv_meta = conv_wrapper.get('conversation', {})
    # Detect if this is Claude Chat data (Anthropic's prod-grok-backend.json)
    # Claude conversations have user_id, system_prompt_id, leaf_response_id
    is_claude = any(k in conv_meta for k in ('user_id', 'system_prompt_id', 'leaf_response_id'))
    platform = 'claude_chat' if is_claude else 'grok'

    conv_id = conv_meta.get('id') or conv_meta.get('conversation_id') or make_id(platform, filepath)
    title = conv_meta.get('title', 'Untitled')
    created = ensure_iso(conv_meta.get('create_time'))
    updated = ensure_iso(conv_meta.get('modify_time'))

    # Build message tree from responses
    responses_raw = conv_wrapper.get('responses', [])
    response_map = {}  # response_id -> response object

    for resp_wrapper in responses_raw:
        resp = resp_wrapper.get('response', resp_wrapper)
        resp_id = resp.get('_id') or resp.get('response_id')
        if resp_id:
            response_map[resp_id] = resp

    # Traverse tree
    messages = []
    processed = set()

    def add_message(resp: Dict):
        """Convert response object to message with FULL extraction."""
        resp_id = resp.get('_id') or resp.get('response_id')
        if resp_id in processed:
            return
        processed.add(resp_id)

        text = resp.get('message', '')
        sender = resp.get('sender', '')
        ts = ensure_iso(resp.get('create_time'))

        # Model: try response.model first, fall back to metadata.requestModelDetails
        model = resp.get('model', '')
        if not model:
            metadata = resp.get('metadata', {})
            if isinstance(metadata, dict):
                req_model = metadata.get('requestModelDetails', {})
                if isinstance(req_model, dict):
                    model = req_model.get('modelId', '')

        role = 'user' if sender.lower() == 'human' else 'assistant'

        # === ATTACHMENTS: Read actual asset content ===
        attachments = []
        file_att_uuids = resp.get('file_attachments', [])
        for uuid in file_att_uuids:
            content = _read_asset_content(asset_dir, uuid)
            att = {
                "type": "grok_asset",
                "uuid": uuid,
                "name": f"grok_asset_{uuid[:8]}",
            }
            if content:
                att["content"] = content
                att["size"] = len(content)
            else:
                att["size"] = 0
            att["mime_type"] = "application/octet-stream"
            attachments.append(att)

        # === THINKING ===
        thinking = resp.get('thinking_trace', '')
        thinking_start = resp.get('thinking_start_time')
        thinking_end = resp.get('thinking_end_time')
        thinking_duration = _calculate_thinking_duration(thinking_start, thinking_end)

        # === STEPS: Full extraction ===
        steps = resp.get('steps', [])
        tools = []
        for step in steps:
            if not isinstance(step, dict):
                continue

            # Extract tagged_text
            tagged_text = step.get('tagged_text', {})
            if tagged_text:
                tools.append({
                    "type": "reasoning_step",
                    "header": tagged_text.get('header', ''),
                    "summary": tagged_text.get('summary', ''),
                    "tagged_text": tagged_text,
                })

            # Extract web_search_results from step
            step_web = step.get('web_search_results', [])
            if step_web:
                tools.append({
                    "type": "step_web_search",
                    "results": step_web,
                })

            # Extract rag_results
            rag = step.get('rag_results', [])
            if rag:
                tools.append({
                    "type": "rag_results",
                    "results": rag,
                })

            # Extract tool_usage_results
            tool_usage = step.get('tool_usage_results', [])
            if tool_usage:
                tools.append({
                    "type": "tool_usage",
                    "results": tool_usage,
                })

        # === CITATIONS ===
        citations = []

        # card_attachments_json (X/Twitter citations)
        cards_json = resp.get('card_attachments_json', [])
        if cards_json:
            card_citations = _parse_card_attachments(cards_json)
            citations.extend(card_citations)

        # xpost_ids
        xpost_ids = resp.get('xpost_ids', [])
        x_references = xpost_ids if xpost_ids else []

        # === WEB SEARCH RESULTS ===
        web_search_results = resp.get('web_search_results', [])
        cited_web_search_results = resp.get('cited_web_search_results', [])

        # Build message metadata
        msg_metadata = {}
        if thinking:
            msg_metadata['thinking_trace'] = thinking
        if thinking_duration is not None:
            msg_metadata['thinking_duration_ms'] = thinking_duration
        if web_search_results:
            msg_metadata['web_search_results'] = web_search_results
        if cited_web_search_results:
            msg_metadata['cited_web_search_results'] = cited_web_search_results
        if x_references:
            msg_metadata['x_references'] = x_references
        if citations:
            msg_metadata['citations'] = citations

        msg = {
            "role": role,
            "text": text.strip() if text else '',
            "timestamp": ts,
            "attachments": attachments,
            "model": model,
            "_response_id": resp_id,
            "_parent_id": resp.get('parent_response_id'),
            "_metadata": msg_metadata,
            "_tools": tools,
        }

        messages.append(msg)

    # Process tree
    roots = []
    for resp in response_map.values():
        parent_id = resp.get('parent_response_id')
        if not parent_id or parent_id not in response_map:
            roots.append(resp)

    roots.sort(key=lambda r: ensure_iso(r.get('create_time')) or '')

    for root in roots:
        queue = [root]
        while queue:
            current = queue.pop(0)
            add_message(current)
            current_id = current.get('_id') or current.get('response_id')
            children = [r for r in response_map.values()
                       if r.get('parent_response_id') == current_id]
            children.sort(key=lambda r: ensure_iso(r.get('create_time')) or '')
            queue.extend(children)

    # Group into exchanges (preserves all metadata)
    exchanges = _group_into_exchanges_with_tools_and_metadata(messages, platform)

    # Detect model
    detected_model = ''
    for msg in messages:
        if msg.get('model'):
            detected_model = msg['model']
            break

    return _build_result(platform, conv_id, title, detected_model, filepath, created, updated, exchanges)


def _parse_grok_format_a(data: Dict, filepath: str, asset_dir: Optional[Path]) -> Optional[Dict]:
    """Grok Format A: responses[] array - FULL extraction."""
    conv_meta = data.get('conversation', {})
    conv_id = conv_meta.get('conversation_id') or conv_meta.get('id') or make_id('grok', filepath)
    title = conv_meta.get('title', 'Untitled')
    created = ensure_iso(conv_meta.get('create_time'))
    updated = ensure_iso(conv_meta.get('modify_time'))

    messages = []
    for resp_wrapper in data.get('responses', []):
        resp = resp_wrapper.get('response', resp_wrapper)
        text = resp.get('message', '')
        sender = resp.get('sender', '')
        ts = ensure_iso(resp.get('create_time'))

        # Model extraction
        model = resp.get('model', '')
        if not model:
            metadata = resp.get('metadata', {})
            if isinstance(metadata, dict):
                req_model = metadata.get('requestModelDetails', {})
                if isinstance(req_model, dict):
                    model = req_model.get('modelId', '')

        role = 'user' if sender.lower() == 'human' else 'assistant'

        # === ATTACHMENTS with content ===
        attachments = []
        for uuid in resp.get('file_attachments', []):
            content = _read_asset_content(asset_dir, uuid)
            att = {
                "type": "grok_asset",
                "uuid": uuid,
                "name": f"grok_asset_{uuid[:8]}",
            }
            if content:
                att["content"] = content
                att["size"] = len(content)
            else:
                att["size"] = 0
            att["mime_type"] = "application/octet-stream"
            attachments.append(att)

        # === THINKING ===
        thinking = resp.get('thinking_trace', '')
        thinking_start = resp.get('thinking_start_time')
        thinking_end = resp.get('thinking_end_time')
        thinking_duration = _calculate_thinking_duration(thinking_start, thinking_end)

        # === STEPS ===
        steps = resp.get('steps', [])
        tools = []
        for step in steps:
            if not isinstance(step, dict):
                continue

            tagged_text = step.get('tagged_text', {})
            if tagged_text:
                tools.append({
                    "type": "reasoning_step",
                    "header": tagged_text.get('header', ''),
                    "summary": tagged_text.get('summary', ''),
                    "tagged_text": tagged_text,
                })

            step_web = step.get('web_search_results', [])
            if step_web:
                tools.append({
                    "type": "step_web_search",
                    "results": step_web,
                })

            rag = step.get('rag_results', [])
            if rag:
                tools.append({
                    "type": "rag_results",
                    "results": rag,
                })

            tool_usage = step.get('tool_usage_results', [])
            if tool_usage:
                tools.append({
                    "type": "tool_usage",
                    "results": tool_usage,
                })

        # === CITATIONS ===
        citations = []
        cards_json = resp.get('card_attachments_json', [])
        if cards_json:
            citations.extend(_parse_card_attachments(cards_json))

        xpost_ids = resp.get('xpost_ids', [])
        x_references = xpost_ids if xpost_ids else []

        # === WEB SEARCH ===
        web_search_results = resp.get('web_search_results', [])
        cited_web_search_results = resp.get('cited_web_search_results', [])

        # Build metadata
        msg_metadata = {}
        if thinking:
            msg_metadata['thinking_trace'] = thinking
        if thinking_duration is not None:
            msg_metadata['thinking_duration_ms'] = thinking_duration
        if web_search_results:
            msg_metadata['web_search_results'] = web_search_results
        if cited_web_search_results:
            msg_metadata['cited_web_search_results'] = cited_web_search_results
        if x_references:
            msg_metadata['x_references'] = x_references
        if citations:
            msg_metadata['citations'] = citations

        # Handle sub-messages (preserve metadata on first only)
        sub_msgs = text.split('\n\n---\n\n') if '\n\n---\n\n' in text else [text]
        for idx, sub in enumerate(sub_msgs):
            sub = sub.strip()
            if sub:
                msg = {
                    "role": role,
                    "text": sub,
                    "timestamp": ts,
                    "attachments": attachments if idx == 0 else [],
                    "model": model,
                    "_metadata": msg_metadata if idx == 0 else {},
                    "_tools": tools if idx == 0 else [],
                }
                messages.append(msg)

    exchanges = _group_into_exchanges_with_tools_and_metadata(messages, 'grok')

    detected_model = ''
    for msg in messages:
        if msg.get('model'):
            detected_model = msg['model']
            break

    return _build_result('grok', conv_id, title, detected_model, filepath, created, updated, exchanges)


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
            "_metadata": {},
            "_tools": [],
        })

    exchanges = _group_into_exchanges_with_tools_and_metadata(messages, 'grok')
    return _build_result('grok', conv_id, title, '', filepath, '', '', exchanges)


# =============================================================================
# GROK TEXT PARSER - Plain text UI exports (JESSE/GROK speaker format)
# =============================================================================

# UI chrome patterns to filter from Grok text exports
_GROK_UI_PATTERNS = [
    re.compile(r'^To view keyboard shortcuts', re.IGNORECASE),
    re.compile(r'^View keyboard shortcuts', re.IGNORECASE),
    re.compile(r'^See new posts', re.IGNORECASE),
    re.compile(r'^Grok \d', re.IGNORECASE),
    re.compile(r'^DeepSearch$', re.IGNORECASE),
    re.compile(r'^Think$', re.IGNORECASE),
    re.compile(r'^Edit Image$', re.IGNORECASE),
    re.compile(r'^Expand for details', re.IGNORECASE),
    re.compile(r'^\d+ web pages?$', re.IGNORECASE),
    re.compile(r'^\d+ sources?$', re.IGNORECASE),
    re.compile(r'^Tap to read', re.IGNORECASE),
    re.compile(r'^Show thinking', re.IGNORECASE),
    re.compile(r'^Completed DeepSearch', re.IGNORECASE),
    re.compile(r'^Thought for', re.IGNORECASE),
    re.compile(r'^Searching for', re.IGNORECASE),
    re.compile(r'^Browsing', re.IGNORECASE),
    re.compile(r'^\d+\+$'),
    re.compile(r'^File$'),
]


def parse_grok_text(filepath: str) -> List[Dict]:
    """Parse Grok plain text transcript (UI export with speaker labels or alternating format)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    fname = Path(filepath).stem

    # Extract conversation ID from filename (e.g., 01_grok-1888708959684735414.txt)
    conv_id_match = re.search(r'grok-(\d+)', fname)
    conv_id = f"grok-{conv_id_match.group(1)}" if conv_id_match else make_id('grok', fname)

    lines = content.split('\n')

    # Filter UI chrome
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append('')
            continue
        is_noise = any(p.match(stripped) for p in _GROK_UI_PATTERNS)
        if not is_noise:
            filtered.append(stripped)

    # Try speaker-label format first (JESSE: / GROK:)
    messages = _try_speaker_label_parse(filtered)

    # Fall back to alternating block heuristic
    if not messages:
        messages = _grok_alternating_blocks(filtered)

    exchanges = _group_into_exchanges(messages, 'grok')

    # Title from first user message
    title = fname
    if messages and messages[0].get('role') == 'user':
        title = messages[0]['text'].split('\n')[0][:80]

    return [_build_result('grok', conv_id, title, '', filepath, '', '', exchanges)]


def _try_speaker_label_parse(lines: List[str]) -> List[Dict]:
    """Parse lines with explicit JESSE:/GROK: speaker labels."""
    messages = []
    current_role = None
    current_text = []
    label_re = re.compile(r'^(JESSE|GROK|Jesse|Grok)\s*:\s*(.*)', re.IGNORECASE)

    label_count = sum(1 for line in lines if label_re.match(line))
    if label_count < 4:
        return []  # Not speaker-label format

    for line in lines:
        m = label_re.match(line)
        if m:
            # Save previous message
            if current_role and current_text:
                messages.append({
                    "role": current_role,
                    "text": '\n'.join(current_text).strip(),
                    "timestamp": "",
                    "attachments": [],
                })
            speaker = m.group(1).upper()
            current_role = 'user' if speaker == 'JESSE' else 'assistant'
            rest = m.group(2).strip()
            current_text = [rest] if rest else []
        elif line.strip():
            current_text.append(line)
        # blank lines: continue current message

    if current_role and current_text:
        messages.append({
            "role": current_role,
            "text": '\n'.join(current_text).strip(),
            "timestamp": "",
            "attachments": [],
        })

    return messages


def _grok_alternating_blocks(lines: List[str]) -> List[Dict]:
    """Parse alternating user/Grok blocks separated by blank lines."""
    blocks = []
    current = []
    for line in lines:
        if line.strip():
            current.append(line)
        else:
            if current:
                blocks.append('\n'.join(current))
                current = []
    if current:
        blocks.append('\n'.join(current))

    messages = []
    role = 'user'
    current_text = []

    for block in blocks:
        if not block.strip():
            continue

        is_short_question = len(block) < 300 and '?' in block
        starts_grok = block.startswith(('Jesse,', 'Hello', 'Alright', 'Given', 'Below'))

        if role == 'user':
            if starts_grok or (current_text and not is_short_question):
                if current_text:
                    messages.append({
                        "role": "user",
                        "text": '\n\n'.join(current_text).strip(),
                        "timestamp": "",
                        "attachments": [],
                    })
                current_text = [block]
                role = 'assistant'
            else:
                current_text.append(block)
        else:
            if is_short_question and not block.startswith(('Yes', 'No', 'However')):
                messages.append({
                    "role": "assistant",
                    "text": '\n\n'.join(current_text).strip(),
                    "timestamp": "",
                    "attachments": [],
                })
                current_text = [block]
                role = 'user'
            else:
                current_text.append(block)

    if current_text:
        messages.append({
            "role": role,
            "text": '\n\n'.join(current_text).strip(),
            "timestamp": "",
            "attachments": [],
        })

    return messages


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
# GEMINI UNIFIED PARSER - All Gemini sources, exchange-level dedup
# =============================================================================

# Base directories for Gemini data
_GEMINI_OLD_BASE = Path("/home/spark/builder-taey/family_transcripts/for_processing/gemini")
_GEMINI_RAW_EXPORT = Path("/home/spark/data/transcripts/raw_exports/gemini")

# Regex for the Activity Log date format: "Jan 5, 2025 at 3:51 AM" or "Yesterday at 10:01 PM"
_ACTIVITY_DATE_RE = re.compile(
    r'(?:(\w+ \d{1,2},? \d{4})|(\w+ \d{1,2})|(\w+))\s+at\s+(\d{1,2}:\d{2})\s*(AM|PM)',
    re.IGNORECASE
)


def _parse_activity_log_date(date_str: str, file_date_hint: str = "2026-02-17") -> str:
    """Parse Google Activity Log date to ISO 8601.

    Handles:
      "Nov 30, 2025 at 1:58 AM"
      "January 4 at 12:59 AM"        (no year - infer from context)
      "Yesterday at 10:01 PM"
      "Today at 5:00 PM"
    """
    if not date_str:
        return ""

    # Clean non-breaking spaces
    date_str = date_str.replace('\u202f', ' ').replace('\xa0', ' ').strip()

    # Try full formats with dateutil-like manual parsing
    # Format 1: "Nov 30, 2025 at 1:58 AM" or "November 30, 2025 at 1:58 AM"
    # Format 2: "January 4 at 12:59 AM" (no year)
    # Format 3: "Yesterday at 10:01 PM" / "Today at ..."

    # Extract time portion
    time_match = re.search(r'(\d{1,2}:\d{2})\s*(AM|PM)', date_str, re.IGNORECASE)
    if not time_match:
        return ""

    time_str = time_match.group(1)
    ampm = time_match.group(2).upper()

    # Parse time
    hour, minute = map(int, time_str.split(':'))
    if ampm == 'PM' and hour != 12:
        hour += 12
    elif ampm == 'AM' and hour == 12:
        hour = 0

    # Parse date portion (everything before "at")
    date_part = date_str.split(' at ')[0].strip() if ' at ' in date_str else ''

    # Handle relative dates
    try:
        ref_date = datetime.strptime(file_date_hint, "%Y-%m-%d")
    except ValueError:
        ref_date = datetime(2026, 2, 17)

    if date_part.lower() == 'yesterday':
        from datetime import timedelta
        d = ref_date - timedelta(days=1)
    elif date_part.lower() == 'today':
        d = ref_date
    else:
        # Try parsing with year: "Nov 30, 2025" or "November 30, 2025"
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                d = datetime.strptime(date_part, fmt)
                break
            except ValueError:
                continue
        else:
            # No year: "January 4" or "Jan 4"
            for fmt in ("%B %d", "%b %d"):
                try:
                    d = datetime.strptime(date_part, fmt)
                    # Infer year: if month > ref month, it's previous year
                    year = ref_date.year
                    if d.month > ref_date.month:
                        year -= 1
                    d = d.replace(year=year)
                    break
                except ValueError:
                    continue
            else:
                return ""

    d = d.replace(hour=hour, minute=minute, second=0)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_gemini_activity_md(filepath: str) -> List[Dict]:
    """Parse Gemini Activity Log .md format.

    Format is repeating blocks:
        Logo for Gemini Apps
        Gemini Apps
        Prompted {user text}
        Details
        event
        {date}
        apps
        Gemini Apps
        chat

        {response text...}
    """
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Infer file date from filename if possible (GEMINI_2025_11_18-2026_02_17_01_12.md)
    fname = Path(filepath).stem
    date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})(?:_(\d{2})_(\d{2}))?$', fname)
    if date_match:
        file_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        file_date = "2026-02-17"

    # Find all blocks starting with "Logo for Gemini Apps"
    exchanges = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'Logo for Gemini Apps':
            # Parse block header
            # Line i:   Logo for Gemini Apps
            # Line i+1: Gemini Apps
            # Line i+2: Prompted {text}
            # Line i+3: Details
            # Line i+4: event
            # Line i+5: {date}
            # Line i+6: apps
            # Line i+7: Gemini Apps
            # Line i+8: chat
            # Line i+9: (blank)
            # Line i+10+: response text until next "Logo for Gemini Apps"

            if i + 5 >= len(lines):
                i += 1
                continue

            # Extract user prompt
            prompt_line = lines[i + 2] if i + 2 < len(lines) else ''
            user_text = prompt_line
            if user_text.startswith('Prompted '):
                user_text = user_text[9:]

            # Extract date
            date_line = lines[i + 5] if i + 5 < len(lines) else ''
            timestamp = _parse_activity_log_date(date_line, file_date)

            # Find response text: skip header lines, collect until next block
            # Response starts after the "chat" line + blank
            resp_start = i + 10  # After "chat" + blank
            # But be flexible - find the "chat" line
            for j in range(i + 3, min(i + 12, len(lines))):
                if lines[j].strip() == 'chat':
                    resp_start = j + 1
                    # Skip blank line after "chat"
                    if resp_start < len(lines) and lines[resp_start].strip() == '':
                        resp_start += 1
                    break

            # Collect response lines until next "Logo for Gemini Apps" or EOF
            resp_lines = []
            k = resp_start
            while k < len(lines):
                if lines[k].strip() == 'Logo for Gemini Apps':
                    break
                resp_lines.append(lines[k])
                k += 1

            response_text = '\n'.join(resp_lines).strip()

            if user_text.strip():
                exchanges.append({
                    "user_text": user_text.strip(),
                    "response_text": response_text,
                    "timestamp": timestamp,
                    "date_str": date_line.strip(),
                })

            i = k  # Jump to next block
        else:
            i += 1

    return exchanges


def _infer_gemini_sessions(exchanges: List[Dict]) -> List[List[Dict]]:
    """Group exchanges into sessions/conversations by timing and content.

    Heuristic: exchanges within 2 hours of each other are same session,
    unless user text starts a new topic (contains greeting/system prompt markers).
    """
    if not exchanges:
        return []

    # Sort by timestamp
    sorted_ex = sorted(exchanges, key=lambda e: e.get('timestamp', '') or '9999')

    sessions = []
    current_session = [sorted_ex[0]]

    NEW_SESSION_MARKERS = [
        'start research', 'hi gemini', 'hey gemini', 'gemini,',
        'hello gemini', 'hi map',
    ]

    for prev, curr in zip(sorted_ex, sorted_ex[1:]):
        ts_prev = prev.get('timestamp', '')
        ts_curr = curr.get('timestamp', '')

        # Check time gap
        gap_hours = 999
        if ts_prev and ts_curr:
            try:
                t1 = datetime.strptime(ts_prev[:19], "%Y-%m-%dT%H:%M:%S")
                t2 = datetime.strptime(ts_curr[:19], "%Y-%m-%dT%H:%M:%S")
                gap_hours = abs((t2 - t1).total_seconds()) / 3600
            except ValueError:
                pass

        # Check if new session
        user_lower = curr.get('user_text', '').lower()[:50]
        is_greeting = any(user_lower.startswith(m) for m in NEW_SESSION_MARKERS)

        if gap_hours > 2 or (gap_hours > 0.5 and is_greeting):
            sessions.append(current_session)
            current_session = [curr]
        else:
            current_session.append(curr)

    if current_session:
        sessions.append(current_session)

    return sessions


def _exchange_hash(user_text: str, response_text: str) -> str:
    """Hash for exchange-level dedup. Uses first 500 chars of each to handle
    minor formatting differences across sources."""
    key = (user_text or '')[:500] + '|||' + (response_text or '')[:500]
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def parse_gemini_all() -> List[Dict]:
    """Unified Gemini parser. Ingests ALL sources, deduplicates at exchange level.

    Sources (in priority order for dedup - earlier source wins):
    1. New .md Activity Log export (raw_exports/gemini/)
    2. oct-nov JSON conversations
    3. new_transcripts JSON conversations
    4. layer_3 universal JSON conversations
    5. enriched JSON conversations (previous pipeline output)

    Artifacts from artifacts/ directory are linked via complete_artifact_mapping.json.
    """
    all_exchanges = []  # List of (exchange_dict, source_name)
    seen_hashes = set()

    def _add_exchanges(exchanges, source):
        """Add exchanges, dedup by hash."""
        added = 0
        for ex in exchanges:
            h = _exchange_hash(ex.get('user_text', ''), ex.get('response_text', ''))
            if h not in seen_hashes:
                seen_hashes.add(h)
                ex['_source'] = source
                ex['_hash'] = h
                all_exchanges.append(ex)
                added += 1
        return added

    print("=" * 70, flush=True)
    print("Gemini Unified Parser - All Sources", flush=True)
    print("=" * 70, flush=True)

    # -------------------------------------------------------------------------
    # Source 1: New .md Activity Log exports
    # -------------------------------------------------------------------------
    md_files = sorted(_GEMINI_RAW_EXPORT.glob("*.md")) if _GEMINI_RAW_EXPORT.exists() else []
    for md_file in md_files:
        exchanges = _parse_gemini_activity_md(str(md_file))
        n = _add_exchanges(exchanges, f"activity_md:{md_file.name}")
        print(f"  activity_md/{md_file.name}: {len(exchanges)} parsed, {n} new", flush=True)

    # -------------------------------------------------------------------------
    # Source 2: oct-nov JSON conversations
    # -------------------------------------------------------------------------
    oct_nov = _GEMINI_OLD_BASE / "oct-nov"
    if oct_nov.exists():
        for jf in sorted(oct_nov.glob("*.json")):
            try:
                data = json.load(open(jf))
                if isinstance(data, list):
                    data = data[0] if data else {}
                msgs = data.get('messages', [])
                exchanges = []
                # Pair user/assistant messages
                i = 0
                while i < len(msgs):
                    if msgs[i].get('role') == 'user':
                        user_text = msgs[i].get('content', '') or msgs[i].get('text', '')
                        resp_text = ''
                        ts = ensure_iso(msgs[i].get('timestamp', ''))
                        if i + 1 < len(msgs) and msgs[i + 1].get('role') in ('assistant', 'model'):
                            resp_text = msgs[i + 1].get('content', '') or msgs[i + 1].get('text', '')
                            i += 2
                        else:
                            i += 1
                        exchanges.append({
                            "user_text": user_text.strip(),
                            "response_text": resp_text.strip(),
                            "timestamp": ts,
                        })
                    else:
                        i += 1
                n = _add_exchanges(exchanges, f"oct_nov:{jf.name}")
                print(f"  oct-nov/{jf.name}: {len(exchanges)} parsed, {n} new", flush=True)
            except Exception as e:
                print(f"  oct-nov/{jf.name}: ERROR {e}", flush=True)

    # -------------------------------------------------------------------------
    # Source 3: new_transcripts JSON
    # -------------------------------------------------------------------------
    new_trans = _GEMINI_OLD_BASE / "new_transcripts"
    if new_trans.exists():
        for jf in sorted(new_trans.glob("*.json")):
            try:
                data = json.load(open(jf))
                items = data if isinstance(data, list) else [data]
                file_exchanges = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    msgs = item.get('messages', [])
                    i = 0
                    while i < len(msgs):
                        if msgs[i].get('role') == 'user':
                            user_text = msgs[i].get('content', '') or msgs[i].get('text', '')
                            resp_text = ''
                            ts = ensure_iso(msgs[i].get('timestamp', ''))
                            if i + 1 < len(msgs) and msgs[i + 1].get('role') in ('assistant', 'model'):
                                resp_text = msgs[i + 1].get('content', '') or msgs[i + 1].get('text', '')
                                i += 2
                            else:
                                i += 1
                            file_exchanges.append({
                                "user_text": user_text.strip(),
                                "response_text": resp_text.strip(),
                                "timestamp": ts,
                            })
                        else:
                            i += 1
                n = _add_exchanges(file_exchanges, f"new_trans:{jf.name}")
                print(f"  new_transcripts/{jf.name}: {len(file_exchanges)} parsed, {n} new", flush=True)
            except Exception as e:
                print(f"  new_transcripts/{jf.name}: ERROR {e}", flush=True)

    # -------------------------------------------------------------------------
    # Source 4: layer_3 universal JSON
    # -------------------------------------------------------------------------
    layer3 = _GEMINI_OLD_BASE / "layer_3"
    if layer3.exists():
        for jf in sorted(layer3.glob("*_universal.json")):
            try:
                data = json.load(open(jf))
                msgs = data.get('messages', [])
                exchanges = []
                i = 0
                while i < len(msgs):
                    if msgs[i].get('role') == 'user':
                        user_text = msgs[i].get('content', '') or msgs[i].get('text', '')
                        resp_text = ''
                        ts = ensure_iso(msgs[i].get('timestamp', ''))
                        if i + 1 < len(msgs) and msgs[i + 1].get('role') in ('assistant', 'model'):
                            resp_text = msgs[i + 1].get('content', '') or msgs[i + 1].get('text', '')
                            i += 2
                        else:
                            i += 1
                        exchanges.append({
                            "user_text": user_text.strip(),
                            "response_text": resp_text.strip(),
                            "timestamp": ts,
                        })
                    else:
                        i += 1
                n = _add_exchanges(exchanges, f"layer3:{jf.name}")
                print(f"  layer_3/{jf.name}: {len(exchanges)} parsed, {n} new", flush=True)
            except Exception as e:
                print(f"  layer_3/{jf.name}: ERROR {e}", flush=True)

    # -------------------------------------------------------------------------
    # Source 5: enriched JSON (previous pipeline output)
    # -------------------------------------------------------------------------
    enriched = _GEMINI_OLD_BASE / "enriched"
    if enriched.exists():
        for jf in sorted(enriched.glob("*.json")):
            try:
                data = json.load(open(jf))
                msgs = data.get('messages', [])
                exchanges = []
                i = 0
                while i < len(msgs):
                    if msgs[i].get('role') == 'user':
                        user_text = msgs[i].get('content', '') or msgs[i].get('text', '')
                        resp_text = ''
                        ts = ensure_iso(msgs[i].get('timestamp', ''))
                        if i + 1 < len(msgs) and msgs[i + 1].get('role') in ('assistant', 'model'):
                            resp_text = msgs[i + 1].get('content', '') or msgs[i + 1].get('text', '')
                            i += 2
                        else:
                            i += 1
                        exchanges.append({
                            "user_text": user_text.strip(),
                            "response_text": resp_text.strip(),
                            "timestamp": ts,
                        })
                    else:
                        i += 1
                n = _add_exchanges(exchanges, f"enriched:{jf.name}")
                # Only print if new content found (enriched has lots of dupes)
                if n > 0:
                    print(f"  enriched/{jf.name}: {len(exchanges)} parsed, {n} new", flush=True)
            except Exception as e:
                print(f"  enriched/{jf.name}: ERROR {e}", flush=True)

    print(f"\n  Total unique exchanges: {len(all_exchanges)}", flush=True)

    # -------------------------------------------------------------------------
    # Load artifact mapping
    # -------------------------------------------------------------------------
    artifact_map = {}  # conversation title -> [artifact filenames]
    artifact_content = {}  # artifact filename -> content
    mapping_file = _GEMINI_OLD_BASE / "complete_artifact_mapping.json"
    artifacts_dir = _GEMINI_OLD_BASE / "artifacts"

    if mapping_file.exists():
        try:
            mapping = json.load(open(mapping_file))
            for conv_title, art_list in mapping.get('conversation_artifact_mapping', {}).items():
                artifact_map[conv_title] = art_list
        except Exception:
            pass

    if artifacts_dir.exists():
        for af in artifacts_dir.iterdir():
            if af.is_file():
                try:
                    artifact_content[af.stem] = af.read_text(encoding='utf-8')
                except Exception:
                    pass

    if artifact_content:
        print(f"  Loaded {len(artifact_content)} artifact files", flush=True)

    # -------------------------------------------------------------------------
    # Group into sessions and build output
    # -------------------------------------------------------------------------
    sessions = _infer_gemini_sessions(all_exchanges)
    print(f"  Inferred {len(sessions)} sessions", flush=True)

    results = []
    for sess_idx, session in enumerate(sessions):
        # Use first exchange timestamp for session ID
        first_ts = session[0].get('timestamp', '')
        first_prompt = session[0].get('user_text', '')[:40]

        conv_id = make_id('gemini', first_ts or str(sess_idx), first_prompt)

        # Generate title from first user prompt
        title = first_prompt.strip()
        if len(title) > 60:
            title = title[:57] + '...'
        if not title:
            title = f"Gemini Session {sess_idx + 1}"

        # Build exchanges in standard format
        exchanges = []
        for idx, ex in enumerate(session):
            user_text = ex.get('user_text', '')
            resp_text = ex.get('response_text', '')

            resp_artifacts = extract_artifacts(resp_text) if resp_text else []
            file_refs = extract_file_references(user_text + '\n' + resp_text)

            exchanges.append({
                "index": idx,
                "timestamp": ex.get('timestamp', ''),
                "user_prompt": user_text,
                "responses": [{
                    "text": resp_text,
                    "tools": [],
                    "artifacts": resp_artifacts,
                }],
                "attachments": [],
                "file_references": file_refs,
            })

        created = session[0].get('timestamp', '')
        updated = session[-1].get('timestamp', '')

        result = _build_result('gemini', conv_id, title, '', 'gemini_unified', created, updated, exchanges)
        # Add source breakdown to metadata
        sources = list(set(ex.get('_source', '') for ex in session))
        result['metadata']['sources'] = sources
        results.append(result)

    # -------------------------------------------------------------------------
    # Attach artifacts to matching sessions
    # -------------------------------------------------------------------------
    if artifact_map:
        attached = 0
        for result in results:
            for ex in result.get('exchanges', []):
                for resp in ex.get('responses', []):
                    resp_text = resp.get('text', '')
                    # Check if any artifact name is mentioned in the response
                    for art_name, art_text in artifact_content.items():
                        # Check if artifact name (or close variant) appears in response
                        if art_name.replace('_', ' ').lower() in resp_text.lower()[:500]:
                            resp['artifacts'].append({
                                "type": "gemini_artifact",
                                "name": art_name,
                                "content": art_text[:50000],  # Cap at 50K chars
                                "fingerprint": hashlib.md5(art_text.encode()).hexdigest()[:16],
                            })
                            attached += 1
        if attached:
            print(f"  Attached {attached} artifacts to exchanges", flush=True)

    total_ex = sum(len(r['exchanges']) for r in results)
    total_art = sum(
        len(a) for r in results for ex in r['exchanges']
        for resp in ex['responses'] for a in [resp.get('artifacts', [])]
    )
    print(f"\n  Output: {len(results)} conversations, {total_ex} exchanges, {total_art} artifacts", flush=True)
    return results


def _parse_gemini_activity_md_as_results(filepath: str) -> List[Dict]:
    """Wrapper: parse single .md file into session-grouped results (for detect_and_parse)."""
    exchanges = _parse_gemini_activity_md(filepath)
    if not exchanges:
        return []

    sessions = _infer_gemini_sessions(exchanges)
    results = []
    for sess_idx, session in enumerate(sessions):
        first_ts = session[0].get('timestamp', '')
        first_prompt = session[0].get('user_text', '')[:40]
        conv_id = make_id('gemini', first_ts or str(sess_idx), first_prompt)
        title = first_prompt.strip()
        if len(title) > 60:
            title = title[:57] + '...'
        if not title:
            title = f"Gemini Session {sess_idx + 1}"

        formatted = []
        for idx, ex in enumerate(session):
            user_text = ex.get('user_text', '')
            resp_text = ex.get('response_text', '')
            formatted.append({
                "index": idx,
                "timestamp": ex.get('timestamp', ''),
                "user_prompt": user_text,
                "responses": [{
                    "text": resp_text,
                    "tools": [],
                    "artifacts": extract_artifacts(resp_text) if resp_text else [],
                }],
                "attachments": [],
                "file_references": extract_file_references(user_text + '\n' + resp_text),
            })

        created = session[0].get('timestamp', '')
        updated = session[-1].get('timestamp', '')
        results.append(_build_result('gemini', conv_id, title, '', filepath, created, updated, formatted))

    return results


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


def _group_into_exchanges_with_tools_and_metadata(messages: List[Dict], platform: str) -> List[Dict]:
    """Group messages with tool and metadata extraction (for Grok, Claude Code).

    Handles consecutive same-role messages by merging metadata properly:
    - Consecutive assistant messages: merge into single response but preserve all metadata
    - Orphaned assistant at start: create exchange with empty user prompt
    - Orphaned user at end: create exchange with empty response
    """
    exchanges = []
    current_user = None
    current_responses = []
    current_attachments = []
    current_timestamp = ''
    idx = 0

    for msg in messages:
        role = msg.get('role', '')

        if role == 'user':
            # Save previous exchange (if any) before starting new user message
            if current_user is not None:
                if current_responses:
                    # Save previous user->assistant exchange
                    resp_texts = []
                    all_tools = []
                    all_metadata = {}
                    all_resp_attachments = []

                    for r in current_responses:
                        if r.get('text'):
                            resp_texts.append(r['text'])
                        all_tools.extend(r.get('_tools', []))
                        all_resp_attachments.extend(r.get('attachments', []))

                        # Merge metadata - preserve all thinking traces
                        meta = r.get('_metadata', {})
                        for k, v in meta.items():
                            if k == 'thinking_trace':
                                # Concatenate thinking traces if multiple
                                if k in all_metadata:
                                    all_metadata[k] += '\n\n---\n\n' + v
                                else:
                                    all_metadata[k] = v
                            elif k not in all_metadata:
                                all_metadata[k] = v

                    resp_text = '\n'.join(resp_texts)
                    file_refs = extract_file_references(current_user + '\n' + resp_text)

                    # Extract file refs from tool inputs
                    for tool in all_tools:
                        inp = tool.get('input', {})
                        fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
                        if fp:
                            refs = extract_file_references(fp)
                            file_refs.extend(refs)
                    file_refs = sorted(set(file_refs))

                    # Combine user attachments and response attachments
                    all_attachments = current_attachments + all_resp_attachments

                    response_obj = {
                        "text": resp_text,
                        "tools": all_tools,
                        "artifacts": extract_artifacts(resp_text),
                    }
                    if all_metadata:
                        response_obj['metadata'] = all_metadata

                    exchanges.append({
                        "index": idx,
                        "timestamp": current_timestamp,
                        "user_prompt": current_user,
                        "responses": [response_obj],
                        "attachments": all_attachments,
                        "file_references": file_refs,
                    })
                    idx += 1

                else:
                    # Consecutive user messages with no assistant response between
                    # Save orphaned user message with empty response
                    file_refs = extract_file_references(current_user)
                    exchanges.append({
                        "index": idx,
                        "timestamp": current_timestamp,
                        "user_prompt": current_user,
                        "responses": [{"text": "", "tools": [], "artifacts": []}],
                        "attachments": current_attachments,
                        "file_references": file_refs,
                    })
                    idx += 1

            elif not current_user and current_responses:
                # Orphaned assistant messages at start - create exchange with empty user
                resp_texts = []
                all_tools = []
                all_metadata = {}
                all_resp_attachments = []

                for r in current_responses:
                    if r.get('text'):
                        resp_texts.append(r['text'])
                    all_tools.extend(r.get('_tools', []))
                    all_resp_attachments.extend(r.get('attachments', []))
                    meta = r.get('_metadata', {})
                    for k, v in meta.items():
                        if k == 'thinking_trace':
                            if k in all_metadata:
                                all_metadata[k] += '\n\n---\n\n' + v
                            else:
                                all_metadata[k] = v
                        elif k not in all_metadata:
                            all_metadata[k] = v

                resp_text = '\n'.join(resp_texts)
                file_refs = extract_file_references(resp_text)
                for tool in all_tools:
                    inp = tool.get('input', {})
                    fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
                    if fp:
                        file_refs.extend(extract_file_references(fp))
                file_refs = sorted(set(file_refs))

                response_obj = {
                    "text": resp_text,
                    "tools": all_tools,
                    "artifacts": extract_artifacts(resp_text),
                }
                if all_metadata:
                    response_obj['metadata'] = all_metadata

                exchanges.append({
                    "index": idx,
                    "timestamp": current_timestamp,
                    "user_prompt": "",
                    "responses": [response_obj],
                    "attachments": all_resp_attachments,
                    "file_references": file_refs,
                })
                idx += 1

            # Start new user message
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
        all_metadata = {}
        all_resp_attachments = []

        for r in current_responses:
            if r.get('text'):
                resp_texts.append(r['text'])
            all_tools.extend(r.get('_tools', []))
            all_resp_attachments.extend(r.get('attachments', []))
            meta = r.get('_metadata', {})
            for k, v in meta.items():
                if k == 'thinking_trace':
                    if k in all_metadata:
                        all_metadata[k] += '\n\n---\n\n' + v
                    else:
                        all_metadata[k] = v
                elif k not in all_metadata:
                    all_metadata[k] = v

        resp_text = '\n'.join(resp_texts)
        file_refs = extract_file_references(current_user + '\n' + resp_text)
        for tool in all_tools:
            inp = tool.get('input', {})
            fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
            if fp:
                file_refs.extend(extract_file_references(fp))
        file_refs = sorted(set(file_refs))

        all_attachments = current_attachments + all_resp_attachments

        response_obj = {
            "text": resp_text,
            "tools": all_tools,
            "artifacts": extract_artifacts(resp_text),
        }
        if all_metadata:
            response_obj['metadata'] = all_metadata

        exchanges.append({
            "index": idx,
            "timestamp": current_timestamp,
            "user_prompt": current_user,
            "responses": [response_obj],
            "attachments": all_attachments,
            "file_references": file_refs,
        })
    elif not current_user and current_responses:
        # Orphaned assistant at end
        resp_texts = []
        all_tools = []
        all_metadata = {}
        all_resp_attachments = []

        for r in current_responses:
            if r.get('text'):
                resp_texts.append(r['text'])
            all_tools.extend(r.get('_tools', []))
            all_resp_attachments.extend(r.get('attachments', []))
            meta = r.get('_metadata', {})
            for k, v in meta.items():
                if k == 'thinking_trace':
                    if k in all_metadata:
                        all_metadata[k] += '\n\n---\n\n' + v
                    else:
                        all_metadata[k] = v
                elif k not in all_metadata:
                    all_metadata[k] = v

        resp_text = '\n'.join(resp_texts)
        file_refs = extract_file_references(resp_text)
        for tool in all_tools:
            inp = tool.get('input', {})
            fp = inp.get('file_path', '') or inp.get('path', '') or inp.get('command', '')
            if fp:
                file_refs.extend(extract_file_references(fp))
        file_refs = sorted(set(file_refs))

        response_obj = {
            "text": resp_text,
            "tools": all_tools,
            "artifacts": extract_artifacts(resp_text),
        }
        if all_metadata:
            response_obj['metadata'] = all_metadata

        exchanges.append({
            "index": idx,
            "timestamp": current_timestamp or (current_responses[0].get('timestamp', '') if current_responses else ''),
            "user_prompt": "",
            "responses": [response_obj],
            "attachments": all_resp_attachments,
            "file_references": file_refs,
        })
    elif current_user and not current_responses:
        # Orphaned user message at end - create exchange with empty response
        exchanges.append({
            "index": idx,
            "timestamp": current_timestamp,
            "user_prompt": current_user,
            "responses": [{"text": "", "tools": [], "artifacts": []}],
            "attachments": current_attachments,
            "file_references": extract_file_references(current_user),
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

    # Markdown files
    if fp.suffix == '.md':
        # Gemini Activity Log format
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.read(500)
            if 'Logo for Gemini Apps' in first_lines or 'Gemini Apps' in first_lines:
                return _parse_gemini_activity_md_as_results(filepath)
        except Exception:
            pass
        # Plain text Grok transcripts with .md extension
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.read(2000)
            if 'grok' in fname or 'GROK' in first_lines[:500]:
                return parse_grok_text(filepath)
        except Exception:
            pass
        return []

    # Plain text Grok transcripts
    if fp.suffix == '.txt':
        # Check if it's a Grok text transcript
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.read(2000)
            if 'grok' in fname or 'GROK' in first_lines[:500] or 'Grok' in first_lines[:500]:
                return parse_grok_text(filepath)
        except Exception:
            pass
        return []

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
            # xAI Grok: has 'chat_messages' AND 'account' (distinguishes from Claude)
            if 'chat_messages' in first and 'account' in first:
                return parse_xai_grok_bulk(filepath)
            # Claude: has 'chat_messages' key (no 'account')
            if 'chat_messages' in first:
                return parse_claude_bulk(filepath)
            # Grok array
            if 'responses' in first or 'conversation' in first:
                return parse_grok_bulk(filepath)

    # Single conversation object
    if isinstance(data, dict):
        # Grok master bulk: {"conversations": [...]}
        if 'conversations' in data:
            return parse_grok_bulk(filepath)
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
    parser.add_argument("--gemini-all", action="store_true",
                        help="Run unified Gemini parser (all sources, exchange-level dedup)")
    args = parser.parse_args()

    if args.gemini_all:
        results = parse_gemini_all()
        if not args.check:
            out = Path(args.output) / "gemini"
            out.mkdir(parents=True, exist_ok=True)
            # Clear existing gemini files first
            existing = list(out.glob("*.json"))
            if existing:
                print(f"\nClearing {len(existing)} existing gemini files...", flush=True)
                for f in existing:
                    f.unlink()
            for r in results:
                safe_id = r['conversation_id'][:16]
                out_file = out / f"{safe_id}.json"
                with open(out_file, 'w') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"Wrote {len(results)} files to {out}", flush=True)
        else:
            for r in results:
                print(json.dumps({
                    "conversation_id": r["conversation_id"],
                    "title": r["title"],
                    "exchanges": r["metadata"]["total_exchanges"],
                    "created": r["created_at"],
                    "updated": r["updated_at"],
                }, indent=2))
        sys.exit(0)

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
