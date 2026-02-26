#!/usr/bin/env python3
"""
Parse Perplexity markdown exports into Schema B JSON for process_transcripts.py.

Input: Raw .md files from Perplexity "Export as Markdown" with filenames like:
    "Dec 20, 2025-Hi Clarity, it's Spark Claude. Quick question abou.md"

Output: Schema B JSON files in parsed/perplexity/:
    {
        "converter": "parse_perplexity_exports",
        "conversation_id": "<hash>",
        "title": "...",
        "source": "perplexity",
        "created_at": "2025-12-20T00:00:00Z",
        "total_exchanges": N,
        "exchanges": [
            {"prompt": "...", "response": "...", "timestamp": "2025-12-20T00:00:00Z"}
        ]
    }

Perplexity export format:
  - First line: <img> logo tag (stripped)
  - Exchanges separated by \n---\n
  - Each exchange: # heading = user prompt, rest = AI response
  - Some exchanges have no # prompt (continuation responses) - merged with previous
  - Footnotes at bottom: [^N_M]: url (stripped from response text)
"""

import os
import re
import sys
import json
import glob
import hashlib
from datetime import datetime
from pathlib import Path

RAW_DIR = "/home/spark/data/transcripts/raw_exports/perplexity"
PARSED_DIR = "/home/spark/data/transcripts/parsed/perplexity"

# Regex for Perplexity logo tag
LOGO_RE = re.compile(r'<img\s+src="https://r2cdn\.perplexity\.ai/[^"]*"[^/>]*/>\s*')

# Regex for footnote references in text: [^1_2] or <span style="display:none">[^1_1]</span>
FOOTNOTE_INLINE_RE = re.compile(r'<span[^>]*>\[\^\d+_\d+\]</span>|\[\^\d+_\d+\]')

# Regex for footnote definitions at end: [^1_1]: https://...
FOOTNOTE_DEF_RE = re.compile(r'^\[\^\d+_\d+\]:\s+\S+.*$', re.MULTILINE)

# End-of-response decorative divider
DIVIDER_RE = re.compile(r'<div align="center">⁂</div>\s*$')

# Date from filename: "MMM DD, YYYY-Title.md"
FILENAME_DATE_RE = re.compile(
    r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(20\d{2})-'
)

# Date patterns that appear in content headers
CONTENT_DATE_RE = re.compile(
    r'\*\*Date:?\*\*\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2})'
)


def parse_filename_date(filename):
    """Extract date from filename prefix like 'Dec 20, 2025-...'"""
    m = FILENAME_DATE_RE.match(filename)
    if not m:
        return None
    month_str, day, year = m.group(1), m.group(2), m.group(3)
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month = month_map.get(month_str)
    if not month:
        return None
    try:
        return datetime(int(year), month, int(day))
    except ValueError:
        return None


def clean_response(text):
    """Remove footnote refs/defs and trailing dividers from response text."""
    text = FOOTNOTE_INLINE_RE.sub('', text)
    text = FOOTNOTE_DEF_RE.sub('', text)
    text = DIVIDER_RE.sub('', text)
    # Clean up trailing whitespace/newlines from removed footnotes
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_user_prompt(text):
    """Heuristic: does this text look like a user message vs AI response?

    User prompts in Perplexity exports typically start with:
    - '# ' heading (Perplexity wraps the user prompt in H1)
    - Short conversational text
    - Questions directed at Perplexity/Clarity
    """
    stripped = text.strip()
    if not stripped:
        return False

    first_line = stripped.split('\n')[0].strip()

    # Strong signal: starts with # (H1 heading) which is how Perplexity formats prompts
    if first_line.startswith('# '):
        return True

    return False


def extract_prompt_response(block):
    """Extract prompt and response from a --- separated block.

    Returns (prompt, response) tuple. If block has no clear prompt,
    returns (None, full_block_as_response).
    """
    lines = block.strip().split('\n')
    if not lines:
        return None, ''

    first_line = lines[0].strip()

    # Check if first line is a # prompt
    if first_line.startswith('# '):
        prompt = first_line[2:].strip()  # Remove '# ' prefix

        # Check if prompt is repeated on next non-empty line (Perplexity sometimes does this)
        remaining_lines = []
        skip_next_blank = True
        for line in lines[1:]:
            if skip_next_blank and not line.strip():
                skip_next_blank = False
                continue
            # If next non-blank line is identical to prompt, skip it
            if skip_next_blank and line.strip() == prompt:
                skip_next_blank = False
                continue
            skip_next_blank = False
            remaining_lines.append(line)

        response = '\n'.join(remaining_lines).strip()
        return prompt, clean_response(response)

    # No # heading - this is a response-only block (or a continuation)
    return None, clean_response(block.strip())


def parse_file(filepath):
    """Parse a single Perplexity markdown export into Schema B format."""
    filename = os.path.basename(filepath)
    session_date = parse_filename_date(filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip logo
    content = LOGO_RE.sub('', content).strip()

    # Split into exchange blocks
    blocks = content.split('\n---\n')

    exchanges = []
    pending_response = None  # For merging continuation responses

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        prompt, response = extract_prompt_response(block)

        if prompt is not None:
            # New prompt+response pair
            # If there was a pending continuation, finalize it
            if pending_response is not None and exchanges:
                exchanges[-1]['response'] += '\n\n' + pending_response
                pending_response = None

            exchanges.append({
                'prompt': prompt,
                'response': response,
            })
            pending_response = None

        else:
            # No prompt - this is a continuation response
            if exchanges:
                # Append to previous exchange's response
                if pending_response:
                    exchanges[-1]['response'] += '\n\n' + pending_response
                pending_response = response
            else:
                # First block has no prompt - treat whole thing as first exchange
                exchanges.append({
                    'prompt': response[:200] if response else '(no prompt)',
                    'response': response,
                })

    # Finalize any pending continuation
    if pending_response and exchanges:
        exchanges[-1]['response'] += '\n\n' + pending_response

    # Add timestamps
    if session_date:
        iso_date = session_date.strftime('%Y-%m-%dT00:00:00Z')
    else:
        iso_date = ''

    # Try to extract more specific dates from content headers
    for ex in exchanges:
        # Check for **Date:** headers in response
        m = CONTENT_DATE_RE.search(ex['response'][:500])
        if m:
            try:
                parsed = datetime.strptime(m.group(1).replace(',', ''), '%B %d %Y')
                ex['timestamp'] = parsed.strftime('%Y-%m-%dT00:00:00Z')
            except ValueError:
                ex['timestamp'] = iso_date
        else:
            ex['timestamp'] = iso_date

    # Generate conversation ID from content hash
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

    # Title from first prompt or filename
    title = exchanges[0]['prompt'][:100] if exchanges else filename

    return {
        'converter': 'parse_perplexity_exports',
        'conversation_id': content_hash,
        'title': title,
        'source': 'perplexity',
        'created_at': iso_date,
        'total_exchanges': len(exchanges),
        'exchanges': exchanges,
    }


def main():
    os.makedirs(PARSED_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, '*.md')))
    print(f"Found {len(files)} raw Perplexity exports")
    print(f"Output: {PARSED_DIR}")
    print()

    total_exchanges = 0
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            result = parse_file(fpath)
            n_ex = result['total_exchanges']
            total_exchanges += n_ex

            # Write output
            out_path = os.path.join(PARSED_DIR, f"{result['conversation_id']}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)

            print(f"  [{i+1}/{len(files)}] {n_ex:3d} exchanges | {result['created_at'][:10] or '??'} | {fname[:65]}")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] ERROR: {e} | {fname[:65]}")

    print(f"\nDone: {len(files)} files → {total_exchanges} total exchanges")
    print(f"Output in: {PARSED_DIR}")


if __name__ == '__main__':
    main()
