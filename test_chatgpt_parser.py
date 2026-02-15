#!/usr/bin/env python3
"""Test the rewritten ChatGPT parser to verify artifact extraction."""

import json
from pathlib import Path
from isma.scripts.parse_raw_exports import parse_chatgpt_bulk

# Test with real ChatGPT export
export_path = "/home/spark/data/transcripts/raw_exports/chatgpt/conversations.json"

if not Path(export_path).exists():
    print(f"ERROR: Export file not found: {export_path}")
    exit(1)

print("Parsing ChatGPT export...")
results = parse_chatgpt_bulk(export_path)

print(f"\nParsed {len(results)} conversations")

# Summary stats
canvas_count = 0
code_interpreter_count = 0
dalle_count = 0
citations_count = 0
models_seen = set()

for conv in results:
    for exchange in conv['exchanges']:
        # Track model
        if exchange.get('model'):
            models_seen.add(exchange['model'])

        for response in exchange.get('responses', []):
            # Track artifacts
            for artifact in response.get('artifacts', []):
                atype = artifact.get('type', '')
                if atype == 'canvas':
                    canvas_count += 1
                    print(f"\nCANVAS ARTIFACT FOUND:")
                    print(f"  Name: {artifact.get('name')}")
                    print(f"  Type: {artifact.get('content_type')}")
                    print(f"  Command: {artifact.get('command')}")
                    print(f"  Content length: {len(artifact.get('content', ''))}")
                elif atype == 'code_interpreter':
                    code_interpreter_count += 1
                    print(f"\nCODE INTERPRETER FOUND:")
                    print(f"  Code length: {len(artifact.get('code', ''))}")
                elif atype == 'code_interpreter_output':
                    print(f"\nCODE INTERPRETER OUTPUT:")
                    print(f"  Output length: {len(artifact.get('output', ''))}")
                elif atype == 'dalle':
                    dalle_count += 1
                    print(f"\nDALL-E ARTIFACT FOUND:")
                    print(f"  Pointer: {artifact.get('asset_pointer', '')[:80]}")

            # Track citations
            if response.get('citations'):
                citations_count += len(response['citations'])

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"  Canvas artifacts: {canvas_count}")
print(f"  Code interpreter: {code_interpreter_count}")
print(f"  DALL-E images: {dalle_count}")
print(f"  Citations: {citations_count}")
print(f"  Models seen: {sorted(models_seen)}")
print(f"{'='*60}")
