#!/usr/bin/env python3
"""Test the enhanced Claude Chat parser on actual exports."""

import json
import sys
from pathlib import Path
from parse_raw_exports import parse_claude_bulk

def analyze_export(filepath: str):
    """Analyze a Claude export and show what content types were captured."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*80}\n")

    results = parse_claude_bulk(filepath)

    for conv in results:
        print(f"Conversation: {conv['title']}")
        print(f"ID: {conv['conversation_id']}")
        print(f"Exchanges: {len(conv['exchanges'])}\n")

        # Count content types across all exchanges
        stats = {
            'total_exchanges': 0,
            'exchanges_with_thinking': 0,
            'exchanges_with_artifacts': 0,
            'exchanges_with_tools': 0,
            'exchanges_with_citations': 0,
            'total_artifacts': 0,
            'total_tools': 0,
            'total_citations': 0,
            'artifact_types': {},
            'tool_types': {},
        }

        for ex in conv['exchanges']:
            stats['total_exchanges'] += 1

            for resp in ex.get('responses', []):
                # Check for thinking
                if resp.get('thinking'):
                    stats['exchanges_with_thinking'] += 1

                # Check for artifacts
                artifacts = resp.get('artifacts', [])
                if artifacts:
                    stats['exchanges_with_artifacts'] += 1
                    stats['total_artifacts'] += len(artifacts)
                    for art in artifacts:
                        art_type = art.get('type', 'unknown')
                        stats['artifact_types'][art_type] = stats['artifact_types'].get(art_type, 0) + 1

                # Check for tools
                tools = resp.get('tools', [])
                if tools:
                    stats['exchanges_with_tools'] += 1
                    stats['total_tools'] += len(tools)
                    for tool in tools:
                        tool_type = tool.get('type', 'unknown')
                        stats['tool_types'][tool_type] = stats['tool_types'].get(tool_type, 0) + 1

                # Check for citations
                citations = resp.get('citations', [])
                if citations:
                    stats['exchanges_with_citations'] += 1
                    stats['total_citations'] += len(citations)

        print("STATISTICS:")
        print(f"  Total exchanges: {stats['total_exchanges']}")
        print(f"  With thinking: {stats['exchanges_with_thinking']}")
        print(f"  With artifacts: {stats['exchanges_with_artifacts']} ({stats['total_artifacts']} total)")
        print(f"  With tools: {stats['exchanges_with_tools']} ({stats['total_tools']} total)")
        print(f"  With citations: {stats['exchanges_with_citations']} ({stats['total_citations']} total)")

        if stats['artifact_types']:
            print("\n  Artifact types:")
            for art_type, count in sorted(stats['artifact_types'].items()):
                print(f"    {art_type}: {count}")

        if stats['tool_types']:
            print("\n  Tool types:")
            for tool_type, count in sorted(stats['tool_types'].items()):
                print(f"    {tool_type}: {count}")

        # Show sample artifacts
        print("\nSAMPLE ARTIFACTS (first 3):")
        artifact_count = 0
        for ex in conv['exchanges']:
            for resp in ex.get('responses', []):
                for art in resp.get('artifacts', []):
                    artifact_count += 1
                    if artifact_count <= 3:
                        print(f"\n  [{artifact_count}] Type: {art.get('type')}")
                        if art.get('type') == 'generated':
                            print(f"      ID: {art.get('id', '')[:50]}")
                            print(f"      Title: {art.get('title', '')[:60]}")
                            print(f"      Content Type: {art.get('content_type')}")
                            print(f"      Command: {art.get('command')}")
                            print(f"      Language: {art.get('language')}")
                            print(f"      Content Length: {len(art.get('content', ''))} chars")
                            print(f"      Preview: {art.get('content', '')[:100]}")
                        elif art.get('type') == 'code':
                            print(f"      Language: {art.get('language')}")
                            print(f"      Content Length: {len(art.get('content', ''))} chars")
                            print(f"      Preview: {art.get('content', '')[:100]}")
                    else:
                        break
                if artifact_count >= 3:
                    break
            if artifact_count >= 3:
                break

        # Show sample tools
        print("\nSAMPLE TOOLS (first 3):")
        tool_count = 0
        for ex in conv['exchanges']:
            for resp in ex.get('responses', []):
                for tool in resp.get('tools', []):
                    tool_count += 1
                    if tool_count <= 3:
                        print(f"\n  [{tool_count}] Type: {tool.get('type')}")
                        print(f"      Input keys: {list(tool.get('input', {}).keys())}")
                        if tool.get('output'):
                            print(f"      Output length: {len(str(tool.get('output')))} chars")
                            print(f"      Output preview: {str(tool.get('output'))[:100]}")
                    else:
                        break
                if tool_count >= 3:
                    break
            if tool_count >= 3:
                break

        # Show sample thinking
        print("\nSAMPLE THINKING (first occurrence):")
        found_thinking = False
        for ex in conv['exchanges']:
            for resp in ex.get('responses', []):
                thinking = resp.get('thinking', '')
                if thinking and not found_thinking:
                    print(f"  Length: {len(thinking)} chars")
                    print(f"  Preview: {thinking[:200]}")
                    found_thinking = True
                    break
            if found_thinking:
                break

        if not found_thinking:
            print("  (No thinking blocks found)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_claude_parser.py <path_to_claude_export.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    analyze_export(filepath)
