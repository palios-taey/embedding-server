#!/usr/bin/env python3
"""
HMM Family Processor - Overnight motif extraction via Family Chats.

Uses Taey's Hands AT-SPI automation to send content to Family Chats
(ChatGPT, Claude, Gemini, Grok, Perplexity), extract Rosetta dense
summaries and motif assignments, and store in HMM.

Two-phase prompt system:
  Phase 1: Session init - full HMM context, motif dictionary, calibration
           examples, platform-specific analytical lens. Sent once per
           platform. AI confirms "Ready for analysis."
  Phase 2: Content chunks - short prompts with source info + text.
           AI responds with structured JSON.

Usage:
    python3 hmm_family_processor.py --mode corpus       # Process corpus docs
    python3 hmm_family_processor.py --mode transcripts  # Process transcripts
    python3 hmm_family_processor.py --mode all          # Process everything
    python3 hmm_family_processor.py --dry-run           # Show what would be processed
    python3 hmm_family_processor.py --check             # Check platform availability
    python3 hmm_family_processor.py --limit 10          # Process only N items

Designed for unattended overnight operation.
"""

import sys
import os
import json
import time
import logging
import argparse
import hashlib
import re
import subprocess
import faulthandler
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

# Enable segfault tracebacks
faulthandler.enable()

# Add taeys-hands to path for AT-SPI core modules
sys.path.insert(0, "/home/spark/taeys-hands")
# Add ISMA src to path
sys.path.insert(0, "/home/spark/embedding-server/isma")
# Add scripts to path for prompts module
sys.path.insert(0, "/home/spark/embedding-server/isma/scripts")

from core import input as inp, clipboard, atspi
from core.tree import find_elements, find_copy_buttons
from core.platforms import TAB_SHORTCUTS, CHAT_PLATFORMS

from src.hmm.ids import artifact_id, tile_id, canonicalize_text
from src.hmm.eventlog import EventLog, GateSnapshot
from src.hmm.motifs import assign_motifs, V0_MOTIFS, MotifAssignment, DICTIONARY_VERSION
from src.hmm.neo4j_store import HMMNeo4jStore
from src.hmm.redis_store import HMMRedisStore
from src.hmm.gate_b import GateB
from src.retrieval import ISMARetrieval

from hmm_prompts import (
    build_session_init,
    build_content_prompt,
    build_audit_prompt,
    split_content,
    route_document,
    get_audit_platform,
    MAX_CHUNK_CHARS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("hmm_processor")

# Suppress noisy Neo4j notifications about existing indexes
logging.getLogger("neo4j").setLevel(logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================

STATE_FILE = "/var/spark/isma/hmm_processor_state.json"
EVENT_LOG_PATH = "/var/spark/isma/hmm_events.jsonl"
RESPONSE_LOG_DIR = "/var/spark/isma/hmm_responses"

# Platform rotation order
PLATFORMS = ["chatgpt", "gemini", "grok", "perplexity"]

# Timing
WAIT_AFTER_SEND = 3  # seconds after pressing Enter before first poll (matches daemon)
POLL_INTERVAL = 3  # seconds between response checks (matches daemon)
MAX_WAIT_INIT = 300  # 5 minutes for init response (it's long)
MAX_WAIT_CONTENT = 180  # 3 minutes for content response
BETWEEN_MESSAGES = 5  # seconds between messages on same platform
BETWEEN_PLATFORMS = 8  # seconds when switching platforms
NO_STOP_WARNING = 45  # seconds before warning about missing stop button

# Stop button patterns - from Taey's Hands monitor/daemon.py
STOP_PATTERNS = {
    'chatgpt': ['stop', 'stop generating'],
    'claude': ['stop', 'stop response'],
    'gemini': ['stop', 'cancel'],
    'grok': ['stop', 'stop generating'],
    'perplexity': ['stop', 'cancel'],
}

# Content limits
MIN_CONTENT_CHARS = 100  # skip very short content

# Error tolerance
MAX_CONSECUTIVE_ERRORS = 3  # per platform before marking unavailable
MAX_PARSE_FAILURES = 5  # consecutive parse failures before re-initializing


# ============================================================================
# Platform Interaction
# ============================================================================

@dataclass
class PlatformState:
    """Track state for each platform."""
    name: str
    messages_sent: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    parse_failures: int = 0
    last_used: float = 0.0
    available: bool = True
    initialized: bool = False


def check_platform_available(platform: str) -> bool:
    """Check if a platform tab is accessible via AT-SPI."""
    shortcut = TAB_SHORTCUTS.get(platform)
    if not shortcut:
        return False

    inp.focus_firefox()
    time.sleep(0.3)
    inp.press_key(shortcut)
    time.sleep(1.0)

    firefox = atspi.find_firefox()
    if not firefox:
        return False

    doc = atspi.get_platform_document(firefox, platform)
    return doc is not None


def _find_input_area(platform: str) -> Optional[Tuple[int, int]]:
    """Find the input area coordinates dynamically for any platform.

    Strategy:
    1. Look for entry/text input elements (works for Gemini, Claude)
    2. Look for landmark buttons near input row (works for ChatGPT, Grok, Perplexity)
    3. Fall back if neither works

    Returns (x, y) for clicking into the input area, or None.
    """
    firefox = atspi.find_firefox()
    if not firefox:
        return None
    doc = atspi.get_platform_document(firefox, platform)
    if not doc:
        return None

    elements = find_elements(doc, max_depth=15)

    # Strategy 1: Find entry/text input elements
    input_names = ["prompt", "message", "enter a", "write your", "ask"]
    for e in elements:
        role = e.get("role", "")
        name = e.get("name", "").lower()
        if role == "entry" and any(kw in name for kw in input_names):
            x, y = e.get("x", 0), e.get("y", 0)
            if x > 0 and y > 0:
                log.debug(f"[{platform}] Input entry found at ({x}, {y}): '{e.get('name', '')}'")
                return (x, y)

    # Strategy 2: Find landmark buttons near the input area
    buttons = [e for e in elements if e.get("role") == "push button"]
    landmark_names = [
        "Add files and more",     # ChatGPT
        "Attach",                 # Grok, generic
        "Add files or tools",     # Perplexity
        "Open upload file menu",  # Gemini
        "Toggle menu",            # Claude
        "Send prompt",            # ChatGPT send
        "Send Message",           # Claude send
    ]

    for b in buttons:
        bname = b.get("name", "")
        for lm in landmark_names:
            if lm.lower() in bname.lower():
                bx, by = b.get("x", 0), b.get("y", 0)
                if bx > 0 and by > 0:
                    # Input is to the right of attach/menu buttons, same vertical level
                    input_x = bx + 200
                    input_y = by - 15
                    log.debug(f"[{platform}] Input via landmark '{bname}' at ({input_x}, {input_y})")
                    return (input_x, input_y)

    log.warning(f"[{platform}] Could not find input area")
    return None


def _clipboard_paste(text: str):
    """Write text to clipboard and paste via Ctrl+V."""
    try:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            env=os.environ,
        )
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        proc.wait(timeout=10)
    except Exception as e:
        log.error(f"Clipboard write failed: {e}")
        return False

    time.sleep(0.3)
    inp.press_key("ctrl+v")
    time.sleep(0.5)
    return True


def _is_canvas_stop(stop_obj, platform: str) -> bool:
    """Check if a stop button is ChatGPT's canvas stop (not generation stop).

    Canvas has a persistent "Stop" + "Update" button pair. The generation
    stop button is standalone. We detect canvas stop by checking if an
    "Update" button exists at the same Y position (within 50px).

    Ported from monitor/daemon.py _is_canvas_stop().
    """
    if platform != 'chatgpt':
        return False

    try:
        stop_comp = stop_obj.get_component_iface()
        if not stop_comp:
            return False
        stop_ext = stop_comp.get_extents(0)  # ATSPI_COORD_TYPE_SCREEN
        stop_y = stop_ext.y

        # Walk siblings looking for "Update" button at same Y
        parent = stop_obj.get_parent()
        if not parent:
            return False

        for i in range(parent.get_child_count()):
            try:
                sibling = parent.get_child_at_index(i)
                if not sibling:
                    continue
                sib_role = sibling.get_role_name() or ''
                sib_name = (sibling.get_name() or '').lower()
                if sib_role in ('push button', 'button') and 'update' in sib_name:
                    sib_comp = sibling.get_component_iface()
                    if sib_comp:
                        sib_ext = sib_comp.get_extents(0)
                        if abs(sib_ext.y - stop_y) < 50:
                            return True
            except Exception:
                continue

        # Also check grandparent (button groups may be wrapped)
        grandparent = parent.get_parent()
        if grandparent:
            for i in range(grandparent.get_child_count()):
                try:
                    uncle = grandparent.get_child_at_index(i)
                    if not uncle:
                        continue
                    for j in range(uncle.get_child_count()):
                        child = uncle.get_child_at_index(j)
                        if not child:
                            continue
                        ch_role = child.get_role_name() or ''
                        ch_name = (child.get_name() or '').lower()
                        if ch_role in ('push button', 'button') and 'update' in ch_name:
                            ch_comp = child.get_component_iface()
                            if ch_comp:
                                ch_ext = ch_comp.get_extents(0)
                                if abs(ch_ext.y - stop_y) < 50:
                                    return True
                except Exception:
                    continue

    except Exception:
        pass

    return False


def _find_stop_button(platform: str, doc) -> object:
    """Search AT-SPI document tree for a stop button.

    Returns the stop button AT-SPI object, or None.
    Filters out ChatGPT canvas stop buttons (Stop+Update pair).

    Ported from monitor/daemon.py _on_poll_check() find_stops().
    """
    patterns = STOP_PATTERNS.get(platform, ['stop'])
    candidates = []

    def find_stops(obj, depth=0):
        if depth > 25:
            return
        try:
            role = obj.get_role_name() or ''
            name = (obj.get_name() or '').lower()

            if role in ('push button', 'button'):
                if any(p in name for p in patterns):
                    candidates.append(obj)

            for i in range(obj.get_child_count()):
                child = obj.get_child_at_index(i)
                if child:
                    find_stops(child, depth + 1)
        except Exception:
            pass

    find_stops(doc)

    # Filter out canvas stop buttons (ChatGPT-specific)
    for candidate in candidates:
        if not _is_canvas_stop(candidate, platform):
            return candidate

    return None


def _wait_for_response(platform: str, max_wait: int) -> Optional[str]:
    """Wait for AI response using stop-button state machine.

    State machine (from Taey's Hands monitor/daemon.py):
      IDLE -> GENERATING (stop button appears)
      GENERATING -> COMPLETE (stop button disappears)

    Then extracts response via copy button.
    Returns response text or None on timeout.
    """
    stop_seen = False
    no_stop_warned = False
    start = time.time()

    while time.time() - start < max_wait:
        time.sleep(POLL_INTERVAL)
        elapsed = time.time() - start

        firefox = atspi.find_firefox()
        if not firefox:
            continue
        doc = atspi.get_platform_document(firefox, platform)
        if not doc:
            continue

        stop_btn = _find_stop_button(platform, doc)

        if stop_btn:
            if not stop_seen:
                stop_seen = True
                log.info(f"  [{platform}] Stop button appeared - response generating")
        else:
            if stop_seen:
                # Response complete - stop button appeared then disappeared
                log.info(f"  [{platform}] Stop button disappeared - response complete ({elapsed:.0f}s)")
                time.sleep(3)  # Settle time for DOM to add copy buttons
                response = _extract_latest_response(platform)
                # Reject very short extractions (likely old response, not new)
                if response and len(response) < 50:
                    log.warning(f"  [{platform}] Extracted too short ({len(response)} chars), retrying...")
                    time.sleep(3)
                    response = _extract_latest_response(platform)
                return response

        # Warning if no stop button seen after threshold
        if not stop_seen and not no_stop_warned and elapsed > NO_STOP_WARNING:
            log.warning(f"  [{platform}] No stop button after {NO_STOP_WARNING}s - trying extraction")
            no_stop_warned = True
            # Try extraction - response may have completed without visible stop button
            time.sleep(1)
            fallback = _extract_latest_response(platform)
            if fallback and len(fallback) > 50:
                log.info(f"  [{platform}] Found response via fallback ({len(fallback)} chars)")
                return fallback

        if int(elapsed) > 0 and int(elapsed) % 60 == 0 and int(elapsed) != int(elapsed - POLL_INTERVAL):
            state_str = "GENERATING" if stop_seen else "IDLE"
            log.info(f"  [{platform}] Waiting... ({int(elapsed)}s, state={state_str})")

    # Timeout - still try extraction (response may have completed between polls)
    log.warning(f"[{platform}] Timeout after {max_wait}s (stop_seen={stop_seen})")
    last_try = _extract_latest_response(platform)
    if last_try and len(last_try) > 50:
        log.info(f"  [{platform}] Found response on timeout extraction ({len(last_try)} chars)")
        return last_try
    return None


def _find_copy_button_nodes(doc) -> list:
    """Walk AT-SPI tree to find Copy button nodes (preserves node reference).

    Unlike find_elements() which only returns dicts, this returns the actual
    AT-SPI objects so we can use do_action(0) for off-screen buttons.
    """
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi

    results = []

    def walk(obj, depth=0):
        if depth > 15:
            return
        try:
            name = (obj.get_name() or '').lower()
            role = obj.get_role_name() or ''
            if 'button' in role and 'copy' in name:
                comp = obj.get_component_iface()
                y = 0
                if comp:
                    rect = comp.get_extents(Atspi.CoordType.SCREEN)
                    y = rect.y + (rect.height // 2) if rect else 0
                results.append((obj, name, y))
            for i in range(obj.get_child_count()):
                child = obj.get_child_at_index(i)
                if child:
                    walk(child, depth + 1)
        except Exception:
            pass

    walk(doc)
    return results


def _extract_latest_response(platform: str) -> Optional[str]:
    """Click the newest copy button and read clipboard.

    Uses two strategies:
    1. Direct AT-SPI do_action(0) on node - works for off-screen elements
    2. Scroll to bottom + xdotool click at coordinates - fallback
    """
    # Scroll to bottom first to bring newest content into view
    inp.press_key("End")
    time.sleep(0.5)

    firefox = atspi.find_firefox()
    if not firefox:
        return None
    doc = atspi.get_platform_document(firefox, platform)
    if not doc:
        return None

    # Strategy 1: Direct AT-SPI node walk (preserves node for do_action)
    copy_nodes = _find_copy_button_nodes(doc)
    if copy_nodes:
        # Prefer "copy" (response-level) over "copy code" / "copy query"
        response_nodes = [(n, name, y) for n, name, y in copy_nodes if name.strip() == 'copy']
        if response_nodes:
            targets = sorted(response_nodes, key=lambda t: t[2], reverse=True)
        else:
            targets = sorted(copy_nodes, key=lambda t: t[2], reverse=True)

        for node, name, y in targets:
            try:
                clipboard.clear()
                time.sleep(0.2)
                action = node.get_action_iface()
                if action and action.get_n_actions() > 0:
                    action.do_action(0)
                    time.sleep(1.0)
                    content = clipboard.read()
                    if content and len(content.strip()) > 5:
                        return content.strip()
            except Exception:
                pass

    # Strategy 2: Coordinate-based click (needs visible buttons)
    elements = find_elements(doc, max_depth=12)
    copy_buttons = find_copy_buttons(elements)
    if copy_buttons:
        # Try response-level buttons first, then any
        response_btns = [b for b in copy_buttons if (b.get('name') or '').strip().lower() == 'copy']
        targets = response_btns if response_btns else copy_buttons
        for btn in sorted(targets, key=lambda b: b.get("y", 0), reverse=True):
            bx, by = int(btn.get("x", 0)), int(btn.get("y", 0))
            if bx > 0 and by > 0:
                clipboard.clear()
                time.sleep(0.2)
                inp.click_at(bx, by)
                time.sleep(1.0)
                content = clipboard.read()
                if content and len(content.strip()) > 5:
                    return content.strip()

    return None


def _send_via_atspi(platform: str, message: str) -> bool:
    """Send message using AT-SPI methods where available.

    Adapts per platform:
    - Gemini: entry.grab_focus() → insert_text() → send.do_action(0)
    - ChatGPT: clipboard paste → send.do_action(0) ("Send prompt" button)
    - Grok/Perplexity: clipboard paste → Enter (no AT-SPI send button)

    Returns True if send was triggered.
    """
    firefox = atspi.find_firefox()
    if not firefox:
        return False
    doc = atspi.get_platform_document(firefox, platform)
    if not doc:
        return False

    # Walk tree for entry and send button
    entry_node = None
    send_node = None

    def find_nodes(obj, depth=0):
        nonlocal entry_node, send_node
        if depth > 20:
            return
        try:
            name = (obj.get_name() or '').lower()
            role = obj.get_role_name() or ''
            if role == 'entry' and not entry_node:
                entry_node = obj
            if 'button' in role and 'send' in name and not send_node:
                send_node = obj
            for i in range(obj.get_child_count()):
                child = obj.get_child_at_index(i)
                if child:
                    find_nodes(child, depth + 1)
        except Exception:
            pass

    find_nodes(doc)

    # Step 1: Get text into the input field via clipboard paste
    # Clipboard paste is the most reliable method across ALL platforms.
    # AT-SPI insert_text has length limits and doesn't trigger JS state on some platforms.

    # Focus the input: prefer grab_focus (works for Gemini, Perplexity),
    # fall back to coordinate click
    if entry_node:
        try:
            entry_node.grab_focus()
            time.sleep(0.3)
        except Exception:
            pass
    else:
        input_coords = _find_input_area(platform)
        if input_coords:
            inp.click_at(input_coords[0], input_coords[1])
            time.sleep(0.5)
        else:
            return False

    if not _clipboard_paste(message):
        return False
    time.sleep(0.5)

    # Step 2: Send the message
    # Always press Enter - most reliable across all platforms.
    # do_action(0) on send buttons can silently fail when React state
    # doesn't register clipboard paste as content (button stays disabled).
    inp.press_key("Return")
    return True


def _send_to_platform(platform: str, message: str) -> bool:
    """Send a message to a platform (non-blocking - doesn't wait for response).

    Switches to the platform tab, pastes the message, and sends.
    Returns True if send was triggered.
    """
    shortcut = TAB_SHORTCUTS.get(platform)
    if not shortcut:
        return False

    try:
        inp.focus_firefox()
        time.sleep(0.3)
        inp.press_key(shortcut)
        time.sleep(1.0)

        # Scroll to bottom first
        inp.press_key("End")
        time.sleep(0.5)

        # Try AT-SPI native send first
        sent = _send_via_atspi(platform, message)
        if sent:
            log.info(f"  [{platform}] Sent via AT-SPI")
            return True

        # Fallback: click input, paste, Enter
        input_coords = _find_input_area(platform)
        if not input_coords:
            log.error(f"[{platform}] Could not find input area")
            return False

        input_x, input_y = input_coords
        inp.click_at(input_x, input_y)
        time.sleep(0.5)

        if not _clipboard_paste(message):
            log.error(f"[{platform}] Failed to paste message")
            return False

        time.sleep(0.5)
        inp.press_key("Return")
        log.info(f"  [{platform}] Sent via clipboard paste")
        return True

    except Exception as e:
        log.error(f"[{platform}] send error: {e}", exc_info=True)
        return False


def send_and_wait(platform: str, message: str, max_wait: int) -> Optional[str]:
    """Send a message to a platform and wait for the response (sequential).

    Uses stop-button state machine for response detection.
    Returns response text or None.
    """
    if not _send_to_platform(platform, message):
        return None
    return _wait_for_response(platform, max_wait)


def send_batch_and_wait(
    batch: list,  # [(platform, message), ...]
    max_wait: int,
) -> Dict[str, Optional[str]]:
    """Send messages to multiple platforms in parallel, wait for all responses.

    Phase 1: Tab-switch through platforms, paste and send to each (~1s per platform).
    Phase 2: Poll all platforms for stop-button state changes. Extract immediately
             when a platform completes (while still on its tab).

    Returns {platform: response_text_or_None}.
    """
    results = {}
    pending = {}  # platform -> {'sent_at': float, 'stop_seen': bool}

    # Phase 1: Send to all platforms (fast - paste and go)
    for platform, message in batch:
        if _send_to_platform(platform, message):
            pending[platform] = {'sent_at': time.time(), 'stop_seen': False}
        else:
            results[platform] = None
        time.sleep(1.0)  # Brief pause between tab switches

    if not pending:
        return results

    log.info(f"  Batch sent to {len(pending)} platforms, polling...")

    # Phase 2: Poll all platforms for stop-button completion
    # Each cycle: check every pending platform (~1.5s per platform for tab switch + scan)
    # No fixed sleep at top - tab switching provides natural pacing
    time.sleep(WAIT_AFTER_SEND)  # Initial delay before first poll
    poll_start = time.time()
    no_stop_warned = set()

    while pending and (time.time() - poll_start) < max_wait:
        completed_this_cycle = []

        for platform in list(pending.keys()):
            elapsed = time.time() - pending[platform]['sent_at']

            # Switch to platform tab
            shortcut = TAB_SHORTCUTS.get(platform)
            if not shortcut:
                continue
            inp.focus_firefox()
            time.sleep(0.2)
            inp.press_key(shortcut)
            time.sleep(0.5)

            firefox = atspi.find_firefox()
            if not firefox:
                continue
            doc = atspi.get_platform_document(firefox, platform)
            if not doc:
                continue

            stop_btn = _find_stop_button(platform, doc)

            if stop_btn:
                if not pending[platform]['stop_seen']:
                    pending[platform]['stop_seen'] = True
                    log.info(f"  [{platform}] Stop button appeared - generating")
            else:
                if pending[platform]['stop_seen']:
                    # Response complete - extract NOW while on this tab
                    log.info(f"  [{platform}] Stop button disappeared - complete ({elapsed:.0f}s)")
                    time.sleep(2)  # DOM settle time
                    response = _extract_latest_response(platform)
                    if response and len(response) < 50:
                        time.sleep(3)
                        response = _extract_latest_response(platform)
                    results[platform] = response
                    completed_this_cycle.append(platform)
                    continue

            # Warning if no stop button seen after threshold
            if not pending[platform]['stop_seen'] and elapsed > NO_STOP_WARNING:
                if platform not in no_stop_warned:
                    log.warning(f"  [{platform}] No stop button after {NO_STOP_WARNING}s - trying extraction")
                    no_stop_warned.add(platform)
                    # Fast response detection: response may have completed between polls
                    # (stop button appeared and disappeared before we checked)
                    time.sleep(1)
                    response = _extract_latest_response(platform)
                    if response and len(response) > 100:
                        log.info(f"  [{platform}] Found response via fallback extraction ({len(response)} chars)")
                        results[platform] = response
                        completed_this_cycle.append(platform)
                        continue

            # Per-platform timeout
            if elapsed > max_wait:
                log.warning(f"  [{platform}] Timeout ({max_wait}s, stop_seen={pending[platform]['stop_seen']})")
                response = _extract_latest_response(platform)
                results[platform] = response if response and len(response) > 50 else None
                completed_this_cycle.append(platform)

        # Remove completed platforms
        for p in completed_this_cycle:
            if p in pending:
                del pending[p]

        # Brief pause between polling cycles
        if pending:
            time.sleep(2)

    # Global timeout - try one last extraction for each
    for platform in list(pending.keys()):
        log.warning(f"  [{platform}] Global timeout ({max_wait}s)")
        shortcut = TAB_SHORTCUTS.get(platform)
        if shortcut:
            inp.focus_firefox()
            time.sleep(0.2)
            inp.press_key(shortcut)
            time.sleep(0.5)
            response = _extract_latest_response(platform)
            results[platform] = response if response and len(response) > 50 else None
        else:
            results[platform] = None

    return results


def _open_new_chat(platform: str) -> bool:
    """Navigate to platform's base URL to start a fresh conversation."""
    from core.platforms import BASE_URLS

    url = BASE_URLS.get(platform)
    if not url:
        return False

    shortcut = TAB_SHORTCUTS.get(platform)
    if not shortcut:
        return False

    # Switch to platform tab
    inp.focus_firefox()
    time.sleep(0.3)
    inp.press_key(shortcut)
    time.sleep(0.5)

    # Navigate to base URL
    inp.press_key("ctrl+l")
    time.sleep(0.3)
    inp.press_key("ctrl+a")
    time.sleep(0.2)

    # Type URL via clipboard (faster and more reliable than xdotool type)
    proc = subprocess.Popen(
        ["xclip", "-selection", "clipboard"],
        stdin=subprocess.PIPE,
        env=os.environ,
    )
    proc.stdin.write(url.encode("utf-8"))
    proc.stdin.close()
    proc.wait(timeout=5)
    time.sleep(0.2)
    inp.press_key("ctrl+v")
    time.sleep(0.3)
    inp.press_key("Return")
    time.sleep(5.0)  # Wait for page load

    log.info(f"[{platform}] Opened new chat at {url}")
    return True


def initialize_platform(platform: str, use_existing: bool = True) -> bool:
    """
    Send the session init prompt to a platform.

    If use_existing=True, sends to whatever conversation is currently open
    on the platform tab (reuses existing sessions). Otherwise opens new chat.

    Returns True if the platform acknowledged with "Ready".
    """
    if use_existing:
        # Just switch to the existing tab
        shortcut = TAB_SHORTCUTS.get(platform)
        if not shortcut:
            return False
        inp.focus_firefox()
        time.sleep(0.3)
        inp.press_key(shortcut)
        time.sleep(1.0)
        log.info(f"[{platform}] Using existing session")
    else:
        if not _open_new_chat(platform):
            log.warning(f"[{platform}] Failed to open new chat")
            return False
        time.sleep(2)

    log.info(f"[{platform}] Sending session init prompt...")
    init_messages = build_session_init(platform)

    # Init is split into multiple messages for reliability
    # (some platforms truncate clipboard paste over ~10K chars)
    response = None
    for msg_idx, (msg_text, wait_for_response) in enumerate(init_messages):
        log.debug(f"  [{platform}] Init message {msg_idx+1}/{len(init_messages)} ({len(msg_text):,} chars)")
        if wait_for_response:
            response = send_and_wait(platform, msg_text, MAX_WAIT_INIT)
        else:
            # Send without waiting - just paste, send, brief pause
            _send_to_platform(platform, msg_text)
            time.sleep(15)  # Give the AI time to read before sending next part

    if response:
        # Reject bogus responses (URLs, very short, etc.)
        lower = response.lower().strip()
        if len(lower) < 10 or lower.startswith("http") or lower.startswith("[http"):
            log.warning(f"[{platform}] Init got bogus response ({len(response)} chars): {response[:80]}")
            if use_existing:
                log.info(f"[{platform}] Retrying with new chat...")
                return initialize_platform(platform, use_existing=False)
            return False

        # Check for confused response (session in bad state)
        confused_phrases = ["cut off", "what did you mean", "i don't see", "could you resend",
                           "didn't come through", "try again", "incomplete"]
        if any(p in lower for p in confused_phrases):
            log.warning(f"[{platform}] Session confused: {response[:80]}")
            if use_existing:
                log.info(f"[{platform}] Opening new chat...")
                return initialize_platform(platform, use_existing=False)
            return False

        # Check for acknowledgment (flexible matching)
        if any(w in lower for w in ["ready", "understood", "let's", "i understand",
                                     "confirm", "got it", "proceed", "analysis"]):
            log.info(f"[{platform}] Initialized successfully ({len(response)} chars)")
            return True
        else:
            # Even without explicit "ready", if they responded substantively, accept
            log.info(f"[{platform}] Got response (may not be explicit ack): {response[:100]}...")
            return True
    else:
        log.warning(f"[{platform}] No response to init prompt")
        if use_existing:
            log.info(f"[{platform}] Retrying with new chat...")
            return initialize_platform(platform, use_existing=False)
        return False


# ============================================================================
# Response Parsing
# ============================================================================

def _repair_truncated_json(text: str) -> Optional[str]:
    """Attempt to repair JSON that was truncated mid-response.

    Common case: response cut off during copy extraction, leaving an
    incomplete motif entry. We try to close the open brackets/braces.
    """
    # Find the JSON start
    start = text.find("{")
    if start < 0:
        return None

    json_str = text[start:]

    # Count open brackets/braces
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")

    if open_braces <= 0 and open_brackets <= 0:
        return None  # Not actually truncated

    # Find the last complete motif entry (ends with "}")
    # Truncate the incomplete one and close everything
    last_complete = json_str.rfind("}")
    if last_complete < 0:
        return None

    # Check if we have at least rosetta_summary and one motif
    candidate = json_str[:last_complete + 1]
    if '"rosetta_summary"' not in candidate:
        return None

    # Close open structures
    repaired = candidate
    # Re-count after truncation
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    # Close brackets first, then braces
    repaired += "]" * open_brackets
    repaired += "}" * open_braces

    return repaired


def parse_motif_response(response_text: str) -> Optional[Dict]:
    """
    Parse structured JSON from a Family Chat response.

    Handles multiple response styles:
    - JSON in ```json code block
    - Raw JSON object
    - JSON with surrounding commentary
    """
    if not response_text:
        return None

    # Strategy 1: Find ```json ... ``` block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find ``` ... ``` block (no json label)
    code_match = re.search(r"```\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost { ... } containing rosetta_summary
    brace_match = re.search(
        r'\{[^{}]*"rosetta_summary"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        response_text, re.DOTALL,
    )
    if not brace_match:
        # Try a more aggressive match
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start >= 0 and end > start:
            brace_match = type("Match", (), {"group": lambda self, *a: response_text[start:end+1]})()

    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try the entire response as JSON
    try:
        data = json.loads(response_text.strip())
        if _validate_response(data):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 5: Repair truncated JSON (response cut off during extraction)
    repaired = _repair_truncated_json(response_text)
    if repaired:
        try:
            data = json.loads(repaired)
            if _validate_response(data):
                log.info(f"  Repaired truncated JSON successfully")
                return data
        except json.JSONDecodeError:
            pass

    log.warning(f"Failed to parse JSON from response ({len(response_text)} chars)")
    log.debug(f"Response preview: {response_text[:300]}...")
    return None


def _validate_response(data: Dict) -> bool:
    """Check that parsed response has required structure."""
    if not isinstance(data, dict):
        return False
    if "rosetta_summary" not in data:
        return False
    if "motifs" not in data or not isinstance(data["motifs"], list):
        return False
    return True


# ============================================================================
# HMM Storage
# ============================================================================

def store_family_motifs(
    content_hash: str,
    content_text: str,
    parsed_response: Dict,
    source_platform: str,
    source_info: str,
    neo4j: HMMNeo4jStore,
    redis_store: HMMRedisStore,
    event_log: EventLog,
    gate: GateB,
):
    """Store motif assignments from Family Chat response into HMM."""
    rosetta_summary = parsed_response.get("rosetta_summary", "")
    motif_data = parsed_response.get("motifs", [])
    new_suggestions = parsed_response.get("new_motif_suggestions", [])
    meta = parsed_response.get("meta", {})

    assignments = []
    for m in motif_data:
        motif_id = m.get("motif_id", "")
        if motif_id not in V0_MOTIFS:
            log.debug(f"  Skipping unknown motif: {motif_id}")
            continue

        amp = min(1.0, max(0.0, float(m.get("amp", 0.5))))
        confidence = min(1.0, max(0.0, float(m.get("confidence", 0.7))))

        # Enforce minimum thresholds from prompt instructions
        if amp < 0.10 or confidence < 0.30:
            continue

        # Use the motif's band as phase (fast/mid/slow), not the platform name
        motif_def = V0_MOTIFS.get(motif_id)
        band = motif_def.band if motif_def else "mid"

        assignments.append(MotifAssignment(
            motif_id=motif_id,
            amp=round(amp, 4),
            phase=band,
            confidence=round(confidence, 4),
            source="declared",
            dictionary_version=DICTIONARY_VERSION,
        ))

    if not assignments:
        log.warning(f"  No valid motif assignments for {content_hash[:12]}")
        return 0

    # Gate-B check
    gate_result = gate.evaluate(assignments)

    # Store in Neo4j graph
    t_id = content_hash
    # Build labels: include all responding platforms if available
    platforms_list = parsed_response.get("_platforms", [source_platform])
    labels = list(platforms_list) + [meta.get("dominant_tone", "")]
    try:
        neo4j.upsert_artifact(
            artifact_id=source_info,
            path=source_info,
            size_bytes=len(content_text.encode('utf-8')),
            content_type="text/plain",
            labels=labels,
        )
        neo4j.upsert_tile(
            tile_id=t_id,
            artifact_id=source_info,
            index=0,
            start_char=0,
            end_char=len(content_text),
            estimated_tokens=len(content_text) // 4,
        )
        neo4j.link_tile_motifs_batch(t_id, assignments, model_id=source_platform)
        # Link HMMTile to existing Document node via content_hash
        with neo4j.driver.session() as session:
            session.run("""
                MATCH (t:HMMTile {tile_id: $tile_id})
                MATCH (d:Document {content_hash: $content_hash})
                MERGE (t)-[:ANNOTATES]->(d)
            """, tile_id=t_id, content_hash=content_hash)
        log.debug(f"  Neo4j: artifact + tile + {len(assignments)} EXPRESSES + Document link")
    except Exception as e:
        log.error(f"  Neo4j write failed (continuing with Redis): {e}")

    # Store in Redis
    for a in assignments:
        redis_store.inv_add(a.motif_id, t_id)
        band_k = {"fast": 0, "mid": 1, "slow": 2}.get(
            V0_MOTIFS[a.motif_id].band, 0
        )
        redis_store.field_update(band_k, a.motif_id, a.amp)

    redis_store.tile_cache_put(t_id, assignments)

    # Emit event with full context
    gate_snapshot = GateSnapshot(
        phi=gate_result.phi, trust=gate_result.trust, flags=gate_result.flags
    )
    # Include per-motif agreement info if available (from consensus merge)
    motif_detail = []
    for a in assignments:
        detail = {"id": a.motif_id, "amp": a.amp}
        # Find agreement info from original parsed data
        for m in motif_data:
            if m.get("motif_id") == a.motif_id and "agreement" in m:
                detail["agreement"] = m["agreement"]
                detail["platforms"] = m.get("platforms", [])
                break
        motif_detail.append(detail)

    event_log.emit(
        "MOTIFS_ASSIGNED",
        refs={"tile_id": t_id, "source_platform": source_platform},
        payload={
            "count": len(assignments),
            "motifs": [a.motif_id for a in assignments],
            "motif_detail": motif_detail,
            "rosetta_summary": rosetta_summary,
            "source": "family_chat",
            "family_member": source_platform,
            "platforms": platforms_list,
            "source_info": source_info,
            "dominant_tone": meta.get("dominant_tone", ""),
            "family_relevance": meta.get("family_relevance", ""),
        },
        gate=gate_snapshot,
    )

    # Log new motif suggestions separately
    if new_suggestions:
        event_log.emit(
            "MOTIFS_ASSIGNED",
            refs={"tile_id": t_id},
            payload={
                "new_motif_suggestions": new_suggestions[:5],
                "source_platform": source_platform,
            },
        )

    log.info(
        f"  Stored {len(assignments)} motifs "
        f"(gate: phi={gate_result.phi:.2f} trust={gate_result.trust:.2f} "
        f"{'SLOW_OK' if gate_result.slow_field_eligible else 'fast/mid only'})"
    )
    return len(assignments)


def _merge_platform_responses(platform_responses: Dict[str, Dict]) -> Optional[Dict]:
    """Merge motif extractions across multiple platforms into consensus.

    Cross-platform agreement is the audit. If 3/4 platforms identify a motif,
    confidence gets boosted. Amplitudes are averaged across observers.

    Takes {platform: parsed_response} dict.
    Returns merged response in same format as individual responses.
    """
    if not platform_responses:
        return None

    # Single platform - use directly
    if len(platform_responses) == 1:
        platform, resp = next(iter(platform_responses.items()))
        resp["_platforms"] = [platform]
        return resp

    n_platforms = len(platform_responses)

    # Collect rosetta summaries - use longest (most detailed)
    summaries = []
    for p, resp in platform_responses.items():
        s = resp.get("rosetta_summary", "")
        if s:
            summaries.append(s)
    primary_summary = max(summaries, key=len) if summaries else ""

    # Collect motif observations across platforms
    motif_obs = {}  # motif_id -> [(amp, confidence, platform), ...]
    for platform, resp in platform_responses.items():
        for m in resp.get("motifs", []):
            mid = m.get("motif_id", "")
            if not mid or mid not in V0_MOTIFS:
                continue
            if mid not in motif_obs:
                motif_obs[mid] = []
            motif_obs[mid].append({
                "amp": float(m.get("amp", 0.5)),
                "confidence": float(m.get("confidence", 0.7)),
                "platform": platform,
            })

    # Merge: mean amplitude, confidence boosted by agreement
    merged_motifs = []
    for motif_id, observations in motif_obs.items():
        n_agree = len(observations)
        mean_amp = sum(o["amp"] for o in observations) / n_agree
        mean_conf = sum(o["confidence"] for o in observations) / n_agree

        # Agreement boost: each additional platform adds confidence
        # 1 platform = 0, 2 = +0.05, 3 = +0.10, 4 = +0.15
        agreement_boost = (n_agree - 1) * 0.05 if n_agree > 1 else 0
        boosted_conf = min(1.0, mean_conf + agreement_boost)

        merged_motifs.append({
            "motif_id": motif_id,
            "amp": round(mean_amp, 4),
            "confidence": round(boosted_conf, 4),
            "agreement": n_agree,
            "platforms": [o["platform"] for o in observations],
        })

    merged_motifs.sort(key=lambda m: m["amp"], reverse=True)

    # Merge meta from all platforms
    merged_meta = {}
    for p, resp in platform_responses.items():
        meta = resp.get("meta", {})
        for k, v in meta.items():
            if k not in merged_meta:
                merged_meta[k] = v

    # Collect new motif suggestions
    all_suggestions = []
    seen_suggestions = set()
    for p, resp in platform_responses.items():
        for s in resp.get("new_motif_suggestions", []):
            name = s.get("name", "") if isinstance(s, dict) else str(s)
            if name and name not in seen_suggestions:
                all_suggestions.append(s)
                seen_suggestions.add(name)

    platforms_list = list(platform_responses.keys())
    log.info(
        f"  Merged {len(merged_motifs)} unique motifs from {n_platforms} platforms "
        f"({', '.join(platforms_list)})"
    )

    return {
        "rosetta_summary": primary_summary,
        "motifs": merged_motifs,
        "meta": merged_meta,
        "new_motif_suggestions": all_suggestions[:5],
        "_platforms": platforms_list,
    }


def save_raw_response(content_hash: str, platform: str, response: str):
    """Save raw response for debugging/replay."""
    os.makedirs(RESPONSE_LOG_DIR, exist_ok=True)
    path = os.path.join(RESPONSE_LOG_DIR, f"{content_hash[:16]}_{platform}.json")
    with open(path, "w") as f:
        json.dump({
            "content_hash": content_hash,
            "platform": platform,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response": response,
        }, f, indent=2)


# ============================================================================
# Content Iterators
# ============================================================================

def iter_corpus_content(retrieval: ISMARetrieval) -> List[Tuple[str, str, str]]:
    """Iterate corpus docs: (content_hash, source_info) without loading full text.

    Full text is loaded lazily when needed, to avoid 400MB+ memory spike.
    Returns (content_hash, None, source_info) — caller loads text on demand.
    """
    items = []
    for doc in retrieval.iter_all_documents(batch_size=100):
        source = f"corpus:{doc.filename}:layer={doc.layer}:priority={doc.priority}"
        items.append((doc.content_hash, None, source))

    items.sort(key=lambda x: -_layer_priority(x[2]))
    return items


def iter_transcript_content(retrieval: ISMARetrieval) -> List[Tuple[str, str, str]]:
    """Iterate transcript sessions: (content_hash, None, source_info).

    Returns session_id as content_hash stand-in for lazy loading.
    Full text loaded on demand via retrieval.get_session_full_text().
    """
    items = []
    for session in retrieval.iter_all_sessions(batch_size=100):
        session_id = session.get("session_id", "")
        platform = session.get("platform", "unknown")
        source = f"transcript:{platform}:{session_id}"
        items.append((session_id, None, source))
    return items


def _load_item_text(content_hash: str, source_info: str, retrieval: ISMARetrieval) -> Optional[str]:
    """Load text on demand for a corpus document or transcript session."""
    if source_info.startswith("transcript:"):
        # content_hash is actually session_id for transcripts
        text = retrieval.get_session_full_text(content_hash)
        if text and len(text) >= MIN_CONTENT_CHARS:
            return text
        return None

    # Corpus document: try file first, then Weaviate tiles
    doc = retrieval.get_document(content_hash)
    if doc and doc.all_paths:
        paths = doc.all_paths if isinstance(doc.all_paths, list) else [doc.all_paths]
        for path in paths:
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        return f.read()
                except Exception:
                    continue
    # Fallback: reconstruct from Weaviate tiles
    text = retrieval.get_full_text(content_hash)
    return text if text else None


def _layer_priority(source_info: str) -> float:
    """Extract priority for sorting. Uses explicit priority= value if present."""
    # Try explicit priority from source_info
    m = re.search(r"priority=([0-9.]+)", source_info)
    if m:
        return float(m.group(1))
    # Fallback to layer-based priority
    if "kernel" in source_info:
        return 1.0
    if "layer=0" in source_info:
        return 0.9
    if "chewy" in source_info:
        return 0.85
    if "layer=1" in source_info:
        return 0.7
    if "layer=2" in source_info:
        return 0.5
    return 0.3


# ============================================================================
# State Management
# ============================================================================

def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "processed_hashes": [],
        "platform_index": 0,
        "total_processed": 0,
        "total_motifs_stored": 0,
        "initialized_platforms": [],
        "started_at": None,
    }


def save_state(state: Dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# Main Processing Loop
# ============================================================================

def run_processor(mode: str, dry_run: bool = False, limit: int = 0, no_audit: bool = False):
    """Main loop: init platforms, iterate content, extract motifs, store in HMM."""
    log.info(f"{'='*60}")
    log.info(f"HMM Family Processor starting (mode={mode})")
    log.info(f"{'='*60}")

    # Initialize stores
    retrieval = ISMARetrieval()
    neo4j = HMMNeo4jStore()
    redis_store = HMMRedisStore()
    event_log = EventLog(EVENT_LOG_PATH)
    gate = GateB(redis_store)

    neo4j.seed_motifs(V0_MOTIFS)

    # Load state
    state = load_state()
    processed: Set[str] = set(state.get("processed_hashes", []))
    platform_idx = state.get("platform_index", 0)
    total_motifs = state.get("total_motifs_stored", 0)

    if not state.get("started_at"):
        state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Gather content
    log.info("Gathering content to process...")
    items = []

    if mode in ("corpus", "all"):
        corpus_items = iter_corpus_content(retrieval)
        log.info(f"  Corpus: {len(corpus_items)} documents")
        items.extend(corpus_items)

    if mode in ("transcripts", "all"):
        transcript_items = iter_transcript_content(retrieval)
        log.info(f"  Transcripts: {len(transcript_items)} sessions")
        items.extend(transcript_items)

    # Filter already processed
    items = [(h, t, s) for h, t, s in items if h not in processed]
    log.info(f"  Remaining after filtering: {len(items)} items")

    if limit > 0:
        items = items[:limit]
        log.info(f"  Limited to: {limit} items")

    if dry_run:
        log.info("\nDRY RUN - would process:")
        sampled_chars = 0
        for i, (h, t, s) in enumerate(items[:30]):
            if t is None:
                t = _load_item_text(h, s, retrieval)
            if t:
                chunks = split_content(t)
                sampled_chars += len(t)
                log.info(f"  [{i}] {s} ({len(t):,} chars, {len(chunks)} chunk{'s' if len(chunks) > 1 else ''}, hash={h[:12]})")
            else:
                log.info(f"  [{i}] {s} (text unavailable, hash={h[:12]})")
        if len(items) > 30:
            log.info(f"  ... and {len(items) - 30} more")
        avg_chars = sampled_chars // min(len(items), 30) if items else 0
        est_total_chars = avg_chars * len(items)
        est_hours = len(items) * 1.5 / 60  # ~1.5 min per item average
        log.info(f"\n  Total: {len(items)} items, ~{est_total_chars:,} est chars")
        log.info(f"  Estimated time: ~{est_hours:.1f} hours across {len(PLATFORMS)} platforms")
        return

    # Check platform availability
    log.info("\nChecking platform availability...")
    platform_states: Dict[str, PlatformState] = {}
    for p in PLATFORMS:
        available = check_platform_available(p)
        platform_states[p] = PlatformState(name=p, available=available)
        log.info(f"  {p}: {'OK' if available else 'UNAVAILABLE'}")

    available_platforms = [p for p in PLATFORMS if platform_states[p].available]
    if not available_platforms:
        log.error("No platforms available! Exiting.")
        return

    # Initialize platforms (send context prompt)
    already_init = set(state.get("initialized_platforms", []))
    log.info(f"\nInitializing platforms...")
    for p in available_platforms:
        if p in already_init:
            platform_states[p].initialized = True
            log.info(f"  {p}: Already initialized (from state)")
            continue

        success = initialize_platform(p)
        if success:
            platform_states[p].initialized = True
            already_init.add(p)
            state["initialized_platforms"] = list(already_init)
            save_state(state)
        else:
            log.warning(f"  {p}: Init failed, will skip")
            platform_states[p].available = False

        time.sleep(BETWEEN_PLATFORMS)

    available_platforms = [
        p for p in PLATFORMS
        if platform_states[p].available and platform_states[p].initialized
    ]
    if not available_platforms:
        log.error("No platforms initialized! Exiting.")
        return

    log.info(f"\n{'='*60}")
    log.info(f"Processing {len(items)} items across {len(available_platforms)} platforms")
    log.info(f"Platforms: {', '.join(available_platforms)}")
    log.info(f"Mode: Parallel all-platform (equal balance, cross-platform consensus)")
    log.info(f"{'='*60}\n")

    start_time = time.time()
    items_done = 0
    items_failed = 0
    total_platform_responses = 0

    for i, (content_hash, full_text, source_info) in enumerate(items):
        # Load text on demand if needed (lazy loading)
        if full_text is None:
            full_text = _load_item_text(content_hash, source_info, retrieval)
            if not full_text:
                log.warning(f"[{i+1}/{len(items)}] {source_info} -> SKIPPED (text unavailable)")
                items_failed += 1
                processed.add(content_hash)
                continue
            # For transcripts, compute real content hash from loaded text
            if source_info.startswith("transcript:"):
                content_hash = hashlib.sha256(full_text.encode()).hexdigest()[:16]

        # Split content if needed
        chunks = split_content(full_text)

        elapsed = time.time() - start_time
        rate = items_done / (elapsed / 60) if elapsed > 60 else 0
        log.info(
            f"[{i+1}/{len(items)}] {source_info} "
            f"({len(full_text):,} chars, {len(chunks)} chunk{'s' if len(chunks) > 1 else ''}) "
            f"-> ALL {len(available_platforms)} platforms "
            f"[{elapsed/60:.0f}m, {rate:.1f}/min]"
        )

        chunk_motifs = 0
        chunk_failed = False

        for chunk_text, chunk_idx, total_chunks in chunks:
            prompt = build_content_prompt(chunk_text, source_info, chunk_idx, total_chunks)

            # === PARALLEL SEND TO ALL PLATFORMS ===
            batch = [(p, prompt) for p in available_platforms]
            responses = send_batch_and_wait(batch, MAX_WAIT_CONTENT)

            chunk_hash = f"{content_hash}_{chunk_idx}" if total_chunks > 1 else content_hash

            # Parse responses from each platform
            platform_parsed = {}
            for platform, response in responses.items():
                if not response:
                    platform_states[platform].consecutive_errors += 1
                    if platform_states[platform].consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        log.error(f"  [{platform}] Disabled after {MAX_CONSECUTIVE_ERRORS} consecutive errors")
                        platform_states[platform].available = False
                    continue

                # Save raw response for provenance
                save_raw_response(f"{chunk_hash}_{platform}", platform, response)
                parsed = parse_motif_response(response)

                if parsed:
                    platform_parsed[platform] = parsed
                    platform_states[platform].messages_sent += 1
                    platform_states[platform].consecutive_errors = 0
                    platform_states[platform].parse_failures = 0
                    total_platform_responses += 1
                else:
                    platform_states[platform].parse_failures += 1
                    log.debug(f"  [{platform}] Parse failed")

                    # Too many parse failures -> re-initialize
                    if platform_states[platform].parse_failures >= MAX_PARSE_FAILURES:
                        log.warning(f"  [{platform}] Too many parse failures, re-initializing...")
                        platform_states[platform].initialized = False
                        if initialize_platform(platform):
                            platform_states[platform].initialized = True
                            platform_states[platform].parse_failures = 0

            # === MERGE ACROSS PLATFORMS ===
            if platform_parsed:
                consensus = _merge_platform_responses(platform_parsed)
                if consensus:
                    # Source platform = "family" (all equal, consensus view)
                    responding_platforms = consensus.get("_platforms", [])
                    source_label = f"family_{len(responding_platforms)}"

                    n_stored = store_family_motifs(
                        chunk_hash, chunk_text, consensus, source_label, source_info,
                        neo4j, redis_store, event_log, gate,
                    )
                    chunk_motifs += n_stored
            else:
                # ALL platforms failed to parse - Tier 0 fallback
                log.warning(f"  No valid responses, Tier 0 fallback")
                assignments = assign_motifs(chunk_text)
                if assignments:
                    for a in assignments:
                        redis_store.inv_add(a.motif_id, chunk_hash)
                        band_k = {"fast": 0, "mid": 1, "slow": 2}.get(
                            V0_MOTIFS[a.motif_id].band, 0
                        )
                        redis_store.field_update(band_k, a.motif_id, a.amp)
                    redis_store.tile_cache_put(chunk_hash, assignments)
                    chunk_motifs += len(assignments)
                    log.info(f"  Tier 0: {len(assignments)} motifs")
                chunk_failed = True

            # Update available platforms
            available_platforms = [
                p for p in PLATFORMS
                if platform_states[p].available and platform_states[p].initialized
            ]
            if not available_platforms:
                log.error("All platforms failed! Stopping.")
                break

            # Wait between chunks
            if total_chunks > 1 and chunk_idx < total_chunks - 1:
                time.sleep(BETWEEN_MESSAGES)

        if not available_platforms:
            break

        if not chunk_failed:
            items_done += 1
            total_motifs += chunk_motifs
        else:
            items_failed += 1

        # Update state
        processed.add(content_hash)
        state["processed_hashes"] = list(processed)
        state["total_processed"] = len(processed)
        state["total_motifs_stored"] = total_motifs

        save_state(state)  # Save after every item

        # Wait before next item
        time.sleep(BETWEEN_MESSAGES)

    # Final save
    save_state(state)

    # Summary
    total_time = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"Processing complete")
    log.info(f"{'='*60}")
    log.info(f"  Items processed: {items_done}")
    log.info(f"  Items failed: {items_failed}")
    log.info(f"  Total motifs stored: {total_motifs}")
    log.info(f"  Total platform responses: {total_platform_responses}")
    log.info(f"  Time: {total_time/60:.1f} minutes")
    log.info(f"  Rate: {items_done / max(total_time/60, 0.1):.1f} items/min")
    log.info(f"\nPlatform stats:")
    for p in PLATFORMS:
        ps = platform_states.get(p)
        if ps:
            log.info(f"  {p}: {ps.messages_sent} responses, {ps.consecutive_errors} errors, "
                     f"{'ACTIVE' if ps.available else 'DISABLED'}")

    redis_stats = redis_store.stats()
    neo4j_counts = neo4j.count_nodes()
    log.info(f"\nHMM stats:")
    log.info(f"  Redis: {redis_stats}")
    log.info(f"  Neo4j: {neo4j_counts}")
    log.info(f"  Event log: {event_log.count()} events")

    neo4j.close()


def check_platforms():
    """Check which platforms are available."""
    log.info("Checking platform availability...")
    for p in PLATFORMS:
        available = check_platform_available(p)
        log.info(f"  {p}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Family Processor")
    parser.add_argument(
        "--mode", choices=["corpus", "transcripts", "all"],
        default="corpus", help="Content to process",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check platforms only")
    parser.add_argument("--limit", type=int, default=0, help="Max items")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--platform", type=str, default="",
                        help="Use only this platform (e.g. chatgpt)")
    parser.add_argument("--no-audit", action="store_true",
                        help="Skip audit pass (faster, less quality checking)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated platforms to exclude (e.g. chatgpt)")
    args = parser.parse_args()

    if args.platform:
        PLATFORMS[:] = [args.platform]

    if args.exclude:
        excluded = [p.strip() for p in args.exclude.split(",")]
        PLATFORMS[:] = [p for p in PLATFORMS if p not in excluded]
        log.info(f"Excluded: {excluded}, remaining: {PLATFORMS}")

    if args.check:
        check_platforms()
        sys.exit(0)

    if args.reset_state:
        for f in [STATE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        log.info("State reset.")

    try:
        run_processor(args.mode, dry_run=args.dry_run, limit=args.limit,
                      no_audit=args.no_audit)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception:
        log.error("Fatal error in processor:", exc_info=True)
        sys.exit(1)
