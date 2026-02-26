#!/usr/bin/env python3
"""
HMM Direct Pipeline v2 — AT-SPI automation with MCP-grade reliability.

Architecture:
  - Uses taeys-hands core modules directly (no MCP round-trips)
  - Spawns monitor daemon for response detection (same as MCP tools)
  - Validates every step: page load, file dialog, send, extraction quality
  - Claude Code runs this via Bash and checks JSON output between cycles

Modes:
  cycle-once  — Send to all 5, wait, extract all. Exit with JSON summary.
  rolling     — Continuous interleaved send/extract. Ctrl+C to stop.
  test <plat> — Test single platform end-to-end.
  status      — Check generating/copy state per platform.
"""

import sys
import os
import time
import json
import subprocess
import logging
import signal
import uuid

import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

# Add taeys-hands to path
TAEY_PATH = os.path.expanduser("~/taeys-hands")
sys.path.insert(0, TAEY_PATH)

from core import input as inp
from core import clipboard as clip
from core import atspi
from core.tree import (
    find_elements, filter_useful_elements,
    find_copy_buttons, find_dropdown_menus
)

# Marionette helper for focus-free mode (Thor)
try:
    from marionette_helper import MarionetteHelper, is_marionette_available
except ImportError:
    MarionetteHelper = None
    is_marionette_available = lambda: False

# ============================================================================
# Logging
# ============================================================================

LOG_FILE = "/tmp/hmm_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ]
)
log = logging.getLogger("hmm_pipeline")


def log_event(event: str, platform: str = "", **kwargs):
    entry = {"ts": time.time(), "event": event, "platform": platform, **kwargs}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.expanduser("~/hmm/scripts")
DISPLAY = os.environ.get("DISPLAY", ":1")
ATSPI_DEPTH = 20  # ChatGPT/Grok need depth 16-20

PLATFORMS = ["chatgpt", "claude", "gemini", "grok", "perplexity"]

TAB_KEYS = {
    "chatgpt": "alt+1", "claude": "alt+2", "gemini": "alt+3",
    "grok": "alt+4", "perplexity": "alt+5",
}

# How to open a fresh chat per platform
NEW_CHAT = {
    "chatgpt": ("url", "https://chatgpt.com/"),
    "claude": ("url", "https://claude.ai/new"),
    "gemini": ("url", "https://gemini.google.com/"),
    "grok": ("url", "https://grok.com"),
    "perplexity": ("url", "https://perplexity.ai/"),
}

SEND_PROMPT = "Analyze the attached context package. Follow the INSTRUCTIONS section at the top. Respond ONLY with the JSON format specified."
# Platform-specific prompts
SEND_PROMPT_PERPLEXITY = "Do NOT search the web. Analyze ONLY the attached file. Follow the INSTRUCTIONS section at the top of the attached document. Respond ONLY with the JSON format specified in the file."
# Short prompt for platforms where xdotool type fails (ChatGPT with file attached)
SEND_PROMPT_SHORT = "Go"

# Attach button names per platform (tried in order)
ATTACH_NAMES = {
    "chatgpt": ["add files"],
    "claude": ["toggle menu", "attach", "add content"],
    "gemini": ["open upload file menu", "upload file menu", "open upload"],
    "grok": ["attach"],
    "perplexity": ["attach", "add file"],
}

# Dropdown item names for file upload (after clicking attach button)
UPLOAD_ITEM_NAMES = ["add photos", "add files", "upload file", "upload files"]

# Send button name pattern per platform
SEND_BUTTON_NAMES = {
    "chatgpt": ["send prompt", "send"],
    "claude": ["send", "reply"],
    "gemini": ["send message", "send", "submit"],
    "grok": ["send"],
    "perplexity": ["submit", "send"],
}

# Stop button indicators
STOP_PATTERNS = ["stop generating", "stop response", "stop", "cancel"]

# Monitor daemon path
DAEMON_PATH = os.path.join(TAEY_PATH, "monitor", "daemon.py")


# ============================================================================
# Setup
# ============================================================================

FIREFOX_WID = None  # Set once at startup, used for --window targeted input
MARIONETTE = None   # Set at startup if Marionette is available (focus-free mode)

def _find_firefox_wid():
    """Find Firefox window ID for targeted input (bypasses focus requirement)."""
    env = {"DISPLAY": DISPLAY}
    result = subprocess.run(
        ["xdotool", "search", "--class", "firefox"],
        capture_output=True, text=True, env=env
    )
    wids = [w.strip() for w in result.stdout.strip().split('\n') if w.strip()]
    if not wids:
        result = subprocess.run(
            ["xdotool", "search", "--name", "Firefox"],
            capture_output=True, text=True, env=env
        )
        wids = [w.strip() for w in result.stdout.strip().split('\n') if w.strip()]
    if wids:
        return wids[0]
    return None


def _unminimize_firefox(wid_str: str):
    """Unminimize Firefox if it's in Iconic state (common after remote restart).
    Uses python-xlib to set WM_STATE to Normal since GNOME/Mutter blocks
    xdotool windowactivate and wmctrl for remotely-launched windows."""
    try:
        env = {"DISPLAY": DISPLAY}
        result = subprocess.run(
            ["xprop", "-id", wid_str, "WM_STATE"],
            capture_output=True, text=True, env=env
        )
        if "Iconic" not in result.stdout:
            return  # Already Normal
        log.info(f"Firefox is minimized (Iconic), unminimizing via Xlib...")
        # Use python-xlib to change WM_STATE
        from Xlib import display, X
        d = display.Display(DISPLAY)
        wid_int = int(wid_str)
        w = d.create_resource_object("window", wid_int)
        w.map()
        wm_state = d.intern_atom("WM_STATE")
        w.change_property(wm_state, wm_state, 32, [1, 0])  # 1 = Normal
        w.configure(stack_mode=X.Above)
        d.flush()
        d.sync()
        log.info(f"Firefox unminimized successfully")
    except ImportError:
        log.warning("python-xlib not available, cannot unminimize")
    except Exception as e:
        log.warning(f"Unminimize failed: {e}")


def _patch_input_for_window(wid: str):
    """Monkey-patch inp.press_key and inp.type_text to use --window WID.
    This sends keystrokes directly to Firefox without requiring keyboard focus."""
    env = inp._get_env()

    _orig_press_key = inp.press_key
    _orig_type_text = inp.type_text

    def targeted_press_key(key, timeout=10):
        try:
            result = subprocess.run(
                ['xdotool', 'key', '--window', wid, key],
                env=env, capture_output=True, timeout=timeout,
            )
            if result.returncode != 0:
                log.warning(f"xdotool key --window {wid} {key} failed: {result.stderr.decode()}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.error(f"xdotool key {key} timed out after {timeout}s")
            return False
        except Exception as e:
            log.error(f"xdotool key {key} error: {e}")
            return False

    def targeted_type_text(text, delay_ms=5, timeout=30):
        actual_timeout = timeout + (len(text) * 0.1)
        try:
            result = subprocess.run(
                ['xdotool', 'type', '--window', wid, '--clearmodifiers',
                 '--delay', str(delay_ms), '--', text],
                env=env, capture_output=True, timeout=actual_timeout,
            )
            if result.returncode != 0:
                log.warning(f"Type --window failed: {result.stderr.decode()}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.error(f"Typing timed out (text length: {len(text)})")
            return False
        except Exception as e:
            log.error(f"Type error: {e}")
            return False

    inp.press_key = targeted_press_key
    inp.type_text = targeted_type_text
    log.info(f"Input patched: all keystrokes target window {wid}")


def setup():
    global FIREFOX_WID, MARIONETTE
    os.environ["DISPLAY"] = DISPLAY
    inp.set_display(DISPLAY)
    clip.set_display(DISPLAY)
    Atspi.init()
    log.info(f"Pipeline v2 | DISPLAY={DISPLAY} | depth={ATSPI_DEPTH}")

    # Marionette: used ONLY for navigation (new chat URL), not for text injection or send.
    # AT-SPI handles file attachment, send, response detection, and extraction.
    # Disable with NO_MARIONETTE=1 if Marionette causes session limit issues.
    if not os.environ.get("NO_MARIONETTE") and MarionetteHelper and is_marionette_available():
        m = MarionetteHelper()
        if m.connect():
            tabs = m.discover_tabs()
            if tabs:
                MARIONETTE = m
                log.info(f"MARIONETTE NAV MODE: {len(tabs)} tabs ({', '.join(tabs.keys())})")
            else:
                log.warning("Marionette connected but no tabs found, falling back")
                m.disconnect()

    # Find Firefox window, unminimize if needed, and patch input to target it directly
    FIREFOX_WID = _find_firefox_wid()
    if FIREFOX_WID:
        _unminimize_firefox(FIREFOX_WID)
        _patch_input_for_window(FIREFOX_WID)
        log.info(f"Firefox WID: {FIREFOX_WID}")
    else:
        log.warning("Firefox not found — falling back to focus-based input")
        _focus_firefox()
    # Dismiss any browser permission popups (microphone, notification, etc.)
    _dismiss_popups()


# ============================================================================
# AT-SPI helpers
# ============================================================================

def _focus_firefox():
    """Best-effort focus for Firefox. With --window patching, this is mostly
    a no-op since keystrokes go directly to the Firefox window."""
    if FIREFOX_WID:
        # Already using window-targeted input, no focus needed
        return
    env = {"DISPLAY": DISPLAY}
    result = subprocess.run(
        ["xdotool", "search", "--class", "firefox"],
        capture_output=True, text=True, env=env
    )
    wids = [w.strip() for w in result.stdout.strip().split('\n') if w.strip()]
    if wids:
        subprocess.run(["wmctrl", "-ia", wids[0]], env=env, capture_output=True)
        time.sleep(0.3)
        subprocess.run(["xdotool", "windowactivate", "--sync", wids[0]], env=env, capture_output=True)
        time.sleep(0.3)


def _dismiss_popups():
    """Dismiss browser permission popups (microphone, notifications, etc.).
    These block keyboard input and prevent Send from working."""
    firefox = atspi.find_firefox()
    if not firefox:
        return
    dismiss_names = {"block", "deny", "dismiss", "not now", "maybe later", "no thanks",
                      "set as default browser"}
    # Partial matches for buttons like "Acknowledge and close" (Gemini)
    # NOTE: "close" alone matches tab close buttons — only match multi-word close buttons
    dismiss_partials = ["acknowledge", "got it"]
    queue = [(firefox, 0)]
    dismissed = 0
    while queue:
        node, depth = queue.pop(0)
        if depth > 10:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()
            if "button" in role and (name in dismiss_names or any(p in name for p in dismiss_partials)):
                x, y = 0, 0
                try:
                    comp = node.get_component_iface()
                    if comp:
                        ext = comp.get_extents(0)
                        x, y = ext.x, ext.y
                except Exception:
                    pass
                if x > 0 and y > 80:  # y > 80 avoids tab bar buttons
                    log.info(f"  Dismissing popup: \"{node.get_name()}\" ({x},{y})")
                    inp.click_at(x, y)
                    time.sleep(0.5)
                    dismissed += 1
            for i in range(node.get_child_count()):
                c = node.get_child_at_index(i)
                if c:
                    queue.append((c, depth + 1))
        except Exception:
            continue
    if dismissed:
        log.info(f"  Dismissed {dismissed} popup(s)")


def _dismiss_gemini_dialogs():
    """Dismiss Gemini feature/promo dialogs via AT-SPI.
    These dialogs (e.g. 'Gemini Feature Information') overlay the entire page
    and have an 'Acknowledge and close' button inside a dialog role element."""
    _, _, scope = get_scope("gemini")
    if not scope:
        return
    queue = [(scope, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > 12:
            continue
        try:
            if node.get_role_name() == "dialog":
                # Found dialog — search inside for dismiss button
                dq = [(node, 0)]
                while dq:
                    dn, dd = dq.pop(0)
                    if dd > 5:
                        continue
                    try:
                        if "button" in dn.get_role_name():
                            btn_name = (dn.get_name() or "").lower()
                            if "acknowledge" in btn_name or "close" in btn_name or "got it" in btn_name:
                                try:
                                    action = dn.get_action_iface()
                                    if action:
                                        action.do_action(0)
                                        log.info(f"  [gemini] Dismissed dialog: '{dn.get_name()}'")
                                        time.sleep(1.0)
                                        return
                                except Exception:
                                    pass
                        for i in range(dn.get_child_count()):
                            c = dn.get_child_at_index(i)
                            if c:
                                dq.append((c, dd + 1))
                    except Exception:
                        continue
            for i in range(node.get_child_count()):
                c = node.get_child_at_index(i)
                if c:
                    queue.append((c, depth + 1))
        except Exception:
            continue


def _run_devtools_js(js_code: str):
    """Open Firefox DevTools console, execute JS, close DevTools."""
    inp.press_key("ctrl+shift+k")
    time.sleep(1.5)
    inp.press_key("ctrl+a")
    time.sleep(0.1)
    inp.press_key("Delete")
    time.sleep(0.1)
    inp.type_text(js_code, delay_ms=1)
    time.sleep(0.3)
    inp.press_key("Return")
    time.sleep(1.0)
    inp.press_key("F12")
    time.sleep(0.5)


def inject_text_via_devtools(text: str) -> bool:
    """Inject text into the active contenteditable element via Firefox DevTools console.

    This bypasses React's synthetic event filtering that blocks xdotool typing
    when files are attached on ChatGPT/Grok/etc. Uses innerHTML which reliably
    triggers React's change detection across all platforms.
    """
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

    js_code = (
        f"var el = document.querySelector('[contenteditable=\"true\"]') || "
        f"document.querySelector('div[role=\"textbox\"]') || "
        f"document.querySelector('textarea'); "
        f"if (el) {{ el.focus(); "
        f"if (el.contentEditable === 'true') {{ "
        f"el.innerHTML = '<p>{escaped}</p>'; "
        f"el.dispatchEvent(new InputEvent('input', {{bubbles: true, data: '{escaped}', inputType: 'insertText'}})); "
        f"}} else {{ "
        f"var s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set; "
        f"s.call(el, '{escaped}'); el.dispatchEvent(new Event('input', {{bubbles: true}})); "
        f"}} }}"
    )

    _run_devtools_js(js_code)
    log.info(f"  Injected text via DevTools ({len(text)} chars)")
    return True


def click_send_via_devtools() -> bool:
    """Click the Send/Submit button via DevTools JS for platforms where
    AT-SPI doesn't expose the Send button (e.g. Grok, Perplexity)."""
    js_code = (
        "var sent = false; "
        # Strategy 1: find button by aria-label or data-testid
        "var btns = document.querySelectorAll('button'); "
        "for (var i = 0; i < btns.length; i++) { "
        "var b = btns[i]; "
        "var al = (b.getAttribute('aria-label')||'').toLowerCase(); "
        "var dn = (b.getAttribute('data-testid')||'').toLowerCase(); "
        "if (al.match(/send|submit/) || dn.match(/send|submit/)) "
        "{ b.click(); sent = true; break; } } "
        # Strategy 2: find form and submit
        "if (!sent) { var form = document.querySelector('form'); "
        "if (form) { var sb = form.querySelector('button[type=submit]'); "
        "if (sb) { sb.click(); sent = true; } } } "
        # Strategy 3: dispatch Enter on the active input
        "if (!sent) { var el = document.activeElement; "
        "if (el) { el.dispatchEvent(new KeyboardEvent('keydown', "
        "{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true})); } }"
    )
    _run_devtools_js(js_code)
    log.info("  Clicked Send via DevTools JS")
    return True


def is_send_enabled(platform: str) -> bool:
    """Check if the Send button is enabled (has ENABLED state)."""
    _, _, scope = get_scope(platform)
    if not scope:
        return False
    queue = [(scope, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > ATSPI_DEPTH:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()
            if "button" in role and any(p in name for p in SEND_BUTTON_NAMES.get(platform, ["send"])):
                ss = node.get_state_set()
                return ss.contains(Atspi.StateType.ENABLED)
            for i in range(node.get_child_count()):
                child = node.get_child_at_index(i)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return False


def get_scope(platform: str):
    """Get AT-SPI scope for a platform. Returns (firefox, doc, scope)."""
    firefox = atspi.find_firefox()
    if not firefox:
        return None, None, None
    doc = atspi.get_platform_document(firefox, platform)
    scope = doc if doc else firefox
    return firefox, doc, scope


def get_elements(platform: str) -> list:
    """Get filtered elements for current platform tab."""
    _, _, scope = get_scope(platform)
    if not scope:
        return []
    return filter_useful_elements(find_elements(scope, max_depth=ATSPI_DEPTH))


def find_button_by_names(elements: list, name_patterns: list) -> dict | None:
    """Find first button matching any of the name patterns."""
    for el in elements:
        name = (el.get("name") or "").lower()
        role = el.get("role", "")
        if "button" in role:
            for pattern in name_patterns:
                if pattern in name:
                    return el
    return None


# ============================================================================
# Core operations
# ============================================================================

def switch_tab(platform: str):
    """Switch to platform tab. ~0.5s. Ensures Firefox focus first."""
    _focus_firefox()
    inp.press_key(TAB_KEYS[platform])
    time.sleep(0.5)


def open_new_chat(platform: str) -> bool:
    """Open fresh chat. Cleans stale files. Verifies clean state. ~5s"""
    _, value = NEW_CHAT[platform]
    inp.press_key("ctrl+l")
    time.sleep(0.3)
    inp.type_text(value, delay_ms=10)
    inp.press_key("Return")
    wait = 8.0 if platform == "gemini" else 4.0
    time.sleep(wait)

    # Dismiss any permission popups that appeared after navigation
    _dismiss_popups()

    # Gemini-specific: dismiss feature/promo dialogs that overlay the page
    if platform == "gemini":
        _dismiss_gemini_dialogs()

    # Clean up stale attached files (from previous failed tests)
    for cleanup_round in range(3):
        elements = get_elements(platform)
        stale_files = [e for e in elements
                       if "remove file" in (e.get("name") or "").lower()
                       and "button" in e.get("role", "")]
        if not stale_files:
            break
        log.info(f"  [{platform}] Removing {len(stale_files)} stale files")
        for f in stale_files:
            inp.click_at(f["x"], f["y"])
            time.sleep(0.4)
        time.sleep(1.0)

    # Verify: page loaded and no existing responses
    for attempt in range(8):
        elements = get_elements(platform)
        copies = find_copy_buttons(elements)
        if len(elements) > 3 and len(copies) == 0:
            return True
        if len(copies) > 0 and attempt < 6:
            log.info(f"  [{platform}] Stale chat detected ({len(copies)} copies), retrying (attempt {attempt+1}/6)...")
            # Navigate to about:blank first to force AT-SPI tree reset
            # DO NOT close/reopen tabs — that breaks Alt+N tab ordering!
            inp.press_key("ctrl+l")
            time.sleep(0.3)
            inp.type_text("about:blank", delay_ms=10)
            inp.press_key("Return")
            time.sleep(3.0)
            inp.press_key("ctrl+l")
            time.sleep(0.3)
            inp.type_text(value, delay_ms=10)
            inp.press_key("Return")
            time.sleep(8.0)  # Longer wait for AT-SPI tree to fully update
            continue
        time.sleep(2.0)

    log.warning(f"  [{platform}] New chat verification incomplete, proceeding anyway")
    return True


def attach_file(platform: str, file_path: str) -> bool:
    """Attach file with full verification. ~5s"""
    firefox, doc, scope = get_scope(platform)
    if not firefox:
        return False

    # Ctrl+U opens file dialog directly — works on ChatGPT, Claude, Perplexity
    # NOT Grok (Ctrl+U opens sidebar) or Gemini (Ctrl+U does nothing)
    if platform in ("chatgpt", "claude", "perplexity"):
        elements = filter_useful_elements(find_elements(scope, max_depth=ATSPI_DEPTH))
        inputs = [e for e in elements if e.get("role") in ("entry", "text", "section", "paragraph")]
        if inputs:
            input_el = max(inputs, key=lambda e: e.get("y", 0))
            inp.click_at(input_el["x"], input_el["y"])
            time.sleep(0.3)
        inp.press_key("ctrl+u")
        time.sleep(1.5)
        if atspi.is_file_dialog_open(firefox):
            log.info(f"  [{platform}] Attach via Ctrl+U")
            return _handle_file_dialog(firefox, file_path)
        log.info(f"  [{platform}] Ctrl+U failed, trying button method")

    # Find attach button with retry
    attach_btn = None
    for attempt in range(8):
        elements = filter_useful_elements(find_elements(scope, max_depth=ATSPI_DEPTH))
        attach_btn = find_button_by_names(elements, ATTACH_NAMES.get(platform, ["attach"]))
        if attach_btn:
            break
        wait = 2.0 + attempt * 0.5
        log.info(f"  [{platform}] Attach button not found (attempt {attempt+1}), waiting {wait:.0f}s...")
        time.sleep(wait)

    if not attach_btn:
        log.error(f"  [{platform}] Attach button not found after retries")
        buttons = [e for e in elements if "button" in e.get("role", "")]
        log.info(f"  [{platform}] Buttons: {[(b.get('name','?')[:40], b.get('x'), b.get('y')) for b in buttons[:12]]}")
        return False

    log.info(f"  [{platform}] Attach: '{attach_btn.get('name')}' @ ({attach_btn['x']},{attach_btn['y']})")

    # Click attach button
    # Use AT-SPI do_action for Gemini and Claude (xdotool fails when Firefox lacks native X focus)
    # Grok: xdotool click (AT-SPI do_action opens wrong menu on Grok)
    if platform == "grok":
        # Grok: xdotool click with extra wait for dropdown to render
        inp.click_at(attach_btn["x"], attach_btn["y"])
        time.sleep(2.5)  # Grok dropdown needs extra time
    elif platform in ("gemini", "claude"):
        attach_names = ATTACH_NAMES.get(platform, ["attach"])
        # Search from scope first, then firefox root as fallback
        attach_obj = _find_button_atspi(scope, attach_names)
        if not attach_obj:
            attach_obj = _find_button_atspi(firefox, attach_names)
        clicked_via_atspi = False
        if attach_obj:
            try:
                action = attach_obj.get_action_iface()
                if action:
                    action.do_action(0)
                    clicked_via_atspi = True
                    time.sleep(2.0)  # Extra wait for dropdown to render
            except Exception:
                pass
        if not clicked_via_atspi:
            log.info(f"  [{platform}] AT-SPI click failed, using xdotool")
            inp.click_at(attach_btn["x"], attach_btn["y"])
            time.sleep(1.5)
    else:
        inp.click_at(attach_btn["x"], attach_btn["y"])
        time.sleep(1.5)

    # Check: file dialog or dropdown?
    if atspi.is_file_dialog_open(firefox):
        return _handle_file_dialog(firefox, file_path)

    # Look for dropdown upload item via targeted AT-SPI search near the button
    # (avoids find_dropdown_menus which has cross-tab contamination issues)
    dropdown_items = _find_menu_items_near(attach_btn["x"], attach_btn["y"])
    log.info(f"  [{platform}] Menu items found near button: {len(dropdown_items)}")
    if dropdown_items:
        attach_name_lower = (attach_btn.get("name") or "").lower()
        log.info(f"  [{platform}] Dropdown: {[i.get('name','?') for i in dropdown_items]}")

        upload_item = None
        for item in dropdown_items:
            item_name = (item.get("name") or "").lower()
            if item_name == attach_name_lower:
                continue  # Skip the button itself
            for term in UPLOAD_ITEM_NAMES:
                if term in item_name:
                    upload_item = item
                    break
            if upload_item:
                break

        # Fallback: any dropdown item with "file" or "photo"
        if not upload_item:
            for item in dropdown_items:
                item_name = (item.get("name") or "").lower()
                if item_name != attach_name_lower and ("file" in item_name or "photo" in item_name):
                    upload_item = item
                    break

        if upload_item:
            log.info(f"  [{platform}] Upload item: '{upload_item.get('name')}'")
            # Click via AT-SPI do_action (React dropdown items don't respond to xdotool)
            clicked = False
            obj = upload_item.get("atspi_obj")
            if obj:
                try:
                    action = obj.get_action_iface()
                    if action:
                        action.do_action(0)
                        clicked = True
                except Exception:
                    pass
            if not clicked:
                inp.click_at(upload_item["x"], upload_item["y"])
            time.sleep(1.0)

            # Wait for file dialog
            for _ in range(10):
                if atspi.is_file_dialog_open(firefox):
                    return _handle_file_dialog(firefox, file_path)
                time.sleep(0.5)

    # Retry: click attach again and re-search for dropdown
    if not dropdown_items and platform in ("grok",):
        log.info(f"  [{platform}] Retrying attach click...")
        inp.press_key("Escape")
        time.sleep(0.5)
        inp.click_at(attach_btn["x"], attach_btn["y"])
        time.sleep(3.0)
        dropdown_items = _find_menu_items_near(attach_btn["x"], attach_btn["y"])
        log.info(f"  [{platform}] Retry menu items: {len(dropdown_items)}")
        if dropdown_items:
            attach_name_lower = (attach_btn.get("name") or "").lower()
            upload_item = None
            for item in dropdown_items:
                item_name = (item.get("name") or "").lower()
                if item_name != attach_name_lower:
                    for term in UPLOAD_ITEM_NAMES:
                        if term in item_name:
                            upload_item = item
                            break
                if upload_item:
                    break
            if not upload_item:
                for item in dropdown_items:
                    item_name = (item.get("name") or "").lower()
                    if item_name != attach_name_lower and ("file" in item_name or "photo" in item_name):
                        upload_item = item
                        break
            if upload_item:
                log.info(f"  [{platform}] Upload item: '{upload_item.get('name')}'")
                obj = upload_item.get("atspi_obj")
                if obj:
                    try:
                        action = obj.get_action_iface()
                        if action:
                            action.do_action(0)
                    except Exception:
                        inp.click_at(upload_item["x"], upload_item["y"])
                else:
                    inp.click_at(upload_item["x"], upload_item["y"])
                time.sleep(1.0)
                for _ in range(10):
                    if atspi.is_file_dialog_open(firefox):
                        return _handle_file_dialog(firefox, file_path)
                    time.sleep(0.5)

    # Final attempt
    time.sleep(1.0)
    if atspi.is_file_dialog_open(firefox):
        return _handle_file_dialog(firefox, file_path)

    log.error(f"  [{platform}] Could not open file dialog")
    return False


def _handle_file_dialog(firefox, file_path: str) -> bool:
    """Handle GTK file dialog with verification. ~2s"""
    time.sleep(0.3)
    inp.press_key("ctrl+l")
    time.sleep(0.5)
    inp.type_text(file_path, delay_ms=5)
    time.sleep(0.3)
    inp.press_key("Return")
    time.sleep(0.5)
    inp.press_key("Return")  # Confirm selection
    time.sleep(0.5)

    # Verify dialog closed (like MCP tools do)
    for _ in range(25):
        if not atspi.is_file_dialog_open(firefox):
            log.info(f"  File attached: {os.path.basename(file_path)}")
            return True
        time.sleep(0.2)

    log.error(f"  File dialog did not close after 5s")
    return False


def _find_menu_items_near(ref_x: int, ref_y: int):
    """Find menu items near a reference point via AT-SPI BFS from Firefox root.

    Searches from the Firefox root (NOT document scope) because React portals
    render dropdown menus outside the document web element in the AT-SPI tree.
    Filters: menu item/check menu item roles, within 200px X and below ref_y.
    """
    firefox = atspi.find_firefox()
    if not firefox:
        return []
    results = []
    menu_roles = {"menu item", "check menu item"}
    queue = [(firefox, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > ATSPI_DEPTH:
            continue
        try:
            role = node.get_role_name()
            name = node.get_name() or ""
            if name and role in menu_roles:
                x, y = 0, 0
                try:
                    comp = node.get_component_iface()
                    if comp:
                        ext = comp.get_extents(0)
                        x, y = ext.x, ext.y
                except Exception:
                    pass
                # Dropdown items: within 200px X, and below the button
                if abs(x - ref_x) < 200 and y >= ref_y - 50:
                    results.append({"name": name, "role": role, "x": x, "y": y, "atspi_obj": node})
            for i in range(node.get_child_count()):
                child = node.get_child_at_index(i)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return results


def _find_button_atspi(scope_node, name_patterns, max_depth=20):
    """Find a button via AT-SPI BFS for do_action. Returns atspi object or None."""
    queue = [(scope_node, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()
            if "button" in role:
                for pattern in name_patterns:
                    if pattern in name:
                        return node
            for i in range(node.get_child_count()):
                child = node.get_child_at_index(i)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return None


def inject_and_send_via_devtools(text: str) -> bool:
    """Combined inject text + click Send in a SINGLE DevTools session.

    This avoids the focus-loss problem of opening/closing DevTools twice.
    Used for platforms where AT-SPI Send button doesn't exist (Grok, Perplexity).
    """
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    js_code = (
        # Inject text
        f"var el = document.querySelector('[contenteditable=\"true\"]') || "
        f"document.querySelector('div[role=\"textbox\"]') || "
        f"document.querySelector('textarea'); "
        f"if (el) {{ el.focus(); "
        f"if (el.contentEditable === 'true') {{ "
        f"el.innerHTML = '<p>{escaped}</p>'; "
        f"el.dispatchEvent(new InputEvent('input', {{bubbles: true, data: '{escaped}', inputType: 'insertText'}})); "
        f"}} else {{ "
        f"var s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set; "
        f"s.call(el, '{escaped}'); el.dispatchEvent(new Event('input', {{bubbles: true}})); "
        f"}} }} "
        # Wait 200ms then click Send
        "setTimeout(function() { "
        "var btns = document.querySelectorAll('button'); "
        "for (var i = 0; i < btns.length; i++) { "
        "var b = btns[i]; "
        "var al = (b.getAttribute('aria-label')||'').toLowerCase(); "
        "var dn = (b.getAttribute('data-testid')||'').toLowerCase(); "
        "if (al.match(/send|submit|prompt/) || dn.match(/send|submit/)) "
        "{ b.click(); break; } } "
        "}, 200);"
    )
    _run_devtools_js(js_code)
    log.info(f"  Injected text + clicked Send via DevTools ({len(text)} chars)")
    return True


def _find_input_atspi(platform: str):
    """Find input element via AT-SPI BFS. Returns (atspi_obj, x, y, w, h) or None.

    Searches deeper than get_elements() for platforms like Gemini where the
    input is at depth 13+ in the AT-SPI tree.
    """
    _, _, scope = get_scope(platform)
    if not scope:
        return None

    queue = [(scope, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > ATSPI_DEPTH:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()

            is_input = False
            if role == "entry":
                is_input = True
            elif role in ("section", "paragraph") and any(kw in name for kw in
                    ["message", "ask", "chat", "prompt", "type", "reply"]):
                is_input = True
            elif role in ("section", "paragraph"):
                # Check for editable state (Grok uses editable sections)
                try:
                    ss = node.get_state_set()
                    if ss.contains(Atspi.StateType.EDITABLE):
                        is_input = True
                except Exception:
                    pass

            if is_input:
                try:
                    comp = node.get_component_iface()
                    if comp:
                        ext = comp.get_extents(0)
                        if ext.x > 50 and ext.y > 100:
                            return (node, ext.x, ext.y, ext.width, ext.height)
                except Exception:
                    pass

            for i in range(node.get_child_count()):
                child = node.get_child_at_index(i)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return None


def send_message(platform: str, message: str) -> bool:
    """Type message and click Send button. ~2-8s

    Send strategy (in order):
    1. grab_focus + xdotool type → AT-SPI Send button (Gemini, Perplexity)
    2. xdotool type → check Send enabled → AT-SPI Send (Claude, ChatGPT without files)
    3. DevTools JS inject → AT-SPI Send (ChatGPT with files)
    4. DevTools JS inject+send combined (Grok — no AT-SPI Send)
    5. Re-focus input → Enter key (final fallback)
    """
    # Step 1: Find input via deep AT-SPI scan
    input_atspi = _find_input_atspi(platform)
    # Track input coordinates for re-focus fallback
    input_x, input_y = 0, 0

    if input_atspi:
        node, ix, iy, iw, ih = input_atspi
        log.info(f"  [{platform}] Input: {node.get_role_name()} '{(node.get_name() or '')[:30]}' @ ({ix},{iy})")

        # grab_focus is critical for Gemini and Perplexity
        try:
            comp = node.get_component_iface()
            if comp:
                comp.grab_focus()
                time.sleep(0.3)
        except Exception:
            pass

        # Click center of input
        input_x = ix + (iw // 2 or 10)
        input_y = iy + (ih // 2 or 10)
        inp.click_at(input_x, input_y)
        time.sleep(0.3)
    else:
        # Fallback: use get_elements() for shallower search
        elements = get_elements(platform)
        entries = [e for e in elements if e.get("role") in ("entry", "text")]
        if not entries:
            entries = [e for e in elements if "editable" in str(e.get("states", []))]
        if not entries:
            entries = [e for e in elements if e.get("role") in ("section", "paragraph") and e.get("y", 0) > 300]
        if not entries:
            entries = [e for e in elements if e.get("role") == "form" and e.get("y", 0) > 200]

        if not entries:
            log.error(f"  [{platform}] Input area not found")
            return False

        input_el = max(entries, key=lambda e: e.get("y", 0))
        input_x = input_el["x"] + (input_el.get("width", 0) // 2 or 10)
        input_y = input_el["y"] + (input_el.get("height", 0) // 2 or 10)
        log.info(f"  [{platform}] Input: role={input_el.get('role')} @ ({input_el['x']},{input_el['y']})")
        inp.click_at(input_x, input_y)
        time.sleep(0.3)

    # Type message via xdotool type (fast, works on all platforms for plain text)
    inp.type_text(message, delay_ms=3)
    time.sleep(0.5)

    send_names = SEND_BUTTON_NAMES.get(platform, ["send"])

    # Strategy 1: Try AT-SPI Send button (works on ChatGPT, Claude, Gemini, Perplexity)
    _, _, scope = get_scope(platform)
    if scope:
        send_obj = _find_button_atspi(scope, send_names)
        if send_obj:
            try:
                action = send_obj.get_action_iface()
                if action:
                    log.info(f"  [{platform}] Send via AT-SPI do_action")
                    action.do_action(0)
                    return True
            except Exception as e:
                log.warning(f"  [{platform}] AT-SPI send failed: {e}")

    # Strategy 2: Enter key (works on all platforms when input is focused)
    log.info(f"  [{platform}] Send via Enter key")
    inp.press_key("Return")
    time.sleep(1.0)

    # Verify something happened (generating or Send button disappeared)
    if is_generating(platform):
        return True

    # Check if Send button is still enabled — if so, xdotool type may have failed
    # (React countermeasure blocks typing when files are attached)
    if is_send_enabled(platform):
        log.info(f"  [{platform}] Send still enabled — xdotool type may have failed")
        # DevTools fallback: inject text + click Send in one shot
        inject_and_send_via_devtools(message if len(message) < 200 else SEND_PROMPT_SHORT)
        time.sleep(1.0)

        # Final backup: re-focus + Enter
        inp.click_at(input_x, input_y)
        time.sleep(0.3)
        inp.press_key("Return")
        time.sleep(0.5)

    return True


def spawn_monitor(platform: str, baseline_copies: int = 0) -> str | None:
    """Spawn monitor daemon (same as MCP tools). Returns monitor_id."""
    monitor_id = uuid.uuid4().hex[:8]
    session_id = uuid.uuid4().hex

    cmd = [
        sys.executable, DAEMON_PATH,
        "--platform", platform,
        "--monitor-id", monitor_id,
        "--baseline-copy-count", str(baseline_copies),
        "--timeout", "3600",
        "--session-id", session_id,
        "--user-message-id", uuid.uuid4().hex,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        log.info(f"  [{platform}] Monitor daemon spawned: {monitor_id} (pid={proc.pid})")
        return monitor_id
    except Exception as e:
        log.error(f"  [{platform}] Failed to spawn monitor: {e}")
        return None


def is_generating(platform: str) -> bool:
    """Check if platform is generating (stop button visible). ~0.5s"""
    elements = get_elements(platform)
    for el in elements:
        name = (el.get("name") or "").lower()
        role = el.get("role", "")
        if "button" in role:
            for pattern in STOP_PATTERNS:
                if pattern in name:
                    # ChatGPT canvas "Stop" paired with "Update" — not generating
                    if platform == "chatgpt" and name == "stop":
                        for other in elements:
                            if (other.get("name") or "").lower() == "update":
                                if abs(other.get("y", 0) - el.get("y", 0)) < 50:
                                    return False
                    return True
    return False


def count_copy_buttons(platform: str) -> int:
    """Count copy buttons on current platform. ~0.5s"""
    elements = get_elements(platform)
    return len(find_copy_buttons(elements))


def _count_grok_copy_buttons() -> int:
    """Count Grok copy buttons using platform document scope."""
    _, doc, scope = get_scope("grok")
    if not scope:
        return 0
    buttons = _find_copy_buttons_atspi(scope, max_depth=25 if not doc else 20)
    if not doc:
        buttons = [b for b in buttons if 0 < b.get("x", 0) < 1400 and 0 < b.get("y", 0) < 1100]
    return len(buttons)


def _find_copy_buttons_atspi(scope_node, max_depth=20):
    """Find Copy buttons with AT-SPI object references for do_action."""
    results = []
    queue = [(scope_node, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").strip().lower()
            if role in ("toggle button", "push button") and name == "copy":
                x, y = 0, 0
                try:
                    comp = node.get_component_iface()
                    if comp:
                        ext = comp.get_extents(0)
                        x, y = ext.x, ext.y
                except Exception:
                    pass
                results.append({"name": "Copy", "x": x, "y": y, "atspi_obj": node})
            for i in range(node.get_child_count()):
                child = node.get_child_at_index(i)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return results


def _click_copy_button(btn, platform):
    """Click a copy button with AT-SPI do_action + xdotool fallback."""
    obj = btn["atspi_obj"]
    clip.clear()
    time.sleep(0.1)

    # Try AT-SPI do_action first
    try:
        action_iface = obj.get_action_iface()
        if action_iface:
            action_iface.do_action(0)
            time.sleep(2.0)
            content = clip.read()
            if content and len(content) > 2:
                return content
    except Exception:
        pass

    # Fallback: scroll to bottom + xdotool click
    inp.press_key("End")
    time.sleep(0.5)
    clip.clear()
    time.sleep(0.1)
    inp.click_at(btn["x"], btn["y"])
    time.sleep(1.5)
    return clip.read()


def extract_response(platform: str) -> str | None:
    """Extract latest response via copy button (AT-SPI + xdotool). ~4s"""
    # Scroll to bottom first — makes response copy buttons visible/accessible
    inp.press_key("End")
    time.sleep(1.0)

    # For Claude: hover over response area to reveal Copy buttons
    if platform == "claude":
        inp.click_at(700, 400)  # Click response area to establish focus
        time.sleep(0.5)

    _, doc, scope = get_scope(platform)
    if not scope:
        return None
    copy_buttons = _find_copy_buttons_atspi(scope if doc else scope, max_depth=25 if not doc else 20)
    # If scope fell back to Firefox root (no doc found), filter to visible area only
    if not doc:
        copy_buttons = [b for b in copy_buttons if 0 < b.get("x", 0) < 1400 and 0 < b.get("y", 0) < 1100]

    # If no copy buttons found, try scrolling and re-scanning
    if not copy_buttons:
        inp.press_key("End")
        time.sleep(1.5)
        _, doc2, scope2 = get_scope(platform)
        if scope2:
            copy_buttons = _find_copy_buttons_atspi(scope2, max_depth=25 if not doc2 else 20)
            if not doc2:
                copy_buttons = [b for b in copy_buttons if 0 < b.get("x", 0) < 1400 and 0 < b.get("y", 0) < 1100]

    if not copy_buttons:
        log.warning(f"  [{platform}] No copy buttons found")
        return None

    log.info(f"  [{platform}] Found {len(copy_buttons)} copy buttons")

    # ChatGPT: filter out canvas copy buttons (x > 1000)
    if platform == "chatgpt":
        main = [b for b in copy_buttons if b.get("x", 0) < 1000]
        if main:
            copy_buttons = main

    # Try all copy buttons and pick the best one (JSON response preferred)
    best_content = None
    best_score = -1

    # Try ALL buttons in reverse order (newest first) — stale chat can leave many old buttons
    for idx in range(len(copy_buttons) - 1, -1, -1):
        btn = copy_buttons[idx]
        content = _click_copy_button(btn, platform)
        if not content or len(content) < 2:
            continue

        cl = content.lower().strip()

        # Detect user's package/prompt — these are ALWAYS skipped
        is_package = (cl.startswith("# instructions")           # Package header
                      or cl.startswith("# theme:")              # Package with theme header
                      or (len(content) > 50000 and "## items for analysis" in cl)  # Large package body
                      or (len(content) > 50000 and "## context anchors" in cl))
        is_prompt = (content.strip() == SEND_PROMPT_SHORT  # exact "Go" match only
                     or SEND_PROMPT[:30] in content
                     or ("analyze the" in cl[:50] and "context package" in cl and len(content) < 500)
                     or ("follow the instructions" in cl and len(content) < 500)
                     or (len(content) < 200 and "json format" in cl))
        if is_package or is_prompt:
            log.info(f"  [{platform}] Button {idx}: user {'package' if is_package else 'prompt'} ({len(content)} chars), skip")
            continue

        # Score: prefer content with JSON response structure, then by length
        # A real JSON response starts with { or has "rosetta" key (our output format)
        trimmed = content.strip()
        is_json_response = (trimmed.startswith("{") and "package_id" in content) or '"rosetta"' in content
        score = len(content) + (100000 if is_json_response else 0)
        log.info(f"  [{platform}] Button {idx}: {len(content)} chars, json={is_json_response}")

        if score > best_score:
            best_score = score
            best_content = content

    if best_content and len(best_content) >= 2:
        log.info(f"  [{platform}] Extracted {len(best_content)} chars from copy button")
        return best_content

    log.warning(f"  [{platform}] Clipboard empty/short ({len(best_content or '')} chars, {len(copy_buttons)} copy buttons found)")
    return None


def assess_quality(platform: str, content: str, elements: list) -> dict:
    """Quality assessment (mirrors MCP extract.py logic)."""
    words = len(content.split())
    assessment = {"ok": True, "words": words, "action": None}

    if platform == "perplexity":
        # Check for Export button = Deep Research summary only
        for el in elements:
            if "export" in (el.get("name") or "").lower() and words < 500:
                assessment["ok"] = False
                assessment["action"] = "export_markdown"
                assessment["reason"] = "Perplexity Deep Research: copy = summary only"
                break

    elif platform == "claude":
        for el in elements:
            if "continue" in (el.get("name") or "").lower() and "button" in el.get("role", ""):
                assessment["ok"] = False
                assessment["action"] = "click_continue"
                assessment["reason"] = "Claude truncated (Continue button visible)"
                break

    elif platform == "chatgpt":
        for el in elements:
            if "show more" in (el.get("name") or "").lower():
                assessment["ok"] = False
                assessment["action"] = "click_show_more"
                assessment["reason"] = "ChatGPT collapsed response"
                break

    if words < 20:
        assessment["ok"] = False
        assessment["action"] = "verify"
        assessment["reason"] = f"Suspiciously short ({words} words)"

    return assessment


# ============================================================================
# Package builder + storage wrappers
# ============================================================================

def build_package(platform: str) -> str | None:
    """Build next package. Returns file path or None."""
    result = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "next", "--platform", platform],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0 and result.stderr:
        log.warning(f"  [{platform}] Builder error: {result.stderr[:200]}")
        return None

    for line in result.stdout.split("\n"):
        if line.startswith("Package ready:"):
            path = line.split(":", 1)[1].strip()
            log.info(f"  [{platform}] Package: {os.path.basename(path)}")
            return path

    if "no items" in result.stdout.lower() or "no themes" in result.stdout.lower():
        log.info(f"  [{platform}] No items available")
    else:
        log.warning(f"  [{platform}] Build output: {result.stdout[:200]}")
    return None


def store_results(response_text: str, platform: str) -> dict:
    """Parse and store via hmm_store_results.py."""
    # Extract JSON
    data = None
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(response_text[start:end + 1])
            except json.JSONDecodeError:
                pass

    if not data:
        return {"ok": False, "error": "json_parse_failed", "chars": len(response_text)}

    tmp_path = f"/tmp/hmm_response_{platform}_{int(time.time())}.json"
    with open(tmp_path, "w") as f:
        json.dump(data, f)

    items = data.get("items", [])
    log.info(f"  [{platform}] Parsed {len(items)} items, storing...")

    result = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_store_results.py",
         tmp_path, "--platform", platform],
        capture_output=True, text=True, timeout=120
    )

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if result.returncode != 0:
        log.error(f"  [{platform}] Store failed: {result.stderr[:200]}")
        return {"ok": False, "error": "store_failed", "items": len(items)}

    log.info(f"  [{platform}] Stored {len(items)} items")
    return {"ok": True, "items": len(items)}


def complete_package(platform: str):
    subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "complete", "--platform", platform],
        capture_output=True, text=True, timeout=30
    )


def fail_package(platform: str, reason: str):
    subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "fail", reason, "--platform", platform],
        capture_output=True, text=True, timeout=30
    )


# ============================================================================
# Orchestration
# ============================================================================

def _send_via_marionette(platform: str, pkg_path: str) -> dict | None:
    """Send package via Marionette (focus-free mode). Returns metadata or None."""
    t0 = time.time()

    # Read package content
    with open(pkg_path, "r") as f:
        content = f.read()
    log.info(f"  [{platform}] Marionette: pkg={os.path.basename(pkg_path)} ({len(content)} chars)")

    # Navigate to new chat
    MARIONETTE.open_new_chat(platform)
    log.info(f"  [{platform}] Marionette: navigated to new chat")

    # Inject: package content + prompt into the input
    prompt = SEND_PROMPT_PERPLEXITY if platform == "perplexity" else SEND_PROMPT
    full_text = content + "\n\n" + prompt
    result = MARIONETTE.inject_text_chunked(full_text)
    if not result.startswith("OK"):
        log.error(f"  [{platform}] Marionette inject failed: {result}")
        fail_package(platform, "inject_failed")
        return None
    log.info(f"  [{platform}] Marionette: injected {len(full_text)} chars ({result})")

    # Spawn monitor for AT-SPI-based response detection
    monitor_id = spawn_monitor(platform, 0)

    # Click Send
    time.sleep(0.5)
    send_result = MARIONETTE.click_send(platform)
    log.info(f"  [{platform}] Marionette: send={send_result}")

    elapsed = time.time() - t0
    log.info(f"  [{platform}] SENT in {elapsed:.1f}s (Marionette)")
    log_event("sent", platform, elapsed=round(elapsed, 1), pkg=os.path.basename(pkg_path), mode="marionette")

    # Post-send baseline — with Marionette text injection, the user's message gets
    # its own copy button. Wait for it to appear before measuring baseline.
    time.sleep(8.0)
    baseline = MARIONETTE.count_copy_buttons()
    # If baseline is still 0, wait a bit more — large messages take time to render
    if baseline == 0:
        time.sleep(5.0)
        baseline = MARIONETTE.count_copy_buttons()
    log.info(f"  [{platform}] Post-send baseline: {baseline} copies (Marionette)")

    return {
        "platform": platform,
        "pkg": pkg_path,
        "baseline_copies": baseline,
        "monitor_id": monitor_id,
        "sent_at": time.time(),
    }


def send_to_platform(platform: str) -> dict | None:
    """Full send: build pkg → new chat → attach → send. Returns metadata or None."""
    t0 = time.time()

    pkg_path = build_package(platform)
    if not pkg_path:
        return None

    # Use Marionette for navigation only (tab switch + new chat URL) if available.
    # File attachment, send, and extraction always use AT-SPI.
    if MARIONETTE and platform in MARIONETTE.tab_handles:
        log.info(f"  [{platform}] Using Marionette for navigation")
        MARIONETTE.open_new_chat(platform)
    else:
        switch_tab(platform)
        open_new_chat(platform)

    if not attach_file(platform, pkg_path):
        fail_package(platform, "attach_failed")
        return None

    # Let upload register — Gemini uploads to Google servers (needs more time for large files)
    if platform == "gemini":
        log.info(f"  [{platform}] Waiting for upload to complete...")
        time.sleep(8.0)
        # Verify: look for file badge in AT-SPI tree
        elements = get_elements(platform)
        file_indicators = [e for e in elements if
                          any(x in (e.get("name") or "").lower() for x in ["remove file", "pkg_", ".md"])]
        if file_indicators:
            log.info(f"  [{platform}] File badge detected: {file_indicators[0].get('name', '')[:50]}")
        else:
            log.warning(f"  [{platform}] No file badge found after upload wait")
    else:
        time.sleep(1.5)

    # Spawn monitor BEFORE sending (critical race condition fix)
    monitor_id = spawn_monitor(platform, 0)

    # Pick platform-appropriate prompt and send method
    prompt = SEND_PROMPT_PERPLEXITY if platform == "perplexity" else SEND_PROMPT

    # ChatGPT: React blocks xdotool typing when file attached
    # Strategy: clipboard paste (more reliable than DevTools inject)
    if platform == "chatgpt":
        # Click input area first to ensure focus
        input_atspi = _find_input_atspi(platform)
        if input_atspi:
            node, ix, iy, iw, ih = input_atspi
            cx = ix + (iw // 2 or 10)
            cy = iy + (ih // 2 or 10)
            inp.click_at(cx, cy)
            time.sleep(0.3)
            try:
                comp = node.get_component_iface()
                if comp:
                    comp.grab_focus()
            except Exception:
                pass
            time.sleep(0.3)

        # Paste prompt via clipboard (bypasses React's synthetic event filter)
        clip.set_display(DISPLAY)
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=prompt.encode(), env={"DISPLAY": DISPLAY}, timeout=2)
        time.sleep(0.2)
        inp.press_key("ctrl+v")
        time.sleep(0.5)
        log.info(f"  [{platform}] Pasted prompt via clipboard ({len(prompt)} chars)")

        # Try AT-SPI Send first
        _, _, scope = get_scope(platform)
        sent = False
        if scope:
            send_obj = _find_button_atspi(scope, SEND_BUTTON_NAMES.get(platform, ["send"]))
            if send_obj:
                try:
                    action = send_obj.get_action_iface()
                    if action:
                        action.do_action(0)
                        sent = True
                        log.info(f"  [{platform}] Send via AT-SPI after clipboard paste")
                except Exception:
                    pass
        if not sent:
            inp.press_key("Return")
            log.info(f"  [{platform}] Send via Enter after clipboard paste")
    elif not send_message(platform, prompt):
        fail_package(platform, "send_failed")
        return None

    elapsed = time.time() - t0
    log.info(f"  [{platform}] SENT in {elapsed:.1f}s")
    log_event("sent", platform, elapsed=round(elapsed, 1), pkg=os.path.basename(pkg_path))

    # Count copy buttons AFTER send as baseline (user message shows copy buttons
    # for file content on Grok/ChatGPT before the AI response appears)
    time.sleep(5.0)  # Wait for user message to fully render
    if platform == "grok":
        baseline = max(_count_grok_copy_buttons(), 1)  # Grok always has >= 1 for user msg
    else:
        baseline = count_copy_buttons(platform)
    log.info(f"  [{platform}] Post-send baseline: {baseline} copies")

    return {
        "platform": platform,
        "pkg": pkg_path,
        "baseline_copies": baseline,
        "monitor_id": monitor_id,
        "sent_at": time.time(),
    }


def wait_for_response(platform: str, baseline: int = 0, max_wait: int = 600) -> bool:
    """Wait until response is ready. Always uses AT-SPI detection."""
    t0 = time.time()
    saw_generating = False

    while _running and time.time() - t0 < max_wait:
        gen = is_generating(platform)
        if platform == "grok":
            copies = _count_grok_copy_buttons()
        else:
            copies = count_copy_buttons(platform)

        if gen:
            saw_generating = True

        # Done: not generating AND new copy button appeared
        elapsed_s = time.time() - t0
        if not gen and copies > baseline:
            if saw_generating or elapsed_s > 15:
                log.info(f"  [{platform}] Response ready ({int(elapsed_s)}s, {copies} copies)")
                return True

        # Bail: never saw generating, no new copies, been waiting 2 min
        if not gen and not saw_generating and (time.time() - t0) > 120:
            log.warning(f"  [{platform}] Never generated, no new copies after 120s")
            return False

        elapsed = int(time.time() - t0)
        if elapsed > 0 and elapsed % 30 == 0:
            status = "generating" if gen else "waiting"
            log.info(f"  [{platform}] {status}... ({elapsed}s, copies={copies})")

        time.sleep(3)

    if not _running:
        log.info(f"  [{platform}] Interrupted by signal")
        return False

    log.warning(f"  [{platform}] Timed out after {max_wait}s")
    return False


def _extract_via_marionette(platform: str) -> str | None:
    """Extract response in Marionette mode. Reads directly from DOM — no AT-SPI
    or clipboard needed. Works without X11 keyboard focus."""
    MARIONETTE.switch_tab(platform)
    time.sleep(0.5)

    # Primary: DOM-based extraction (no focus or clipboard needed)
    text = MARIONETTE.get_response_text()
    if text and len(text) > 20:
        return text

    # Fallback: try AT-SPI extraction (may fail without focus)
    log.warning(f"  [{platform}] Marionette DOM extraction failed, trying AT-SPI fallback")
    return extract_response(platform)


def extract_and_store(platform: str) -> bool:
    """Extract response, assess quality, store results. ~3s"""
    t0 = time.time()

    # Use Marionette for tab switch if available, AT-SPI for extraction always
    if MARIONETTE and platform in (MARIONETTE.tab_handles if MARIONETTE else {}):
        MARIONETTE.switch_tab(platform)
        time.sleep(0.5)
    else:
        switch_tab(platform)
        time.sleep(0.5)
    response = extract_response(platform)

    if not response:
        log.warning(f"  [{platform}] No response extracted")
        fail_package(platform, "no_response")
        return False

    # Quality assessment — handle Continue button by clicking and re-extracting
    for cont_attempt in range(3):
        elements = get_elements(platform)
        quality = assess_quality(platform, response, elements)

        if quality.get("action") == "click_continue":
            log.info(f"  [{platform}] Clicking Continue button (attempt {cont_attempt + 1})")
            for el in elements:
                if "continue" in (el.get("name") or "").lower() and "button" in el.get("role", ""):
                    obj = el.get("atspi_obj")
                    if obj:
                        try:
                            action_iface = obj.get_action_iface()
                            if action_iface:
                                action_iface.do_action(0)
                        except Exception:
                            inp.click_at(el["x"], el["y"])
                    else:
                        inp.click_at(el["x"], el["y"])
                    break
            # Wait for continuation to complete
            time.sleep(3)
            baseline_copies = count_copy_buttons(platform)
            wait_for_response(platform, baseline_copies - 1, max_wait=120)
            new_response = extract_response(platform)
            if new_response and len(new_response) > len(response):
                response = new_response
            continue
        break

    if not quality["ok"] and quality.get("action") != "click_continue":
        log.warning(f"  [{platform}] Quality issue: {quality.get('reason', '?')}")

    log.info(f"  [{platform}] Response: {len(response)} chars, {quality['words']} words")
    # Debug: log first 300 chars to diagnose extraction issues
    preview = response[:300].replace('\n', '\\n')
    log.info(f"  [{platform}] Preview: {preview}")
    result = store_results(response, platform)
    if result["ok"]:
        complete_package(platform)
        elapsed = time.time() - t0
        log.info(f"  [{platform}] STORED in {elapsed:.1f}s ({result['items']} items, {quality['words']} words)")
        log_event("stored", platform, elapsed=round(elapsed, 1),
                  items=result["items"], words=quality["words"])
        return True
    else:
        log.error(f"  [{platform}] Store failed: {result}")
        fail_package(platform, result.get("error", "unknown"))
        return False


# ============================================================================
# Pipeline modes
# ============================================================================

def cycle_once():
    """Single cycle: send all → wait → extract all. Exit with JSON."""
    t_start = time.time()
    sent_meta = {}

    # Phase 1: SEND
    log.info("=" * 60)
    log.info("PHASE 1: SEND")
    log.info("=" * 60)

    for platform in PLATFORMS:
        try:
            result = send_to_platform(platform)
            if result:
                sent_meta[platform] = result
            else:
                log.info(f"  [{platform}] Skipped")
        except Exception as e:
            log.error(f"  [{platform}] Send error: {e}")

    if not sent_meta:
        log.info("Nothing sent.")
        print(json.dumps({"sent": 0, "extracted": 0, "elapsed": 0}))
        return

    send_time = time.time() - t_start
    log.info(f"Sent to {len(sent_meta)} platforms in {send_time:.1f}s")

    # Phase 2: WAIT + EXTRACT
    log.info("=" * 60)
    log.info("PHASE 2: WAIT + EXTRACT")
    log.info("=" * 60)

    extracted = 0
    for platform, meta in sent_meta.items():
        try:
            switch_tab(platform)
            time.sleep(0.5)
            baseline = meta.get("baseline_copies", 0)

            if wait_for_response(platform, baseline, max_wait=300):
                if extract_and_store(platform):
                    extracted += 1
            else:
                fail_package(platform, "timeout")
        except Exception as e:
            log.error(f"  [{platform}] Extract error: {e}")
            fail_package(platform, f"exception")

    total = time.time() - t_start
    summary = {
        "sent": len(sent_meta),
        "sent_to": list(sent_meta.keys()),
        "extracted": extracted,
        "elapsed": round(total, 1),
    }
    log.info(f"CYCLE: sent={summary['sent']}, extracted={extracted}, time={total:.1f}s")
    log_event("cycle_complete", **summary)
    print(json.dumps(summary))


def rolling_pipeline():
    """Continuous rolling pipeline. Ctrl+C to stop."""
    in_flight = {}  # platform -> metadata
    cycle = 0

    while _running:
        cycle += 1
        log.info(f"\n{'='*60}")
        log.info(f"ROLLING CYCLE {cycle}")
        log.info(f"{'='*60}")

        extracted = 0
        sent = 0

        for platform in PLATFORMS:
            if not _running:
                break

            # Extract previous if in-flight and ready
            if platform in in_flight:
                switch_tab(platform)
                time.sleep(0.5)

                if is_generating(platform):
                    log.info(f"  [{platform}] Still generating, skip")
                    continue

                baseline = in_flight[platform].get("baseline_copies", 0)
                if count_copy_buttons(platform) <= baseline:
                    log.info(f"  [{platform}] No new copies yet")
                    continue

                if extract_and_store(platform):
                    extracted += 1
                del in_flight[platform]

            # Send next
            try:
                result = send_to_platform(platform)
                if result:
                    in_flight[platform] = result
                    sent += 1
            except Exception as e:
                log.error(f"  [{platform}] Error: {e}")

        log.info(f"Cycle {cycle}: sent={sent}, extracted={extracted}")
        log_event("rolling_cycle", cycle=cycle, sent=sent, extracted=extracted)

        if sent == 0 and extracted == 0:
            if in_flight:
                log.info("Waiting 30s for in-flight...")
                time.sleep(30)
            else:
                log.info("ALL DONE")
                break


def test_platform(platform: str):
    """Test single platform end-to-end."""
    log.info(f"=== TEST: {platform} ===")

    result = send_to_platform(platform)
    if not result:
        log.error("Send failed!")
        return

    baseline = result.get("baseline_copies", 0)
    log.info("Waiting for response...")

    switch_tab(platform)
    if wait_for_response(platform, baseline, max_wait=300):
        if extract_and_store(platform):
            log.info("SUCCESS!")
        else:
            log.error("Store failed!")
    else:
        log.error("No response!")


def simple_test(platform: str):
    """Simple E2E test: send a math question, wait, extract. No package builder needed."""
    log.info(f"=== SIMPLE TEST: {platform} ===")
    test_msg = "What is the square root of 144? Reply with ONLY the number."

    switch_tab(platform)
    time.sleep(1.0)

    # Navigate to fresh chat
    open_new_chat(platform)

    # Count baseline copies
    if platform == "grok":
        baseline = _count_grok_copy_buttons()
    else:
        baseline = count_copy_buttons(platform)
    log.info(f"  [{platform}] Baseline copies: {baseline}")

    # Send
    if not send_message(platform, test_msg):
        log.error(f"  [{platform}] Send failed!")
        return

    log.info(f"  [{platform}] Message sent, waiting for response...")

    # Wait
    if not wait_for_response(platform, baseline, max_wait=60):
        log.error(f"  [{platform}] No response!")
        return

    # Extract
    time.sleep(2.0)
    response = extract_response(platform)
    if response:
        log.info(f"  [{platform}] Response ({len(response)} chars): {response[:200]}")
        if "12" in response:
            log.info(f"  [{platform}] PASS - correct answer found")
        else:
            log.warning(f"  [{platform}] WARN - '12' not found in response")
    else:
        log.error(f"  [{platform}] No response extracted!")


def check_status():
    """Check all platforms."""
    results = {}
    for platform in PLATFORMS:
        switch_tab(platform)
        time.sleep(0.5)
        gen = is_generating(platform)
        copies = count_copy_buttons(platform)
        results[platform] = {"generating": gen, "copies": copies}
        log.info(f"  [{platform}] generating={gen}, copies={copies}")
    print(json.dumps(results))


# ============================================================================
# Entry point
# ============================================================================

_running = True

def _signal_handler(sig, frame):
    global _running
    log.info("Signal received, stopping...")
    _running = False

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def loop_platform(platform: str, max_cycles: int = 100):
    """Continuous single-platform loop. Ctrl+C to stop."""
    successes = 0
    failures = 0
    cycle = 0

    while _running and cycle < max_cycles:
        cycle += 1
        log.info(f"\n{'='*60}")
        log.info(f"LOOP {cycle} [{platform}] (ok={successes}, fail={failures})")
        log.info(f"{'='*60}")

        try:
            result = send_to_platform(platform)
            if not result:
                log.info("No packages available, done!")
                break

            baseline = result.get("baseline_copies", 0)
            switch_tab(platform)

            if wait_for_response(platform, baseline, max_wait=600):
                if extract_and_store(platform):
                    successes += 1
                else:
                    failures += 1
            else:
                fail_package(platform, "timeout")
                failures += 1
        except Exception as e:
            log.error(f"Cycle error: {e}")
            failures += 1
            time.sleep(5)

        time.sleep(2)

    log.info(f"LOOP DONE: {successes} ok, {failures} fail in {cycle} cycles")


def multi_loop(platforms: list, max_cycles: int = 200):
    """Multi-platform interleaved loop. Sends to each platform, then waits and extracts.

    While platform A generates, sends to B, C, etc.
    After all sent, polls for responses and extracts as they complete.
    """
    successes = 0
    failures = 0
    cycle = 0

    while _running and cycle < max_cycles:
        cycle += 1
        log.info(f"\n{'='*60}")
        log.info(f"MULTI LOOP {cycle} (ok={successes}, fail={failures}, platforms={platforms})")
        log.info(f"{'='*60}")

        in_flight = {}

        # Phase 1: Send to all platforms
        for platform in platforms:
            if not _running:
                break
            try:
                result = send_to_platform(platform)
                if result:
                    in_flight[platform] = result
                else:
                    log.info(f"  [{platform}] No packages available")
            except Exception as e:
                log.error(f"  [{platform}] Send error: {e}")

        if not in_flight:
            log.info("No platforms had packages. Done!")
            break

        # Phase 2: Wait and extract from each (sequential to avoid tab conflicts)
        for platform in list(in_flight.keys()):
            if not _running:
                break
            try:
                switch_tab(platform)
                time.sleep(0.5)
                baseline = in_flight[platform].get("baseline_copies", 0)
                max_wait = 600 if platform == "chatgpt" else 300

                if wait_for_response(platform, baseline, max_wait=max_wait):
                    if extract_and_store(platform):
                        successes += 1
                    else:
                        failures += 1
                else:
                    fail_package(platform, "timeout")
                    failures += 1
            except Exception as e:
                log.error(f"  [{platform}] Extract error: {e}")
                fail_package(platform, "exception")
                failures += 1

        time.sleep(2)

    log.info(f"MULTI LOOP DONE: {successes} ok, {failures} fail in {cycle} cycles")


def main():
    setup()

    mode = sys.argv[1] if len(sys.argv) > 1 else "cycle-once"

    if mode == "cycle-once":
        cycle_once()
    elif mode == "rolling":
        rolling_pipeline()
    elif mode == "test":
        platform = sys.argv[2] if len(sys.argv) > 2 else "chatgpt"
        test_platform(platform)
    elif mode == "loop":
        platform = sys.argv[2] if len(sys.argv) > 2 else "grok"
        max_c = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        loop_platform(platform, max_c)
    elif mode == "multi":
        # Usage: multi grok,claude,chatgpt [max_cycles]
        plats = sys.argv[2].split(",") if len(sys.argv) > 2 else ["grok", "claude"]
        max_c = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        multi_loop(plats, max_c)
    elif mode == "simple-test":
        platform = sys.argv[2] if len(sys.argv) > 2 else "chatgpt"
        simple_test(platform)
    elif mode == "simple-test-all":
        for p in PLATFORMS:
            simple_test(p)
    elif mode == "status":
        check_status()
    else:
        print(f"Usage: {sys.argv[0]} [cycle-once|rolling|test <plat>|loop <plat> [N]|multi plat1,plat2 [N]|simple-test <plat>|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
