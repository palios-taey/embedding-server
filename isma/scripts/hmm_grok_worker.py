#!/usr/bin/env python3
"""HMM Grok Worker — Continuous single-platform enrichment loop.

Sends packages to Grok one at a time, waits for response, extracts, stores.
Designed to run in tmux on Thor.

Usage:
    python3 hmm_grok_worker.py [--max-cycles N] [--platform grok]
"""

import sys
import os
import time
import json
import subprocess
import logging
import signal

# Setup
DISPLAY = os.environ.get("DISPLAY", ":1")
os.environ["DISPLAY"] = DISPLAY
SCRIPT_DIR = os.path.expanduser("~/hmm/scripts")
TAEY_PATH = os.path.expanduser("~/taeys-hands")
sys.path.insert(0, TAEY_PATH)

import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

from core import input as inp
from core import clipboard as clip
from core import atspi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/hmm_grok_worker.log"),
    ]
)
log = logging.getLogger("grok_worker")

_running = True
def _sigint(sig, frame):
    global _running
    log.info("Stopping after current cycle...")
    _running = False
signal.signal(signal.SIGINT, _sigint)


def setup():
    inp.set_display(DISPLAY)
    clip.set_display(DISPLAY)
    Atspi.init()
    log.info("Grok Worker started | DISPLAY=%s", DISPLAY)


def build_package(platform="grok"):
    """Build next package via hmm_package_builder.py."""
    result = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "next", "--platform", platform],
        capture_output=True, text=True, timeout=60
    )
    for line in result.stdout.split("\n"):
        if line.startswith("Package ready:"):
            path = line.split(":", 1)[1].strip()
            log.info("Package: %s", os.path.basename(path))
            return path
    if "no items" in result.stdout.lower():
        log.info("No more items available")
    else:
        log.warning("Builder output: %s", result.stdout[:300])
    return None


def complete_package(platform="grok"):
    subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "complete", "--platform", platform],
        capture_output=True, text=True, timeout=30
    )


def fail_package(platform="grok", reason="unknown"):
    subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_package_builder.py",
         "fail", reason, "--platform", platform],
        capture_output=True, text=True, timeout=30
    )


def switch_to_grok():
    """Focus Firefox and switch to Grok tab."""
    result = subprocess.run(
        ["xdotool", "search", "--class", "firefox"],
        capture_output=True, text=True, env={"DISPLAY": DISPLAY}
    )
    wids = [w.strip() for w in result.stdout.strip().split('\n') if w.strip()]
    if wids:
        subprocess.run(["xdotool", "windowactivate", wids[0]], env={"DISPLAY": DISPLAY})
        time.sleep(0.3)
    inp.press_key("alt+4")
    time.sleep(0.5)


def open_new_chat():
    """Navigate to fresh Grok chat."""
    inp.press_key("ctrl+l")
    time.sleep(0.3)
    inp.type_text("https://grok.com", delay_ms=10)
    inp.press_key("Return")
    time.sleep(4.0)

    # Verify clean state (no copy buttons)
    doc = get_grok_doc()
    if doc:
        copies = find_copy_buttons(doc)
        if copies:
            log.info("Stale chat (%d copies), retrying...", len(copies))
            inp.press_key("ctrl+l")
            time.sleep(0.3)
            inp.type_text("https://grok.com", delay_ms=10)
            inp.press_key("Return")
            time.sleep(4.0)


def get_grok_doc():
    """Find Grok's document node."""
    firefox = atspi.find_firefox()
    if not firefox:
        return None
    queue = [(firefox, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > 8:
            continue
        try:
            if node.get_role_name() == "document web":
                try:
                    iface = node.get_document_iface()
                    if iface:
                        url = iface.get_document_attribute_value("DocURL") or ""
                        if "grok.com" in url:
                            return node
                except Exception:
                    pass
            for k in range(node.get_child_count()):
                child = node.get_child_at_index(k)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    # Fallback: use Firefox root but flag it
    log.warning("No Grok document found, using Firefox root")
    return firefox


def find_copy_buttons(scope):
    """Find copy buttons within scope."""
    results = []
    queue = [(scope, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > 25:
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
                results.append({"x": x, "y": y, "obj": node})
            for k in range(node.get_child_count()):
                child = node.get_child_at_index(k)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return results


def find_input():
    """Find Grok's editable input."""
    doc = get_grok_doc()
    if not doc:
        return None
    queue = [(doc, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > 20:
            continue
        try:
            role = node.get_role_name()
            if role in ("section", "paragraph"):
                ss = node.get_state_set()
                if ss and ss.contains(gi.repository.Atspi.StateType.EDITABLE):
                    return node
            elif role == "entry":
                return node
            for k in range(node.get_child_count()):
                child = node.get_child_at_index(k)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return None


def attach_file(file_path):
    """Attach file to Grok chat."""
    # Find attach button
    doc = get_grok_doc()
    if not doc:
        log.error("No Grok document for attach")
        return False

    queue = [(doc, 0)]
    attach_btn = None
    while queue:
        node, depth = queue.pop(0)
        if depth > 15:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()
            if "button" in role and "attach" in name:
                attach_btn = node
                break
            for k in range(node.get_child_count()):
                child = node.get_child_at_index(k)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue

    if not attach_btn:
        # Fallback: try Ctrl+U
        log.info("No attach button, trying Ctrl+U")
        inp.press_key("ctrl+u")
        time.sleep(2)
    else:
        # Click attach button
        try:
            action = attach_btn.get_action_iface()
            if action:
                action.do_action(0)
                time.sleep(1)
        except Exception:
            comp = attach_btn.get_component_iface()
            if comp:
                ext = comp.get_extents(0)
                inp.click_at(ext.x + ext.width // 2, ext.y + ext.height // 2)
                time.sleep(1)

        # Look for "Upload a file" menu item
        time.sleep(0.5)
        firefox = atspi.find_firefox()
        queue2 = [(firefox, 0)]
        upload_item = None
        while queue2:
            n2, d2 = queue2.pop(0)
            if d2 > 10:
                continue
            try:
                r2 = n2.get_role_name()
                nm2 = (n2.get_name() or "").lower()
                if "menu item" in r2 and "upload" in nm2 and "file" in nm2:
                    upload_item = n2
                    break
                for k in range(n2.get_child_count()):
                    ch = n2.get_child_at_index(k)
                    if ch:
                        queue2.append((ch, d2 + 1))
            except Exception:
                continue

        if upload_item:
            log.info("Upload item: '%s'", upload_item.get_name())
            try:
                action = upload_item.get_action_iface()
                if action:
                    action.do_action(0)
            except Exception:
                comp = upload_item.get_component_iface()
                if comp:
                    ext = comp.get_extents(0)
                    inp.click_at(ext.x + 5, ext.y + 5)
            time.sleep(2)

    # Handle file dialog
    time.sleep(1)
    inp.press_key("ctrl+l")
    time.sleep(0.3)
    abs_path = os.path.abspath(file_path)
    inp.type_text(abs_path, delay_ms=5)
    time.sleep(0.3)
    inp.press_key("Return")
    time.sleep(2)

    log.info("File attached: %s", os.path.basename(file_path))
    return True


def send_message(msg):
    """Type and send message."""
    input_node = find_input()
    if input_node:
        try:
            input_node.grab_focus()
            time.sleep(0.3)
            comp = input_node.get_component_iface()
            if comp:
                ext = comp.get_extents(0)
                inp.click_at(ext.x + ext.width // 2, ext.y + ext.height // 2)
                time.sleep(0.3)
        except Exception:
            pass

    inp.type_text(msg, delay_ms=3)
    time.sleep(0.5)
    inp.press_key("Return")
    time.sleep(1.0)
    log.info("Message sent")
    return True


def is_generating():
    """Check if Grok is generating (stop button visible)."""
    doc = get_grok_doc()
    if not doc:
        return False
    queue = [(doc, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > 15:
            continue
        try:
            role = node.get_role_name()
            name = (node.get_name() or "").lower()
            if "button" in role and "stop" in name:
                return True
            for k in range(node.get_child_count()):
                child = node.get_child_at_index(k)
                if child:
                    queue.append((child, depth + 1))
        except Exception:
            continue
    return False


def wait_for_response(baseline_copies=0, max_wait=600):
    """Wait until response is ready."""
    t0 = time.time()
    saw_generating = False

    while time.time() - t0 < max_wait:
        gen = is_generating()
        if gen:
            saw_generating = True

        doc = get_grok_doc()
        copies = len(find_copy_buttons(doc)) if doc else 0

        if not gen and copies > baseline_copies:
            elapsed = int(time.time() - t0)
            log.info("Response ready (%ds, %d copies)", elapsed, copies)
            return True

        if not gen and not saw_generating and (time.time() - t0) > 120:
            log.warning("Never generated, no copies after 120s")
            return False

        elapsed = int(time.time() - t0)
        if elapsed > 0 and elapsed % 30 == 0:
            status = "generating" if gen else "waiting"
            log.info("  %s... (%ds, copies=%d)", status, elapsed, copies)

        time.sleep(3)

    log.warning("Timed out after %ds", max_wait)
    return False


def extract_response():
    """Extract response via copy button."""
    inp.press_key("End")
    time.sleep(1)

    doc = get_grok_doc()
    if not doc:
        return None

    copies = find_copy_buttons(doc)
    if not copies:
        inp.press_key("End")
        time.sleep(1.5)
        doc = get_grok_doc()
        copies = find_copy_buttons(doc) if doc else []

    if not copies:
        log.warning("No copy buttons found")
        return None

    log.info("Found %d copy buttons", len(copies))

    # Click last copy button
    btn = copies[-1]
    clip.clear()
    time.sleep(0.2)
    try:
        action = btn["obj"].get_action_iface()
        if action:
            action.do_action(0)
            time.sleep(2)
    except Exception:
        pass

    content = clip.read()
    if content and len(content) >= 2:
        log.info("Extracted %d chars", len(content))
        return content

    # Retry with xdotool click fallback
    clip.clear()
    time.sleep(0.2)
    inp.click_at(btn["x"], btn["y"])
    time.sleep(2)
    content = clip.read()
    if content and len(content) >= 2:
        log.info("Extracted %d chars (xdotool fallback)", len(content))
        return content

    log.warning("Clipboard empty after extraction")
    return None


def store_response(response_text, platform="grok"):
    """Parse JSON and store via hmm_store_results.py."""
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
        # Save raw response for debugging
        raw_path = f"/tmp/hmm_raw_{platform}_{int(time.time())}.txt"
        with open(raw_path, "w") as f:
            f.write(response_text)
        log.error("JSON parse failed, raw saved to %s (%d chars)", raw_path, len(response_text))
        return False

    items = data.get("items", [])
    log.info("Parsed %d items from %s", len(items), data.get("package_id", "?"))

    tmp_path = f"/tmp/hmm_response_{platform}_{int(time.time())}.json"
    with open(tmp_path, "w") as f:
        json.dump(data, f)

    result = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/hmm_store_results.py",
         tmp_path, "--platform", platform],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        log.error("Store failed: %s", result.stderr[:200])
        return False

    log.info("Stored %d items", len(items))
    return True


def run_cycle(platform="grok"):
    """Single cycle: build → send → wait → extract → store."""
    t0 = time.time()

    # Build package
    pkg_path = build_package(platform)
    if not pkg_path:
        return None

    # Navigate to new chat
    switch_to_grok()
    open_new_chat()

    # Attach file
    if not attach_file(pkg_path):
        fail_package(platform, "attach_failed")
        return False

    time.sleep(1.5)

    # Send prompt
    prompt = "Analyze the attached context package. Follow the INSTRUCTIONS section at the top. Respond ONLY with the JSON format specified."
    if not send_message(prompt):
        fail_package(platform, "send_failed")
        return False

    # Wait for response (up to 10 min)
    if not wait_for_response(baseline_copies=0, max_wait=600):
        fail_package(platform, "timeout")
        return False

    time.sleep(2)

    # Extract
    response = extract_response()
    if not response:
        fail_package(platform, "no_response")
        return False

    # Store
    if store_response(response, platform):
        complete_package(platform)
        elapsed = time.time() - t0
        log.info("CYCLE COMPLETE: %s in %.1fs", os.path.basename(pkg_path), elapsed)
        return True
    else:
        fail_package(platform, "store_failed")
        return False


def main():
    setup()
    platform = "grok"
    max_cycles = 100

    if len(sys.argv) > 1 and sys.argv[1] == "--platform":
        platform = sys.argv[2]
    if "--max-cycles" in sys.argv:
        idx = sys.argv.index("--max-cycles")
        max_cycles = int(sys.argv[idx + 1])

    successes = 0
    failures = 0
    cycle = 0

    while _running and cycle < max_cycles:
        cycle += 1
        log.info("=" * 60)
        log.info("CYCLE %d (successes=%d, failures=%d)", cycle, successes, failures)
        log.info("=" * 60)

        try:
            result = run_cycle(platform)
            if result is None:
                log.info("No more packages, done!")
                break
            elif result:
                successes += 1
            else:
                failures += 1
                # Brief pause after failure
                time.sleep(5)
        except Exception as e:
            log.error("Cycle error: %s", e)
            failures += 1
            time.sleep(10)

        # Brief pause between cycles
        time.sleep(2)

    log.info("DONE: %d successes, %d failures in %d cycles", successes, failures, cycle)


if __name__ == "__main__":
    main()
