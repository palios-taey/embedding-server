"""Marionette helper for focus-free Firefox automation.

Used on machines where Firefox lacks native WM keyboard focus
(e.g., Thor with GNOME anti-focus-stealing).

Provides:
  - Tab switching via WebDriver protocol
  - URL navigation
  - JS execution (text injection, button clicking)
  - File content injection via execCommand('insertText')
  - Response extraction directly from DOM (no clipboard needed)
"""
import socket
import json
import time
import logging

log = logging.getLogger("hmm_pipeline")

MARIONETTE_PORT = 2828
PLATFORM_TAB_NAMES = {
    "chatgpt": "chatgpt",
    "claude": "claude",
    "gemini": "gemini",
    "grok": "grok",
    "perplexity": "perplexity",
}

NEW_CHAT_URLS = {
    "chatgpt": "https://chatgpt.com/",
    "claude": "https://claude.ai/new",
    "gemini": "https://gemini.google.com/",
    "grok": "https://grok.com",
    "perplexity": "https://perplexity.ai/",
}


class MarionetteHelper:
    """Focus-free Firefox automation via Marionette protocol."""

    def __init__(self, host='127.0.0.1', port=MARIONETTE_PORT, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.msg_id = 0
        self.session_id = None
        self.tab_handles = {}  # platform -> handle

    def connect(self) -> bool:
        """Connect to Marionette and create session.

        Handles stale sessions: if NewSession fails, disconnects and retries
        with a fresh TCP connection (Marionette allows only one session).
        """
        for attempt in range(2):
            try:
                if self.sock:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))

                # Read handshake
                hs = self._read_response()
                if not hs:
                    log.warning(f"Marionette: empty handshake (attempt {attempt+1})")
                    if attempt == 0:
                        # Stale session — try to clean up
                        try:
                            self._send('WebDriver:DeleteSession')
                        except Exception:
                            pass
                        try:
                            self.sock.close()
                        except Exception:
                            pass
                        self.sock = None
                        time.sleep(2)
                        continue
                    return False

                # Try to delete any existing session (idempotent)
                if attempt == 0:
                    try:
                        self._send('WebDriver:DeleteSession')
                    except Exception:
                        pass
                    # Reconnect after delete (Marionette closes socket on DeleteSession)
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(self.timeout)
                    self.sock.connect((self.host, self.port))
                    self._read_response()  # re-read handshake

                # Create new session
                resp = self._send('WebDriver:NewSession', {
                    'capabilities': {'alwaysMatch': {'acceptInsecureCerts': True}}
                })
                if resp and len(resp) > 3 and resp[2] is None and resp[3]:
                    val = resp[3]
                    self.session_id = val.get('sessionId') if isinstance(val, dict) else None
                    if self.session_id:
                        log.info(f"Marionette connected (session={self.session_id[:12]})")
                        return True

                log.warning(f"Marionette: NewSession failed (attempt {attempt+1}): {resp}")
                if attempt == 0:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = None
                    time.sleep(2)
                    continue
                return False

            except Exception as e:
                log.error(f"Marionette connect failed (attempt {attempt+1}): {e}")
                if attempt == 0:
                    try:
                        if self.sock:
                            self.sock.close()
                    except Exception:
                        pass
                    self.sock = None
                    time.sleep(2)
                    continue
                return False

        return False

    def disconnect(self):
        """Close connection."""
        if self.sock:
            try:
                self._send('WebDriver:DeleteSession')
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def reconnect(self) -> bool:
        """Reconnect after connection loss."""
        self.disconnect()
        time.sleep(1)
        return self.connect()

    def discover_tabs(self) -> dict:
        """Map platform names to tab handles."""
        resp = self._send('WebDriver:GetWindowHandles')
        handles = self._val(resp)
        if not isinstance(handles, list):
            log.warning(f"Unexpected handles: {handles}")
            return {}

        self.tab_handles = {}
        for h in handles:
            self._send('WebDriver:SwitchToWindow', {'handle': h})
            title = self._val(self._send('WebDriver:GetTitle')) or ''
            for platform, pattern in PLATFORM_TAB_NAMES.items():
                if pattern in title.lower() and platform not in self.tab_handles:
                    self.tab_handles[platform] = h
                    break

        log.info(f"Marionette tabs: {list(self.tab_handles.keys())}")
        return self.tab_handles

    def switch_tab(self, platform: str) -> bool:
        """Switch to a platform tab."""
        handle = self.tab_handles.get(platform)
        if not handle:
            log.warning(f"No tab handle for {platform}")
            return False
        self._send('WebDriver:SwitchToWindow', {'handle': handle})
        return True

    def navigate(self, url: str, wait: float = 8.0) -> bool:
        """Navigate to URL and wait for page load."""
        resp = self._send('WebDriver:Navigate', {'url': url})
        error = resp[2] if resp and len(resp) > 2 else None
        if error:
            log.warning(f"Navigate error: {error}")
            return False
        time.sleep(wait)
        return True

    def open_new_chat(self, platform: str) -> bool:
        """Switch to platform tab and navigate to new chat URL."""
        if not self.switch_tab(platform):
            return False
        url = NEW_CHAT_URLS.get(platform)
        if not url:
            return False
        return self.navigate(url)

    def inject_text(self, text: str) -> str:
        """Inject text into the active contenteditable via execCommand.

        This triggers React's change handlers properly, unlike clipboard paste
        or AT-SPI editable_text.insert_text().

        Returns result string or error.
        """
        # Escape for JS string literal
        escaped = (text
                   .replace("\\", "\\\\")
                   .replace("'", "\\'")
                   .replace("\n", "\\n")
                   .replace("\r", ""))

        js = f"""
        var eds = document.querySelectorAll('[contenteditable="true"]');
        if (eds.length === 0) return 'NO_CONTENTEDITABLE';
        var ed = eds[eds.length - 1];
        ed.focus();
        document.execCommand('selectAll', false, null);
        document.execCommand('delete', false, null);
        document.execCommand('insertText', false, '{escaped}');
        ed.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText'}}));
        return 'OK:' + ed.textContent.length;
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        return str(self._val(resp))

    def inject_text_chunked(self, text: str, chunk_size: int = 50000) -> str:
        """Inject large text in chunks to avoid JS string literal limits."""
        if len(text) <= chunk_size:
            return self.inject_text(text)

        # First chunk: clear and insert
        first = text[:chunk_size]
        result = self.inject_text(first)
        if not result.startswith('OK'):
            return result

        # Subsequent chunks: append via execCommand
        offset = chunk_size
        while offset < len(text):
            chunk = text[offset:offset + chunk_size]
            escaped = (chunk
                       .replace("\\", "\\\\")
                       .replace("'", "\\'")
                       .replace("\n", "\\n")
                       .replace("\r", ""))
            js = f"""
            var eds = document.querySelectorAll('[contenteditable="true"]');
            if (eds.length === 0) return 'NO_CONTENTEDITABLE';
            var ed = eds[eds.length - 1];
            // Move cursor to end
            var sel = window.getSelection();
            sel.selectAllChildren(ed);
            sel.collapseToEnd();
            document.execCommand('insertText', false, '{escaped}');
            return 'OK:' + ed.textContent.length;
            """
            resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
            result = str(self._val(resp))
            if not result.startswith('OK'):
                return f'CHUNK_FAIL@{offset}: {result}'
            offset += chunk_size
            time.sleep(0.2)

        return f'OK:{len(text)}'

    def click_send(self, platform: str = "claude") -> str:
        """Click the Send/Submit button via JS."""
        js = """
        var btns = document.querySelectorAll('button');
        for (var b of btns) {
            var al = (b.getAttribute('aria-label') || '').toLowerCase();
            if (al.match(/send|submit/) && !b.disabled) { b.click(); return 'CLICKED: ' + al; }
        }
        // Fallback: data-testid
        var tb = document.querySelectorAll('button[data-testid*="send"]');
        if (tb.length > 0 && !tb[0].disabled) { tb[0].click(); return 'CLICKED_TESTID'; }
        return 'NO_SEND_BUTTON';
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        return str(self._val(resp))

    def count_elements(self, selector: str) -> int:
        """Count DOM elements matching a CSS selector."""
        js = f"return document.querySelectorAll('{selector}').length;"
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        val = self._val(resp)
        return int(val) if val is not None else 0

    def is_generating(self) -> bool:
        """Check if the AI is currently generating (stop button visible)."""
        js = """
        var btns = document.querySelectorAll('button');
        for (var b of btns) {
            var al = (b.getAttribute('aria-label') || '').toLowerCase();
            if (al.match(/stop/)) return true;
        }
        return false;
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        return self._val(resp) == True

    def count_copy_buttons(self) -> int:
        """Count copy buttons (response indicators)."""
        js = """
        var count = 0;
        var btns = document.querySelectorAll('button');
        for (var b of btns) {
            var al = (b.getAttribute('aria-label') || '').toLowerCase();
            if (al.includes('copy')) count++;
        }
        return count;
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        val = self._val(resp)
        return int(val) if val is not None else 0

    def get_response_text(self) -> str | None:
        """Extract the last AI response text directly from the DOM.

        Tries multiple strategies:
        1. Walk up from last Copy button to find response container
        2. Look for response-specific DOM structures
        3. Get innerText of last prose/markdown block

        Returns the response text or None if not found.
        """
        # Strategy 1: Walk up from last Copy button to find response container
        result = self._exec_js("""
        var copies = [];
        document.querySelectorAll('button').forEach(function(b) {
            var al = (b.getAttribute('aria-label') || '').toLowerCase();
            if (al.includes('copy')) copies.push(b);
        });
        if (copies.length === 0) return null;

        var btn = copies[copies.length - 1];
        var el = btn.parentElement;
        var bestText = null;
        var bestLen = 0;

        // Walk up, looking for the narrowest container with substantial text
        for (var i = 0; i < 15 && el; i++) {
            var text = el.innerText || el.textContent || '';
            text = text.trim();

            // Skip tiny containers and the whole page
            if (text.length > 30 && text.length < 500000) {
                // Check: this container should have the copy button but not be the whole chat
                var innerCopies = el.querySelectorAll('button[aria-label*="Copy"], button[aria-label*="copy"]');
                if (innerCopies.length > 0 && innerCopies.length <= 3) {
                    // This is likely a single message container
                    if (text.length > bestLen) {
                        bestText = text;
                        bestLen = text.length;
                    }
                    // Don't go further up - we found the best container
                    if (i > 3) break;
                }
            }
            el = el.parentElement;
        }

        return bestText;
        """)
        if result and len(result) > 20:
            return self._clean_response_text(result)

        # Strategy 2: Look for markdown/prose content blocks
        result = self._exec_js("""
        // Find all prose/markdown blocks (Claude uses these for responses)
        var blocks = document.querySelectorAll('.prose, [class*="markdown"], [class*="message-content"]');
        if (blocks.length === 0) return null;
        var last = blocks[blocks.length - 1];
        return last.innerText || last.textContent;
        """)
        if result and len(result) > 20:
            return self._clean_response_text(result)

        log.warning("Marionette: could not extract response from DOM")
        return None

    def _clean_response_text(self, text: str) -> str:
        """Clean extracted response text by removing thinking markers."""
        # Claude's thinking section appears before the actual response
        # Common patterns: "Thinking about..." text followed by "Done" then the response
        lines = text.split('\n')
        cleaned = []
        skip_thinking = True

        for line in lines:
            stripped = line.strip()
            # Skip common thinking prefixes
            if skip_thinking:
                if stripped.lower() in ('done', 'done.'):
                    skip_thinking = False
                    continue
                # If line looks like thinking output, skip
                thinking_markers = [
                    'thinking about', 'let me', 'i need to', 'analyzing',
                    'reading the', 'looking at', 'considering',
                    'synthesized context', 'read the', 'view truncated',
                ]
                if any(stripped.lower().startswith(m) for m in thinking_markers):
                    continue
                # If we see what looks like actual content (JSON, structured text),
                # stop skipping
                if stripped.startswith('{') or stripped.startswith('[') or stripped.startswith('#'):
                    skip_thinking = False

            if not skip_thinking:
                cleaned.append(line)

        result = '\n'.join(cleaned).strip()
        # If cleaning removed EVERYTHING, return original
        if not result:
            return text.strip()
        return result

    def click_continue(self) -> bool:
        """Click Continue Generating button if present."""
        js = """
        var btns = document.querySelectorAll('button');
        for (var b of btns) {
            var text = (b.textContent || '').toLowerCase();
            var label = (b.getAttribute('aria-label') || '').toLowerCase();
            if (text.includes('continue') || label.includes('continue')) {
                b.click();
                return true;
            }
        }
        return false;
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        return self._val(resp) == True

    def has_continue_button(self) -> bool:
        """Check if Continue Generating button is present."""
        js = """
        var btns = document.querySelectorAll('button');
        for (var b of btns) {
            var text = (b.textContent || '').toLowerCase();
            var label = (b.getAttribute('aria-label') || '').toLowerCase();
            if (text.includes('continue') || label.includes('continue')) return true;
        }
        return false;
        """
        resp = self._send('WebDriver:ExecuteScript', {'script': js, 'args': []})
        return self._val(resp) == True

    def get_page_title(self) -> str:
        """Get current page title."""
        resp = self._send('WebDriver:GetTitle')
        return str(self._val(resp) or '')

    def get_page_url(self) -> str:
        """Get current page URL via JS (WebDriver:GetCurrentUrl not supported)."""
        result = self._exec_js("return window.location.href;")
        return str(result or '')

    # -- Internal --

    def _send(self, method, params=None):
        """Send Marionette command and read response."""
        self.msg_id += 1
        cmd = [0, self.msg_id, method, params or {}]
        data = json.dumps(cmd)
        packet = f"{len(data)}:{data}".encode()
        try:
            self.sock.sendall(packet)
            return self._read_response()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            log.error(f"Marionette connection error: {e}")
            return None

    def _read_response(self):
        """Read length-prefixed JSON response."""
        buf = b""
        while b":" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            buf += chunk

        colon_idx = buf.index(b":")
        size = int(buf[:colon_idx])
        data = buf[colon_idx + 1:]

        while len(data) < size:
            data += self.sock.recv(min(size - len(data), 65536))

        return json.loads(data[:size])

    def _val(self, resp):
        """Extract value from Marionette response [type, id, error, value]."""
        if not resp or len(resp) < 4:
            return None
        if resp[2]:  # error
            return None
        val = resp[3]
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val

    def _exec_js(self, script):
        """Execute synchronous JavaScript and return the value."""
        resp = self._send('WebDriver:ExecuteScript', {'script': script, 'args': []})
        return self._val(resp)


def is_marionette_available(host='127.0.0.1', port=MARIONETTE_PORT) -> bool:
    """Quick check if Marionette is listening."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False
