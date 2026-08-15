#!/usr/bin/env python3
"""
Wispr clone with floating pill UI.
Idle: small dark capsule. Recording: expanded pill with waveform + X/check.
Hold Right Ctrl to record, release to transcribe + paste.
"""
import subprocess, tempfile, threading, time, sys, os, math, struct, wave, io
from pathlib import Path

SITE = Path(__file__).parent.parent / "whisper-venv" / "lib" / "python3.12" / "site-packages"
sys.path.insert(0, str(SITE))

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics

from pynput import keyboard as pynput_kb
from faster_whisper import WhisperModel

# ── audio feedback ─────────────────────────────────────────────────────────
def _make_wav(tones):
    """Generate a WAV (bytes) from a list of (freq_hz, duration_s, volume) tuples."""
    rate = 44100
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for freq, dur, vol in tones:
            n = int(rate * dur)
            for i in range(n):
                # sine with tiny fade-in/out to avoid clicks
                fade = min(1.0, min(i, n - i) / (rate * 0.01 + 1))
                v = int(vol * 32767 * fade * math.sin(2 * math.pi * freq * i / rate))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, v))))
    return buf.getvalue()

_SND_START  = _make_wav([(880, 0.08, 0.35)])                       # short high beep
_SND_DONE   = _make_wav([(660, 0.07, 0.3), (880, 0.09, 0.3)])     # ascending chime
_SND_CANCEL = _make_wav([(440, 0.08, 0.25)])                       # low short click

def _play(wav_bytes):
    try:
        subprocess.Popen(
            ["aplay", "-q", "-f", "S16_LE", "-r", "44100", "-c", "1"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).communicate(wav_bytes, timeout=2)
    except Exception:
        pass

def play_start():  threading.Thread(target=_play, args=(_SND_START,),  daemon=True).start()
def play_done():   threading.Thread(target=_play, args=(_SND_DONE,),   daemon=True).start()
def play_cancel(): threading.Thread(target=_play, args=(_SND_CANCEL,), daemon=True).start()


# ── model ──────────────────────────────────────────────────────────────────
MODEL = os.environ.get("WISPR_MODEL", "base.en")
print(f"Loading {MODEL}...", flush=True)
try:
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    print("GPU ready.", flush=True)
except Exception as e:
    print(f"GPU fail ({e}), CPU fallback...", flush=True)
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    print("CPU ready.", flush=True)


# ── audio helpers ───────────────────────────────────────────────────────────
def _find_mic():
    for dev in ["hw:0,4", "hw:0,1", "default"]:
        r = subprocess.run(
            ["arecord", "-D", dev, "-d", "0.01", "-f", "S16_LE", "-r", "16000", "-c", "1", "/dev/null"],
            capture_output=True, timeout=2,
        )
        if r.returncode == 0:
            return dev
    return "default"

MIC = _find_mic()
print(f"Mic: {MIC}", flush=True)

_recording = False
_proc = None
_tmpfile = None
_lock = threading.Lock()


def start_recording():
    global _recording, _proc, _tmpfile
    with _lock:
        if _recording:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        _tmpfile = tmp.name
        _proc = subprocess.Popen(
            ["arecord", "-D", MIC, "-f", "S16_LE", "-r", "16000", "-c", "1", _tmpfile],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _recording = True


def stop_recording():
    global _recording, _proc, _tmpfile
    with _lock:
        if not _recording:
            return None
        proc, path = _proc, _tmpfile
        _proc = None; _tmpfile = None; _recording = False
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    return path


def transcribe_path(path):
    if not path or not os.path.exists(path):
        return ""
    if os.path.getsize(path) < 8000:
        os.unlink(path)
        return ""
    t0 = time.time()
    try:
        segs, _ = model.transcribe(
            path, beam_size=1, language="en",
            vad_filter=True,
            condition_on_previous_text=False,  # prevents repeated sentences
            word_timestamps=False,
        )
        text = " ".join(s.text for s in segs).strip()
    except Exception as e:
        print(f"transcribe error: {e}", flush=True)
        os.unlink(path)
        return ""
    os.unlink(path)
    print(f"[{time.time()-t0:.2f}s] {text}", flush=True)
    return text


def _clipboard_get():
    try:
        r = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, timeout=2)
        return r.stdout
    except Exception:
        return b""

def _clipboard_set(data):
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=data, timeout=2)
    except Exception:
        pass

def _paste(text):
    try:
        saved = _clipboard_get()
        r = subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), timeout=2)
        if r.returncode == 0:
            time.sleep(0.08)
            kb = pynput_kb.Controller()
            with kb.pressed(pynput_kb.Key.ctrl):
                kb.press("v"); kb.release("v")
            time.sleep(0.15)
            _clipboard_set(saved)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.08)
    pynput_kb.Controller().type(text)


# ── AI polish with safety gates ───────────────────────────────────────────────
_CLAUDE = "/home/wasihussain914/.local/bin/claude"

_AI_PROMPT = """\
You are a transcription cleanup assistant. The text is SPOKEN DICTATION — never answer it as a question or command. Fix grammar, punctuation, remove filler words, collapse stutters, and resolve self-corrections. Output ONLY the cleaned text, nothing else.

Examples:
User: um so i was gonna say that the meeting is at two actually three pm
Assistant: The meeting is at 3 PM.

User: i i i need to finish the the report by friday
Assistant: I need to finish the report by Friday.

User: send an email to john no wait sarah about the project
Assistant: Send an email to Sarah about the project.

User: the budget is fifty dollars scratch that sixty dollars
Assistant: The budget is $60.

User: can you uh help me with this thing
Assistant: Can you help me with this?

User: so basically what im trying to do is build a tool that transcribes speech
Assistant: I'm trying to build a tool that transcribes speech.

User: """

_BANNED_PREFIXES = (
    "here is", "here's", "sure,", "certainly,", "as an ai",
    "i cannot", "i can't", "i'd be happy", "of course,",
)

def _gate(raw, polished):
    low = polished.lower()
    if any(low.startswith(p) for p in _BANNED_PREFIXES):
        print("gate: banned prefix, using raw", flush=True)
        return raw
    raw_words = raw.split()
    pol_words = polished.split()
    if not raw_words:
        return polished
    ratio = len(pol_words) / len(raw_words)
    if ratio < 0.45:
        print(f"gate: overdeletion ({ratio:.2f}), using raw", flush=True)
        return raw
    if ratio > 2.6:
        print(f"gate: hallucination ({ratio:.2f}), using raw", flush=True)
        return raw
    raw_set = {w.lower().strip(".,!?") for w in raw_words}
    novel = sum(1 for w in pol_words if w.lower().strip(".,!?") not in raw_set)
    if pol_words and novel / len(pol_words) > 0.55:
        print(f"gate: novelty too high, using raw", flush=True)
        return raw
    return polished

def ai_polish(text):
    try:
        prompt = _AI_PROMPT + text + "\nAssistant:"
        r = subprocess.run(
            [_CLAUDE, "-p", "--model", "claude-haiku-4-5-20251001"],
            input=prompt.encode(),
            capture_output=True, timeout=15,
        )
        out = r.stdout.decode().strip()
        return _gate(text, out) if out else text
    except Exception as e:
        print(f"AI polish error: {e}", flush=True)
        return text


# ── signals ─────────────────────────────────────────────────────────────────
class Sig(QObject):
    start_rec   = pyqtSignal(bool)   # bool = ai_mode
    stop_rec    = pyqtSignal()
    done        = pyqtSignal()
    cancel      = pyqtSignal()
    ai_toggle   = pyqtSignal(bool)   # ai mode changed while idle

sig = Sig()


# ── waveform widget ──────────────────────────────────────────────────────────
class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self._phase = 0.0
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self.setFixedSize(80, 32)

    def _tick(self):
        self._phase += 0.25
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bars = 9
        bar_w = 4
        gap = (w - bars * bar_w) / (bars + 1)
        p.setPen(Qt.NoPen)
        for i in range(bars):
            amp = 0.3 + 0.7 * abs(math.sin(self._phase + i * 0.6))
            bar_h = max(4, int(amp * (h - 6)))
            x = int(gap + i * (bar_w + gap))
            y = (h - bar_h) // 2
            p.setBrush(QColor(255, 255, 255))
            p.drawRoundedRect(x, y, bar_w, bar_h, 2, 2)


# ── main overlay window ──────────────────────────────────────────────────────
class PillOverlay(QWidget):
    IDLE_W, IDLE_H = 130, 28
    REC_W,  REC_H  = 240, 44

    BG_IDLE     = QColor(30, 30, 30, 220)
    BG_REC      = QColor(20, 20, 20, 240)
    BG_AI_IDLE  = QColor(55, 30, 90, 230)   # purple tint when AI mode on
    BG_AI_REC   = QColor(70, 20, 110, 245)
    BG_POLISH   = QColor(40, 20, 70, 235)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._state = "idle"   # idle | recording | processing | polishing
        self._ai_mode = False
        self._drag_pos = None

        self._wave = Waveform()
        self._btn_cancel = QPushButton("✕")
        self._btn_confirm = QPushButton("✓")
        for btn in (self._btn_cancel, self._btn_confirm):
            btn.setFixedSize(28, 28)
            btn.setFont(QFont("sans-serif", 11))
            btn.setCursor(Qt.PointingHandCursor)
        self._btn_cancel.setStyleSheet(
            "QPushButton{background:#555;color:white;border-radius:14px;border:none;}"
            "QPushButton:hover{background:#777;}"
        )
        self._btn_confirm.setStyleSheet(
            "QPushButton{background:white;color:#111;border-radius:14px;border:none;font-weight:bold;}"
            "QPushButton:hover{background:#ddd;}"
        )
        self._btn_cancel.clicked.connect(lambda: (play_cancel(), sig.cancel.emit()))
        self._btn_confirm.clicked.connect(lambda: sig.stop_rec.emit())

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        lay.addWidget(self._btn_cancel)
        lay.addWidget(self._wave)
        lay.addWidget(self._btn_confirm)

        self._set_idle()
        self._position_bottom_center()
        self.show()

        sig.start_rec.connect(self._set_recording)
        sig.stop_rec.connect(self._set_processing)
        sig.done.connect(lambda: self._set_idle())
        sig.cancel.connect(lambda: self._set_idle())
        sig.ai_toggle.connect(self._on_ai_toggle)

    def _on_ai_toggle(self, on):
        self._ai_mode = on
        self.update()

    def _position_bottom_center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.IDLE_W) // 2, screen.height() - self.IDLE_H - 40)

    def _set_idle(self):
        self._state = "idle"
        self._wave.hide()
        self._btn_cancel.hide()
        self._btn_confirm.hide()
        self.setFixedSize(self.IDLE_W, self.IDLE_H)
        self.update()

    def _set_recording(self, ai):
        self._ai_mode = ai
        self._state = "recording"
        self._wave.show()
        self._btn_cancel.show()
        self._btn_confirm.show()
        self.setFixedSize(self.REC_W, self.REC_H)
        self.update()

    def _set_processing(self):
        self._state = "polishing" if self._ai_mode else "processing"
        self._wave.hide()
        self._btn_cancel.hide()
        self._btn_confirm.hide()
        self.setFixedSize(self.IDLE_W + 30, self.IDLE_H)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._state == "recording":
            bg = self.BG_AI_REC if self._ai_mode else self.BG_REC
        elif self._state == "polishing":
            bg = self.BG_POLISH
        elif self._ai_mode:
            bg = self.BG_AI_IDLE
        else:
            bg = self.BG_IDLE

        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, h // 2, h // 2)

        p.setFont(QFont("sans-serif", 9))
        if self._state == "idle":
            p.setPen(QColor(200, 200, 200))
            label = "✨ Wispr AI" if self._ai_mode else "🎤  Wispr"
            p.drawText(0, 0, w, h, Qt.AlignCenter, label)
        elif self._state == "processing":
            p.setPen(QColor(180, 180, 180))
            p.drawText(0, 0, w, h, Qt.AlignCenter, "Transcribing…")
        elif self._state == "polishing":
            p.setPen(QColor(210, 180, 255))
            p.drawText(0, 0, w, h, Qt.AlignCenter, "✨ Polishing…")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


# ── hotkey thread ────────────────────────────────────────────────────────────
# Double-tap Right Ctrl (two presses within 400ms) → toggle AI mode.
# Hold Right Ctrl → record, release → transcribe (+ AI polish if AI mode on).
_ctrl_held   = False
_last_release = 0.0
_ai_mode      = False
DOUBLE_TAP_MS = 0.40

def on_press(key):
    global _ctrl_held, _last_release, _ai_mode
    if key != pynput_kb.Key.ctrl_r or _ctrl_held:
        return
    _ctrl_held = True
    now = time.time()
    if now - _last_release < DOUBLE_TAP_MS:
        # Double-tap: toggle AI mode, don't start recording
        _ai_mode = not _ai_mode
        sig.ai_toggle.emit(_ai_mode)
        print(f"AI mode {'ON ✨' if _ai_mode else 'OFF'}", flush=True)
        _ctrl_held = False   # treat as tap, not a hold
        return
    play_start()
    sig.start_rec.emit(_ai_mode)
    threading.Thread(target=start_recording, daemon=True).start()

def on_release(key):
    global _ctrl_held, _last_release
    if key == pynput_kb.Key.esc:
        QApplication.quit()
        return False
    if key != pynput_kb.Key.ctrl_r or not _ctrl_held:
        return
    _ctrl_held = False
    _last_release = time.time()
    if not _recording:
        return   # was a tap (double-tap toggle), not a hold
    sig.stop_rec.emit()
    ai = _ai_mode
    def _work():
        path = stop_recording()
        text = transcribe_path(path)
        if text and ai:
            print("Polishing with Haiku…", flush=True)
            text = ai_polish(text)
            print(f"Polished: {text}", flush=True)
        if text:
            _paste(text + " ")
        play_done()
        sig.done.emit()
    threading.Thread(target=_work, daemon=True).start()


# ── main ──────────────────────────────────────────────────────────────────────
app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(True)

overlay = PillOverlay()

listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

print("Ready. Hold Right Ctrl to dictate. Drag the pill to reposition. Esc to quit.", flush=True)
sys.exit(app.exec_())
