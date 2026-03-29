import cv2
import numpy as np
import mediapipe as mp

# ── Colors (BGR) ─────────────────────────────────────────────────────
BG_MAIN = (15, 10, 10)
BG_TOPBAR = (32, 22, 22)
BG_PANEL = (24, 17, 17)
BG_PILL = (32, 22, 22)
BG_PILL_FIRST = (46, 30, 30)

TEXT_PRIMARY = (240, 240, 240)
TEXT_GRAY = (120, 102, 102)
TEXT_WHITE = (255, 255, 255)

GREEN = (122, 219, 77)
BLUE = (219, 158, 77)

HAND_DOT = (122, 219, 77)
HAND_LINE = (80, 160, 50)

MOOD_COLORS = {
    "happy":    (122, 219, 77),
    "sad":      (77, 77, 219),
    "urgent":   (77, 77, 219),
    "angry":    (77, 77, 219),
    "neutral":  (120, 102, 102),
    "grateful": (180, 100, 180),
    "confused": (77, 219, 219),
}

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# Layout for ~1728x1117 render
TOPBAR_H = 48
SIGNED_H = 88
SAID_H = 88
TOP_UI_H = TOPBAR_H + SIGNED_H + SAID_H
BOTTOM_H = 150

_mp_hands = mp.solutions.hands
_HAND_CONNECTIONS = _mp_hands.HAND_CONNECTIONS


# ── Hand Skeleton ────────────────────────────────────────────────────

def draw_landmarks(frame, raw_landmarks):
    if raw_landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = []
    for lm in raw_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for c in _HAND_CONNECTIONS:
        cv2.line(frame, pts[c[0]], pts[c[1]], HAND_LINE, 2)
    for p in pts:
        cv2.circle(frame, p, 5, HAND_DOT, -1)


# ── Letter Prediction HUD ───────────────────────────────────────────

def draw_letter_overlay(frame, letter, confidence, top_predictions=None):
    if not letter:
        return
    h, w = frame.shape[:2]
    bx, by = 32, h - BOTTOM_H - 140
    bw, bh = 240, 120

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Large letter
    cv2.putText(frame, str(letter), (bx + 16, by + 74), FONT, 3.5, TEXT_WHITE, 3)

    # Confidence
    conf = f"{int(confidence * 100)}%"
    cv2.putText(frame, conf, (bx + 130, by + 60), FONT, 1.0, GREEN, 2)

    # Alts
    if top_predictions and len(top_predictions) > 1:
        alt_x = bx + 20
        for pred_letter, _ in top_predictions[1:3]:
            cv2.putText(frame, pred_letter, (alt_x, by + 105), FONT, 0.85, TEXT_GRAY, 1)
            alt_x += 42


# ── Candidate Progress ───────────────────────────────────────────────

def draw_candidate_progress(frame, candidate_letter, progress_float):
    if candidate_letter is None or progress_float <= 0:
        return
    h, w = frame.shape[:2]
    bar_y = h - BOTTOM_H - 3
    fill = int(w * min(progress_float, 1.0))
    color = TEXT_WHITE if progress_float >= 0.85 else GREEN
    cv2.rectangle(frame, (0, bar_y), (fill, bar_y + 3), color, -1)


# ── YOU SIGNED + THEY SAID (top panels) ─────────────────────────────

def draw_conversation_panel(frame, asl_sentence, transcript, is_listening,
                            mood="neutral", translated=None, lang="EN"):
    h, w = frame.shape[:2]
    px = 36

    # ── YOU SIGNED ──
    ys = TOPBAR_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ys), (w, ys + SIGNED_H), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.rectangle(frame, (0, ys), (5, ys + SIGNED_H), GREEN, -1)

    cv2.putText(frame, "YOU SIGNED", (px, ys + 22), FONT_S, 0.50, GREEN, 1)

    if asl_sentence:
        cv2.putText(frame, asl_sentence, (px, ys + 58), FONT, 1.05, TEXT_PRIMARY, 2)

        # Mood pill
        mood_color = MOOD_COLORS.get(mood, MOOD_COLORS["neutral"])
        mood_str = f"[{mood.upper()}]"
        sent_size = cv2.getTextSize(asl_sentence, FONT, 1.05, 2)[0]
        pill_x = px + sent_size[0] + 16
        pill_y = ys + 38
        pill_size = cv2.getTextSize(mood_str, FONT, 0.55, 1)[0]
        pw = pill_size[0] + 14
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pw, pill_y + 26), mood_color, 1)
        cv2.putText(frame, mood_str, (pill_x + 7, pill_y + 20), FONT, 0.55, mood_color, 1)

        if lang not in ("EN", "EL") and translated and translated != asl_sentence:
            cv2.putText(frame, translated, (px, ys + 80), FONT, 0.65, GREEN, 1)
    else:
        cv2.putText(frame, "Start signing...", (px, ys + 58), FONT, 1.05, TEXT_GRAY, 1)

    cv2.line(frame, (0, ys + SIGNED_H), (w, ys + SIGNED_H), (40, 35, 35), 1)

    # ── THEY SAID ──
    ts = ys + SIGNED_H
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, ts), (w, ts + SAID_H), BG_PANEL, -1)
    cv2.addWeighted(overlay2, 0.90, frame, 0.10, 0, frame)
    cv2.rectangle(frame, (0, ts), (5, ts + SAID_H), BLUE, -1)

    cv2.putText(frame, "THEY SAID", (px, ts + 22), FONT_S, 0.50, BLUE, 1)

    # Mic icon
    mic_x = w - 40
    mic_cy = ts + SAID_H // 2
    mic_c = BLUE if is_listening else TEXT_GRAY
    cv2.rectangle(frame, (mic_x - 4, mic_cy - 10), (mic_x + 4, mic_cy + 5), mic_c, -1)
    cv2.ellipse(frame, (mic_x, mic_cy - 10), (4, 4), 0, 180, 360, mic_c, -1)
    cv2.ellipse(frame, (mic_x, mic_cy + 3), (8, 7), 0, 0, 180, mic_c, 1)
    cv2.line(frame, (mic_x, mic_cy + 10), (mic_x, mic_cy + 15), mic_c, 1)

    if is_listening:
        cv2.circle(frame, (px, ts + 56), 5, BLUE, -1)
        cv2.putText(frame, "Listening...", (px + 16, ts + 60), FONT, 0.90, BLUE, 1)
    elif transcript:
        if len(transcript) <= 50:
            cv2.putText(frame, transcript, (px, ts + 60), FONT, 0.90, TEXT_PRIMARY, 1)
        else:
            mid = len(transcript) // 2
            sp = transcript.rfind(" ", 0, mid + 10)
            if sp == -1:
                sp = mid
            cv2.putText(frame, transcript[:sp].strip(), (px, ts + 48), FONT, 0.75, TEXT_PRIMARY, 1)
            cv2.putText(frame, transcript[sp:].strip(), (px, ts + 72), FONT, 0.75, TEXT_PRIMARY, 1)
    else:
        cv2.putText(frame, "Press F to listen...", (px, ts + 60), FONT, 0.90, TEXT_GRAY, 1)

    cv2.line(frame, (0, ts + SAID_H), (w, ts + SAID_H), (40, 35, 35), 1)


# ── Bottom Panel ─────────────────────────────────────────────────────

def draw_word_panel(frame, buffer_str, suggestions, word_count=0,
                    language_display="EN -- English"):
    h, w = frame.shape[:2]
    pt = h - BOTTOM_H
    px = 36

    cv2.rectangle(frame, (0, pt), (w, h), BG_MAIN, -1)

    # Current word large
    display = buffer_str if buffer_str else ""
    cv2.putText(frame, display, (px, pt + 50), FONT, 1.6, TEXT_PRIMARY, 2)

    word_size = cv2.getTextSize(display, FONT, 1.6, 2)[0] if display else (0, 0)
    cursor_x = px + word_size[0] + 4
    cv2.line(frame, (cursor_x, pt + 14), (cursor_x, pt + 54), TEXT_PRIMARY, 2)

    # Suggestion pills
    pill_y = pt + 86
    pill_x = px
    for i, suggestion in enumerate(suggestions[:3]):
        if suggestion:
            num_text = f"[{i + 1}]"
            full = f"{num_text} {suggestion}"
            text_size = cv2.getTextSize(full, FONT, 0.60, 1)[0]
            pw = max(text_size[0] + 28, 140)
            bg = BG_PILL_FIRST if i == 0 else BG_PILL
            cv2.rectangle(frame, (pill_x, pill_y - 14), (pill_x + pw, pill_y + 18), bg, -1)
            cv2.putText(frame, num_text, (pill_x + 12, pill_y + 10), FONT, 0.55, TEXT_GRAY, 1)
            num_size = cv2.getTextSize(num_text, FONT, 0.55, 1)[0]
            cv2.putText(frame, suggestion, (pill_x + 12 + num_size[0] + 6, pill_y + 10),
                        FONT, 0.60, TEXT_PRIMARY, 1)
            pill_x += pw + 12

    # Keybind hints
    hints = "SPACE=confirm - D=delete - 1/2/3=pick - F=listen - L=lang - S=speak - C=clear - ESC=quit - "
    cv2.putText(frame, hints, (px, h - 12), FONT_S, 0.42, TEXT_GRAY, 1)
    hints_size = cv2.getTextSize(hints, FONT_S, 0.42, 1)[0]
    cv2.putText(frame, "PPP=EMERGENCY", (px + hints_size[0], h - 12), FONT_S, 0.42, (100, 100, 180), 1)


# ── Title Bar ────────────────────────────────────────────────────────

def draw_status_bar(frame, wili_connected, model_loaded,
                    language_display="EN -- English"):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, TOPBAR_H), BG_TOPBAR, -1)

    cv2.putText(frame, "SignBridge", (24, TOPBAR_H // 2 + 9), FONT, 0.75, TEXT_WHITE, 1)

    # Language
    lang_code = language_display[:2]
    flag = "GR" if lang_code == "EL" else lang_code

    # Globe
    gx = w - 72
    gy = TOPBAR_H // 2
    cv2.circle(frame, (gx, gy), 10, TEXT_GRAY, 1)
    cv2.line(frame, (gx - 10, gy), (gx + 10, gy), TEXT_GRAY, 1)
    cv2.ellipse(frame, (gx, gy), (4, 10), 0, 0, 360, TEXT_GRAY, 1)

    cv2.putText(frame, flag, (gx + 16, gy + 7), FONT, 0.60, TEXT_WHITE, 1)

    cv2.line(frame, (0, TOPBAR_H), (w, TOPBAR_H), (40, 35, 35), 1)


# ── Demo Label ───────────────────────────────────────────────────────

def draw_demo_label(frame):
    h, w = frame.shape[:2]
    y = h - BOTTOM_H - 16
    cv2.putText(frame, "SignBridge", (w - 160, y - 22), FONT, 0.50, TEXT_GRAY, 1)
    cv2.putText(frame, "GrizzHacks 8", (w - 160, y), FONT, 0.50, (60, 50, 50), 1)


# ── Emergency Overlay ────────────────────────────────────────────────

def draw_emergency_overlay(frame):
    h, w = frame.shape[:2]

    # Full red background
    frame[:] = (0, 0, 180)

    # Inner border
    cv2.rectangle(frame, (8, 8), (w - 8, h - 8), (0, 0, 140), 6)

    # "EMERGENCY" — large centered
    text = "EMERGENCY"
    size = cv2.getTextSize(text, FONT, 4.5, 5)[0]
    x = (w - size[0]) // 2
    y = int(h * 0.35)
    cv2.putText(frame, text, (x, y), FONT, 4.5, TEXT_WHITE, 5)

    # White line below
    line_y = y + 30
    cv2.line(frame, (40, line_y), (w - 40, line_y), TEXT_WHITE, 2)

    # "CALL 911"
    text2 = "CALL 911"
    size2 = cv2.getTextSize(text2, FONT, 3.0, 4)[0]
    x2 = (w - size2[0]) // 2
    cv2.putText(frame, text2, (x2, int(h * 0.55)), FONT, 3.0, TEXT_WHITE, 4)

    # "I NEED IMMEDIATE HELP"
    text3 = "I NEED IMMEDIATE HELP"
    size3 = cv2.getTextSize(text3, FONT, 1.2, 2)[0]
    x3 = (w - size3[0]) // 2
    cv2.putText(frame, text3, (x3, int(h * 0.70)), FONT, 1.2, (200, 200, 255), 2)

    # "Press ESC to dismiss"
    text4 = "Press ESC to dismiss"
    size4 = cv2.getTextSize(text4, FONT, 0.6, 1)[0]
    x4 = (w - size4[0]) // 2
    cv2.putText(frame, text4, (x4, int(h * 0.88)), FONT, 0.6, (160, 160, 200), 1)


# ── Utility ──────────────────────────────────────────────────────────

def apply_mirror_flip(frame):
    return cv2.flip(frame, 1)
