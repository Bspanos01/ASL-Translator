import cv2
import socket
import threading
import time
from dotenv import load_dotenv

load_dotenv()

from core.hand_tracker import HandTracker
from core.classifier import ASLClassifier
from core.letter_buffer import LetterBuffer
from core.autocomplete import get_suggestions_cached, build_sentence
from core.speaker import say, say_with_mood
from core.mood import detect_mood
from core.translator import (
    translate, cycle_language, get_current_language,
    get_voice_for_language, get_language_display,
)
from core.listener import start_recording, stop_recording, is_active, get_transcript, clear_transcript
from core import projector
from core.emergency import activate_emergency
from server.app import run_server, update_state
from ui.display import (
    draw_landmarks,
    draw_letter_overlay,
    draw_word_panel,
    draw_conversation_panel,
    draw_status_bar,
    draw_candidate_progress,
    draw_demo_label,
    draw_emergency_overlay,
    apply_mirror_flip,
)

# Constants
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RENDER_WIDTH = 1728
RENDER_HEIGHT = 1117
AUTOCOMPLETE_MIN_LETTERS = 3
MIN_CONFIDENCE = 0.50
DEBOUNCE_FRAMES = 8
COOLDOWN_FRAMES = 6


def main():
    # Initialize components
    tracker = HandTracker()
    classifier = ASLClassifier()
    buffer = LetterBuffer(debounce_frames=DEBOUNCE_FRAMES, min_confidence=MIN_CONFIDENCE, cooldown_frames=COOLDOWN_FRAMES)

    # Connect FREE-WILi (non-blocking)
    projector.connect()

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[Main] ERROR: Cannot open webcam")
        return

    # Fullscreen window
    cv2.namedWindow("SignBridge - ASL Translator", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("SignBridge - ASL Translator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start web server for phone view
    from server.app import SECRET_TOKEN
    server_port = 5050
    server_thread = threading.Thread(target=lambda: run_server(port=server_port), daemon=True)
    server_thread.start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    phone_url = f"http://{local_ip}:{server_port}/?token={SECRET_TOKEN}"
    print(f"[Server] Phone view: {phone_url}")
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(phone_url)
        qr.print_ascii(invert=True)
    except ImportError:
        pass

    print("[Main] ASL Translator running. Press ESC to quit.")

    suggestions = []
    last_word_str = ""
    sentence = ""
    translated_sentence = ""
    transcript = ""
    current_mood = "neutral"
    current_lang = "EN"
    f_held_frames = 0  # counts frames since last F keypress
    emergency_mode = False
    p_tap_times = []

    while True:
        ret, cam_frame = cap.read()
        if not ret:
            break

        cam_frame = apply_mirror_flip(cam_frame)

        # Hand tracking on original webcam frame
        norm_landmarks, raw_landmarks = tracker.get_both(cam_frame)
        letter, confidence = classifier.predict_letter(norm_landmarks)
        top_preds = classifier.get_top_predictions(norm_landmarks, n=3)

        # CONFIRM gesture (thumbs up) — auto-confirm current word
        if letter == "CONFIRM" and confidence >= MIN_CONFIDENCE:
            # Don't feed CONFIRM into the letter buffer — handle it directly
            # Use same debounce logic: count consecutive CONFIRM frames
            if not hasattr(buffer, '_confirm_count'):
                buffer._confirm_count = 0
            buffer._confirm_count += 1
            if buffer._confirm_count >= DEBOUNCE_FRAMES and buffer.get_word_so_far():
                word = buffer.confirm_word()
                if word:
                    sentence = build_sentence(buffer.words) if buffer.words else word
                    current_mood = detect_mood(sentence)
                    current_lang = get_current_language()
                    translated_sentence = translate(sentence, current_lang)
                    voice = get_voice_for_language(current_lang)
                    say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
                    suggestions = []
                    last_word_str = ""
                buffer._confirm_count = 0
            letter = None  # don't pass to buffer
        else:
            if hasattr(buffer, '_confirm_count'):
                buffer._confirm_count = 0

        # Buffer logic
        prev_word = buffer.get_word_so_far()
        buffer.add_letter(letter, confidence)
        status = buffer.get_status()
        current_word = status["word_so_far"]

        # Auto-confirm ILY as "I LOVE YOU" when it passes debounce
        if current_word != prev_word and current_word.endswith("ILY"):
            # Remove ILY from buffer and confirm as a word
            buffer.buffer = buffer.buffer[:-1]  # remove the "ILY" entry
            if buffer.buffer:
                # Confirm whatever was before ILY first
                buffer.confirm_word()
            buffer.buffer = ["I LOVE YOU"]
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            current_mood = detect_mood(sentence)
            current_lang = get_current_language()
            translated_sentence = translate(sentence, current_lang)
            voice = get_voice_for_language(current_lang)
            say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
            projector.show_word("I LOVE YOU")
            projector.flash_leds()
            suggestions = []
            last_word_str = ""
            status = buffer.get_status()
            current_word = status["word_so_far"]

        # Autocomplete when word changes and hits min length
        if current_word != last_word_str:
            last_word_str = current_word
            if len(current_word) >= AUTOCOMPLETE_MIN_LETTERS:
                suggestions = get_suggestions_cached(current_word)
            else:
                suggestions = []

        # Send current letter to FREE-WILi
        if letter and confidence > MIN_CONFIDENCE:
            projector.show_letter(letter, confidence)

        # Check for new transcript from listener — speak it back
        new_t = get_transcript()
        if new_t and new_t != transcript:
            transcript = new_t
            say(transcript)

        # Push state to web server for phone view
        update_state(
            asl_sentence=sentence,
            transcript=transcript,
            mood=current_mood,
            current_word=current_word,
            suggestions=suggestions,
            language=get_current_language(),
            is_listening=is_active(),
            translated=translated_sentence,
        )

        # Upscale webcam frame to render resolution for sharp text
        frame = cv2.resize(cam_frame, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)

        if emergency_mode:
            draw_emergency_overlay(frame)
        else:
            # Draw all UI layers
            draw_status_bar(frame, projector.is_connected(), classifier.is_ready(),
                            language_display=get_language_display())
            draw_landmarks(frame, raw_landmarks)
            draw_letter_overlay(frame, letter, confidence, top_preds)
            draw_demo_label(frame)
            draw_candidate_progress(frame, status["candidate"], status["candidate_progress"])
            draw_conversation_panel(frame, sentence, transcript, is_active(), current_mood,
                                    translated=translated_sentence, lang=get_current_language())
            draw_word_panel(frame, current_word, suggestions, buffer.get_word_count(),
                            language_display=get_language_display())

        cv2.imshow("SignBridge - ASL Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        # F key release detection — if recording and F not pressed for 3 frames, stop
        if is_active():
            if key in (ord('f'), ord('F')):
                f_held_frames = 0
            else:
                f_held_frames += 1
                if f_held_frames >= 10:
                    stop_recording()

        # ESC — dismiss emergency or quit
        if key == 27:
            if emergency_mode:
                emergency_mode = False
                print("[EMERGENCY] dismissed")
            else:
                break

        # SPACE — confirm word
        elif key == 32:
            word = buffer.confirm_word()
            if word:
                if buffer.words:
                    sentence = build_sentence(buffer.words)
                else:
                    sentence = word
                current_mood = detect_mood(sentence)
                current_lang = get_current_language()
                translated_sentence = translate(sentence, current_lang)
                voice = get_voice_for_language(current_lang)
                say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
                projector.show_word(word)
                projector.flash_leds()
            suggestions = []
            last_word_str = ""

        # D — delete last letter
        elif key in (ord('d'), ord('D')):
            buffer.backspace()

        # 1/2/3 — pick suggestion
        elif key == ord('1') and len(suggestions) > 0:
            buffer.buffer = list(suggestions[0].upper())
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            current_mood = detect_mood(sentence)
            current_lang = get_current_language()
            translated_sentence = translate(sentence, current_lang)
            voice = get_voice_for_language(current_lang)
            say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        elif key == ord('2') and len(suggestions) > 1:
            buffer.buffer = list(suggestions[1].upper())
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            current_mood = detect_mood(sentence)
            current_lang = get_current_language()
            translated_sentence = translate(sentence, current_lang)
            voice = get_voice_for_language(current_lang)
            say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        elif key == ord('3') and len(suggestions) > 2:
            buffer.buffer = list(suggestions[2].upper())
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            current_mood = detect_mood(sentence)
            current_lang = get_current_language()
            translated_sentence = translate(sentence, current_lang)
            voice = get_voice_for_language(current_lang)
            say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        # J key — insert J
        elif key in (ord('j'), ord('J')):
            buffer.buffer.append("J")
            buffer.last_added = "J"

        # Z key — insert Z
        elif key in (ord('z'), ord('Z')):
            buffer.buffer.append("Z")
            buffer.last_added = "Z"

        # S — speak entire sentence
        elif key in (ord('s'), ord('S')):
            if sentence:
                current_lang = get_current_language()
                translated_sentence = translate(sentence, current_lang)
                voice = get_voice_for_language(current_lang)
                say_with_mood(translated_sentence, current_mood, voice_override=voice, lang=current_lang)

        # L — cycle language
        elif key in (ord('l'), ord('L')):
            current_lang = cycle_language()

        # F — push-to-talk (hold to record, release to transcribe)
        elif key in (ord('f'), ord('F')):
            if not is_active():
                clear_transcript()
                transcript = ""
            start_recording()
            f_held_frames = 0

        # T — clear transcript
        elif key in (ord('t'), ord('T')):
            clear_transcript()
            transcript = ""

        # C — clear all
        elif key in (ord('c'), ord('C')):
            buffer.clear_all()
            sentence = ""
            suggestions = []
            last_word_str = ""
            projector.show_text("CLEARED")

        # X — clear last word
        elif key in (ord('x'), ord('X')):
            buffer.clear_last_word()
            sentence = build_sentence(buffer.words) if buffer.words else ""

        # P — triple tap for emergency
        elif key in (ord('p'), ord('P')):
            p_tap_times.append(time.time())
            p_tap_times = p_tap_times[-3:]
            if len(p_tap_times) == 3 and (p_tap_times[-1] - p_tap_times[0]) < 1.2:
                emergency_mode = True
                p_tap_times = []
                activate_emergency()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    projector.disconnect()


if __name__ == "__main__":
    main()
