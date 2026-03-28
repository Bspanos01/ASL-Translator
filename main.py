import cv2
from dotenv import load_dotenv

load_dotenv()

from core.hand_tracker import HandTracker
from core.classifier import ASLClassifier
from core.letter_buffer import LetterBuffer
from core.autocomplete import get_suggestions_cached, build_sentence
from core.speaker import say_letter, say_word, say_sentence
from core import projector
from ui.display import (
    draw_landmarks,
    draw_letter_overlay,
    draw_word_panel,
    draw_sentence,
    draw_status_bar,
    draw_candidate_progress,
    apply_mirror_flip,
)

# Constants
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
AUTOCOMPLETE_MIN_LETTERS = 3
MIN_CONFIDENCE = 0.50
DEBOUNCE_FRAMES = 8


def main():
    # Initialize components
    tracker = HandTracker()
    classifier = ASLClassifier()
    buffer = LetterBuffer(debounce_frames=DEBOUNCE_FRAMES, min_confidence=MIN_CONFIDENCE)

    # Connect FREE-WILi (non-blocking)
    projector.connect()

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[Main] ERROR: Cannot open webcam")
        return

    print("[Main] ASL Translator running. Press ESC to quit.")

    suggestions = []
    last_word_str = ""
    sentence = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_mirror_flip(frame)

        # Hand tracking + classification
        norm_landmarks, raw_landmarks = tracker.get_both(frame)
        letter, confidence = classifier.predict_letter(norm_landmarks)
        top_preds = classifier.get_top_predictions(norm_landmarks, n=3)

        # Buffer logic
        prev_len = len(buffer.buffer)
        buffer.add_letter(letter, confidence)
        if len(buffer.buffer) > prev_len:
            say_letter(buffer.buffer[-1])
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

        # Draw all UI layers
        draw_landmarks(frame, raw_landmarks)
        draw_sentence(frame, sentence)
        draw_letter_overlay(frame, letter, confidence, top_preds)
        draw_candidate_progress(frame, status["candidate"], status["candidate_progress"])
        draw_word_panel(frame, current_word, suggestions, buffer.get_word_count())
        draw_status_bar(frame, projector.is_connected(), classifier.is_ready())

        cv2.imshow("ASL Translator — GrizzHacks 8", frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC — quit
        if key == 27:
            break

        # SPACE — confirm word
        elif key == 32:
            word = buffer.confirm_word()
            if word:
                if buffer.words:
                    sentence = build_sentence(buffer.words)
                else:
                    sentence = word
                say_sentence(sentence)
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
            say_sentence(sentence)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        elif key == ord('2') and len(suggestions) > 1:
            buffer.buffer = list(suggestions[1].upper())
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            say_sentence(sentence)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        elif key == ord('3') and len(suggestions) > 2:
            buffer.buffer = list(suggestions[2].upper())
            word = buffer.confirm_word()
            sentence = build_sentence(buffer.words) if buffer.words else word
            say_sentence(sentence)
            projector.show_word(word)
            projector.flash_leds()
            suggestions = []
            last_word_str = ""

        # J key — insert J
        elif key in (ord('j'), ord('J')):
            buffer.buffer.append("J")
            buffer.last_added = "J"
            say_letter("J")

        # Z key — insert Z
        elif key in (ord('z'), ord('Z')):
            buffer.buffer.append("Z")
            buffer.last_added = "Z"
            say_letter("Z")

        # S — speak entire sentence
        elif key in (ord('s'), ord('S')):
            if sentence:
                say_sentence(sentence)

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

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    projector.disconnect()


if __name__ == "__main__":
    main()
