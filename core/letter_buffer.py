class LetterBuffer:
    def __init__(self, debounce_frames=20, min_confidence=0.75):
        self.buffer = []
        self.words = []
        self.candidate = None
        self.candidate_count = 0
        self.last_added = None
        self.debounce_frames = debounce_frames
        self.min_confidence = min_confidence
        self._no_hand_count = 0
        self._no_hand_reset = 5  # frames without hand to reset last_added

    def add_letter(self, letter, confidence):
        """Called every frame with the predicted letter and confidence.
        When hand disappears briefly (pulled away), resets so the same letter
        can be typed again for consecutive letters like LL, SS.
        """
        if confidence < self.min_confidence or letter is None:
            self.candidate = None
            self.candidate_count = 0
            self._no_hand_count += 1
            # Hand gone for a few frames = reset for consecutive letter
            if self._no_hand_count >= self._no_hand_reset:
                self.last_added = None
            return

        self._no_hand_count = 0

        if letter == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = letter
            self.candidate_count = 1

        if self.candidate_count >= self.debounce_frames and letter != self.last_added:
            self.buffer.append(letter)
            self.last_added = letter
            self.candidate_count = 0

    def reset_last_added(self):
        """Manually reset so the same letter can be added again."""
        self.last_added = None

    def get_word_so_far(self):
        return "".join(self.buffer)

    def get_word_count(self):
        return len(self.words)

    def confirm_word(self):
        """Confirm the current buffer as a word. Returns the confirmed word."""
        word = "".join(self.buffer)
        if word:
            self.words.append(word)
        self.buffer = []
        self.last_added = None
        return word

    def backspace(self):
        """Remove last letter from buffer."""
        if self.buffer:
            self.buffer.pop()
            self.last_added = self.buffer[-1] if self.buffer else None

    def get_full_sentence(self):
        return " ".join(self.words)

    def clear_last_word(self):
        """Remove the last confirmed word."""
        if self.words:
            self.words.pop()

    def clear_all(self):
        """Reset everything to initial state."""
        self.buffer = []
        self.words = []
        self.candidate = None
        self.candidate_count = 0
        self.last_added = None
        self._no_hand_count = 0

    def get_status(self):
        progress = self.candidate_count / self.debounce_frames if self.debounce_frames > 0 else 0.0
        return {
            "word_so_far": self.get_word_so_far(),
            "words": list(self.words),
            "sentence": self.get_full_sentence(),
            "candidate": self.candidate,
            "candidate_progress": min(progress, 1.0),
        }
