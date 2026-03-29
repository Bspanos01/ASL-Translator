class LetterBuffer:
    def __init__(self, debounce_frames=20, min_confidence=0.75, cooldown_frames=6):
        self.buffer = []
        self.words = []
        self.candidate = None
        self.candidate_count = 0
        self.last_added = None
        self.debounce_frames = debounce_frames
        self.min_confidence = min_confidence
        self._no_hand_count = 0
        self._no_hand_reset = 5  # frames without hand to reset last_added
        self.cooldown_frames = cooldown_frames  # require gap after adding a letter
        self._cooldown_remaining = 0  # frames left in cooldown
        self._gap_seen = False  # whether a low-confidence gap was seen during cooldown

    def add_letter(self, letter, confidence):
        """Called every frame with the predicted letter and confidence.
        After a letter is accepted, a brief low-confidence gap (hand transition)
        must occur before the next letter can register. This prevents jibberish
        from hand transitions between signs.
        """
        if confidence < self.min_confidence or letter is None:
            self.candidate = None
            self.candidate_count = 0
            self._no_hand_count += 1
            # Hand gone for a few frames = reset for consecutive letter
            if self._no_hand_count >= self._no_hand_reset:
                self.last_added = None
            # Low confidence counts as seeing a gap during cooldown
            if self._cooldown_remaining > 0:
                self._gap_seen = True
                self._cooldown_remaining -= 1
            return

        self._no_hand_count = 0

        # During cooldown, wait for a gap before accepting new letters
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if not self._gap_seen:
                # No gap yet — ignore this prediction (it's transition noise)
                return

        if letter == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = letter
            self.candidate_count = 1

        if self.candidate_count >= self.debounce_frames and letter != self.last_added:
            self.buffer.append(letter)
            self.last_added = letter
            self.candidate_count = 0
            # Start cooldown — require a gap before next letter
            self._cooldown_remaining = self.cooldown_frames
            self._gap_seen = False

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
            "sentence": " ".join(self.words),
            "candidate": self.candidate,
            "candidate_progress": min(progress, 1.0),
        }
