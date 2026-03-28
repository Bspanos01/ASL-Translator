import subprocess


def say(text):
    """Speak text using macOS 'say' command. Non-blocking."""
    try:
        subprocess.Popen(["say", "-r", "200", text])
    except Exception:
        pass


def say_letter(letter):
    """Speak a single letter A-Z."""
    if letter and len(letter) == 1 and letter.isalpha():
        say(letter.lower())


def say_word(word):
    """Speak a full word."""
    if word:
        say(word)


def say_sentence(sentence):
    """Speak a full sentence."""
    if sentence:
        say(sentence)
