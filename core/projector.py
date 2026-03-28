import time

_device = None


def connect():
    """Try to connect to FREE-WILi hardware. Returns True if connected."""
    global _device
    try:
        from freewili import FreeWili
        _device = FreeWili.find_first().expect("No FREE-WILi found")
        _device.open()
        print("[Projector] FREE-WILi connected")
        return True
    except Exception as e:
        _device = None
        print(f"[Projector] FREE-WILi not available: {e}")
        return False


def is_connected():
    return _device is not None


def show_text(text, clear_first=True):
    """Display text on the FREE-WILi screen."""
    global _device
    if _device is None:
        return
    try:
        text = text[:64]
        if clear_first:
            _device.clear_display()
        _device.display_text(text)
    except Exception as e:
        print(f"[Projector] Display error: {e}")
        _device = None


def show_letter(letter, confidence):
    show_text(f"{letter}  {int(confidence * 100)}%")


def show_word(word):
    show_text(f"WORD: {word}")


def show_sentence(sentence):
    show_text(sentence[:64])


def flash_leds(color_tuple=(0, 255, 0), duration=0.3):
    """Briefly light the board LEDs on word confirmation."""
    global _device
    if _device is None:
        return
    try:
        for i in range(7):
            _device.set_board_leds(i, color_tuple[0], color_tuple[1], color_tuple[2])
        time.sleep(duration)
        for i in range(7):
            _device.set_board_leds(i, 0, 0, 0)
    except Exception as e:
        print(f"[Projector] LED error: {e}")
        _device = None


def disconnect():
    """Close the FREE-WILi connection."""
    global _device
    if _device is not None:
        try:
            _device.close()
        except Exception:
            pass
        _device = None
        print("[Projector] FREE-WILi disconnected")
