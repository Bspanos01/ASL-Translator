import threading
import time

from core.speaker import say_with_mood


def activate_emergency():
    print("[EMERGENCY] activated")

    def _speak():
        say_with_mood("EMERGENCY. I need help immediately. Please call 911.",
                      mood="urgent", lang="EN")
        time.sleep(4)
        say_with_mood("EMERGENCY. I need help immediately. Please call 911.",
                      mood="urgent", lang="EN")

    threading.Thread(target=_speak, daemon=True).start()
