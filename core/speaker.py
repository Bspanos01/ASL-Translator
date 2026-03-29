import os
import subprocess
import threading
import tempfile

from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

try:
    from elevenlabs import ElevenLabs
    from elevenlabs.types import VoiceSettings
    if ELEVENLABS_API_KEY:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        USE_ELEVENLABS = True
        print("[Speaker] ElevenLabs connected")
    else:
        client = None
        USE_ELEVENLABS = False
        print("[Speaker] No ELEVENLABS_API_KEY, using macOS say fallback")
except Exception as e:
    client = None
    USE_ELEVENLABS = False
    VoiceSettings = None
    print(f"[Speaker] ElevenLabs not available: {e}")

# Default voice — Sarah (Mature, Reassuring, Confident)
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
VOICE_SETTINGS = {
    "stability": 0.75,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}

# All languages use the default voice (Sarah) — ElevenLabs turbo v2.5
# handles multilingual text with any voice, and free-tier accounts
# can only use their own voices, not library voices.
def _resolve_voice_id(voice_name=None):
    """Always use the default voice. Free-tier ElevenLabs can't access library voices."""
    return VOICE_ID


def _kill_all_audio():
    """Kill any currently playing audio."""
    try:
        subprocess.run(["killall", "say"], capture_output=True)
    except Exception:
        pass
    try:
        subprocess.run(["killall", "afplay"], capture_output=True)
    except Exception:
        pass


def _speak_elevenlabs(text, voice_name=None, lang="EN"):
    """Speak text using ElevenLabs API."""
    try:
        vid = _resolve_voice_id(voice_name)
        # Use multilingual model for non-English, turbo for English
        model = "eleven_multilingual_v2" if lang != "EN" else "eleven_turbo_v2_5"
        audio_iter = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=model,
            output_format="pcm_22050",
            voice_settings=VoiceSettings(**VOICE_SETTINGS),
        )
        # Collect raw PCM bytes
        pcm_data = b"".join(audio_iter)

        # Write as WAV so afplay can handle it
        import wave
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(22050)
            wf.writeframes(pcm_data)
        tmp.close()

        subprocess.run(["afplay", tmp.name])
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        print(f"[Speaker] ElevenLabs OK ({model}): \"{text}\"")
    except Exception as e:
        print(f"[Speaker] ElevenLabs FAILED: {e} — falling back to macOS say")
        subprocess.Popen(["say", "-r", "200", text])


def _speak_fallback(text):
    """macOS say command fallback."""
    try:
        subprocess.Popen(["say", "-r", "200", text])
    except Exception:
        pass


def say(text):
    """Speak text in a background thread."""
    if not text:
        return
    _kill_all_audio()
    if USE_ELEVENLABS:
        thread = threading.Thread(target=_speak_elevenlabs, args=(text,), daemon=True)
    else:
        thread = threading.Thread(target=_speak_fallback, args=(text,), daemon=True)
    thread.start()


def say_with_mood(sentence, mood="neutral", voice_override=None, lang="EN"):
    """Speak sentence. Kills any in-progress audio first."""
    if not sentence:
        return
    _kill_all_audio()
    print(f"[Speaker] mood={mood}, voice={voice_override or 'default'}, lang={lang}")
    if USE_ELEVENLABS:
        thread = threading.Thread(target=_speak_elevenlabs, args=(sentence, voice_override, lang), daemon=True)
    else:
        thread = threading.Thread(target=_speak_fallback, args=(sentence,), daemon=True)
    thread.start()
