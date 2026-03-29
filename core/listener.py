import threading
import tempfile
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

_recognizer = sr.Recognizer()

is_listening = False
last_transcript = ""
_thread = None
_audio_chunks = []
_stream = None

SAMPLE_RATE = 16000


def _audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk while stream is open."""
    if is_listening:
        _audio_chunks.append(indata.copy())


def start_recording():
    """Begin capturing audio. Call every frame while F is held."""
    global is_listening, _stream, _audio_chunks
    if not is_listening:
        _audio_chunks = []
        is_listening = True
        try:
            _stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=_audio_callback,
            )
            _stream.start()
        except Exception as e:
            print(f"[Listener] Mic error: {e}")
            is_listening = False


def stop_recording():
    """Stop capturing and transcribe in a background thread."""
    global is_listening, _stream, _thread
    if not is_listening:
        return
    is_listening = False

    # Stop the audio stream
    if _stream is not None:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
        _stream = None

    # Only transcribe if we have at least 0.5 seconds of audio
    chunks = list(_audio_chunks)
    total_samples = sum(c.shape[0] for c in chunks) if chunks else 0
    if total_samples < SAMPLE_RATE * 0.5:
        print("[Listener] Recording too short, skipping transcription")
        return
    _thread = threading.Thread(target=_transcribe, args=(chunks,), daemon=True)
    _thread.start()


def _transcribe(chunks):
    """Transcribe recorded audio chunks."""
    global last_transcript
    if not chunks:
        return
    tmp_path = None
    try:
        audio_data = np.concatenate(chunks, axis=0)

        # Save to temp wav
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_path, audio_data, SAMPLE_RATE)

        # Transcribe
        with sr.AudioFile(tmp_path) as source:
            audio = _recognizer.record(source)
        text = _recognizer.recognize_google(audio, language="en-US")
        last_transcript = text.upper()[:200]  # cap length for UI safety
    except sr.UnknownValueError:
        last_transcript = ""
    except sr.RequestError:
        last_transcript = "[NO CONNECTION]"
    except Exception:
        last_transcript = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def get_transcript():
    """Return the last transcript."""
    return last_transcript


def clear_transcript():
    """Clear the stored transcript."""
    global last_transcript
    last_transcript = ""


def is_active():
    """Return True if currently recording."""
    return is_listening
