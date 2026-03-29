import os

from dotenv import load_dotenv

load_dotenv()

USE_GEMINI = False
 
_client = None
if USE_GEMINI:
    try:
        from google import genai
        _api_key = os.getenv("GEMINI_API_KEY")
        if _api_key:
            _client = genai.Client(api_key=_api_key)
        else:
            print("[Mood] No GEMINI_API_KEY found, using offline mood detection")
    except Exception as e:
        print(f"[Mood] Gemini not available: {e}")

MODEL = "gemini-2.0-flash-lite"

VALID_MOODS = {"happy", "sad", "urgent", "angry", "neutral", "grateful", "confused"}


def detect_mood_offline(sentence):
    """Fast keyword-based mood detection. No API needed."""
    s = sentence.lower()

    # Check most specific first
    if any(w in s for w in ["thank you", "grateful", "appreciate"]):
        return "grateful"
    if any(w in s for w in ["help", "emergency", "please", "now", "urgent"]):
        return "urgent"
    if any(w in s for w in ["angry", "hate", "stop", "frustrated"]):
        return "angry"
    if any(w in s for w in ["thank", "love", "happy", "great", "good", "awesome"]):
        return "happy"
    if any(w in s for w in ["sorry", "sad", "miss", "hurt", "tired"]):
        return "sad"
    if any(w in s for w in ["?", "confused", "understand", "what", "why"]):
        return "confused"
    return "neutral"


def detect_mood_gemini(sentence):
    """Use Gemini to detect sentence mood."""
    if _client is None:
        return "neutral"
    try:
        prompt = (
            "Analyze the emotional tone of this sentence and respond with "
            "exactly one word from this list: "
            "happy, sad, urgent, angry, neutral, grateful, confused.\n"
            f"Sentence: {sentence}\n"
            "Respond with only the single mood word, nothing else."
        )
        response = _client.models.generate_content(model=MODEL, contents=prompt)
        mood = response.text.strip().lower()
        if mood in VALID_MOODS:
            return mood
        return "neutral"
    except Exception:
        return "neutral"


def detect_mood(sentence):
    """Detect mood — tries Gemini if enabled, otherwise offline."""
    if USE_GEMINI and _client is not None:
        try:
            return detect_mood_gemini(sentence)
        except Exception:
            return detect_mood_offline(sentence)
    return detect_mood_offline(sentence)
