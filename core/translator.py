import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_client = None
USE_GEMINI = False

if GEMINI_API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        print("[Translator] Gemini API configured")
    except Exception as e:
        print(f"[Translator] Gemini not available: {e}")
else:
    print("[Translator] No GEMINI_API_KEY, using offline translation only")

MODEL = "gemini-2.0-flash"

LANGUAGES = {
    "EN": {"name": "English",  "elevenlabs_voice": "Sarah",  "flag": "EN"},
    "ES": {"name": "Spanish",  "elevenlabs_voice": "Sarah",  "flag": "ES"},
    "FR": {"name": "French",   "elevenlabs_voice": "Sarah",  "flag": "FR"},
    "IT": {"name": "Italian",  "elevenlabs_voice": "Sarah",  "flag": "IT"},
    "EL": {"name": "Greek",    "elevenlabs_voice": "Sarah",  "flag": "GR"},
}

LANGUAGE_ORDER = ["EN", "ES", "FR", "IT", "EL"]

current_language = "EN"

# Offline translation dictionary — common words and phrases
_OFFLINE = {
    "ES": {
        "HELLO": "HOLA",
        "HELLO MY NAME IS": "HOLA MI NOMBRE ES",
        "MY NAME IS": "MI NOMBRE ES",
        "HI": "HOLA",
        "HELP": "AYUDA",
        "PLEASE": "POR FAVOR",
        "THANK YOU": "GRACIAS",
        "THANKS": "GRACIAS",
        "YES": "SI",
        "NO": "NO",
        "SORRY": "LO SIENTO",
        "WATER": "AGUA",
        "FOOD": "COMIDA",
        "MORE": "MAS",
        "I LOVE YOU": "TE AMO",
        "LOVE": "AMOR",
        "GOOD": "BUENO",
        "BAD": "MALO",
        "HOW ARE YOU": "COMO ESTAS",
        "MY NAME": "MI NOMBRE",
        "WHAT": "QUE",
        "WHERE": "DONDE",
        "WHEN": "CUANDO",
        "WHY": "POR QUE",
        "WHO": "QUIEN",
        "HAPPY": "FELIZ",
        "SAD": "TRISTE",
        "HUNGRY": "HAMBRIENTO",
        "TIRED": "CANSADO",
        "COLD": "FRIO",
        "HOT": "CALIENTE",
        "HOME": "CASA",
        "FAMILY": "FAMILIA",
        "FRIEND": "AMIGO",
        "SCHOOL": "ESCUELA",
        "WORK": "TRABAJO",
        "STOP": "PARA",
        "GO": "IR",
        "COME": "VEN",
        "WANT": "QUIERO",
        "NEED": "NECESITO",
        "LIKE": "ME GUSTA",
        "GOODBYE": "ADIOS",
        "BYE": "ADIOS",
        "GOOD MORNING": "BUENOS DIAS",
        "GOOD NIGHT": "BUENAS NOCHES",
        "NICE": "BONITO",
        "BEAUTIFUL": "HERMOSO",
        "EAT": "COMER",
        "DRINK": "BEBER",
        "SLEEP": "DORMIR",
        "NAME": "NOMBRE",
    },
    "FR": {
        "HELLO": "BONJOUR",
        "HELLO MY NAME IS": "BONJOUR JE M'APPELLE",
        "MY NAME IS": "JE M'APPELLE",
        "HI": "SALUT",
        "HELP": "AIDE",
        "PLEASE": "S'IL VOUS PLAIT",
        "THANK YOU": "MERCI",
        "THANKS": "MERCI",
        "YES": "OUI",
        "NO": "NON",
        "SORRY": "DESOLE",
        "WATER": "EAU",
        "FOOD": "NOURRITURE",
        "MORE": "PLUS",
        "I LOVE YOU": "JE T'AIME",
        "LOVE": "AMOUR",
        "GOOD": "BON",
        "BAD": "MAUVAIS",
        "HOW ARE YOU": "COMMENT ALLEZ-VOUS",
        "MY NAME": "MON NOM",
        "WHAT": "QUOI",
        "WHERE": "OU",
        "WHEN": "QUAND",
        "WHY": "POURQUOI",
        "WHO": "QUI",
        "HAPPY": "HEUREUX",
        "SAD": "TRISTE",
        "HUNGRY": "AFFAME",
        "TIRED": "FATIGUE",
        "COLD": "FROID",
        "HOT": "CHAUD",
        "HOME": "MAISON",
        "FAMILY": "FAMILLE",
        "FRIEND": "AMI",
        "SCHOOL": "ECOLE",
        "WORK": "TRAVAIL",
        "STOP": "ARRETEZ",
        "GO": "ALLEZ",
        "COME": "VENEZ",
        "WANT": "JE VEUX",
        "NEED": "J'AI BESOIN",
        "LIKE": "J'AIME",
        "GOODBYE": "AU REVOIR",
        "BYE": "AU REVOIR",
        "GOOD MORNING": "BONJOUR",
        "GOOD NIGHT": "BONNE NUIT",
        "NICE": "BEAU",
        "BEAUTIFUL": "MAGNIFIQUE",
        "EAT": "MANGER",
        "DRINK": "BOIRE",
        "SLEEP": "DORMIR",
        "NAME": "NOM",
    },
    "IT": {
        "HELLO": "CIAO",
        "HELLO MY NAME IS": "CIAO MI CHIAMO",
        "MY NAME IS": "MI CHIAMO",
        "HI": "CIAO",
        "HELP": "AIUTO",
        "PLEASE": "PER FAVORE",
        "THANK YOU": "GRAZIE",
        "THANKS": "GRAZIE",
        "YES": "SI",
        "NO": "NO",
        "SORRY": "MI DISPIACE",
        "WATER": "ACQUA",
        "FOOD": "CIBO",
        "MORE": "PIU",
        "I LOVE YOU": "TI AMO",
        "LOVE": "AMORE",
        "GOOD": "BUONO",
        "BAD": "CATTIVO",
        "HOW ARE YOU": "COME STAI",
        "MY NAME": "IL MIO NOME",
        "WHAT": "COSA",
        "WHERE": "DOVE",
        "WHEN": "QUANDO",
        "WHY": "PERCHE",
        "WHO": "CHI",
        "HAPPY": "FELICE",
        "SAD": "TRISTE",
        "HUNGRY": "AFFAMATO",
        "TIRED": "STANCO",
        "COLD": "FREDDO",
        "HOT": "CALDO",
        "HOME": "CASA",
        "FAMILY": "FAMIGLIA",
        "FRIEND": "AMICO",
        "SCHOOL": "SCUOLA",
        "WORK": "LAVORO",
        "STOP": "FERMA",
        "GO": "VAI",
        "COME": "VIENI",
        "WANT": "VOGLIO",
        "NEED": "HO BISOGNO",
        "LIKE": "MI PIACE",
        "GOODBYE": "ARRIVEDERCI",
        "BYE": "CIAO",
        "GOOD MORNING": "BUONGIORNO",
        "GOOD NIGHT": "BUONANOTTE",
        "NICE": "BELLO",
        "BEAUTIFUL": "BELLISSIMO",
        "EAT": "MANGIARE",
        "DRINK": "BERE",
        "SLEEP": "DORMIRE",
        "NAME": "NOME",
    },
    "EL": {
        "HELLO": "\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5",
        "HELLO MY NAME IS": "\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5, \u03bc\u03b5 \u03bb\u03ad\u03bd\u03b5",
        "MY NAME IS": "\u039c\u03b5 \u03bb\u03ad\u03bd\u03b5",
        "HI": "\u0393\u03b5\u03b9\u03b1",
        "HELP": "\u0392\u03bf\u03ae\u03b8\u03b5\u03b9\u03b1",
        "PLEASE": "\u03a0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce",
        "THANK YOU": "\u0395\u03c5\u03c7\u03b1\u03c1\u03b9\u03c3\u03c4\u03ce",
        "THANKS": "\u0395\u03c5\u03c7\u03b1\u03c1\u03b9\u03c3\u03c4\u03ce",
        "YES": "\u039d\u03b1\u03b9",
        "NO": "\u038c\u03c7\u03b9",
        "SORRY": "\u03a3\u03c5\u03b3\u03b3\u03bd\u03ce\u03bc\u03b7",
        "WATER": "\u039d\u03b5\u03c1\u03cc",
        "FOOD": "\u03a6\u03b1\u03b3\u03b7\u03c4\u03cc",
        "MORE": "\u03a0\u03b5\u03c1\u03b9\u03c3\u03c3\u03cc\u03c4\u03b5\u03c1\u03bf",
        "I LOVE YOU": "\u03a3\u2019\u03b1\u03b3\u03b1\u03c0\u03ce",
        "LOVE": "\u0391\u03b3\u03ac\u03c0\u03b7",
        "GOOD": "\u039a\u03b1\u03bb\u03cc",
        "BAD": "\u039a\u03b1\u03ba\u03cc",
        "HOW ARE YOU": "\u03a4\u03b9 \u03ba\u03ac\u03bd\u03b5\u03b9\u03c2",
        "MY NAME": "\u03a4\u03bf \u03cc\u03bd\u03bf\u03bc\u03ac \u03bc\u03bf\u03c5",
        "WHAT": "\u03a4\u03b9",
        "WHERE": "\u03a0\u03bf\u03cd",
        "WHEN": "\u03a0\u03cc\u03c4\u03b5",
        "WHY": "\u0393\u03b9\u03b1\u03c4\u03af",
        "WHO": "\u03a0\u03bf\u03b9\u03bf\u03c2",
        "HAPPY": "\u0395\u03c5\u03c4\u03c5\u03c7\u03b9\u03c3\u03bc\u03ad\u03bd\u03bf\u03c2",
        "SAD": "\u039b\u03c5\u03c0\u03b7\u03bc\u03ad\u03bd\u03bf\u03c2",
        "HUNGRY": "\u03a0\u03b5\u03b9\u03bd\u03ac\u03c9",
        "TIRED": "\u039a\u03bf\u03c5\u03c1\u03b1\u03c3\u03bc\u03ad\u03bd\u03bf\u03c2",
        "COLD": "\u039a\u03c1\u03cd\u03bf",
        "HOT": "\u0396\u03ad\u03c3\u03c4\u03b7",
        "HOME": "\u03a3\u03c0\u03af\u03c4\u03b9",
        "FAMILY": "\u039f\u03b9\u03ba\u03bf\u03b3\u03ad\u03bd\u03b5\u03b9\u03b1",
        "FRIEND": "\u03a6\u03af\u03bb\u03bf\u03c2",
        "SCHOOL": "\u03a3\u03c7\u03bf\u03bb\u03b5\u03af\u03bf",
        "WORK": "\u0394\u03bf\u03c5\u03bb\u03b5\u03b9\u03ac",
        "STOP": "\u03a3\u03c4\u03b1\u03bc\u03ac\u03c4\u03b1",
        "GO": "\u03a0\u03ac\u03bc\u03b5",
        "COME": "\u0388\u03bb\u03b1",
        "WANT": "\u0398\u03ad\u03bb\u03c9",
        "NEED": "\u03a7\u03c1\u03b5\u03b9\u03ac\u03b6\u03bf\u03bc\u03b1\u03b9",
        "LIKE": "\u039c\u03bf\u03c5 \u03b1\u03c1\u03ad\u03c3\u03b5\u03b9",
        "GOODBYE": "\u0391\u03bd\u03c4\u03af\u03bf",
        "BYE": "\u0393\u03b5\u03b9\u03b1",
        "GOOD MORNING": "\u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1",
        "GOOD NIGHT": "\u039a\u03b1\u03bb\u03b7\u03bd\u03cd\u03c7\u03c4\u03b1",
        "NICE": "\u03a9\u03c1\u03b1\u03af\u03bf",
        "BEAUTIFUL": "\u038c\u03bc\u03bf\u03c1\u03c6\u03bf",
        "EAT": "\u03a4\u03c1\u03ce\u03c9",
        "DRINK": "\u03a0\u03af\u03bd\u03c9",
        "SLEEP": "\u039a\u03bf\u03b9\u03bc\u03ac\u03bc\u03b1\u03b9",
        "NAME": "\u038c\u03bd\u03bf\u03bc\u03b1",
    },
}


def _offline_translate(text, target_lang_code):
    """Translate using built-in dictionary. Matches longest phrases first."""
    lang_dict = _OFFLINE.get(target_lang_code, {})
    upper = text.upper().strip()

    # Try exact match first
    if upper in lang_dict:
        return lang_dict[upper]

    # Try longest prefix phrases first (e.g. "HELLO MY NAME IS BILL")
    # Sort phrases by length descending so longer matches win
    phrases = sorted(lang_dict.keys(), key=len, reverse=True)
    result = upper
    for phrase in phrases:
        if phrase in result and len(phrase) > 1:
            result = result.replace(phrase, lang_dict[phrase], 1)

    if result != upper:
        return result

    # Fallback: word-by-word
    words = upper.split()
    translated = []
    for word in words:
        if word in lang_dict:
            translated.append(lang_dict[word])
        else:
            translated.append(word)
    result = " ".join(translated)

    if result != upper:
        return result
    return text


def get_current_language():
    return current_language


def cycle_language():
    global current_language
    idx = LANGUAGE_ORDER.index(current_language)
    idx = (idx + 1) % len(LANGUAGE_ORDER)
    current_language = LANGUAGE_ORDER[idx]
    name = LANGUAGES[current_language]["name"]
    print(f"[Translator] Language set to {name}")
    return current_language


def translate(text, target_lang_code):
    if target_lang_code == "EN":
        return text

    # Try Gemini first
    if USE_GEMINI and _client is not None:
        try:
            language_name = LANGUAGES[target_lang_code]["name"]
            prompt = (
                f"Translate this text to {language_name}. "
                f"Return ONLY the translated text, nothing else, "
                f"no explanation, no quotes.\n"
                f"Text: {text}"
            )
            response = _client.models.generate_content(model=MODEL, contents=prompt)
            translated = response.text.strip()[:200]  # cap length for UI safety
            print(f"[Translator] Gemini: {text} -> {translated} ({language_name})")
            return translated
        except Exception as e:
            print(f"[Translator] Gemini failed, using offline: {e}")

    # Offline fallback
    translated = _offline_translate(text, target_lang_code)
    print(f"[Translator] Offline: {text} -> {translated} ({LANGUAGES[target_lang_code]['name']})")
    return translated


def get_voice_for_language(lang_code):
    lang = LANGUAGES.get(lang_code)
    if lang:
        return lang["elevenlabs_voice"]
    return "Sarah"


def get_language_display():
    lang = LANGUAGES.get(current_language, LANGUAGES["EN"])
    return f"{current_language} — {lang['name']}"
