import json
import os

from dotenv import load_dotenv

load_dotenv()

# Set to True to use Gemini API, False for offline-only (no API calls)
USE_GEMINI_API = False

_client = None
if USE_GEMINI_API:
    try:
        from google import genai
        _api_key = os.getenv("GEMINI_API_KEY")
        if _api_key:
            _client = genai.Client(api_key=_api_key)
            print("[Autocomplete] Gemini API enabled")
        else:
            print("[Autocomplete] No GEMINI_API_KEY found, using offline mode")
    except Exception as e:
        print(f"[Autocomplete] Gemini client not available: {e}")
else:
    print("[Autocomplete] Running in offline mode")

_cache = {}

MODEL = "gemini-2.0-flash-lite"

# --- Offline word list (common English words) ---
_WORD_LIST = sorted(set([
    "ABLE", "ABOUT", "ABOVE", "ACCEPT", "ACROSS", "ACT", "ADD", "AFRAID",
    "AFTER", "AGAIN", "AGREE", "AIR", "ALL", "ALLOW", "ALMOST", "ALONE",
    "ALONG", "ALREADY", "ALSO", "ALWAYS", "AMONG", "AND", "ANGRY", "ANIMAL",
    "ANOTHER", "ANSWER", "ANY", "APPEAR", "APPLE", "ARE", "AREA", "ARM",
    "AROUND", "ART", "ASK", "AWAY", "BABY", "BACK", "BAD", "BAG", "BALL",
    "BANK", "BASE", "BATH", "BE", "BEAR", "BEAT", "BEAUTIFUL", "BECAUSE",
    "BECOME", "BED", "BEFORE", "BEGIN", "BEHIND", "BELIEVE", "BELOW", "BEST",
    "BETTER", "BETWEEN", "BIG", "BIRD", "BIT", "BLACK", "BLOOD", "BLUE",
    "BOARD", "BOAT", "BODY", "BONE", "BOOK", "BORN", "BOTH", "BOTTOM",
    "BOX", "BOY", "BRAIN", "BREAD", "BREAK", "BRING", "BROTHER", "BROWN",
    "BUILD", "BURN", "BUS", "BUSY", "BUT", "BUY", "CALL", "CAME", "CAN",
    "CAR", "CARD", "CARE", "CARRY", "CASE", "CAT", "CATCH", "CAUSE",
    "CENTER", "CERTAIN", "CHAIR", "CHANGE", "CHECK", "CHILD", "CHILDREN",
    "CHOOSE", "CHURCH", "CITY", "CLASS", "CLEAN", "CLEAR", "CLOSE", "CLOUD",
    "COAT", "COLD", "COLLEGE", "COLOR", "COME", "COMMON", "COMMUNITY",
    "COMPANY", "COMPLETE", "COMPUTER", "COOL", "COULD", "COUNTRY", "COURSE",
    "COVER", "CREATE", "CROSS", "CRY", "CUT", "DAD", "DANCE", "DANGER",
    "DARK", "DAUGHTER", "DAY", "DEAD", "DEAL", "DEAR", "DEATH", "DECIDE",
    "DEEP", "DEVELOP", "DID", "DIE", "DIFFERENT", "DINNER", "DOCTOR", "DOG",
    "DOLLAR", "DONE", "DOOR", "DOWN", "DRAW", "DREAM", "DRESS", "DRINK",
    "DRIVE", "DROP", "DURING", "EACH", "EAR", "EARLY", "EARTH", "EAST",
    "EAT", "EDUCATION", "EIGHT", "EITHER", "ELSE", "END", "ENERGY", "ENJOY",
    "ENOUGH", "ENTER", "EVEN", "EVENING", "EVER", "EVERY", "EVERYONE",
    "EVERYTHING", "EXAMPLE", "EXCEPT", "EXCITED", "EXERCISE", "EXPECT",
    "EXPERIENCE", "EYE", "FACE", "FACT", "FALL", "FAMILY", "FAR", "FAST",
    "FATHER", "FAVORITE", "FEAR", "FEEL", "FEET", "FEW", "FIELD", "FIGHT",
    "FILL", "FINAL", "FIND", "FINE", "FINGER", "FINISH", "FIRE", "FIRST",
    "FISH", "FIVE", "FLOOR", "FLY", "FOLLOW", "FOOD", "FOOT", "FOR",
    "FORCE", "FOREIGN", "FORGET", "FORM", "FORWARD", "FOUND", "FOUR",
    "FREE", "FRIDAY", "FRIEND", "FROM", "FRONT", "FULL", "FUN", "FUTURE",
    "GAME", "GARDEN", "GATE", "GAVE", "GET", "GIRL", "GIVE", "GLAD",
    "GLASS", "GO", "GOD", "GOLD", "GONE", "GOOD", "GOT", "GOVERNMENT",
    "GREAT", "GREEN", "GREW", "GROUND", "GROUP", "GROW", "GUESS", "GUY",
    "HAIR", "HALF", "HALL", "HAND", "HAPPEN", "HAPPY", "HARD", "HAS",
    "HAVE", "HE", "HEAD", "HEALTH", "HEAR", "HEART", "HEAT", "HEAVY",
    "HELD", "HELLO", "HELP", "HER", "HERE", "HIGH", "HILL", "HIM", "HIS",
    "HIT", "HOLD", "HOLE", "HOME", "HOPE", "HORSE", "HOSPITAL", "HOT",
    "HOTEL", "HOUR", "HOUSE", "HOW", "HOWEVER", "HUGE", "HUMAN", "HUNDRED",
    "HUNGRY", "HURT", "ICE", "IDEA", "IMPORTANT", "IN", "INCLUDE", "INDEED",
    "INSIDE", "INSTEAD", "INTEREST", "INTO", "IS", "IT", "JOB", "JOIN",
    "JUMP", "JUST", "KEEP", "KEY", "KID", "KILL", "KIND", "KING", "KITCHEN",
    "KNEW", "KNOW", "LAND", "LANGUAGE", "LARGE", "LAST", "LATE", "LAUGH",
    "LAW", "LAY", "LEAD", "LEARN", "LEAST", "LEAVE", "LEFT", "LEG", "LESS",
    "LET", "LETTER", "LEVEL", "LIFE", "LIFT", "LIGHT", "LIKE", "LINE",
    "LIST", "LISTEN", "LITTLE", "LIVE", "LONG", "LOOK", "LORD", "LOSE",
    "LOST", "LOT", "LOVE", "LOW", "LUCK", "LUNCH", "MACHINE", "MADE",
    "MAIN", "MAJOR", "MAKE", "MAN", "MANY", "MAP", "MARK", "MARKET",
    "MATTER", "MAY", "MAYBE", "ME", "MEAN", "MEET", "MEMBER", "MEMORY",
    "MEN", "MIGHT", "MILE", "MILLION", "MIND", "MINE", "MINUTE", "MISS",
    "MODERN", "MOM", "MOMENT", "MONDAY", "MONEY", "MONTH", "MOON", "MORE",
    "MORNING", "MOST", "MOTHER", "MOUNTAIN", "MOUTH", "MOVE", "MOVIE",
    "MUCH", "MUSIC", "MUST", "MY", "MYSELF", "NAME", "NATION", "NATURE",
    "NEAR", "NECESSARY", "NEED", "NEVER", "NEW", "NEWS", "NEXT", "NICE",
    "NIGHT", "NINE", "NO", "NONE", "NORTH", "NOSE", "NOT", "NOTE",
    "NOTHING", "NOTICE", "NOW", "NUMBER", "OF", "OFF", "OFFER", "OFFICE",
    "OFTEN", "OLD", "ON", "ONCE", "ONE", "ONLY", "OPEN", "OR", "ORDER",
    "OTHER", "OUR", "OUT", "OUTSIDE", "OVER", "OWN", "PAGE", "PAIN",
    "PAIR", "PAPER", "PARENT", "PARK", "PART", "PARTY", "PASS", "PAST",
    "PAY", "PEACE", "PEOPLE", "PERFECT", "PERHAPS", "PERIOD", "PERSON",
    "PICK", "PICTURE", "PIECE", "PLACE", "PLAN", "PLANT", "PLAY", "PLEASE",
    "POINT", "POLICE", "POOR", "POPULAR", "POSSIBLE", "POWER", "PRACTICE",
    "PREPARE", "PRESENT", "PRESIDENT", "PRETTY", "PRICE", "PROBLEM",
    "PRODUCE", "PROGRAM", "PROJECT", "PROMISE", "PROTECT", "PROVIDE",
    "PUBLIC", "PULL", "PURPOSE", "PUSH", "PUT", "QUESTION", "QUICK",
    "QUIET", "QUITE", "RACE", "RAIN", "RAISE", "RAN", "RATHER", "REACH",
    "READ", "READY", "REAL", "REALLY", "REASON", "RECEIVE", "RECORD",
    "RED", "REGION", "REMEMBER", "REPORT", "REST", "RESULT", "RETURN",
    "RICH", "RIDE", "RIGHT", "RING", "RISE", "RIVER", "ROAD", "ROCK",
    "ROLE", "ROOM", "RULE", "RUN", "SAFE", "SAID", "SAME", "SAT", "SAVE",
    "SAW", "SAY", "SCHOOL", "SEA", "SEASON", "SEAT", "SECOND", "SEE",
    "SEEM", "SELL", "SEND", "SENSE", "SENTENCE", "SERVE", "SET", "SEVEN",
    "SEVERAL", "SHALL", "SHE", "SHIP", "SHORT", "SHOT", "SHOULD", "SHOW",
    "SICK", "SIDE", "SIGN", "SIMPLE", "SINCE", "SING", "SISTER", "SIT",
    "SIX", "SIZE", "SKILL", "SLEEP", "SMALL", "SMILE", "SNOW", "SO",
    "SOCIAL", "SOME", "SON", "SONG", "SOON", "SORRY", "SORT", "SOUND",
    "SOUTH", "SPACE", "SPEAK", "SPECIAL", "SPEND", "SPOKE", "SPORT",
    "SPRING", "STAND", "STAR", "START", "STATE", "STAY", "STEP", "STILL",
    "STOP", "STORE", "STORY", "STREET", "STRONG", "STUDENT", "STUDY",
    "SUCH", "SUDDENLY", "SUMMER", "SUN", "SURE", "SURPRISE", "SWEET",
    "TABLE", "TAKE", "TALK", "TEACH", "TEAM", "TELL", "TEN", "TEST",
    "THAN", "THANK", "THAT", "THE", "THEIR", "THEM", "THEN", "THERE",
    "THESE", "THEY", "THING", "THINK", "THIS", "THOSE", "THOUGH", "THOUGHT",
    "THREE", "THROUGH", "THROW", "TIME", "TO", "TODAY", "TOGETHER", "TOLD",
    "TOMORROW", "TONIGHT", "TOO", "TOOK", "TOP", "TOTAL", "TOUCH", "TOWARD",
    "TOWN", "TRADE", "TREE", "TRIP", "TROUBLE", "TRUE", "TRUST", "TRY",
    "TURN", "TWELVE", "TWENTY", "TWO", "TYPE", "UNDER", "UNDERSTAND",
    "UNITED", "UNTIL", "UP", "UPON", "US", "USE", "USUAL", "VALUE", "VERY",
    "VISIT", "VOICE", "WAIT", "WALK", "WALL", "WANT", "WAR", "WARM",
    "WASH", "WATCH", "WATER", "WAY", "WE", "WEAR", "WEATHER", "WEEK",
    "WEIGHT", "WELCOME", "WELL", "WENT", "WERE", "WEST", "WHAT", "WHEN",
    "WHERE", "WHETHER", "WHICH", "WHILE", "WHITE", "WHO", "WHOLE", "WHY",
    "WIDE", "WIFE", "WILL", "WIN", "WIND", "WINDOW", "WINTER", "WISH",
    "WITH", "WITHOUT", "WOMAN", "WOMEN", "WONDER", "WOOD", "WORD", "WORK",
    "WORLD", "WORRY", "WORST", "WORTH", "WOULD", "WRITE", "WRONG", "YARD",
    "YEAH", "YEAR", "YES", "YET", "YOU", "YOUNG", "YOUR",
]))


def _offline_suggestions(partial_word, n=3):
    """Fast prefix match against built-in word list."""
    prefix = partial_word.upper()
    matches = [w for w in _WORD_LIST if w.startswith(prefix)]
    return matches[:n]


def get_suggestions(partial_word, n=3):
    """Get word completions — uses Gemini if enabled, otherwise offline."""
    if not partial_word or len(partial_word) < 2:
        return []

    if _client is None:
        return _offline_suggestions(partial_word, n)

    try:
        prompt = (
            f'You complete partial words for a sign language translator. '
            f'The user has signed: "{partial_word}". '
            f'Return ONLY a JSON array of exactly {n} common English words '
            f'starting with these letters. No explanation. '
            f'Example format: ["HELLO","HELP","HELD"]'
        )
        response = _client.models.generate_content(model=MODEL, contents=prompt)
        text = response.text.strip()
        words = json.loads(text)
        return words[:n]
    except Exception as e:
        print(f"[Autocomplete] API error: {e}")
        return _offline_suggestions(partial_word, n)


def build_sentence(words):
    """Build sentence — uses Gemini if enabled, otherwise simple join."""
    if not words:
        return ""
    if _client is None:
        return " ".join(words)
    try:
        prompt = (
            f"Given these signed words: {words}. "
            f"Return one clean grammatically correct English sentence using these words. "
            f"Respond with only the sentence, nothing else."
        )
        response = _client.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Autocomplete] Sentence error: {e}")
        return " ".join(words)


def get_suggestions_cached(partial_word, n=3):
    """Cached version of get_suggestions."""
    key = (partial_word.upper(), n)
    if key in _cache:
        return _cache[key]
    result = get_suggestions(partial_word, n)
    _cache[key] = result
    return result
