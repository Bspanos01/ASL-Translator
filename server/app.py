import secrets
import threading

from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SECRET_TOKEN = secrets.token_urlsafe(6)

_lock = threading.Lock()

state = {
    "asl_sentence": "",
    "transcript": "",
    "mood": "neutral",
    "current_word": "",
    "suggestions": [],
    "language": "EN",
    "is_listening": False,
    "translated": "",
}


def _check_token():
    if request.args.get("token") != SECRET_TOKEN:
        abort(403)


def update_state(**kwargs):
    with _lock:
        for k, v in kwargs.items():
            if k in state:
                state[k] = v


@app.route("/")
def index():
    _check_token()
    return render_template("index.html", token=SECRET_TOKEN)


@app.route("/state")
def get_state():
    _check_token()
    with _lock:
        return jsonify(state)


@app.route("/update", methods=["POST"])
def post_update():
    if request.headers.get("X-Internal") != "signly":
        abort(403)
    data = request.json or {}
    update_state(**data)
    return jsonify({"ok": True})


def run_server(host="0.0.0.0", port=5050):
    try:
        print(f"[Server] Starting on {host}:{port}...")
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[Server] FAILED TO START: {e}")
