import os
import sys
from flask import Flask, request, jsonify

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag_pipeline.rag import answer_question

app = Flask(__name__)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    k = data.get("k", 3)  # optional, defaults to 3

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # answer_question returns a dict: {"answer": ..., "sources": [...]}
    result = answer_question(question, k=int(k))

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
