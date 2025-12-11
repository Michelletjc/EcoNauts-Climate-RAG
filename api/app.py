from flask import Flask, request, jsonify
from rag_pipeline.rag import answer_question

app = Flask(__name__)


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    Request:
        {"question": "your question here"}
    Response:
        {"answer": "...", "sources": [...]}
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    answer, sources = answer_question(question)

    return jsonify({"answer": answer, "sources": sources}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
