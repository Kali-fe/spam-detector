"""
app.py — Application Flask pour la détection de spam.
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Chargement du modèle ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "spam_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Modèle introuvable. Lancez d'abord : python train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message vide."}), 400

    # Prédiction + probabilités
    label = model.predict([message])[0]
    proba = model.predict_proba([message])[0]

    ham_pct  = round(float(proba[0]) * 100, 1)
    spam_pct = round(float(proba[1]) * 100, 1)

    return jsonify({
        "is_spam":   bool(label == 1),
        "label":     "SPAM" if label == 1 else "Message normal",
        "ham_pct":   ham_pct,
        "spam_pct":  spam_pct,
        "confidence": spam_pct if label == 1 else ham_pct,
    })


# ── Démarrage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
