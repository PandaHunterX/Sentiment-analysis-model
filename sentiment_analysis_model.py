from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your sentiment model and vectorizer
with open("random_forest_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Vectorize input text
    text_vec = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vec)[0]

    # Get confidence score (probability of predicted class)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_vec)[0]
        confidence = float(max(proba))  # highest class probability
    else:
        confidence = None  # if model doesn't support probabilities

    return jsonify({
        "sentiment": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
