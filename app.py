from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Lazy-loaded models
classifier = None
generator = None

def get_classifier():
    global classifier
    if classifier is None:
        from transformers import pipeline
        classifier = pipeline(
            "text-classification",
            model="./fake_news_model"
        )
    return classifier

def get_generator():
    global generator
    if generator is None:
        from transformers import pipeline
        generator = pipeline("text-generation", model="gpt2")
    return generator


def generate_explanation(text, label):
    gen = get_generator()
    
    prompt = f"Headline: {text}\nThis news is {label} because"
    
    result = gen(prompt, max_length=120)
    output = result[0]["generated_text"]
    
    return output.replace(prompt, "").strip()


label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    text = text[:512]
    
    model = get_classifier()
    result = model(text)[0]
    
    label = label_map.get(result["label"], result["label"])
    confidence = result["score"]
    
    explanation = generate_explanation(text, label)
    
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 3),
        "explanation": explanation
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)