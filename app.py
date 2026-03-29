from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_explanation(text, label):
    prompt = f"Headline: {text}\nThis news is {label} because"
    
    result = generator(prompt, max_length=120)
    output = result[0]["generated_text"]
    
    return output.replace(prompt, "").strip()

app = Flask(__name__)
CORS(app)

# Load model
classifier = pipeline(
    "text-classification",
    model="./fake_news_model"
)

label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}

# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    text = text[:512]
    
    result = classifier(text)[0]
    
    label = label_map.get(result["label"], result["label"])
    confidence = result["score"]
    
    # 🔥 ADD THIS
    explanation = generate_explanation(text, label)
    
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 3),
        "explanation": explanation   # 👈 NEW
    })

if __name__ == "__main__":
    app.run(debug=True)