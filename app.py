from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None

    if request.method == "POST":
        text = request.form["text"]
        predictions_list = emotion_classifier(text)  # this is always a list of lists
        if isinstance(predictions_list, list) and len(predictions_list) > 0:
            predictions = predictions_list[0]  # first inner list
            if isinstance(predictions, list):
                # now find the label with the highest score
                emotion = max(predictions, key=lambda x: x.get('score', 0)).get('label', None)
            else:
                # fallback if predictions is a single dict
                emotion = predictions.get('label', None)
        else:
            emotion = "Unknown"

    return render_template("index.html", emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
