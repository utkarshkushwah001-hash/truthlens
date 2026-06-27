from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# upload folder
UPLOAD_FOLDER = "uploads"

# create folder automatically if missing
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/analyze", methods=["POST"])
def analyze():

    # check if image exists
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    image = request.files["image"]

    # save image
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    # get file size in KB
    file_size = os.path.getsize(filepath) / 1024

    # scoring logic
    score = 50

    if file_size > 2000:      # bigger than 2MB
        score += 20

    elif file_size < 200:     # smaller than 200KB
        score -= 20

    # result logic
    if score >= 70:
        verdict = "Likely Real Image"

    elif score >= 40:
        verdict = "Uncertain Result"

    else:
        verdict = "Likely AI Generated"

    return jsonify({
        "filename": image.filename,
        "file_size_kb": round(file_size, 2),
        "score": score,
        "verdict": verdict
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)