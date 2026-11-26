import sys
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
import os


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_FOLDER = BASE_DIR / "uploads"
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "pcb" / "prometheus_v1" / "weights" / "best.pt"

UPLOAD_FOLDER.mkdir(exist_ok=True)

# Flask configuration
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
model = None

if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print("PROMETHEUS-X AI model loaded!")
else:
    print("AI model is still training or not available.")


# FRONTEND
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")
# DISPLAY ANNOTATED RESULTS
@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(BASE_DIR / "results", filename)


# IMAGE UPLOAD AND PROCESSING
@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)

    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    image_path = UPLOAD_FOLDER / filename

    file.save(str(image_path))

    image = cv2.imread(str(image_path))

    if image is None:
        os.remove(image_path)
        return jsonify({"error": "Invalid image"}), 400

    height, width, channels = image.shape

    if model is not None:
        results = model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False
        )

        result = results[0]
        print("YOLO RAW DETECTION COUNT:", len(result.boxes))

        # Generate annotated PCB image
        annotated_image = result.plot()

        output_filename = "annotated_" + filename
        output_path = BASE_DIR / "results" / output_filename

        cv2.imwrite(str(output_path), annotated_image)

        detections = []

        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "box": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

        inspection = {
            "status": "completed",
            "detections": detections,
            "count": len(detections)
        }
    else:
        inspection = {
            "status": "pending",
            "message": "Trained YOLO model is not available yet."
        }

    return jsonify({
    "message": "Image uploaded and processed successfully!",
    "filename": filename,
    "width": width,
    "height": height,
    "channels": channels,
    "inspection": inspection,
    "annotated_image": "/results/" + output_filename if model is not None else None
})

if __name__ == "__main__":
    app.run(debug=True)