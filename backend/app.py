
import sys
import uuid
import cv2
import json
import csv

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
NORMAL_FOLDER = BASE_DIR / "datasets" / "anomaly" / "normal"

MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "runs" /
    "pcb" / "prometheus_v1" / "weights" / "best.pt"
)

ANALYSIS_FOLDER = (
    BASE_DIR / "runs" / "detect" / "prometheus_final_analysis"
)

CALIBRATION_FOLDER = (
    BASE_DIR / "runs" / "detect" / "yolo_calibration_results"
)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models"))

from ultralytics import YOLO
from patchcore_detector import PCBPatchCore


# ==========================================
# SEVERITY ASSESSMENT
# ==========================================

def assess_region_severity(region):

    maximum_score = float(region.get("maximum_score", 0))

    # Provisional heuristic, not a defect probability.

    severity_score = min(
        100,
        max(0, round(maximum_score * 4))
    )

    if severity_score >= 85:
        severity = "Critical"
    elif severity_score >= 65:
        severity = "High"
    elif severity_score >= 35:
        severity = "Moderate"
    else:
        severity = "Low"

    region["severity"] = severity
    region["severity_score"] = severity_score

    return region


# ==========================================
# FLASK
# ==========================================

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


# ==========================================
# LOAD YOLO
# ==========================================

model = None

if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print("PROMETHEUS-X YOLO model loaded!")
else:
    print("WARNING: YOLO model unavailable.")


# ==========================================
# LOAD PATCHCORE
# ==========================================

patchcore = None

if NORMAL_FOLDER.exists():

    try:
        patchcore = PCBPatchCore(memory_limit=5000)
        patchcore.fit(NORMAL_FOLDER)

        print("PROMETHEUS-X PatchCore ready!")

    except Exception as error:
        patchcore = None
        print("PatchCore initialization failed:", error)

else:
    print("WARNING: PatchCore normal folder unavailable.")


# ==========================================
# FRONTEND
# ==========================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==========================================
# RESULT IMAGES
# ==========================================

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "yolo_available": model is not None,
        "patchcore_available": patchcore is not None
    })


# ==========================================
# MODEL EVALUATION DATA HELPERS
# ==========================================

def read_csv_file(path):

    if not path.exists():
        return []

    try:
        with open(
            path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            return list(csv.DictReader(file))

    except Exception as error:
        print("CSV READ ERROR:", path, error)
        return []


def read_json_file(path):

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as error:
        print("JSON READ ERROR:", path, error)
        return None


# ==========================================
# MODEL EVALUATION API
# ==========================================

@app.route("/api/analysis")
def get_analysis():

    final_report = read_json_file(
        ANALYSIS_FOLDER / "final_analysis_report.json"
    )

    calibration_metrics = read_csv_file(
        CALIBRATION_FOLDER / "calibration_metrics.csv"
    )

    false_detection_by_class = read_csv_file(
        ANALYSIS_FOLDER / "false_detection_by_class.csv"
    )

    incorrect_detections = read_csv_file(
        ANALYSIS_FOLDER / "incorrect_detections.csv"
    )

    return jsonify({

        "available": True,

        "analysis_ready": final_report is not None,

        "calibration_metrics": calibration_metrics,

        "shift_analysis": (
            final_report.get("shift_analysis")
            if final_report else None
        ),

        "false_detection_analysis": (
            final_report.get("false_detection_analysis")
            if final_report else None
        ),

        "false_detection_by_class": false_detection_by_class,

        "incorrect_detections": incorrect_detections[:100]

    })


# ==========================================
# PCB INSPECTION
# ==========================================

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

    unique_id = uuid.uuid4().hex[:8]

    image_path = UPLOAD_FOLDER / f"{unique_id}_{filename}"

    file.save(str(image_path))

    image = cv2.imread(str(image_path))

    if image is None:
        image_path.unlink(missing_ok=True)
        return jsonify({"error": "Invalid image"}), 400

    height, width, channels = image.shape

    # ======================================
    # YOLO INSPECTION
    # ======================================

    detections = []
    annotated_url = None

    if model is not None:

        results = model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False
        )

        result = results[0]

        print("YOLO DETECTIONS:", len(result.boxes))

        annotated_filename = f"annotated_{unique_id}.jpg"

        cv2.imwrite(
            str(RESULTS_FOLDER / annotated_filename),
            result.plot()
        )

        annotated_url = f"/results/{annotated_filename}"

        for box in result.boxes:

            detections.append({

                "class": result.names[int(box.cls[0])],

                "confidence": round(
                    float(box.conf[0]), 3
                ),

                "box": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

            })

    # ======================================
    # PATCHCORE ANOMALY INSPECTION
    # ======================================

    anomaly = {
        "available": False,
        "message": "PatchCore is unavailable."
    }

    if patchcore is not None:

        try:

            patch_result = patchcore.inspect(image_path)

            heatmap_filename = f"heatmap_{unique_id}.jpg"
            overlay_filename = f"overlay_{unique_id}.jpg"

            heatmap_bgr = cv2.cvtColor(
                patch_result["heatmap"],
                cv2.COLOR_RGB2BGR
            )

            overlay_bgr = cv2.cvtColor(
                patch_result["overlay"],
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                str(RESULTS_FOLDER / heatmap_filename),
                heatmap_bgr
            )

            cv2.imwrite(
                str(RESULTS_FOLDER / overlay_filename),
                overlay_bgr
            )

            # ==================================
            # LOCALIZATION + SEVERITY
            # ==================================

            regions = patch_result.get("regions", [])

            assessed_regions = []

            for region in regions:

                region = dict(region)

                assessed_region = assess_region_severity(region)

                assessed_regions.append(assessed_region)

            anomaly = {

                "available": True,

                "status": "experimental",

                "maximum_score": round(
                    float(patch_result["anomaly_score"]), 4
                ),

                "mean_score": round(
                    float(patch_result["mean_anomaly_score"]), 4
                ),

                "regions": assessed_regions,

                "region_count": len(assessed_regions),

                "localization_threshold": patch_result.get(
                    "localization_threshold", 0.65
                ),

                "severity_method": "Provisional heuristic",

                "severity_note": (
                    "Experimental severity scores based on raw "
                    "anomaly strength. Not validated defect severity."
                ),

                "heatmap": f"/results/{heatmap_filename}",

                "overlay": f"/results/{overlay_filename}",

                "note": (
                    "Experimental anomaly scores. "
                    "Not calibrated defect probabilities."
                )

            }

            print(
                "PATCHCORE LOCALIZED REGIONS:",
                len(assessed_regions)
            )

        except Exception as error:

            print("PATCHCORE INSPECTION ERROR:", error)

            anomaly = {
                "available": False,
                "message": str(error)
            }

    # ======================================
    # COMBINED REPORT
    # ======================================

    inspection = {

        "status": "completed",

        "detection_count": len(detections),

        "detections": detections,

        "yolo_available": model is not None,

        "anomaly_detection": anomaly

    }

    return jsonify({

        "message": "PCB inspection completed.",

        "filename": filename,

        "width": width,

        "height": height,

        "channels": channels,

        "inspection": inspection,

        "annotated_image": annotated_url

    })


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
