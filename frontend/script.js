
const imageInput = document.getElementById("imageInput");
const cameraButton = document.getElementById("cameraButton");
const captureButton = document.getElementById("captureButton");
const inspectButton = document.getElementById("inspectButton");

const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");

const statusBox = document.getElementById("status");
const report = document.getElementById("report");

const yoloResults = document.getElementById("yoloResults");
const patchcoreResults = document.getElementById("patchcoreResults");
const analysisDashboard = document.getElementById("analysisDashboard");

let selectedImage = null;
let cameraStream = null;


// ==========================================
// HELPERS
// ==========================================

function addText(parent, tag, text, className = "") {

    const element = document.createElement(tag);
    element.textContent = text;

    if (className) {
        element.className = className;
    }

    parent.appendChild(element);

    return element;
}


function addMetric(parent, label, value) {

    const card = document.createElement("div");
    card.className = "metric";

    addText(card, "span", label);
    addText(card, "strong", value);

    parent.appendChild(card);
}


function addResultImage(parent, title, url) {

    if (!url) return;

    const card = document.createElement("div");
    card.className = "image-card";

    addText(card, "h3", title);

    const image = document.createElement("img");

    image.src = url + "?t=" + Date.now();
    image.alt = title;

    card.appendChild(image);
    parent.appendChild(card);
}


function createTable(parent, headers, rows) {

    const wrapper = document.createElement("div");
    wrapper.className = "table-wrap";

    const table = document.createElement("table");

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach(header => {
        addText(headerRow, "th", header);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    rows.forEach(row => {

        const tr = document.createElement("tr");

        row.forEach(value => {
            addText(tr, "td", value);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    parent.appendChild(wrapper);
}


function formatNumber(value, digits = 4) {

    const number = Number(value);

    return Number.isFinite(number)
        ? number.toFixed(digits)
        : "N/A";
}


// ==========================================
// IMAGE UPLOAD
// ==========================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {

        statusBox.textContent = "Please select a valid image.";
        return;
    }

    selectedImage = file;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    statusBox.textContent = "PCB image selected successfully.";

    report.innerHTML = "";

});


// ==========================================
// OPEN CAMERA
// ==========================================

cameraButton.addEventListener("click", async function () {

    try {

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Camera API unavailable. Use localhost or HTTPS.");
        }

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        camera.srcObject = cameraStream;

        camera.style.display = "block";

        captureButton.disabled = false;

        statusBox.textContent = "Camera is ready.";

    } catch (error) {

        statusBox.textContent = "Camera access failed: " + error.message;

    }

});


// ==========================================
// CAPTURE IMAGE
// ==========================================

captureButton.addEventListener("click", function () {

    if (!camera.videoWidth || !camera.videoHeight) {

        statusBox.textContent = "Camera is not ready yet.";
        return;
    }

    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;

    canvas.getContext("2d").drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(function (blob) {

        if (!blob) {

            statusBox.textContent = "Image capture failed.";
            return;
        }

        selectedImage = new File(
            [blob],
            "captured_pcb.jpg",
            { type: "image/jpeg" }
        );

        preview.src = URL.createObjectURL(blob);
        preview.style.display = "block";

        statusBox.textContent = "Camera image captured successfully.";

    }, "image/jpeg");

});


// ==========================================
// YOLO RESULTS
// ==========================================

function displayYOLO(inspection, annotatedImage) {

    yoloResults.innerHTML = "";

    addResultImage(
        yoloResults,
        "YOLO Annotated PCB",
        annotatedImage
    );

    const detections = inspection.detections || [];

    const metrics = document.createElement("div");
    metrics.className = "metric-grid";

    addMetric(metrics, "Detected Components", detections.length);

    addMetric(
        metrics,
        "YOLO Availability",
        inspection.yolo_available ? "Available" : "Unavailable"
    );

    yoloResults.appendChild(metrics);

    if (detections.length === 0) {

        addText(
            yoloResults,
            "p",
            "No objects detected above the confidence threshold."
        );

        return;
    }

    const rows = detections.map((detection, index) => [

        index + 1,

        detection.class,

        (Number(detection.confidence) * 100).toFixed(1) + "%",

        detection.box.join(", ")

    ]);

    createTable(
        yoloResults,
        ["#", "Component", "Confidence", "Bounding Box"],
        rows
    );

}


// ==========================================
// PATCHCORE RESULTS
// ==========================================

function displayPatchCore(anomaly) {

    patchcoreResults.innerHTML = "";

    if (!anomaly || !anomaly.available) {

        addText(
            patchcoreResults,
            "p",
            "PatchCore unavailable: " +
            (anomaly?.message || "Unknown error")
        );

        return;
    }

    const metrics = document.createElement("div");
    metrics.className = "metric-grid";

    addMetric(
        metrics,
        "Maximum Anomaly Score",
        formatNumber(anomaly.maximum_score)
    );

    addMetric(
        metrics,
        "Mean Anomaly Score",
        formatNumber(anomaly.mean_score)
    );

    addMetric(
        metrics,
        "Localized Regions",
        anomaly.region_count ?? (anomaly.regions || []).length
    );

    patchcoreResults.appendChild(metrics);

    addText(
        patchcoreResults,
        "h3",
        "Localized Anomaly Regions"
    );

    const regions = anomaly.regions || [];

    if (regions.length === 0) {

        addText(
            patchcoreResults,
            "p",
            "No regions crossed the localization threshold."
        );

    } else {

        const rows = regions.map((region, index) => [

            region.region_id ?? index + 1,

            region.severity || "Unknown",

            `${region.severity_score ?? "N/A"}/100`,

            formatNumber(region.maximum_score, 2),

            `${region.width ?? "N/A"} x ${region.height ?? "N/A"}`,

            region.area_patches ?? "N/A"

        ]);

        createTable(
            patchcoreResults,
            [
                "Region",
                "Severity",
                "Score",
                "Max Anomaly",
                "Size (px)",
                "Patches"
            ],
            rows
        );

    }

    const images = document.createElement("div");
    images.className = "image-grid";

    addResultImage(
        images,
        "Anomaly Heatmap",
        anomaly.heatmap
    );

    addResultImage(
        images,
        "PCB Anomaly Overlay",
        anomaly.overlay
    );

    patchcoreResults.appendChild(images);

    addText(
        patchcoreResults,
        "p",
        anomaly.severity_note || anomaly.note ||
        "Experimental anomaly analysis."
    );

}


// ==========================================
// COMBINED INSPECTION REPORT
// ==========================================

function displayResults(data) {

    report.innerHTML = "";

    const inspection = data.inspection || {};

    addText(
        report,
        "h3",
        "PROMETHEUS-X Inspection Report"
    );

    addText(
        report,
        "p",
        `Image: ${data.filename} | Size: ${data.width} x ${data.height}`
    );

    addText(
        report,
        "p",
        "Inspection completed successfully."
    );

    displayYOLO(
        inspection,
        data.annotated_image
    );

    displayPatchCore(
        inspection.anomaly_detection
    );

}


// ==========================================
// SEND IMAGE TO BACKEND
// ==========================================

inspectButton.addEventListener("click", async function () {

    if (!selectedImage) {

        statusBox.textContent =
            "Please upload or capture a PCB image first.";

        return;
    }

    const formData = new FormData();

    formData.append("image", selectedImage);

    statusBox.textContent = "Processing PCB image...";

    inspectButton.disabled = true;

    report.innerHTML = "";

    yoloResults.innerHTML = "Processing YOLO...";
    patchcoreResults.innerHTML = "Processing PatchCore...";

    try {

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(data.error || "Inspection failed.");

        }

        statusBox.textContent = data.message;

        displayResults(data);

    } catch (error) {

        statusBox.textContent = "Error: " + error.message;

        yoloResults.textContent = "Inspection results unavailable.";
        patchcoreResults.textContent = "Inspection results unavailable.";

    } finally {

        inspectButton.disabled = false;

    }

});


// ==========================================
// CALIBRATION RESULTS
// ==========================================

function displayCalibration(data) {

    addText(
        analysisDashboard,
        "h3",
        "YOLO Confidence Calibration"
    );

    const metrics = data.calibration_metrics || [];

    if (metrics.length === 0) {

        addText(
            analysisDashboard,
            "p",
            "Calibration metrics are not available."
        );

        return;
    }

    const rows = metrics.map(item => [

        item.method || item.Method || "Unknown",

        formatNumber(item.brier_score ?? item.Brier_Score),

        formatNumber(item.log_loss ?? item.Log_Loss),

        formatNumber(item.ece ?? item.ECE)

    ]);

    createTable(
        analysisDashboard,
        ["Method", "Brier Score", "Log Loss", "ECE"],
        rows
    );

}


// ==========================================
// DISTRIBUTION SHIFT ANALYSIS
// ==========================================

function displayShiftAnalysis(shift) {

    addText(
        analysisDashboard,
        "h3",
        "Prediction Distribution Analysis"
    );

    if (!shift) {

        addText(
            analysisDashboard,
            "p",
            "Shift analysis report is not available."
        );

        return;
    }

    const metrics = document.createElement("div");
    metrics.className = "metric-grid";

    const fields = [
        ["Calibration Groups", "calibration_groups"],
        ["Testing Groups", "testing_groups"],
        ["Group Overlap", "group_overlap"],
        ["Calibration Mean Confidence", "calibration_mean_confidence"],
        ["Testing Mean Confidence", "testing_mean_confidence"],
        ["KS Statistic", "ks_statistic"],
        ["KS P-Value", "ks_pvalue"]
    ];

    fields.forEach(([label, key]) => {

        if (shift[key] !== undefined) {

            addMetric(
                metrics,
                label,
                formatNumber(shift[key])
            );

        }

    });

    analysisDashboard.appendChild(metrics);

    if (shift.detection_correctness) {

        addText(
            analysisDashboard,
            "p",
            "Detection correctness information is available in the report."
        );

    }

}


// ==========================================
// FALSE DETECTION ANALYSIS
// ==========================================

function displayFalseDetection(data) {

    addText(
        analysisDashboard,
        "h3",
        "False Detection Analysis"
    );

    const summary = data.false_detection_analysis;

    if (!summary) {

        addText(
            analysisDashboard,
            "p",
            "False detection summary is not available."
        );

    } else {

        const metrics = document.createElement("div");
        metrics.className = "metric-grid";

        Object.entries(summary).forEach(([key, value]) => {

            if (
                typeof value === "number" ||
                typeof value === "string"
            ) {

                addMetric(
                    metrics,
                    key.replaceAll("_", " "),
                    typeof value === "number"
                        ? formatNumber(value)
                        : value
                );

            }

        });

        analysisDashboard.appendChild(metrics);

    }

    const classes = data.false_detection_by_class || [];

    addText(
        analysisDashboard,
        "h3",
        "Class-wise False Detection Results"
    );

    if (classes.length === 0) {

        addText(
            analysisDashboard,
            "p",
            "Class-wise results are not available."
        );

    } else {

        const headers = Object.keys(classes[0]);

        const rows = classes.map(item =>
            headers.map(key => item[key] ?? "N/A")
        );

        createTable(
            analysisDashboard,
            headers,
            rows
        );

    }

    const incorrect = data.incorrect_detections || [];

    addText(
        analysisDashboard,
        "h3",
        "Incorrect Detection Samples"
    );

    addText(
        analysisDashboard,
        "p",
        `Displaying ${incorrect.length} available records (maximum 100).`
    );

    if (incorrect.length > 0) {

        const headers = Object.keys(incorrect[0]);

        const rows = incorrect.map(item =>
            headers.map(key => item[key] ?? "N/A")
        );

        createTable(
            analysisDashboard,
            headers,
            rows
        );

    }

}


// ==========================================
// LOAD MODEL EVALUATION DASHBOARD
// ==========================================

async function loadAnalysisDashboard() {

    analysisDashboard.innerHTML = "";

    addText(
        analysisDashboard,
        "p",
        "Loading evaluation data..."
    );

    try {

        const response = await fetch("/api/analysis");

        if (!response.ok) {

            throw new Error("Could not load evaluation API.");

        }

        const data = await response.json();

        analysisDashboard.innerHTML = "";

        if (!data.analysis_ready) {

            addText(
                analysisDashboard,
                "p",
                "Final analysis report has not been generated yet."
            );

            addText(
                analysisDashboard,
                "p",
                "Run: python models\\run_final_analysis.py",
                "note"
            );

        }

        displayCalibration(data);

        displayShiftAnalysis(data.shift_analysis);

        displayFalseDetection(data);

        addText(
            analysisDashboard,
            "p",
            "Note: This analysis evaluates recorded detections. It does not measure missed objects (false negatives).",
            "note"
        );

        addText(
            analysisDashboard,
            "p",
            "Calibration metrics are evaluated on the held-out dataset. Raw YOLO confidence is retained unless calibration demonstrates improvement.",
            "note"
        );

    } catch (error) {

        analysisDashboard.innerHTML = "";

        addText(
            analysisDashboard,
            "p",
            "Evaluation dashboard error: " + error.message,
            "error"
        );

    }

}


// ==========================================
// STOP CAMERA WHEN PAGE CLOSES
// ==========================================

window.addEventListener("beforeunload", function () {

    if (cameraStream) {

        cameraStream.getTracks().forEach(track => track.stop());

    }

});


// ==========================================
// INITIALIZE DASHBOARD
// ==========================================

loadAnalysisDashboard();
