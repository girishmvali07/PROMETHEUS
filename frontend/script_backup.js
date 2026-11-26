
const imageInput = document.getElementById("imageInput");
const cameraButton = document.getElementById("cameraButton");
const captureButton = document.getElementById("captureButton");
const inspectButton = document.getElementById("inspectButton");

const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");

const statusBox = document.getElementById("status");
const report = document.getElementById("report");

let selectedImage = null;
let cameraStream = null;


// IMAGE UPLOAD

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


// OPEN CAMERA

cameraButton.addEventListener("click", async function () {
    try {
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


// CAPTURE IMAGE

captureButton.addEventListener("click", function () {
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;

    canvas.getContext("2d").drawImage(
        camera, 0, 0, canvas.width, canvas.height
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


// DISPLAY INSPECTION RESULTS

function displayResults(data) {
    report.innerHTML = "";

    const inspection = data.inspection;

    const heading = document.createElement("h3");
    heading.textContent = "PROMETHEUS-X Inspection Report";
    report.appendChild(heading);

    const details = document.createElement("p");
    details.textContent =
        `Image: ${data.filename} | Size: ${data.width} x ${data.height}`;
    report.appendChild(details);
    if (data.annotated_image) {
    const annotated = document.createElement("img");

    annotated.src = data.annotated_image + "?t=" + Date.now();
    annotated.alt = "Annotated PCB inspection result";
    annotated.style.width = "100%";
    annotated.style.maxWidth = "800px";
    annotated.style.marginTop = "15px";
    annotated.style.borderRadius = "10px";

    report.appendChild(annotated);
}

    if (inspection.status !== "completed") {
        const message = document.createElement("p");
        message.textContent = inspection.message;
        report.appendChild(message);
        return;
    }

    const count = document.createElement("h4");
    count.textContent = `Detected Objects: ${inspection.count}`;
    report.appendChild(count);

    if (inspection.count === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No objects detected above the confidence threshold.";
        report.appendChild(empty);
        return;
    }

    inspection.detections.forEach((detection, index) => {
        const item = document.createElement("div");

        const name = document.createElement("h4");
        name.textContent = `${index + 1}. ${detection.class}`;

        const confidence = document.createElement("p");
        confidence.textContent =
            `Confidence: ${(detection.confidence * 100).toFixed(1)}%`;

        const coordinates = document.createElement("p");
        coordinates.textContent =
            `Bounding box: ${detection.box.join(", ")}`;

        item.appendChild(name);
        item.appendChild(confidence);
        item.appendChild(coordinates);

        report.appendChild(item);
    });
}


// SEND IMAGE TO BACKEND

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

    } finally {
        inspectButton.disabled = false;
    }
});


// STOP CAMERA WHEN PAGE CLOSES

window.addEventListener("beforeunload", function () {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
});
