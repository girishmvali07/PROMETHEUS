
from pathlib import Path
import csv

import numpy as np

from patchcore_detector import PCBPatchCore


# ==========================================
# PROMETHEUS-X PATCHCORE VALIDATION
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATASET_FOLDER = Path(
    r"D:\PROMETHEUS_DATA\PCB_MC\images\PCB-MC-A\kfold_data\fold_1"
)

NORMAL_FOLDER = (
    PROJECT_FOLDER / "datasets" / "anomaly" / "normal"
)

VALID_IMAGES = DATASET_FOLDER / "valid" / "images"
VALID_LABELS = DATASET_FOLDER / "valid" / "labels"

RESULTS_FOLDER = PROJECT_FOLDER / "results"

MISSING_COMPONENT_IDS = {
    15, 16, 17, 18, 19, 20, 21, 22
}


def get_missing_component_label(label_path):

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    with open(label_path, "r") as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(float(parts[0]))

            if class_id in MISSING_COMPONENT_IDS:
                return 1

    return 0


def calculate_auc(positive_scores, negative_scores):

    if not positive_scores or not negative_scores:
        return None

    wins = 0
    ties = 0

    for positive in positive_scores:

        for negative in negative_scores:

            if positive > negative:
                wins += 1

            elif positive == negative:
                ties += 1

    total = len(positive_scores) * len(negative_scores)

    return (wins + 0.5 * ties) / total


def main():

    print("\n========== PROMETHEUS-X VALIDATION ==========")

    if not VALID_IMAGES.exists():
        raise FileNotFoundError(VALID_IMAGES)

    if not VALID_LABELS.exists():
        raise FileNotFoundError(VALID_LABELS)

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    images = sorted(
        image for image in VALID_IMAGES.iterdir()
        if image.suffix.lower() in image_extensions
    )

    if not images:
        raise ValueError("No validation images found.")

    print("Validation images:", len(images))

    detector = PCBPatchCore(memory_limit=5000)

    detector.fit(NORMAL_FOLDER)

    positive_max = []
    negative_max = []

    positive_mean = []
    negative_mean = []

    records = []

    for index, image_path in enumerate(images):

        label_path = VALID_LABELS / (
            image_path.stem + ".txt"
        )

        ground_truth = get_missing_component_label(
            label_path
        )

        result = detector.inspect(image_path)

        max_score = result["anomaly_score"]
        mean_score = result["mean_anomaly_score"]

        if ground_truth == 1:

            positive_max.append(max_score)
            positive_mean.append(mean_score)

        else:

            negative_max.append(max_score)
            negative_mean.append(mean_score)

        records.append({
            "image": image_path.name,
            "missing_component_label": ground_truth,
            "maximum_anomaly_score": max_score,
            "mean_anomaly_score": mean_score
        })

        if (index + 1) % 10 == 0:

            print(
                f"Evaluated {index + 1}/{len(images)} images"
            )

    max_auc = calculate_auc(
        positive_max,
        negative_max
    )

    mean_auc = calculate_auc(
        positive_mean,
        negative_mean
    )

    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        RESULTS_FOLDER / "patchcore_validation.csv"
    )

    with open(report_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "missing_component_label",
                "maximum_anomaly_score",
                "mean_anomaly_score"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n========== VALIDATION RESULTS ==========")

    print("Total images:", len(images))
    print("Images with missing-component labels:", len(positive_max))
    print("Images without missing-component labels:", len(negative_max))

    print("Maximum-score AUC:", max_auc)
    print("Mean-score AUC:", mean_auc)

    print("CSV report:", report_path)

    print("\nPROMETHEUS-X VALIDATION COMPLETED.")


if __name__ == "__main__":
    main()
