import os, json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

base = "runs/detect"
out = os.path.join(base, "prometheus_final_analysis")
os.makedirs(out, exist_ok=True)

train = pd.read_csv(os.path.join(base, "yolo_calibration_train.csv"))
test = pd.read_csv(os.path.join(base, "yolo_calibration_test.csv"))

for df in (train, test):
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df.dropna(subset=["confidence", "correct", "class_id"], inplace=True)

# 1. Distribution shift analysis
ks = ks_2samp(train["confidence"], test["confidence"])
train_accuracy = float(train["correct"].mean())
test_accuracy = float(test["correct"].mean())

shift = {
    "calibration_groups": int(train.pcb_group.nunique()),
    "testing_groups": int(test.pcb_group.nunique()),
    "group_overlap": len(set(train.pcb_group) & set(test.pcb_group)),
    "calibration_confidence_mean": float(train.confidence.mean()),
    "testing_confidence_mean": float(test.confidence.mean()),
    "confidence_mean_difference": float(test.confidence.mean() - train.confidence.mean()),
    "confidence_ks_statistic": float(ks.statistic),
    "confidence_ks_p_value": float(ks.pvalue),
    "calibration_detection_correctness": train_accuracy,
    "testing_detection_correctness": test_accuracy,
    "correctness_difference": test_accuracy - train_accuracy
}

# 2. False detection analysis
test["incorrect"] = (test["correct"] == 0).astype(int)

by_class = test.groupby("class_id").agg(
    total_detections=("correct", "size"),
    correct_detections=("correct", "sum"),
    incorrect_detections=("incorrect", "sum"),
    average_confidence=("confidence", "mean")
).reset_index()

by_class["incorrect_detection_rate"] = (
    by_class["incorrect_detections"] / by_class["total_detections"]
)

by_class = by_class.sort_values(
    "incorrect_detection_rate", ascending=False
)

overall = {
    "total_test_detections": int(len(test)),
    "incorrect_detections": int(test.incorrect.sum()),
    "correct_detections": int(test.correct.sum()),
    "incorrect_detection_rate": float(test.incorrect.mean()),
    "correct_detection_rate": float(test.correct.mean())
}

# Save reports
by_class.to_csv(os.path.join(out, "false_detection_by_class.csv"), index=False)
test[test.incorrect == 1].to_csv(
    os.path.join(out, "incorrect_detections.csv"), index=False
)

report = {
    "shift_analysis": shift,
    "false_detection_analysis": overall,
    "interpretation": {
        "confidence_shift": "Statistical distribution comparison; not proof of image-domain shift.",
        "incorrect_detections": "Based on the existing correct flag and its matching criteria.",
        "limitations": "Analysis covers recorded detections only; missed objects are not measured."
    }
}

with open(os.path.join(out, "final_analysis_report.json"), "w") as f:
    json.dump(report, f, indent=4)

print("\n========== PROMETHEUS-X FINAL ANALYSIS ==========")
print("\nSHIFT ANALYSIS")
for k, v in shift.items():
    print(f"{k}: {v}")

print("\nFALSE DETECTION ANALYSIS")
for k, v in overall.items():
    print(f"{k}: {v}")

print("\nCLASS-WISE INCORRECT DETECTIONS")
print(by_class.to_string(index=False))

print("\nReports saved to:", out)
print("1. final_analysis_report.json")
print("2. false_detection_by_class.csv")
print("3. incorrect_detections.csv")
print("\n========== ANALYSIS COMPLETED ==========")
