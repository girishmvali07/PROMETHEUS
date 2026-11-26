import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

BASE = "runs/detect"
TRAIN = os.path.join(BASE, "yolo_calibration_train.csv")
TEST = os.path.join(BASE, "yolo_calibration_test.csv")
SOURCE = os.path.join(BASE, "yolo_confidence_calibration.csv")
OUT = os.path.join(BASE, "yolo_calibration_results")
os.makedirs(OUT, exist_ok=True)

print("\n========== PROMETHEUS-X YOLO CALIBRATION ==========")

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
source = pd.read_csv(SOURCE)

required = {"confidence", "correct", "pcb_group"}
for name, df in [("TRAIN", train), ("TEST", test)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV missing columns: {missing}")

for name, df in [("TRAIN", train), ("TEST", test)]:
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df.dropna(subset=["confidence", "correct", "pcb_group"], inplace=True)
    df["confidence"] = df["confidence"].clip(0, 1)
    df["correct"] = df["correct"].astype(int)

overlap = set(train.pcb_group) & set(test.pcb_group)
if overlap:
    raise ValueError(f"DATA LEAKAGE: {len(overlap)} overlapping groups!")

if train.correct.nunique() < 2:
    raise ValueError("Training data must contain both correct and incorrect predictions.")

X_train = train[["confidence"]].values
y_train = train["correct"].values
X_test = test[["confidence"]].values
y_test = test["correct"].values

print("Calibration groups:", train.pcb_group.nunique())
print("Testing groups:", test.pcb_group.nunique())
print("Calibration predictions:", len(train))
print("Testing predictions:", len(test))
print("Group overlap:", len(overlap))
print("Third CSV source rows:", len(source))

platt = LogisticRegression(max_iter=1000)
platt.fit(X_train, y_train)
platt_test = platt.predict_proba(X_test)[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(train.confidence.values, y_train)
iso_test = iso.predict(test.confidence.values)

def metrics(name, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    brier = brier_score_loss(y_test, probabilities)
    nll = log_loss(y_test, probabilities, labels=[0, 1])

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    bin_rows = []

    for i in range(10):
        mask = (probabilities >= bins[i]) & (
            probabilities < bins[i + 1] if i < 9
            else probabilities <= bins[i + 1]
        )
        if mask.any():
            avg_conf = float(probabilities[mask].mean())
            accuracy = float(y_test[mask].mean())
            count = int(mask.sum())
            ece += count / len(y_test) * abs(avg_conf - accuracy)
            bin_rows.append({
                "method": name,
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "count": count,
                "mean_confidence": avg_conf,
                "accuracy": accuracy
            })

    return {
        "method": name,
        "brier_score": float(brier),
        "log_loss": float(nll),
        "ECE": float(ece)
    }, bin_rows

results = []
all_bins = []

for name, predictions in [
    ("Raw YOLO", test.confidence.values),
    ("Platt Scaling", platt_test),
    ("Isotonic Regression", iso_test)
]:
    result, bins = metrics(name, predictions)
    results.append(result)
    all_bins.extend(bins)

    print("\n", name)
    print("Brier score:", round(result["brier_score"], 5))
    print("Log loss:", round(result["log_loss"], 5))
    print("ECE:", round(result["ECE"], 5))

pd.DataFrame(results).to_csv(
    os.path.join(OUT, "calibration_metrics.csv"), index=False
)
pd.DataFrame(all_bins).to_csv(
    os.path.join(OUT, "reliability_bins.csv"), index=False
)

test_output = test.copy()
test_output["raw_confidence"] = test.confidence
test_output["platt_calibrated"] = platt_test
test_output["isotonic_calibrated"] = iso_test
test_output.to_csv(
    os.path.join(OUT, "calibrated_test_predictions.csv"), index=False
)

with open(os.path.join(OUT, "calibration_report.json"), "w") as f:
    json.dump({
        "calibration_groups": int(train.pcb_group.nunique()),
        "testing_groups": int(test.pcb_group.nunique()),
        "calibration_predictions": len(train),
        "testing_predictions": len(test),
        "group_overlap": len(overlap),
        "source_csv_rows": len(source),
        "metrics": results,
        "note": "Metrics are conditional on YOLO detections. They do not measure missed-object recall or guarantee production reliability."
    }, f, indent=4)

print("\n========== CALIBRATION COMPLETED ==========")
print("Results folder:", OUT)
print("Files created:")
print("1. calibration_metrics.csv")
print("2. reliability_bins.csv")
print("3. calibrated_test_predictions.csv")
print("4. calibration_report.json")
