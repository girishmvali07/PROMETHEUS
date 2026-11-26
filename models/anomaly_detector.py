
import os

# Limit CPU thread usage to reduce memory pressure
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import timm

from PIL import Image
from torchvision import transforms

torch.set_num_threads(1)


class PCBAnomalyDetector:

    def __init__(self):

        self.device = "cpu"

        print("Loading anomaly detection model...")

        self.model = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.memory_bank = None

        print("Anomaly feature extractor loaded.")

    def extract_features(self, image_path):

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image).unsqueeze(0)

        with torch.no_grad():

            features = self.model.forward_features(image)

            features = torch.nn.functional.adaptive_avg_pool2d(
                features, (1, 1)
            )

        return features.flatten().cpu().numpy()

    def fit(self, normal_folder):

        folder = Path(normal_folder)

        images = list(folder.glob("*.jpg"))

        if not images:
            raise ValueError("No candidate normal PCB images found.")

        features = []

        print(
            f"Learning from {len(images)} candidate normal images..."
        )

        for index, image_path in enumerate(images):

            features.append(
                self.extract_features(image_path)
            )

            if (index + 1) % 25 == 0:
                print(
                    f"Processed {index + 1}/{len(images)} images"
                )

        self.memory_bank = np.asarray(
            features,
            dtype=np.float32
        )

        print("Anomaly reference memory bank created.")

    def inspect(self, image_path):

        if self.memory_bank is None:
            raise RuntimeError(
                "Train the anomaly detector first."
            )

        query = self.extract_features(image_path)

        distances = np.linalg.norm(
            self.memory_bank - query,
            axis=1
        )

        anomaly_score = float(np.min(distances))

        return {
            "anomaly_score": anomaly_score,
            "status": "inspection_required"
        }


if __name__ == "__main__":

    print("\n========== PROMETHEUS-X ==========")

    detector = PCBAnomalyDetector()

    project_folder = Path(__file__).resolve().parent.parent

    normal_folder = (
        project_folder
        / "datasets"
        / "anomaly"
        / "normal"
    )

    detector.fit(normal_folder)

    # Validation dataset paths

    valid_folder = Path(
        r"D:\PROMETHEUS_DATA\PCB_MC\images\PCB-MC-A\kfold_data\fold_1\valid"
    )

    image_folder = valid_folder / "images"
    label_folder = valid_folder / "labels"

    if not image_folder.exists() or not label_folder.exists():
        raise FileNotFoundError(
            "Validation images or labels folder not found."
        )

    # Missing-component class IDs: 15 to 22

    missing_classes = set(range(15, 23))

    # Match image filenames with label filenames

    image_lookup = {
        image.stem: image
        for image in image_folder.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    scores = defaultdict(list)

    processed = 0
    skipped = 0

    label_files = list(label_folder.glob("*.txt"))

    print("\nSTARTING BATCH ANOMALY EVALUATION...")
    print("Validation label files:", len(label_files))

    for label_file in label_files:

        image_path = image_lookup.get(label_file.stem)

        if image_path is None:
            skipped += 1
            continue

        lines = label_file.read_text().splitlines()

        has_missing_component = any(
            line.strip()
            and int(line.split()[0]) in missing_classes
            for line in lines
        )

        group = (
            "With missing-component labels"
            if has_missing_component
            else "Without missing-component labels"
        )

        result = detector.inspect(str(image_path))

        scores[group].append(
            result["anomaly_score"]
        )

        processed += 1

        if processed % 20 == 0:
            print(
                f"Processed {processed}/{len(label_files)} images"
            )

    # Display evaluation results

    print("\n========== EVALUATION RESULTS ==========")

    print("Total processed:", processed)
    print("Skipped:", skipped)

    positive = scores["With missing-component labels"]
    negative = scores["Without missing-component labels"]

    for group, values in scores.items():

        if values:

            print("\n" + group)
            print("Count:", len(values))
            print("Mean score:", float(np.mean(values)))
            print("Median score:", float(np.median(values)))
            print("Minimum:", float(np.min(values)))
            print("Maximum:", float(np.max(values)))

    # Pairwise ranking evaluation

    if positive and negative:

        auc = sum(
            (p > n) + 0.5 * (p == n)
            for p in positive
            for n in negative
        ) / (len(positive) * len(negative))

        print("\nPairwise ranking AUC:", round(float(auc), 4))

        print(
            "AUC above 0.5 means higher scores tend to occur "
            "in the missing-component-labelled group."
        )

    else:

        print(
            "\nCould not calculate AUC because one evaluation group is empty."
        )

    print("\nPROMETHEUS-X BATCH EVALUATION COMPLETED.")
