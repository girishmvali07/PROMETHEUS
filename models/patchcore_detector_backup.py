
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import cv2

from PIL import Image
from torchvision import transforms

torch.set_num_threads(1)


class PCBPatchCore:

    def __init__(self, memory_limit=5000):

        self.device = "cpu"
        self.memory_limit = memory_limit
        self.memory_bank = None

        print("Loading PatchCore feature extractor...")

        self.model = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("PatchCore feature extractor loaded.")

    def extract_patch_features(self, image_path):

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image).unsqueeze(0)

        with torch.no_grad():

            features = self.model.forward_features(image)

            features = F.adaptive_avg_pool2d(
                features, (14, 14)
            )

            features = features.squeeze(0)
            features = features.permute(1, 2, 0)

            features = features.reshape(
                -1, features.shape[-1]
            )

        return features.cpu().numpy().astype(np.float32)

    def fit(self, normal_folder):

        folder = Path(normal_folder)
        images = list(folder.glob("*.jpg"))

        if not images:
            raise ValueError(
                "No candidate normal PCB images found."
            )

        memory_features = []
        rng = np.random.default_rng(42)

        print(
            f"Building patch memory from {len(images)} images..."
        )

        for index, image_path in enumerate(images):

            features = self.extract_patch_features(
                image_path
            )

            sample_count = min(20, len(features))

            selected_indices = rng.choice(
                len(features),
                size=sample_count,
                replace=False
            )

            memory_features.append(
                features[selected_indices]
            )

            if (index + 1) % 10 == 0:
                print(
                    f"Processed {index + 1}/{len(images)} images"
                )

        memory = np.concatenate(
            memory_features,
            axis=0
        )

        selected_count = min(
            self.memory_limit,
            len(memory)
        )

        selected_indices = rng.choice(
            len(memory),
            size=selected_count,
            replace=False
        )

        self.memory_bank = memory[selected_indices]

        print(
            "Patch memory bank created:",
            len(self.memory_bank),
            "patches"
        )

    def inspect(self, image_path):

        if self.memory_bank is None:
            raise RuntimeError(
                "Build the patch memory bank first."
            )

        query_features = self.extract_patch_features(
            image_path
        )

        anomaly_scores = []

        memory_tensor = torch.from_numpy(
            self.memory_bank
        )

        for start in range(0, len(query_features), 16):

            chunk = torch.from_numpy(
                query_features[start:start + 16]
            )

            distances = torch.cdist(
                chunk,
                memory_tensor
            )

            nearest_distances = distances.min(
                dim=1
            ).values

            anomaly_scores.extend(
                nearest_distances.numpy().tolist()
            )

        score_map = np.asarray(
            anomaly_scores,
            dtype=np.float32
        ).reshape(14, 14)

        with Image.open(image_path) as image:
            original = np.asarray(
                image.convert("RGB")
            )

        original_height, original_width = original.shape[:2]

        # Normalize anomaly scores for visualization.
        normalized_map = (
            (score_map - score_map.min()) /
            (score_map.max() - score_map.min() + 1e-8)
        )

        gray_map = np.uint8(255 * normalized_map)

        # Resize the grayscale map.
        gray_map = cv2.resize(
            gray_map,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC
        )

        # Generate a coloured heatmap.
        colored_heatmap = cv2.applyColorMap(
            gray_map,
            cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Overlay heatmap on original PCB.
        overlay = cv2.addWeighted(
            original,
            0.60,
            colored_heatmap,
            0.40,
            0
        )

        return {
            "anomaly_score": float(score_map.max()),
            "mean_anomaly_score": float(score_map.mean()),
            "heatmap": colored_heatmap,
            "gray_heatmap": gray_map,
            "overlay": overlay,
            "score_map": score_map,
            "status": "inspection_required"
        }


if __name__ == "__main__":

    print("\n========== PROMETHEUS-X PATCHCORE ==========")

    project_folder = Path(__file__).resolve().parent.parent

    normal_folder = (
        project_folder
        / "datasets"
        / "anomaly"
        / "normal"
    )

    detector = PCBPatchCore(memory_limit=5000)

    detector.fit(normal_folder)

    test_image = Path(
        r"D:\PROMETHEUS_DATA\PCB_MC\images\PCB-MC-A\kfold_data\fold_1\valid\images\ArduinoMega_Bottom_png_jpg.rf.0afe03bafd6bb2937ad7509946fd36d0.jpg"
    )

    if not test_image.exists():
        raise FileNotFoundError(
            f"Test image not found: {test_image}"
        )

    result = detector.inspect(test_image)

    results_folder = project_folder / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    # Save original PCB.
    original_output = results_folder / "patchcore_original.png"

    Image.open(test_image).convert("RGB").save(
        original_output
    )

    # Save coloured heatmap.
    heatmap_path = results_folder / "patchcore_colored_heatmap.png"

    Image.fromarray(result["heatmap"]).save(
        heatmap_path
    )

    # Save PCB with anomaly overlay.
    overlay_path = results_folder / "patchcore_overlay.png"

    Image.fromarray(result["overlay"]).save(
        overlay_path
    )

    print("\n========== INSPECTION RESULT ==========")

    print("Maximum anomaly score:", result["anomaly_score"])
    print("Mean anomaly score:", result["mean_anomaly_score"])

    print("Original PCB:", original_output)
    print("Colored heatmap:", heatmap_path)
    print("Overlay:", overlay_path)

    print("\nPROMETHEUS-X PATCHCORE TEST COMPLETED.")


