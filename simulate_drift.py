"""
simulate_drift.py

Purpose:
    Simulates a "production data stream" with realistic drift by taking the
    held-out test set (images the model has NEVER trained on) and applying
    randomized perturbations (brightness, contrast, blur) to mimic gradual
    real-world changes like lighting shifts, camera wear, or environmental
    variation on a factory floor.

    Ground truth labels are preserved via folder structure, since perturbing
    pixels does not change what the object actually is (a perturbed defective
    part is still defective).

Input:
    archive/casting_data/casting_data/test/  (class-subfolder structure, e.g. def_front/, ok_front/)

Output:
    production_stream/  (same class-subfolder structure, perturbed images)
"""

import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ---- Config ----
SOURCE_DIR = Path("archive/casting_data/casting_data/test")
OUTPUT_DIR = Path("production_stream")

# Randomized perturbation pool.
# Each image gets a RANDOM SUBSET and RANDOM INTENSITY of these effects,
# simulating gradual, uneven environmental drift rather than one fixed filter.
def get_random_perturbation():
    effects = []

    # Brightness/contrast jitter (simulates lighting changes over time)
    if random.random() < 0.7:
        brightness = random.uniform(0.6, 1.5)   # darker to brighter
        contrast = random.uniform(0.7, 1.4)
        effects.append(transforms.ColorJitter(brightness=(brightness, brightness),
                                                contrast=(contrast, contrast)))

    # Slight blur (simulates camera focus drift / motion / dust on lens)
    if random.random() < 0.4:
        kernel = random.choice([3, 5])
        effects.append(transforms.GaussianBlur(kernel_size=kernel, sigma=(0.3, 1.5)))

    # Fallback: if neither triggered, force at least a mild brightness shift
    # so every image has SOME drift applied.
    if not effects:
        effects.append(transforms.ColorJitter(brightness=0.2))

    return transforms.Compose(effects)


def simulate_drift():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {SOURCE_DIR}")

    total_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class_dir = OUTPUT_DIR / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(class_dir.glob("*.*"))
        print(f"Processing class '{class_name}': {len(image_paths)} images")

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")
                continue

            perturb = get_random_perturbation()
            perturbed_img = perturb(img)

            out_path = out_class_dir / img_path.name
            perturbed_img.save(out_path)
            total_images += 1

    print(f"\nDone. {total_images} perturbed images written to '{OUTPUT_DIR}/'")
    print("Folder structure mirrors the source, so labels remain implicit "
          "(same as your original ImageFolder-based training setup).")


if __name__ == "__main__":
    simulate_drift()