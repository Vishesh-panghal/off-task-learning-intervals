"""
Prepares deterministic, stratified splits and provenance hashes for
applied fine-tuning regime datasets:
- Target Task A: ISIC 2019 dermoscopy (8 classes, 25,331 images)
- Auxiliary Task B: Imagenette (10 classes, 13,394 images)

Produces:
- isic2019_splits.json
- imagenette_splits.json
- isic2019_probe_2000.json (deterministic 2,000 validation images for CKA & Rank)
- data_provenance_manifest.json (SHA-256 hashes of all artifacts)
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def sha256_file(filepath):
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def stratified_split_indices(labels, train_frac=0.80, val_frac=0.10, seed=42):
    """
    Deterministic stratified split into train, val, and test subsets.
    Guarantees exact class proportions across all splits.
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    
    train_indices = []
    val_indices = []
    test_indices = []

    for c in unique_classes:
        class_idxs = np.where(labels == c)[0]
        rng.shuffle(class_idxs)
        n = len(class_idxs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        
        train_idx = class_idxs[:n_train]
        val_idx = class_idxs[n_train:n_train + n_val]
        test_idx = class_idxs[n_train + n_val:]

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    # Sort indices for deterministic ordering
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    return train_indices, val_indices, test_indices


def prepare_isic2019(ground_truth_csv, images_dir=None, output_dir="./data_cache/splits", seed=42):
    """
    Parses ISIC 2019 ground truth, verifies labels, and generates stratified splits.
    """
    df = pd.read_csv(ground_truth_csv)
    print(f"[ISIC 2019] Loaded {len(df)} rows from {ground_truth_csv}")

    # Check columns
    for col in ISIC_CLASSES:
        if col not in df.columns:
            raise ValueError(f"Missing required class column '{col}' in {ground_truth_csv}")

    # Convert one-hot to integer labels 0..7
    label_matrix = df[ISIC_CLASSES].values
    int_labels = np.argmax(label_matrix, axis=1)
    
    # Verify exactly one 1.0 per row
    row_sums = label_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        invalid_rows = np.where(~np.isclose(row_sums, 1.0))[0]
        raise ValueError(f"Found {len(invalid_rows)} rows without a clean one-hot label in ISIC 2019.")

    image_ids = df["image"].tolist()

    # Stratified 80/10/10 split
    train_idx, val_idx, test_idx = stratified_split_indices(int_labels, train_frac=0.80, val_frac=0.10, seed=seed)

    print(f"[ISIC 2019] Split counts -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Verify class distribution across splits
    def get_dist(idxs):
        sub_labels = int_labels[idxs]
        counts = {ISIC_CLASSES[i]: int((sub_labels == i).sum()) for i in range(len(ISIC_CLASSES))}
        return counts

    train_dist = get_dist(train_idx)
    val_dist = get_dist(val_idx)
    test_dist = get_dist(test_idx)

    # Construct split record
    splits_record = {
        "dataset": "ISIC_2019",
        "num_classes": len(ISIC_CLASSES),
        "classes": ISIC_CLASSES,
        "seed": seed,
        "total_samples": len(df),
        "split_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "class_distributions": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
        "train": [{"image_id": image_ids[i], "label": int(int_labels[i]), "class_name": ISIC_CLASSES[int_labels[i]]} for i in train_idx],
        "val": [{"image_id": image_ids[i], "label": int(int_labels[i]), "class_name": ISIC_CLASSES[int_labels[i]]} for i in val_idx],
        "test": [{"image_id": image_ids[i], "label": int(int_labels[i]), "class_name": ISIC_CLASSES[int_labels[i]]} for i in test_idx],
    }

    os.makedirs(output_dir, exist_ok=True)
    isic_splits_path = os.path.join(output_dir, "isic2019_splits.json")
    with open(isic_splits_path, "w") as f:
        json.dump(splits_record, f, indent=2)
    print(f"[ISIC 2019] Saved splits to {isic_splits_path}")

    # Generate fixed 2,000-image diagnostic probe set (from validation subset)
    rng_probe = np.random.RandomState(seed + 100)
    val_len = len(val_idx)
    probe_sample_size = min(2000, val_len)
    probe_indices_in_val = sorted(rng_probe.choice(val_len, size=probe_sample_size, replace=False).tolist())
    probe_samples = [splits_record["val"][i] for i in probe_indices_in_val]

    probe_record = {
        "dataset": "ISIC_2019_Diagnostic_Probe",
        "seed": seed + 100,
        "source": "val",
        "num_samples": len(probe_samples),
        "samples": probe_samples,
    }
    probe_path = os.path.join(output_dir, "isic2019_probe_2000.json")
    with open(probe_path, "w") as f:
        json.dump(probe_record, f, indent=2)
    print(f"[ISIC 2019] Saved 2,000-sample diagnostic probe set to {probe_path}")

    return isic_splits_path, probe_path, splits_record


def prepare_imagenette(imagenette_root, output_dir="./data_cache/splits", seed=42):
    """
    Parses extracted Imagenette dataset directory and builds train/val records.
    Standard fast.ai Imagenette classes: 10 synsets.
    """
    imagenette_path = Path(imagenette_root)
    # Check if 'train' or 'val' subfolders exist or if flat
    train_dir = imagenette_path / "train" if (imagenette_path / "train").exists() else imagenette_path
    val_dir = imagenette_path / "val" if (imagenette_path / "val").exists() else None

    # Discover classes
    subdirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(subdirs)}

    all_samples = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = train_dir / cls_name
        for img_path in sorted(cls_dir.glob("*.*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                all_samples.append({
                    "rel_path": str(img_path.relative_to(imagenette_path)),
                    "label": cls_idx,
                    "class_name": cls_name,
                    "orig_split": "train",
                })

    if val_dir and val_dir.exists():
        for cls_name, cls_idx in class_to_idx.items():
            cls_vdir = val_dir / cls_name
            if cls_vdir.exists():
                for img_path in sorted(cls_vdir.glob("*.*")):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        all_samples.append({
                            "rel_path": str(img_path.relative_to(imagenette_path)),
                            "label": cls_idx,
                            "class_name": cls_name,
                            "orig_split": "val",
                        })

    print(f"[Imagenette] Found {len(all_samples)} total images across {len(class_to_idx)} classes.")

    # Stratified 90/10 split
    labels = np.array([s["label"] for s in all_samples])
    train_idx, val_idx, _ = stratified_split_indices(labels, train_frac=0.90, val_frac=0.10, seed=seed)

    imagenette_record = {
        "dataset": "Imagenette",
        "num_classes": len(class_to_idx),
        "classes": subdirs,
        "seed": seed,
        "total_samples": len(all_samples),
        "split_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
        },
        "train": [all_samples[i] for i in train_idx],
        "val": [all_samples[i] for i in val_idx],
    }

    os.makedirs(output_dir, exist_ok=True)
    imagenette_splits_path = os.path.join(output_dir, "imagenette_splits.json")
    with open(imagenette_splits_path, "w") as f:
        json.dump(imagenette_record, f, indent=2)
    print(f"[Imagenette] Saved splits to {imagenette_splits_path}")

    return imagenette_splits_path, imagenette_record


def build_provenance_manifest(file_paths, output_dir="./data_cache/splits"):
    """
    Computes SHA-256 hashes of all artifacts and outputs a provenance manifest.
    """
    manifest = {
        "generated_files": {},
    }
    for p in file_paths:
        if os.path.exists(p):
            h = sha256_file(p)
            size = os.path.getsize(p)
            manifest["generated_files"][os.path.basename(p)] = {
                "path": p,
                "sha256": h,
                "bytes": size,
            }

    manifest_path = os.path.join(output_dir, "data_provenance_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Saved provenance manifest to {manifest_path}")
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset splits and provenance hashes for fine-tuning regime.")
    parser.add_argument("--isic_csv", type=str, default="./data_cache/isic2019/ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--imagenette_dir", type=str, default="./data_cache/imagenette2")
    parser.add_argument("--output_dir", type=str, default="./data_cache/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generated_files = []

    if os.path.exists(args.isic_csv):
        isic_splits_path, probe_path, _ = prepare_isic2019(args.isic_csv, output_dir=args.output_dir, seed=args.seed)
        generated_files.extend([args.isic_csv, isic_splits_path, probe_path])
    else:
        print(f"[Warning] ISIC Ground Truth CSV not found at {args.isic_csv}")

    if os.path.exists(args.imagenette_dir):
        imgnet_splits_path, _ = prepare_imagenette(args.imagenette_dir, output_dir=args.output_dir, seed=args.seed)
        generated_files.append(imgnet_splits_path)
    else:
        print(f"[Warning] Imagenette dir not found at {args.imagenette_dir}")

    if generated_files:
        manifest_path, manifest = build_provenance_manifest(generated_files, output_dir=args.output_dir)
        print("\n--- PROVENANCE HASHES ---")
        for fname, info in manifest["generated_files"].items():
            print(f"{fname}: {info['sha256']}")


if __name__ == "__main__":
    main()

