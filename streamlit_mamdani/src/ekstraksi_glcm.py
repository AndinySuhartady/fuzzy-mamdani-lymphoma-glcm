"""Ekstraksi fitur GLCM untuk dataset histopatologi CLL, FL, dan MCL.

Jalankan dari mana saja:
    py D:\\Proyek_Komputasi_Lunak\\src\\ekstraksi_glcm.py --limit-per-class 100
    py D:\\Proyek_Komputasi_Lunak\\src\\ekstraksi_glcm.py --limit-per-class 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

CLASSES = ("CLL", "FL", "MCL")
EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
FEATURES = ("contrast", "homogeneity", "energy", "correlation")
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def extract_features(path: Path, size: int = 256, levels: int = 32) -> dict[str, float]:
    with Image.open(path) as source:
        source.seek(0)
        # RGB terlebih dahulu agar TIFF multikanal dapat dikonversi konsisten.
        rgb = source.convert("RGB")
        gray = ImageOps.grayscale(rgb)
        gray = ImageOps.fit(gray, (size, size), method=Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)

    quantized = (pixels.astype(np.uint16) * levels // 256).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=list(ANGLES),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    result = {}
    for feature in FEATURES:
        per_angle = graycoprops(glcm, feature)[0]
        for degree, value in zip((0, 45, 90, 135), per_angle):
            result[f"{feature}_{degree}"] = float(value)
        result[feature] = float(np.mean(per_angle))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekstraksi GLCM dataset limfoma")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "hasil_glcm" / "hasil_glcm.csv")
    parser.add_argument("--limit-per-class", type=int, default=100, help="0 = semua citra per kelas")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=32)
    args = parser.parse_args()
    if args.limit_per_class < 0 or args.size < 2 or not (2 <= args.levels <= 256):
        parser.error("limit harus >=0, size >=2, dan levels antara 2 sampai 256")

    if not args.dataset.is_dir():
        parser.error(f"Folder dataset tidak ditemukan: {args.dataset}")

    rows = []
    errors = []
    for label in CLASSES:
        folder = args.dataset / label
        if not folder.is_dir():
            parser.error(f"Folder kelas tidak ditemukan: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if args.limit_per_class:
            paths = paths[:args.limit_per_class]
        print(f"{label}: {len(paths)} file diproses")
        for path in tqdm(paths, desc=f"GLCM {label}", unit="gambar"):
            try:
                values = extract_features(path, args.size, args.levels)
                rows.append({"nama_file": path.name, "kelas": label, **values})
            except Exception as exc:
                errors.append({"nama_file": str(path), "kelas": label, "error": str(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSelesai: {len(rows)} gambar berhasil diproses")
    print(f"CSV: {args.output}")
    if rows:
        print("\nRata-rata fitur per kelas (eksplorasi saja):")
        print(pd.DataFrame(rows).groupby("kelas")[list(FEATURES)].agg(["mean", "std"]).round(5).to_string())
    if errors:
        error_file = args.output.with_name("error_glcm.csv")
        pd.DataFrame(errors).to_csv(error_file, index=False, encoding="utf-8-sig")
        print(f"PERINGATAN: {len(errors)} file gagal dibaca; lihat {error_file}")


if __name__ == "__main__":
    main()
