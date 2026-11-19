
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "hasil_glcm" / "hasil_glcm.csv"
OUTPUT = ROOT / "hasil_glcm" / "analisis"

FEATURES = ["contrast", "homogeneity", "energy", "correlation"]
CLASSES = ["CLL", "FL", "MCL"]


def group_id(filename):
    # Contoh: sj-03-2810_001.tif -> sj-03-2810
    return re.sub(r"_\d+$", "", Path(filename).stem)


def eta_squared(df, feature):
    overall = df[feature].mean()
    ss_between = sum(
        len(g) * (g[feature].mean() - overall) ** 2
        for _, g in df.groupby("kelas")
    )
    ss_total = ((df[feature] - overall) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


def split_groups(df, target=0.20):
    # Kelompok gambar yang sama tidak boleh terbagi.
    rng = np.random.default_rng(42)
    test_indices = []

    for label in CLASSES:
        subset = df[df["kelas"] == label]
        groups = [
            g.index.to_numpy()
            for _, g in subset.groupby("group_id")
        ]

        if len(groups) < 2:
            raise ValueError(
                f"Kelas {label} memiliki kurang dari 2 kelompok."
            )

        rng.shuffle(groups)
        chosen = []
        target_count = len(subset) * target

        for group in groups:
            if len(chosen) >= target_count:
                break
            if len(chosen) + len(group) >= len(subset):
                continue
            chosen.extend(group.tolist())

        test_indices.extend(chosen)

    test = df.loc[test_indices].copy()
    train = df.drop(index=test_indices).copy()

    return train, test


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    required = ["nama_file", "kelas"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    df["group_id"] = df["nama_file"].map(group_id)

    train, test = split_groups(df)

    train.to_csv(OUTPUT / "data_latih.csv", index=False)
    test.to_csv(OUTPUT / "data_uji.csv", index=False)

    stats = train.groupby("kelas")[FEATURES].agg(
        ["mean", "std", "min", "median", "max"]
    )
    stats.to_csv(OUTPUT / "statistik_data_latih.csv")

    ranking = pd.DataFrame({
        "fitur": FEATURES,
        "eta_squared": [
            eta_squared(train, f) for f in FEATURES
        ]
    }).sort_values("eta_squared", ascending=False)

    ranking.to_csv(
        OUTPUT / "peringkat_fitur.csv", index=False
    )

    train[FEATURES].corr().to_csv(
        OUTPUT / "korelasi_antarfitur_latih.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, feature in zip(axes.flat, FEATURES):
        values = [
            train.loc[train["kelas"] == c, feature]
            for c in CLASSES
        ]
        ax.boxplot(values, tick_labels=CLASSES)
        ax.set_title(feature.capitalize())
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Nilai GLCM")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Distribusi Fitur GLCM - Data Latih")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "boxplot_data_latih.png", dpi=180
    )
    plt.close(fig)

    # Pastikan tidak ada kelompok yang bocor.
    train_groups = set(zip(train["kelas"], train["group_id"]))
    test_groups = set(zip(test["kelas"], test["group_id"]))
    assert train_groups.isdisjoint(test_groups)

    print("\n=== PEMBAGIAN DATA ===")
    print("Data latih:", len(train))
    print("Data uji:", len(test))
    print("\nData latih per kelas:")
    print(train["kelas"].value_counts().sort_index())
    print("\nData uji per kelas:")
    print(test["kelas"].value_counts().sort_index())

    print("\n=== PERINGKAT FITUR ===")
    print(ranking.to_string(index=False))
    print("\nHasil tersimpan di:", OUTPUT)


if __name__ == "__main__":
    main()
