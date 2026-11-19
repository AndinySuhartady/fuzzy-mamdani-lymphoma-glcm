
"""Eksperimen V4: Confidence-Weighted Fuzzy Mamdani.

Perbandingan V3 dan V4 dengan validasi leave-one-prefix-group-out.
Hanya menggunakan data_latih.csv. Tidak membaca data_uji.csv.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from fuzzy_mamdani_v3 import (
    BASE, LABELS, TERMS,
    fit, firing, predict, metrics
)

OUT = BASE / "fuzzy_mamdani_v4"

FEATURES = (
    "correlation",
    "contrast",
    "homogeneity"
)

STRATEGY = "class_balanced"


def calculate_weights(votes):
    """Bobot dominasi kelas untuk setiap aturan."""
    total = votes.sum(axis=1)

    weights = np.zeros(len(total), dtype=float)
    active = total > 1e-12

    weights[active] = (
        votes[active].max(axis=1) / total[active]
    )

    return weights


def predict_v4(df, feats, knots, rules, weights):
    """Prediksi dengan aktivasi aturan berbobot."""
    acts = firing(df, feats, knots)

    weighted_acts = acts * weights[None, :]

    heights = np.column_stack([
        weighted_acts[:, rules == k].max(axis=1)
        if np.any(rules == k)
        else np.zeros(len(df))
        for k in range(len(LABELS))
    ])

    areas = heights - heights**2 / 2

    winners = areas.argmax(axis=1)
    best = areas.max(axis=1)

    ties = (
        np.abs(areas - best[:, None]) < 1e-10
    ).sum(axis=1) > 1

    preds = np.array(LABELS, dtype=object)[winners]

    preds[best <= 1e-12] = "TIDAK_TERKLASIFIKASI"
    preds[(best > 1e-12) & ties] = "AMBIGU"

    return preds, areas


def cross_validate_both(train):
    """Evaluasi V3 dan V4 pada fold yang identik."""
    pred_v3 = np.empty(len(train), dtype=object)
    pred_v4 = np.empty(len(train), dtype=object)

    group_values = train["group_id"].to_numpy()
    groups = sorted(train["group_id"].unique())

    for gid in groups:
        mask = group_values == gid

        fit_df = train.loc[~mask]
        val_df = train.loc[mask]

        knots, rules, votes = fit(
            fit_df, FEATURES, STRATEGY
        )

        weights = calculate_weights(votes)

        pred_v3[mask], _ = predict(
            val_df, FEATURES, knots, rules
        )

        pred_v4[mask], _ = predict_v4(
            val_df, FEATURES, knots, rules, weights
        )

    return pred_v3, pred_v4


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(BASE / "data_latih.csv")

    required = {
        "nama_file", "kelas", "group_id", *FEATURES
    }

    if not required.issubset(train.columns):
        raise ValueError(
            f"Kolom hilang: {required - set(train.columns)}"
        )

    if train[list(FEATURES)].isna().any().any():
        raise ValueError("Terdapat nilai fitur kosong.")

    if not np.isfinite(
        train[list(FEATURES)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Terdapat nilai fitur tidak hingga.")

    if set(train["kelas"]) != set(LABELS):
        raise ValueError("Label kelas tidak sesuai.")

    if train["group_id"].isna().any():
        raise ValueError("Ada group_id kosong.")

    if train.groupby("group_id")["kelas"].nunique().max() != 1:
        raise ValueError("Satu group_id memuat beberapa kelas.")

    print("=" * 65)
    print("PERBANDINGAN FUZZY MAMDANI V3 DAN V4")
    print("=" * 65)

    print("Data latih:", len(train))
    print("Jumlah grup:", train["group_id"].nunique())
    print("Fitur:", FEATURES)
    print("Strategi:", STRATEGY)

    pred_v3, pred_v4 = cross_validate_both(train)

    actual = train["kelas"].to_numpy()

    score_v3 = metrics(actual, pred_v3)
    score_v4 = metrics(actual, pred_v4)

    comparison = pd.DataFrame([
        {"model": "V3", **score_v3},
        {"model": "V4", **score_v4}
    ])

    comparison.to_csv(
        OUT / "perbandingan_v3_v4.csv",
        index=False
    )

    predictions = train[
        ["nama_file", "kelas", "group_id"]
    ].copy()

    predictions = predictions.rename(
        columns={"kelas": "aktual"}
    )

    predictions["prediksi_v3"] = pred_v3
    predictions["prediksi_v4"] = pred_v4

    predictions.to_csv(
        OUT / "prediksi_cv_v3_v4.csv",
        index=False
    )

    # Model final dilatih hanya pada seluruh data latih.
    knots, rules, votes = fit(
        train, FEATURES, STRATEGY
    )

    weights = calculate_weights(votes)

    import itertools

    rows = []

    for i, terms in enumerate(
        itertools.product(TERMS, repeat=len(FEATURES))
    ):
        row = {
            "rule_id": i + 1,
            **dict(zip(FEATURES, terms)),
            "consequent": (
                LABELS[rules[i]]
                if rules[i] >= 0
                else "TIDAK_AKTIF"
            ),
            "weight": float(weights[i])
        }

        for k, label in enumerate(LABELS):
            row[f"vote_{label}"] = float(votes[i, k])

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUT / "bobot_aturan_v4.csv",
        index=False
    )

    params = {
        "features": FEATURES,
        "strategy": STRATEGY,
        "knots": {
            f: knots[f].tolist()
            for f in FEATURES
        },
        "rules": rules.tolist(),
        "weights": weights.tolist(),
        "validation": "leave-one-prefix-group-out",
        "decision": "maximum clipped output area",
        "test_data_used": False
    }

    (OUT / "parameter_model_v4.json").write_text(
        json.dumps(params, indent=2),
        encoding="utf-8"
    )

    print("\nHASIL VALIDASI")
    print(comparison.to_string(index=False))

    delta = (
        score_v4["macro_f1"] - score_v3["macro_f1"]
    )

    print(f"\nSelisih macro-F1 V4 - V3: {delta:+.4f}")
    print("Prediksi berubah:", int((pred_v3 != pred_v4).sum()))

    print("\nOutput:", OUT)
    print("Data uji tidak digunakan.")
    print("Hasil bersifat eksploratif, bukan diagnosis klinis.")


if __name__ == "__main__":
    main()
