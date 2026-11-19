"""Fuzzy Mamdani V2: tiga kanal kelas nominal CLL, FL, MCL.

Jalankan dari akar proyek Windows:
    py src/fuzzy_mamdani_v2.py
    py src/fuzzy_mamdani_v2.py --evaluate-test

Default: evaluasi silang berbasis group_id hanya pada data latih.
--evaluate-test: evaluasi data uji SATU KALI setelah model dipilih.
Eksperimen citra histopatologi, BUKAN alat diagnosis klinis.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "hasil_glcm" / "analisis"
OUT = BASE / "fuzzy_mamdani_v2"
FEATURES = ("correlation", "contrast")
LABELS = ("CLL", "FL", "MCL")
TERMS = ("rendah", "sedang", "tinggi")
GRID = np.linspace(0.0, 1.0, 1001)
# Output masing-masing kelas berada pada domain dukungan 0..1, bukan satu sumbu penyakit.
# Bentuk output sama untuk ketiga kelas agar luas agregasi dapat dibandingkan.
OUTPUT_SHAPE = np.maximum(0.0, 1.0 - np.abs(GRID - 0.5) / 0.5)


def memberships(x, knots):
    q1, q2, q3 = map(float, knots)
    if not (q1 < q2 < q3):
        raise ValueError(f"Kuartil harus naik tegas, diperoleh {knots}")
    low = np.clip((q2 - x) / (q2 - q1), 0.0, 1.0)
    mid = np.clip(min((x - q1) / (q2 - q1), (q3 - x) / (q3 - q2)), 0.0, 1.0)
    high = np.clip((x - q2) / (q3 - q2), 0.0, 1.0)
    return np.array([low, mid, high], dtype=float)


def fit(train):
    knots = {f: train[f].quantile([.25, .5, .75]).tolist() for f in FEATURES}
    votes = np.zeros((3, 3, 3), dtype=float)
    for row in train.itertuples(index=False):
        a = memberships(float(getattr(row, FEATURES[0])), knots[FEATURES[0]])
        b = memberships(float(getattr(row, FEATURES[1])), knots[FEATURES[1]])
        cls = LABELS.index(row.kelas)
        votes[:, :, cls] += np.minimum(a[:, None], b[None, :])
    priors = train["kelas"].value_counts().reindex(LABELS, fill_value=0).to_numpy()
    rules = []
    for i in range(3):
        for j in range(3):
            v = votes[i, j]
            winner = int(np.argmax(v)) if v.sum() else int(np.argmax(priors))
            rules.append({"correlation": TERMS[i], "contrast": TERMS[j],
                          "output": LABELS[winner], "support": float(v[winner]),
                          **{f"votes_{c}": float(v[k]) for k, c in enumerate(LABELS)}})
    return knots, rules


def predict_one(row, knots, rules):
    a = memberships(float(row[FEATURES[0]]), knots[FEATURES[0]])
    b = memberships(float(row[FEATURES[1]]), knots[FEATURES[1]])
    aggregates = {c: np.zeros_like(GRID) for c in LABELS}
    fired = []
    for i in range(3):
        for j in range(3):
            alpha = float(min(a[i], b[j]))
            if alpha <= 0:
                continue
            rule = rules[i * 3 + j]
            label = rule["output"]
            aggregates[label] = np.maximum(aggregates[label], np.minimum(alpha, OUTPUT_SHAPE))
            fired.append(f"{TERMS[i]} & {TERMS[j]} -> {label} ({alpha:.5f})")
    # Centroid dihitung per kanal, tetapi TIDAK dipakai untuk menentukan kelas:
    # centroid dari tiga himpunan simetris identik akan selalu ~0.5.
    # Luas agregasi (integral derajat keanggotaan) menjadi skor dukungan kelas.
    scores = {c: float(np.trapezoid(aggregates[c], GRID)) for c in LABELS}
    centroids = {c: (float(np.trapezoid(GRID * aggregates[c], GRID) / scores[c])
                     if scores[c] > 0 else np.nan) for c in LABELS}
    best = max(scores.values())
    if best <= 0:
        pred = "TIDAK_TERKLASIFIKASI"
    else:
        winners = [c for c in LABELS if abs(scores[c] - best) < 1e-10]
        pred = winners[0] if len(winners) == 1 else "AMBIGU"
    return pred, scores, centroids, fired


def evaluate(df, name, knots, rules, save=True):
    records = []
    for _, row in df.iterrows():
        pred, scores, centroids, fired = predict_one(row, knots, rules)
        records.append({"nama_file": row["nama_file"], "aktual": row["kelas"],
                        "prediksi": pred, **{f"skor_{c}": scores[c] for c in LABELS},
                        **{f"centroid_{c}": centroids[c] for c in LABELS},
                        "aturan_aktif": "; ".join(fired)})
    result = pd.DataFrame(records)
    matrix = pd.crosstab(pd.Categorical(result["aktual"], categories=LABELS),
                         pd.Categorical(result["prediksi"], categories=[*LABELS, "AMBIGU", "TIDAK_TERKLASIFIKASI"]),
                         rownames=["Aktual"], colnames=["Prediksi"], dropna=False)
    acc = float((result["aktual"] == result["prediksi"]).mean())
    recalls = {c: float(((result["aktual"] == c) & (result["prediksi"] == c)).sum() /
                        max(1, (result["aktual"] == c).sum())) for c in LABELS}
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUT / f"prediksi_{name}.csv", index=False)
        matrix.to_csv(OUT / f"confusion_matrix_{name}.csv")
    print(f"\n{name.upper()}: n={len(df)}, akurasi={acc:.2%}, macro-recall={np.mean(list(recalls.values())):.2%}")
    print(matrix.to_string())
    return acc


def validate(df):
    required = {"nama_file", "kelas", "group_id", *FEATURES}
    if not required.issubset(df.columns):
        raise ValueError(f"Kolom tidak ada: {sorted(required - set(df.columns))}")
    if df[list(FEATURES)].isna().any().any() or not np.isfinite(df[list(FEATURES)].to_numpy()).all():
        raise ValueError("Ada fitur kosong/tidak hingga")
    if not set(df["kelas"]).issubset(LABELS):
        raise ValueError("Label kelas tidak dikenal")


def cross_validate(train):
    # Leave-one-group-out; semua citra satu kelompok ditahan bersama.
    groups = list(train.groupby(["kelas", "group_id"], sort=True))
    results = []
    for (label, gid), held in groups:
        fit_df = train.drop(index=held.index)
        if set(fit_df["kelas"]) != set(LABELS):
            raise ValueError(f"Kelompok {gid}: kelas hilang dari fold pelatihan")
        knots, rules = fit(fit_df)
        for _, row in held.iterrows():
            pred, _, _, _ = predict_one(row, knots, rules)
            results.append({"nama_file": row["nama_file"], "aktual": row["kelas"],
                            "prediksi": pred, "group_id": gid})
    cv = pd.DataFrame(results)
    cv.to_csv(OUT / "prediksi_cv_latih.csv", index=False)
    accuracy = (cv["aktual"] == cv["prediksi"]).mean()
    print(f"\nVALIDASI GROUP-HELD-OUT (data latih): {len(groups)} fold, n={len(cv)}, akurasi={accuracy:.2%}")
    print(pd.crosstab(cv["aktual"], cv["prediksi"]).to_string())
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true", help="Gunakan data uji final satu kali")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(BASE / "data_latih.csv")
    validate(train)
    knots, rules = fit(train)
    pd.DataFrame(rules).to_csv(OUT / "aturan_fuzzy_v2.csv", index=False)
    (OUT / "parameter_keanggotaan_v2.json").write_text(json.dumps(knots, indent=2), encoding="utf-8")
    print("\nATURAN V2 (berdasarkan data latih):")
    print(pd.DataFrame(rules)[["correlation", "contrast", "output", "support"]].to_string(index=False))
    cross_validate(train)
    evaluate(train, "latih", knots, rules)
    if args.evaluate_test:
        test = pd.read_csv(BASE / "data_uji.csv")
        validate(test)
        overlap = set(zip(train["kelas"], train["group_id"])) & set(zip(test["kelas"], test["group_id"]))
        if overlap:
            raise ValueError(f"Kelompok bocor latih/uji: {overlap}")
        evaluate(test, "uji", knots, rules)
    else:
        print("\nData uji sengaja tidak digunakan. Pakai --evaluate-test hanya setelah model final.")
    print("\nFile keluaran:", OUT)
    print("Bukan alat diagnosis medis.")

if __name__ == "__main__":
    main()
