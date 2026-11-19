
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fuzzy_mamdani_v3 import LABELS, predict
from evaluasi_final_v3 import load_frozen_model

BASE = ROOT / "hasil_glcm" / "analisis"
OUT = BASE / "pemeriksaan_300_citra"


def periksa_dataset(nama_csv, kelompok, feats, knots, consequents):
    path = BASE / nama_csv
    df = pd.read_csv(path)

    kolom_wajib = {"nama_file", "kelas", *feats}
    kolom_hilang = kolom_wajib - set(df.columns)

    if kolom_hilang:
        raise ValueError(f"Kolom hilang: {kolom_hilang}")

    if not df["kelas"].isin(LABELS).all():
        raise ValueError("Terdapat label kelas tidak dikenal")

    if not np.isfinite(df[list(feats)].to_numpy(dtype=float)).all():
        raise ValueError("Terdapat fitur kosong atau tidak valid")

    prediksi, areas = predict(
        df, feats, knots, consequents
    )

    hasil = df[["nama_file", "kelas"]].copy()
    hasil = hasil.rename(columns={"kelas": "aktual"})
    hasil["kelompok"] = kelompok

    if "group_id" in df.columns:
        hasil["group_id"] = df["group_id"]

    for fitur in feats:
        hasil[fitur] = df[fitur].to_numpy()

    hasil["prediksi"] = prediksi

    for i, label in enumerate(LABELS):
        hasil[f"luas_{label}"] = areas[:, i]

    hasil["benar"] = hasil["aktual"] == hasil["prediksi"]
    hasil["status"] = np.where(
        hasil["benar"], "BENAR", "SALAH"
    )

    return hasil


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    feats, knots, consequents = load_frozen_model()

    latih = periksa_dataset(
        "data_latih.csv", "LATIH",
        feats, knots, consequents
    )

    uji = periksa_dataset(
        "data_uji.csv", "UJI",
        feats, knots, consequents
    )

    semua = pd.concat(
        [latih, uji], ignore_index=True
    )

    salah = semua[~semua["benar"]].copy()

    semua.to_csv(
        OUT / "hasil_semua_citra.csv", index=False
    )

    salah.to_csv(
        OUT / "daftar_salah_klasifikasi.csv",
        index=False
    )

    print("\n=== PEMERIKSAAN FUZZY MAMDANI V3 ===")
    print("Fitur:", ", ".join(feats))
    print("Jumlah aturan:", len(consequents))

    for nama, data in [
        ("LATIH", latih),
        ("UJI", uji),
        ("SELURUH DATA", semua)
    ]:
        jumlah = len(data)
        benar = int(data["benar"].sum())
        salah_count = jumlah - benar

        print(f"\n--- {nama} ---")
        print("Jumlah citra:", jumlah)
        print("Benar:", benar)
        print("Salah:", salah_count)
        print(f"Akurasi: {benar / jumlah:.2%}")

        print("\nConfusion matrix:")
        print(pd.crosstab(
            data["aktual"],
            data["prediksi"]
        ).to_string())

    print("\n=== DAFTAR KESALAHAN ===")
    print(
        salah[
            ["nama_file", "kelompok", "aktual", "prediksi"]
        ].to_string(index=False)
    )

    print("\nHasil disimpan di:")
    print(OUT)
    print("\nModel V3 tidak diubah.")
    print("Hasil ini bukan diagnosis medis.")


if __name__ == "__main__":
    main()
