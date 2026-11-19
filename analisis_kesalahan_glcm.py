
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "hasil_glcm" / "analisis"
FOLDER = BASE / "pemeriksaan_300_citra"

df = pd.read_csv(FOLDER / "hasil_semua_citra.csv")

FITUR = ["correlation", "contrast", "homogeneity"]

# Fokus evaluasi utama: data uji
uji = df[df["kelompok"] == "UJI"].copy()

# Analisis tambahan: seluruh dataset
semua = df.copy()

def analisis(data, nama):
    print(f"\n{'=' * 60}")
    print(f"ANALISIS {nama}")
    print(f"{'=' * 60}")

    print("\n1. RATA-RATA FITUR BERDASARKAN KELAS ASLI")
    print(
        data.groupby("aktual")[FITUR]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\n2. RATA-RATA FITUR BERDASARKAN")
    print("   KELAS ASLI DAN HASIL PREDIKSI")
    print(
        data.groupby(["aktual", "prediksi"])[FITUR]
        .agg(["count", "mean"])
        .round(4)
        .to_string()
    )

    print("\n3. POLA KESALAHAN TERBANYAK")
    salah = data[data["aktual"] != data["prediksi"]]

    print(
        salah.groupby(["aktual", "prediksi"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n4. KESALAHAN BERDASARKAN GROUP_ID")
    if "group_id" in data.columns:
        print(
            salah.groupby(["aktual", "group_id"])
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )

    return data.groupby(
        ["aktual", "prediksi"]
    )[FITUR].mean().reset_index()


hasil_uji = analisis(uji, "DATA UJI")
hasil_semua = analisis(semua, "SELURUH 300 CITRA")

hasil_uji.to_csv(
    FOLDER / "analisis_fitur_per_prediksi_uji.csv",
    index=False
)

hasil_semua.to_csv(
    FOLDER / "analisis_fitur_per_prediksi_semua.csv",
    index=False
)

print("\nAnalisis selesai. Model tidak diubah.")
