"""Grafik evaluasi Mamdani V3 vs V4, berdasarkan hasil eksperimen proyek.
Jalankan: python grafik_evaluasi_fuzzy_v3_v4.py
Dependensi: pip install pandas numpy matplotlib
Tempatkan script di root Proyek_Komputasi_Lunak; file CSV CV opsional.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "gambar_laporan"
OUT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'gambar_laporan'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':150,'savefig.dpi':220})
COLORS = {'V3':'#3569A8','V4':'#D27B39'}

def locate(filename):
    candidates = [ROOT / filename, ROOT / 'hasil_glcm' / 'analisis' / 'fuzzy_mamdani_v4' / filename,
                  ROOT / 'hasil_glcm' / 'analisis' / filename]
    return next((p for p in candidates if p.exists()), None)

def save(name):
    plt.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Tersimpan:', OUT / name)

# Angka hasil cross-validation terdokumentasi dari eksperimen V3 dan V4.
# Jangan mengganti dengan skor pengujian 81 citra (protokol berbeda).
metrics = pd.DataFrame({
    'Metrik':['Accuracy','Macro-F1','Macro-recall'],
    'V3':[0.388128,0.356850,0.383150],
    'V4':[0.369863,0.333111,0.365035],
})
p = locate('perbandingan_v3_v4.csv')
if p:
    print('CSV perbandingan tersedia:', p, '\nPeriksa kesesuaian metrik:', pd.read_csv(p).head().to_string(index=False))

# Gambar 4 — performa
fig, ax = plt.subplots(figsize=(9,4.8))
x = np.arange(len(metrics)); width=.32
for offset, model in [(-width/2,'V3'),(width/2,'V4')]:
    bars = ax.bar(x+offset, metrics[model]*100, width, color=COLORS[model],label=model)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.6,f'{b.get_height():.2f}%',ha='center',fontsize=9)
ax.set_xticks(x,metrics['Metrik']); ax.set_ylim(0,49)
ax.set_ylabel('Nilai (%)'); ax.set_title('Perbandingan evaluasi cross-validation V3 dan V4')
ax.legend(frameon=False); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
fig.tight_layout(); save('gambar4_perbandingan_performa.png')

# Gambar 5 — perubahan prediksi, 219 data CV: 7 benar->salah, 3 salah->benar,
# 1 salah->salah (kelas prediksi berubah); 208 prediksi tetap sama.
fig, ax = plt.subplots(figsize=(8.7,4.8))
labels=['Benar → Salah','Salah → Benar','Salah → Salah\n(kelas berubah)','Prediksi tetap']
values=[7,3,1,208]
bars=ax.bar(labels,values,color=['#C75D5D','#4E9A75','#B49A56','#9CA9BA'])
for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,v+2,str(v),ha='center')
ax.set_ylim(0,230);ax.set_ylabel('Jumlah sampel CV');ax.set_title('Dampak pembobotan aturan V4 pada 219 prediksi CV')
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.tight_layout();save('gambar5_perubahan_prediksi.png')

# Gambar 6 — bobot 27 aturan, dibaca langsung dari parameter model yang dilatih.
params = locate('parameter_model_v4.json')
if params:
    model=json.loads(params.read_text(encoding='utf-8'))
    weights=np.asarray(model['weights'],dtype=float)
    rules=np.asarray(model['rules'])
    if len(weights)!=27: raise ValueError('Jumlah bobot aturan bukan 27')
    classes=['CLL','FL','MCL']
    colors=['#7468AD','#D0A248','#C46D88']
    fig,ax=plt.subplots(figsize=(11,4.8))
    ax.bar(np.arange(1,28),weights,color=[colors[int(k)] for k in rules])
    ax.set_xticks(np.arange(1,28));ax.set_xlabel('Nomor aturan fuzzy (R1–R27)')
    ax.set_ylabel('Bobot aturan V4');ax.set_ylim(0,1.12)
    ax.set_title('Distribusi bobot 27 aturan Fuzzy Mamdani V4')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=label) for c,label in zip(colors,classes)],title='Konsekuen',ncol=3,frameon=False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();save('gambar6_bobot_aturan_v4.png')
else:
    print('LEWATI bobot aturan: parameter_model_v4.json belum ditemukan.')


# Gambar 7 — Confusion Matrix V3 dan V4

# ==========================================
# GAMBAR 7 — CONFUSION MATRIX V3 DAN V4
# Tanpa scikit-learn
# ==========================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

CSV_PREDIKSI = (
    ROOT
    / "hasil_glcm"
    / "analisis"
    / "fuzzy_mamdani_v4"
    / "prediksi_cv_v3_v4.csv"
)

OUTPUT = ROOT / "gambar_laporan"
OUTPUT.mkdir(parents=True, exist_ok=True)

KELAS = ["CLL", "FL", "MCL"]

if CSV_PREDIKSI.exists():

    df = pd.read_csv(CSV_PREDIKSI)

    kolom_wajib = [
        "aktual",
        "prediksi_v3",
        "prediksi_v4",
    ]

    for kolom in kolom_wajib:
        if kolom not in df.columns:
            raise ValueError(
                f"Kolom '{kolom}' tidak ditemukan."
            )

    # Samakan format label kelas
    for kolom in kolom_wajib:
        df[kolom] = (
            df[kolom]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        label_tidak_dikenal = (
            set(df[kolom]) - set(KELAS)
        )

        if label_tidak_dikenal:
            raise ValueError(
                f"Label tidak dikenal pada {kolom}: "
                f"{label_tidak_dikenal}"
            )

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8)
    )

    for ax, (nama_model, kolom_prediksi) in zip(
        axes,
        [
            ("V3 — Tanpa Bobot", "prediksi_v3"),
            ("V4 — Dengan Bobot", "prediksi_v4"),
        ],
    ):

        cm = pd.crosstab(
            pd.Categorical(
                df["aktual"],
                categories=KELAS,
            ),
            pd.Categorical(
                df[kolom_prediksi],
                categories=KELAS,
            ),
            dropna=False,
        ).to_numpy()

        ax.imshow(cm, cmap="Blues")

        ax.set_title(nama_model, fontweight="bold")
        ax.set_xlabel("Kelas Prediksi")
        ax.set_ylabel("Kelas Aktual")

        ax.set_xticks(range(len(KELAS)))
        ax.set_yticks(range(len(KELAS)))

        ax.set_xticklabels(KELAS)
        ax.set_yticklabels(KELAS)

        batas_warna = cm.max() / 2

        for i in range(len(KELAS)):
            for j in range(len(KELAS)):
                nilai = cm[i, j]

                ax.text(
                    j, i, str(nilai),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=(
                        "white"
                        if nilai > batas_warna
                        else "black"
                    ),
                )

        akurasi = np.trace(cm) / cm.sum()

        ax.text(
            0.5, -0.20,
            f"Accuracy CV: {akurasi:.2%}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
        )

    fig.suptitle(
        "Perbandingan Confusion Matrix "
        "Fuzzy Mamdani V3 dan V4",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])

    lokasi = (
        OUTPUT
        / "gambar7_confusion_matrix_v3_v4.png"
    )

    fig.savefig(
        lokasi,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Tersimpan: {lokasi}")

else:
    print(
        f"File prediksi CV tidak ditemukan: "
        f"{CSV_PREDIKSI}"
    )
