# GUI GLCM + Fuzzy Mamdani V3/V4

GUI akademik Streamlit untuk mengunggah citra histopatologi dan membandingkan inferensi **V3 (baseline)** dengan **V4 (confidence-weighted)** pada model yang telah dilatih.

## Menjalankan

Buka terminal di folder ini:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Yang ditampilkan

- Pratinjau citra dan ekstraksi GLCM (256×256, grayscale, 32 level, jarak 1, empat sudut).
- Nilai empat fitur GLCM, serta fuzzifikasi tiga fitur terpilih.
- Aktivasi 27 aturan: MIN V3, bobot V4, dan aktivasi berbobot.
- Agregasi MAX, luas keluaran, prediksi V3 dan V4, grafik perbandingan.
- Pemilihan tampilan V3, V4, atau keduanya.

## Struktur berkas

- `app.py`: antarmuka Streamlit.
- `src/ekstraksi_glcm.py`: ekstraksi fitur citra (asli).
- `src/fuzzy_mamdani_v3.py`: inferensi V3 (asli).
- `src/fuzzy_mamdani_v4.py`: inferensi V4 (asli).
- `hasil_glcm/analisis/fuzzy_mamdani_v3/`: parameter dan aturan model V3.
- `hasil_glcm/analisis/fuzzy_mamdani_v4/parameter_model_v4.json`: parameter, konsekuen, dan bobot V4.

GUI **tidak** memerlukan dataset atau CSV data latih untuk prediksi. Model tidak dilatih ulang. Untuk menilai kinerja, gunakan hasil validasi penelitian, bukan citra unggahan tunggal.

**Peringatan:** prototipe akademik, bukan perangkat diagnosis klinis. Luas agregasi bukan probabilitas.
