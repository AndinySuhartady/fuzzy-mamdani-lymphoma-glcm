# Fuzzy Mamdani for Lymphoma Histopathology Classification Using GLCM

**Evaluasi Pembobotan Aturan Fuzzy Mamdani V3 dan V4 pada Klasifikasi Citra Histopatologi Limfoma CLL, FL, dan MCL**

Proyek penelitian eksploratif untuk mata kuliah **Komputasi Lunak**, Program Studi Informatika, Universitas Diponegoro (2026).

## Deskripsi Proyek

Penelitian ini mengimplementasikan metode **Gray-Level Co-occurrence Matrix (GLCM)** untuk ekstraksi fitur tekstur citra histopatologi dan **Fuzzy Mamdani** untuk klasifikasi tiga jenis limfoma:

- Chronic Lymphocytic Leukemia (CLL)
- Follicular Lymphoma (FL)
- Mantle Cell Lymphoma (MCL)

Penelitian membandingkan dua konfigurasi model:

- **V3 (Baseline):** Fuzzy Mamdani tanpa pembobotan aturan.
- **V4 (Pengembangan):** Fuzzy Mamdani dengan bobot aturan berdasarkan dominasi dukungan kelas pada data latih.

## Dataset

Dataset penelitian terdiri atas **300 citra histopatologi**, dengan pembagian:

| Kelas | Data Latih | Data Uji | Total |
|---|---:|---:|---:|
| CLL | 77 | 23 | 100 |
| FL | 65 | 35 | 100 |
| MCL | 77 | 23 | 100 |
| **Total** | **219** | **81** | **300** |

Pemisahan dilakukan berdasarkan `group_id` yang diturunkan dari prefiks nama berkas. Prefiks tersebut belum diverifikasi sebagai identitas pasien.

## Metodologi

1. Persiapan dataset citra histopatologi.
2. Ekstraksi fitur tekstur GLCM.
3. Pemilihan fitur correlation, contrast, dan homogeneity.
4. Pembentukan fungsi keanggotaan fuzzy rendah, sedang, dan tinggi.
5. Pembentukan 27 aturan fuzzy.
6. Implementasi model V3 dan V4.
7. Evaluasi menggunakan validasi silang berbasis kelompok.
8. Perbandingan accuracy, macro-F1, macro-recall, dan confusion matrix.

## Hasil Evaluasi

Hasil validasi silang pada 219 sampel data latih:

| Metrik | V3 | V4 |
|---|---:|---:|
| Accuracy | 38,81% | 36,99% |
| Macro-F1 | 35,69% | 33,31% |
| Macro-recall | 38,31% | 36,50% |

Pembobotan aturan pada V4 belum meningkatkan performa dibandingkan V3. Hasil ini menunjukkan bahwa penambahan bobot berdasarkan dominasi kelas tidak selalu memperbaiki kemampuan klasifikasi.

## Struktur Repository

```text
src/                  # Implementasi GLCM dan Fuzzy Mamdani
diagram/              # Diagram sistem dan skrip visualisasi
gambar_laporan/       # Grafik hasil penelitian
hasil_glcm/           # Hasil ekstraksi dan evaluasi
streamlit_mamdani/    # Aplikasi GUI Streamlit
```

## GUI Streamlit

Proyek juga mencakup prototipe antarmuka berbasis Streamlit untuk mendemonstrasikan proses klasifikasi menggunakan model fuzzy.

Petunjuk dan dependensi aplikasi tersedia di direktori `streamlit_mamdani/`.

## Dokumentasi GUI

Aplikasi klasifikasi citra histopatologi limfoma dikembangkan menggunakan Python dan Streamlit.

Video demonstrasi memperlihatkan:
1. Tampilan utama aplikasi.
2. Proses unggah citra histopatologi.
3. Proses klasifikasi menggunakan fitur GLCM dan Fuzzy Mamdani.
4. Hasil prediksi kelas CLL, FL, atau MCL.

**Video Demonstrasi:** [Tonton melalui Google Drive]https://drive.google.com/drive/folders/1CUlLL4nfroAUC7Zj73jzblvgZ-QrJkb7?usp=sharing


**Catatan:** Aplikasi merupakan prototipe penelitian akademik dan tidak ditujukan untuk diagnosis klinis.

## Keterbatasan

- Performa klasifikasi masih relatif rendah.
- Jumlah dan keragaman dataset terbatas.
- Identitas pasien belum diverifikasi.
- Dataset uji pernah diperiksa selama eksplorasi sehingga tidak dapat dianggap sebagai holdout yang sepenuhnya independen.
- Sistem merupakan prototipe akademik dan bukan perangkat diagnosis medis.

## Pengembang

**Andiny Khaerany Suhartady**

**Jessica Laurencia Panjaitan**

Program Studi Informatika

Universitas Diponegoro

2026

## Tujuan Penggunaan

Repository ini disediakan untuk dokumentasi dan pembelajaran akademik mengenai ekstraksi fitur tekstur serta penerapan Fuzzy Mamdani pada klasifikasi citra histopatologi.
