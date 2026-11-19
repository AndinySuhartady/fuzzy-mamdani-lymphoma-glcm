"""GUI demonstrasi akademik GLCM + Mamdani V3/V4 (parameter terlatih)."""
from pathlib import Path
import json
import sys
import tempfile
import hashlib
import itertools

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from ekstraksi_glcm import extract_features
from fuzzy_mamdani_v3 import LABELS, TERMS, membership, firing, predict
from fuzzy_mamdani_v4 import predict_v4

BASE = ROOT / 'hasil_glcm' / 'analisis'
SUPPORTED = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp']

st.set_page_config(page_title='GLCM + Fuzzy Mamdani | V3 vs V4', page_icon='🔬', layout='wide')
st.title('🔬 Klasifikasi Tekstur Histopatologi Limfoma')
st.caption('GLCM · Fuzzy Mamdani V3 (baseline) dan V4 (pembobotan aturan) · CLL / FL / MCL')
st.warning('Prototipe penelitian akademik. Bukan alat diagnosis medis. Luas keluaran bukan probabilitas atau tingkat keyakinan klinis.')

@st.cache_data(show_spinner=False)
def load_models():
    v3_dir = BASE / 'fuzzy_mamdani_v3'
    v4_dir = BASE / 'fuzzy_mamdani_v4'
    p3 = json.loads((v3_dir / 'parameter_model_terpilih.json').read_text(encoding='utf-8'))
    r3 = pd.read_csv(v3_dir / 'aturan_model_terpilih.csv')
    p4 = json.loads((v4_dir / 'parameter_model_v4.json').read_text(encoding='utf-8'))
    features = tuple(p3['fitur'])
    if features != tuple(p4['features']):
        raise ValueError('Urutan fitur V3 dan V4 berbeda; perbandingan tidak aman.')
    if p3['strategi'] != p4['strategy']:
        raise ValueError('Strategi pembentukan aturan V3 dan V4 berbeda.')
    if any(not np.allclose(p3['knots'][f], p4['knots'][f], rtol=0, atol=1e-12) for f in features):
        raise ValueError('Parameter keanggotaan V3 dan V4 berbeda.')
    n = 3 ** len(features)
    if len(r3) != n or len(p4['rules']) != n or len(p4['weights']) != n:
        raise ValueError('Jumlah aturan/bobot tidak sesuai dengan kombinasi fitur.')
    for f in features:
        if f not in r3.columns or f not in p3['knots']:
            raise ValueError(f'Kolom/parameter fitur tidak ditemukan: {f}')
    mapping = {label: i for i, label in enumerate(LABELS)}
    rules3 = np.array([mapping.get(str(v), -1) for v in r3['output']], dtype=int)
    rules4 = np.asarray(p4['rules'], dtype=int)
    weights = np.asarray(p4['weights'], dtype=float)
    if not np.array_equal(rules3, rules4):
        raise ValueError('Konsekuen aturan V3 dan V4 berbeda; jangan tampilkan sebagai perbandingan setara.')
    if np.any((rules4 < -1) | (rules4 >= len(LABELS))):
        raise ValueError('Kode kelas aturan tidak valid.')
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError('Bobot aturan harus berada dalam rentang [0, 1].')
    return features, p3['knots'], rules3, weights

try:
    feats, knots, rules, weights = load_models()
except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
    st.error(f'Gagal memuat model terlatih: {exc}')
    st.stop()

with st.sidebar:
    st.header('Konfigurasi')
    selection = st.radio('Tampilan hasil', ['Bandingkan V3 & V4', 'V3 saja', 'V4 saja'], index=0)
    st.write('**Fitur:** ' + ', '.join(feats))
    st.write(f'**Aturan:** {len(rules)}')
    st.write('**Inferensi:** MIN → (× bobot pada V4) → MAX')
    st.write('**Keputusan:** luas keluaran segitiga terpotong terbesar')
    st.caption('Model final dimuat dari berkas hasil pelatihan, tanpa melatih ulang saat unggah citra.')

upload = st.file_uploader('Unggah citra histopatologi', type=SUPPORTED)
if upload is None:
    st.info('Unggah citra untuk memulai. Ekstraksi mengikuti konfigurasi asli: grayscale, 256×256, 32 tingkat, GLCM jarak 1 dan 4 arah.')
    st.stop()

payload = upload.getvalue()
digest = hashlib.sha256(payload).hexdigest()
left, right = st.columns([1, 1.3], gap='large')
with left:
    st.subheader('Citra masukan')
    try:
        with Image.open(upload) as im:
            st.image(im.convert('RGB'), caption=upload.name, use_container_width=True)
    except Exception as exc:
        st.error(f'Citra tidak dapat dibaca: {exc}')
        st.stop()
with right:
    st.subheader('Ekstraksi dan inferensi')
    run = st.button('🔍 Jalankan klasifikasi', type='primary', use_container_width=True)
    if run:
        try:
            with st.spinner('Menghitung fitur GLCM dan inferensi kedua model…'):
                suffix = Path(upload.name).suffix.lower()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / ('input' + suffix)
                    path.write_bytes(payload)
                    values = extract_features(path)
                frame = pd.DataFrame([{f: float(values[f]) for f in feats}])
                pred3, areas3 = predict(frame, feats, knots, rules)
                pred4, areas4 = predict_v4(frame, feats, knots, rules, weights)
                acts = firing(frame, feats, knots)[0]
                st.session_state['analysis'] = dict(digest=digest, name=upload.name, values=values,
                    pred3=str(pred3[0]), pred4=str(pred4[0]), areas3=areas3[0].tolist(),
                    areas4=areas4[0].tolist(), firing=acts.tolist())
        except Exception as exc:
            st.error(f'Klasifikasi gagal: {exc}')
            st.stop()
    result = st.session_state.get('analysis')
    if result is None or result['digest'] != digest:
        st.info('Klik **Jalankan klasifikasi** untuk menghitung hasil pada citra ini.')
        st.stop()

show3 = selection != 'V4 saja'
show4 = selection != 'V3 saja'
cols = st.columns(2 if show3 and show4 else 1)
if show3:
    with cols[0]:
        st.metric('Prediksi V3 · tanpa bobot', result['pred3'])
if show4:
    with cols[-1]:
        st.metric('Prediksi V4 · berbobot', result['pred4'])
if show3 and show4:
    if result['pred3'] == result['pred4']:
        st.info('Kedua konfigurasi menghasilkan kelas yang sama untuk citra ini.')
    else:
        st.warning('Kedua konfigurasi menghasilkan prediksi berbeda untuk citra ini.')

st.subheader('1 · Fitur GLCM (rata-rata empat arah)')
feature_names = ['correlation', 'contrast', 'homogeneity', 'energy']
feature_df = pd.DataFrame({'Fitur': feature_names, 'Nilai': [result['values'][f] for f in feature_names]})
st.dataframe(feature_df.style.format({'Nilai':'{:.6f}'}), hide_index=True, use_container_width=True)

st.subheader('2 · Fuzzifikasi')
mu_rows = []
for f in feats:
    x = float(result['values'][f])
    mu = membership(np.array([x]), np.asarray(knots[f]))[0]
    mu_rows.append({'Fitur': f, 'Nilai masukan': x, 'Rendah': mu[0], 'Sedang': mu[1], 'Tinggi': mu[2]})
mu_df = pd.DataFrame(mu_rows)
st.dataframe(mu_df.style.format({'Nilai masukan':'{:.6f}', 'Rendah':'{:.6f}', 'Sedang':'{:.6f}', 'Tinggi':'{:.6f}'}), hide_index=True, use_container_width=True)
with st.expander('Parameter kuartil Q1, Q2, Q3'):
    st.dataframe(pd.DataFrame([{'Fitur': f, 'Q1': knots[f][0], 'Q2': knots[f][1], 'Q3': knots[f][2]} for f in feats]).style.format({'Q1':'{:.6f}', 'Q2':'{:.6f}', 'Q3':'{:.6f}'}), hide_index=True, use_container_width=True)

st.subheader('3 · Aktivasi aturan fuzzy')
raw = np.asarray(result['firing'], dtype=float)
weighted = raw * weights
term_combinations = list(itertools.product(TERMS, repeat=len(feats)))
rows = []
for i, terms in enumerate(term_combinations):
    rows.append({'Aturan':f'R{i+1}', **dict(zip(feats, terms)),
        'Konsekuen': LABELS[rules[i]] if rules[i] >= 0 else 'TIDAK_AKTIF',
        'Aktivasi MIN (V3)': raw[i], 'Bobot V4': weights[i], 'Aktivasi berbobot (V4)': weighted[i]})
rule_df = pd.DataFrame(rows)
active = rule_df[rule_df['Aktivasi MIN (V3)'] > 1e-12]
st.caption(f'{len(active)} dari {len(rule_df)} aturan memiliki aktivasi MIN > 0 pada citra ini.')
st.dataframe(active.style.format({'Aktivasi MIN (V3)':'{:.6f}', 'Bobot V4':'{:.6f}', 'Aktivasi berbobot (V4)':'{:.6f}'}), hide_index=True, use_container_width=True)
with st.expander('Tampilkan seluruh 27 aturan'):
    st.dataframe(rule_df.style.format({'Aktivasi MIN (V3)':'{:.6f}', 'Bobot V4':'{:.6f}', 'Aktivasi berbobot (V4)':'{:.6f}'}), hide_index=True, use_container_width=True)

st.subheader('4 · Agregasi MAX dan luas keluaran')
areas3 = np.asarray(result['areas3'], dtype=float)
areas4 = np.asarray(result['areas4'], dtype=float)
heights3 = np.array([raw[rules == k].max() if np.any(rules == k) else 0. for k in range(len(LABELS))])
heights4 = np.array([weighted[rules == k].max() if np.any(rules == k) else 0. for k in range(len(LABELS))])
area_df = pd.DataFrame({'Kelas': LABELS, 'Tinggi MAX V3': heights3, 'Luas V3': areas3,
    'Tinggi MAX V4': heights4, 'Luas V4': areas4})
st.dataframe(area_df.style.format({c:'{:.6f}' for c in area_df.columns if c != 'Kelas'}), hide_index=True, use_container_width=True)
chart = area_df.set_index('Kelas')[[c for c in ['Luas V3', 'Luas V4'] if (c == 'Luas V3' and show3) or (c == 'Luas V4' and show4)]]
st.bar_chart(chart)
st.caption('A(h) = h − h²/2 untuk segitiga keluaran (0; 0,5; 1). Nilai luas bukan probabilitas. Model tidak menggunakan centroid global untuk memilih kelas.')

with st.expander('5 · Langkah perhitungan yang dapat ditelusuri'):
    st.markdown('**Aktivasi aturan:** αᵣ = MIN(μ₁, μ₂, μ₃). **V4:** α′ᵣ = αᵣ × wᵣ.')
    st.markdown('**Agregasi:** hₖ = MAX(aktivasi aturan dengan konsekuen kelas k). **Luas:** Aₖ = hₖ − hₖ²/2.')
    for k, label in enumerate(LABELS):
        if show3:
            st.write(f'V3 {label}: A = {heights3[k]:.6f} − ({heights3[k]:.6f})²/2 = {areas3[k]:.6f}')
        if show4:
            st.write(f'V4 {label}: A = {heights4[k]:.6f} − ({heights4[k]:.6f})²/2 = {areas4[k]:.6f}')
    st.caption('Bobot V4 dihitung saat pelatihan dari dukungan kelas seimbang; bukan dihitung ulang dari citra unggahan.')

st.caption('Hasil inferensi ini memakai parameter model final yang dilatih pada data latih. Hasil satu citra bukan ukuran akurasi model; penilaian performa memerlukan validasi terpisah.')
