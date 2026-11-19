"""Evaluasi satu kali model Mamdani V3 yang sudah dibekukan.
Jalankan dari akar proyek: py src/evaluasi_final_v3.py
Tidak melatih ulang, mengubah parameter, atau memilih model berdasarkan data uji.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fuzzy_mamdani_v3 import LABELS, TERMS, predict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'hasil_glcm' / 'analisis'
MODEL = BASE / 'fuzzy_mamdani_v3'
OUT = MODEL / 'evaluasi_final'


def load_frozen_model():
    params = json.loads((MODEL / 'parameter_model_terpilih.json').read_text(encoding='utf-8'))
    feats = tuple(params['fitur'])
    if not feats or len(feats) != len(set(feats)):
        raise ValueError('Daftar fitur model tidak valid')
    knots = {f: np.asarray(params['knots'][f], dtype=float) for f in feats}
    for f, q in knots.items():
        if q.shape != (3,) or not np.isfinite(q).all() or not (q[0] < q[1] < q[2]):
            raise ValueError(f'Kuartil tidak valid untuk {f}')
    if params.get('output_shape') != 'segitiga simetris 0,0.5,1':
        raise ValueError('Bentuk output tidak cocok dengan fungsi prediksi V3')
    rules_df = pd.read_csv(MODEL / 'aturan_model_terpilih.csv')
    expected = list(itertools.product(TERMS, repeat=len(feats)))
    if len(rules_df) != len(expected):
        raise ValueError(f'Diharapkan {len(expected)} aturan, didapat {len(rules_df)}')
    if not set([*feats, 'output']).issubset(rules_df.columns):
        raise ValueError('Kolom aturan tidak lengkap')
    actual = list(rules_df[list(feats)].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Urutan aturan tidak cocok dengan urutan produk kartesius V3')
    mapping = {c: i for i, c in enumerate(LABELS)}
    mapping['TIDAK_AKTIF'] = -1
    if not rules_df['output'].isin(mapping).all():
        raise ValueError('Konsekuen aturan tidak dikenal')
    consequents = rules_df['output'].map(mapping).to_numpy(dtype=int)
    return feats, knots, consequents


def main():
    feats, knots, consequents = load_frozen_model()
    test = pd.read_csv(BASE / 'data_uji.csv')
    needed = {'nama_file', 'kelas', *feats}
    if not needed.issubset(test.columns):
        raise ValueError(f'Kolom data uji hilang: {sorted(needed - set(test.columns))}')
    if not test['kelas'].isin(LABELS).all():
        raise ValueError('Ada label kelas data uji yang tidak dikenal')
    x = test[list(feats)].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Ada nilai fitur uji kosong / tidak hingga')
    preds, areas = predict(test, feats, knots, consequents)
    y = test['kelas'].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    results = test[['nama_file', 'kelas']].rename(columns={'kelas': 'aktual'}).copy()
    if 'group_id' in test:
        results['group_id'] = test['group_id']
    for f in feats:
        results[f] = test[f].to_numpy()
    results['prediksi'] = preds
    for k, label in enumerate(LABELS):
        results[f'luas_{label}'] = areas[:, k]
    results['benar'] = y == preds
    results.to_csv(OUT / 'prediksi_uji_final_v3.csv', index=False)
    columns = [*LABELS, 'AMBIGU', 'TIDAK_TERKLASIFIKASI']
    cm = pd.crosstab(pd.Categorical(y, categories=LABELS),
                     pd.Categorical(preds, categories=columns),
                     rownames=['Aktual'], colnames=['Prediksi'], dropna=False)
    cm.to_csv(OUT / 'confusion_matrix_uji_final_v3.csv')
    rows = []
    for label in LABELS:
        tp = int(((y == label) & (preds == label)).sum())
        fp = int(((y != label) & (preds == label)).sum())
        fn = int(((y == label) & (preds != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({'kelas': label, 'support': int((y == label).sum()),
                     'precision': precision, 'recall': recall, 'f1': f1})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(OUT / 'metrik_per_kelas_v3.csv', index=False)
    summary = {'n_uji': len(test), 'benar': int((y == preds).sum()),
               'accuracy': float((y == preds).mean()),
               'macro_precision': float(per_class['precision'].mean()),
               'macro_recall': float(per_class['recall'].mean()),
               'macro_f1': float(per_class['f1'].mean()),
               'jumlah_ambigu': int((preds == 'AMBIGU').sum()),
               'jumlah_tidak_terklasifikasi': int((preds == 'TIDAK_TERKLASIFIKASI').sum()),
               'fitur': list(feats), 'jumlah_aturan': len(consequents),
               'catatan': 'Model dibekukan sebelum evaluasi; hasil hanya eksperimen tekstur, bukan diagnosis.'}
    (OUT / 'ringkasan_evaluasi_final_v3.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('=== EVALUASI FINAL MAMDANI V3 (MODEL DIBEKUKAN) ===')
    print('Fitur:', ', '.join(feats), '| aturan:', len(consequents))
    print(f"Data uji: {summary['n_uji']}, benar: {summary['benar']}")
    print(f"Accuracy: {summary['accuracy']:.2%} | Macro-F1: {summary['macro_f1']:.2%}")
    print('\nConfusion matrix:')
    print(cm.to_string())
    print('\nMetrik per kelas:')
    print(per_class.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nFile tersimpan di:', OUT)
    print('PERINGATAN: Jangan mengubah model berdasarkan hasil data uji ini.')
    print('Bukan alat diagnosis medis.')

if __name__ == '__main__':
    main()
