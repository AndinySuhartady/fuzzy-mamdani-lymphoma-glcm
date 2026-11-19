"""Perbandingan leave-one-prefix-group-out V1 dan V2 tanpa data uji.
Simpan di folder src, lalu jalankan dari akar proyek dengan Python.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import fuzzy_mamdani as v1
import fuzzy_mamdani_v2 as v2

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'hasil_glcm' / 'analisis'
OUT = BASE / 'perbandingan_validasi'
LABELS = ('CLL', 'FL', 'MCL')


def metrics(df, model):
    actual = df['aktual'].to_numpy()
    pred = df['prediksi'].to_numpy()
    accuracy = float(np.mean(actual == pred))
    precision, recall, f1 = [], [], []
    for cls in LABELS:
        tp = int(np.sum((actual == cls) & (pred == cls)))
        fp = int(np.sum((actual != cls) & (pred == cls)))
        fn = int(np.sum((actual == cls) & (pred != cls)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        precision.append(p); recall.append(r); f1.append(f)
    return {'model': model, 'accuracy': accuracy, 'macro_precision': float(np.mean(precision)),
            'macro_recall': float(np.mean(recall)), 'macro_f1': float(np.mean(f1)),
            **{f'recall_{c}': r for c, r in zip(LABELS, recall)}}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE / 'data_latih.csv')
    required = {'nama_file', 'kelas', 'group_id', 'correlation', 'contrast'}
    if not required.issubset(df.columns):
        raise ValueError('Kolom hilang: ' + str(required - set(df.columns)))
    if df[list(v1.FEATURES)].isna().any().any():
        raise ValueError('Ada nilai fitur kosong')
    if not set(df['kelas']).issubset(LABELS):
        raise ValueError('Label kelas tidak dikenal')
    # Kelompok berdasarkan group_id saja, bukan (kelas, group_id), agar prefix
    # yang kebetulan sama pada dua kelas tidak terpecah ke train dan validasi.
    groups = sorted(df['group_id'].unique())
    print(f'Data latih: {len(df)} citra, {len(groups)} kelompok')
    records = []
    shape_v1 = v1.output_sets()
    for fold, gid in enumerate(groups, 1):
        held = df[df['group_id'] == gid]
        fit_df = df[df['group_id'] != gid]
        if set(fit_df['kelas']) != set(LABELS):
            raise ValueError(f'Fold {gid}: ada kelas yang hilang dari data pelatihan')
        knots_v1 = {f: fit_df[f].quantile([.25, .5, .75]).tolist() for f in v1.FEATURES}
        rules_v1 = v1.fit_rules(fit_df, knots_v1)
        knots_v2, rules_v2 = v2.fit(fit_df)
        for _, row in held.iterrows():
            pred1, _, _ = v1.predict_one(row, knots_v1, rules_v1, shape_v1)
            pred2, _, _, _ = v2.predict_one(row, knots_v2, rules_v2)
            for name, pred in [('V1', pred1), ('V2', pred2)]:
                records.append({'fold': fold, 'group_id': gid, 'nama_file': row['nama_file'],
                                'aktual': row['kelas'], 'model': name, 'prediksi': pred})
    all_predictions = pd.DataFrame(records)
    all_predictions.to_csv(OUT / 'prediksi_cv_v1_v2.csv', index=False)
    summary = []
    for name in ['V1', 'V2']:
        d = all_predictions[all_predictions['model'] == name]
        summary.append(metrics(d, name))
        cm = pd.crosstab(pd.Categorical(d['aktual'], categories=LABELS),
                         pd.Categorical(d['prediksi'], categories=[*LABELS, 'AMBIGU', 'TIDAK_TERKLASIFIKASI']),
                         rownames=['Aktual'], colnames=['Prediksi'], dropna=False)
        cm.to_csv(OUT / f'confusion_matrix_cv_{name.lower()}.csv')
        print(f'\n{name}: akurasi={summary[-1]["accuracy"]:.2%}; macro-F1={summary[-1]["macro_f1"]:.2%}')
        print(cm.to_string())
    pd.DataFrame(summary).to_csv(OUT / 'ringkasan_perbandingan.csv', index=False)
    group_audit = (all_predictions.assign(benar=lambda d: d.aktual == d.prediksi)
                   .groupby(['model', 'group_id'], as_index=False)
                   .agg(jumlah=('benar','size'), benar=('benar','sum'), akurasi=('benar','mean')))
    group_audit.to_csv(OUT / 'audit_per_kelompok.csv', index=False)
    print('\nRingkasan:')
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nFile tersimpan di:', OUT)
    print('Data uji tidak dibaca. Pengelompokan berdasarkan prefix nama file, bukan ID pasien terverifikasi.')

if __name__ == '__main__':
    run()
