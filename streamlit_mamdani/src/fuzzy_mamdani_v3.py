"""Eksperimen Mamdani V3: seleksi fitur dan bobot aturan via leave-one-prefix-group-out.

Jalankan dari akar proyek: py src/fuzzy_mamdani_v3.py
Hanya membaca hasil_glcm/analisis/data_latih.csv; tidak membaca data_uji.csv.
Output kanal kelas: segitiga simetris di [0,1]; MIN implikasi, MAX agregasi,
centroid tiap kanal (0.5 bila aktif), keputusan berdasar luas agregasi.
Untuk data klinis penelitian eksploratif, bukan diagnosis.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'hasil_glcm' / 'analisis'
OUT = BASE / 'fuzzy_mamdani_v3'
LABELS = ('CLL', 'FL', 'MCL')
TERMS = ('rendah', 'sedang', 'tinggi')
FEATURES_ALL = ('correlation', 'contrast', 'homogeneity', 'energy')
# 6 kombinasi 2 fitur + 4 kombinasi 3 fitur
CANDIDATES = [x for n in (2, 3) for x in itertools.combinations(FEATURES_ALL, n)]
STRATEGIES = ('raw', 'class_balanced')


def membership(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Array (N,3): bahu kiri, segitiga tengah, bahu kanan."""
    a, b, c = map(float, q)
    if not a < b < c:
        # Kuartil berimpit: pisahkan dengan epsilon relatif agar stabil.
        spread = max(abs(a), abs(b), abs(c), 1.0) * 1e-9
        b = max(b, a + spread)
        c = max(c, b + spread)
    low = np.clip((b-x)/(b-a), 0, 1)
    mid = np.clip(np.minimum((x-a)/(b-a), (c-x)/(c-b)), 0, 1)
    high = np.clip((x-b)/(c-b), 0, 1)
    return np.column_stack([low, mid, high])


def firing(df: pd.DataFrame, feats: tuple[str, ...], knots: dict) -> np.ndarray:
    """MIN antarfungsi keanggotaan, (N, 3**d)."""
    mu = [membership(df[f].to_numpy(dtype=float), knots[f]) for f in feats]
    indices = list(itertools.product(range(3), repeat=len(feats)))
    return np.column_stack([np.minimum.reduce([mu[j][:, idx[j]] for j in range(len(feats))]) for idx in indices])


def fit(df: pd.DataFrame, feats: tuple[str, ...], strategy: str):
    knots = {f: df[f].quantile([.25,.5,.75]).to_numpy(dtype=float) for f in feats}
    acts = firing(df, feats, knots)
    y = df['kelas'].to_numpy()
    counts = np.array([(y == c).sum() for c in LABELS], dtype=float)
    if np.any(counts == 0):
        raise ValueError('Ada kelas yang tidak ada di data pelatihan fold')
    votes = np.column_stack([acts[y == c].sum(axis=0) for c in LABELS])
    if strategy == 'class_balanced':
        votes = votes / counts[None, :]
    # aturan dengan dukungan nol tidak diaktifkan; tidak membuat konsekuen palsu
    consequents = np.where(votes.sum(axis=1) > 1e-12, votes.argmax(axis=1), -1)
    return knots, consequents, votes


def predict(df: pd.DataFrame, feats, knots, consequents):
    acts = firing(df, feats, knots)
    # MAX gabungan rule sekelas; karena bentuk output sama, alpha terbesar cukup.
    heights = np.column_stack([
        acts[:, consequents == k].max(axis=1) if np.any(consequents == k)
        else np.zeros(len(df)) for k in range(3)
    ])
    # Integral min(alpha, triangle(0,0.5,1)) dz = alpha - alpha^2/2.
    areas = heights - heights**2/2
    winners = areas.argmax(axis=1)
    best = areas.max(axis=1)
    ties = (np.abs(areas - best[:,None]) < 1e-10).sum(axis=1) > 1
    preds = np.array(LABELS, dtype=object)[winners]
    preds[best <= 1e-12] = 'TIDAK_TERKLASIFIKASI'
    preds[(best > 1e-12) & ties] = 'AMBIGU'
    return preds, areas


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    p, r, f = [], [], []
    for c in LABELS:
        tp = int(((y==c)&(pred==c)).sum())
        fp = int(((y!=c)&(pred==c)).sum())
        fn = int(((y==c)&(pred!=c)).sum())
        pc = tp/(tp+fp) if tp+fp else 0.0
        rc = tp/(tp+fn) if tp+fn else 0.0
        fc = 2*pc*rc/(pc+rc) if pc+rc else 0.0
        p.append(pc); r.append(rc); f.append(fc)
    return {'accuracy':float((y==pred).mean()), 'macro_f1':float(np.mean(f)),
            'macro_recall':float(np.mean(r)),
            **{f'recall_{c}':r[i] for i,c in enumerate(LABELS)}}


def cross_validate(df, feats, strategy, groups):
    predictions = np.empty(len(df), dtype=object)
    for gid in groups:
        mask = df['group_id'].to_numpy() == gid
        fit_df = df.loc[~mask]
        knots, rules, _ = fit(fit_df, feats, strategy)
        predictions[mask], _ = predict(df.loc[mask], feats, knots, rules)
    return predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(BASE / 'data_latih.csv')
    required = {'nama_file','kelas','group_id', *FEATURES_ALL}
    if not required.issubset(train.columns):
        raise ValueError(f'Kolom hilang: {sorted(required-set(train.columns))}')
    if train[list(FEATURES_ALL)].isna().any().any() or not np.isfinite(train[list(FEATURES_ALL)].to_numpy(dtype=float)).all():
        raise ValueError('Nilai fitur kosong atau tidak hingga')
    if set(train['kelas']) != set(LABELS):
        raise ValueError('Label kelas tidak sesuai')
    if train['group_id'].isna().any() or train.groupby('group_id')['kelas'].nunique().max() != 1:
        raise ValueError('group_id kosong atau satu group_id memiliki lebih dari satu kelas')
    groups = sorted(train['group_id'].unique())
    print(f'Data latih: {len(train)} citra, {len(groups)} kelompok. Data uji TIDAK dibaca.')
    results = []
    predictions_by_id = {}
    for feats in CANDIDATES:
        for strategy in STRATEGIES:
            preds = cross_validate(train, feats, strategy, groups)
            scores = metrics(train['kelas'].to_numpy(), preds)
            key = '+'.join(feats)+'__'+strategy
            predictions_by_id[key] = preds
            results.append({'id':key,'fitur':'+'.join(feats),'jumlah_fitur':len(feats),
                            'jumlah_aturan':3**len(feats),'strategi':strategy,**scores})
            print(f'{key:62s} accuracy={scores["accuracy"]:.2%} macro-F1={scores["macro_f1"]:.2%}')
    summary = pd.DataFrame(results).sort_values(['macro_f1','accuracy'], ascending=False).reset_index(drop=True)
    summary.to_csv(OUT/'peringkat_validasi_v3.csv', index=False)
    best = summary.iloc[0]
    best_id = best['id']
    feats = tuple(best['fitur'].split('+'))
    strategy = best['strategi']
    cv = train[['nama_file','kelas','group_id']].rename(columns={'kelas':'aktual'}).copy()
    cv['prediksi'] = predictions_by_id[best_id]
    cv.to_csv(OUT/'prediksi_cv_terbaik.csv', index=False)
    matrix = pd.crosstab(pd.Categorical(cv['aktual'],categories=LABELS),
                         pd.Categorical(cv['prediksi'],categories=[*LABELS,'AMBIGU','TIDAK_TERKLASIFIKASI']),
                         rownames=['Aktual'],colnames=['Prediksi'],dropna=False)
    matrix.to_csv(OUT/'confusion_matrix_cv_terbaik.csv')
    knots, rules, votes = fit(train, feats, strategy)
    rule_rows = []
    for i, terms in enumerate(itertools.product(TERMS, repeat=len(feats))):
        record = {f:term for f,term in zip(feats,terms)}
        record.update({'output':LABELS[rules[i]] if rules[i]>=0 else 'TIDAK_AKTIF',
                       **{f'vote_{c}':float(votes[i,k]) for k,c in enumerate(LABELS)}})
        rule_rows.append(record)
    pd.DataFrame(rule_rows).to_csv(OUT/'aturan_model_terpilih.csv',index=False)
    (OUT/'parameter_model_terpilih.json').write_text(json.dumps({
        'fitur':feats,'strategi':strategy,'knots':{f:knots[f].tolist() for f in feats},
        'output_shape':'segitiga simetris 0,0.5,1',
        'decision':'luas agregasi MAX terbesar; centroid kanal 0.5 bila aktif',
        'validation':'leave-one-prefix-group-out; seluruh parameter dipelajari ulang per fold'
    },indent=2),encoding='utf-8')
    print('\n=== MODEL TERBAIK MENURUT MACRO-F1 CV ===')
    print(summary[['fitur','strategi','accuracy','macro_f1','recall_CLL','recall_FL','recall_MCL']].head(10).to_string(index=False))
    print('\nTERPILIH:',best_id, f'| accuracy={best["accuracy"]:.2%}, macro-F1={best["macro_f1"]:.2%}')
    print(matrix.to_string())
    print('\nKeluaran:',OUT)
    print('PERINGATAN: hasil CV dipakai memilih model, sehingga skor terbaik bersifat optimistis.')
    print('81 data uji tidak digunakan. Pengelompokan prefix bukan ID pasien terverifikasi.')
    print('Bukan alat diagnosis medis.')

if __name__ == '__main__':
    main()
