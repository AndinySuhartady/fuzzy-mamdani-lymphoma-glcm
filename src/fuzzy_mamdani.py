"""Prototipe klasifikasi tekstur CLL/FL/MCL dengan Fuzzy Mamdani.
Jalankan dari akar proyek: py src/fuzzy_mamdani.py
Hanya untuk penelitian citra, bukan diagnosis klinis.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'hasil_glcm' / 'analisis'
OUT = BASE / 'fuzzy_mamdani'
FEATURES = ('correlation', 'contrast')
LABELS = ('CLL', 'FL', 'MCL')
TERMS = ('rendah', 'sedang', 'tinggi')
# Kode numerik hanyalah pengkodean kelas nominal, BUKAN urutan klinis.
CENTERS = {'CLL': 20.0, 'FL': 50.0, 'MCL': 80.0}
GRID = np.linspace(0, 100, 1001)


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = (x-a)/(b-a) if b > a else np.zeros_like(x)
    right = (c-x)/(c-b) if c > b else np.zeros_like(x)
    return np.maximum(0, np.minimum(left, right))


def memberships(x, knots):
    q1, q2, q3 = knots
    # Bahu kiri, segitiga tengah, bahu kanan; tetap terdefinisi di luar rentang data.
    low = np.clip((q2-x)/(q2-q1), 0, 1)
    mid = float(trimf(x, q1, q2, q3))
    high = np.clip((x-q2)/(q3-q2), 0, 1)
    return np.array([low, mid, high], dtype=float)


def output_sets():
    # Himpunan output segitiga terpisah, sehingga centroid lebih mudah ditafsirkan.
    return {name: trimf(GRID, center-12, center, center+12)
            for name, center in CENTERS.items()}


def fit_rules(train, knots):
    # Derajat dukungan kelas untuk tiap kombinasi 3x3, bukan aturan buatan.
    votes = np.zeros((3, 3, 3), dtype=float)
    for _, row in train.iterrows():
        m1 = memberships(float(row[FEATURES[0]]), knots[FEATURES[0]])
        m2 = memberships(float(row[FEATURES[1]]), knots[FEATURES[1]])
        cls = LABELS.index(row['kelas'])
        for i in range(3):
            for j in range(3):
                votes[i, j, cls] += min(m1[i], m2[j])
    prior = train['kelas'].value_counts().reindex(LABELS, fill_value=0).to_numpy()
    rules = []
    for i in range(3):
        for j in range(3):
            v = votes[i, j]
            winner = int(np.argmax(v)) if v.sum() > 0 else int(np.argmax(prior))
            rules.append({'correlation': TERMS[i], 'contrast': TERMS[j],
                          'output': LABELS[winner], 'support': round(float(v[winner]), 5),
                          'votes_CLL': round(float(v[0]), 5),
                          'votes_FL': round(float(v[1]), 5),
                          'votes_MCL': round(float(v[2]), 5)})
    return rules


def predict_one(row, knots, rules, shapes):
    a = memberships(float(row[FEATURES[0]]), knots[FEATURES[0]])
    b = memberships(float(row[FEATURES[1]]), knots[FEATURES[1]])
    aggregated = np.zeros_like(GRID)
    fired = []
    for i in range(3):
        for j in range(3):
            rule = rules[i*3+j]
            alpha = min(a[i], b[j])  # AND = MIN
            if alpha > 0:
                aggregated = np.maximum(aggregated,
                                        np.minimum(alpha, shapes[rule['output']]))  # MAX
                fired.append((f"{TERMS[i]} & {TERMS[j]} -> {rule['output']}", round(float(alpha), 5)))
    mass = np.trapezoid(aggregated, GRID)
    if mass <= 0:
        return 'TIDAK_TERKLASIFIKASI', np.nan, fired
    crisp = float(np.trapezoid(GRID * aggregated, GRID) / mass)  # centroid
    # Pemetaan centroid ke kode kelas terdekat adalah keputusan rekayasa,
    # bukan sifat ordinal CLL/FL/MCL.
    pred = min(LABELS, key=lambda name: abs(crisp - CENTERS[name]))
    return pred, crisp, fired


def evaluate(df, name, knots, rules, shapes):
    rows = []
    for _, row in df.iterrows():
        pred, crisp, fired = predict_one(row, knots, rules, shapes)
        rows.append({'nama_file': row['nama_file'], 'aktual': row['kelas'],
                     'prediksi': pred, 'centroid': crisp,
                     'aturan_aktif': '; '.join(f'{r} ({a})' for r, a in fired)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f'prediksi_{name}.csv', index=False)
    matrix = pd.crosstab(pd.Categorical(result['aktual'], categories=LABELS),
                         pd.Categorical(result['prediksi'], categories=LABELS),
                         rownames=['Aktual'], colnames=['Prediksi'], dropna=False)
    matrix.to_csv(OUT / f'confusion_matrix_{name}.csv')
    accuracy = (result['aktual'] == result['prediksi']).mean()
    print(f'\n{name.upper()}: n={len(df)}, akurasi={accuracy:.2%}')
    print(matrix.to_string())
    return accuracy


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = BASE / 'data_latih.csv'
    if not train_path.exists():
        raise FileNotFoundError(f'Tidak ditemukan: {train_path}')
    train = pd.read_csv(train_path)
    for f in FEATURES:
        if f not in train.columns or train[f].isna().any():
            raise ValueError(f'Fitur hilang/NaN: {f}')
    if not set(train['kelas']).issubset(set(LABELS)):
        raise ValueError('Ada label kelas tidak dikenal')
    knots = {f: train[f].quantile([.25, .5, .75]).to_numpy().tolist() for f in FEATURES}
    rules = fit_rules(train, knots)
    pd.DataFrame(rules).to_csv(OUT / 'aturan_fuzzy.csv', index=False)
    (OUT / 'parameter_keanggotaan.json').write_text(json.dumps(knots, indent=2), encoding='utf-8')
    print('Titik Q1, median, Q3 (dihitung dari data latih):', knots)
    print('\n9 aturan hasil pembobotan data latih:')
    print(pd.DataFrame(rules)[['correlation','contrast','output','support']].to_string(index=False))
    shapes = output_sets()
    evaluate(train, 'latih', knots, rules, shapes)
    test_path = BASE / 'data_uji.csv'
    if test_path.exists():
        test = pd.read_csv(test_path)
        if 'group_id' in train and 'group_id' in test:
            overlap = set(zip(train['kelas'], train['group_id'])) & set(zip(test['kelas'], test['group_id']))
            if overlap:
                raise ValueError(f'Kebocoran kelompok data: {overlap}')
        evaluate(test, 'uji', knots, rules, shapes)
    else:
        print('\nData uji belum tersedia; evaluasi uji dilewati.')
    print('\nHasil:', OUT)
    print('PERINGATAN: hasil hanya eksplorasi tekstur, bukan diagnosis medis.')


if __name__ == '__main__':
    main()
