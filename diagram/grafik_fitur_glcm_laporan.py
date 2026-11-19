"""Visualisasi fitur GLCM asli dan fungsi keanggotaan model V4.
Jalankan dari folder proyek: python diagram/grafik_fitur_glcm_laporan.py
Paket: python -m pip install pandas numpy matplotlib pillow scikit-image
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'diagram' else Path(__file__).resolve().parent
OUT = ROOT / 'gambar_laporan'
OUT.mkdir(parents=True, exist_ok=True)
ANALISIS = ROOT / 'hasil_glcm' / 'analisis'
CSV = ANALISIS / 'data_latih.csv'
PARAM = ANALISIS / 'fuzzy_mamdani_v4' / 'parameter_model_v4.json'
for path in (CSV, PARAM):
    if not path.exists():
        raise FileNotFoundError(f'File penelitian tidak ditemukan: {path}')
df = pd.read_csv(CSV)
params = json.loads(PARAM.read_text(encoding='utf-8'))
features = ['correlation', 'contrast', 'homogeneity']
classes = ['CLL', 'FL', 'MCL']
for col in ['kelas', 'nama_file', *features]:
    if col not in df:
        raise ValueError(f'Kolom CSV tidak ditemukan: {col}')
colors = {'CLL':'#5366ae', 'FL':'#c18b32', 'MCL':'#b35d80'}

# Fuzzy shoulder kiri/kanan dan segitiga tengah dari titik kuartil.
def membership(x, knots):
    a,b,c = map(float, knots)
    low = np.clip((b-x)/(b-a), 0, 1)
    medium = np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))
    high = np.clip((x-b)/(c-b), 0, 1)
    return low, medium, high

# GAMBAR 8: distribusi fitur dan fungsi keanggotaan asli.
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
for i, feature in enumerate(features):
    ax = axes[0,i]
    for cls in classes:
        vals = pd.to_numeric(df.loc[df['kelas']==cls, feature], errors='coerce').dropna()
        ax.hist(vals, bins=18, density=True, alpha=.42, color=colors[cls], label=cls)
    knots = params['knots'][feature]
    for q in knots:
        ax.axvline(q, color='#333333', ls='--', lw=.9, alpha=.7)
    ax.set(title=f'Distribusi {feature.title()}', xlabel=feature.title(), ylabel='Kepadatan')
    if i == 0: ax.legend(fontsize=9)
    ax = axes[1,i]
    values = pd.to_numeric(df[feature], errors='coerce').dropna().to_numpy()
    span = max(float(np.ptp(values)), 1e-6)
    x = np.linspace(min(values.min(),knots[0])-.08*span, max(values.max(),knots[2])+.08*span, 500)
    low,mid,high = membership(x,knots)
    for y,label,color in zip((low,mid,high),('Rendah','Sedang','Tinggi'),('#326bb0','#36845b','#d65b5b')):
        ax.plot(x,y,label=label,color=color,lw=2)
    for q in knots: ax.axvline(q,color='#777',ls=':',lw=.8)
    ax.set(title=f'Keanggotaan {feature.title()}',xlabel=f'{feature.title()} | q1={knots[0]:.4f}, q2={knots[1]:.4f}, q3={knots[2]:.4f}',ylabel='Derajat keanggotaan',ylim=(-.04,1.06))
    if i == 0: ax.legend(fontsize=9)
fig.suptitle('Distribusi Fitur GLCM dan Fungsi Keanggotaan Fuzzy (Data Latih n=219)',fontsize=16,weight='bold')
p = OUT / 'gambar8_distribusi_fitur_dan_keanggotaan.png'
fig.savefig(p,dpi=250,bbox_inches='tight'); plt.close(fig); print('Tersimpan:',p)

# GAMBAR 9: per kelas; scatter benar-benar menggunakan nilai fitur CSV, bukan titik acak.
fig, axes = plt.subplots(3,3,figsize=(16,12),constrained_layout=True)
for i,cls in enumerate(classes):
    sub = df[df['kelas']==cls]
    ax=axes[0,i]
    ax.scatter(sub['correlation'],sub['contrast'],s=18,alpha=.65,color=colors[cls],edgecolors='none')
    ax.set(xlabel='Correlation',ylabel='Contrast',title=f'{cls}: sebaran Correlation–Contrast (n={len(sub)})')
    ax=axes[1,i]
    ax.hist(sub['correlation'],bins=14,color=colors[cls],alpha=.85)
    ax.axvline(params['knots']['correlation'][1],color='black',ls='--',lw=1,label='q2 latih')
    ax.set(xlabel='Correlation',ylabel='Jumlah citra',title=f'{cls}: histogram Correlation')
    if i==0: ax.legend(fontsize=9)
    ax=axes[2,i]
    imgdir=ROOT/'dataset'/cls
    path=next((imgdir/name for name in sub['nama_file'] if (imgdir/name).exists()),None)
    if path is not None:
        with Image.open(path) as im:
            rgb=np.asarray(im.convert('RGB'))
        ax.imshow(rgb)
        ax.set_title(f'{cls}: citra asli — {path.name}',fontsize=10)
        ax.axis('off')
    else:
        ax.text(.5,.5,f'Citra asli {cls} tidak ditemukan\ndi {imgdir}',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
fig.suptitle('Perbandingan Karakteristik Fitur dan Citra Asli CLL, FL, MCL',fontsize=16,weight='bold')
p=OUT/'gambar9_perbandingan_fitur_per_kelas.png'
fig.savefig(p,dpi=250,bbox_inches='tight');plt.close(fig);print('Tersimpan:',p)
print('Selesai. Sumber: data_latih.csv, parameter_model_v4.json, dan citra dataset jika tersedia.')
