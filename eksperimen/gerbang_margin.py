"""
Gerbang margin — kapan pencocokan lokal layak dijalankan.

    MODEL=MIEWID DATASET=reunion    python3 gerbang_margin.py --matcher xfeat --k 40
    MODEL=MIEWID DATASET=amvrakikos python3 gerbang_margin.py \
        --matcher vm-xfeat-lightglue-512-kepala_gt --k 20

Kenapa ada
----------
Pencocokan lokal terbukti +33 pp di Amvrakikos (stage-1 lemah, 57%) dan
-34 pp di Reunion (stage-1 kuat, 84,5%). Kalau pipeline memilih berdasarkan
NAMA DATASET, ia tidak bisa dipakai di lapangan — foto dari pantai tidak
berlabel "Zakynthos".

Jadi keputusannya harus diambil dari sinyal yang terukur pada query itu
sendiri. Kandidatnya: **margin** stage-1, yaitu selisih cosine peringkat 1
dan peringkat 2. Margin besar = "mirip DAN tidak ada saingan dekat".

Gerbangnya: margin >= ambang -> pakai stage-1 apa adanya (murah);
margin < ambang -> jalankan matcher lokal + fusi terkalibrasi (mahal).

PERINGATAN OVERFITTING
----------------------
Menyapu ambang lalu melaporkan yang terbaik adalah cara paling mudah menipu
diri sendiri: dengan 168 query dan puluhan ambang, selalu ada satu yang
kelihatan bagus karena kebetulan. Maka:

  * ambang dipilih **leave-one-query-out** — untuk tiap query, ambang
    ditentukan dari 167 query lainnya, lalu diterapkan ke query itu. Angka
    LOO inilah yang boleh dipercaya.
  * sapuan penuh tetap dicetak sebagai diagnosa, dan ditandai ORACLE karena
    ia melihat jawaban. Jangan pernah kutip angka oracle sebagai hasil.
"""

import argparse
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

import protokol as P

HASIL = P.dir_hasil()


def siapkan(kondisi, matcher, k):
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    E = np.load(os.path.join(HASIL, f"emb_{kondisi}.npy"))
    Eg, Eq = E[:len(gal)], E[len(gal):]
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    S = Eq @ Eg.T
    S[s_q[:, None] != s_g[None, :]] = -np.inf
    idx = np.argsort(-S, axis=1)[:, :k]
    cos = np.take_along_axis(S, idx, 1)
    benar = id_g[idx] == id_q[:, None]
    M = np.load(os.path.join(HASIL, f"rerank_{matcher}_k{k}.npy"))
    if M.shape != cos.shape:
        raise SystemExit(f"bentuk skor lokal {M.shape} != {cos.shape}")
    return cos, M.astype(float), benar


def kalibrasi_loo(x, y, milik):
    keluar = np.zeros_like(x, dtype=float)
    for q in np.unique(milik):
        lain = milik != q
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x[lain], y[lain])
        keluar[milik == q] = iso.predict(x[milik == q])
    return keluar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--kondisi", default="raw")
    a = ap.parse_args()

    cos, M, benar = siapkan(a.kondisi, a.matcher, a.k)
    nq = len(cos)
    milik = np.repeat(np.arange(nq), a.k)
    p_g = kalibrasi_loo(cos.ravel(), benar.ravel().astype(float), milik)
    p_l = kalibrasi_loo(M.ravel(), benar.ravel().astype(float), milik)
    fus = (0.5 * (p_g + p_l)).reshape(cos.shape)

    b_stage1 = benar[np.arange(nq), np.argmax(cos, 1)]
    b_cal = benar[np.arange(nq), np.argmax(fus, 1)]
    margin = cos[:, 0] - cos[:, 1]          # cos sudah terurut menurun

    def akur(mask_pakai_lokal):
        """Gabungkan: query yang digerbang -> cal, sisanya -> stage1."""
        return np.where(mask_pakai_lokal, b_cal, b_stage1)

    ambang = np.unique(np.round(margin, 4))
    print(f"\n{P.DATASET} · {a.matcher} · k={a.k} · n={nq}")
    print(f"  stage-1 saja      : {b_stage1.mean()*100:6.2f}%")
    print(f"  lokal terkalibrasi: {b_cal.mean()*100:6.2f}%   "
          f"(dipakai untuk SEMUA query)")

    # ---- oracle: ambang terbaik kalau boleh melihat jawaban
    skor = [(akur(margin < t).mean(), t) for t in ambang]
    terbaik, t_best = max(skor)
    hemat = (margin >= t_best).mean() * 100
    print(f"  ORACLE ambang     : {terbaik*100:6.2f}%  pada margin<{t_best:.4f}"
          f"  ({hemat:.0f}% query lewat gerbang, hemat waktu)")
    print("  ^ JANGAN dikutip sebagai hasil: ambangnya dipilih dengan melihat"
          " jawaban.")

    # ---- LOO: ambang dipilih tanpa melihat query yang dinilai
    keluar = np.zeros(nq, bool)
    for i in range(nq):
        lain = np.ones(nq, bool)
        lain[i] = False
        sk = [(np.where(margin[lain] < t, b_cal[lain], b_stage1[lain]).mean(), t)
              for t in ambang]
        t_i = max(sk)[1]
        keluar[i] = b_cal[i] if margin[i] < t_i else b_stage1[i]
    print(f"  LOO gerbang       : {keluar.mean()*100:6.2f}%   <-- angka yang sah")

    import statistik as S_
    for nama, v in (("cal-semua", b_cal), ("LOO gerbang", keluar)):
        d = S_.bootstrap_delta(b_stage1 * 100.0, v * 100.0)
        m = S_.mcnemar(b_stage1, v)
        print(f"    {nama:12} vs stage-1: {d['delta']:+6.2f} "
              f"[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] p={m['p_value']:.4g}")


if __name__ == "__main__":
    main()
