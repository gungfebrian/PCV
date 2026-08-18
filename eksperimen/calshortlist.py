"""
Fusi skor terkalibrasi (CalShortlist) atas shortlist stage-1.

    MODEL=MIEWID DATASET=reunion python3 calshortlist.py --matcher xfeat --k 40

Kenapa ada
----------
`rerank.py` MENGGANTI urutan stage-1 dengan skor matcher lokal. Itu bekerja
spektakuler saat stage-1 buruk (MegaDescriptor 25,00 -> 69,05) dan menghancurkan
saat stage-1 bagus (MiewID 84,52 -> 50,00, p=2,4e-13). Fusi peringkat naif
(RRF) pun masih -23,81.

Masalahnya bukan matcher-nya, melainkan bahwa dua skor yang skalanya sama
sekali berbeda — cosine 0..1 dan jumlah inlier 0..300 — tidak bisa
dibandingkan atau dirata-ratakan begitu saja. CalShortlist memetakan keduanya
ke probabilitas lebih dulu (isotonic regression), baru dirata-ratakan.

    f = 0.5 * (p_global + p_lokal)

Ini resep WildFusion (Cermak et al. 2024), dan dipakai produksi di
turtle-identification-be.

Kebocoran ditutup dengan leave-one-query-out
--------------------------------------------
Kalibrator dipasang pada pasangan dari SEMUA query KECUALI query yang sedang
dinilai. Kalau dipasang pada seluruh data termasuk query itu sendiri,
angkanya naik palsu — kalibrator sudah melihat jawabannya.

Protokol §3 tidak berubah: split, kunci sisi, dan himpunan query identik,
jadi McNemar berpasangan terhadap stage-1 tetap sah.
"""

import argparse
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

import protokol as P

HASIL = P.dir_hasil()


def shortlist(Eq, Eg, sisi_q, sisi_g, k):
    """Matriks similarity stage-1 + indeks top-k, dengan kunci sisi.

    Kunci sisi dipasang sebagai -inf SEBELUM argsort, sama seperti
    protokol.evaluasi_manual — bukan difilter setelahnya.
    """
    S = Eq @ Eg.T
    S[sisi_q[:, None] != sisi_g[None, :]] = -np.inf
    idx = np.argsort(-S, axis=1)[:, :k]
    return S, idx


def kalibrasi_loo(x, y, milik):
    """Peta skor -> probabilitas, leave-one-query-out.

    x       skor datar (n_query * k,)
    y       label benar/salah (n_query * k,)
    milik   query pemilik tiap pasangan (n_query * k,)
    """
    keluar = np.zeros_like(x, dtype=np.float64)
    for q in np.unique(milik):
        lain = milik != q
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x[lain], y[lain])
        keluar[milik == q] = iso.predict(x[milik == q])
    return keluar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", default="xfeat")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--kondisi", default="raw")
    a = ap.parse_args()

    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    E = np.load(os.path.join(HASIL, f"emb_{a.kondisi}.npy"))
    Eg, Eq = E[:len(gal)], E[len(gal):]
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    S, idx = shortlist(Eq, Eg, s_q, s_g, a.k)
    M = np.load(os.path.join(HASIL, f"rerank_{a.matcher}_k{a.k}.npy"))
    if M.shape != idx.shape:
        raise SystemExit(f"bentuk skor lokal {M.shape} != shortlist {idx.shape} — "
                         f"k atau kondisi tidak cocok dengan run rerank")

    cos = np.take_along_axis(S, idx, 1)
    benar = id_g[idx] == id_q[:, None]
    milik = np.repeat(np.arange(len(qry)), a.k)

    p_glob = kalibrasi_loo(cos.ravel(), benar.ravel().astype(float), milik)
    p_lok = kalibrasi_loo(M.ravel().astype(float), benar.ravel().astype(float), milik)
    fus = (0.5 * (p_glob + p_lok)).reshape(cos.shape)

    def metrik(skor):
        urut = np.argsort(-skor, axis=1)
        b = np.take_along_axis(benar, urut, 1)
        ap_ = np.zeros(len(qry))
        for i in range(len(qry)):
            hit = np.flatnonzero(b[i])
            if len(hit):
                ap_[i] = np.mean((np.arange(len(hit)) + 1) / (hit + 1))
        return {"rank1": b[:, 0], "rank5": b[:, :5].any(1), "ap": ap_}

    import statistik as S_

    hasil = {"stage1": metrik(cos), "lokal": metrik(M.astype(float)),
             "cal": metrik(fus)}
    print(f"\n{a.matcher}  k={a.k}  n={len(qry)}   (kalibrator leave-one-query-out)")
    print(f"{'':10}{'R-1':>7}{'R-5':>7}{'mAP':>8}   {'ΔR-1 vs stage1 (95% CI)':>28}{'p':>10}")
    base = hasil["stage1"]
    for nama, h in hasil.items():
        r1, r5 = h["rank1"].mean() * 100, h["rank5"].mean() * 100
        m = h["ap"].mean() * 100
        if nama == "stage1":
            print(f"{nama:10}{r1:7.2f}{r5:7.2f}{m:8.2f}{'—':>28}{'—':>10}")
            continue
        d = S_.bootstrap_delta(base["rank1"] * 100.0, h["rank1"] * 100.0)
        mc = S_.mcnemar(base["rank1"].astype(bool), h["rank1"].astype(bool))
        ci = f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        print(f"{nama:10}{r1:7.2f}{r5:7.2f}{m:8.2f}{ci:>28}{mc['p_value']:10.3g}")

    keluar = os.path.join(HASIL, f"calshortlist_{a.matcher}_k{a.k}.json")
    with open(keluar, "w") as f:
        json.dump({**P.metadata_run(kat), "matcher": a.matcher, "k": a.k,
                   "tabel": {n: {"rank1": float(h["rank1"].mean() * 100),
                                 "rank5": float(h["rank5"].mean() * 100),
                                 "mAP": float(h["ap"].mean() * 100),
                                 "n": len(qry)} for n, h in hasil.items()}},
                  f, indent=2)
    print(f"\ndisimpan: {keluar}")


if __name__ == "__main__":
    main()
