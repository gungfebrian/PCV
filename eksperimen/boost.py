"""
Menaikkan akurasi TANPA menyentuh gambar sama sekali.

    MODEL=L python3 boost.py                 # semua metode, kondisi raw
    MODEL=L python3 boost.py --kondisi resize512
    MODEL=L python3 boost.py --metode csls   # satu saja

Semua metode di sini bekerja pada matriks embedding yang SUDAH ADA
(`emb_*.npy`). Tidak ada gambar yang dibaca ulang, tidak ada model yang
dijalankan. Konsekuensinya: satu putaran penuh selesai dalam hitungan detik,
bukan jam. Itu sebabnya bagian ini dikerjakan lebih dulu daripada YOLO atau
matcher baru — kalau ada yang gratis, habiskan dulu yang gratis.

Yang diuji
----------
alpha-QE   Query expansion: query diganti rata-rata berbobot dirinya + top-k
           tetangga galeri. Bobot = similarity^alpha, jadi tetangga yang ragu
           hampir tidak ikut. Kalau tetangga teratas salah, ia justru menarik
           query menjauh — metode ini menguatkan apa pun yang sudah ada.
DBA        Sama, tapi diterapkan ke sisi GALERI. Boleh dilakukan sekali di
           awal karena galeri tidak berubah saat dipakai.
CSLS       Koreksi hubness. Ada vektor galeri yang jadi "hub": dekat dengan
           semua orang, jadi menang di banyak query tanpa alasan. CSLS
           mengurangi skor tiap kandidat sebanyak rata-rata kedekatannya
           dengan tetangganya sendiri, jadi hub kehilangan keuntungan gratis.
k-recip    Jaccard atas himpunan tetangga k-reciprocal (Zhong dkk. 2017).
           Dua foto dianggap mirip kalau mereka SALING ada di daftar tetangga
           satu sama lain, bukan cuma searah.
whiten     PCA-whitening. Dimensi berenergi besar biasanya berisi variasi
           yang tidak relevan (cahaya, latar pasir). Whitening menyamakan
           energi antar dimensi supaya bukan dimensi itu yang menentukan.
ensemble   Gabung embedding L-384 dan T-224, masing-masing dinormalisasi
           dulu supaya bobotnya seimbang.

Kejujuran yang wajib disebut
----------------------------
1. `whiten` dipasang HANYA dari galeri. Memasangnya dari query adalah
   kebocoran: di dunia nyata query datang satu per satu dan statistiknya
   belum diketahui.
2. `k-recip` di sini sengaja ditulis versi yang **bisa dipakai satu query
   sekali jalan** — himpunan tetangga query hanya diambil dari galeri, tidak
   pernah dari query lain. Implementasi asli Zhong dkk. memakai seluruh batch
   query sekaligus, dan itu TIDAK berlaku untuk kamera langsung.
3. Semua metode dijalankan TERPISAH PER SISI. Statistik tetangga sisi kiri
   tidak boleh bocor ke sisi kanan, sama seperti pencocokannya.
"""

import argparse
import json
import os

import numpy as np

import protokol as P
from evaluasi import breakdown, muat
from statistik import bootstrap_delta, mcnemar

HASIL = P.dir_hasil()


def _norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------- metode
def polos(Sq, Eq, Eg):
    """Baseline: cosine apa adanya."""
    return Sq


def alpha_qe(Sq, Eq, Eg, k=3, alpha=3.0):
    """Query expansion berbobot.

    q' = normalize(q + sum_i (s_i^alpha) * g_i) atas top-k galeri.

    alpha=3 membuat tetangga dengan similarity 0,5 hanya menyumbang 0,125 —
    cukup tegas untuk menahan tetangga yang meragukan tanpa harus memilih
    ambang keras.
    """
    idx = np.argsort(-Sq, axis=1)[:, :k]
    w = np.take_along_axis(Sq, idx, 1).clip(0) ** alpha
    tambah = np.einsum("qk,qkd->qd", w, Eg[idx])
    return _norm(Eq + tambah) @ Eg.T


def dba(Sq, Eq, Eg, k=2, alpha=3.0):
    """Database-side augmentation: tiap vektor galeri diperkaya tetangganya."""
    Sg = Eg @ Eg.T
    np.fill_diagonal(Sg, -np.inf)                 # jangan jadi tetangga sendiri
    idx = np.argsort(-Sg, axis=1)[:, :k]
    w = np.take_along_axis(Sg, idx, 1).clip(0) ** alpha
    Eg2 = _norm(Eg + np.einsum("gk,gkd->gd", w, Eg[idx]))
    return Eq @ Eg2.T


def csls(Sq, Eq, Eg, k=5):
    """Cross-domain Similarity Local Scaling.

    s'(q,g) = 2*s(q,g) - r(q) - r(g)

    r(x) = rata-rata similarity x ke k tetangga terdekatnya di sisi seberang.
    Kandidat yang memang dekat dengan SEMUA orang (hub) punya r besar, jadi
    keuntungan gratisnya dipotong.
    """
    def rerata_atas(S, k):
        kk = min(k, S.shape[1])
        return np.sort(S, axis=1)[:, -kk:].mean(1)

    rq = rerata_atas(Sq, k)
    rg = rerata_atas(Sq.T, k)
    return 2 * Sq - rq[:, None] - rg[None, :]


def k_reciprocal(Sq, Eq, Eg, k1=8, k2=3, lam=0.3):
    """Jarak Jaccard atas himpunan tetangga k-reciprocal.

    Versi yang bisa dipakai per query: himpunan tetangga sebuah query hanya
    diambil dari GALERI, tidak pernah dari query lain. Jadi hasilnya tidak
    berubah kalau query lain tidak ada — syarat mutlak untuk kamera langsung.
    """
    ng = len(Eg)
    k1 = min(k1, ng)
    Sg = Eg @ Eg.T

    def vektor(S, sendiri=None):
        """Bobot Gaussian atas tetangga k-reciprocal, dinormalisasi."""
        n = len(S)
        V = np.zeros((n, ng), np.float32)
        tetangga_g = np.argsort(-Sg, axis=1)[:, :k1]
        for i in range(n):
            kandidat = np.argsort(-S[i])[:k1]
            # saling: g menganggap i cukup dekat juga
            saling = [g for g in kandidat
                      if (sendiri is not None and sendiri == g)
                      or S[i, g] >= S[i, tetangga_g[g][-1]]]
            if not saling:
                saling = list(kandidat[:1])
            d = 1.0 - S[i, saling]
            w = np.exp(-d / max(d.std(), 1e-3))
            V[i, saling] = w / w.sum()
        return V

    Vq = vektor(Sq)
    Vg = np.zeros((ng, ng), np.float32)
    tetangga_g = np.argsort(-Sg, axis=1)[:, :k1]
    for g in range(ng):
        s = tetangga_g[g]
        d = 1.0 - Sg[g, s]
        w = np.exp(-d / max(d.std(), 1e-3))
        Vg[g, s] = w / w.sum()

    # ekspansi lokal: rata-ratakan V dengan k2 tetangga terdekatnya
    if k2 > 1:
        kk = min(k2, ng)
        Vg = Vg[np.argsort(-Sg, axis=1)[:, :kk]].mean(1)
        Vq = Vg[np.argsort(-Sq, axis=1)[:, :kk]].mean(1)

    # Jaccard: 1 - sum(min)/sum(max)
    mn = np.minimum(Vq[:, None, :], Vg[None, :, :]).sum(-1)
    mx = np.maximum(Vq[:, None, :], Vg[None, :, :]).sum(-1)
    jac = 1.0 - mn / np.maximum(mx, 1e-12)
    return -((1 - lam) * jac + lam * (1.0 - Sq))     # skor = -jarak


def whiten(Sq, Eq, Eg, eps=1e-3, buang=0):
    """PCA-whitening yang DIPASANG DARI GALERI SAJA.

    Memasangnya dari query adalah kebocoran transduktif: di lapangan query
    datang satu per satu dan statistiknya belum ada.
    """
    mu = Eg.mean(0)
    U, s, _ = np.linalg.svd((Eg - mu).T @ (Eg - mu) / max(len(Eg) - 1, 1))
    if buang:                       # buang komponen berenergi terbesar
        U, s = U[:, buang:], s[buang:]
    W = U / np.sqrt(s + eps)
    return _norm((Eq - mu) @ W) @ _norm((Eg - mu) @ W).T


METODE = {
    "polos": polos,
    "alpha_qe": alpha_qe,
    "dba": dba,
    "csls": csls,
    "k_recip": k_reciprocal,
    "whiten": whiten,
}

LABEL = {
    "polos": "baseline cosine",
    "alpha_qe": "alpha query expansion (k=3, a=3)",
    "dba": "database-side augmentation (k=2)",
    "csls": "CSLS koreksi hubness (k=5)",
    "k_recip": "k-reciprocal Jaccard (k1=8, k2=3, lam=0.3)",
    "whiten": "PCA-whitening dipasang dari galeri",
}


# ------------------------------------------------------------- eksekusi
def jalankan(fn, Eq, Eg, sisi_q, sisi_g, **kw):
    """Terapkan metode TERPISAH PER SISI, lalu rakit ulang jadi satu matriks.

    Ini bukan sekadar rapi: tetangga, hub, dan statistik whitening sisi kiri
    tidak boleh ikut menentukan hasil sisi kanan. Kalau digabung, kunci sisi
    di protokol §3 bocor lewat pintu belakang tanpa memunculkan error apa pun.
    """
    S = np.full((len(Eq), len(Eg)), -np.inf, np.float64)
    for s in P.SISI:
        iq = np.flatnonzero(sisi_q == s)
        ig = np.flatnonzero(sisi_g == s)
        if not len(iq) or not len(ig):
            continue
        sub_q, sub_g = Eq[iq], Eg[ig]
        S[np.ix_(iq, ig)] = fn(sub_q @ sub_g.T, sub_q, sub_g, **kw)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kondisi", default="raw")
    ap.add_argument("--metode", default=None,
                    help="jalankan satu metode saja")
    ap.add_argument("--ensemble", action="store_true",
                    help="gabung embedding L dan T sebelum menjalankan metode")
    a = ap.parse_args()

    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    Eg, Eq = muat(a.kondisi, gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    sisi_g = np.array([r["side"] for r in gal])
    sisi_q = np.array([r["side"] for r in qry])

    catatan = ""
    if a.ensemble:
        lain = HASIL.replace(f"_{P.MODEL}_", "_T_" if P.MODEL == "L" else "_L_")
        p = os.path.join(lain, f"emb_{a.kondisi}.npy")
        if not os.path.exists(p):
            raise SystemExit(f"embedding pasangan tidak ada: {p}")
        E2 = np.load(p)
        Eg = _norm(np.hstack([Eg, _norm(E2[:len(gal)])]))
        Eq = _norm(np.hstack([Eq, _norm(E2[len(gal):])]))
        catatan = f" [ensemble dengan {os.path.basename(lain)}]"

    dasar = P.metrik_dari_matriks(
        jalankan(polos, Eq, Eg, sisi_q, sisi_g), id_q, id_g)
    print(f"kondisi {a.kondisi}{catatan}   n={len(id_q)}")
    print(f"{'metode':<34} {'R1':>6} {'R5':>6} {'mAP':>6} "
          f"{'dR1':>7} {'p':>9}  {'CI95 mAP':>18}")
    print("-" * 96)

    keluar = {}
    for nama, fn in METODE.items():
        if a.metode and nama != a.metode:
            continue
        try:
            S = jalankan(fn, Eq, Eg, sisi_q, sisi_g)
            h = P.metrik_dari_matriks(S, id_q, id_g)
        except Exception as e:
            print(f"{LABEL[nama]:<34} GAGAL: {str(e).splitlines()[0][:40]}")
            continue
        r = P.ringkas(h)
        mc = mcnemar(dasar["rank1"], h["rank1"])
        bs = bootstrap_delta(dasar["ap"], h["ap"])
        d1 = r["rank1"] - float(dasar["rank1"].mean() * 100)
        tanda = "*" if mc["p_value"] < 0.05 else " "
        print(f"{LABEL[nama]:<34} {r['rank1']:6.2f} {r['rank5']:6.2f} "
              f"{r['mAP']:6.2f} {d1:+7.2f} {mc['p_value']:9.2e}{tanda} "
              f"[{bs['ci95'][0]*100:+6.2f},{bs['ci95'][1]*100:+6.2f}]")
        keluar[nama] = {"label": LABEL[nama], **r, "delta_rank1": d1,
                        "mcnemar": mc, "bootstrap_mAP": bs,
                        **breakdown(h, qry)}

    print("\n* = p < 0,05 (McNemar berpasangan). CI95 adalah selisih mAP, "
          "bukan mAP-nya sendiri.")
    print("Ingat: n =", len(id_q),
          "- satu prediksi berubah kira-kira", f"{100/len(id_q):.2f}", "poin.")

    nama_berkas = f"boost_{a.kondisi}{'_ens' if a.ensemble else ''}.json"
    with open(os.path.join(HASIL, nama_berkas), "w") as f:
        json.dump({"kondisi": a.kondisi, "ensemble": a.ensemble,
                   "n": len(id_q), "hasil": keluar}, f, indent=2)
    print("ditulis:", nama_berkas)


if __name__ == "__main__":
    main()
