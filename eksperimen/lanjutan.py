"""
Di luar preprocessing — tiga cara lain menaikkan akurasi, semua GRATIS.

Preprocessing sudah dijawab: tidak ada yang membantu. Pertanyaan berikutnya
yang wajar: kalau bukan piksel, apa yang bisa diubah tanpa melatih model?

Tiga yang diuji di sini. Ketiganya memakai embedding yang SUDAH dihitung —
nol forward pass tambahan, jadi ongkosnya nyaris nol:

  1. konsensus  — gabungkan bukti dari foto kiri DAN kanan individu yang sama
  2. hubness    — koreksi "gallery hub", foto gallery yang jadi tetangga
                  terdekat bagi terlalu banyak query
  3. fusi_TL    — gabungkan embedding MegaDescriptor-T dan L

Protokol §3 tetap dikunci: split sama, sisi tetap tidak boleh silang, query
sama persis, jadi McNemar berpasangan tetap sah.

    MODEL=L python3 lanjutan.py
"""

import json
import os

import numpy as np

import protokol as P
from evaluasi import breakdown, evaluasi, muat
from statistik import bootstrap_delta, mcnemar

HASIL = P.dir_hasil()


def _matriks(Eq, Eg, s_q, s_g):
    S = Eq @ Eg.T
    S[s_q[:, None] != s_g[None, :]] = -np.inf
    return S


def _dari_matriks(S, id_q, id_g):
    """Metrik dari matriks similarity apa pun — bentuk keluaran sama dengan
    P.evaluasi_manual, supaya bisa diuji berpasangan terhadap baseline."""
    urut = np.argsort(-S, axis=1)
    benar = (id_g[urut] == id_q[:, None]) & np.isfinite(
        np.take_along_axis(S, urut, 1))
    ap = np.zeros(len(id_q))
    for i in range(len(id_q)):
        hit = np.flatnonzero(benar[i])
        if len(hit):
            ap[i] = np.mean((np.arange(len(hit)) + 1) / (hit + 1))
    return {"rank1": benar[:, 0], "rank5": benar[:, :5].any(1), "ap": ap}


# ---------------------------------------------------------- 1. konsensus
def konsensus(S, qry, gal):
    """Foto kiri dan kanan individu yang sama harus sepakat pada satu nama.

    Kiri dan kanan adalah pola sisik yang BERBEDA, jadi keduanya membawa bukti
    yang saling bebas. Dua bukti bebas yang sepakat jauh lebih kuat dari satu.

    JUJUR SOAL ONGKOSNYA: ini bukan perbaikan gratis dari sisi data — ia butuh
    DUA foto per penyu saat identifikasi, bukan satu. Yang gratis hanya
    komputasinya. Konsekuensi UX harus ditulis di laporan.

    Skor tiap query di-z-score dulu; tanpa itu, foto dengan sebaran similarity
    lebar akan mendominasi penjumlahan.
    """
    S = S.copy()
    Z = np.full_like(S, -np.inf)
    for i in range(len(S)):
        m = np.isfinite(S[i])
        v = S[i, m]
        Z[i, m] = (v - v.mean()) / max(v.std(), 1e-9)

    # peta identitas gallery -> kolom, per sisi
    kol = {}
    for j, g in enumerate(gal):
        kol[(g["identity"], g["side"])] = j

    # pasangkan query kiri & kanan milik individu + tahun yang sama
    pasangan = {}
    for i, q in enumerate(qry):
        pasangan.setdefault((q["identity"], q["year"]), {})[q["side"]] = i

    ident = sorted({g["identity"] for g in gal})
    idx_id = {k: i for i, k in enumerate(ident)}
    Sgab = np.full((len(qry), len(ident)), -np.inf)

    for (_, _), p in pasangan.items():
        skor = np.zeros(len(ident))
        n = 0
        for sisi, i in p.items():
            for k in ident:
                j = kol.get((k, sisi))
                if j is not None and np.isfinite(Z[i, j]):
                    skor[idx_id[k]] += Z[i, j]
            n += 1
        for i in p.values():
            Sgab[i] = skor / max(n, 1)

    return Sgab, np.array(ident)


# ------------------------------------------------------------ 2. hubness
def koreksi_hubness(S):
    """Kurangi rata-rata similarity tiap foto GALLERY terhadap semua query.

    Di ruang berdimensi tinggi muncul "hub": beberapa foto gallery menjadi
    tetangga terdekat bagi terlalu banyak query, apa pun identitasnya. Foto
    seperti itu memakan peringkat-1 dan menutupi jawaban yang benar.
    Mengurangi rata-rata kolom membuat tiap foto gallery bersaing pada
    kelebihannya sendiri, bukan pada popularitas umumnya.
    """
    S = S.copy()
    M = np.isfinite(S)
    rata = np.where(M, S, np.nan)
    rata = np.nanmean(rata, axis=0, keepdims=True)
    return np.where(M, S - rata, -np.inf)


# --------------------------------------------------------------- 3. fusi
def fusi(nama="raw"):
    """Gabungkan embedding T-224 dan L-384 (masing-masing L2-norm, lalu
    disambung dan di-L2-norm lagi). Dua model melihat gambar pada resolusi
    berbeda, jadi kesalahannya tidak sepenuhnya sama."""
    keluar = []
    for m in ("T", "L"):
        d = os.path.join(P.dir_hasil(model=m), f"emb_{nama}.npy")
        if not os.path.exists(d):
            return None
        E = np.load(d)
        keluar.append(E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9))
    E = np.concatenate(keluar, axis=1)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- utama
def main():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    Eg, Eq = muat("raw", gal, qry)
    S = _matriks(Eq, Eg, s_q, s_g)

    varian = {"raw (baseline)": evaluasi("raw", gal, qry)}

    Sk, ident = konsensus(S, qry, gal)
    varian["konsensus dua sisi"] = _dari_matriks(Sk, id_q, ident)

    varian["koreksi hubness"] = _dari_matriks(koreksi_hubness(S), id_q, id_g)

    Ef = fusi("raw")
    if Ef is not None:
        Sf = _matriks(Ef[len(gal):], Ef[:len(gal)], s_q, s_g)
        varian["fusi T-224 + L-384"] = _dari_matriks(Sf, id_q, id_g)
        varian["fusi + hubness"] = _dari_matriks(koreksi_hubness(Sf), id_q, id_g)
        Sfk, ident2 = konsensus(Sf, qry, gal)
        varian["fusi + konsensus"] = _dari_matriks(Sfk, id_q, ident2)

    base = varian["raw (baseline)"]
    tabel = {}
    for nama, h in varian.items():
        b = {"label": nama,
             "rank1": float(h["rank1"].mean() * 100),
             "rank5": float(h["rank5"].mean() * 100),
             "mAP": float(h["ap"].mean() * 100),
             "n": int(len(h["ap"]))}
        if nama != "raw (baseline)":
            b["delta_rank1"] = bootstrap_delta(
                base["rank1"].astype(float) * 100, h["rank1"].astype(float) * 100)
            b["mcnemar_rank1"] = mcnemar(base["rank1"].astype(bool),
                                         h["rank1"].astype(bool))
            b["delta_mAP"] = bootstrap_delta(base["ap"] * 100, h["ap"] * 100)
        b.update(breakdown(h, qry))
        tabel[nama] = b

    with open(os.path.join(HASIL, "lanjutan.json"), "w") as f:
        json.dump(tabel, f, indent=2)

    print(f"{'Varian':24} {'R-1':>6} {'R-5':>6} {'mAP':>6} "
          f"{'ΔR-1 vs raw (95% CI)':>26} {'p':>10}")
    for nama, b in tabel.items():
        if "delta_rank1" not in b:
            print(f"{nama:24} {b['rank1']:6.2f} {b['rank5']:6.2f} {b['mAP']:6.2f} "
                  f"{'—':>26} {'—':>10}")
            continue
        d = b["delta_rank1"]
        ci = f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        tanda = "*" if b["mcnemar_rank1"]["p_value"] < 0.05 else " "
        print(f"{nama:24} {b['rank1']:6.2f} {b['rank5']:6.2f} {b['mAP']:6.2f} "
              f"{ci:>26} {b['mcnemar_rank1']['p_value']:9.3g}{tanda}")
    print("\n* = signifikan pada p < 0.05 (McNemar berpasangan)")
    print("Konsensus memakai DUA foto per penyu — bukan perbandingan setara "
          "dengan varian satu foto. Lihat catatan di docstring.")


if __name__ == "__main__":
    main()
