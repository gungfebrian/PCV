"""
Fusi kiri + kanan — satu penyu, satu jawaban.

    MODEL=MIEWID DATASET=reunion python3 dua_sisi.py --kondisi raw

Kenapa ada
----------
Protokol §3 mengunci pencarian per sisi: query kiri hanya dicari di galeri
kiri. Itu benar dan tidak diubah di sini. Tapi metrik yang kita laporkan
selama ini menilai tiap FOTO, padahal di lapangan satu penyu difoto dua sisi
sekaligus dan yang dibutuhkan cuma SATU nama.

Jadi ini bukan pelonggaran protokol, melainkan metrik yang berbeda:
identity top-1 dengan satu tebakan per penyu, bukan per foto. Dua pencarian
tetap berjalan terpisah dan tetap terkunci sisi; yang digabung hanya
keputusannya di akhir.

Aturan yang dibandingkan
------------------------
Semua memakai skor stage-1 yang sama, tanpa hitung ulang embedding.

  satu_foto   rata-rata top-1 per foto (baseline, = angka yang biasa kita pakai)
  sepakat     benar hanya kalau KEDUA sisi menunjuk nama yang sama dan benar
  fallback    kalau sepakat -> nama itu; kalau tidak -> rank-1 dengan MARGIN
              lebih besar (skor#1 - skor#2 pada sisi itu)
  unik_top5   benar hanya kalau tepat satu nama muncul di top-5 kedua sisi

PERINGATAN dari pengukuran repo turtle-identification-be: `unik_top5`
terdengar seperti penyelamat tapi di Hawksbill justru LEBIH BURUK daripada
satu foto (67,6% vs 72,1%). Dimasukkan di sini supaya bisa dibantah dengan
data kita sendiri, bukan supaya dipakai.

Margin, bukan skor mentah, yang dipakai untuk memilih sisi pemenang: cosine
tertinggi hanya berarti "mirip", sedangkan margin berarti "mirip DAN tidak
ada saingan dekat" — itu yang menandakan keyakinan.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import protokol as P

HASIL = P.dir_hasil()


def peringkat(Eq, Eg, sisi_q, sisi_g, id_g):
    """Untuk tiap query: urutan identitas galeri + skornya, terkunci sisi."""
    S = Eq @ Eg.T
    S[sisi_q[:, None] != sisi_g[None, :]] = -np.inf
    urut = np.argsort(-S, axis=1)
    nama = id_g[urut]
    skor = np.take_along_axis(S, urut, 1)
    return nama, skor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kondisi", default="raw")
    a = ap.parse_args()

    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    E = np.load(os.path.join(HASIL, f"emb_{a.kondisi}.npy"))
    Eg, Eq = E[:len(gal)], E[len(gal):]
    id_g = np.array([r["identity"] for r in gal])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    nama, skor = peringkat(Eq, Eg, s_q, s_g, id_g)

    # Kelompokkan query per individu. Hanya individu dengan DUA sisi yang bisa
    # difusikan — yang cuma punya satu sisi dilaporkan terpisah, tidak
    # diam-diam dihitung sebagai kemenangan aturan fusi.
    per_ind = defaultdict(list)
    for i, r in enumerate(qry):
        per_ind[r["identity"]].append(i)
    pasangan = {k: v for k, v in per_ind.items() if len(v) == 2}
    tunggal = {k: v for k, v in per_ind.items() if len(v) != 2}

    hasil = {k: [] for k in ("satu_foto", "sepakat", "fallback", "unik_top5")}
    n_sepakat = n_sepakat_benar = 0
    for ident, (i, j) in sorted(pasangan.items()):
        a_nama, a_skor = nama[i], skor[i]
        b_nama, b_skor = nama[j], skor[j]
        hasil["satu_foto"] += [a_nama[0] == ident, b_nama[0] == ident]

        setuju = a_nama[0] == b_nama[0]
        if setuju:
            n_sepakat += 1
            n_sepakat_benar += a_nama[0] == ident
        hasil["sepakat"].append(setuju and a_nama[0] == ident)

        if setuju:
            tebak = a_nama[0]
        else:
            m_a = a_skor[0] - a_skor[1]
            m_b = b_skor[0] - b_skor[1]
            tebak = a_nama[0] if m_a >= m_b else b_nama[0]
        hasil["fallback"].append(tebak == ident)

        irisan = set(a_nama[:5]) & set(b_nama[:5])
        hasil["unik_top5"].append(len(irisan) == 1 and irisan.pop() == ident)

    n = len(pasangan)
    print(f"\n{P.DATASET} · {a.kondisi} · {n} individu berpasangan "
          f"({len(hasil['satu_foto'])} foto)"
          + (f" · {len(tunggal)} individu tidak berpasangan, DIKECUALIKAN"
             if tunggal else ""))
    print(f"  rank-1 kedua sisi sepakat: {n_sepakat}/{n}, "
          f"dan {n_sepakat_benar} dari yang sepakat itu benar")
    print()

    import statistik as S_
    base = np.array(hasil["satu_foto"], bool)
    # Baseline diringkas ke satuan INDIVIDU: rata-rata dua sisi, jadi nilainya
    # 0 / 0,5 / 1. Karena itu BUKAN keputusan biner tunggal per individu,
    # McNemar TIDAK berlaku di sini — memaksakannya (mis. dengan ambang
    # b>=0,5) diam-diam mengubah baseline jadi "minimal satu sisi benar",
    # yang pertanyaannya berbeda. Yang sah: bootstrap berpasangan atas selisih
    # per individu.
    b2 = base.reshape(-1, 2).mean(1)
    print(f"{'aturan':12}{'benar/n':>12}{'akurasi':>10}"
          f"{'Δ vs satu foto (bootstrap 95% CI)':>36}")
    for k in ("satu_foto", "sepakat", "fallback", "unik_top5"):
        v = np.array(hasil[k], bool)
        akur = v.mean() * 100
        if k == "satu_foto":
            print(f"{k:12}{f'{v.sum()}/{len(v)}':>12}{akur:9.2f}%{'—':>36}")
            continue
        d = S_.bootstrap_delta(b2 * 100.0, v * 100.0)
        tanda = "SIG" if d["signifikan"] else "ns"
        ci = f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] {tanda}"
        print(f"{k:12}{f'{v.sum()}/{len(v)}':>12}{akur:9.2f}%{ci:>36}")

    # Breakdown per spesies — wajib menurut protokol, dan di sinilah fusi
    # dua sisi diharapkan paling berguna (spesies yang stage-1-nya lemah).
    spesies = {}
    for ident, (i, _) in sorted(pasangan.items()):
        spesies.setdefault(qry[i].get("species", "?"), []).append(ident)
    if len(spesies) > 1:
        print(f"\n{'spesies':12}{'n':>5}{'satu foto':>12}{'fallback':>11}{'Δ':>8}")
        urutan = [k for k, _ in sorted(pasangan.items())]
        for sp, daftar in sorted(spesies.items()):
            m = np.array([i in set(daftar) for i in urutan])
            print(f"{sp:12}{m.sum():5d}{b2[m].mean()*100:11.2f}%"
                  f"{np.array(hasil['fallback'])[m].mean()*100:10.2f}%"
                  f"{(np.array(hasil['fallback'])[m].mean()-b2[m].mean())*100:+8.2f}")

    keluar = os.path.join(HASIL, f"dua_sisi_{a.kondisi}.json")
    with open(keluar, "w") as f:
        json.dump({**P.metadata_run(kat), "kondisi": a.kondisi,
                   "n_individu": n, "n_sepakat": n_sepakat,
                   "n_sepakat_benar": n_sepakat_benar,
                   "akurasi": {k: float(np.mean(v) * 100)
                               for k, v in hasil.items()}}, f, indent=2)
    print(f"\ndisimpan: {keluar}")


if __name__ == "__main__":
    main()
