"""
Grid stage-2: matcher x preprocessing. Semua angka dihitung, tidak ada yang diketik.

Menjawab pertanyaan "MegaDescriptor + <matcher>, dan bagaimana kalau
gambarnya dipreprocessing dulu sebelum matching". Stage-1 SELALU memakai
embedding raw — yang divariasikan hanya gambar yang masuk ke matcher stage-2,
supaya efek yang terukur benar-benar berasal dari stage-2.

    MODEL=L python3 grid_rerank.py            # jalankan yang belum ada
    MODEL=L python3 grid_rerank.py --lapor    # tabel + JSON untuk UI

Matcher yang belum punya bobot dilewati dengan status jelas, BUKAN diisi
angka perkiraan. Lihat `unduh_matcher.py`.
"""

import argparse
import json
import os
import time

import numpy as np

import protokol as P
import rerank as R

K = 20
MATCHER = ["xfeat", "sift", "aliked", "roma"]
KONDISI = list(P.KONDISI)
OUT = os.path.join(R.HASIL, "grid_rerank.json")


def sudah_lengkap(nama, k=K):
    p = os.path.join(R.HASIL, f"rerank_{nama}_k{k}.npy.progress")
    if not os.path.exists(p):
        return False
    kat = P.baca_katalog()
    _, qry = P.bangun_split(kat)
    return int(open(p).read()) >= len(qry)


def tugas():
    """Semua kombinasi yang belum selesai, matcher terblokir dilewati."""
    keluar = []
    for m in MATCHER:
        if m in R.MATCHER_TERBLOKIR and not R.bobot_ada(m):
            continue
        for k in KONDISI:
            nama = m if k == "raw" else f"{m}-{k}"
            if not sudah_lengkap(nama):
                keluar.append((m, k, nama))
    return keluar


def jalankan(budget=35.0):
    sisa = tugas()
    if not sisa:
        print("semua kombinasi sudah lengkap")
        return True
    m, k, nama = sisa[0]
    print(f"[{len(sisa)} tersisa] {nama}")
    matcher = R.buat_matcher(m, k)
    selesai, _, _ = R.jalankan(matcher, K, budget=budget)
    return selesai and len(sisa) == 1


def lapor():
    from statistik import bootstrap_delta, mcnemar
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    sp = np.array([r["species"] for r in qry])

    dasar = None
    baris = []
    lewat = []
    for m in MATCHER:
        if m in R.MATCHER_TERBLOKIR and not R.bobot_ada(m):
            lewat.append({"matcher": m, "alasan": R.MATCHER_TERBLOKIR[m]})
            continue
        for k in KONDISI:
            nama = m if k == "raw" else f"{m}-{k}"
            if not sudah_lengkap(nama):
                continue
            H, _, kk = R.evaluasi_rerank(nama, K)
            if dasar is None:
                dasar = H["stage1"]
                baris.append({
                    "matcher": "-", "kondisi": "-",
                    "label": "stage-1 saja (MegaDescriptor)",
                    "rank1": float(dasar["rank1"].mean() * 100),
                    "rank5": float(dasar["rank5"].mean() * 100),
                    "mAP": float(dasar["ap"].mean() * 100),
                    "hijau": float(dasar["rank1"][sp == "Green"].mean() * 100),
                    "sisik": float(dasar["rank1"][sp == "Hawksbill"].mean() * 100),
                })
            for mode in ("murni", "rrf"):
                h = H[mode]
                d = bootstrap_delta(dasar["rank1"].astype(float) * 100,
                                    h["rank1"].astype(float) * 100)
                mc = mcnemar(dasar["rank1"].astype(bool), h["rank1"].astype(bool))
                dm = bootstrap_delta(dasar["ap"] * 100, h["ap"] * 100)
                baris.append({
                    "matcher": m, "kondisi": k, "mode": mode,
                    "label": f"MegaDesc + {m.upper()}"
                             + ("" if k == "raw" else f" ({P.LABEL[k]})")
                             + f" · {mode}",
                    "rank1": float(h["rank1"].mean() * 100),
                    "rank5": float(h["rank5"].mean() * 100),
                    "mAP": float(h["ap"].mean() * 100),
                    "hijau": float(h["rank1"][sp == "Green"].mean() * 100),
                    "sisik": float(h["rank1"][sp == "Hawksbill"].mean() * 100),
                    "delta_rank1": d, "delta_mAP": dm, "mcnemar_rank1": mc,
                    "hijau_mcnemar": mcnemar(
                        dasar["rank1"][sp == "Green"].astype(bool),
                        h["rank1"][sp == "Green"].astype(bool)),
                    "sisik_mcnemar": mcnemar(
                        dasar["rank1"][sp == "Hawksbill"].astype(bool),
                        h["rank1"][sp == "Hawksbill"].astype(bool)),
                })

    hasil = {**P.metadata_run(kat),
             "k": K, "n": len(qry),
             "dibuat": time.strftime("%Y-%m-%d %H:%M"),
             "baris": baris, "belum_ada_bobot": lewat}
    with open(OUT, "w") as f:
        json.dump(hasil, f, indent=2)

    print(f"\nGrid stage-2  k={K}  n={len(qry)}  ({P.DATASET} / {P.MODEL})")
    print(f"{'Konfigurasi':44} {'R-1':>6} {'R-5':>6} {'mAP':>6} "
          f"{'hijau':>6} {'sisik':>6} {'ΔR-1':>8} {'p':>8}")
    for b in baris:
        if "delta_rank1" not in b:
            print(f"{b['label']:44} {b['rank1']:6.2f} {b['rank5']:6.2f} "
                  f"{b['mAP']:6.2f} {b['hijau']:6.2f} {b['sisik']:6.2f} "
                  f"{'—':>8} {'—':>8}")
            continue
        p = b["mcnemar_rank1"]["p_value"]
        print(f"{b['label']:44} {b['rank1']:6.2f} {b['rank5']:6.2f} "
              f"{b['mAP']:6.2f} {b['hijau']:6.2f} {b['sisik']:6.2f} "
              f"{b['delta_rank1']['delta']:+8.2f} {p:8.3f}"
              + ("*" if p < 0.05 else ""))
    if lewat:
        print("\nBELUM BISA DIJALANKAN (bobot tidak ada, angkanya TIDAK dikarang):")
        for x in lewat:
            print(f"  {x['matcher']:8} {x['alasan'][:88]}")
        print("  -> jalankan `unduh_matcher.py` di Mac untuk mengisinya")
    return hasil


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lapor", action="store_true")
    ap.add_argument("--budget", type=float, default=35.0)
    a = ap.parse_args()
    if a.lapor:
        lapor()
    else:
        if jalankan(a.budget):
            lapor()
