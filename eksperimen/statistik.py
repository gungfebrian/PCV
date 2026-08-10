"""
Uji berpasangan §5 — McNemar untuk rank-1, bootstrap CI untuk selisih mAP.

Himpunan query identik di semua kondisi, jadi perbandingan WAJIB berpasangan.
Uji tak-berpasangan (mis. two-proportion z-test) akan melebih-lebihkan
ketidakpastian dan menyembunyikan efek yang nyata.

    python3 statistik.py
"""

import json
import os

import numpy as np

import protokol as P
from evaluasi import breakdown, evaluasi

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")
BASELINE = "raw"
B_BOOT = 10000
SEED_BOOT = 0      # hanya untuk resampling bootstrap, BUKAN untuk split data


def mcnemar(a, b):
    """a, b = vektor boolean benar/salah pada query yang SAMA.

    Dipakai versi exact binomial, bukan chi-square: jumlah discordant di sini
    bisa kecil, dan chi-square tidak akurat di ekor.
    """
    from scipy.stats import binomtest
    n01 = int(np.sum(~a & b))      # baseline salah, kondisi benar
    n10 = int(np.sum(a & ~b))      # baseline benar, kondisi salah
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return {"n01_baseline_salah_kondisi_benar": n01,
            "n10_baseline_benar_kondisi_salah": n10,
            "n_discordant": n, "p_value": float(p)}


def bootstrap_delta(x, y, B=B_BOOT, seed=SEED_BOOT):
    """CI 95% untuk mean(y) - mean(x) dengan resampling query BERPASANGAN.

    Indeks yang sama dipakai untuk x dan y di tiap replikasi — kalau tidak,
    korelasi antar kondisi hilang dan CI-nya jadi terlalu lebar.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    d = (y[idx] - x[idx]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float((y - x).mean()), "ci95": [float(lo), float(hi)],
            "signifikan": bool(lo > 0 or hi < 0)}


def main():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    sisi = np.array([r["side"] for r in qry])

    H = {}
    for k in P.KONDISI:
        try:
            H[k] = evaluasi(k, gal, qry)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"{k}: dilewati — {e}")

    base = H[BASELINE]
    tabel = {}
    for k, h in H.items():
        baris = {
            "label": P.LABEL[k],
            "rank1": float(h["rank1"].mean() * 100),
            "rank5": float(h["rank5"].mean() * 100),
            "mAP": float(h["ap"].mean() * 100),
            "n": int(len(h["ap"])),
        }
        if k != BASELINE:
            baris["delta_rank1"] = bootstrap_delta(
                base["rank1"].astype(float) * 100, h["rank1"].astype(float) * 100)
            baris["mcnemar_rank1"] = mcnemar(base["rank1"].astype(bool),
                                             h["rank1"].astype(bool))
            baris["delta_mAP"] = bootstrap_delta(base["ap"] * 100, h["ap"] * 100)
        baris.update(breakdown(h, qry))
        tabel[k] = baris

    with open(os.path.join(HASIL, "statistik.json"), "w") as f:
        json.dump(tabel, f, indent=2)

    # ---- cetak tabel utama (§9)
    print(f"{'Kondisi':28} {'R-1':>6} {'R-5':>6} {'mAP':>6} "
          f"{'ΔR-1 vs raw (95% CI)':>26} {'p (McNemar)':>12}")
    for k, b in tabel.items():
        if k == BASELINE:
            print(f"{b['label']:28} {b['rank1']:6.2f} {b['rank5']:6.2f} "
                  f"{b['mAP']:6.2f} {'—':>26} {'—':>12}")
        else:
            d = b["delta_rank1"]
            ci = f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
            print(f"{b['label']:28} {b['rank1']:6.2f} {b['rank5']:6.2f} "
                  f"{b['mAP']:6.2f} {ci:>26} "
                  f"{b['mcnemar_rank1']['p_value']:12.3g}")

    print("\nΔ mAP vs raw (bootstrap 95% CI, resampling query berpasangan)")
    for k, b in tabel.items():
        if k == BASELINE:
            continue
        d = b["delta_mAP"]
        print(f"  {b['label']:28} {d['delta']:+6.2f} "
              f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  "
              f"{'signifikan' if d['signifikan'] else 'TIDAK signifikan'}")

    spesies = sorted(tabel[BASELINE]["per_spesies"])
    print("\nBreakdown Rank-1 per sisi & per spesies")
    kepala = f"{'Kondisi':28} {'kiri':>8} {'kanan':>8}"
    for sp in spesies:
        kepala += f" {sp:>10}"
    print(kepala)
    for k, b in tabel.items():
        baris = (f"{b['label']:28} {b['per_sisi']['left']['rank1']:8.2f} "
                 f"{b['per_sisi']['right']['rank1']:8.2f}")
        for sp in spesies:
            baris += f" {b['per_spesies'][sp]['rank1']:10.2f}"
        print(baris)
    if not spesies:
        print("  (spesies tidak tersedia di dataset ini)")


if __name__ == "__main__":
    main()
