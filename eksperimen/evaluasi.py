"""
Evaluasi satu/semua kondisi + validasi metrik manual vs wildlife-tools.

    python3 evaluasi.py                 # semua kondisi yang embedding-nya lengkap
    python3 evaluasi.py --validasi      # manual vs wildlife-tools, checkpoint §4
"""

import json
import os
import sys

import numpy as np

import protokol as P

HASIL = P.dir_hasil()


def muat(nama, gal, qry):
    E = np.load(os.path.join(HASIL, f"emb_{nama}.npy"))
    prog = int(open(os.path.join(HASIL, f"emb_{nama}.progress")).read())
    if prog != len(gal) + len(qry):
        raise RuntimeError(f"{nama} belum lengkap: {prog}/{len(gal)+len(qry)}")
    return E[:len(gal)], E[len(gal):]


def evaluasi(nama, gal, qry):
    Eg, Eq = muat(nama, gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])
    return P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)


def _potong(hasil, mask):
    return {"rank1": float(hasil["rank1"][mask].mean() * 100),
            "rank5": float(hasil["rank5"][mask].mean() * 100),
            "mAP": float(hasil["ap"][mask].mean() * 100),
            "n": int(mask.sum())}


def breakdown(hasil, qry):
    """Pecah per sisi dan per spesies (§4).

    Spesies hanya ada di ReunionTurtles; di SeaTurtleIDHeads dict-nya kosong,
    bukan diisi angka palsu.
    """
    sisi = np.array([r["side"] for r in qry])
    spes = np.array([r.get("species") or "" for r in qry])
    return {
        "per_sisi": {s: _potong(hasil, sisi == s) for s in P.SISI},
        "per_spesies": {sp: _potong(hasil, spes == sp)
                        for sp in sorted(set(spes) - {""})},
    }


# ------------------------------------------------ validasi metrik (§4)
def validasi_wildlife_tools(gal, qry):
    """Bandingkan metrik manual dengan evaluasi bawaan wildlife-tools.

    wildlife-tools memakai KnnClassifier + CosineSimilarity untuk rank-k.
    Kunci sisi diterapkan dengan cara yang sama (similarity beda-sisi = -inf),
    kalau tidak, dua angka ini memang tidak boleh dibandingkan.
    """
    try:
        import torch
        from wildlife_tools.inference import KnnClassifier
        from wildlife_tools.similarity.cosine import CosineSimilarity
    except ImportError as e:
        return {"status": "TERBLOKIR", "alasan": f"wildlife-tools tidak terpasang: {e}"}

    Eg, Eq = muat("raw", gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    # Matriks similarity dari wildlife-tools, bukan dari kode kita.
    S = np.asarray(CosineSimilarity()(query=Eq, database=Eg)["cosine"],
                   dtype=np.float64).copy()
    S[s_q[:, None] != s_g[None, :]] = -np.inf   # kunci sisi, sama seperti manual

    # rank-1 lewat KnnClassifier bawaan (k=1)
    pred1 = KnnClassifier(k=1, database_labels=id_g)(S)
    r1_wt = float((pred1 == id_q).mean() * 100)

    # rank-5: top-5 dari matriks yang sama
    top5 = torch.tensor(S).topk(k=5, dim=1).indices.numpy()
    r5_wt = float((id_g[top5] == id_q[:, None]).any(1).mean() * 100)

    man = P.ringkas(evaluasi("raw", gal, qry))
    d1, d5 = abs(r1_wt - man["rank1"]), abs(r5_wt - man["rank5"])
    return {"status": "OK",
            "wildlife_tools": {"rank1": r1_wt, "rank5": r5_wt,
                               "sumber": "CosineSimilarity + KnnClassifier(k=1)"},
            "manual": {"rank1": man["rank1"], "rank5": man["rank5"]},
            "selisih_rank1": d1, "selisih_rank5": d5,
            "cocok": bool(d1 < 1e-9 and d5 < 1e-9),
            "catatan_mAP": ("wildlife-tools 0.0.9 tidak menyediakan mAP siap "
                            "pakai; mAP hanya dari implementasi manual.")}


def main():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)

    if "--validasi" in sys.argv:
        r = validasi_wildlife_tools(gal, qry)
        print(json.dumps(r, indent=2))
        with open(os.path.join(HASIL, "validasi_metrik.json"), "w") as f:
            json.dump(r, f, indent=2)
        return

    ringkasan, mentah = {}, {}
    for k in P.KONDISI:
        try:
            h = evaluasi(k, gal, qry)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"{k:9} dilewati — {e}")
            continue
        ringkasan[k] = P.ringkas(h) | breakdown(h, qry)
        mentah[k] = {"rank1": h["rank1"].astype(np.int8), "ap": h["ap"]}
        r = ringkasan[k]
        print(f"{k:9} R1 {r['rank1']:5.2f}  R5 {r['rank5']:5.2f}  "
              f"mAP {r['mAP']:5.2f}  (n={r['n']})")

    with open(os.path.join(HASIL, "ringkasan.json"), "w") as f:
        json.dump(ringkasan, f, indent=2)
    np.savez(os.path.join(HASIL, "mentah.npz"),
             **{f"{k}_{m}": v for k, d in mentah.items() for m, v in d.items()})


if __name__ == "__main__":
    main()
