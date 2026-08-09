"""
Kumpulkan kasus gagal (§9): query yang rank-1-nya salah, beserta top-5 gallery.

Dipilih yang informatif, bukan yang acak:
  - 'yakin_tapi_salah'  : similarity top-1 tinggi tapi identitas salah
  - 'jawaban_jauh'      : jawaban benar ada tapi peringkatnya jauh
  - 'nyaris'            : jawaban benar di peringkat 2

    python3 kasus_gagal.py
"""

import json
import os

import numpy as np

import protokol as P
from evaluasi import muat

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")


def kumpulkan(kondisi="raw", n_per_jenis=2):
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    Eg, Eq = muat(kondisi, gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    S = Eq @ Eg.T
    S[s_q[:, None] != s_g[None, :]] = -np.inf
    urut = np.argsort(-S, axis=1)

    kasus = []
    for i in range(len(qry)):
        top = urut[i][np.isfinite(S[i, urut[i]])]
        if len(top) == 0 or id_g[top[0]] == id_q[i]:
            continue
        benar = np.flatnonzero(id_g[top] == id_q[i])
        peringkat_benar = int(benar[0]) + 1 if len(benar) else None
        kasus.append({
            "query_path": os.path.relpath(qry[i]["path"], P.REPO),
            "query_id": qry[i]["identity"],
            "query_side": qry[i]["side"],
            "query_year": qry[i]["year"],
            "skor_top1": float(S[i, top[0]]),
            "margin_top1_top2": float(S[i, top[0]] - S[i, top[1]]) if len(top) > 1 else None,
            "peringkat_jawaban_benar": peringkat_benar,
            "top5": [{"path": os.path.relpath(gal[j]["path"], P.REPO),
                      "id": gal[j]["identity"], "year": gal[j]["year"],
                      "skor": float(S[i, j]),
                      "benar": bool(id_g[j] == id_q[i])} for j in top[:5]],
        })

    yakin = sorted(kasus, key=lambda k: -k["skor_top1"])[:n_per_jenis]
    jauh = sorted([k for k in kasus if k["peringkat_jawaban_benar"]],
                  key=lambda k: -k["peringkat_jawaban_benar"])[:n_per_jenis]
    nyaris = [k for k in kasus if k["peringkat_jawaban_benar"] == 2][:n_per_jenis]

    pilih, seen = [], set()
    for jenis, grup in (("yakin_tapi_salah", yakin), ("jawaban_jauh", jauh),
                        ("nyaris_peringkat_2", nyaris)):
        for k in grup:
            if k["query_path"] in seen:
                continue
            seen.add(k["query_path"])
            pilih.append(dict(k, jenis=jenis))

    return {"kondisi": kondisi, "total_gagal": len(kasus),
            "total_query": len(qry), "contoh": pilih[:5]}


if __name__ == "__main__":
    r = kumpulkan()
    with open(os.path.join(HASIL, "kasus_gagal.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(f"{r['total_gagal']}/{r['total_query']} query gagal di rank-1")
    for k in r["contoh"]:
        print(f"  [{k['jenis']:18}] {k['query_id']} {k['query_side']:5} "
              f"{k['query_year']}  top1={k['top5'][0]['id']} "
              f"skor={k['skor_top1']:.3f} "
              f"jawaban_benar_di_peringkat={k['peringkat_jawaban_benar']}")
