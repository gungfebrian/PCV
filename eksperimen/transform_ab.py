"""
A/B transform input — sanity check §8 butir 4, di seluruh himpunan query.

Pertanyaan: "ukuran input & normalisasi sesuai ekspektasi model" itu artinya
apa persisnya? config.json MegaDescriptor menyebut crop_pct 0.9 + bicubic +
center crop. Tapi crop_pct dirancang untuk foto pemandangan penuh, sedangkan
SeaTurtleIDHeads sudah berupa crop kepala yang ketat — center crop 0.9 di atas
crop yang sudah ketat berarti membuang sisik di tepi.

Jadi "sesuai config" belum tentu "benar untuk data ini". Diuji, bukan diasumsikan.

    python3 transform_ab.py          # resumable, panggil sampai selesai
    python3 transform_ab.py --uji    # McNemar + bootstrap setelah lengkap
"""

import json
import os
import sys
import time

import cv2
import numpy as np

import protokol as P

HASIL = P.AKAR_HASIL
OUT = os.path.join(HASIL, "emb_raw_squash.npy")
PROG = OUT + ".progress"


def squash(rgb):
    """Cara lama di repo: INTER_AREA langsung ke 224x224, tanpa center crop.
    Rasio aspek dirusak, tapi tidak ada piksel yang dibuang."""
    rgb = cv2.resize(rgb, (P.UKURAN, P.UKURAN), interpolation=cv2.INTER_AREA)
    return ((rgb.astype(np.float32) / 255.0 - P.MEAN) / P.STD).transpose(2, 0, 1)


def hitung(budget=32):
    import torch
    model, _ = P.muat_model()
    torch.set_num_threads(4)
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    paths = [r["path"] for r in gal] + [r["path"] for r in qry]
    E = np.load(OUT) if os.path.exists(OUT) else np.zeros((len(paths), P.DIM), np.float32)
    d = int(open(PROG).read()) if os.path.exists(PROG) else 0
    t0 = time.time()
    while d < len(paths) and time.time() - t0 < budget:
        xs = [squash(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
              for p in paths[d:d + 32]]
        with torch.no_grad():
            v = model(torch.from_numpy(np.stack(xs))).float().numpy()
        E[d:d + len(xs)] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        d += len(xs)
        np.save(OUT, E)
        open(PROG, "w").write(str(d))
    print(f"raw_squash {d}/{len(paths)}")
    return d == len(paths)


def uji():
    from evaluasi import evaluasi
    from statistik import bootstrap_delta, mcnemar
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    E = np.load(OUT)
    Eg, Eq = E[:len(gal)], E[len(gal):]
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    sq = P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)
    kn = evaluasi("raw", gal, qry)
    r = {
        "kanonik_bicubic_crop0.9": P.ringkas(kn),
        "squash_INTER_AREA_224": P.ringkas(sq),
        "delta_rank1_squash_minus_kanonik": bootstrap_delta(
            kn["rank1"].astype(float) * 100, sq["rank1"].astype(float) * 100),
        "mcnemar": mcnemar(kn["rank1"].astype(bool), sq["rank1"].astype(bool)),
        "delta_mAP_squash_minus_kanonik": bootstrap_delta(kn["ap"] * 100, sq["ap"] * 100),
    }
    print(json.dumps(r, indent=2))
    with open(os.path.join(HASIL, "transform_ab.json"), "w") as f:
        json.dump(r, f, indent=2)


if __name__ == "__main__":
    if "--uji" in sys.argv:
        uji()
    else:
        hitung()
