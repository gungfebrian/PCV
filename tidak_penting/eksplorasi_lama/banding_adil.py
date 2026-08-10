"""
Bandingkan MegaDescriptor-T generik vs ArcFace fine-tuned, protokol IDENTIK.

Kenapa skrip terpisah: evaluasi bawaan latih_arcface.py memakai 164 individu
galeri, sedangkan angka pembanding 60.6% di Notion memakai 20 individu.
Makin banyak individu di galeri, makin sulit tugasnya — jadi kedua angka itu
TIDAK sebanding, dan membandingkannya langsung akan menyesatkan.

Di sini keduanya diuji pada daftar individu, galeri, dan foto uji yang sama
persis. Split tetap time-aware: galeri dari tahun-tahun awal, uji hanya dari
tahun terakhir yang tidak pernah dilihat model saat latihan.

Jalankan:
    .venv/bin/python banding_adil.py            # 20 individu (sebanding Notion)
    .venv/bin/python banding_adil.py --n 164    # skala penuh
"""

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "dataset_penyu", "SeaTurtleIDHeads")


def bagi_time_aware(n_individu):
    """Galeri dari tahun awal, uji dari tahun terakhir. Sama dengan protokol
    latih_arcface.py supaya foto uji tidak pernah dilihat model."""
    with open(os.path.join(DATA, "annotations.json")) as f:
        anot = json.load(f)

    byind = {}
    for im in anot["images"]:
        ind = im["path"].split("/")[1]
        try:
            d = datetime.strptime(im["date"].split()[0], "%Y:%m:%d")
        except (ValueError, KeyError):
            continue
        p = os.path.join(DATA, im["path"])
        if os.path.exists(p):
            byind.setdefault(ind, []).append((d, p))

    gal, uji = {}, []
    for ind in sorted(byind):
        lst = sorted(byind[ind])
        tahun = sorted({d.year for d, _ in lst})
        if len(tahun) < 2 or len(lst) < 8:
            continue
        th_uji = tahun[-1]
        awal = [p for d, p in lst if d.year != th_uji]
        akhir = [p for d, p in lst if d.year == th_uji]
        if len(awal) < 4 or not akhir:
            continue
        gal[ind] = awal[:4]
        uji += [(p, ind) for p in akhir[:10]]
        if len(gal) >= n_individu:
            break
    # buang foto uji milik individu yang tidak masuk galeri
    uji = [(p, k) for p, k in uji if k in gal]
    return gal, uji


def ukur(varian, gal_paths, uji):
    import megadescriptor as md
    import turtle_mode as tm
    from pipeline import Params

    p = Params()
    tm.set_crop(False)
    tm.set_masking(False)
    md.muat(varian)

    def emb(path):
        im = cv2.imread(path)
        return md.deskriptor(tm.align(im, p)[0], varian) if im is not None else None

    gal = {k: [v for v in (emb(q) for q in ps) if v is not None]
           for k, ps in gal_paths.items()}

    b = t5 = n = 0
    sama, beda = [], []
    for q, truth in uji:
        v = emb(q)
        if v is None:
            continue
        rk = sorted(((k, (1 - max(float(np.dot(v, w)) for w in vs)) / 2)
                     for k, vs in gal.items()), key=lambda x: x[1])
        b += rk[0][0] == truth
        t5 += truth in [r[0] for r in rk[:5]]
        n += 1
        for k, d in rk:
            (sama if k == truth else beda).append(d)

    sama, beda = np.array(sama), np.array(beda)
    return {"top1": b / n * 100, "top5": t5 / n * 100, "n": n,
            "sama": (sama.mean(), sama.std()),
            "beda": (beda.mean(), beda.std()),
            "pisah": beda.mean() - sama.mean(),
            "tumpang": (sama > beda.mean()).mean() * 100}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    gal, uji = bagi_time_aware(a.n)
    print(f"protokol time-aware: {len(gal)} individu galeri, {len(uji)} foto uji\n")

    hasil = {}
    for varian, nama in (("T", "MegaDescriptor-T generik"),
                         ("arcface", "ArcFace fine-tuned")):
        try:
            hasil[varian] = ukur(varian, gal, uji)
        except FileNotFoundError as e:
            print(f"{nama}: dilewati ({e})")
            continue
        r = hasil[varian]
        print(f"{nama:26} Top-1 {r['top1']:5.1f}%  Top-5 {r['top5']:5.1f}%")
        print(f"{'':26} SAMA {r['sama'][0]:.4f}±{r['sama'][1]:.4f}  "
              f"BEDA {r['beda'][0]:.4f}±{r['beda'][1]:.4f}")
        print(f"{'':26} pemisahan {r['pisah']:+.4f}  "
              f"tumpang tindih {r['tumpang']:.0f}%\n")

    if len(hasil) == 2:
        d1 = hasil["arcface"]["top1"] - hasil["T"]["top1"]
        dp = hasil["arcface"]["pisah"] - hasil["T"]["pisah"]
        print(f"SELISIH ArcFace vs generik: Top-1 {d1:+.1f} poin, "
              f"pemisahan {dp:+.4f}")
