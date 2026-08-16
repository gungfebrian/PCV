"""
Sanity check §8 — dijalankan SEBELUM menyalahkan preprocessing.

Baseline rank-1 di bawah 50% wajib memicu skrip ini. Tiap dugaan diuji dengan
angka, bukan dengan keyakinan.

    python3 sanity.py
"""

import json
import os

import numpy as np

import protokol as P
from evaluasi import muat

HASIL = P.AKAR_HASIL


def _ev(Eq, Eg, id_q, id_g, s_q, s_g):
    return P.ringkas(P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g))


def main():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    Eg, Eq = muat("raw", gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    out = {}

    # 3. Embedding sudah L2-normalized?
    out["norma_embedding"] = {
        "gallery_min": float(np.linalg.norm(Eg, axis=1).min()),
        "gallery_max": float(np.linalg.norm(Eg, axis=1).max()),
        "query_min": float(np.linalg.norm(Eq, axis=1).min()),
        "query_max": float(np.linalg.norm(Eq, axis=1).max()),
    }

    # 1. Kebocoran sisi: kalau sisi DIBUKA, angkanya harus TURUN.
    #    Kalau justru naik, berarti pemisahan sisi yang salah, bukan lawannya.
    terkunci = _ev(Eq, Eg, id_q, id_g, s_q, s_g)
    bocor = _ev(Eq, Eg, id_q, id_g,
                np.zeros(len(s_q), int), np.zeros(len(s_g), int))
    out["sisi"] = {"terkunci": terkunci, "dibuka_bocor": bocor,
                   "delta_rank1_kalau_bocor": bocor["rank1"] - terkunci["rank1"]}

    # 2. Arah split: tukar gallery<->query. Kalau hasilnya mirip, arah tidak
    #    menentukan; kalau jauh lebih baik, split kita terbalik.
    balik = _ev(Eg, Eq, id_g, id_q, s_g, s_q)
    out["arah_split"] = {"normal_g1_q2": terkunci, "dibalik_g2_q1": balik,
                         "delta_rank1": balik["rank1"] - terkunci["rank1"]}

    # 4. Transform input: kanonik (bicubic + center crop 0.9) vs cara lama di
    #    repo (INTER_AREA langsung ke 224x224, tanpa center crop).
    out["transform"] = bandingkan_transform(gal, qry, id_g, id_q, s_g, s_q)

    # 5. Query tanpa pasangan di gallery
    out["split"] = P.periksa_split(gal, qry)

    # Bonus: berapa banyak gallery identity yang tidak pernah jadi jawaban
    # (distraktor). Makin banyak, makin sulit — ini menjelaskan angka rendah.
    out["distraktor"] = {
        "identitas_gallery": int(len(set(id_g))),
        "identitas_query": int(len(set(id_q))),
        "identitas_gallery_yang_tak_pernah_benar":
            int(len(set(id_g) - set(id_q))),
    }

    print(json.dumps(out, indent=2))
    with open(os.path.join(HASIL, "sanity.json"), "w") as f:
        json.dump(out, f, indent=2)


def bandingkan_transform(gal, qry, id_g, id_q, s_g, s_q):
    """Hitung ulang embedding sampel kecil dengan transform lama vs kanonik."""
    import cv2
    import torch

    model, _ = P.muat_model()
    torch.set_num_threads(4)

    def lama(rgb):
        rgb = cv2.resize(rgb, (P.UKURAN, P.UKURAN), interpolation=cv2.INTER_AREA)
        f = rgb.astype(np.float32) / 255.0
        return ((f - P.MEAN) / P.STD).transpose(2, 0, 1)

    def jalankan(tf, paths):
        keluar = []
        for i in range(0, len(paths), 16):
            xs = [tf(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                  for p in paths[i:i + 16]]
            with torch.no_grad():
                keluar.append(model(torch.from_numpy(np.stack(xs))).float().numpy())
        E = np.concatenate(keluar)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    # sampel: 300 query pertama + seluruh gallery sisi yang sama akan mahal,
    # jadi dipakai subset gallery 600 foto. Angka absolutnya tidak dipakai —
    # yang dipakai hanya SELISIH antara dua transform pada subset yang sama.
    qi = np.arange(0, len(qry), max(1, len(qry) // 300))[:300]
    gi = np.arange(0, len(gal), max(1, len(gal) // 600))[:600]
    pq = [qry[i]["path"] for i in qi]
    pg = [gal[i]["path"] for i in gi]

    hasil = {}
    for nama, tf in (("kanonik_bicubic_crop0.9", P.transform_kanonik),
                     ("lama_INTER_AREA_squash", lama)):
        Eq, Eg = jalankan(tf, pq), jalankan(tf, pg)
        hasil[nama] = P.ringkas(P.evaluasi_manual(
            Eq, Eg, id_q[qi], id_g[gi], s_q[qi], s_g[gi]))
    hasil["delta_rank1_kanonik_minus_lama"] = (
        hasil["kanonik_bicubic_crop0.9"]["rank1"]
        - hasil["lama_INTER_AREA_squash"]["rank1"])
    hasil["catatan"] = ("subset 300 query x 600 gallery — angka absolut tidak "
                        "sebanding dengan tabel utama, hanya selisihnya yang dipakai")
    return hasil


if __name__ == "__main__":
    main()
