"""
BANDINGKAN DUA FOTO — apakah ini penyu yang sama, atau cuma spesies yang sama?

Pengenalan satu-lawan-banyak menyembunyikan kelemahan: sistem selalu punya
jawaban, karena ia hanya perlu memilih yang paling dekat dari galeri. Yang
tidak terlihat adalah apakah "paling dekat" itu benar-benar dekat.

Perbandingan satu-lawan-satu memaksa pertanyaan yang sebenarnya:
    "Apakah dua foto ini individu yang sama?" -> ya / tidak / ragu

Dua bukti ditampilkan berdampingan supaya bisa saling memeriksa:
    GLOBAL  jarak MegaDescriptor  — meringkas seluruh gambar jadi satu angka
    LOKAL   pasangan pola sisik   — menunjukkan BAGIAN MANA yang cocok

Kalau keduanya sepakat, keyakinannya tinggi. Kalau global bilang mirip tapi
tidak ada pasangan sisik yang cocok, itu tanda kemiripan datang dari latar
atau dari ciri spesies — bukan dari identitas individu.
"""

import os

import cv2
import numpy as np

import turtle_mode as tm

# Diukur pada 20 individu TurtleID2022 (MegaDescriptor-T, 7140 pasangan):
#     pasangan individu SAMA  jarak 0.3101 +/- 0.0988
#     pasangan individu BEDA  jarak 0.3588 +/- 0.0788
# Selisih rata-ratanya hanya 0.0487 sementara simpangannya ~0.09, jadi kedua
# distribusi hampir seluruhnya bertumpang tindih. Ambang di bawah ini adalah
# titik pisah terbaik yang bisa ditemukan, dan akurasi seimbangnya cuma 61%.
AMBANG = 0.290
SAMA_MEAN, SAMA_STD = 0.3101, 0.0988
BEDA_MEAN, BEDA_STD = 0.3588, 0.0788

# Di bawah ambang tapi selisihnya tipis -> jawab "ragu", jangan memaksa.
ZONA_RAGU = 0.03


def bandingkan(bgr_a, bgr_b, varian="T"):
    """Bandingkan dua foto. Return dict berisi kedua bukti dan putusannya."""
    from pipeline import Params
    import megadescriptor as md

    p = Params()
    pa, _ = tm.align(bgr_a, p)
    pb, _ = tm.align(bgr_b, p)

    va = md.deskriptor(pa, varian)
    vb = md.deskriptor(pb, varian)
    jarak = max(0.0, min(1.0, (1.0 - float(np.dot(va, vb))) / 2.0))

    sisik = None
    try:
        import pola_sisik
        if pola_sisik.tersedia():
            sisik = pola_sisik.cocokkan(pa, pb)
    except (ImportError, RuntimeError):
        pass

    # Probabilitas terkalibrasi dari distribusi terukur (turtle_mode.STATS,
    # ikut dataset aktif) — inilah angka yang layak ditunjukkan ke pengguna.
    prob = tm.prob_sama(jarak)
    if prob >= 0.65:
        putusan, warna = "INDIVIDU SAMA", (0, 220, 0)
    elif prob >= 0.45:
        putusan, warna = "RAGU", (0, 220, 220)
    else:
        putusan, warna = "INDIVIDU BERBEDA", (0, 140, 255)

    # Seberapa jauh jarak ini dari kedua distribusi, dalam satuan simpangan
    # baku. Angka ini memperlihatkan langsung bahwa kedua kemungkinan hampir
    # sama masuk akalnya — itulah inti masalahnya.
    z_sama = (jarak - SAMA_MEAN) / SAMA_STD
    z_beda = (jarak - BEDA_MEAN) / BEDA_STD

    return {"jarak": jarak, "prob": prob, "putusan": putusan, "warna": warna,
            "sisik": sisik, "patch_a": pa, "patch_b": pb,
            "z_sama": z_sama, "z_beda": z_beda,
            "ambang": AMBANG}


def gambar(hasil, label_a="Foto A", label_b="Foto B", kebenaran=None):
    """Susun tampilan perbandingan: dua foto, pasangan sisik, dan putusannya."""
    S = tm.PATCH
    atas = 84                    # ruang untuk putusan di bagian atas
    bawah = 92                   # ruang untuk rincian angka di bagian bawah
    kanvas = np.full((S + atas + bawah, S * 2 + 20, 3), 22, np.uint8)

    a = hasil["patch_a"].copy()
    b = hasil["patch_b"].copy()
    sisik = hasil["sisik"]

    # Garis antar pola sisik yang cocok, digambar melintasi kedua foto.
    if sisik and sisik["jumlah"]:
        sk = S / 384.0           # koordinat pola_sisik memakai IMG_SIZE=384
        for i in range(min(60, sisik["jumlah"])):
            pa = (sisik["titik_a"][i] * sk).astype(int)
            pb = (sisik["titik_b"][i] * sk).astype(int)
            warna = tuple(int(c) for c in cv2.applyColorMap(
                np.uint8([[int(255 * i / max(1, min(60, sisik["jumlah"])))]]),
                cv2.COLORMAP_HSV)[0][0])
            cv2.circle(a, tuple(pa), 3, warna, -1)
            cv2.circle(b, tuple(pb), 3, warna, -1)
            cv2.line(kanvas, (pa[0], pa[1] + atas),
                     (pb[0] + S + 20, pb[1] + atas), warna, 1, cv2.LINE_AA)

    kanvas[atas:atas + S, :S] = a
    kanvas[atas:atas + S, S + 20:] = b

    # Garis penghubung digambar sebelum foto ditempel, jadi ditimpa. Gambar
    # ulang di atas foto supaya pasangannya terlihat menyeberang.
    if sisik and sisik["jumlah"]:
        sk = S / 384.0
        for i in range(min(60, sisik["jumlah"])):
            pa = (sisik["titik_a"][i] * sk).astype(int)
            pb = (sisik["titik_b"][i] * sk).astype(int)
            warna = tuple(int(c) for c in cv2.applyColorMap(
                np.uint8([[int(255 * i / max(1, min(60, sisik["jumlah"])))]]),
                cv2.COLORMAP_HSV)[0][0])
            cv2.line(kanvas, (pa[0], pa[1] + atas),
                     (pb[0] + S + 20, pb[1] + atas), warna, 1, cv2.LINE_AA)

    F = cv2.FONT_HERSHEY_SIMPLEX
    pr = hasil.get("prob", 0.5)
    cv2.putText(kanvas, f"{hasil['putusan']}  -  {pr:.0%}", (14, 34), F, 0.9,
                hasil["warna"], 2)
    cv2.putText(kanvas, f"P(individu sama) {pr:.1%}   jarak {hasil['jarak']:.4f}",
                (14, 62), F, 0.55, (210, 210, 220), 1)

    if kebenaran is not None:
        benar = (kebenaran == (hasil["putusan"] == "INDIVIDU SAMA"))
        teks = "SEHARUSNYA SAMA" if kebenaran else "SEHARUSNYA BEDA"
        cv2.putText(kanvas, f"{teks}  {'BENAR' if benar else 'SALAH'}",
                    (S + 30, 34), F, 0.7,
                    (0, 220, 0) if benar else (60, 60, 255), 2)

    cv2.putText(kanvas, label_a, (14, atas + S + 24), F, 0.55, (200, 200, 210), 1)
    cv2.putText(kanvas, label_b, (S + 34, atas + S + 24), F, 0.55, (200, 200, 210), 1)

    y = atas + S + 52
    n = sisik["jumlah"] if sisik else 0
    cv2.putText(kanvas, f"GLOBAL  MegaDescriptor  jarak {hasil['jarak']:.4f}",
                (14, y), F, 0.5, (170, 200, 255), 1)
    cv2.putText(kanvas, f"LOKAL   pola sisik cocok  {n} pasang",
                (14, y + 22), F, 0.5,
                (170, 255, 170) if n else (140, 140, 150), 1)
    cv2.putText(kanvas,
                f"jarak ini {hasil['z_sama']:+.2f} SD dari rata2 SAMA, "
                f"{hasil['z_beda']:+.2f} SD dari rata2 BEDA",
                (S + 30, y), F, 0.45, (190, 190, 130), 1)
    cv2.putText(kanvas,
                "kedua distribusi bertumpang tindih - lihat catatan modul",
                (S + 30, y + 22), F, 0.45, (150, 150, 160), 1)
    return kanvas


def pasangan_dataset(sama=True, seed=None):
    """Ambil sepasang foto dari dataset: individu sama atau berbeda.

    Return (path_a, path_b, label_a, label_b, kebenaran).
    """
    rng = np.random.default_rng(seed)
    root = tm.GALERI_DEFAULT
    if not os.path.isdir(root):
        return None
    inds = [d for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]

    def fotos(i):
        d = os.path.join(root, i)
        fs = []
        for akar, _, ns in os.walk(d):
            fs += [os.path.join(akar, n) for n in sorted(ns)
                   if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        return fs

    if sama:
        kandidat = [i for i in inds if len(fotos(i)) >= 2]
        if not kandidat:
            return None
        i = kandidat[rng.integers(len(kandidat))]
        fs = fotos(i)
        a, b = rng.choice(len(fs), 2, replace=False)
        return fs[a], fs[b], i, i, True

    if len(inds) < 2:
        return None
    i, j = rng.choice(len(inds), 2, replace=False)
    fa, fb = fotos(inds[i]), fotos(inds[j])
    if not fa or not fb:
        return None
    return (fa[rng.integers(len(fa))], fb[rng.integers(len(fb))],
            inds[i], inds[j], False)
