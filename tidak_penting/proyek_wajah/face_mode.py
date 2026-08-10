"""
Mode MUKA MANUSIA — Face ID untuk wajah, memakai model bawaan OpenCV.

Beda dengan mode penyu, di sini kita pakai model yang memang dirancang untuk
wajah, bukan untuk satwa:

    DETECT   YuNet  — detektor wajah ringan, sekaligus memberi 5 titik landmark
    ALIGN    alignCrop() — memutar & memotong wajah berdasarkan posisi mata,
             sehingga wajah miring tetap jadi tegak
    DESCRIBE SFace — embedding 128 dimensi khusus wajah
    MATCH    cosine ke galeri terdaftar

Langkah ALIGN inilah yang tidak dimiliki pipeline penyu kita. Karena posisi
mata diketahui, wajah bisa diputar ke posisi baku. Itu salah satu alasan
pengenalan wajah jauh lebih akurat daripada re-ID penyu kita sekarang.

Model diunduh oleh unduh_model_wajah.py ke models_wajah/.
"""

import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODEL = os.path.join(BASE_DIR, "models_wajah")
YUNET = os.path.join(DIR_MODEL, "yunet.onnx")
SFACE = os.path.join(DIR_MODEL, "sface.onnx")

# SFace: dua wajah dianggap orang yang sama kalau cosine similarity >= 0.363
# (angka resmi dokumentasi OpenCV).
#
# PENTING soal satuan: Galeri.kenali() memakai jarak = (1 - similarity) / 2,
# bukan (1 - similarity). Konversinya harus ikut pembagian 2 itu. Versi awal
# memakai 1 - 0.363 = 0.637 dan itu SALAH — dua kali terlalu longgar, sehingga
# wajah dengan similarity -0.02 (praktis tidak mirip sama sekali) tetap
# diterima dan tiga orang berbeda dikenali sebagai nama yang sama.
AMBANG_SFACE = (1.0 - 0.363) / 2.0        # = 0.3185

# Gerbang kualitas. Wajah kecil atau berskor rendah menghasilkan embedding yang
# tidak stabil — pada pengujian, wajah 83x139 piksel di pinggir frame salah
# dikenali sebagai orang lain dengan jarak 0.40 (jauh di bawah ambang). Jadi
# wajah seperti itu tetap digambar kotaknya, tapi tidak dipakai untuk mengenali
# maupun mendaftarkan.
# Diturunkan setelah pengujian: dengan 110px/0.85, wajah kedua dalam frame
# (skor 0.78) ikut tertolak sehingga hanya satu orang yang pernah dikenali.
# Angka sekarang masih membuang crop yang benar-benar tidak layak — wajah
# 83x139px yang dulu salah dikenali tetap tertolak oleh syarat ukuran.
MIN_UKURAN = 90       # sisi terpendek kotak wajah, piksel
MIN_SKOR = 0.75

# Kandidat teratas harus lebih dekat sejauh ini dari kandidat kedua sebelum
# diakui. Kalau dua orang sama-sama dekat, sistem menjawab "ragu" — itu jawaban
# yang benar, dan jauh lebih baik daripada menyebut nama yang salah.
MIN_MARGIN = 0.06

_detektor = None
_pengenal = None


def tersedia():
    return os.path.exists(YUNET) and os.path.exists(SFACE)


def _muat():
    global _detektor, _pengenal
    if _detektor is None:
        if not tersedia():
            raise FileNotFoundError(
                "Model wajah belum ada. Jalankan: "
                ".venv/bin/python unduh_model_wajah.py")
        # Ukuran input diatur ulang tiap frame lewat setInputSize().
        _detektor = cv2.FaceDetectorYN.create(YUNET, "", (320, 320),
                                              score_threshold=0.7)
        _pengenal = cv2.FaceRecognizerSF.create(SFACE, "")
    return _detektor, _pengenal


def deteksi(bgr):
    """Return array wajah (N x 15): x,y,w,h, 5 landmark, skor."""
    det, _ = _muat()
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, wajah = det.detect(bgr)
    return wajah if wajah is not None else np.empty((0, 15), np.float32)


def deskriptor(bgr, wajah_row):
    """ALIGN + DESCRIBE: luruskan wajah lalu ubah jadi embedding 128-dim."""
    _, rec = _muat()
    aligned = rec.alignCrop(bgr, wajah_row)
    v = rec.feature(aligned).flatten().astype(np.float32)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v), aligned


def jalankan(frame, faceid, params=None):
    # params tidak dipakai: YuNet punya ambangnya sendiri dan tidak
    # bergantung pada slider Canny/blur milik pipeline kartu & penyu.
    # Tetap diterima agar tanda tangannya senada dengan mode lain.
    """Pipeline wajah. Tanda tangan senada dengan mode lain."""
    from pipeline import Stage

    stages = []
    h, w = frame.shape[:2]

    if not tersedia():
        kosong = np.zeros((300, 500, 3), np.uint8)
        cv2.putText(kosong, "Model wajah belum diunduh", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        stages.append(Stage("asli", "Model Belum Ada", kosong, "",
                            "Jalankan: .venv/bin/python unduh_model_wajah.py"))
        return stages, None

    wajah = deteksi(frame)

    # --- tahap 1: frame dengan kotak + identitas
    vis = frame.copy()
    hasil_utama = None
    vek_utama = None
    aligned_utama = None
    semua = []          # hasil pengenalan untuk SETIAP wajah yang lolos

    # Urutkan dari yang terbesar: subjek utama hampir selalu wajah terdekat,
    # bukan wajah pertama yang kebetulan ditemukan detektor.
    if len(wajah):
        wajah = wajah[np.argsort(-(wajah[:, 2] * wajah[:, 3]))]

    for i, f in enumerate(wajah):
        x, y, bw, bh = (int(v) for v in f[:4])

        if min(bw, bh) < MIN_UKURAN or f[-1] < MIN_SKOR:
            # Terlalu kecil/ragu untuk dipercaya: tetap digambar supaya terlihat
            # bahwa wajahnya TERDETEKSI, cuma tidak dipakai untuk mengenali.
            alasan = (f"terlalu kecil {min(bw, bh)}px" if min(bw, bh) < MIN_UKURAN
                      else f"kurang jelas {f[-1]:.2f}")
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (120, 120, 120), 2)
            cv2.putText(vis, alasan, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 2)
            continue

        try:
            vek, aligned = deskriptor(frame, f)
        except cv2.error:
            continue

        r = faceid.kenali(vek, ambang=AMBANG_SFACE,
                          min_margin=MIN_MARGIN) if faceid else None
        # Tiga keadaan yang harus jelas dibedakan, karena tindakan lanjutannya
        # berbeda: sudah terdaftar (tidak perlu apa-apa), terdeteksi tapi belum
        # terdaftar (pencet Daftarkan), dan galeri masih kosong.
        if r and r["status"] == "dikenal":
            label, warna = f"SUDAH TERDAFTAR: {r['nama']}", (0, 255, 0)
        elif r and r["status"] == "ragu":
            label, warna = (f"RAGU: {r['kandidat']}? beda tipis", (0, 255, 255))
        elif r and r["status"] == "tidak dikenal":
            label, warna = "BELUM TERDAFTAR - pencet Daftarkan", (0, 165, 255)
        else:
            label, warna = "GALERI KOSONG - pencet Daftarkan", (255, 200, 0)

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), warna, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(vis, (x, max(0, y - th - 14)), (x + tw + 12, y), warna, -1)
        cv2.putText(vis, label, (x + 6, max(th, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        if r and r["status"] != "kosong":
            # Jarak ditaruh di bawah kotak: berguna untuk menilai seberapa yakin,
            # tapi tidak boleh menutupi label utama.
            cv2.putText(vis, f"jarak {r['jarak']:.3f} / ambang {r['ambang']:.3f}",
                        (x, y + bh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)
        # 5 landmark: 2 mata, hidung, 2 sudut mulut — dasar penyelarasan.
        for j in range(5):
            cv2.circle(vis, (int(f[4 + j*2]), int(f[5 + j*2])), 2, (255, 0, 255), -1)

        semua.append({
            "nama": r["nama"] if r else None,
            "status": r["status"] if r else "kosong",
            "jarak": r["jarak"] if r else 1.0,
            "kandidat": r.get("kandidat") if r else None,
            "kotak": (x, y, bw, bh),
            "utama": hasil_utama is None,   # wajah terbesar yang lolos
        })
        if hasil_utama is None:          # wajah terbesar yang lolos = subjek utama
            hasil_utama, vek_utama, aligned_utama = r, vek, aligned

    stages.append(Stage(
        "asli", "1. Frame + Deteksi Wajah", vis, f"{len(wajah)} wajah terdeteksi",
        "YuNet mendeteksi wajah sekaligus 5 titik landmark (dua mata, hidung, "
        "dua sudut mulut) — titik ungu. Landmark itu yang dipakai untuk "
        "meluruskan wajah di tahap berikutnya."))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stages.append(Stage("gray", "2. Grayscale", gray, "cvtColor BGR -> GRAY",
                        "Ditampilkan untuk perbandingan dengan mode lain. "
                        "YuNet sendiri bekerja pada citra berwarna."))

    if aligned_utama is not None:
        stages.append(Stage(
            "align", "3. Wajah Diluruskan (ALIGN)", aligned_utama, "112x112",
            "alignCrop memutar dan memotong wajah memakai posisi mata, jadi "
            "wajah miring pun menjadi tegak dan berukuran baku. Inilah yang "
            "TIDAK dimiliki pipeline penyu kita — dan salah satu sebab utama "
            "pengenalan wajah jauh lebih akurat."))

        sisi = int(np.sqrt(vek_utama.size))
        vv = vek_utama[:sisi*sisi].reshape(sisi, sisi)
        vv = cv2.normalize(vv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        stages.append(Stage(
            "vektor", "4. Embedding SFace", cv2.applyColorMap(
                cv2.resize(vv, (224, 224), interpolation=cv2.INTER_NEAREST),
                cv2.COLORMAP_VIRIDIS),
            f"{vek_utama.size} dimensi, |v| = 1.0",
            "Embedding 128-dim khusus wajah. Ambang resmi OpenCV: cosine "
            "similarity 0.363 (jarak 0.637). Berbeda dari penyu, ambang ini "
            "sudah tervalidasi di jutaan wajah."))
    else:
        stages.append(Stage("align", "3. Tidak Ada Wajah",
                            np.zeros((112, 112, 3), np.uint8), "—",
                            "Arahkan wajah ke kamera. Kalau tetap tidak "
                            "terdeteksi, coba perbaiki pencahayaan."))

    hasil = {"nama": hasil_utama["nama"] if hasil_utama else None,
             "skor": hasil_utama["jarak"] if hasil_utama else 1.0,
             "peringkat": hasil_utama["peringkat"] if hasil_utama else [],
             "diterima": bool(hasil_utama and hasil_utama["status"] == "dikenal"),
             "vektor": vek_utama, "faceid": hasil_utama,
             "jumlah_wajah": len(wajah), "semua_wajah": semua}
    return stages, hasil
