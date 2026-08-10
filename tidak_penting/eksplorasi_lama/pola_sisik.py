"""
POLA SISIK — pencocokan fitur lokal ALIKED + LightGlue.

Meniru pendekatan turtle-identification-poc/be/app/local_features.py.

Kenapa ini penting: MegaDescriptor menghasilkan SATU vektor untuk seluruh
gambar. Kalau penyunya bergeser, terbalik, atau sebagian tertutup, vektor itu
ikut berubah banyak. Fitur lokal bekerja sebaliknya — ia mencari titik-titik
khas pada pola sisik, lalu memasangkannya satu per satu antara dua foto.
Sepuluh pasang sisik yang cocok adalah bukti identitas yang jauh lebih kuat
daripada satu skor kemiripan global.

Ini juga yang membuat POC Anda memakai:
    skor_akhir = max(skor_megadescriptor, skor_aliked)

ALIKED : mencari keypoint + deskriptor lokal (pengganti SIFT yang terlatih)

CATATAN: LightGlue TIDAK dipakai di sini.
POC Anda memakai ALIKED + LightGlue. LightGlue versi kornia 0.8.3 di
lingkungan ini mengembalikan NOL pasangan bahkan ketika sebuah gambar
dicocokkan dengan dirinya sendiri (sudah diuji di CPU dan MPS, dengan pruning
dimatikan) — jadi bobotnya tidak berfungsi sebagaimana mestinya.

Sebagai gantinya dipakai mutual nearest-neighbour + ratio test Lowe langsung
di atas deskriptor ALIKED. Ekstraksi ALIKED-nya sendiri normal. Kalau nanti
LightGlue diperbaiki, cukup ganti isi cocokkan(); bentuk keluarannya sudah
sama.
"""

import numpy as np

IMG_SIZE = 384          # sama dengan aliked_image_size di config POC
MAX_KEYPOINTS = 512     # sama dengan aliked_max_keypoints
RATIO_LOWE = 0.9        # ratio test; makin kecil makin ketat
MIN_SIMILARITY = 0.5    # cosine minimum sebelum pasangan dianggap sah

_alat = {}


def tersedia():
    try:
        import kornia  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _muat():
    """Muat ALIKED + LightGlue sekali saja."""
    if _alat:
        return _alat
    import torch
    from kornia.feature import ALIKED, LightGlue

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # LightGlue harus dibuat dengan nama fitur yang sama dengan ekstraktornya,
    # karena bobot pemasangannya khusus per jenis deskriptor.
    _alat["ekstraktor"] = ALIKED(max_num_keypoints=MAX_KEYPOINTS).to(device).eval()
    _alat["pemasang"] = LightGlue(features="aliked").to(device).eval()
    _alat["device"] = device
    return _alat


def _ke_tensor(bgr):
    import cv2
    import torch
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(_muat()["device"])


def ekstrak(bgr):
    """Cari keypoint pola sisik pada satu gambar.

    Return (koordinat Nx2, deskriptor NxD) dalam skala IMG_SIZE.
    """
    import torch
    a = _muat()
    with torch.no_grad():
        # ALIKED kornia mengembalikan list[ALIKEDFeatures], satu per gambar —
        # bukan dict seperti implementasi LightGlue aslinya.
        fitur = a["ekstraktor"](_ke_tensor(bgr))[0]
    return fitur.keypoints, fitur.descriptors


def cocokkan(bgr_a, bgr_b, ratio=RATIO_LOWE):
    """Pasangkan pola sisik dua gambar.

    Memakai mutual nearest-neighbour + ratio test, BUKAN LightGlue.
    Alasannya ada di catatan modul: LightGlue kornia 0.8.3 mengembalikan nol
    pasangan bahkan untuk gambar yang dicocokkan dengan dirinya sendiri.

    Dua saringan yang dipakai:
      mutual NN  — A memilih B sebagai terdekat DAN B memilih A. Membuang
                   keypoint yang "menempel" ke banyak calon sekaligus.
      ratio test — jarak ke terdekat harus jauh lebih kecil daripada ke
                   terdekat kedua. Kalau dua calon sama miripnya, pasangan itu
                   tidak informatif dan dibuang (kriteria Lowe).

    Return dict: jumlah pasangan, koordinatnya, dan skor ternormalisasi
    (jumlah / MAX_KEYPOINTS) — sama seperti normalize_aliked_count() di POC.
    """
    kp_a, ds_a = ekstrak(bgr_a)
    kp_b, ds_b = ekstrak(bgr_b)
    kp_a = kp_a.cpu().numpy()
    kp_b = kp_b.cpu().numpy()
    A = ds_a.cpu().numpy()
    B = ds_b.cpu().numpy()

    if len(A) == 0 or len(B) == 0:
        return {"jumlah": 0, "titik_a": np.zeros((0, 2)), "titik_b": np.zeros((0, 2)),
                "kepercayaan": np.zeros(0), "kp_a": kp_a, "kp_b": kp_b, "skor": 0.0}

    # Deskriptor ALIKED sudah L2-normalized, jadi dot product = cosine.
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    sim = A @ B.T

    urut = np.argsort(-sim, axis=1)
    terbaik = urut[:, 0]
    kedua = urut[:, 1] if sim.shape[1] > 1 else urut[:, 0]
    s1 = sim[np.arange(len(A)), terbaik]
    s2 = sim[np.arange(len(A)), kedua]

    # Ratio test pada jarak (1 - cosine); ambang Lowe berlaku untuk jarak.
    d1, d2 = 1.0 - s1, 1.0 - s2
    lolos_ratio = d1 < ratio * np.maximum(d2, 1e-9)

    # Mutual: pilihan terbaik B untuk kolom itu harus balik ke A.
    balik = np.argmax(sim, axis=0)
    mutual = balik[terbaik] == np.arange(len(A))

    valid = lolos_ratio & mutual & (s1 >= MIN_SIMILARITY)
    idx_a = np.nonzero(valid)[0]
    idx_b = terbaik[valid]

    return {"jumlah": int(valid.sum()),
            "titik_a": kp_a[idx_a] if len(idx_a) else np.zeros((0, 2)),
            "titik_b": kp_b[idx_b] if len(idx_b) else np.zeros((0, 2)),
            "kepercayaan": s1[valid] if len(idx_a) else np.zeros(0),
            "kp_a": kp_a, "kp_b": kp_b,
            "skor": min(1.0, int(valid.sum()) / MAX_KEYPOINTS)}


def gambar_pasangan(bgr_a, bgr_b, hasil, maks=80):
    """Gambar dua foto berdampingan dengan garis antar pola sisik yang cocok.

    Inilah tampilan yang Anda minta: pola sisik di sebelah kanan dipasangkan
    ke pola sisik di foto acuan, sehingga terlihat BAGIAN MANA yang membuat
    sistem yakin ini penyu yang sama.
    """
    import cv2
    A = cv2.resize(bgr_a, (IMG_SIZE, IMG_SIZE))
    B = cv2.resize(bgr_b, (IMG_SIZE, IMG_SIZE))
    kanvas = np.zeros((IMG_SIZE, IMG_SIZE * 2 + 20, 3), np.uint8)
    kanvas[:, :IMG_SIZE] = A
    kanvas[:, IMG_SIZE + 20:] = B

    # Semua keypoint dulu (redup), supaya terlihat mana yang ditemukan tapi
    # tidak berpasangan.
    for p in hasil["kp_a"]:
        cv2.circle(kanvas, (int(p[0]), int(p[1])), 1, (90, 90, 90), -1)
    for p in hasil["kp_b"]:
        cv2.circle(kanvas, (int(p[0]) + IMG_SIZE + 20, int(p[1])), 1, (90, 90, 90), -1)

    n = min(maks, len(hasil["titik_a"]))
    for i in range(n):
        pa = hasil["titik_a"][i]
        pb = hasil["titik_b"][i]
        xa, ya = int(pa[0]), int(pa[1])
        xb, yb = int(pb[0]) + IMG_SIZE + 20, int(pb[1])
        # Warna berputar supaya garis yang bertumpuk tetap bisa dibedakan.
        warna = tuple(int(c) for c in cv2.applyColorMap(
            np.uint8([[int(255 * i / max(1, n))]]), cv2.COLORMAP_HSV)[0][0])
        cv2.line(kanvas, (xa, ya), (xb, yb), warna, 1, cv2.LINE_AA)
        cv2.circle(kanvas, (xa, ya), 3, warna, -1)
        cv2.circle(kanvas, (xb, yb), 3, warna, -1)

    teks = f"{hasil['jumlah']} pola sisik cocok  (skor {hasil['skor']:.3f})"
    cv2.rectangle(kanvas, (0, 0), (IMG_SIZE * 2 + 20, 30), (0, 0, 0), -1)
    cv2.putText(kanvas, teks, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if hasil["jumlah"] else (0, 165, 255), 2)
    return kanvas
