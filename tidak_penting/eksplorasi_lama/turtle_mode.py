"""
Mode PENYU — pipeline re-identification untuk foto sisik penyu (TurtleID2022).

Kenapa beda dari mode kartu: kartu itu persegi panjang, jadi bisa dicari 4
pojoknya lalu di-warp. Penyu tidak punya bentuk baku, jadi langkah ALIGN-nya
diganti "ambil wilayah kepala terbesar lalu paksa ke ukuran tetap".

Kerangkanya tetap sama persis dengan mode kartu, dan sama dengan MegaDescriptor:

    DETECT   -> temukan wilayah objek        (kontur terbesar)
    ALIGN    -> paksa ke ukuran baku         (crop + resize 224x224)
    DESCRIBE -> jadikan vektor angka         (lihat catatan di bawah)
    MATCH    -> cosine ke galeri individu    (identik dengan pgvector Anda)

CATATAN PENTING soal DESCRIBE
-----------------------------
deskriptor_baseline() di bawah BUKAN MegaDescriptor. Dia cuma citra grayscale
yang diperkecil dan dinormalisasi — baseline lemah yang sengaja dipakai supaya
UI ini jalan tanpa PyTorch dan tanpa unduhan model 1.5 GB.

Yang penting: bentuk antarmukanya sudah sama dengan MegaDescriptor —
    vektor = deskriptor(gambar_bgr) -> np.ndarray 1-D, sudah L2-normalized
    skor   = 1 - cosine(a, b)
Jadi untuk memakai model asli, cukup ganti satu fungsi ini. Lihat
deskriptor_megadescriptor() untuk kerangkanya.
"""

import os

import cv2
import numpy as np

from pipeline import Stage

# Dataset penyu milik pengguna, di luar folder PCV.
GALERI_DEFAULT = "/Users/gung/Documents/CodingProject/by_individual"

PATCH = 224          # ukuran baku setelah ALIGN
MAKS_PER_INDIVIDU = 4   # batasi biar galeri cepat dimuat


# --------------------------------------------------------------- descriptor
def deskriptor_baseline(bgr, sisi=64):
    """Baseline: grayscale + CLAHE + perkecil + L2-normalize.

    CLAHE menonjolkan pola sisik yang jadi ciri individu, mirip sidik jari.
    Hasilnya vektor 1-D yang sudah dinormalisasi, jadi dot-product = cosine.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    kecil = cv2.resize(gray, (sisi, sisi), interpolation=cv2.INTER_AREA)
    v = kecil.astype(np.float32).ravel()
    v -= v.mean()                       # buang pengaruh terang-gelap global
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ------------------------------------------------------- kalibrasi jarak
# Statistik jarak MegaDescriptor-T yang DIUKUR (bukan diasumsikan) pada 20
# individu per dataset — dokumentasi lengkap di Notion bagian 17 & 19.
# Dipakai untuk menerjemahkan jarak mentah menjadi probabilitas: pengguna
# tidak bisa menafsirkan "jarak 0.31", tapi bisa menafsirkan "72% sama".
STATS = {
    # foto utuh bawah air (by_individual): distribusi hampir bertumpang tindih
    "utuh":   {"sama": (0.3101, 0.0988), "beda": (0.3588, 0.0788),
               "ambang": 0.290, "akurasi": 61.0},
    # crop kepala (SeaTurtleIDHeads): pemisahan 3.1x lebih lebar
    "kepala": {"sama": (0.2660, 0.1022), "beda": (0.4193, 0.0785),
               "ambang": 0.355, "akurasi": 79.5},
    # ArcFace fine-tuned di SeaTurtleIDHeads (latih_arcface.py),
    # diukur dengan split time-aware — Top-1 46.0%.
    "arcface": {"sama": (0.2796, 0.1117), "beda": (0.4595, 0.0574),
                 "ambang": 0.370, "akurasi": 85.6},
}
STATS_AKTIF = "utuh"


def set_stats(kunci):
    global STATS_AKTIF
    if kunci in STATS:
        STATS_AKTIF = kunci


def prob_sama(jarak, kunci=None):
    """Terjemahkan jarak cosine -> P(individu sama), lewat rasio likelihood
    dua distribusi Gaussian yang diukur dari data berlabel.

    Prinsipnya sama dengan kalibrasi isotonic di turtle-identification-poc
    (be/app/calibration.py): skor mentah tidak boleh disodorkan ke pengguna,
    ia harus dipetakan dulu ke probabilitas lewat data berlabel. Bedanya di
    sini bentuk distribusinya diasumsikan Gaussian karena datanya (bagian 17
    dan 19) memang mendekati itu.

    PENTING dibaca bersama akurasi seimbangnya: pada dataset "utuh" akurasi
    kalibrasinya cuma 61%, jadi probabilitas di rentang tengah memang akan
    sering berkisar 40-60% — itu bukan bug, itu sistem yang jujur mengakui
    distribusinya bertumpang tindih.
    """
    import math
    s = STATS[kunci or STATS_AKTIF]
    m_s, sd_s = s["sama"]
    m_b, sd_b = s["beda"]

    # Simpangan digabung (pooled). Kalau tiap distribusi memakai simpangannya
    # sendiri, rasio likelihood-nya TIDAK monoton: di ekor kanan jauh,
    # distribusi SAMA yang lebih lebar menang lagi, sehingga jarak 0.55 bisa
    # mendapat probabilitas lebih tinggi daripada 0.42. Probabilitas yang naik
    # saat bukti memburuk tidak bisa dipercaya pengguna.
    sd = (sd_s + sd_b) / 2.0

    def pdf(x, m):
        return math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    p_s = pdf(jarak, m_s)
    p_b = pdf(jarak, m_b)
    if p_s + p_b <= 0:
        return 0.0 if jarak > m_b else 1.0
    return p_s / (p_s + p_b)


# Deskriptor yang sedang dipakai. Diganti lewat set_deskriptor() supaya galeri
# dan query dijamin memakai fungsi yang sama — kalau berbeda, skornya ngawur.
_deskriptor_aktif = deskriptor_baseline


def set_deskriptor(fn):
    global _deskriptor_aktif
    _deskriptor_aktif = fn


def deskriptor_aktif(bgr):
    return _deskriptor_aktif(bgr)


def deskriptor_megadescriptor(bgr, varian="T"):
    """MegaDescriptor asli. Sudah terpasang — lihat megadescriptor.py.

    Akurasi pada 20 individu TurtleID2022, 200 foto held-out:
        baseline piksel      Top-1  6.0%   Top-5 30.5%
        MegaDescriptor-T     Top-1 21.0%   Top-5 51.5%
        MegaDescriptor-L     Top-1 27.0%   Top-5 61.0%
        tebak acak           Top-1  5.0%
    """
    import megadescriptor
    return megadescriptor.deskriptor(bgr, varian=varian)


def cocokkan_cosine(vek, galeri):
    """MATCH: cosine distance ke seluruh galeri. Sama dengan pgvector <=>.

    Return (nama_terbaik, skor_terbaik, peringkat). Skor 0 = identik,
    supaya arah "makin kecil makin mirip" konsisten dengan SAD di mode kartu.
    """
    if not galeri:
        return "galeri kosong", 1.0, []
    peringkat = []
    for nama, vecs in galeri.items():
        # Satu individu punya beberapa foto; ambil yang paling mirip.
        best = max(float(np.dot(vek, v)) for v in vecs)
        peringkat.append((nama, max(0.0, min(1.0, (1.0 - best) / 2.0))))
    peringkat.sort(key=lambda x: x[1])
    return peringkat[0][0], peringkat[0][1], peringkat


# Warna latar pengganti, menyamai mask_background_fill="gray" di POC.
# Abu-abu netral dipilih supaya tidak menciptakan tepi kontras palsu di batas
# mask — latar hitam justru melahirkan keypoint baru tepat di siluetnya.
ISI_MASK = (128, 128, 128)

# Masking DIMATIKAN secara bawaan karena terbukti merugikan.
#
# Diukur pada 20 individu TurtleID2022, 200 foto held-out, MegaDescriptor-T:
#     foto utuh tanpa apa-apa   Top-1 21.0%   Top-5 51.5%
#     crop ROI kontur           Top-1 17.0%   Top-5 39.0%
#     foto utuh + masking       Top-1 13.5%   Top-5 41.5%
#     crop ROI + masking        Top-1  9.5%   Top-5 38.0%
#
# Masking hanya menolong kalau mask-nya benar. GrabCut memilih wilayah yang
# kebetulan kontras, bukan penyunya — hanya 18.7% keypoint yang jatuh di objek,
# padahal luas objeknya 24.2%. Mask yang salah menghapus ciri asli dan
# menciptakan tepi buatan yang dibaca model sebagai ciri palsu.
#
# Dibiarkan ada supaya percobaannya bisa diulang: set_masking(True).
MASKING = False

# Crop ROI juga DIMATIKAN secara bawaan, alasan yang sama: 17.0% vs 21.0%.
# Kontur terbesar sering menemukan riak pasir, bukan penyu, jadi crop-nya
# membuang justru bagian yang membawa identitas. Sampai ada detektor kepala
# yang benar, memberi foto utuh ke descriptor lebih baik daripada memberi
# potongan yang salah.
#
# Kotak ROI TETAP dihitung dan digambar di tahap "Wilayah Objek" — supaya
# terlihat betapa seringnya ia meleset. Yang berubah hanya: hasilnya tidak
# lagi dipakai untuk memotong masukan descriptor.
CROP_ROI = False


def set_crop(aktif):
    global CROP_ROI
    CROP_ROI = bool(aktif)


def set_masking(aktif):
    global MASKING
    MASKING = bool(aktif)


def buang_latar(bgr, kotak=None, iterasi=3):
    """Buang latar dengan GrabCut, sisakan objek utama di atas abu-abu.

    GrabCut diberi kotak awal sebagai tebakan "objek ada di dalam sini", lalu
    ia menyempurnakan batasnya sendiri dengan model warna foreground/background.

    Return (gambar_termask, mask_boolean). Kalau segmentasi gagal atau
    menghasilkan area terlalu kecil, gambar asli dikembalikan apa adanya —
    lebih baik tidak melakukan masking daripada menghapus penyunya.
    """
    h, w = bgr.shape[:2]
    if kotak is None:
        # Tanpa petunjuk, anggap objek ada di 80% bagian tengah.
        m = 0.1
        kotak = (int(w * m), int(h * m), int(w * (1 - 2 * m)), int(h * (1 - 2 * m)))
    else:
        x0, y0, x1, y1 = kotak
        kotak = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, kotak, bgd, fgd, iterasi, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return bgr, None

    depan = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    # Kalau tersisa < 5% piksel, GrabCut kemungkinan besar salah dan malah
    # menghapus objeknya. Lebih aman kembalikan gambar utuh.
    if depan.sum() < 0.05 * h * w:
        return bgr, None

    keluar = bgr.copy()
    keluar[~depan] = ISI_MASK
    return keluar, depan


def align(bgr, params):
    """DETECT + ALIGN: potong ke wilayah objek terbesar, paksa ke PATCH x PATCH.

    Galeri DAN query wajib lewat fungsi yang sama. Kalau galeri dihitung dari
    gambar utuh sementara query dari hasil crop, deskriptornya tidak sebanding
    dan pencocokannya ngawur — persis seperti template kartu yang harus
    diproses sama dengan kartu dari kamera.
    """
    h, w = bgr.shape[:2]
    if params.scale != 1.0:
        small = cv2.resize(bgr, (int(w * params.scale), int(h * params.scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    k = max(1, params.blur | 1)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (k, k), 0),
                      params.canny_lo, params.canny_hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(bgr, (PATCH, PATCH), interpolation=cv2.INTER_AREA), None

    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    pad = int(0.08 * max(cw, ch))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(sw, x + cw + pad), min(sh, y + ch + pad)
    if CROP_ROI:
        f = 1.0 / params.scale
        crop = bgr[int(y0 * f):int(y1 * f), int(x0 * f):int(x1 * f)]
        if crop.size == 0:
            crop = bgr
    else:
        crop = bgr          # foto utuh — terukur lebih baik, lihat catatan atas
    patch = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)

    # Masking dilakukan SETELAH crop, di ukuran baku: GrabCut mahal, dan pada
    # 224x224 jauh lebih cepat daripada di resolusi penuh. Karena align()
    # dipakai galeri maupun query, keduanya otomatis diperlakukan sama.
    if MASKING:
        patch, _ = buang_latar(patch)
    return patch, (x0, y0, x1, y1)


# Satu patch acuan per individu, disimpan saat galeri dibangun. Dipakai untuk
# visualisasi pola sisik — deskriptor saja tidak bisa digambar pasangannya.
_patch_acuan = {}


def muat_galeri(folder=None, deskriptor=None, params=None, maks_individu=None):
    """Baca <folder>/<id>/**.jpg dan hitung deskriptornya.

    maks_individu membatasi jumlah individu yang dimuat — SeaTurtleIDHeads
    punya 400 individu dan memuat semuanya membekukan UI berpuluh detik.
    """
    from pipeline import Params
    deskriptor = deskriptor or deskriptor_aktif
    _patch_acuan.clear()
    folder = folder or GALERI_DEFAULT
    params = params or Params()
    galeri, contoh = {}, []
    if not os.path.isdir(folder):
        return galeri, contoh, f"Folder galeri tidak ada: {folder}"

    daftar_ind = sorted(os.listdir(folder))
    if maks_individu:
        daftar_ind = daftar_ind[:maks_individu]
    for ind in daftar_ind:
        d = os.path.join(folder, ind)
        if not os.path.isdir(d):
            continue
        # Struktur datanya bertingkat (individu/view/file), jadi telusuri semua.
        files = []
        for akar, _, names in os.walk(d):
            files += [os.path.join(akar, n) for n in sorted(names)
                      if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        vecs = []
        for p in files[:MAKS_PER_INDIVIDU]:
            img = cv2.imread(p)
            if img is None:
                continue
            patch, _ = align(img, params)      # jalur yang sama dengan query
            vecs.append(deskriptor(patch))
            _patch_acuan.setdefault(ind, patch)
            if len(contoh) < 40:
                contoh.append(p)
        if vecs:
            galeri[ind] = vecs
    return galeri, contoh, f"{len(galeri)} individu dimuat dari {os.path.basename(folder)}"


# ----------------------------------------------------------------- pipeline
def gambar_kotak(frame, kotak, params, label, warna, titik=None):
    """Overlay penanda area + pola sisik + label identitas pada frame penuh.

    Kotak sengaja digambar TIPIS: ia cuma penanda area, bukan yang dipakai
    untuk mengenali. Yang menentukan identitas adalah sebaran titik pola sisik,
    jadi titik-titik itulah yang ditonjolkan.
    """
    if kotak is None:
        return frame
    f = 1.0 / params.scale
    x0, y0, x1, y1 = (int(v * f) for v in kotak)
    out = frame.copy()

    if titik is not None and len(titik):
        # Skala titik dari ruang patch (224) ke frame penuh.
        h, w = frame.shape[:2]
        sx, sy = w / PATCH, h / PATCH
        for x, y in titik.astype(int):
            cv2.circle(out, (int(x * sx), int(y * sy)), 2, (0, 255, 255), -1)

    cv2.rectangle(out, (x0, y0), (x1, y1), warna, 1)
    # Latar solid di belakang teks: label putih di atas pasir terang tidak terbaca.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (x0, max(0, y0 - th - 12)), (x0 + tw + 12, y0), warna, -1)
    cv2.putText(out, label, (x0 + 6, max(th, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return out


def jalankan(frame, galeri, params, faceid=None):
    """Pipeline penyu. Tanda tangannya sama dengan pipeline.jalankan().

    Kalau `faceid` (faceid_penyu.Galeri) diberikan, tahap 1 menampilkan kotak
    deteksi beserta hasil pengenalannya — inilah tampilan realtime-nya.
    """
    stages = []
    h, w = frame.shape[:2]
    patch_awal, kotak_awal = align(frame, params)

    # Kenali dulu supaya hasilnya bisa digambar di frame tahap 1.
    # Titik pola dihitung sekali di sini supaya bisa dipakai di tahap 1
    # (overlay di frame) maupun tahap "Titik Pola Sisik" nanti.
    titik_pola = None
    try:
        import pola_sisik
        if pola_sisik.tersedia():
            kp, _ = pola_sisik.ekstrak(patch_awal)
            titik_pola = kp.cpu().numpy() * (PATCH / pola_sisik.IMG_SIZE)
    except (ImportError, RuntimeError):
        pass

    id_hasil = None
    if faceid is not None:
        id_hasil = faceid.kenali(deskriptor_aktif(patch_awal))
        # Jarak mentah tidak bisa ditafsirkan pengguna; probabilitas bisa.
        id_hasil["prob"] = prob_sama(id_hasil["jarak"])
        pr = id_hasil["prob"]
        if id_hasil["status"] == "dikenal":
            label, warna = f"{id_hasil['nama']}  {pr:.0%} sama", (0, 255, 0)
        elif id_hasil["status"] == "tidak dikenal":
            label = f"TIDAK DIKENAL  ({pr:.0%} thd terdekat)"
            warna = (0, 165, 255)
        else:
            label, warna = "BELUM ADA YANG TERDAFTAR", (200, 200, 200)
        frame_tampil = gambar_kotak(frame, kotak_awal, params, label, warna,
                                    titik=titik_pola)
    else:
        frame_tampil = frame

    stages.append(Stage(
        "asli", "1. Frame Asli + Deteksi", frame_tampil, f"{w}x{h} piksel, BGR",
        "Foto penyu mentah dengan kotak deteksi dan hasil Face ID. Tidak "
        "seperti kartu, penyu tidak punya bentuk geometris yang bisa "
        "diandalkan — jadi strategi ALIGN-nya harus berbeda."))

    if params.scale != 1.0:
        small = cv2.resize(frame, (int(w * params.scale), int(h * params.scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()
    sh, sw = small.shape[:2]
    stages.append(Stage("resize", "2. Perkecil", small,
                        f"skala {params.scale:.2f} -> {sw}x{sh}",
                        "Sama seperti mode kartu: mempercepat langkah berikutnya."))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    stages.append(Stage("gray", "3. Grayscale", gray, "cvtColor BGR -> GRAY",
                        "Pola sisik penyu adalah soal tekstur terang-gelap, "
                        "bukan warna. Warna justru berubah-ubah tergantung air "
                        "dan cahaya, jadi membuangnya membantu."))

    # CLAHE: ini langkah khas penyu yang tidak ada di pipeline kartu.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    stages.append(Stage(
        "clahe", "4. CLAHE (kontras lokal)", clahe, "clipLimit 2.0, tile 8x8",
        "Histogram equalization yang bekerja per petak kecil, bukan seluruh "
        "gambar. Ini menaikkan kontras batas antar sisik tanpa membakar bagian "
        "yang sudah terang. Untuk re-ID penyu ini langkah paling berpengaruh — "
        "pola sisik itulah identitasnya."))

    k = max(1, params.blur | 1)
    blur = cv2.GaussianBlur(clahe, (k, k), 0)
    stages.append(Stage("blur", "5. Gaussian Blur", blur, f"kernel {k}x{k}",
                        "Meredam noise sensor supaya tepi sisik tidak pecah."))

    edges = cv2.Canny(blur, params.canny_lo, params.canny_hi)
    stages.append(Stage(
        "edges", "6. Deteksi Tepi (Canny)", edges,
        f"ambang {params.canny_lo} / {params.canny_hi}",
        "Di sinilah pola sisik terlihat sebagai jaringan garis. Geser slider "
        "Canny dan perhatikan: ambang terlalu tinggi menghapus sisik halus, "
        "terlalu rendah membanjiri gambar dengan tekstur air."))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = small.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
    stages.append(Stage("contours", "7. Semua Kontur", vis,
                        f"{len(contours)} kontur",
                        "Berbeda dari kartu, di sini kontur TIDAK dipakai untuk "
                        "mencari bentuk tertentu — cuma untuk menemukan wilayah "
                        "objek terbesar."))

    # ALIGN: ambil kotak pembatas kontur terbesar sebagai wilayah objek.
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        pad = int(0.08 * max(cw, ch))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(sw, x + cw + pad), min(sh, y + ch + pad)
        vis_box = small.copy()
        cv2.rectangle(vis_box, (x0, y0), (x1, y1), (0, 255, 0), 2)
        stages.append(Stage("roi", "8. Wilayah Objek", vis_box,
                            f"kotak {x1-x0}x{y1-y0}",
                            "Kotak pembatas kontur terbesar, diberi margin 8%. "
                            "Ini pengganti kasar dari detektor kepala penyu. "
                            "Di produksi, bagian inilah yang diganti model "
                            "deteksi terlatih."))
        # Ambil crop dari frame resolusi penuh, bukan dari versi kecil.
        f = 1.0 / params.scale
        crop = frame[int(y0*f):int(y1*f), int(x0*f):int(x1*f)]
        if crop.size == 0:
            crop = frame
    else:
        crop = frame
        stages.append(Stage("roi", "8. Wilayah Objek", frame, "tidak ada kontur",
                            "Tidak ada kontur ditemukan; memakai seluruh frame."))

    # Pakai align() yang sama dengan galeri — jangan hitung ulang sendiri,
    # karena perbedaan sekecil apa pun bikin skornya tidak sebanding.
    patch, _ = align(frame, params)
    stages.append(Stage("patch", "9. Patch Baku (ALIGN)", patch,
                        f"-> {PATCH}x{PATCH}",
                        "Semua gambar dipaksa ke ukuran sama sebelum dijadikan "
                        "vektor. Wajib: deskriptor hanya bisa dibandingkan kalau "
                        "masukannya berukuran sama."))

    patch_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    stages.append(Stage("desc", "10. Sidik Jari (DESCRIBE)", patch_clahe,
                        "baseline: CLAHE + 64x64 + L2-norm",
                        "INI descriptor-nya, dan ini titik tukar ke "
                        "MegaDescriptor. Baseline sekarang cuma piksel yang "
                        "diperkecil — lemah terhadap perubahan sudut pandang. "
                        "MegaDescriptor menghasilkan vektor 1536-dim yang tahan "
                        "pose dan pencahayaan. Antarmukanya sudah sama, jadi "
                        "tinggal ganti satu fungsi."))

    vek = deskriptor_aktif(patch)
    nama, skor, peringkat = cocokkan_cosine(vek, galeri)
    referensi_bgr = _patch_acuan.get(peringkat[0][0]) if peringkat else None
    hasil_sisik = None

    # Tampilkan vektor sebagai gambar supaya "sidik jari" itu kelihatan wujudnya.
    sisi = int(np.sqrt(vek.size))
    vv = vek[:sisi*sisi].reshape(sisi, sisi)
    vv = cv2.normalize(vv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages.append(Stage("vektor", "11. Vektor Deskriptor", cv2.applyColorMap(
        cv2.resize(vv, (PATCH, PATCH), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_VIRIDIS),
        f"{vek.size} dimensi, |v| = 1.0",
        "Vektor yang sama digambar sebagai peta warna. Inilah yang disimpan "
        "di pgvector pada proyek turtle-identification-poc Anda, dan "
        "dibandingkan dengan operator cosine."))

    # --- Titik pola sisik digambar langsung di atas penyunya.
    # Kotak hanya penanda area tipis; yang benar-benar dipakai untuk mengenali
    # adalah sebaran titik ini. Menampilkannya membuat jelas berapa banyak
    # titik yang jatuh di penyu dan berapa yang terbuang ke pasir.
    if titik_pola is not None:
        vis_pola = patch.copy()
        for x, y in titik_pola.astype(int):
            if 0 <= x < PATCH and 0 <= y < PATCH:
                # Titik kuning kecil; menumpuk di area bertekstur seperti sisik.
                cv2.circle(vis_pola, (x, y), 2, (0, 255, 255), -1)
        stages.append(Stage(
            "titik_pola", "12. Titik Pola Sisik", vis_pola,
            f"{len(titik_pola)} titik ALIKED",
            "Setiap titik kuning adalah pola lokal yang dianggap khas. "
            "INILAH yang sebenarnya dipakai untuk mengenali — bukan kotaknya. "
            "Perhatikan sebarannya: kalau titik-titik menumpuk di pasir dan "
            "riak air alih-alih di sisik kepala, maka yang dicocokkan sistem "
            "adalah lokasi pemotretan, bukan penyunya."))

    # --- Pola sisik: cocokkan keypoint lokal ke foto acuan individu terbaik.
    # Ini pelengkap MegaDescriptor, bukan pengganti — POC memakai
    # skor_akhir = max(skor_mega, skor_lokal).
    if referensi_bgr is not None:
        try:
            import pola_sisik
            if pola_sisik.tersedia():
                cocok = pola_sisik.cocokkan(patch, referensi_bgr)
                stages.append(Stage(
                    "sisik", "12. Pola Sisik (ALIKED)",
                    pola_sisik.gambar_pasangan(patch, referensi_bgr, cocok),
                    f"{cocok['jumlah']} pasang, skor {cocok['skor']:.3f}",
                    "Kiri foto uji, kanan foto acuan. Garis berwarna "
                    "menghubungkan pola sisik yang dianggap sama. Berbeda dari "
                    "MegaDescriptor yang memberi satu skor global, di sini "
                    "terlihat BAGIAN MANA yang cocok. Titik abu-abu = keypoint "
                    "yang ditemukan tapi tidak berpasangan — perhatikan berapa "
                    "banyak yang jatuh di pasir dan air, bukan di penyunya."))
                hasil_sisik = cocok
        except (ImportError, RuntimeError) as e:
            print(f"[pola_sisik] dilewati: {e}")

    hasil = {"nama": nama, "skor": skor, "peringkat": peringkat[:10],
             "diterima": skor <= params.conf_max,
             "vektor": vek, "faceid": id_hasil,
             "sisik": hasil_sisik}
    return stages, hasil
