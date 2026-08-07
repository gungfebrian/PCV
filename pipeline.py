"""
Pipeline pengenalan kartu, dipecah jadi tahap-tahap bernama supaya bisa dilihat satu per satu.

Alur besarnya sama dengan pola re-identification (mis. MegaDescriptor untuk penyu):

    DETECT  -> temukan objek di frame          (Canny + contour + approxPolyDP)
    ALIGN   -> luruskan ke bentuk baku         (warpPerspective 300x420)
    DESCRIBE-> ubah jadi "sidik jari" numerik  (grayscale + threshold Otsu)
    MATCH   -> cari tetangga terdekat di galeri (SAD ke 52 template)

Kalau nanti mau dipakai untuk penyu, yang diganti cuma DESCRIBE dan MATCH:
buat kelas baru dengan method describe()/match() yang sama, sisanya tetap.
"""

import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["Spade", "Heart", "Diamond", "Club"]

# Ukuran kartu setelah diluruskan. Semua template dipaksa ke ukuran ini juga,
# supaya SAD membandingkan piksel yang benar-benar sejajar.
CARD_W, CARD_H = 300, 420


class Params:
    """Semua angka yang bisa diutak-atik dari slider di UI."""

    def __init__(self):
        self.scale = 0.5          # perkecil frame dulu biar deteksi cepat
        self.blur = 5             # ukuran kernel Gaussian (harus ganjil)
        self.canny_lo = 50
        self.canny_hi = 150
        self.thresh = 150         # ambang biner sebelum Otsu mengoreksinya
        self.area_div = 150       # area minimum kontur = (w*h) / area_div
        self.epsilon = 0.02       # toleransi approxPolyDP, relatif ke keliling
        self.conf_max = 0.5       # di atas ini dianggap "tidak diketahui"

    def copy(self):
        p = Params()
        p.__dict__.update(self.__dict__)
        return p


class Stage:
    """Satu tahap pipeline: gambar hasil + penjelasan singkat."""

    def __init__(self, key, title, image, note="", detail=""):
        self.key = key
        self.title = title
        self.image = image      # BGR atau grayscale, siap ditampilkan
        self.note = note        # ringkas, tampil di bawah thumbnail
        self.detail = detail    # panjang, tampil saat stage dipilih


def urutkan_titik(pts):
    """Urutkan 4 pojok jadi [kiri-atas, kanan-atas, kanan-bawah, kiri-bawah].

    Lihat penjelasan_titik.md. Ringkasnya: x+y paling kecil pasti kiri-atas dan
    paling besar pasti kanan-bawah; sisanya dibedakan pakai x-y.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def muat_template(folder=None, params=None):
    """Baca 52 template kartu dan ubah jadi bentuk yang siap dibandingkan.

    Template diproses dengan langkah yang sama persis dengan kartu dari kamera
    (resize -> gray -> threshold). Kalau berbeda, SAD-nya tidak adil.
    """
    # Pakai deck HD hasil unduh_template.py kalau ada: gambarnya bersih dan
    # lengkap 52, sedangkan Templatekartu/ berisi foto dan kurang 2 kartu.
    if folder is None:
        hd = os.path.join(BASE_DIR, "Templatekartu_hd")
        folder = hd if os.path.isdir(hd) else os.path.join(BASE_DIR, "Templatekartu")
    params = params or Params()

    templates = {}
    hilang = []
    for suit in SUITS:
        for rank in RANKS:
            path = os.path.join(folder, f"{rank}_{suit}.jpg")
            img = cv2.imread(path)
            if img is None:
                hilang.append(f"{rank}_{suit}")
                continue
            img = cv2.resize(img, (CARD_W, CARD_H))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(
                gray, params.thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            templates[f"{rank} {suit}"] = thr
    return templates, hilang


def deskripsi(kartu_warped, params):
    """DESCRIBE: ubah kartu yang sudah lurus jadi citra biner ("sidik jari").

    Untuk penyu, fungsi ini yang diganti embedding MegaDescriptor.
    """
    gray = cv2.cvtColor(kartu_warped, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(
        gray, params.thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return gray, thr


def cocokkan(desc, templates, params):
    """MATCH: bandingkan sidik jari ke seluruh galeri, ambil yang paling mirip.

    Metrik: SAD (Sum of Absolute Differences) — makin kecil makin mirip.
    Dinormalisasi ke 0..1 dengan membagi selisih maksimum yang mungkin,
    supaya angkanya bisa dibandingkan antar ukuran gambar.

    Return: (nama_terbaik, skor_terbaik, peringkat) — peringkat = list
    (nama, skor) terurut dari paling mirip.
    """
    if not templates:
        return "tidak diketahui", 1.0, []

    peringkat = []
    a = desc.astype(np.float32)
    max_sad = desc.shape[0] * desc.shape[1] * 255.0

    for nama, tmpl in templates.items():
        if tmpl.shape != desc.shape:
            tmpl = cv2.resize(tmpl, (desc.shape[1], desc.shape[0]))
        sad = float(np.sum(np.abs(a - tmpl.astype(np.float32))))
        peringkat.append((nama, max(0.0, min(1.0, sad / max_sad))))

    peringkat.sort(key=lambda x: x[1])
    nama, skor = peringkat[0]
    if skor > params.conf_max:
        nama = "tidak diketahui"
    return nama, skor, peringkat


def peta_selisih(desc, tmpl):
    """Gambar di mana persisnya kartu dan template berbeda.

    Merah = beda, hitam = sama. Ini yang bikin SAD gampang dimengerti:
    angkanya tinggi karena area merahnya luas.
    """
    if tmpl.shape != desc.shape:
        tmpl = cv2.resize(tmpl, (desc.shape[1], desc.shape[0]))
    diff = cv2.absdiff(desc, tmpl)
    out = np.zeros((*diff.shape, 3), dtype=np.uint8)
    out[:, :, 2] = diff          # taruh selisih di kanal merah
    return out


def jalankan(frame, templates, params):
    """Jalankan seluruh pipeline dan kembalikan daftar Stage untuk ditampilkan.

    Selalu mengembalikan tahap-tahap awal walaupun tidak ada kartu yang ketemu,
    supaya pengguna tetap bisa melihat di mana prosesnya berhenti.
    """
    stages = []
    h, w = frame.shape[:2]

    stages.append(Stage(
        "asli", "1. Frame Asli", frame,
        f"{w}x{h} piksel, BGR",
        "Gambar mentah dari kamera atau file. OpenCV menyimpan warna dengan "
        "urutan Biru-Hijau-Merah (BGR), bukan RGB."))

    # --- Perkecil dulu: deteksi kontur di gambar kecil jauh lebih cepat,
    # dan pojok yang ditemukan tinggal dikali balik ke ukuran asli.
    if params.scale != 1.0:
        small = cv2.resize(frame, (int(w * params.scale), int(h * params.scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()
    sh, sw = small.shape[:2]
    stages.append(Stage(
        "resize", "2. Perkecil", small,
        f"skala {params.scale:.2f} -> {sw}x{sh}",
        "Mengecilkan gambar mempercepat semua langkah berikutnya. Koordinat "
        "pojok yang ditemukan nanti dikalikan balik ke ukuran asli, jadi "
        "hasil akhirnya tetap resolusi penuh."))

    # --- Warna dibuang: bentuk kartu bisa dikenali dari terang-gelap saja.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    stages.append(Stage(
        "gray", "3. Grayscale", gray,
        "cvtColor BGR -> GRAY",
        "Dari 3 kanal warna jadi 1 kanal keabuan. Tepi kartu ditentukan oleh "
        "perbedaan terang-gelap, jadi warna tidak diperlukan dan cuma bikin "
        "perhitungan 3x lebih berat."))

    # --- Blur: hilangkan bintik halus supaya Canny tidak menemukan tepi palsu.
    k = max(1, params.blur | 1)          # kernel Gaussian wajib ganjil
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    stages.append(Stage(
        "blur", "4. Gaussian Blur", blur,
        f"kernel {k}x{k}",
        "Meredam noise dan tekstur halus. Tanpa ini, Canny akan menandai "
        "butiran kecil sebagai tepi dan kontur kartu jadi patah-patah. "
        "Kernel makin besar makin halus, tapi tepi asli ikut melemah."))

    # --- Canny: cari lokasi perubahan terang-gelap yang tajam.
    edges = cv2.Canny(blur, params.canny_lo, params.canny_hi)
    stages.append(Stage(
        "edges", "5. Deteksi Tepi (Canny)", edges,
        f"ambang {params.canny_lo} / {params.canny_hi}",
        "Canny memakai dua ambang. Piksel di atas ambang atas pasti tepi; "
        "yang di antara dua ambang hanya jadi tepi kalau menempel pada tepi "
        "kuat. Itu sebabnya garisnya tipis dan menyambung."))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_all = small.copy()
    cv2.drawContours(vis_all, contours, -1, (0, 255, 255), 1)
    stages.append(Stage(
        "contours", "6. Semua Kontur", vis_all,
        f"{len(contours)} kontur ditemukan",
        "findContours merangkai piksel tepi jadi kurva tertutup. RETR_EXTERNAL "
        "hanya mengambil kontur terluar, jadi gambar di dalam kartu diabaikan. "
        "Sebagian besar kontur ini sampah dan disaring di langkah berikutnya."))

    # --- Saring: kartu itu besar DAN punya tepat 4 sudut.
    area_min = (sw * sh) / max(1.0, params.area_div)
    vis_quad = small.copy()
    kandidat = []
    for c in contours:
        if cv2.contourArea(c) <= area_min:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, params.epsilon * peri, True)
        if len(approx) == 4:
            kandidat.append(approx)
            cv2.drawContours(vis_quad, [approx], -1, (0, 255, 0), 2)
            for pt in approx.reshape(4, 2):
                cv2.circle(vis_quad, tuple(int(v) for v in pt), 5, (0, 0, 255), -1)
    stages.append(Stage(
        "quads", "7. Saring Segi-4", vis_quad,
        f"{len(kandidat)} kandidat (area > {area_min:.0f} px)",
        "Dua saringan sekaligus. Pertama buang kontur yang terlalu kecil. "
        "Lalu approxPolyDP menyederhanakan kurva jadi poligon; yang tersisa "
        "dengan tepat 4 titik dianggap kartu. Titik merah adalah pojoknya."))

    if not kandidat:
        stages.append(Stage(
            "gagal", "8. Tidak Ada Kartu", np.zeros((CARD_H, CARD_W, 3), np.uint8),
            "tidak ada kontur segi-4",
            "Pipeline berhenti di sini. Coba turunkan ambang Canny, kecilkan "
            "'Area /', atau naikkan epsilon supaya poligonnya lebih longgar."))
        return stages, None

    # Ambil kandidat terbesar, lalu kembalikan koordinatnya ke skala asli.
    terbesar = max(kandidat, key=cv2.contourArea)
    asli = (terbesar.reshape(4, 2) / params.scale).astype(np.float32)
    src = urutkan_titik(asli)

    vis_titik = frame.copy()
    label = ["KIRI-ATAS", "KANAN-ATAS", "KANAN-BAWAH", "KIRI-BAWAH"]
    warna = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    cv2.polylines(vis_titik, [src.astype(np.int32)], True, (255, 255, 255), 2)
    for i, pt in enumerate(src):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(vis_titik, p, 8, warna[i], -1)
        cv2.putText(vis_titik, f"{i}:{label[i]}", (p[0] + 10, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna[i], 2)
    stages.append(Stage(
        "titik", "8. Urutkan Pojok", vis_titik,
        "urutan: KA -> KnA -> KnB -> KiB",
        "Pojok dari approxPolyDP urutannya acak. Kalau langsung dipakai, hasil "
        "warp bisa terbalik atau terputar. urutkan_titik() memakai x+y untuk "
        "menemukan kiri-atas & kanan-bawah, lalu x-y untuk dua sisanya."))

    dst = np.array([[0, 0], [CARD_W - 1, 0],
                    [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))
    stages.append(Stage(
        "warp", "9. Luruskan (Warp)", warped,
        f"-> {CARD_W}x{CARD_H}",
        "Transformasi perspektif memetakan 4 pojok tadi ke persegi panjang "
        "baku. Kartu yang miring atau terlihat menyerong jadi tegak lurus. "
        "Ini langkah ALIGN — tanpa ini, template matching mustahil akurat."))

    wgray, wthr = deskripsi(warped, params)
    stages.append(Stage(
        "wgray", "10. Kartu Grayscale", wgray,
        "cvtColor BGR -> GRAY",
        "Kartu yang sudah lurus dibuang warnanya, sama seperti langkah 3."))
    stages.append(Stage(
        "desc", "11. Sidik Jari (Threshold)", wthr,
        f"Otsu, ambang awal {params.thresh}",
        "Otsu memilih sendiri ambang terbaik dengan melihat sebaran histogram, "
        "jadi tahan terhadap perubahan pencahayaan. Hasilnya citra biner: "
        "tinta jadi putih, kertas jadi hitam. INI descriptor-nya — untuk penyu, "
        "bagian inilah yang diganti embedding MegaDescriptor."))

    nama, skor, peringkat = cocokkan(wthr, templates, params)

    if peringkat:
        top = peringkat[0][0]
        stages.append(Stage(
            "banding", "12. vs Template Terbaik", templates[top],
            f"template: {top}",
            "Template pemenang, diproses dengan langkah yang persis sama. "
            "Kalau template diproses berbeda dari kartu kamera, "
            "perbandingannya tidak adil dan hasilnya ngawur."))
        stages.append(Stage(
            "selisih", "13. Peta Selisih (SAD)", peta_selisih(wthr, templates[top]),
            f"skor {skor:.4f} (0 = identik)",
            "Merah menandai piksel yang berbeda. SAD menjumlahkan seluruh "
            "selisih itu jadi satu angka, lalu dibagi selisih maksimum yang "
            "mungkin supaya jadi 0..1. Makin sedikit merah, makin cocok. "
            "Inilah 'jarak' — persis peran cosine distance pada embedding."))

    hasil = {
        "nama": nama,
        "skor": skor,
        "peringkat": peringkat[:10],
        "diterima": skor <= params.conf_max,
    }
    return stages, hasil
