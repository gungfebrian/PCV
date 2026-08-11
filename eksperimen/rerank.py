"""
Stage 2 — re-ranking kandidat stage 1 dengan local feature matching.

    Stage 1  MegaDescriptor-L-384 frozen -> cosine -> top-k kandidat
    Stage 2  cocokkan sisik satu per satu -> urutkan ulang top-k

Protokol §3 TIDAK berubah: split tetap deterministik berbasis tahun, sisi
tetap dikunci (kandidat sudah difilter per sisi di stage 1), query identik,
jadi McNemar berpasangan tetap sah.

DUA HAL YANG HARUS DIBACA SEBELUM MEMPERCAYAI ANGKA DI SINI
-----------------------------------------------------------
1. **Plafon re-ranking = recall@k stage 1.** Re-ranking hanya mengurutkan
   ulang kandidat yang sudah ada. Kalau jawaban benar tidak masuk top-k, tidak
   ada matcher yang bisa menolongnya. Di ReunionTurtles galeri per sisi cuma
   84 foto, jadi k=84 berarti me-rank ulang SELURUH galeri dan plafonnya 100%.
   Pertanyaan "k berapa" di sini murni soal waktu komputasi, bukan soal
   keadilan perbandingan.

2. **Hanya ada SATU foto galeri per individu per sisi.** Jadi tiap kandidat
   cuma punya satu pasangan gambar untuk dicocokkan — tidak ada agregasi
   multi-foto yang biasanya membuat local matching jauh lebih stabil.

MATCHER
-------
Yang jalan tanpa jaringan: SIFT, AKAZE, ORB (klasik, tanpa bobot unduhan).
Yang BUTUH bobot dari host yang diblokir di lingkungan ini: ALIKED, XFeat,
RoMa — lihat `MATCHER_TERBLOKIR`. Antarmukanya sudah disiapkan, jadi begitu
bobotnya ada, ketiganya tinggal dicolok tanpa mengubah apa pun yang lain.

    MODEL=L python3 rerank.py --matcher sift --k 84
    MODEL=L python3 rerank.py --uji
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import protokol as P
from evaluasi import breakdown, muat

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")
SISI_PROSES = 800          # sisi terpanjang gambar saat matching
RATIO_LOWE = 0.8
AMBANG_RANSAC = 4.0

MATCHER_TERBLOKIR = {
    "aliked": ("bobot sudah ada di bobot_matcher/aliked-n16.pth, TAPI kornia "
               "0.8.2 yang terpasang tidak punya kelas ALIKED "
               "(`kornia.feature.aliked` tidak ada). Butuh kornia versi lain "
               "atau implementasi resmi ALIKED, yang reponya diblokir di sini."),
    "xfeat": ("bobot XFeat dari github.com/verlab/accelerated_features — 403. "
              "CATATAN: paket PyPI bernama `xfeat` BUKAN XFeat CVPR 2024, "
              "melainkan pustaka feature engineering tabular dari 2020. "
              "Jangan sampai tertukar."),
    "roma": ("bobot roma_outdoor.pth + dinov2_vitl14 sudah ada, TAPI paket "
             "`romatch` tidak bisa dipasang di aarch64 Linux: poselib dan "
             "fused-local-corr tidak punya wheel. Hanya bisa dijalankan di Mac."),
}


def baca_kondisi(path, kondisi):
    """Baca gambar dan terapkan kondisi — SATU jalur untuk semua matcher.

    Menangani dua jenis kondisi sekaligus:
      - berbasis array (resize, CLAHE, ...) lewat `P.KONDISI`
      - berbasis berkas (potongan kepala YOLO) lewat `P.KONDISI_BERKAS`,
        yang potongannya sudah dihitung sebelumnya dan disimpan ke disk

    Mengembalikan None kalau gambarnya tidak ada ATAU kondisi berkas tidak
    punya entri untuk foto itu. Pemanggil harus menangani None secara sadar;
    jatuh diam-diam ke gambar penuh akan mencampur dua kondisi berbeda ke
    dalam satu angka tanpa memunculkan error.
    """
    if kondisi in getattr(P, "KONDISI_BERKAS", {}):
        return P.KONDISI_BERKAS[kondisi](path)
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return P.KONDISI[kondisi](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _inlier(src, dst):
    """Jumlah inlier setelah RANSAC MAGSAC — SATU-SATUNYA definisi skor.

    Kalau pasangan mentahnya kurang dari 4, homografi tidak bisa diestimasi
    sama sekali, jadi yang dikembalikan adalah jumlah pasangan mentah itu
    (0-3). Perilaku ini sengaja dipertahankan persis seperti sebelum
    refactor supaya semua angka yang sudah dilaporkan tetap sah.
    """
    if src is None or len(src) < 4:
        return 0.0 if src is None else float(len(src))
    _, mask = cv2.findHomography(np.float32(src).reshape(-1, 1, 2),
                                 np.float32(dst).reshape(-1, 1, 2),
                                 cv2.USAC_MAGSAC, AMBANG_RANSAC)
    return float(mask.sum()) if mask is not None else 0.0


# ------------------------------------------------------------- matcher
#
# KONTRAK MATCHER — dipakai eksperimen DAN aplikasi.
#
#   .ekstrak(path)          fitur dari berkas          (dipakai eksperimen)
#   .ekstrak_array(rgb)     fitur dari array RGB       (dipakai aplikasi/kamera)
#   .korespondensi(a, b)    -> (src, dst) Nx2 sebelum RANSAC, atau (None, None)
#   .skor(a, b)             -> jumlah inlier setelah RANSAC
#   .KOORD_ASLI             True kalau koordinat yang dikembalikan sudah dalam
#                           ukuran gambar ASLI, bukan ukuran hasil resize
#   .PUNYA_KEYPOINT         False untuk dense matcher (RoMa) yang tidak punya
#                           keypoint per gambar — aplikasi menampilkan "-"
#
# Kontrak ini ada karena aplikasi sebelumnya menebak jenis matcher lewat
# `hasattr(mm, "X")` dan `mm.det`. RoMa tidak punya keduanya, jadi begitu
# dipilih di UI aplikasinya langsung mati dengan
# `AttributeError: 'RoMa' object has no attribute 'det'`. Menebak tipe seperti
# itu akan rusak lagi setiap kali satu matcher baru ditambahkan.
class Klasik:
    KOORD_ASLI = False       # .ekstrak mengembalikan koordinat pada skala resize
    PUNYA_KEYPOINT = True

    """Detector klasik OpenCV. Tanpa bobot unduhan, jadi selalu bisa jalan.

    Skor = jumlah inlier setelah RANSAC, bukan jumlah match mentah. Match
    mentah gampang dipalsukan oleh tekstur berulang (pasir, riak air); inlier
    memaksa pasangan itu konsisten dengan satu transformasi geometris.
    """

    def __init__(self, nama="sift", n=2048, kondisi="raw"):
        self.kondisi = kondisi
        self.nama = nama if kondisi == "raw" else f"{nama}-{kondisi}"
        nama = nama
        if nama == "sift":
            self.det = cv2.SIFT_create(nfeatures=n)
            self.norm = cv2.NORM_L2
        elif nama == "akaze":
            self.det = cv2.AKAZE_create()
            self.norm = cv2.NORM_HAMMING
        elif nama == "orb":
            self.det = cv2.ORB_create(nfeatures=n)
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError(nama)

    def ekstrak(self, path):
        if self.kondisi == "raw":
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            # preprocessing dipakai dari protokol yang sama dengan eksperimen
            # stage-1, bukan ditulis ulang di sini
            rgb = baca_kondisi(path, self.kondisi)
            im = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if im is None:
            return None
        s = SISI_PROSES / max(im.shape)
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.det.detectAndCompute(im, None)
        if des is None or len(kp) < 8:
            return None
        return np.float32([k.pt for k in kp]), des

    def ekstrak_array(self, rgb):
        """Sama dengan .ekstrak tapi dari array RGB di memori (frame kamera).

        Preprocessing TIDAK diterapkan lagi di sini: pemanggil sudah
        memberikan gambar yang sudah dipraproses. Menerapkannya dua kali
        adalah bug yang mudah terjadi dan tidak memunculkan error apa pun.
        """
        im = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = SISI_PROSES / max(im.shape)
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.det.detectAndCompute(im, None)
        if des is None or len(kp) < 8:
            return None
        return np.float32([k.pt for k in kp]), des

    def korespondensi(self, a, b):
        if a is None or b is None:
            return None, None
        m = cv2.BFMatcher(self.norm).knnMatch(a[1], b[1], k=2)
        baik = [x for x, y in m
                if len([x, y]) == 2 and x.distance < RATIO_LOWE * y.distance]
        if not baik:
            return None, None
        return (a[0][[x.queryIdx for x in baik]],
                b[0][[x.trainIdx for x in baik]])

    def skor(self, a, b):
        src, dst = self.korespondensi(a, b)
        return _inlier(src, dst)


class XFeat:
    """XFeat (CVPR 2024). Arsitektur direkonstruksi dari state_dict —
    lihat `xfeat_lokal.py` untuk dua pemeriksaan yang membuktikannya benar.

    Memakai matcher bawaannya (mutual NN + ambang cosine), bukan ratio test
    Lowe seperti SIFT. Yang disamakan antar metode adalah SKOR akhirnya:
    jumlah inlier setelah RANSAC MAGSAC, dengan ambang yang sama persis.
    """

    KOORD_ASLI = True        # xfeat_lokal.ekstrak sudah membagi dengan skala
    PUNYA_KEYPOINT = True

    def __init__(self, kondisi="raw"):
        import xfeat_lokal as X
        self.X = X
        self.model = X.muat()
        self.kondisi = kondisi
        self.nama = "xfeat" if kondisi == "raw" else f"xfeat-{kondisi}"

    def ekstrak(self, path):
        if self.kondisi == "raw":
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            rgb = baca_kondisi(path, self.kondisi)
            im = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if im is None:
            return None
        return self.X.ekstrak(self.model, im, sisi=SISI_PROSES)

    def ekstrak_array(self, rgb):
        return self.X.ekstrak(self.model, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                              sisi=SISI_PROSES)

    def korespondensi(self, a, b):
        if a is None or b is None:
            return None, None
        return self.X.cocokkan(a, b)

    def skor(self, a, b):
        return _inlier(*self.korespondensi(a, b))


class ALIKED:
    """ALIKED lewat kornia, bobot dimuat dari berkas LOKAL.

    Arsitekturnya sudah ada di kornia yang terpasang; yang hilang cuma
    bobotnya. Membiarkan kornia mengunduh sendiri tidak berhasil di sini
    (github release 403), jadi bobotnya dibaca dari `bobot_matcher/`.

    Matcher-nya memakai mutual nearest neighbour + ambang cosine, sama seperti
    XFeat — deskriptor terlatih padat, dan ratio test Lowe membuangnya hampir
    semua (terbukti di XFeat: 2 pasangan saja antar gambar berbeda).
    """

    KOORD_ASLI = True        # .ekstrak sudah membagi pts dengan skala
    PUNYA_KEYPOINT = True

    def __init__(self, kondisi="raw"):
        import inspect
        import kornia.feature as KF
        import torch
        self.torch = torch
        # Signature ALIKED berbeda antar versi kornia: sebagian menerima
        # `pretrained=`, sebagian tidak, dan yang tidak menerimanya melempar
        # `TypeError: got an unexpected keyword argument 'pretrained'` — itu
        # yang terjadi di Mac. Jadi kwarg-nya diperiksa dulu, bukan ditebak.
        sig = inspect.signature(KF.ALIKED.__init__).parameters
        kw = {}
        if "model_name" in sig:
            kw["model_name"] = "aliked-n16"
        if "pretrained" in sig:
            kw["pretrained"] = False       # bobot dimuat manual di bawah
        self.model = KF.ALIKED(**kw)
        p = os.path.join(BOBOT, "aliked-n16.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} tidak ada — jalankan unduh_matcher.py")
        sd = torch.load(p, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        hasil = self.model.load_state_dict(sd, strict=False)
        if len(hasil.missing_keys) > 20:
            raise RuntimeError(
                f"bobot ALIKED tidak cocok dengan kornia: "
                f"{len(hasil.missing_keys)} key hilang — versi kornia berbeda")
        self.model.eval()
        self.kondisi = kondisi
        self.nama = "aliked" if kondisi == "raw" else f"aliked-{kondisi}"

    def ekstrak(self, path):
        import kornia
        im = cv2.imread(path) if self.kondisi == "raw" else None
        if self.kondisi != "raw":
            bgr = cv2.imread(path)
            if bgr is None:
                return None
            rgb = P.KONDISI[self.kondisi](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if im is None:
            return None
        s = SISI_PROSES / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        t = kornia.image_to_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                                   False).float() / 255.0
        with self.torch.no_grad():
            kp, _, des = self.model(t)
        pts = kp[0].numpy() / s
        return pts.astype(np.float32), des[0].numpy().astype(np.float32)

    def ekstrak_array(self, rgb):
        import kornia
        im = rgb
        s = SISI_PROSES / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
        t = kornia.image_to_tensor(im, False).float() / 255.0
        with self.torch.no_grad():
            kp, _, des = self.model(t)
        return ((kp[0].numpy() / s).astype(np.float32),
                des[0].numpy().astype(np.float32))

    def korespondensi(self, a, b):
        if a is None or b is None:
            return None, None
        import xfeat_lokal as X
        return X.cocokkan(a, b, min_cossim=0.8)

    def skor(self, a, b):
        return _inlier(*self.korespondensi(a, b))


class VisMatch:
    """Pembungkus `vismatch` — satu antarmuka untuk 50+ matcher.

    https://github.com/gmberton/vismatch

    Ini menggantikan kelas RoMa buatan sendiri di bawah, dan sekaligus membuka
    LoMa, ALIKED, LoFTR, DeDoDe, dan puluhan lain tanpa menulis satu kelas per
    model. Bobotnya diunduh otomatis dari HuggingFace saat matcher dibuat.

    **HANYA BISA DIJALANKAN DI MAC.** `vismatch` butuh `poselib`, yang tidak
    punya wheel untuk aarch64 Linux — sama persis dengan hambatan `romatch`.
    Jadi kelas ini belum pernah dijalankan dan harus dianggap belum teruji
    sampai lolos uji self-match.

    Nama matcher yang relevan untuk pekerjaan ini:
        sparse : xfeat, aliked-lightglue, xfeat-lightglue, sift-lightglue,
                 dedode, loma, loma-r
        dense  : roma, tiny-roma, minima-roma
        semi   : loftr, eloftr, xfeat-star

    `resize` sengaja dijadikan parameter: sapu ukuran menunjukkan resolusi
    matching adalah tuas terbesar yang ditemukan sejauh ini (256 -> 512
    menambah +15,5 poin Rank-1 pada XFeat).
    """

    KOORD_ASLI = False       # koordinat pada resolusi `resize`, bukan asli
    PUNYA_KEYPOINT = False   # sebagian model vismatch dense: tidak ada kp/gambar

    def __init__(self, nama_model="loma", kondisi="raw", resize=512,
                 perangkat=None):
        try:
            from vismatch import get_matcher
        except ImportError as e:
            raise SystemExit(
                f"paket `vismatch` tidak ada: {e}\n"
                "Di Mac:  ../.venv/bin/pip install vismatch\n"
                "Di sandbox Linux ARM ini memang tidak bisa dipasang — "
                "vismatch butuh poselib yang tidak punya wheel aarch64.")
        import torch
        dev = perangkat or ("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_matcher(nama_model, device=dev)
        self.resize = resize
        self.kondisi = kondisi
        self.nama_model = nama_model
        self.nama = (f"vm-{nama_model}-{resize}" if kondisi == "raw"
                     else f"vm-{nama_model}-{resize}-{kondisi}")

    def ekstrak(self, path):
        """vismatch memuat gambar sendiri. Preprocessing dari protokol tetap
        diterapkan lebih dulu supaya kondisi seperti resize368 atau CLAHE
        memakai fungsi yang sama dengan seluruh eksperimen."""
        if self.kondisi == "raw":
            return self.model.load_image(path, resize=self.resize)
        bgr = cv2.imread(path)
        if bgr is None:
            return None
        rgb = P.KONDISI[self.kondisi](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tmp = os.path.join(BOBOT, "_tmp_praproses.png")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return self.model.load_image(tmp, resize=self.resize)

    def ekstrak_array(self, rgb):
        """vismatch hanya menerima jalur berkas, jadi frame kamera ditulis ke
        berkas sementara. Ini memang I/O per frame — konsekuensi wajar dari
        matcher yang tidak punya jalur masuk dari memori."""
        tmp = os.path.join(BOBOT, "_tmp_frame.png")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return self.model.load_image(tmp, resize=self.resize)

    def korespondensi(self, a, b):
        """Titik inlier dari vismatch, kalau modelnya menyediakannya.

        Nama kuncinya berbeda antar versi, jadi dicari beberapa kemungkinan.
        Kalau tidak ada satu pun, kembalikan kosong — aplikasi cukup tidak
        menggambar garis. Menebak koordinat lebih buruk daripada tidak
        menggambar apa pun.
        """
        if a is None or b is None:
            return None, None
        r = self.model(a, b)
        for k0, k1 in (("inlier_kpts0", "inlier_kpts1"),
                       ("inliers0", "inliers1"),
                       ("mkpts0", "mkpts1")):
            if k0 in r and r[k0] is not None and len(r[k0]):
                return np.float32(r[k0]), np.float32(r[k1])
        return None, None

    def skor(self, a, b):
        """Skor = num_inliers dari vismatch.

        Sengaja memakai angka bawaannya, BUKAN menghitung RANSAC sendiri:
        tiap matcher punya estimator yang cocok untuknya, dan memaksakan satu
        RANSAC seragam justru merugikan sebagian model. Konsekuensinya harus
        disebut saat membandingkan dengan SIFT dan XFeat di repo ini, yang
        skornya dari MAGSAC dengan ambang identik.
        """
        if a is None or b is None:
            return 0.0
        return float(self.model(a, b)["num_inliers"])


class RoMa:
    """RoMa (CVPR 2024) — dense matcher dengan backbone DINOv2.

    BELUM PERNAH DIJALANKAN. Kelas ini ditulis supaya tinggal dipakai di Mac,
    tapi **tidak bisa diverifikasi dari lingkungan eksperimen** karena paket
    `romatch` butuh `poselib` dan `fused-local-corr` yang tidak punya wheel
    untuk aarch64 Linux. Anggap kode di bawah belum teruji sampai ia benar-
    benar jalan sekali dan lolos uji self-match.

    Uji pertama yang WAJIB dilakukan sebelum mempercayai angkanya: cocokkan
    sebuah gambar dengan dirinya sendiri. Kalau hasilnya bukan ratusan
    korespondensi dengan inlier hampir sempurna, ada yang salah — persis
    seperti yang terjadi pada XFeat saat matcher-nya keliru.

    Ongkos: dense matching jauh lebih berat dari XFeat. Perkirakan 1-3 detik
    per pasangan di CPU, jauh lebih cepat di MPS. Untuk k=20 x 168 query =
    3.360 pasangan, itu 1-3 jam di CPU.
    """

    KOORD_ASLI = True        # to_pixel_coordinates mengembalikan koordinat asli
    PUNYA_KEYPOINT = False   # dense: korespondensi hanya ada untuk PASANGAN

    def __init__(self, kondisi="raw", perangkat=None):
        try:
            from romatch import roma_outdoor
        except ImportError as e:
            raise SystemExit(
                f"paket `romatch` tidak ada: {e}\n"
                "Di Mac: ../.venv/bin/pip install romatch\n"
                "Di sandbox Linux ARM ini paketnya memang tidak bisa dipasang "
                "(poselib / fused-local-corr tidak punya wheel aarch64).")
        import torch
        self.torch = torch
        dev = perangkat or ("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = roma_outdoor(device=dev)
        self.dev = dev
        self.kondisi = kondisi
        self.nama = "roma" if kondisi == "raw" else f"roma-{kondisi}"

    def ekstrak(self, path):
        """RoMa mencocokkan PASANGAN gambar sekaligus, tidak mengekstrak
        deskriptor per gambar. Jadi di sini hanya jalurnya yang disimpan;
        pekerjaan sebenarnya terjadi di .skor(). Konsekuensinya tidak ada
        cache deskriptor — itu sebabnya RoMa jauh lebih mahal."""
        return path if os.path.exists(path) else None

    def ekstrak_array(self, rgb):
        """RoMa membaca dari jalur berkas, jadi frame kamera ditulis dulu ke
        berkas sementara. Tidak ada jalan pintas: modelnya memang bekerja per
        PASANGAN gambar, bukan per gambar."""
        tmp = os.path.join(BOBOT, "_tmp_frame_roma.png")
        cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return tmp

    def korespondensi(self, a, b):
        if a is None or b is None:
            return None, None
        from PIL import Image
        with self.torch.no_grad():
            warp, certainty = self.model.match(a, b, device=self.dev)
            match, _ = self.model.sample(warp, certainty)
            kpA, kpB = self.model.to_pixel_coordinates(
                match, *Image.open(a).size[::-1], *Image.open(b).size[::-1])
        return kpA.cpu().numpy(), kpB.cpu().numpy()

    def skor(self, a, b):
        return _inlier(*self.korespondensi(a, b))


class Normalisasi:
    """Bungkus matcher lain dan ganti CARA SKOR DIHITUNG, bukan matchernya.

    Kenapa ini dicoba: analisis kegagalan menunjukkan pasangan yang BENAR
    hampir tidak pernah gagal mendapat inlier (cuma 0-2% yang di bawah 4),
    tapi pasangan yang SALAH sering mendapat inlier lebih banyak lagi.
    Artinya masalahnya bukan "pasangan benar tidak cocok", melainkan
    "pasangan salah terlalu mudah menang".

    Tersangka utamanya: jumlah inlier mentah memberi keuntungan gratis pada
    foto galeri yang bertekstur ramai atau beresolusi besar — foto seperti itu
    menghasilkan banyak korespondensi dengan APA PUN. Sama persis dengan
    masalah hub pada retrieval, dan obatnya juga sama: bagi dengan ukurannya.

        rasio    inlier / min(jumlah keypoint kedua gambar)
        presisi  inlier / jumlah pasangan mentah sebelum RANSAC
                 -> mengukur seberapa konsisten geometrinya, terlepas dari
                    berapa banyak pasangan yang sempat diusulkan
    """

    def __init__(self, dalam, mode="rasio"):
        self.dalam = dalam
        self.mode = mode
        self.nama = f"{dalam.nama}-{mode}"
        self.kondisi = dalam.kondisi
        self.KOORD_ASLI = getattr(dalam, "KOORD_ASLI", True)
        self.PUNYA_KEYPOINT = getattr(dalam, "PUNYA_KEYPOINT", False)

    def ekstrak(self, path):
        return self.dalam.ekstrak(path)

    def ekstrak_array(self, rgb):
        return self.dalam.ekstrak_array(rgb)

    def korespondensi(self, a, b):
        return self.dalam.korespondensi(a, b)

    def skor(self, a, b):
        src, dst = self.korespondensi(a, b)
        inl = _inlier(src, dst)
        if inl <= 0:
            return 0.0
        if self.mode == "presisi":
            n = len(src)
        elif self.PUNYA_KEYPOINT and a is not None and b is not None:
            n = min(len(a[0]), len(b[0]))
        else:
            n = len(src)
        # dikali 1000 supaya tetap terbaca sebagai bilangan, bukan 0,0xx
        return 1000.0 * inl / max(n, 1)


BOBOT = os.path.join(P.BASE, "bobot_matcher")


def bobot_ada(nama):
    """True kalau bobot matcher terlatih sudah tersedia di mesin ini.

    Dipakai supaya UI dan laporan bisa membedakan "belum dijalankan" dari
    "tidak bisa dijalankan" — dan supaya tidak ada slot yang diam-diam diisi
    angka perkiraan.
    """
    if nama == "xfeat":
        return os.path.exists(os.path.join(BOBOT, "xfeat.pt"))
    if nama == "aliked":
        # Bobot saja TIDAK cukup: kornia yang terpasang harus punya kelas
        # ALIKED-nya. kornia 0.8.2 tidak punya (`kornia.feature.aliked` tidak
        # ada). Memeriksa berkas saja akan membuat grid mencoba menjalankannya
        # lalu gagal di tengah jalan.
        if not os.path.exists(os.path.join(BOBOT, "aliked-n16.pth")):
            return False
        try:
            import kornia.feature as KF
            return hasattr(KF, "ALIKED")
        except Exception:
            return False
    if nama == "roma":
        # Bobot saja tidak cukup: paket `romatch` butuh poselib dan
        # fused-local-corr yang tidak punya wheel untuk aarch64 Linux.
        try:
            import romatch  # noqa: F401
        except Exception:
            return False
        return os.path.exists(os.path.join(BOBOT, "roma_outdoor.pth"))
    return True


def buat_matcher(nama, kondisi="raw"):
    # Nama berawalan "vm:" diteruskan ke vismatch, mis. "vm:loma",
    # "vm:roma", "vm:aliked-lightglue". Ukuran opsional: "vm:loma@448".
    if nama.startswith("vm:"):
        spec = nama[3:]
        model, _, uk = spec.partition("@")
        return VisMatch(model, kondisi, resize=int(uk) if uk else 512)
    if nama == "xfeat" and bobot_ada("xfeat"):
        return XFeat(kondisi)
    if nama == "aliked" and bobot_ada("aliked"):
        return ALIKED(kondisi)
    if nama == "roma" and bobot_ada("roma"):
        return RoMa(kondisi)
    if nama in MATCHER_TERBLOKIR and not bobot_ada(nama):
        raise SystemExit(
            f"matcher '{nama}' tidak bisa dijalankan di lingkungan ini:\n"
            f"  {MATCHER_TERBLOKIR[nama]}\n"
            f"Antarmukanya sudah siap — sediakan bobotnya lalu tambahkan kelas "
            f"dengan metode .ekstrak(path) dan .skor(a, b).")
    return Klasik(nama, kondisi=kondisi)


# ------------------------------------------------------- inti re-rank
# Embedding mana yang dipakai stage 1. Default "raw" supaya seluruh angka
# yang sudah dilaporkan tetap sah. Diubah lewat --stage1 untuk menguji apakah
# stage 1 juga terbantu oleh crop kepala.
KONDISI_STAGE1 = os.environ.get("STAGE1", "raw")


def kandidat_stage1(gal, qry, k):
    """Top-k kandidat per query dari stage 1. Sisi sudah dikunci di sini."""
    Eg, Eq = muat(KONDISI_STAGE1, gal, qry)
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])
    S = Eq @ Eg.T
    S[s_q[:, None] != s_g[None, :]] = -np.inf
    urut = np.argsort(-S, axis=1)
    n_sisi = int(np.isfinite(S[0]).sum())
    k = min(k, n_sisi)
    return S, urut[:, :k], k


def jalankan(matcher, k, budget=None):
    """Hitung skor matching untuk tiap pasangan (query, kandidat). Resumable."""
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    S, top, k = kandidat_stage1(gal, qry, k)

    tag = "" if KONDISI_STAGE1 == "raw" else f"_s1-{KONDISI_STAGE1}"
    out = os.path.join(HASIL, f"rerank_{matcher.nama}{tag}_k{k}.npy")
    prog = out + ".progress"
    M = np.load(out) if os.path.exists(out) else np.full((len(qry), k), -1.0, np.float32)
    d = int(open(prog).read()) if os.path.exists(prog) else 0

    cache = {}

    def fitur(i, daftar):
        if i not in cache:
            cache[i] = matcher.ekstrak(daftar[i]["path"])
        return cache[i]

    t0 = time.time()
    n_pas = 0
    while d < len(qry):
        fq = matcher.ekstrak(qry[d]["path"])
        for j in range(k):
            M[d, j] = matcher.skor(fq, fitur(int(top[d, j]), gal))
            n_pas += 1
        d += 1
        np.save(out, M)
        open(prog, "w").write(str(d))
        if budget and time.time() - t0 > budget:
            break
    dt = time.time() - t0
    print(f"{matcher.nama} k={k}: {d}/{len(qry)} query"
          + (f"  ({n_pas} pasangan, {dt / max(n_pas, 1) * 1000:.1f} ms/pasangan)"
             if n_pas else ""))
    return d == len(qry), out, k


# ----------------------------------------------------------- evaluasi
def _skor_dari_urutan(urut, S_asli):
    """Urutan baru -> matriks skor yang menghasilkan urutan itu.

    Metrik TIDAK dihitung di sini. Ia dikembalikan ke `P.metrik_dari_matriks`
    supaya stage-2 memakai jalur kode yang sama persis dengan stage-1 —
    termasuk mask sisi. Percobaan pertama menulis ulang metrik di modul ini,
    dan langsung kehilangan mask itu: mAP stage-1 terbaca 19.52 padahal
    seharusnya 37.40, karena foto sisi seberang ikut dihitung sebagai jawaban
    benar. Tidak ada error, hanya angka yang salah.
    """
    S = np.full_like(S_asli, -np.inf)
    n = urut.shape[1]
    for i in range(urut.shape[0]):
        sah = np.isfinite(S_asli[i, urut[i]])
        S[i, urut[i][sah]] = -np.arange(n)[sah].astype(S.dtype)
    return S


def _tag_s1():
    return "" if KONDISI_STAGE1 == "raw" else f"_s1-{KONDISI_STAGE1}"


def evaluasi_rerank(nama_matcher, k):
    """Bandingkan stage-1 saja vs stage-1 + re-rank, pada query yang sama.

    Dua cara menggabungkan diuji, keduanya TANPA parameter yang disetel di
    data uji — menyetel bobot fusi di test set adalah overfitting yang
    menghasilkan angka bagus dan kesimpulan palsu:
      murni  : urutkan top-k hanya dengan skor matcher, sisanya tetap
      rrf    : reciprocal rank fusion peringkat stage-1 dan peringkat matcher
    """
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    S, top, k = kandidat_stage1(gal, qry, k)
    M = np.load(os.path.join(HASIL, f"rerank_{nama_matcher}{_tag_s1()}_k{k}.npy"))

    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    penuh = np.argsort(-S, axis=1)
    sisa = penuh[:, k:]

    # stage 1 saja — lewat jalur metrik yang sama dengan seluruh eksperimen
    dasar = P.metrik_dari_matriks(S, id_q, id_g)

    # murni: dalam top-k, urutkan dengan skor matcher; seri dipecah oleh
    # peringkat stage-1 supaya hasilnya deterministik
    murni = np.empty_like(penuh)
    rrf = np.empty_like(penuh)
    for i in range(len(qry)):
        s1 = np.arange(k)
        o = np.lexsort((s1, -M[i]))
        murni[i] = np.concatenate([top[i][o], sisa[i]])
        r1 = 1.0 / (60 + s1 + 1)
        r2 = 1.0 / (60 + np.argsort(np.argsort(-M[i])) + 1)
        o2 = np.lexsort((s1, -(r1 + r2)))
        rrf[i] = np.concatenate([top[i][o2], sisa[i]])

    return {"stage1": dasar,
            "murni": P.metrik_dari_matriks(_skor_dari_urutan(murni, S), id_q, id_g),
            "rrf": P.metrik_dari_matriks(_skor_dari_urutan(rrf, S), id_q, id_g),
            }, qry, k


def lapor(nama_matcher, k):
    from statistik import bootstrap_delta, mcnemar
    H, qry, k = evaluasi_rerank(nama_matcher, k)
    dasar = H["stage1"]
    tabel = {}
    for nama, h in H.items():
        b = {"label": nama, "rank1": float(h["rank1"].mean() * 100),
             "rank5": float(h["rank5"].mean() * 100),
             "mAP": float(h["ap"].mean() * 100), "n": int(len(h["ap"]))}
        if nama != "stage1":
            b["delta_rank1"] = bootstrap_delta(
                dasar["rank1"].astype(float) * 100, h["rank1"].astype(float) * 100)
            b["mcnemar_rank1"] = mcnemar(dasar["rank1"].astype(bool),
                                         h["rank1"].astype(bool))
            b["delta_mAP"] = bootstrap_delta(dasar["ap"] * 100, h["ap"] * 100)
        b.update(breakdown(h, qry))
        tabel[nama] = b

    with open(os.path.join(HASIL,
                           f"rerank_{nama_matcher}{_tag_s1()}_k{k}.json"), "w") as f:
        json.dump({"matcher": nama_matcher, "k": k,
                   "cv2": cv2.__version__, "model": P.MODEL,
                   "dataset": P.DATASET,
                   "dataset_hash": P.hash_dataset(P.baca_katalog()),
                   "tabel": tabel}, f, indent=2)

    print(f"\n{nama_matcher}  k={k}  n={tabel['stage1']['n']}")
    print(f"{'':8} {'R-1':>6} {'R-5':>6} {'mAP':>6} "
          f"{'ΔR-1 (95% CI)':>24} {'p':>9}")
    for nama, b in tabel.items():
        if "delta_rank1" not in b:
            print(f"{nama:8} {b['rank1']:6.2f} {b['rank5']:6.2f} {b['mAP']:6.2f} "
                  f"{'—':>24} {'—':>9}")
            continue
        d = b["delta_rank1"]
        ci = f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        tanda = "*" if b["mcnemar_rank1"]["p_value"] < 0.05 else " "
        print(f"{nama:8} {b['rank1']:6.2f} {b['rank5']:6.2f} {b['mAP']:6.2f} "
              f"{ci:>24} {b['mcnemar_rank1']['p_value']:8.3g}{tanda}")
    print("\nΔ mAP:")
    for nama, b in tabel.items():
        if "delta_mAP" not in b:
            continue
        d = b["delta_mAP"]
        print(f"  {nama:8} {d['delta']:+6.2f} [{d['ci95'][0]:+.2f}, "
              f"{d['ci95'][1]:+.2f}]  "
              f"{'signifikan' if d['signifikan'] else 'tidak signifikan'}")
    sp = sorted(tabel["stage1"].get("per_spesies", {}))
    if sp:
        print("\nRank-1 per spesies & sisi:")
        print(f"{'':8} " + " ".join(f"{s:>10}" for s in sp) +
              f" {'kiri':>7} {'kanan':>7}")
        for nama, b in tabel.items():
            print(f"{nama:8} " +
                  " ".join(f"{b['per_spesies'][s]['rank1']:10.2f}" for s in sp) +
                  f" {b['per_sisi']['left']['rank1']:7.2f}"
                  f" {b['per_sisi']['right']['rank1']:7.2f}")
    return tabel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", default="sift")
    ap.add_argument("--k", type=int, default=84)
    ap.add_argument("--budget", type=float, default=35.0)
    ap.add_argument("--kondisi", default="raw")
    ap.add_argument("--lapor", action="store_true")
    ap.add_argument("--skor", default="inlier",
                    choices=["inlier", "rasio", "presisi"],
                    help="cara skor dihitung; lihat kelas Normalisasi")
    a = ap.parse_args()

    if a.lapor:
        lapor(a.matcher, a.k)
    else:
        m = buat_matcher(a.matcher, a.kondisi)
        if a.skor != "inlier":
            m = Normalisasi(m, a.skor)
        selesai, _, k = jalankan(m, a.k, budget=a.budget)
        if selesai:
            lapor(m.nama, k)
