"""
Tes tata letak aplikasi live — tanpa layar, tanpa kamera, tanpa model.

`tampilan.py` sengaja dibuat murni supaya bisa diuji begini. Yang diuji:
kanvas terbentuk pada ukuran yang benar, area gambar tidak tertimpa sidebar,
dan **area klik cocok dengan yang tergambar** — pergeseran satu piksel di sini
membuat tombol menekan hal yang salah tanpa ada error.

    python3 uji_tampilan.py            # tes
    python3 uji_tampilan.py --png      # + tulis pratinjau.png untuk dilihat
"""

import sys
import traceback

import numpy as np

import tampilan as T

_daftar = []


def uji(nama):
    def bungkus(fn):
        _daftar.append((nama, fn))
        return fn
    return bungkus


def state_uji():
    return {
        "kondisi": "wb",
        # Sengaja mencerminkan daftar kondisi yang SEBENARNYA dipakai (11),
        # bukan versi pendek. Tes tata letak yang memakai daftar lebih pendek
        # dari kenyataan tidak akan pernah menangkap masalah overflow.
        "kondisi_pilihan": [("raw", "Raw (baseline)"), ("crop", "Crop kepala"),
                            ("wb", "White balance"), ("clahe", "CLAHE"),
                            ("gray", "Grayscale"), ("crop_wb", "Crop + WB"),
                            ("resize368", "Resize seragam 368x368"),
                            ("resize256", "Resize seragam 256x256"),
                            ("resize320", "Resize seragam 320x320"),
                            ("resize448", "Resize seragam 448x448"),
                            ("resize512", "Resize seragam 512x512")],
        "matcher": "xfeat",
        "matcher_pilihan": [("sift", "SIFT"), ("xfeat", "XFEAT")],
        "sumber": "dataset",
        "sumber_pilihan": [("kamera", "kamera / foto"),
                           ("dataset", "jelajah dataset")],
        "info_query": "8/168  Calvin 2021",
        "geser": 0,
        "rerank": "rrf",
        "rerank_pilihan": [("off", "mati (stage-1 saja)"),
                           ("murni", "murni — skor inlier"),
                           ("rrf", "RRF — cosine + inlier")],
        "sisi": "left",
        "ambang": 0.45,
        "bbox": True,
        "keypoint": False,
        "match": False,
        "jeda": False,
        "tahap": "Asli",
    }


def telemetry_uji():
    return {"ms_infer": 61.4, "ms_pra": 2.1, "fps": 14.3,
            "input_w": 1280, "input_h": 720, "ukuran": "384x384",
            "n_gallery": 84, "n_id": 84, "cos1": 0.6261,
            "ms_match": 41.2, "n_kp": 1873, "inlier": 12}


def top5_uji():
    rng = np.random.default_rng(0)
    return [{"nama": n, "skor": s, "inlier": 20 - 3 * i,
             "img": rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)}
            for i, (n, s) in enumerate([("Zippy", 0.626), ("Taleen", 0.591), ("Samy", 0.544),
                         ("Baguette", 0.502), ("Enez", 0.481)])]


def frame_uji(h=720, w=1280):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------ tes
@uji("kanvas: ukuran keluaran persis seperti yang diminta")
def _():
    for lebar, tinggi in [(1280, 760), (1600, 900), (1024, 640)]:
        k = T.susun(frame_uji(), state_uji(), telemetry_uji(),
                    {"nama": "Zippy", "skor": 0.626, "margin": 0.035},
                    top5_uji(), lebar, tinggi)
        assert k.shape == (tinggi, lebar, 3), f"{k.shape} != {(tinggi, lebar, 3)}"
        assert k.dtype == np.uint8
    return "3 ukuran, bentuk & dtype benar"


@uji("kanvas: tidak crash saat frame kosong / galeri kosong / tanpa hasil")
def _():
    k = T.susun(None, state_uji(), telemetry_uji() | {"cos1": None},
                None, [], 1280, 760)
    assert k.shape == (760, 1280, 3)
    return "frame None + top5 kosong ditangani"


@uji("tata letak: isi frame terkurung di area gambar, tidak bocor ke panel")
def _():
    """Tidak memakai deteksi warna — penanda UI bisa kebetulan sewarna dengan
    frame dan memberi alarm palsu. Yang diuji: render dua frame berbeda, lalu
    pastikan piksel yang BERUBAH hanya ada di dalam area gambar. Kalau frame
    bocor ke sidebar atau strip, piksel di sana ikut berubah."""
    lebar, tinggi = 1280, 760
    st, tel = state_uji(), telemetry_uji()
    a = T.susun(np.full((400, 600, 3), 20, np.uint8), st, tel, None, [],
                lebar, tinggi)
    b = T.susun(np.full((400, 600, 3), 230, np.uint8), st, tel, None, [],
                lebar, tinggi)
    beda = (a != b).any(axis=2)

    bocor_sidebar = int(beda[:, :T.LEBAR_SIDEBAR].sum())
    assert bocor_sidebar == 0, f"{bocor_sidebar} piksel frame bocor ke sidebar"
    bocor_strip = int(beda[tinggi - T.TINGGI_STRIP:, :].sum())
    assert bocor_strip == 0, f"{bocor_strip} piksel frame bocor ke strip top-5"

    di_area = int(beda[:tinggi - T.TINGGI_STRIP, T.LEBAR_SIDEBAR:-T.LEBAR_PANEL].sum())
    assert di_area > 10000, f"frame nyaris tidak tergambar ({di_area} piksel)"
    return f"{di_area} piksel berubah, semuanya di dalam area gambar"


@uji("tata letak: area gambar memang mendapat porsi terbesar")
def _():
    lebar, tinggi = 1280, 760
    luas_total = lebar * tinggi
    luas_gambar = (lebar - T.LEBAR_SIDEBAR) * (tinggi - T.TINGGI_STRIP)
    rasio = luas_gambar / luas_total
    assert rasio > 0.60, f"area gambar cuma {rasio:.0%} — kontrol terlalu makan tempat"
    return f"area gambar {rasio:.0%} dari kanvas"


@uji("klik: setiap baris yang tergambar bisa diklik dan mengembalikan kunci")
def _():
    st = state_uji()
    baris = T.bangun_baris_sidebar(st)
    geo = T.geometri_sidebar(baris, 760)
    bisa_diklik = [b for b in baris if b[2] is not None]
    assert len(geo) == len(bisa_diklik), \
        f"{len(geo)} area klik vs {len(bisa_diklik)} baris — tidak sinkron"
    for i, y0, y1 in geo:
        tengah = (y0 + y1) // 2
        got = T.klik_sidebar(st, 100, tengah, 760)
        assert got == baris[i][2], \
            f"klik di y={tengah} -> {got}, harusnya {baris[i][2]}"
    return f"{len(geo)} area klik semuanya cocok dengan yang tergambar"


@uji("klik: di luar sidebar dan di celah antar baris mengembalikan None")
def _():
    st = state_uji()
    assert T.klik_sidebar(st, T.LEBAR_SIDEBAR + 5, 100, 760) is None, \
        "klik di area gambar tidak boleh menekan tombol"
    assert T.klik_sidebar(st, 100, 5, 760) is None, "klik di judul"
    return "None di luar area kontrol"


@uji("klik: semua kondisi preprocessing bisa dipilih")
def _():
    st = state_uji()
    baris = T.bangun_baris_sidebar(st)
    kondisi = {b[2][1] for b in baris if b[2] and b[2][0] == "kondisi"}
    harap = {k for k, _ in st["kondisi_pilihan"]}
    assert kondisi == harap, f"{kondisi} != {harap}"
    return f"{len(kondisi)} kondisi bisa diklik"


@uji("sidebar: kondisi aktif ditandai berbeda dari yang tidak aktif")
def _():
    a = T.susun(frame_uji(), state_uji() | {"kondisi": "raw"},
                telemetry_uji(), None, [], 1280, 760)
    b = T.susun(frame_uji(), state_uji() | {"kondisi": "gray"},
                telemetry_uji(), None, [], 1280, 760)
    beda = (a[:, :T.LEBAR_SIDEBAR] != b[:, :T.LEBAR_SIDEBAR]).any(axis=2).sum()
    assert beda > 200, f"sidebar nyaris tidak berubah ({beda} piksel) — " \
                       "penanda aktif tidak terlihat"
    return f"{beda} piksel berbeda antar pilihan"


@uji("putusan: TIDAK DIKENAL saat skor di bawah ambang, nama saat di atas")
def _():
    st = state_uji() | {"ambang": 0.60}
    bawah = T.susun(frame_uji(), st, telemetry_uji(),
                    {"nama": "Zippy", "skor": 0.40, "margin": 0.01},
                    [], 1280, 760)
    atas = T.susun(frame_uji(), st, telemetry_uji(),
                   {"nama": "Zippy", "skor": 0.90, "margin": 0.20},
                   [], 1280, 760)
    assert (bawah != atas).any(), "tampilan tidak berubah di sekitar ambang"
    return "ambang mengubah putusan"


@uji("strip: top-5 tetap tampil walau putusannya TIDAK DIKENAL")
def _():
    """Ini poin review yang paling penting: kotak 'TIDAK DIKENAL' tanpa
    kandidat membuang informasi yang justru dibutuhkan saat analisis gagal."""
    st = state_uji() | {"ambang": 0.99}      # semua di bawah ambang
    k = T.susun(frame_uji(), st, telemetry_uji(),
                {"nama": "Zippy", "skor": 0.62, "margin": 0.03},
                top5_uji(), 1280, 760)
    strip = k[760 - T.TINGGI_STRIP:, T.LEBAR_SIDEBAR:-T.LEBAR_PANEL]
    kosong = T.susun(frame_uji(), st, telemetry_uji(), None, [], 1280, 760)
    strip_kosong = kosong[760 - T.TINGGI_STRIP:, T.LEBAR_SIDEBAR:-T.LEBAR_PANEL]
    assert (strip != strip_kosong).any(), "thumbnail top-5 tidak tergambar"
    return "5 kandidat tetap tampil di bawah ambang"


@uji("strip: label nama & skor muat di dalam kanvas, tidak terpotong")
def _():
    """Label skor sempat terpotong keluar kanvas karena tinggi thumbnail
    dihitung tanpa menyisakan ruang untuk DUA baris teks. Tidak ada error —
    hanya angka yang hilang dari layar. Karena itu dikunci di sini."""
    lebar, tinggi = 1280, 760
    k = T.susun(frame_uji(), state_uji(), telemetry_uji(),
                {"nama": "Zippy", "skor": 0.626, "margin": 0.035},
                top5_uji(), lebar, tinggi)
    # dua baris teks terakhir berada di y0+26+kh+12 dan +24
    y0 = tinggi - T.TINGGI_STRIP
    h_total = T.TINGGI_STRIP
    kh = h_total - 26 - 32
    baris_terakhir = y0 + 26 + kh + 24
    assert baris_terakhir <= tinggi - 2, \
        f"baris skor di y={baris_terakhir}, kanvas cuma {tinggi} — terpotong"
    # dan piksel teksnya memang ada (baris itu tidak seragam warna panel)
    pita = k[baris_terakhir - 9:baris_terakhir + 2, T.LEBAR_SIDEBAR:-T.LEBAR_PANEL]
    assert len(np.unique(pita.reshape(-1, 3), axis=0)) > 1, \
        "baris skor kosong — teks tidak tergambar"
    return f"baris skor terakhir y={baris_terakhir} < {tinggi}"


@uji("tahap: strip pipeline lengkap dan tiap tahap bisa diklik")
def _():
    """Review lama meminta 'proses lengkapnya' terlihat, bukan cuma hasil
    akhir. Tiap tahap harus bisa dipilih untuk dilihat besar."""
    tahap = [(n, frame_uji(200, 260)) for n in
             ("Asli", "Praproses", "Tepi", "Align", "Keypoint")]
    lebar = 1500 - T.LEBAR_SIDEBAR - T.LEBAR_PANEL
    geo = T.geometri_tahap(tahap, T.LEBAR_SIDEBAR, 0, lebar)
    assert len(geo) == 5, f"cuma {len(geo)} tahap tergambar dari 5"
    for nama, x0, y0, x1, y1 in geo:
        got = T.klik_tahap(tahap, (x0 + x1) // 2, (y0 + y1) // 2,
                           T.LEBAR_SIDEBAR, 0, lebar)
        assert got == ("tahap", nama), f"klik {nama} -> {got}"
    return f"{len(geo)} tahap, semua area klik cocok"


@uji("overlay: bbox & keypoint benar-benar tergambar dan bisa dimatikan")
def _():
    st = state_uji()
    tahap = [("Asli", frame_uji(300, 400))]
    dasar = T.susun(frame_uji(300, 400), st | {"bbox": False, "keypoint": False},
                    telemetry_uji(), None, [], 1500, 880, tahap=tahap,
                    bbox=(50, 40, 200, 150),
                    keypoint=np.array([[60., 50.], [120., 90.], [200., 140.]]))
    dg_bbox = T.susun(frame_uji(300, 400), st | {"bbox": True, "keypoint": False},
                      telemetry_uji(), None, [], 1500, 880, tahap=tahap,
                      bbox=(50, 40, 200, 150), keypoint=None)
    dg_kp = T.susun(frame_uji(300, 400), st | {"bbox": False, "keypoint": True},
                    telemetry_uji(), None, [], 1500, 880, tahap=tahap,
                    bbox=None,
                    keypoint=np.array([[60., 50.], [120., 90.], [200., 140.]]))
    assert (dasar != dg_bbox).any(), "bbox tidak tergambar"
    assert (dasar != dg_kp).any(), "keypoint tidak tergambar"
    return "keduanya tergambar dan bisa dimatikan terpisah"


@uji("match: panel garis inlier menggantikan tampilan utama saat aktif")
def _():
    """Panel paling informatif: kalau garis mendarat di karang bukan di sisik,
    penyebab kegagalan terlihat tanpa membaca angka."""
    st = state_uji() | {"match": True}
    pas = [((30., 40.), (35., 45.)), ((80., 90.), (85., 95.))]
    a = T.susun(frame_uji(300, 400), st, telemetry_uji(), None, [], 1500, 880,
                pasangan=pas, gambar_kandidat=frame_uji(300, 400))
    b = T.susun(frame_uji(300, 400), st | {"match": False}, telemetry_uji(),
                None, [], 1500, 880, pasangan=pas,
                gambar_kandidat=frame_uji(300, 400))
    assert (a != b).any(), "panel match tidak berubah saat diaktifkan"
    return "panel match aktif/nonaktif"


@uji("strip: jumlah inlier ikut ditampilkan saat stage-2 hidup")
def _():
    st = state_uji()
    dgn = T.susun(frame_uji(), st, telemetry_uji(), None, top5_uji(), 1500, 880)
    tanpa = T.susun(frame_uji(), st, telemetry_uji(), None,
                    [dict(t, inlier=None) for t in top5_uji()], 1500, 880)
    y0 = 880 - T.TINGGI_STRIP
    assert (dgn[y0:] != tanpa[y0:]).any(), "angka inlier tidak tergambar"
    return "inlier tampil di strip"


@uji("panel: akurasi dari statistik.json, bukan angka yang diketik")
def _():
    """Persentase hardcoded di visualizer lama sudah lama tidak cocok dengan
    hasil sebenarnya. Panel harus kosong kalau datanya tidak ada."""
    kosong = T.susun(frame_uji(), state_uji(), telemetry_uji(), None, [],
                     1500, 880, ringkas=None)
    isi = T.susun(frame_uji(), state_uji(), telemetry_uji(), None, [],
                  1500, 880, ringkas=[("Rank-1", "25.00%"), ("mAP", "37.40%")])
    x = 1500 - T.LEBAR_PANEL
    assert (kosong[:, x:] != isi[:, x:]).any(), "panel akurasi tidak berubah"
    return "kosong vs terisi berbeda"


@uji("teks: karakter non-ASCII disaring, tidak muncul sebagai '?'")
def _():
    """Font Hershey OpenCV hanya ASCII. Em-dash di label membuat
    'STAGE 2 - RE-RANK' tergambar jadi 'STAGE 2 ??? RE-RANK'. Tidak ada error,
    hanya UI yang terlihat rusak."""
    for masuk, harap in [("STAGE 2 — RE-RANK", "STAGE 2 - RE-RANK"),
                         ("murni — skor inlier", "murni - skor inlier"),
                         ("input → model", "input -> model"),
                         ("Δ Rank-1 ±2", "d Rank-1 +/-2")]:
        got = T.ascii_aman(masuk)
        assert got == harap, f"{masuk!r} -> {got!r}, harusnya {harap!r}"
        assert got.isascii(), f"{got!r} masih non-ASCII"
    # semua label yang benar-benar dipakai sidebar harus lolos
    for jenis, label, kunci, aktif in T.bangun_baris_sidebar(state_uji()):
        assert T.ascii_aman(label).isascii()
        assert "?" not in T.ascii_aman(label), f"label '{label}' jadi '?'"
    return "em-dash & panah diganti, semua label sidebar bersih"


@uji("keypoint: titik tersebar di seluruh gambar, tidak mengkerut ke pojok")
def _():
    """Bug nyata yang ketahuan dari layar, bukan dari error.

    Ekstraktor mengembalikan koordinat dalam ukuran gambar ASLI. Kode tampilan
    sempat membaginya lagi dengan faktor resize; untuk foto yang lebih kecil
    dari SISI_PROSES faktornya >1, sehingga SEMUA titik mengkerut ke pojok
    kiri-atas. Tidak ada exception, hanya overlay yang salah.

    Uji: gambar titik yang tersebar di seluruh bidang, lalu pastikan piksel
    kuning yang tergambar juga tersebar — bukan terkumpul di satu kuadran.
    """
    h, w = 400, 600
    kanvas = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(10, w - 10, 400),
                    rng.uniform(10, h - 10, 400)], axis=1)
    T.gambar_keypoint(kanvas, pts, 1.0, 0, 0)
    ys, xs = np.where(kanvas.any(axis=2))
    assert len(xs) > 0, "tidak ada titik tergambar"
    # sebaran harus menutupi mayoritas bidang, bukan satu pojok
    assert xs.max() > 0.75 * w and ys.max() > 0.75 * h, \
        f"titik cuma sampai ({xs.max()}, {ys.max()}) dari ({w}, {h}) — mengkerut"
    kiri_atas = ((xs < w / 2) & (ys < h / 2)).mean()
    assert kiri_atas < 0.45, \
        f"{kiri_atas:.0%} titik menumpuk di kuadran kiri-atas — skala ganda"

    # dan skala 0.5 memang HARUS mengecilkan (kontrol positif)
    kanvas2 = np.zeros((h, w, 3), np.uint8)
    T.gambar_keypoint(kanvas2, pts, 0.5, 0, 0)
    ys2, xs2 = np.where(kanvas2.any(axis=2))
    assert xs2.max() < xs.max() * 0.65, "skala tidak berpengaruh sama sekali"
    return f"tersebar sampai ({xs.max()}, {ys.max()}), kiri-atas {kiri_atas:.0%}"


@uji("keypoint: koordinat dipetakan dari gambar praproses ke frame asli")
def _():
    """Bug kedua yang ketahuan dari layar. Keypoint diekstrak dari gambar
    SETELAH preprocessing (mis. resize368 -> 368x368), tapi digambar di atas
    frame ASLI yang ukurannya lain. Tanpa pemetaan, titik meluber keluar
    gambar atau menumpuk di satu sudut.

    Kondisi seperti resize368 mengubah lebar dan tinggi dengan faktor BERBEDA,
    jadi skalanya harus per sumbu — satu angka skala tidak cukup.
    """
    for (fw, fh), (pw, ph) in [((253, 227), (368, 368)),
                               ((800, 600), (368, 368)),
                               ((640, 480), (512, 512))]:
        rng = np.random.default_rng(0)
        kp = np.stack([rng.uniform(0, pw, 300), rng.uniform(0, ph, 300)], 1)
        kp_asli = kp * np.array([fw / pw, fh / ph], np.float32)
        assert kp_asli[:, 0].max() <= fw + 1, \
            f"x meluber: {kp_asli[:, 0].max():.0f} > {fw}"
        assert kp_asli[:, 1].max() <= fh + 1, \
            f"y meluber: {kp_asli[:, 1].max():.0f} > {fh}"
        # dan harus benar-benar memakai lebar penuh, bukan mengkerut
        assert kp_asli[:, 0].max() > 0.8 * fw and kp_asli[:, 1].max() > 0.8 * fh, \
            "titik mengkerut, tidak memakai bidang penuh"
    return "3 kombinasi ukuran, semua muat dan mengisi penuh"


@uji("panel: pesan galat menyebut perintah perbaikannya, bukan 'belum ada'")
def _():
    """Kalau embedding galeri untuk sebuah kondisi belum dihitung, panel harus
    memberi tahu PERINTAH yang memperbaikinya. 'belum ada' saja membuat alat
    ini selalu perlu ditanyakan ke orang lain."""
    tel = telemetry_uji() | {
        "galat": "embedding galeri untuk 'resize368' belum ada - jalankan: "
                 "MODEL=T python3 jalankan.py resize368"}
    dgn = T.susun(frame_uji(), state_uji(), tel, None, [], 1500, 880)
    tanpa = T.susun(frame_uji(), state_uji(), telemetry_uji(), None, [],
                    1500, 880)
    x = 1500 - T.LEBAR_PANEL
    assert (dgn[:, x:] != tanpa[:, x:]).any(), "pesan galat tidak tergambar"
    return "galat tampil di panel putusan"


@uji("sidebar: bisa di-scroll dan area klik ikut bergeser")
def _():
    """Sidebar sekarang punya 11 kondisi preprocessing + matcher + mode, jadi
    isinya melebihi tinggi layar. Yang paling berbahaya bukan isinya terpotong,
    tapi kalau geseran hanya diterapkan saat MENGGAMBAR dan tidak saat
    hit-test klik — tombol akan menekan hal yang salah tanpa error apa pun."""
    st = state_uji()
    baris = T.bangun_baris_sidebar(st)
    maks = T.batas_geser(baris, 880)
    assert maks > 0, "isi sidebar tidak melebihi layar, tes ini tidak bermakna"

    for geser in (0, 60, maks):
        st["geser"] = geser
        geo = T.geometri_sidebar(baris, 880, geser)
        for i, y0, y1 in geo:
            tengah = (y0 + y1) // 2
            if tengah < 44 or tengah > 880:      # tergulung keluar layar
                continue
            got = T.klik_sidebar(st, 100, tengah, 880)
            assert got == baris[i][2], \
                f"geser={geser}: klik y={tengah} -> {got}, harusnya {baris[i][2]}"

    # menggulung harus benar-benar mengubah tampilan
    a = T.susun(frame_uji(), state_uji() | {"geser": 0}, telemetry_uji(),
                None, [], 1500, 880)
    b = T.susun(frame_uji(), state_uji() | {"geser": maks}, telemetry_uji(),
                None, [], 1500, 880)
    assert (a[:, :T.LEBAR_SIDEBAR] != b[:, :T.LEBAR_SIDEBAR]).any(), \
        "sidebar tidak berubah saat digulung"
    return f"batas geser {maks}px, area klik cocok di 3 posisi"


@uji("panel: opsi terbaik tampil dan menandai apakah sedang dipakai")
def _():
    t1 = telemetry_uji() | {"terbaik": {
        "label": "xfeat + resize512 - murni (k=50)", "rank1": 75.0,
        "rank5": 83.33, "mAP": 79.0, "aktif": True}}
    t2 = telemetry_uji() | {"terbaik": dict(t1["terbaik"], aktif=False)}
    kosong = T.susun(frame_uji(), state_uji(), telemetry_uji(), None, [],
                     1500, 880)
    aktif = T.susun(frame_uji(), state_uji(), t1, None, [], 1500, 880)
    tidak = T.susun(frame_uji(), state_uji(), t2, None, [], 1500, 880)
    x = 1500 - T.LEBAR_PANEL
    assert (kosong[:, x:] != aktif[:, x:]).any(), "panel opsi terbaik tidak tampil"
    assert (aktif[:, x:] != tidak[:, x:]).any(), \
        "tidak membedakan 'sedang dipakai' dari 'bukan yang dipakai'"
    return "tampil, dan status aktif/tidak dibedakan"


@uji("kamera IP: URL salah ketik dibetulkan otomatis")
def _():
    """Tiga kesalahan URL kamera IP yang paling sering terjadi, dan ketiganya
    gagal tanpa pesan berguna dari OpenCV: `https:` (DroidCam melayani HTTP
    polos), kurang `//`, dan kurang path `/video`."""
    import sys as _s
    _s.path.insert(0, "../eksperimen")
    from penyu_live import normalisasi_url as n
    kasus = [
        ("https:10.64.53.103:4747", "http://10.64.53.103:4747/video"),
        ("http://10.64.53.103:4747", "http://10.64.53.103:4747/video"),
        ("10.64.53.103:4747", "http://10.64.53.103:4747/video"),
        ("http://192.168.1.5:8080", "http://192.168.1.5:8080/video"),
        # yang sudah benar tidak boleh diutak-atik
        ("http://10.64.53.103:4747/video", "http://10.64.53.103:4747/video"),
        ("http://192.168.1.5:8081/mjpeg", "http://192.168.1.5:8081/mjpeg"),
        ("rtsp://cam.local/stream", "rtsp://cam.local/stream"),
    ]
    for masuk, harap in kasus:
        got = n(masuk)
        assert got == harap, f"{masuk!r} -> {got!r}, harusnya {harap!r}"
    return f"{len(kasus)} bentuk URL ditangani"


@uji("muat_pas: rasio aspek terjaga, tidak ada gambar yang dipaksa gepeng")
def _():
    for h, w in [(100, 400), (400, 100), (200, 200)]:
        im = np.full((h, w, 3), 200, np.uint8)
        out, skala, ox, oy = T.muat_pas(im, 300, 300)
        assert out.shape == (300, 300, 3)
        assert abs(skala - min(300 / w, 300 / h)) < 1e-6, "skala salah"
        isi = (out == 200).all(axis=2)
        ys, xs = np.where(isi)
        rasio_asli = w / h
        rasio_out = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        assert abs(rasio_out - rasio_asli) / rasio_asli < 0.05, \
            f"{w}x{h}: rasio berubah {rasio_asli:.2f} -> {rasio_out:.2f}"
    return "rasio aspek terjaga di 3 bentuk"


def main():
    lolos = gagal = 0
    for nama, fn in _daftar:
        try:
            catatan = fn()
            lolos += 1
            print(f"  \033[32mOK\033[0m   {nama}")
            if "-v" in sys.argv and catatan:
                print(f"       {catatan}")
        except Exception as e:
            gagal += 1
            print(f"  \033[31mGAGAL\033[0m {nama}")
            print(f"       {type(e).__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()

    if "--png" in sys.argv:
        import cv2
        k = T.susun(frame_uji(), state_uji(), telemetry_uji(),
                    {"nama": "Zippy", "skor": 0.626, "margin": 0.035},
                    top5_uji(), 1280, 760)
        cv2.imwrite("pratinjau.png", k)
        print("\npratinjau.png ditulis")

    print(f"\n{lolos} lolos, {gagal} gagal")
    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
