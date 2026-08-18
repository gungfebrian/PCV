"""
PROTOKOL TERKUNCI — split, matching, metrik.

Ini satu-satunya tempat protokol §3 didefinisikan. Semua kondisi preprocessing
memanggil fungsi yang sama persis dari sini, sehingga tidak ada jalan bagi satu
kondisi untuk diam-diam memakai setup berbeda.

Aturan yang dikunci (spesifikasi §3):
    gallery   = semua foto tahun PERTAMA
    query     = semua foto tahun KEDUA (terakhir)
    split     = deterministik berbasis tahun, tanpa seed, tanpa acak
    matching  = cosine similarity atas embedding yang sudah di-L2 normalize
    sisi      = query kiri HANYA dicari di gallery kiri, kanan ke kanan
    model     = MegaDescriptor, bobot dan versi sama di semua run

DUA DATASET
-----------
`DATASET=reunion` (default) — ReunionTurtles sesuai spesifikasi: 84 individu,
336 foto, 50 hijau + 34 sisik, profil L/R di dua tahun. Foto UTUH bawah air
dengan latar karang. Punya kolom spesies, jadi breakdown §4 lengkap.

`DATASET=seaturtleheads` — SeaTurtleIDHeads, dipakai saat ReunionTurtles belum
tersedia. Skala jauh lebih besar (1.246 query) tapi tidak punya spesies dan
fotonya sudah berupa crop kepala. Dipertahankan sebagai pembanding skala.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
AKAR_HASIL = os.path.join(BASE, "hasil")
DATASET = os.environ.get("DATASET", "reunion")
DATA = os.path.join(REPO, "dataset_penyu",
                    {"reunion": "ReunionTurtles",
                     "seaturtleheads": "SeaTurtleIDHeads",
                     "zakynthos": "Zakynthos",
                     "amvrakikos": "AmvrakikosTurtles"}[DATASET])

SISI = ("left", "right")          # ketat: topleft/topright/top/front/below dibuang

# Bobot dibaca dari cache HF lokal, tanpa jaringan. HF_HOME dipakai kalau ada.
_KANDIDAT_HF = [
    os.environ.get("HF_HOME"),
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    "/sessions/optimistic-vibrant-cerf/mnt/huggingface",
]


# Backbone global. T-224 dan L-384 adalah varian MegaDescriptor: T cepat,
# L jauh lebih akurat tapi ~20x lebih berat di CPU. Keduanya sudah ada di
# cache HF lokal.
#
# MIEWID adalah backbone KEDUA, bukan pengganti. Ditambahkan 2026-08-18 setelah
# repo turtle-identification-be mengukur MegaDescriptor sebagai global ranker
# yang lemah untuk same-side head photo-ID (Reunion Green MD 23% vs MiewID 94%)
# — dan angka MD mereka cocok dengan angka kita (22.0% / 29.4%), jadi itu bukan
# bug implementasi melainkan pilihan model. Lihat
# docs/temuan/2026-08-18-miewid-vs-megadescriptor.md.
#
# Protokol §3 TIDAK dilanggar selama satu aturan dipegang: tiap kondisi
# dibandingkan HANYA dengan baseline dari backbone yang sama. dir_hasil()
# sudah memisahkan folder per (dataset, model, transform), jadi run MIEWID
# tidak bisa mengontaminasi angka MegaDescriptor.
MODEL = os.environ.get("MODEL", "T")
NAMA_HF = {"T": "BVRA/MegaDescriptor-T-224",
           "L": "BVRA/MegaDescriptor-L-384",
           "MIEWID": "conservationxlabs/miewid-msv3"}[MODEL]
REPO_HF = "models--" + NAMA_HF.replace("/", "--")


def cari_snapshot(repo=None):
    repo = repo or REPO_HF
    for akar in _KANDIDAT_HF:
        if not akar:
            continue
        d = os.path.join(akar, "hub", repo, "snapshots")
        if os.path.isdir(d):
            for s in sorted(os.listdir(d)):
                if os.path.exists(os.path.join(d, s, "config.json")):
                    return os.path.join(d, s)
    raise FileNotFoundError(
        f"cache HF untuk {repo} tidak ditemukan di {[k for k in _KANDIDAT_HF if k]}")


SNAP_T = None                     # diisi malas oleh muat_model()

# Dibaca dari config.json varian yang dipakai, BUKAN diketik manual —
# memakai angka lain akan menggeser seluruh distribusi embedding tanpa error.
_CFG = json.load(open(os.path.join(cari_snapshot(), "config.json")))
if MODEL == "MIEWID":
    # Pengecualian yang disengaja. config.json MiewID TIDAK punya
    # `pretrained_cfg`: ia model HF custom (trust_remote_code), bukan model
    # timm, jadi tidak ada yang bisa dibaca dari sana. Sumber angka di bawah
    # adalah implementasi rujukan di repo turtle-identification-be:
    #   app/core/config.py:41              -> miewid_img_size = 440
    #   app/services/ml/miewid_embedder.py -> Resize([440, 440]) + ToTensor()
    #                                         + Normalize(mean/std ImageNet)
    # Resize dengan dua angka di torchvision = squash tanpa center-crop, jadi
    # TRANSFORM="squash" milik kita memang jalur yang sama; CROP_PCT dipaksa
    # 1.0 supaya transform_cfg tidak diam-diam memotong tepi.
    #
    # DIM 2152 di-hardcode di hulu (`final_in_features` di
    # modeling_miewid.py). muat_model() memverifikasinya ke lebar output
    # sungguhan, jadi kalau hulu berubah kita dapat error, bukan angka yang
    # pelan-pelan salah.
    UKURAN = 440
    CROP_PCT = 1.0
    DIM = 2152
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
else:
    UKURAN = _CFG["pretrained_cfg"]["input_size"][1]
    CROP_PCT = _CFG["pretrained_cfg"]["crop_pct"]
    DIM = _CFG["num_features"]
    MEAN = np.array(_CFG["pretrained_cfg"]["mean"], np.float32)
    STD = np.array(_CFG["pretrained_cfg"]["std"], np.float32)


# --------------------------------------------------------------- dataset
def _katalog_reunion(data):
    """data.csv -> list dict(path, identity, position, year, species).

    Path dibangun ulang persis seperti wildlife-datasets:
        Species/Turtle_ID/tahun/Photo_name
    Orientasi diambil dari nama berkas (`Nama_2019_L.webp`), bukan dari kolom
    terpisah — L/R dipetakan ke left/right.
    """
    keluar = []
    with open(os.path.join(data, "data.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tahun = datetime.strptime(r["Date"], "%m/%d/%Y").year
            batang = os.path.splitext(r["Photo_name"])[0]
            ori = {"L": "left", "R": "right"}.get(batang.split("_")[2])
            if ori is None:
                continue
            keluar.append({
                "path": os.path.join(data, r["Species"], r["Turtle_ID"],
                                     str(tahun), r["Photo_name"]),
                # Turtle_ID unik lintas spesies, tapi digabung supaya aman
                "identity": f"{r['Species']}/{r['Turtle_ID']}",
                "position": ori,
                "year": tahun,
                # Tanggal LENGKAP, bukan cuma tahun. Dipakai `periksa_curiga.py`
                # untuk mendeteksi foto dari sesi pemotretan yang sama, yang
                # nyaris identik dan membuat angka naik semu.
                "date": datetime.strptime(r["Date"], "%m/%d/%Y").date().isoformat(),
                "species": r["Species"],
            })
    return keluar


def _katalog_seaturtleheads(data):
    with open(os.path.join(data, "annotations.json")) as f:
        anot = json.load(f)
    ann = {a["image_id"]: a for a in anot["annotations"]}
    keluar = []
    for im in anot["images"]:
        a = ann[im["id"]]
        try:
            tahun = int(im["date"][:4])
        except (KeyError, ValueError):
            continue
        keluar.append({
            "path": os.path.join(data, im["path"]),
            "identity": a["identity"],
            "position": a["position"],
            "year": tahun,
            # 'YYYY:MM:DD HH:MM:SS' -> 'YYYY-MM-DD'
            "date": im["date"][:10].replace(":", "-"),
            "species": None,          # tidak tersedia di dataset ini
        })
    return keluar


def _katalog_zakynthos(data):
    """ZakynthosTurtles — loggerhead (Caretta caretta), Yunani.

    Kenapa dataset ini penting: ia satu-satunya dari tiga yang punya
    **tanggal DAN orientasi DAN bounding box kepala** sekaligus.

      - tanggal  -> split berbasis tahun (protokol §3) benar-benar bisa dibuat
      - orientasi-> kunci sisi bisa diterapkan
      - bbox     -> hipotesis "crop kepala menolong" bisa diuji dengan kotak
                    SEBENARNYA, tanpa perlu melatih YOLO lebih dulu

    Yang terakhir itu penting secara metodologis: kalau crop kepala dengan
    kotak ground-truth pun tidak menolong, melatih detektor tidak akan
    menolong juga. Menguji hipotesisnya lebih dulu jauh lebih murah daripada
    membangun detektornya.

    Spesies ketiga: ReunionTurtles hijau + sisik, ini penyu tempayan. Jadi
    hasil di sini sekaligus uji generalisasi lintas spesies.
    """
    keluar = []
    with open(os.path.join(data, "annotations.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["orientation"] not in SISI:
                continue
            keluar.append({
                "path": os.path.join(data, "images", r["path"]),
                "identity": r["identity"],
                "position": r["orientation"],
                "year": int(r["date"].split("_")[-1]),   # DD_MM_YYYY
                "date": "-".join(reversed(r["date"].split("_"))),
                "species": "Loggerhead",
            })
    return keluar


def _katalog_amvrakikos(data):
    """AmvrakikosTurtles — loggerhead, Teluk Amvrakikos, Yunani.

    200 foto / 50 individu, rentang 4,4 tahun. Struktur mirip Zakynthos:
    punya bbox anotasi manusia DAN orientasi, jadi plafon "crop sempurna"
    bisa dihitung tanpa melatih apa pun.

    SATU PERBEDAAN PENTING, dan ini bisa menggagalkan seluruh run:
    tanggal TIDAK ada di annotations.csv. wildlife-datasets membacanya dari
    **EXIF tiap berkas gambar**. Kalau foto pernah lewat alat yang membuang
    EXIF (banyak yang begitu), tanggalnya hilang dan split berbasis tahun
    protokol §3 tidak bisa dibangun.

    Karena itu fungsi ini MELEMPAR ERROR kalau tanggal tidak terbaca, bukan
    diam-diam memakai tahun 0 atau membuang barisnya. Split yang salah jauh
    lebih berbahaya daripada run yang gagal — run gagal langsung kelihatan.

    Orientasi diambil dari potongan ketiga nama berkas, sama seperti
    wildlife-datasets. Nilai 'top' dibuang karena bukan profil samping.
    """
    p_csv = os.path.join(data, "annotations.csv")
    if not os.path.exists(p_csv):
        raise FileNotFoundError(
            f"{p_csv} tidak ada. Unduh dulu:\n"
            f"  kaggle datasets download -d wildlifedatasets/amvrakikosturtles")

    keluar, tanpa_tanggal, orientasi_lain = [], [], set()
    with open(p_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nama = r["image_name"]
            bagian = os.path.splitext(nama)[0].split("_")
            if len(bagian) < 3:
                raise ValueError(
                    f"nama berkas '{nama}' tidak sesuai pola "
                    f"<identitas>_<?>_<orientasi>... — katalog tidak bisa "
                    f"dibangun tanpa menebak, dan menebak di sini berarti "
                    f"identitas atau sisi bisa salah")
            ori = bagian[2]
            if ori not in SISI:
                orientasi_lain.add(ori)
                continue
            path = os.path.join(data, "images", nama)
            tgl = _tanggal_exif(path)
            if tgl is None:
                tanpa_tanggal.append(nama)
                continue
            keluar.append({
                "path": path,
                "identity": bagian[0],
                "position": ori,
                "year": int(tgl[:4]),
                "date": tgl,
                "species": "Loggerhead",
            })

    if tanpa_tanggal:
        raise ValueError(
            f"{len(tanpa_tanggal)} dari {len(tanpa_tanggal) + len(keluar)} foto "
            f"tidak punya tanggal EXIF (contoh: {tanpa_tanggal[:3]}).\n"
            f"AmvrakikosTurtles menyimpan tanggal HANYA di EXIF, bukan di CSV. "
            f"Tanpa tanggal, split gallery/query berbasis tahun (protokol §3) "
            f"tidak bisa dibangun dan angkanya tidak akan berarti apa-apa.\n"
            f"Kemungkinan sebabnya: berkas diunduh lewat alat yang membuang "
            f"metadata, atau diekstrak ulang. Unduh ulang dari Kaggle.")
    if orientasi_lain:
        print(f"  amvrakikos: {len(orientasi_lain)} orientasi non-samping "
              f"dibuang: {sorted(orientasi_lain)}")
    return keluar


def _tanggal_exif(path):
    """'YYYY-MM-DD' dari EXIF DateTimeOriginal, atau None.

    Dipakai hanya oleh AmvrakikosTurtles. Sengaja memakai Pillow langsung
    dan bukan wildlife-datasets, supaya eksperimen ini tidak bergantung pada
    pustaka itu saat runtime.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "AmvrakikosTurtles butuh Pillow untuk membaca tanggal EXIF: "
            "../.venv/bin/pip install pillow")
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            # 36867 DateTimeOriginal, 306 DateTime
            for tag in (36867, 306):
                v = exif.get(tag)
                if not v:
                    # DateTimeOriginal sering ada di IFD Exif, bukan IFD0
                    try:
                        v = exif.get_ifd(0x8769).get(tag)
                    except Exception:
                        v = None
                if v:
                    return str(v)[:10].replace(":", "-")
    except Exception:
        return None
    return None


def kotak_kepala_gt(path):
    """Bounding box kepala dari anotasi manusia, bukan dari detektor.

    Ada untuk Zakynthos (`bbox.csv`, kolom `label_name` difilter ke 'head')
    dan AmvrakikosTurtles (`annotations.csv`, kolom bbox langsung).

    PERINGATAN untuk Amvrakikos: berkasnya tidak punya kolom `label_name`,
    jadi tidak ada cara program memastikan kotak itu kepala dan bukan
    seluruh badan penyu. Periksa mata dengan `--pratinjau-bbox` sekali
    sebelum mempercayai angka apa pun yang memakainya sebagai plafon.

    Mengembalikan (x, y, w, h) atau None.
    """
    global _BBOX_GT
    if _BBOX_GT is None:
        _BBOX_GT = {}
        p_zak = os.path.join(DATA, "bbox.csv")
        p_amv = os.path.join(DATA, "annotations.csv")
        if os.path.exists(p_zak):
            with open(p_zak, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r["label_name"] != "head":
                        continue
                    _BBOX_GT[r["image_name"]] = (
                        int(r["bbox_x"]), int(r["bbox_y"]),
                        int(r["bbox_width"]), int(r["bbox_height"]))
        elif DATASET == "amvrakikos" and os.path.exists(p_amv):
            with open(p_amv, encoding="utf-8-sig") as f:
                baca = csv.DictReader(f)
                perlu = {"bbox_x", "bbox_y", "bbox_width", "bbox_height"}
                if not perlu <= set(baca.fieldnames or []):
                    raise ValueError(
                        f"{p_amv} tidak punya kolom {sorted(perlu)}. "
                        f"Yang ada: {baca.fieldnames}")
                for r in baca:
                    try:
                        _BBOX_GT[r["image_name"]] = (
                            int(float(r["bbox_x"])), int(float(r["bbox_y"])),
                            int(float(r["bbox_width"])),
                            int(float(r["bbox_height"])))
                    except (TypeError, ValueError):
                        continue        # baris tanpa kotak — sah, jadi None
    return _BBOX_GT.get(os.path.basename(path))


_BBOX_GT = None


def baca_katalog(data=DATA):
    kat = {"reunion": _katalog_reunion,
           "seaturtleheads": _katalog_seaturtleheads,
           "zakynthos": _katalog_zakynthos,
           "amvrakikos": _katalog_amvrakikos}[DATASET](data)
    hilang = [r["path"] for r in kat if not os.path.exists(r["path"])]
    if hilang:
        raise FileNotFoundError(
            f"{len(hilang)} berkas di katalog tidak ada di disk, "
            f"contoh: {hilang[0]}")
    return kat


def bangun_split(katalog):
    """Split deterministik per SISI. Tidak ada acak, tidak ada seed.

    Untuk tiap sisi s dan tiap individu:
        tahun yang ada di sisi s -> y1 = paling awal, y2 = paling akhir
        y1 != y2  -> foto y1 masuk gallery, foto y2 masuk query
        y1 == y2  -> foto itu tetap masuk gallery sebagai DISTRAKTOR
                     (individu itu tidak pernah jadi jawaban benar, tapi ikut
                      menyulitkan pencarian — ini sengaja, biar jujur)
    """
    gallery, query = [], []
    for s in SISI:
        per_ind = defaultdict(lambda: defaultdict(list))
        for r in katalog:
            if r["position"] == s:
                per_ind[r["identity"]][r["year"]].append(r)
        for ident in sorted(per_ind):
            tahun = sorted(per_ind[ident])
            y1, y2 = tahun[0], tahun[-1]
            gallery += [dict(r, side=s, role="gallery") for r in per_ind[ident][y1]]
            if y2 != y1:
                query += [dict(r, side=s, role="query") for r in per_ind[ident][y2]]
    gallery.sort(key=lambda r: r["path"])
    query.sort(key=lambda r: r["path"])
    return gallery, query


def periksa_split(gallery, query):
    """Sanity check §8 — dikembalikan sebagai dict, bukan dicetak diam-diam."""
    g_key = {(r["identity"], r["side"]) for r in gallery}
    yatim = [r for r in query if (r["identity"], r["side"]) not in g_key]
    tahun_salah = [r for r in query
                   if any(g["identity"] == r["identity"] and g["side"] == r["side"]
                          and g["year"] >= r["year"] for g in gallery)]
    return {
        "n_gallery": len(gallery),
        "n_query": len(query),
        "n_identitas_gallery": len({r["identity"] for r in gallery}),
        "n_identitas_query": len({r["identity"] for r in query}),
        "query_tanpa_pasangan_sisi_sama": len(yatim),
        "query_yang_tahunnya_tidak_lebih_baru": len(tahun_salah),
        "gallery_per_sisi": {s: sum(r["side"] == s for r in gallery) for s in SISI},
        "query_per_sisi": {s: sum(r["side"] == s for r in query) for s in SISI},
        "query_per_spesies": {
            sp: sum(r.get("species") == sp for r in query)
            for sp in sorted({r.get("species") for r in query} - {None})},
    }


def hash_dataset(katalog):
    """Hash daftar berkas + ukuran. Cukup untuk mendeteksi dataset berubah."""
    h = hashlib.sha256()
    for r in sorted(katalog, key=lambda x: x["path"]):
        h.update(os.path.relpath(r["path"], DATA).encode())
        h.update(str(os.path.getsize(r["path"])).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------- preprocessing (§6)
def _white_balance(rgb):
    """Gray-world. Membuang cast biru-hijau air dengan menyamakan rata-rata
    tiap kanal ke rata-rata global."""
    f = rgb.astype(np.float32)
    rata = f.reshape(-1, 3).mean(0)
    f *= rata.mean() / np.maximum(rata, 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def _clahe(rgb):
    """CLAHE pada kanal L saja — kalau dikerjakan di RGB, warnanya rusak."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0,
                                   tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _grayscale(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _resize(rgb, n):
    """Samakan semua foto ke n x n. Lihat _resize368 untuk penjelasan lengkap."""
    return cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA)


def _resize368(rgb):
    """Samakan SEMUA foto ke 368x368 sebelum masuk pipeline.

    Foto ReunionTurtles ukurannya beragam, dari 253x227 sampai 800x600.
    Kondisi ini menyeragamkannya lebih dulu, jadi tiap gambar punya resolusi
    efektif yang sama saat masuk model dan saat masuk matcher stage-2.

    CATATAN: ini BUKAN mengubah ukuran input model. MegaDescriptor-L-384
    adalah Swin dengan `fixed_input_size=True` dan menolak apa pun selain
    384x384 (`AssertionError: Input height (368) doesn't match model (384)`).
    Jadi urutannya: asli -> 368x368 -> transform kanonik -> 384x384 -> model.
    Artinya ada langkah naik 368->384; kalau kondisi ini menang, kemenangannya
    datang dari penyeragaman resolusi, bukan dari angka 368 itu sendiri.
    """
    return cv2.resize(rgb, (368, 368), interpolation=cv2.INTER_AREA)


def _crop_tengah(rgb, frac):
    h, w = rgb.shape[:2]
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y, x = (h - ch) // 2, (w - cw) // 2
    return rgb[y:y + ch, x:x + cw]


# Kondisi §6. Nilai = fungsi rgb(uint8) -> rgb(uint8), dijalankan pada resolusi
# asli SEBELUM resize kanonik model.
KONDISI = {
    "raw":       lambda im: im,
    "crop":      lambda im: _crop_tengah(im, 0.70),
    "wb":        _white_balance,
    "clahe":     _clahe,
    "gray":      _grayscale,
    "crop_wb":   lambda im: _white_balance(_crop_tengah(im, 0.70)),
    "resize368": _resize368,
    # Sapu ukuran: 368 dipilih karena kebetulan disebut, bukan dioptimasi.
    # Kalau yang menolong memang penyeragaman skala (bukan angka 368),
    # ukuran lain harus memberi efek serupa. Diuji, bukan diasumsikan.
    "resize256": lambda im: _resize(im, 256),
    "resize320": lambda im: _resize(im, 320),
    "resize448": lambda im: _resize(im, 448),
    "resize512": lambda im: _resize(im, 512),
    # Lanjutan sapu: sampai 512 efeknya masih monoton naik, jadi titik
    # baliknya belum ketemu. Kalau 640 dan 768 tetap naik, yang menolong
    # adalah RESOLUSI, bukan penyeragaman skala — dua penjelasan yang
    # berbeda dan hanya bisa dipisahkan dengan mengukur lebih jauh.
    "resize640": lambda im: _resize(im, 640),
    "resize768": lambda im: _resize(im, 768),
}


# Kondisi yang bekerja pada BERKAS, bukan pada array — potongannya sudah
# dihitung sebelumnya oleh detektor dan disimpan ke disk. Dipisah dari
# KONDISI karena fungsinya butuh tahu jalur asal gambar, bukan cuma pikselnya.
def kepala_gt(path, ukuran=512, margin=0.18):
    """Crop kepala dari kotak ANOTASI MANUSIA, lalu seragamkan ke 512x512.

    Hanya bisa dipakai di Zakynthos. Nilainya besar secara metodologis:
    hipotesis "membuang latar menolong" bisa diuji dengan kotak yang
    sempurna, TANPA melatih detektor lebih dulu. Kalau dengan kotak
    ground-truth pun tidak menolong, melatih YOLO tidak akan menolong —
    dan itu menghemat berhari-hari kerja.

    Margin 18% sengaja ditambahkan: kepala yang terpotong lebih merugikan
    daripada sedikit latar yang ikut terbawa.
    """
    kotak = kotak_kepala_gt(path)
    if kotak is None:
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    x, y, bw, bh = kotak
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    potong = cv2.resize(bgr[y0:y1, x0:x1], (ukuran, ukuran),
                        interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(potong, cv2.COLOR_BGR2RGB)


def berkas_kepala(path):
    """Potongan kepala hasil YOLO untuk sebuah foto, atau None.

    Mengembalikan None kalau detektornya gagal pada foto itu. Pemanggil WAJIB
    memutuskan secara sadar apa yang dilakukan terhadap kasus gagal —
    membiarkannya diam-diam jatuh kembali ke gambar penuh akan mencampur dua
    kondisi berbeda di dalam satu angka.
    """
    global _PETA_KEPALA
    if _PETA_KEPALA is None:
        d = os.path.join(os.path.dirname(BASE), "dataset_penyu",
                         f"{DATASET}_kepala")
        p = os.path.join(d, "peta.json")
        if not os.path.exists(p):
            raise SystemExit(
                f"potongan kepala belum ada di {d}\n"
                "Jalankan di Mac:  python3 yolo_kepala.py --potong\n"
                "(bobot YOLO diunduh dari GitHub, yang diblokir di sandbox)")
        _PETA_KEPALA = {k: os.path.join(d, v["berkas"])
                        for k, v in json.load(open(p)).items()}
    q = _PETA_KEPALA.get(path)
    if q is None:
        return None
    bgr = cv2.imread(q)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def berkas_kepala_lintas(path):
    """Potongan dari detektor lintas-domain (mixed_best / kepala_lintas.pt).

    Beda dari kepala_gt: ini kotak DETEKTOR, bukan anotasi manusia. Jadi
    inilah yang benar-benar didapat produksi — kepala_gt cuma plafonnya.
    """
    global _PETA_KEPALA_LINTAS
    if _PETA_KEPALA_LINTAS is None:
        directory = os.path.join(
            os.path.dirname(BASE), "dataset_penyu", f"{DATASET}_kepala_lintas"
        )
        mapping_path = os.path.join(directory, "peta.json")
        if not os.path.exists(mapping_path):
            raise SystemExit(
                f"potongan detektor lintas-domain belum ada di {directory}\n"
                "Jalankan latih_detektor_lintas_domain.py --crop lebih dulu"
            )
        # Kunci peta.json adalah path ABSOLUT saat potongan dibuat — dan
        # potongan itu dibuat dari worktree, jadi prefiksnya
        # .worktrees/turtle-reid-prototype/dataset_penyu/... sedangkan katalog
        # repo utama menunjuk dataset_penyu/... langsung. Berkasnya sama, cuma
        # jalannya beda. Dicocokkan lewat nama berkas, yang unik dalam satu
        # dataset (kotak_kepala_gt juga sudah memakai basename).
        _PETA_KEPALA_LINTAS = {
            os.path.basename(key): os.path.join(directory, value["berkas"])
            for key, value in json.load(open(mapping_path)).items()
        }
    crop_path = _PETA_KEPALA_LINTAS.get(os.path.basename(path))
    if crop_path is None:
        return None
    bgr = cv2.imread(crop_path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_PETA_KEPALA = None
_PETA_KEPALA_LINTAS = None

# Kontrak: path -> array RGB, atau None kalau tidak tersedia untuk foto itu.
def _pembaca_potongan(tag):
    """Pabrik pembaca potongan dari dataset_penyu/{DATASET}_{tag}/peta.json.

    Dipakai kondisi kepala_cNN — potongan detektor pada ambang keyakinan NN%,
    dibuat oleh potong_kepala.py dengan geometri yang dikunci ke kepala_gt().
    Kuncinya nama berkas, bukan path absolut.
    """
    simpan = {}

    def baca(path):
        if tag not in simpan:
            d = os.path.join(os.path.dirname(BASE), "dataset_penyu",
                             f"{DATASET}_{tag}")
            p = os.path.join(d, "peta.json")
            if not os.path.exists(p):
                raise SystemExit(
                    f"potongan '{tag}' belum ada di {d}\n"
                    f"Jalankan dulu:  DATASET={DATASET} python3 potong_kepala.py "
                    f"--konf 0.{tag[-2:]}")
            simpan[tag] = {k: os.path.join(d, v["berkas"])
                           for k, v in json.load(open(p)).items()}
        f = simpan[tag].get(os.path.basename(path))
        if f is None:
            return None
        bgr = cv2.imread(f)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return baca


KONDISI_BERKAS = {"kepala": berkas_kepala,
                  "kepala_lintas": berkas_kepala_lintas,
                  "kepala_gt": kepala_gt,
                  "kepala_c25": _pembaca_potongan("kepala_c25"),
                  "kepala_c05": _pembaca_potongan("kepala_c05")}

LABEL = {
    "raw": "Raw (baseline)",
    "crop": "Crop kepala (center 70%)",
    "wb": "White balance (gray-world)",
    "clahe": "CLAHE (L, clip 2.0)",
    "gray": "Grayscale",
    "crop_wb": "Crop + white balance",
    "resize368": "Resize seragam 368x368",
    "resize256": "Resize seragam 256x256",
    "resize320": "Resize seragam 320x320",
    "resize448": "Resize seragam 448x448",
    "resize512": "Resize seragam 512x512",
    "resize640": "Resize seragam 640x640",
    "resize768": "Resize seragam 768x768",
    "kepala": "Crop kepala YOLO 512x512",
    "kepala_lintas": "Crop kepala YOLO lintas-domain 512x512",
    "kepala_c25": "Crop YOLO ambang 0,25",
    "kepala_c05": "Crop YOLO ambang 0,05",
    "kepala_gt": "Crop kepala anotasi manusia 512x512",
}

# Label pendek untuk UI berkolom sempit. Tanpa ini, kelima kondisi resize
# terpotong menjadi "Resize seragam ." yang persis sama satu sama lain —
# tombolnya tetap berfungsi, tapi tidak ada cara membedakan mana yang mana.
LABEL_PENDEK = {
    "raw": "Raw",
    "crop": "Crop kepala",
    "wb": "White balance",
    "clahe": "CLAHE",
    "gray": "Grayscale",
    "crop_wb": "Crop + WB",
    "resize256": "256 px",
    "resize320": "320 px",
    "resize368": "368 px",
    "resize448": "448 px",
    "resize512": "512 px",
    "resize640": "640 px",
    "resize768": "768 px",
    "kepala": "Kepala YOLO",
    "kepala_lintas": "Kepala YOLO lintas",
    "kepala_c25": "YOLO conf .25",
    "kepala_c05": "YOLO conf .05",
    "kepala_gt": "Kepala anotasi",
}


# ----------------------------------------------------------- embedding
# Transform input yang dipakai SEMUA kondisi. Dikunci di satu tempat.
#
# "squash" menang telak atas "cfg" pada dataset ini: Rank-1 55.78 vs 47.59,
# selisih +8.19 [+5.94, +10.43], McNemar p=1.9e-12 (lihat transform_ab.py).
# Alasannya masuk akal: crop_pct 0.9 dirancang untuk foto pemandangan penuh,
# sedangkan SeaTurtleIDHeads SUDAH berupa crop kepala yang ketat — center crop
# di atas crop ketat berarti membuang sisik di tepi, persis informasi yang
# dipakai untuk identifikasi.
#
# Jadi "sesuai config model" ternyata BUKAN "benar untuk data ini". Ini diuji,
# bukan diasumsikan — dan inilah kenapa baseline sempat di bawah 50%.
TRANSFORM = os.environ.get("TRANSFORM", "squash")   # "squash" | "cfg"


def dir_hasil(dataset=None, model=None, transform=None):
    """Folder hasil untuk satu kombinasi (dataset, model, transform).

    Tanpa argumen, mengembalikan folder run yang sedang aktif. Argumennya
    dipakai kalau perlu menunjuk run lain — mis. membandingkan MODEL T vs L
    atau membaca hasil dataset pembanding.
    """
    return os.path.join(AKAR_HASIL, "_".join((dataset or DATASET,
                                              model or MODEL,
                                              transform or TRANSFORM)))


def metadata_run(katalog=None):
    """Provenance minimum yang wajib ada di header tiap run.

    Semua nilainya DITURUNKAN dari variabel run, tidak ada yang diketik
    tangan. Versi lama menulis nama dataset sebagai literal di jalankan.py,
    dan akibatnya setiap header.json — reunion, zakynthos, amvrakikos —
    berlabel "SeaTurtleIDHeads". Hash-nya tetap benar, jadi angka
    eksperimennya tidak terpengaruh, tapi labelnya menyesatkan.
    """
    kat = baca_katalog() if katalog is None else katalog
    return {"dataset": DATASET,
            "dataset_dir": os.path.basename(DATA),
            "dataset_hash": hash_dataset(kat),
            "model": NAMA_HF,
            "transform": TRANSFORM,
            "input_size": UKURAN,
            "crop_pct": CROP_PCT,
            "numpy": np.__version__}


def transform_squash(rgb):
    """INTER_AREA langsung ke 224x224. Rasio aspek rusak, tapi tidak ada
    piksel yang dibuang."""
    rgb = cv2.resize(rgb, (UKURAN, UKURAN), interpolation=cv2.INTER_AREA)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def transform_cfg(rgb):
    """Resize bicubic ke 224/0.9 lalu center-crop 224, normalisasi ImageNet.
    Ini yang tertulis di config.json MegaDescriptor."""
    sisi_resize = int(round(UKURAN / CROP_PCT))
    h, w = rgb.shape[:2]
    skala = sisi_resize / min(h, w)
    rgb = cv2.resize(rgb, (max(1, int(round(w * skala))),
                           max(1, int(round(h * skala)))),
                     interpolation=cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    y, x = (h - UKURAN) // 2, (w - UKURAN) // 2
    rgb = rgb[y:y + UKURAN, x:x + UKURAN]
    f = rgb.astype(np.float32) / 255.0
    return ((f - MEAN) / STD).transpose(2, 0, 1)


def transform_kanonik(rgb):
    return transform_squash(rgb) if TRANSFORM == "squash" else transform_cfg(rgb)


def _muat_miewid(snap):
    """MiewID-msv3 dari cache HF lokal, tanpa AutoModel.from_pretrained.

    `AutoModel.from_pretrained(..., trust_remote_code=True)` — jalur yang
    dipakai turtle-identification-be — GAGAL di transformers 5.x:

        AttributeError: 'MiewIdNet' object has no attribute
        'all_tied_weights_keys'

    Remote code MiewID ditulis untuk transformers 4.45. Daripada menurunkan
    versi transformers seluruh venv, kelas aslinya diimpor langsung dari
    snapshot dan bobotnya dimuat manual — pola yang sama dengan cabang
    MegaDescriptor di bawah. Kodenya tetap kode hulu, bukan tulisan ulang.

    `strict=True` yang menjaga kebenarannya: kalau satu bobot saja tidak
    cocok, ini melempar, bukan diam-diam memakai bobot acak.
    """
    import importlib.util
    import sys
    import types

    import timm
    import torch
    from safetensors.torch import load_file

    paket = "_miewid_snap"
    if paket not in sys.modules:
        pkg = types.ModuleType(paket)
        pkg.__path__ = [snap]
        sys.modules[paket] = pkg

    def modul(nama):
        penuh = f"{paket}.{nama}"
        if penuh in sys.modules:
            return sys.modules[penuh]
        spec = importlib.util.spec_from_file_location(
            penuh, os.path.join(snap, f"{nama}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[penuh] = m
        spec.loader.exec_module(m)
        return m

    modul("heads")
    konfig_mod = modul("configuration_miewid")

    # __init__ hulu memaksa pretrained=True, yang menarik bobot ImageNet
    # efficientnetv2_rw_m dari jaringan lalu langsung ditimpa safetensors.
    # Unduhan itu sia-sia, dan protokol ini sengaja berjalan tanpa jaringan.
    asli = timm.create_model
    timm.create_model = lambda *a, **k: asli(*a, **{**k, "pretrained": False})
    try:
        model_mod = modul("modeling_miewid")
    finally:
        timm.create_model = asli

    cfg = json.load(open(os.path.join(snap, "config.json")))
    obj = konfig_mod.MiewIdNetConfig(
        **{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map")})
    model = model_mod.MiewIdNet(obj)
    model.load_state_dict(load_file(os.path.join(snap, "model.safetensors")),
                          strict=True)
    model.eval()

    # DIM tidak bisa dibaca dari config.json, jadi dibuktikan ke model asli.
    with torch.no_grad():
        lebar = model(torch.zeros(2, 3, UKURAN, UKURAN)).shape[1]
    if lebar != DIM:
        raise RuntimeError(f"DIM MiewID berubah di hulu: {lebar} != {DIM}")

    cfg["architecture"] = cfg["architectures"][0]
    return model, cfg


def muat_model(snap=None):
    """MegaDescriptor-T-224 dari cache HF lokal, tanpa jaringan.

    checkpoint_filter_fn WAJIB: checkpoint BVRA memakai tata letak Swin gaya
    lama (downsample di akhir stage). timm >=1.0 menaruh downsample di awal
    stage berikutnya. Tanpa filter itu, load_state_dict(strict=False) akan
    'berhasil' dengan bobot downsample acak — tidak ada error, hanya angka
    yang pelan-pelan salah.
    """
    import timm
    import torch
    from timm.models.swin_transformer import checkpoint_filter_fn

    global SNAP_T
    snap = snap or cari_snapshot()
    SNAP_T = snap
    if MODEL == "MIEWID":
        return _muat_miewid(snap)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(snap, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd = sd["model"] if "model" in sd else sd
    conv = {k: v for k, v in checkpoint_filter_fn(sd, model).items()
            if not k.startswith("head.fc")}
    hasil = model.load_state_dict(conv, strict=False)
    if hasil.missing_keys or hasil.unexpected_keys:
        raise RuntimeError(f"bobot tidak cocok: missing={hasil.missing_keys[:3]} "
                           f"unexpected={hasil.unexpected_keys[:3]}")
    model.eval()
    return model, cfg


def embed(paths, kondisi, model, batch=16, threads=4):
    """List path -> matriks embedding L2-normalized (n, DIM)."""
    import torch
    torch.set_num_threads(threads)
    # Dua jenis kondisi: berbasis array (resize, CLAHE, ...) dan berbasis
    # BERKAS (crop kepala, yang butuh tahu jalur asal untuk mencari kotaknya).
    # Kalau kondisi berkas tidak punya entri untuk sebuah foto, di sini
    # sengaja MELEMPAR error, bukan diam-diam memakai gambar penuh — kalau
    # jatuh diam-diam, satu angka akan mencampur dua kondisi berbeda.
    # Kondisi KOMPOSIT "berkas+array", mis. "kepala_gt+clahe": potong kepala
    # dulu, baru terapkan transform array di atas potongannya. Ada supaya
    # pertanyaan "apakah preprocessing menolong SETELAH latar dibuang" bisa
    # diukur, bukan ditebak. Urutannya sengaja crop-dulu: preprocessing yang
    # dihitung dari seluruh frame (white balance gray-world, CLAHE) memberi
    # hasil berbeda kalau latar masih ikut, dan latar itu justru yang mau
    # dibuang.
    tambahan = None
    if "+" in kondisi:
        dasar, sisa = kondisi.split("+", 1)
        if dasar not in KONDISI_BERKAS or sisa not in KONDISI:
            raise KeyError(f"kondisi komposit '{kondisi}' tidak sah: "
                           f"'{dasar}' harus kondisi berkas dan "
                           f"'{sisa}' harus kondisi array")
        berkas, tambahan = KONDISI_BERKAS[dasar], KONDISI[sisa]
        fn = None
    else:
        berkas = KONDISI_BERKAS.get(kondisi)
        fn = None if berkas else KONDISI[kondisi]
    keluar = np.zeros((len(paths), DIM), np.float32)
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            if berkas:
                rgb = berkas(p)
                if rgb is None:
                    raise RuntimeError(
                        f"kondisi '{kondisi}' tidak punya potongan untuk {p}. "
                        "Query yang gagal dideteksi tidak boleh diam-diam "
                        "diganti gambar penuh.")
                if tambahan is not None:
                    rgb = tambahan(rgb)
                xs.append(transform_kanonik(rgb))
                continue
            bgr = cv2.imread(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xs.append(transform_kanonik(fn(rgb)))
        x = torch.from_numpy(np.stack(xs))
        with torch.no_grad():
            v = model(x).float().numpy()
        keluar[i:i + len(xs)] = v
    n = np.linalg.norm(keluar, axis=1, keepdims=True)
    return keluar / np.maximum(n, 1e-9)


# -------------------------------------------------------------- metrik
def metrik_dari_matriks(S, id_q, id_g):
    """Matriks similarity -> rank1 / rank5 / AP. SATU-SATUNYA tempat metrik
    dihitung di repo ini.

    Pasangan yang tidak sah ditandai -inf oleh pemanggil. Mask `sah` di bawah
    WAJIB: tiap individu punya foto galeri di KEDUA sisi, jadi tanpa mask itu
    foto sisi seberang ikut terhitung sebagai jawaban benar kedua di peringkat
    jauh, dan mAP anjlok tanpa ada error apa pun. Justru karena jebakan ini
    halus, fungsinya tidak boleh ditulis ulang di tempat lain — stage-2
    re-ranking pun memanggil fungsi yang sama.
    """
    urut = np.argsort(-S, axis=1)
    benar = id_g[urut] == id_q[:, None]
    benar &= np.isfinite(np.take_along_axis(S, urut, 1))

    ap = np.zeros(len(id_q))
    for i in range(len(id_q)):
        hit = np.flatnonzero(benar[i])
        if len(hit):
            ap[i] = np.mean((np.arange(len(hit)) + 1) / (hit + 1))
    return {"rank1": benar[:, 0], "rank5": benar[:, :5].any(1), "ap": ap}


def evaluasi_manual(Eq, Eg, id_q, id_g, sisi_q, sisi_g):
    """Implementasi manual §4. Sengaja pendek supaya bisa dibaca sekali lihat.

    Kebocoran sisi ditutup dengan menyetel similarity pasangan beda-sisi ke
    -inf, bukan dengan memfilter setelah ranking — kalau difilter belakangan,
    peringkatnya sudah tercemar.
    """
    S = Eq @ Eg.T                                   # cosine, sudah L2-normalized
    S[sisi_q[:, None] != sisi_g[None, :]] = -np.inf  # kunci sisi
    return metrik_dari_matriks(S, id_q, id_g)


def ringkas(hasil):
    return {"rank1": float(hasil["rank1"].mean() * 100),
            "rank5": float(hasil["rank5"].mean() * 100),
            "mAP": float(hasil["ap"].mean() * 100),
            "n": int(len(hasil["ap"]))}


# Sapu MARGIN — mengukur dosis-respons, bukan sekadar satu titik.
#
# Margin besar = kotak diperlebar = kepala mengisi porsi lebih kecil dari
# potongan. Ini meniru "kepala kecil di dalam frame" secara terkendali, tanpa
# mengganti dataset. Kalau akurasinya turun mulus seiring margin membesar,
# yang menentukan memang PORSI KEPALA DI FRAME - bukan sesuatu yang khas
# Zakynthos. Dan itu memberi aturan yang bisa dipakai untuk memutuskan kapan
# YOLO diperlukan di dataset mana pun.
for _m in (0.5, 1.0, 2.0, 4.0):
    KONDISI_BERKAS[f"kepala_m{int(_m*100)}"] = (
        lambda path, _mm=_m: kepala_gt(path, margin=_mm))
    LABEL[f"kepala_m{int(_m*100)}"] = f"Kepala + margin {_m:g}x"
    LABEL_PENDEK[f"kepala_m{int(_m*100)}"] = f"kepala m{_m:g}"
