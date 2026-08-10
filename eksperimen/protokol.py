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
DATASET = os.environ.get("DATASET", "reunion")
DATA = os.path.join(REPO, "dataset_penyu",
                    {"reunion": "ReunionTurtles",
                     "seaturtleheads": "SeaTurtleIDHeads"}[DATASET])

SISI = ("left", "right")          # ketat: topleft/topright/top/front/below dibuang

# Bobot dibaca dari cache HF lokal, tanpa jaringan. HF_HOME dipakai kalau ada.
_KANDIDAT_HF = [
    os.environ.get("HF_HOME"),
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    "/sessions/optimistic-vibrant-cerf/mnt/huggingface",
]


# Varian MegaDescriptor. T-224 cepat; L-384 jauh lebih akurat tapi ~20x lebih
# berat di CPU. Keduanya sudah ada di cache HF lokal.
MODEL = os.environ.get("MODEL", "T")
REPO_HF = {"T": "models--BVRA--MegaDescriptor-T-224",
           "L": "models--BVRA--MegaDescriptor-L-384"}[MODEL]


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
            "species": None,          # tidak tersedia di dataset ini
        })
    return keluar


def baca_katalog(data=DATA):
    kat = (_katalog_reunion if DATASET == "reunion"
           else _katalog_seaturtleheads)(data)
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
}

LABEL = {
    "raw": "Raw (baseline)",
    "crop": "Crop kepala (center 70%)",
    "wb": "White balance (gray-world)",
    "clahe": "CLAHE (L, clip 2.0)",
    "gray": "Grayscale",
    "crop_wb": "Crop + white balance",
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
    fn = KONDISI[kondisi]
    keluar = np.zeros((len(paths), DIM), np.float32)
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
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
