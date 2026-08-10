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
    "aliked": "bobot ALIKED dari github release / HF — keduanya 403 di sini",
    "xfeat": ("bobot XFeat dari github.com/verlab/accelerated_features — 403. "
              "CATATAN: paket PyPI bernama `xfeat` BUKAN XFeat CVPR 2024, "
              "melainkan pustaka feature engineering tabular dari 2020. "
              "Jangan sampai tertukar."),
    "roma": ("RoMa butuh backbone DINOv2; dl.fbaipublicfiles.com dan HF "
             "keduanya 403 di sini"),
}


# ------------------------------------------------------------- matcher
class Klasik:
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
            bgr = cv2.imread(path)
            if bgr is None:
                return None
            rgb = P.KONDISI[self.kondisi](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if im is None:
            return None
        s = SISI_PROSES / max(im.shape)
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.det.detectAndCompute(im, None)
        if des is None or len(kp) < 8:
            return None
        return np.float32([k.pt for k in kp]), des

    def skor(self, a, b):
        if a is None or b is None:
            return 0.0
        pa, da, pb, db = a[0], a[1], b[0], b[1]
        m = cv2.BFMatcher(self.norm).knnMatch(da, db, k=2)
        baik = [x for x, y in m if len([x, y]) == 2 and x.distance < RATIO_LOWE * y.distance]
        if len(baik) < 4:
            return float(len(baik))
        src = pa[[x.queryIdx for x in baik]].reshape(-1, 1, 2)
        dst = pb[[x.trainIdx for x in baik]].reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, AMBANG_RANSAC)
        return float(mask.sum()) if mask is not None else 0.0


class XFeat:
    """XFeat (CVPR 2024). Arsitektur direkonstruksi dari state_dict —
    lihat `xfeat_lokal.py` untuk dua pemeriksaan yang membuktikannya benar.

    Memakai matcher bawaannya (mutual NN + ambang cosine), bukan ratio test
    Lowe seperti SIFT. Yang disamakan antar metode adalah SKOR akhirnya:
    jumlah inlier setelah RANSAC MAGSAC, dengan ambang yang sama persis.
    """

    def __init__(self, kondisi="raw"):
        import xfeat_lokal as X
        self.X = X
        self.model = X.muat()
        self.kondisi = kondisi
        self.nama = "xfeat" if kondisi == "raw" else f"xfeat-{kondisi}"

    def ekstrak(self, path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if self.kondisi == "raw" else None
        if self.kondisi != "raw":
            bgr = cv2.imread(path)
            if bgr is None:
                return None
            rgb = P.KONDISI[self.kondisi](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if im is None:
            return None
        return self.X.ekstrak(self.model, im, sisi=SISI_PROSES)

    def skor(self, a, b):
        src, dst = self.X.cocokkan(a, b)
        if src is None or len(src) < 4:
            return 0.0 if src is None else float(len(src))
        _, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                     cv2.USAC_MAGSAC, AMBANG_RANSAC)
        return float(mask.sum()) if mask is not None else 0.0


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
        try:
            import kornia.feature as KF
            KF.ALIKED()
            return True
        except Exception:
            return False
    if nama == "roma":
        try:
            import romatch  # noqa: F401
            return True
        except Exception:
            return False
    return True


def buat_matcher(nama, kondisi="raw"):
    if nama == "xfeat" and bobot_ada("xfeat"):
        return XFeat(kondisi)
    if nama in MATCHER_TERBLOKIR and not bobot_ada(nama):
        raise SystemExit(
            f"matcher '{nama}' tidak bisa dijalankan di lingkungan ini:\n"
            f"  {MATCHER_TERBLOKIR[nama]}\n"
            f"Antarmukanya sudah siap — sediakan bobotnya lalu tambahkan kelas "
            f"dengan metode .ekstrak(path) dan .skor(a, b).")
    return Klasik(nama, kondisi=kondisi)


# ------------------------------------------------------- inti re-rank
def kandidat_stage1(gal, qry, k):
    """Top-k kandidat per query dari stage 1. Sisi sudah dikunci di sini."""
    Eg, Eq = muat("raw", gal, qry)
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

    out = os.path.join(HASIL, f"rerank_{matcher.nama}_k{k}.npy")
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
    M = np.load(os.path.join(HASIL, f"rerank_{nama_matcher}_k{k}.npy"))

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

    with open(os.path.join(HASIL, f"rerank_{nama_matcher}_k{k}.json"), "w") as f:
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
    a = ap.parse_args()

    if a.lapor:
        lapor(a.matcher, a.k)
    else:
        m = buat_matcher(a.matcher, a.kondisi)
        selesai, _, k = jalankan(m, a.k, budget=a.budget)
        if selesai:
            lapor(a.matcher, k)
