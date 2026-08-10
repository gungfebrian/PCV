"""
Re-ID penyu realtime dari kamera. HANYA penyu — mode kartu dan wajah dibuang.

    ../.venv/bin/python penyu_live.py                    # webcam 0
    ../.venv/bin/python penyu_live.py --kamera 1
    ../.venv/bin/python penyu_live.py --foto uji.jpg     # tanpa kamera

    # kamera IP (DroidCam / IP Webcam). URL salah ketik dibetulkan otomatis:
    ../.venv/bin/python penyu_live.py --url 10.64.53.103:4747
    MODEL=T ../.venv/bin/python penyu_live.py --url 10.64.53.103:4747

Untuk kamera IP pakai `MODEL=T`. L-384 butuh ~600 ms per frame di CPU, jadi
stream-nya akan tertinggal jauh; T-224 sekitar 15x lebih cepat. Akurasinya
turun (25.00% -> 18.45% di ReunionTurtles), dan itu trade-off yang disengaja
untuk penggunaan live.

Kenapa dipisah dari `eksperimen/`
---------------------------------
Spesifikasi §1 menyebut realtime kamera sebagai hal yang TIDAK dikerjakan.
Aplikasi ini ada di luar folder eksperimen supaya protokol yang dikunci tidak
tercemar. Tapi ia **mengimpor** `eksperimen/protokol.py` untuk transform,
embedding, dan preprocessing, serta `rerank.py` untuk stage-2 — jadi apa yang
tampil di layar memakai jalur kode yang sama persis dengan angka di laporan.
Kalau dibiarkan terpisah, demo dan eksperimen akan pelan-pelan berbeda tanpa
ada yang sadar.

Pipeline yang ditampilkan penuh, bukan hanya hasil akhirnya:

    DETECT    kontur terbesar -> bounding box   (heuristik, sering meleset)
    ALIGN     resize ke ukuran input model
    DESCRIBE  MegaDescriptor frozen -> embedding L2-normalized
    MATCH     cosine ke galeri, dikunci per sisi
    RE-RANK   local feature (SIFT / XFeat) -> inlier RANSAC -> urutkan top-k

Yang SENGAJA tidak ditampilkan: persentase akurasi yang diketik di tombol.
Angka seperti "(26.5%)" di visualizer lama adalah string hardcoded yang sudah
lama tidak cocok dengan hasil sebenarnya. Akurasi dibaca dari
`eksperimen/hasil/*/statistik.json`, atau tidak ditampilkan sama sekali.
"""

import argparse
import json
import os
import re
import socket
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(REPO, "eksperimen"))

import protokol as P            # noqa: E402
import rerank as R              # noqa: E402
import tampilan as T            # noqa: E402

HASIL = os.path.join(REPO, "eksperimen", "hasil",
                     f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")


# ------------------------------------------------------------- DETECT
def deteksi_bbox(bgr):
    """Kontur terbesar. HEURISTIK, dan diketahui sering meleset.

    Dokumentasi pipeline lama mencatat crop dari kotak ini menurunkan Top-1
    dari 26.5% ke 17.0%, karena kontur terbesar sering menemukan riak pasir
    alih-alih penyunya. Kotaknya tetap dihitung dan digambar supaya
    kesalahannya terlihat — bukan supaya dipakai. Identifikasi tetap memakai
    frame penuh.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    tepi = cv2.Canny(g, 50, 150)
    tepi = cv2.dilate(tepi, np.ones((5, 5), np.uint8), iterations=2)
    kont, _ = cv2.findContours(tepi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kont:
        return None, tepi
    c = max(kont, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * bgr.shape[0] * bgr.shape[1]:
        return None, tepi
    return cv2.boundingRect(c), tepi


# ------------------------------------------------------------- mesin
class Mesin:
    def __init__(self, pakai_rerank=True):
        self.model, self.cfg = P.muat_model()
        kat = P.baca_katalog()
        self.gal, _ = P.bangun_split(kat)
        self.matcher = {}
        self.tersedia = []
        for nama in ("sift", "xfeat", "aliked", "roma"):
            try:
                if nama in R.MATCHER_TERBLOKIR and not R.bobot_ada(nama):
                    continue
                self.matcher[nama] = R.buat_matcher(nama)
                self.tersedia.append(nama)
            except Exception as e:
                print(f"  matcher {nama} dilewati: {str(e).splitlines()[0][:70]}")
        self.sift = self.matcher.get("sift")
        self.cache_emb = {}
        self.cache_thumb = {}
        self.cache_fitur = {}

    # ---- galeri
    def emb_galeri(self, kondisi):
        if kondisi not in self.cache_emb:
            p = os.path.join(HASIL, f"emb_{kondisi}.npy")
            self.cache_emb[kondisi] = (np.load(p)[:len(self.gal)]
                                       if os.path.exists(p) else None)
        return self.cache_emb[kondisi]

    def indeks_sisi(self, sisi):
        return [i for i, r in enumerate(self.gal) if r["side"] == sisi]

    def thumb(self, i):
        if i not in self.cache_thumb:
            im = cv2.imread(self.gal[i]["path"])
            self.cache_thumb[i] = im
        return self.cache_thumb[i]

    def fitur(self, i, nama):
        """Cache per matcher — deskriptor SIFT dan XFeat tidak bisa ditukar."""
        kunci = (nama, i)
        if kunci not in self.cache_fitur:
            self.cache_fitur[kunci] = self.matcher[nama].ekstrak(
                self.gal[i]["path"])
        return self.cache_fitur[kunci]

    # ---- inferensi
    def kenali(self, bgr, kondisi, sisi, mode_rerank="off", k=20,
               nama_matcher="xfeat"):
        import torch
        t = {}
        t0 = time.time()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_pra = P.KONDISI[kondisi](rgb)
        x = P.transform_kanonik(rgb_pra)
        t["ms_pra"] = (time.time() - t0) * 1000

        t1 = time.time()
        with torch.no_grad():
            v = self.model(torch.from_numpy(x[None])).float().numpy()[0]
        t["ms_infer"] = (time.time() - t1) * 1000
        v /= max(np.linalg.norm(v), 1e-9)

        E = self.emb_galeri(kondisi)
        idx = self.indeks_sisi(sisi)
        if E is None or not idx:
            return None, [], t, rgb_pra, None, None

        s = E[idx] @ v
        urut = np.argsort(-s)[:max(k, 5)]

        inlier = None
        pasangan = None
        t["ms_match"] = 0.0
        mm = self.matcher.get(nama_matcher)
        if mode_rerank != "off" and mm is not None:
            t2 = time.time()
            fq = self._fitur_frame(rgb_pra, nama_matcher)
            sk = np.array([mm.skor(fq, self.fitur(idx[j], nama_matcher))
                           for j in urut])
            t["ms_match"] = (time.time() - t2) * 1000
            if mode_rerank == "murni":
                o = np.lexsort((np.arange(len(urut)), -sk))
            else:
                r1 = 1.0 / (60 + np.arange(len(urut)) + 1)
                r2 = 1.0 / (60 + np.argsort(np.argsort(-sk)) + 1)
                o = np.lexsort((np.arange(len(urut)), -(r1 + r2)))
            urut, sk = urut[o], sk[o]
            inlier = sk
            pasangan = self._pasangan(fq, self.fitur(idx[int(urut[0])],
                                                      nama_matcher), nama_matcher)

        top5 = [{"nama": self.gal[idx[j]]["identity"].split("/")[-1],
                 "skor": float(s[j]),
                 "inlier": (float(inlier[n]) if inlier is not None else None),
                 "img": self.thumb(idx[j])}
                for n, j in enumerate(urut[:5])]
        margin = float(s[urut[0]] - s[urut[1]]) if len(urut) > 1 else 0.0
        hasil = {"nama": top5[0]["nama"], "skor": top5[0]["skor"],
                 "margin": margin}
        t["inlier"] = int(inlier[0]) if inlier is not None else "-"
        return hasil, top5, t, rgb_pra, pasangan, self.thumb(idx[int(urut[0])])

    def _fitur_frame(self, rgb, nama="xfeat"):
        """Ekstrak dari frame kamera (bukan dari path) untuk matcher terpilih."""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mm = self.matcher[nama]
        if hasattr(mm, "X"):                       # XFeat
            return mm.X.ekstrak(mm.model, g, sisi=R.SISI_PROSES)
        sc = R.SISI_PROSES / max(g.shape)
        if sc < 1:
            g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        kp, des = mm.det.detectAndCompute(g, None)
        if des is None or len(kp) < 8:
            return None
        return np.float32([k.pt for k in kp]), des

    def _pasangan(self, a, b, nama="xfeat", maks=60):
        """Pasangan inlier untuk digambar. Sengaja mengembalikan koordinat,
        bukan cuma jumlah — supaya kelihatan DI MANA korespondensinya mendarat.
        Kalau garisnya di karang dan bukan di sisik, itu penjelasan langsung."""
        if a is None or b is None:
            return []
        mm = self.matcher[nama]
        if hasattr(mm, "X"):                       # XFeat: mutual NN, bukan ratio
            s_, d_ = mm.X.cocokkan(a, b)
            if s_ is None or len(s_) < 4:
                return []
            src, dst = s_.reshape(-1, 1, 2), d_.reshape(-1, 1, 2)
        else:
            m = cv2.BFMatcher(mm.norm).knnMatch(a[1], b[1], k=2)
            baik = [x for x, y in m if x.distance < R.RATIO_LOWE * y.distance]
            if len(baik) < 4:
                return []
            src = a[0][[x.queryIdx for x in baik]].reshape(-1, 1, 2)
            dst = b[0][[x.trainIdx for x in baik]].reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, R.AMBANG_RANSAC)
        if mask is None:
            return []
        return [(tuple(src[i][0]), tuple(dst[i][0]))
                for i in np.flatnonzero(mask.ravel())[:maks]]


# ------------------------------------------------------------------ state
def state_awal():
    return {
        "kondisi": "raw",
        "kondisi_pilihan": [(k, P.LABEL[k]) for k in P.KONDISI],
        "matcher": "xfeat",
        "matcher_pilihan": [],          # diisi saat start dari bobot yang ada
        "rerank": "off",
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


def akurasi_terukur():
    p = os.path.join(HASIL, "statistik.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p)).get("raw", {})
    out = [("Rank-1", f"{s['rank1']:.2f}%"), ("Rank-5", f"{s['rank5']:.2f}%"),
           ("mAP", f"{s['mAP']:.2f}%")]
    for sp, v in sorted(s.get("per_spesies", {}).items()):
        out.append((f"R-1 {sp.lower()}", f"{v['rank1']:.2f}%"))
    return out


def normalisasi_url(u):
    """Betulkan kesalahan URL kamera IP yang paling sering terjadi.

    Tiga jebakan, semuanya gagal tanpa pesan yang berguna dari OpenCV:
      - `https:` -> DroidCam dan IP Webcam melayani HTTP polos, bukan HTTPS
      - `http:10.0.0.5` -> kurang `//`, urllib/OpenCV membacanya sebagai path
      - tanpa path -> port 4747 (DroidCam) butuh `/video`,
        port 8080 (IP Webcam Android) butuh `/video` juga

    Dikembalikan sebagai fungsi murni supaya bisa diuji tanpa kamera.
    """
    if not u:
        return u
    u = u.strip()
    u = re.sub(r"^https(?=:)", "http", u)          # https -> http
    u = re.sub(r"^(https?:)(?!//)", r"\1//", u)    # http:1.2.3.4 -> http://1.2.3.4
    if not re.match(r"^\w+://", u):
        u = "http://" + u
    sisa = u.split("://", 1)[1]
    if "/" not in sisa:                            # belum ada path
        port = sisa.rsplit(":", 1)[-1] if ":" in sisa else ""
        if port in ("4747", "8080"):
            u = u.rstrip("/") + "/video"
    return u


def cek_hidup(u, timeout=3.0):
    """Ping soket sebelum menyerahkannya ke OpenCV.

    `cv2.VideoCapture` pada IP yang mati menggantung sekitar 30 detik tanpa
    memberi tahu apa pun. Catatan pipeline lama sudah menyebut masalah ini;
    pengecekan singkat di sini membuat kegagalan langsung terbaca.
    """
    m = re.match(r"^\w+://([^/:]+)(?::(\d+))?", u or "")
    if not m:
        return True, ""
    host, port = m.group(1), int(m.group(2) or 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as e:
        return False, f"{host}:{port} tidak menjawab dalam {timeout:.0f} detik ({e})"


def buka_sumber(a):
    if a.foto:
        im = cv2.imread(a.foto)
        if im is None:
            raise SystemExit(f"tidak bisa membaca {a.foto}")
        return None, im

    if a.url:
        url = normalisasi_url(a.url)
        if url != a.url:
            print(f"URL dibetulkan: {a.url}  ->  {url}")
        hidup, pesan = cek_hidup(url)
        if not hidup:
            raise SystemExit(
                f"{pesan}\n"
                "Periksa: (1) HP dan Mac di Wi-Fi yang sama, (2) aplikasi "
                "kamera di HP sedang berjalan dan menampilkan IP itu, "
                "(3) coba buka URL-nya di browser dulu.")
        cap = cv2.VideoCapture(url)
    else:
        cap = cv2.VideoCapture(a.kamera)

    if not cap.isOpened():
        raise SystemExit(
            "sumber tidak terbuka. Untuk webcam: System Settings -> Privacy & "
            "Security -> Camera. Untuk kamera IP: buka URL-nya di browser "
            "dulu untuk memastikan formatnya benar. Atau pakai --foto untuk "
            "uji tanpa kamera.")
    return cap, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--url", default=None)
    ap.add_argument("--foto", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lebar", type=int, default=1500)
    ap.add_argument("--tinggi", type=int, default=880)
    a = ap.parse_args()

    print(f"model  : {P.MODEL} ({P.UKURAN}x{P.UKURAN}, dim {P.DIM})")
    acc = akurasi_terukur()
    print("akurasi terukur:", acc if acc else
          "belum ada statistik.json — tidak ditampilkan daripada usang")

    mesin = Mesin()
    st = state_awal()
    st["matcher_pilihan"] = [(m, m.upper()) for m in mesin.tersedia]
    if st["matcher"] not in mesin.tersedia and mesin.tersedia:
        st["matcher"] = mesin.tersedia[0]
    print("matcher tersedia:", ", ".join(mesin.tersedia) or "(tidak ada)")
    cap, diam = buka_sumber(a)
    ringkas = akurasi_terukur()

    JUDUL = "Re-ID Penyu — engineer tool"
    cv2.namedWindow(JUDUL, cv2.WINDOW_AUTOSIZE)
    klik = {"aksi": None}
    tahap_kini = []

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        klik["aksi"] = (T.klik_sidebar(st, x, y)
                        or T.klik_tahap(tahap_kini, x, y, T.LEBAR_SIDEBAR, 0,
                                        a.lebar - T.LEBAR_SIDEBAR - T.LEBAR_PANEL))

    cv2.setMouseCallback(JUDUL, on_mouse)

    frame = diam
    hasil, top5, pasangan, kand = None, [], None, None
    tel = {"ms_infer": 0, "ms_pra": 0, "ms_match": 0, "fps": 0,
           "input_w": 0, "input_h": 0, "ukuran": f"{P.UKURAN}x{P.UKURAN}",
           "n_gallery": 0, "n_kp": "-", "inlier": "-"}
    t_akhir, fps, perlu = time.time(), 0.0, True

    while True:
        if cap is not None and not st["jeda"]:
            ok, f = cap.read()
            if not ok:
                break
            frame, perlu = f, True

        if frame is not None and perlu:
            hasil, top5, t, rgb_pra, pasangan, kand = mesin.kenali(
                frame, st["kondisi"], st["sisi"], st["rerank"], a.k,
                st["matcher"])
            tel.update(t)
            bbox, tepi = deteksi_bbox(frame)
            pra_bgr = cv2.cvtColor(rgb_pra, cv2.COLOR_RGB2BGR)
            align = cv2.cvtColor(
                (np.clip(P.transform_kanonik(rgb_pra).transpose(1, 2, 0)
                         * P.STD + P.MEAN, 0, 1) * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR)
            fq = mesin._fitur_frame(rgb_pra, st["matcher"])
            kp = fq[0] if fq else None
            tel["n_kp"] = len(kp) if kp is not None else "-"
            kp_vis = pra_bgr.copy()
            if kp is not None:
                sc = R.SISI_PROSES / max(pra_bgr.shape[:2])
                T.gambar_keypoint(kp_vis, kp, 1 / max(sc, 1e-9), 0, 0)
            tahap_kini = [("Asli", frame), ("Praproses", pra_bgr),
                          ("Tepi", tepi), ("Align", align),
                          ("Keypoint", kp_vis)]
            tel["input_w"], tel["input_h"] = frame.shape[1], frame.shape[0]
            tel["n_gallery"] = len(mesin.indeks_sisi(st["sisi"]))
            perlu = False

        sekarang = time.time()
        fps = 0.9 * fps + 0.1 / max(sekarang - t_akhir, 1e-6)
        t_akhir = sekarang
        tel["fps"] = fps

        peta = dict(tahap_kini)
        utama = peta.get(st["tahap"], frame)
        kanvas = T.susun(utama, st, tel, hasil, top5, a.lebar, a.tinggi,
                         tahap=tahap_kini, bbox=bbox if frame is not None else None,
                         keypoint=kp if st["tahap"] == "Asli" else None,
                         pasangan=pasangan, gambar_kandidat=kand,
                         ringkas=ringkas)
        cv2.imshow(JUDUL, kanvas)

        if klik["aksi"]:
            jenis, nilai = klik["aksi"]
            klik["aksi"] = None
            if jenis in ("kondisi", "sisi", "rerank", "tahap", "matcher"):
                st[jenis] = nilai
                perlu = True
            elif jenis == "toggle":
                st[nilai] = not st[nilai]
            elif jenis == "ambang":
                st["ambang"] = round((st["ambang"] + 0.05) % 1.0, 2)
            elif jenis == "aksi":
                if nilai == "jeda":
                    st["jeda"] = not st["jeda"]
                elif nilai == "keluar":
                    break
                elif nilai == "simpan":
                    n = f"tangkapan_{int(time.time())}.png"
                    cv2.imwrite(n, kanvas)
                    print("disimpan:", n)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            st["jeda"] = not st["jeda"]
        if k == ord("s"):
            n = f"tangkapan_{int(time.time())}.png"
            cv2.imwrite(n, kanvas)
            print("disimpan:", n)
        if k in (ord("["), ord("]")):
            u = list(P.KONDISI)
            st["kondisi"] = u[(u.index(st["kondisi"])
                               + (1 if k == ord("]") else -1)) % len(u)]
            perlu = True
        if k == ord("f"):
            st["sisi"] = "right" if st["sisi"] == "left" else "left"
            perlu = True
        if k == ord("r"):
            u = ["off", "murni", "rrf"]
            st["rerank"] = u[(u.index(st["rerank"]) + 1) % 3]
            perlu = True
        if k == ord("b"):
            st["bbox"] = not st["bbox"]
        if k == ord("k"):
            st["keypoint"] = not st["keypoint"]
        if k == ord("m"):
            st["match"] = not st["match"]
        if k == ord("-"):
            st["ambang"] = max(0.0, round(st["ambang"] - 0.05, 2))
        if k in (ord("+"), ord("=")):
            st["ambang"] = min(1.0, round(st["ambang"] + 0.05, 2))

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
