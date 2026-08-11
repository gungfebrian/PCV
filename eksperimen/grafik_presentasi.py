"""
Flow chart + seluruh chart untuk presentasi. Keluaran SVG.

    MODEL=L python3 grafik_presentasi.py

SVG dipilih, bukan PNG: tajam di proyektor berapa pun resolusinya, ukurannya
kecil (beberapa KB) sehingga bisa ditempel langsung ke Notion, dan teksnya
tetap bisa dicari. Digambar tangan tanpa matplotlib supaya tidak ada
ketergantungan tambahan dan tiap koordinat bisa diatur persis.

ATURAN: setiap angka di berkas ini dibaca dari berkas hasil, tidak diketik
manual. Kalau sebuah run belum ada, chart-nya melewatkan titik itu daripada
menampilkan angka karangan.
"""

import json
import os

import protokol as P

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")
KELUAR = os.path.join(P.BASE, "grafik")

# palet — kontras tinggi, aman untuk proyektor dan untuk buta warna merah-hijau
BG = "#ffffff"
TINTA = "#1b1b1b"
REDUP = "#6b6b6b"
GARIS = "#d4d4d4"
BIRU = "#2563eb"
HIJAU = "#15803d"
JINGGA = "#ea580c"
MERAH = "#b91c1c"
UNGU = "#7c3aed"
ABU = "#9ca3af"

FONT = ("font-family='Inter, -apple-system, Segoe UI, Helvetica, Arial, "
        "sans-serif'")


MONO = "font-family='SF Mono, Menlo, Consolas, monospace'"


def _t(x, y, s, uk=13, w=TINTA, anchor="start", tebal=400, mono=False):
    """xml:space='preserve' WAJIB: SVG memampatkan spasi berurutan menjadi
    satu, jadi kolom yang diratakan dengan spasi akan berantakan tanpa itu."""
    f = MONO if mono else FONT
    return (f"<text x='{x}' y='{y}' {f} font-size='{uk}' fill='{w}' "
            f"text-anchor='{anchor}' font-weight='{tebal}' "
            f"xml:space='preserve'>{_esc(s)}</text>")


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _kotak(x, y, w, h, isi, tepi=None, r=8, lebar_tepi=1.5):
    st = f" stroke='{tepi}' stroke-width='{lebar_tepi}'" if tepi else ""
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' "
            f"fill='{isi}'{st}/>")


def _svg(w, h, isi, judul):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
            f"width='{w}' height='{h}' role='img' aria-label='{_esc(judul)}'>"
            f"<rect width='{w}' height='{h}' fill='{BG}'/>"
            + "".join(isi) + "</svg>")


def _tulis(nama, isi):
    os.makedirs(KELUAR, exist_ok=True)
    p = os.path.join(KELUAR, nama)
    with open(p, "w") as f:
        f.write(isi)
    print(f"  {nama:34} {len(isi) / 1024:5.1f} KB")
    return p


# ------------------------------------------------------------ baca hasil
def baca(nama):
    p = os.path.join(HASIL, nama)
    return json.load(open(p)) if os.path.exists(p) else None


def rerank_r1(matcher, k, mode="murni"):
    """Rank-1 sebuah run, atau None kalau runnya belum ada.

    Mengembalikan None (bukan 0, bukan tebakan) supaya chart melewatkan
    titiknya alih-alih menggambar angka yang tidak pernah diukur.
    """
    d = baca(f"rerank_{matcher}_k{k}.json")
    if not d:
        return None
    return d.get("tabel", {}).get(mode, {}).get("rank1")


# ==================================================== 1. FLOW CHART PIPELINE
def flowchart_pipeline():
    W, H = 1180, 620
    o = [_t(24, 34, "Pipeline Re-ID Penyu — dua tahap", 21, TINTA, tebal=700),
         _t(24, 56, "MegaDescriptor menyaring, XFeat memutuskan. "
                    "Tidak ada bobot yang dilatih pada penyu.", 13, REDUP)]

    def tahap(x, y, w, h, no, judul, isi, warna, catatan=None):
        b = [_kotak(x, y, w, h, "#ffffff", warna, 10, 2),
             _kotak(x, y, 42, 26, warna, None, 6),
             _t(x + 21, y + 18, no, 13, "#ffffff", "middle", 700),
             _t(x + 52, y + 18, judul, 14, TINTA, tebal=700)]
        for i, s in enumerate(isi):
            b.append(_t(x + 14, y + 46 + i * 17, s, 11.5, REDUP))
        if catatan:
            b.append(_t(x + 14, y + h - 12, catatan, 11, warna, tebal=600))
        return b

    def panah(x1, y1, x2, y2, label=None, warna=REDUP):
        b = [f"<path d='M {x1} {y1} L {x2} {y2}' stroke='{warna}' "
             f"stroke-width='2' fill='none' marker-end='url(#p)'/>"]
        if label:
            b.append(_t((x1 + x2) / 2, y1 - 8, label, 10.5, warna,
                        "middle", 600))
        return b

    o.append("<defs><marker id='p' viewBox='0 0 10 10' refX='9' refY='5' "
             "markerWidth='6' markerHeight='6' orient='auto'>"
             f"<path d='M 0 0 L 10 5 L 0 10 z' fill='{REDUP}'/>"
             "</marker></defs>")

    y1 = 86
    o += tahap(24, y1, 200, 116, "1", "FOTO MASUK",
               ["kamera / berkas / dataset", "ukuran bebas, cahaya bebas"],
               ABU, "1 foto = 1 query")
    o += panah(228, y1 + 58, 268, y1 + 58)
    o += tahap(272, y1, 230, 116, "2", "SISI DITENTUKAN",
               ["kiri atau kanan kepala", "dikunci: kiri hanya dicari",
                "di galeri kiri"],
               MERAH, "protokol §3 — tidak boleh silang")
    o += panah(506, y1 + 58, 546, y1 + 58)
    o += tahap(550, y1, 262, 116, "3", "MEGADESCRIPTOR-L",
               ["Swin-L 384x384, BEKU", "embedding 1536-dim",
                "di-L2-normalize"],
               BIRU, "cosine ke seluruh galeri sisi itu")
    o += panah(816, y1 + 58, 856, y1 + 58)
    o += tahap(860, y1, 296, 116, "4", "TOP-k KANDIDAT",
               ["k = 84 berarti SEMUA galeri", "recall@84 = 100%"],
               BIRU, "Rank-1 di sini baru 25,00%")

    # turun ke stage 2
    o.append(f"<path d='M 1008 {y1 + 116} L 1008 258 L 150 258 L 150 288' "
             f"stroke='{REDUP}' stroke-width='2' fill='none' "
             f"marker-end='url(#p)'/>")
    o.append(_t(580, 250, "kandidat diteruskan ke tahap kedua", 11, REDUP,
                "middle", 600))

    y2 = 292
    o += tahap(24, y2, 252, 128, "5", "RESIZE 512x512",
               ["query DAN kandidat", "disamakan skalanya",
                "bukan demi resolusi"],
               JINGGA, "efeknya mendatar di 448-512")
    o += panah(280, y2 + 64, 320, y2 + 64)
    o += tahap(324, y2, 252, 128, "6", "XFEAT",
               ["keypoint + deskriptor", "BEKU, dilatih di MegaDepth",
                "belum pernah lihat penyu"],
               UNGU, "mutual NN + cosine 0,82")
    o += panah(580, y2 + 64, 620, y2 + 64)
    o += tahap(624, y2, 252, 128, "7", "RANSAC MAGSAC",
               ["homografi, ambang 4 px", "skor = JUMLAH inlier",
                "rasio & presisi dicoba, kalah"],
               UNGU, "geometri harus konsisten")
    o += panah(880, y2 + 64, 920, y2 + 64)
    o += tahap(924, y2, 232, 128, "8", "URUT ULANG",
               ["'murni' = skor inlier saja", "RRF jauh lebih buruk di sini"],
               HIJAU, "Rank-1 75,60%")

    # kaki: hasil
    d = baca("rerank_xfeat-resize512_k84.json")
    m = (d or {}).get("tabel", {}).get("murni", {})
    o += [_kotak(24, 452, 1132, 74, "#f0fdf4", HIJAU, 10, 2),
          _t(44, 480, "HASIL AKHIR", 12, HIJAU, tebal=700),
          _t(44, 506, f"Rank-1 {m.get('rank1', 0):.2f}%", 17, TINTA, tebal=700),
          _t(210, 506, f"Rank-5 {m.get('rank5', 0):.2f}%", 15, TINTA),
          _t(372, 506, f"mAP {m.get('mAP', 0):.2f}%", 15, TINTA),
          _t(510, 506, "n = 168, ReunionTurtles", 13, REDUP),
          _t(730, 506, "+50,60 poin di atas stage-1", 13, HIJAU, tebal=700),
          _t(960, 506, "p = 1,0e-22", 13, HIJAU, tebal=600)]

    o.append(_t(24, 556, "Kenapa dua tahap: MegaDescriptor cepat tapi kasar "
                         "(25%); XFeat teliti tapi lambat (15 ms/pasangan).",
               11.5, REDUP))
    o.append(_t(24, 574, "Tahap 1 memangkas 84 kandidat jadi daftar pendek, "
                         "tahap 2 memeriksa sisik satu per satu.", 11.5, REDUP))
    o.append(_t(24, 596, "Angka dari ReunionTurtles, 168 query. Belum "
                         "direplikasi di dataset kedua.", 11, MERAH,
               tebal=600))
    return _svg(W, H, o, "Flow chart pipeline Re-ID penyu")


# ============================================== 2. PERJALANAN AKURASI (bar)
def chart_perjalanan():
    langkah = [
        ("MegaDescriptor-L saja", rerank_r1("xfeat-resize512", 84, "stage1"),
         ABU, "titik awal"),
        ("+ XFeat (tanpa resize)", rerank_r1("xfeat", 20), BIRU, "k=20"),
        ("+ resize 512", rerank_r1("xfeat-resize512", 20), JINGGA, "k=20"),
        ("+ k=50", rerank_r1("xfeat-resize512", 50), UNGU, "k=50"),
        ("+ k=84 (semua galeri)", rerank_r1("xfeat-resize512", 84), HIJAU,
         "k=84"),
    ]
    langkah = [x for x in langkah if x[1] is not None]

    W, H = 900, 130 + len(langkah) * 62
    x0, lebar = 300, 480
    o = [_t(24, 36, "Dari 25% ke 75,6% — apa yang benar-benar menaikkannya",
            20, TINTA, tebal=700),
         _t(24, 58, "Rank-1, ReunionTurtles, n = 168. Tiap batang menumpuk "
                    "di atas yang sebelumnya.", 12.5, REDUP)]

    for i in (0, 25, 50, 75, 100):
        x = x0 + lebar * i / 100
        o.append(f"<line x1='{x}' y1='88' x2='{x}' y2='{H - 46}' "
                 f"stroke='{GARIS}' stroke-width='1'/>")
        o.append(_t(x, H - 28, f"{i}%", 11, REDUP, "middle"))

    sebelum = None
    for i, (nama, v, warna, tag) in enumerate(langkah):
        y = 104 + i * 62
        o.append(_t(x0 - 14, y + 22, nama, 13, TINTA, "end", 600))
        o.append(_kotak(x0, y, lebar * v / 100, 30, warna, None, 4))
        o.append(_t(x0 + lebar * v / 100 + 10, y + 21, f"{v:.2f}%", 14,
                    warna, tebal=700))
        o.append(_t(x0 + 8, y + 46, tag, 10.5, REDUP))
        if sebelum is not None:
            d = v - sebelum
            o.append(_t(x0 + lebar * v / 100 + 74, y + 21,
                        f"{d:+.2f}", 12, HIJAU if d > 0 else MERAH, tebal=600))
        sebelum = v

    o.append(_t(24, H - 10, "Lompatan terbesar bukan dari model baru, tapi "
                            "dari menyamakan skala gambar dan memperbanyak "
                            "kandidat yang diperiksa.", 11.5, REDUP))
    return _svg(W, H, o, "Perjalanan akurasi")


# ================================================ 3. SAPU UKURAN (garis)
def chart_ukuran():
    titik = []
    for n in (256, 320, 368, 448, 512, 640, 768):
        v = rerank_r1(f"xfeat-resize{n}", 20)
        if v is not None:
            titik.append((n, v))
    dasar = rerank_r1("xfeat", 20)

    W, H = 880, 470
    x0, y0, w, h = 90, 90, 700, 280
    lo, hi = 40, 70
    o = [_t(24, 36, "Sapu ukuran: naik lalu MENDATAR, bukan naik terus",
            20, TINTA, tebal=700),
         _t(24, 58, "XFeat murni, k = 20. Hipotesis 'makin besar makin baik' "
                    "ditolak oleh dua titik terakhir.", 12.5, REDUP)]

    def px(n):
        return x0 + w * (n - 220) / (800 - 220)

    def py(v):
        return y0 + h - h * (v - lo) / (hi - lo)

    for v in range(lo, hi + 1, 5):
        o.append(f"<line x1='{x0}' y1='{py(v)}' x2='{x0 + w}' y2='{py(v)}' "
                 f"stroke='{GARIS}' stroke-width='1'/>")
        o.append(_t(x0 - 12, py(v) + 4, f"{v}%", 11, REDUP, "end"))

    if dasar is not None:
        o.append(f"<line x1='{x0}' y1='{py(dasar)}' x2='{x0 + w}' "
                 f"y2='{py(dasar)}' stroke='{MERAH}' stroke-width='1.5' "
                 f"stroke-dasharray='6 4'/>")
        o.append(_t(x0 + w - 4, py(dasar) - 8,
                    f"tanpa resize {dasar:.2f}%", 11, MERAH, "end", 600))

    d = " ".join(f"{'M' if i == 0 else 'L'} {px(n)} {py(v)}"
                 for i, (n, v) in enumerate(titik))
    o.append(f"<path d='{d}' stroke='{JINGGA}' stroke-width='3' fill='none' "
             f"stroke-linejoin='round'/>")

    puncak = max(titik, key=lambda t: t[1])[1] if titik else 0
    for n, v in titik:
        atas = v >= puncak - 0.01
        o.append(f"<circle cx='{px(n)}' cy='{py(v)}' r='{7 if atas else 5}' "
                 f"fill='{HIJAU if atas else JINGGA}'/>")
        o.append(_t(px(n), py(v) - 15, f"{v:.1f}", 11.5,
                    HIJAU if atas else TINTA, "middle", 700))
        o.append(_t(px(n), y0 + h + 24, str(n), 11.5, REDUP, "middle"))

    o.append(_t(x0 + w / 2, y0 + h + 48, "ukuran resize (piksel)", 12,
                REDUP, "middle", 600))

    # daerah dataran
    o.append(_kotak(px(430), y0, px(790) - px(430), h,
                    "rgba(0,0,0,0)", HIJAU, 6, 1.5))
    o.append(_t(px(610), y0 + 18, "dataran — selisihnya di dalam noise",
                11.5, HIJAU, "middle", 600))

    o.append(_t(24, H - 34, "Penting: kondisi 'tanpa resize' TIDAK berarti "
                            "resolusi kecil - foto asli dipotong ke sisi 800 "
                            "px, lebih besar dari 512.", 11.5, REDUP))
    o.append(_t(24, H - 16, "Jadi yang menolong adalah SKALA YANG SERAGAM "
                            "antar foto, bukan jumlah piksel. Itu dua "
                            "penjelasan yang berbeda.", 11.5, TINTA,
               tebal=600))
    return _svg(W, H, o, "Sapu ukuran resize")


# ============================== 4. k vs LANGIT-LANGIT (recall stage-1)
def chart_k(recall):
    baris = []
    for k in (20, 30, 40, 50, 84):
        v = rerank_r1("xfeat-resize512", k)
        if v is not None:
            baris.append((k, v, recall.get(k)))

    W, H = 940, 190 + len(baris) * 62
    x0, lebar = 190, 600
    o = [_t(24, 36, "Sweet spot k: naik tajam sampai 40, lalu hampir datar",
            20, TINTA, tebal=700),
         _t(24, 58, "Batang terang = langit-langit (recall stage-1). "
                    "Batang gelap = hasil sebenarnya setelah XFeat.",
            12.5, REDUP)]

    for i in (0, 25, 50, 75, 100):
        x = x0 + lebar * i / 100
        o.append(f"<line x1='{x}' y1='86' x2='{x}' y2='{H - 92}' "
                 f"stroke='{GARIS}'/>")
        o.append(_t(x, H - 74, f"{i}%", 11, REDUP, "middle"))

    sblm = None
    for i, (k, v, rc) in enumerate(baris):
        y = 100 + i * 62
        o.append(_t(x0 - 14, y + 26, f"k = {k}", 14, TINTA, "end", 700))
        o.append(_t(x0 - 14, y + 42, f"{168 * k} pasangan", 10, REDUP, "end"))
        if rc:
            o.append(_kotak(x0, y, lebar * rc / 100, 40, "#dbeafe", None, 4))
            o.append(_t(x0 + lebar * rc / 100 + 8, y + 15,
                        f"langit-langit {rc:.1f}%", 10.5, BIRU, tebal=600))
        o.append(_kotak(x0, y + 8, lebar * v / 100, 24, HIJAU, None, 4))
        o.append(_t(x0 + lebar * v / 100 + 8, y + 34, f"{v:.2f}%", 13,
                    HIJAU, tebal=700))
        if sblm is not None:
            d = v - sblm
            o.append(_t(x0 + lebar * v / 100 + 74, y + 34, f"{d:+.2f}", 11,
                        HIJAU if d > 1 else ABU, tebal=600))
        sblm = v

    o.append(_t(24, H - 62, "n = 168 query (ReunionTurtles). Kolom pasangan = "
                            "berapa perbandingan gambar yang harus dihitung.",
               11, REDUP))
    o.append(_t(24, H - 42, "Sweet spot praktis: k = 40. Menaikkannya ke 84 "
                            "melipatgandakan ongkos untuk tambahan 1,79 poin.",
               12, TINTA, tebal=700))
    o.append(_t(24, H - 20, "Pada k = 84 langit-langitnya 100% tapi hasilnya "
                            "75,6% - penghambatnya XFeat, bukan pencarian "
                            "kandidat.", 11.5, REDUP))
    return _svg(W, H, o, "Pengaruh k")


# ================================= 5. METODE YANG DICOBA DAN GAGAL
def chart_gagal():
    b = baca("boost_raw.json")
    baris = []
    if b:
        dasar = b["hasil"]["polos"]["rank1"]
        for kunci, d in b["hasil"].items():
            if kunci == "polos":
                continue
            baris.append((d["label"], d["rank1"] - dasar,
                          d["mcnemar"]["p_value"]))
    baris += [("skor = inlier/keypoint (rasio)",
               (rerank_r1("xfeat-resize512-rasio", 20) or 0)
               - (rerank_r1("xfeat-resize512", 20) or 0), None),
              ("skor = inlier/pasangan (presisi)",
               (rerank_r1("xfeat-resize512-presisi", 20) or 0)
               - (rerank_r1("xfeat-resize512", 20) or 0), None)]

    W, H = 900, 150 + len(baris) * 44
    xm, skala = 470, 26          # px per poin
    o = [_t(24, 36, "Yang dicoba dan TIDAK berhasil", 20, TINTA, tebal=700),
         _t(24, 58, "Hasil negatif tetap hasil. Semua diuji berpasangan "
                    "(McNemar) pada query yang sama.", 12.5, REDUP)]

    o.append(f"<line x1='{xm}' y1='84' x2='{xm}' y2='{H - 56}' "
             f"stroke='{TINTA}' stroke-width='1.5'/>")
    o.append(_t(xm, 78, "tanpa perubahan", 10.5, TINTA, "middle", 600))

    for i, (nama, d, p) in enumerate(baris):
        y = 96 + i * 44
        o.append(_t(xm - 190, y + 18, nama[:44], 12, TINTA, "end"))
        w = abs(d) * skala
        warna = HIJAU if d > 0 else MERAH
        if p is not None and p < 0.05 and d < 0:
            warna = MERAH
        elif p is None or p >= 0.05:
            warna = ABU if abs(d) < 3 else warna
        o.append(_kotak(xm if d > 0 else xm - w, y + 2, max(w, 2), 22,
                        warna, None, 3))
        o.append(_t(xm + (w + 8 if d > 0 else -w - 8), y + 18,
                    f"{d:+.2f}", 12, warna, "start" if d > 0 else "end", 700))
        ket = ("tidak signifikan" if p is None or p >= 0.05
               else f"SIGNIFIKAN LEBIH BURUK  p={p:.3f}")
        o.append(_t(xm + 116 if d > 0 else xm + 116, y + 18, ket, 10.5,
                    MERAH if p is not None and p < 0.05 else REDUP,
                    tebal=600 if p is not None and p < 0.05 else 400))

    o.append(_t(24, H - 32, "k-reciprocal rusak karena tiap individu hanya "
                            "punya SATU foto galeri per sisi - tetangga "
                            "terdekat sebuah foto galeri", 11.5, REDUP))
    o.append(_t(24, H - 14, "selalu individu yang BERBEDA, jadi seluruh "
                            "gagasan 'himpunan tetangga bersama' tidak punya "
                            "bahan untuk bekerja.", 11.5, REDUP))
    return _svg(W, H, o, "Metode yang gagal")


# ===================================== 6. PER SPESIES DAN SISI
def chart_pecahan():
    d = baca("rerank_xfeat-resize512_k84.json")
    if not d:
        return None
    t = d["tabel"]
    W, H = 900, 420
    o = [_t(24, 36, "Pecahan per spesies dan sisi — konfigurasi terbaik",
            20, TINTA, tebal=700),
         _t(24, 58, "Selisih spesies +8,4 poin, tapi TIDAK signifikan "
                    "(Fisher exact p = 0,283 pada n yang lebih kecil).",
            12.5, REDUP)]

    kel = [("Green", t["murni"]["per_spesies"]["Green"], HIJAU),
           ("Hawksbill", t["murni"]["per_spesies"]["Hawksbill"], JINGGA),
           ("sisi kiri", t["murni"]["per_sisi"]["left"], BIRU),
           ("sisi kanan", t["murni"]["per_sisi"]["right"], UNGU)]
    s1 = [("Green", t["stage1"]["per_spesies"]["Green"]),
          ("Hawksbill", t["stage1"]["per_spesies"]["Hawksbill"]),
          ("sisi kiri", t["stage1"]["per_sisi"]["left"]),
          ("sisi kanan", t["stage1"]["per_sisi"]["right"])]

    x0, lebar = 190, 560
    for i in (0, 25, 50, 75, 100):
        x = x0 + lebar * i / 100
        o.append(f"<line x1='{x}' y1='84' x2='{x}' y2='{H - 78}' "
                 f"stroke='{GARIS}'/>")
        o.append(_t(x, H - 60, f"{i}%", 11, REDUP, "middle"))

    for i, ((nama, v, warna), (_, v1)) in enumerate(zip(kel, s1)):
        y = 100 + i * 68
        o.append(_t(x0 - 14, y + 26, f"{nama}  (n={v['n']})", 13, TINTA,
                    "end", 600))
        o.append(_kotak(x0, y, lebar * v1["rank1"] / 100, 16, ABU, None, 3))
        o.append(_t(x0 + lebar * v1["rank1"] / 100 + 8, y + 13,
                    f"{v1['rank1']:.1f} stage-1", 10.5, REDUP))
        o.append(_kotak(x0, y + 20, lebar * v["rank1"] / 100, 26, warna,
                        None, 3))
        o.append(_t(x0 + lebar * v["rank1"] / 100 + 8, y + 39,
                    f"{v['rank1']:.2f}%", 13, warna, tebal=700))

    o.append(_t(24, H - 30, "Perhatikan arahnya BERBALIK: di stage-1 "
                            "hawksbill lebih unggul, setelah XFeat green yang "
                            "unggul.", 11.5, REDUP))
    o.append(_t(24, H - 12, "Dengan n sekecil ini, itu alasan untuk tidak "
                            "menyimpulkan apa pun tentang bias spesies.",
               11.5, TINTA, tebal=600))
    return _svg(W, H, o, "Pecahan spesies dan sisi")


# ============================ 7. FLOW CHART KEPUTUSAN (apa berikutnya)
def flowchart_keputusan():
    W, H = 1100, 560
    o = [_t(24, 34, "Peta keputusan: penghambatnya sudah pindah", 21, TINTA,
            tebal=700),
         _t(24, 56, "Semua tuas di tahap 1 sudah habis dicoba. Yang tersisa "
                    "ada di kualitas gambar dan matcher.", 13, REDUP)]
    o.append("<defs><marker id='q' viewBox='0 0 10 10' refX='9' refY='5' "
             "markerWidth='6' markerHeight='6' orient='auto'>"
             f"<path d='M 0 0 L 10 5 L 0 10 z' fill='{REDUP}'/>"
             "</marker></defs>")

    def blok(x, y, w, h, judul, isi, warna, latar="#ffffff", mono=False):
        b = [_kotak(x, y, w, h, latar, warna, 10, 2),
             _t(x + 14, y + 24, judul, 13.5, warna, tebal=700)]
        for i, s in enumerate(isi):
            b.append(_t(x + 14, y + 46 + i * 16, s, 10.5 if mono else 11.5,
                        TINTA, mono=mono))
        return b

    o += blok(24, 84, 320, 150, "SUDAH DICOBA — TAHAP 1",
              ["alpha-QE            +0,60  tidak sig.",
               "database augment.   -0,60  tidak sig.",
               "CSLS hubness        +1,79  tidak sig.",
               "PCA-whitening       +0,60  tidak sig.",
               "k-reciprocal        -7,14  LEBIH BURUK"],
              MERAH, "#fef2f2", mono=True)
    o += blok(24, 250, 320, 128, "SUDAH DICOBA — SKOR",
              ["inlier mentah       62,50  terbaik",
               "inlier / keypoint   61,90  seri",
               "inlier / pasangan   55,95  lebih buruk"],
              MERAH, "#fef2f2", mono=True)
    o += blok(24, 394, 320, 118, "SUDAH DICOBA — UKURAN & k",
              ["ukuran mendatar di 448-512",
               "k = 84 hanya +0,60 dari k = 50",
               "langit-langit sudah 100%"],
              MERAH, "#fef2f2")

    o.append(f"<path d='M 352 300 L 412 300' stroke='{REDUP}' "
             f"stroke-width='2' marker-end='url(#q)'/>")

    o += blok(420, 210, 300, 180, "PENGHAMBAT SEKARANG",
              ["Pada k = 84 langit-langitnya 100%,",
               "tapi hasilnya 75,60%.",
               "",
               "Jadi 24,4% query gagal karena",
               "XFEAT menaruh foto yang SALAH",
               "di atas yang benar - bukan karena",
               "kandidatnya tidak ikut terambil."],
              JINGGA, "#fff7ed")

    o.append(f"<path d='M 728 300 L 788 300' stroke='{REDUP}' "
             f"stroke-width='2' marker-end='url(#q)'/>")

    o += blok(796, 96, 280, 128, "TUAS 1 — GAMBAR",
              ["YOLO deteksi kepala -> crop",
               "menghapus pasir, tangan, karang",
               "yang ikut menghasilkan inlier palsu",
               "",
               "berlaku untuk foto lapangan"],
              HIJAU, "#f0fdf4")
    o += blok(796, 240, 280, 118, "TUAS 2 — MATCHER LAIN",
              ["ALIKED, LoMa lewat vismatch",
               "RoMa dicoret: terlalu berat",
               "(1-3 detik per pasangan)"],
              HIJAU, "#f0fdf4")
    o += blok(796, 374, 280, 138, "TUAS 3 — LEBIH BANYAK DATA",
              ["Zindi Turtle Recall:",
               "2.145 foto, 100 individu",
               "+ 10.658 foto, 2.231 individu",
               "",
               "TANPA tanggal - protokol split",
               "berbasis tahun tidak bisa dipakai"],
              BIRU, "#eff6ff")

    o.append(_t(24, H - 16, "Kotak merah = jalan buntu yang sudah terukur. "
                            "Menjalankannya lagi tidak akan mengubah apa pun.",
               11.5, REDUP))
    return _svg(W, H, o, "Peta keputusan")


# ================== 8. DOSIS-RESPONS: porsi kepala vs akurasi
def chart_dosis_kepala():
    """Menjawab 'apakah YOLO membantu semua dataset atau cuma satu'.

    Margin crop diperlebar bertahap pada dataset yang SAMA, sehingga kepala
    mengisi porsi yang makin kecil. Kalau akurasinya turun mulus, yang
    menentukan memang PORSI KEPALA — bukan sesuatu yang khas satu dataset.
    """
    H_ = os.path.join(P.BASE, "hasil", "zakynthos_L_squash")

    def r1(nm):
        p = os.path.join(H_, f"rerank_{nm}_k40.json")
        if not os.path.exists(p):
            return None
        return json.load(open(p))["tabel"]["murni"]["rank1"]

    titik = [(54.1, r1("xfeat-kepala_gt"), "crop ketat\n(margin 0,18x)"),
             (25.0, r1("xfeat-kepala_m50"), "margin 0,5x"),
             (11.1, r1("xfeat-kepala_m100"), "margin 1x"),
             (4.0, r1("xfeat-kepala_m200"), "margin 2x"),
             (2.05, r1("xfeat-resize512"), "TANPA crop\nframe penuh")]
    titik = [t for t in titik if t[1] is not None]

    W, H = 940, 620
    x0, y0, w, h = 96, 100, 660, 300
    o = [_t(24, 36, "Jawabannya bukan dataset - tapi seberapa besar kepala "
                    "di frame", 20, TINTA, tebal=700),
         _t(24, 58, "Dataset yang SAMA (Zakynthos, n=80, k=40). Yang diubah "
                    "hanya lebar kotak crop.", 12.5, REDUP)]

    import math

    def px(f):
        return x0 + w * (math.log10(f) - math.log10(1.6)) / \
            (math.log10(70) - math.log10(1.6))

    def py(v):
        return y0 + h - h * v / 70

    for v in range(0, 71, 10):
        o.append(f"<line x1='{x0}' y1='{py(v)}' x2='{x0 + w}' y2='{py(v)}' "
                 f"stroke='{GARIS}'/>")
        o.append(_t(x0 - 12, py(v) + 4, f"{v}%", 11, REDUP, "end"))
    for f in (2, 5, 10, 25, 50):
        o.append(_t(px(f), y0 + h + 22, f"{f}%", 11.5, REDUP, "middle"))

    d = " ".join(f"{'M' if i == 0 else 'L'} {px(f)} {py(v)}"
                 for i, (f, v, _) in enumerate(sorted(titik)))
    o.append(f"<path d='{d}' stroke='{JINGGA}' stroke-width='3' fill='none'/>")

    for f, v, lab in titik:
        atas = f > 40
        o.append(f"<circle cx='{px(f)}' cy='{py(v)}' r='7' "
                 f"fill='{HIJAU if atas else (MERAH if f < 5 else JINGGA)}'/>")
        o.append(_t(px(f), py(v) - 16, f"{v:.2f}%", 13, TINTA, "middle", 700))
        for j, s in enumerate(lab.split("\n")):
            o.append(_t(px(f), y0 + h + 46 + j * 14, s, 10.5, REDUP, "middle"))

    o.append(_t(x0 + w / 2, y0 + h + 94,
                "porsi kepala dari gambar yang masuk ke matcher (skala log)",
                12, REDUP, "middle", 600))

    # zona
    o.append(_kotak(x0, y0, px(9) - x0, h, "rgba(0,0,0,0)", MERAH, 6, 1.5))
    o.append(_t((x0 + px(9)) / 2, y0 + 20, "YOLO WAJIB", 12, MERAH,
                "middle", 700))
    o.append(_kotak(px(20), y0, x0 + w - px(20), h, "rgba(0,0,0,0)",
                    HIJAU, 6, 1.5))
    o.append(_t((px(20) + x0 + w) / 2, y0 + 20, "YOLO TIDAK PERLU", 12, HIJAU,
                "middle", 700))

    o.append(_t(24, H - 62, "Aturan yang bisa dipakai: kalau kepala mengisi "
                            "> 25% gambar, crop tidak menolong banyak. Di "
                            "bawah 10%, akurasinya runtuh.", 12, TINTA,
               tebal=700))
    o.append(_t(24, H - 40, "Zakynthos median 2,05% -> YOLO wajib. "
                            "SeaTurtleIDHeads & Zindi sudah berupa crop "
                            "kepala -> YOLO tidak perlu.", 11.5, REDUP))
    o.append(_t(24, H - 18, "ReunionTurtles foto sudah close-up (600x800) dan "
                            "crop tengah 70% sudah diuji: tidak menolong. "
                            "Jadi kemungkinan besar tidak perlu juga.",
               11.5, REDUP))
    return _svg(W, H, o, "Dosis-respons porsi kepala")


def main():
    print(f"menulis SVG ke {KELUAR}/")
    import numpy as np
    from evaluasi import muat
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    Eg, Eq = muat("raw", gal, qry)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    sg = np.array([r["side"] for r in gal])
    sq = np.array([r["side"] for r in qry])
    S = Eq @ Eg.T
    S[sq[:, None] != sg[None, :]] = -np.inf
    benar = id_g[np.argsort(-S, axis=1)] == id_q[:, None]
    recall = {k: float(benar[:, :k].any(1).mean() * 100)
              for k in (20, 30, 40, 50, 84)}

    _tulis("01_flowchart_pipeline.svg", flowchart_pipeline())
    _tulis("02_perjalanan_akurasi.svg", chart_perjalanan())
    _tulis("03_sapu_ukuran.svg", chart_ukuran())
    _tulis("04_pengaruh_k.svg", chart_k(recall))
    _tulis("05_metode_gagal.svg", chart_gagal())
    p = chart_pecahan()
    if p:
        _tulis("06_spesies_dan_sisi.svg", p)
    _tulis("07_flowchart_keputusan.svg", flowchart_keputusan())
    _tulis("08_dosis_kepala.svg", chart_dosis_kepala())
    print("selesai.")


if __name__ == "__main__":
    main()
