"""
SVG ringkas untuk Notion — ditulis tangan, bukan keluaran matplotlib.

Kenapa tidak pakai `grafik.py` saja: matplotlib menghasilkan 14–25 KB per
berkas. SVG di sini ~2 KB, bisa dibaca manusia, dan bisa dikirim ke Notion
sebagai konten teks tanpa unggah biner. `grafik.py` tetap dipakai untuk
laporan dan slide yang butuh kualitas cetak.

Semua angka dibaca dari hasil/*.json — tidak ada yang diketik tangan.

    MODEL=L python3 grafik_notion.py
"""

import json
import os

import protokol as P

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")
KELUAR = os.path.join(P.BASE, "grafik")
os.makedirs(KELUAR, exist_ok=True)

ABU, HIJAU, MERAH, BIRU, GARIS = "#6e7781", "#1a7f37", "#c0392b", "#1f6feb", "#d0d7de"
FONT = "font-family='-apple-system,Segoe UI,Roboto,sans-serif'"


def _muat(n):
    p = os.path.join(HASIL, n)
    return json.load(open(p)) if os.path.exists(p) else None


def _tulis(nama, isi):
    p = os.path.join(KELUAR, nama)
    open(p, "w").write(isi)
    print(f"  {nama}  ({len(isi)} byte)")
    return p


def t(x, y, s, uk=11, w="#24292f", anchor="start", tebal="normal"):
    s = (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{uk}' "
            f"fill='{w}' text-anchor='{anchor}' font-weight='{tebal}'>{s}</text>")


def _warna(delta, p):
    if p >= 0.05:
        return ABU
    return HIJAU if delta > 0 else MERAH


# ------------------------------------------------------- 1. batang
def batang(stat):
    """Rank-1 tiap kondisi + garis baseline."""
    k = list(stat)
    W, H = 720, 300
    kiri, atas, tinggi_plot = 210, 44, 200
    maks = max(s["rank1"] for s in stat.values()) * 1.25
    lebar_maks = W - kiri - 130
    base = stat["raw"]["rank1"]

    o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         t(12, 20, "Rank-1 per kondisi preprocessing", 13, "#24292f",
           tebal="600"),
         t(12, 36, f"ReunionTurtles · MegaDescriptor-L-384 · n={stat['raw']['n']} query",
           10, ABU)]

    hb = tinggi_plot / len(k) - 8
    for i, key in enumerate(k):
        s = stat[key]
        y = atas + i * (tinggi_plot / len(k))
        d = s.get("delta_rank1")
        pval = s["mcnemar_rank1"]["p_value"] if d else None
        c = BIRU if d is None else _warna(d["delta"], pval)
        lb = s["rank1"] / maks * lebar_maks
        o.append(t(kiri - 8, y + hb * 0.72, s["label"], 10.5, "#24292f", "end"))
        o.append(f"<rect x='{kiri}' y='{y}' width='{lb:.1f}' height='{hb:.1f}' "
                 f"fill='{c}' rx='2'/>")
        o.append(t(kiri + lb + 6, y + hb * 0.72, f"{s['rank1']:.2f}%", 10.5))
        if d:
            o.append(t(kiri + lb + 52, y + hb * 0.72,
                       f"{d['delta']:+.2f}  p={pval:.3f}", 9.5, c))

    xb = kiri + base / maks * lebar_maks
    o.append(f"<line x1='{xb:.1f}' y1='{atas - 6}' x2='{xb:.1f}' "
             f"y2='{atas + tinggi_plot}' stroke='{BIRU}' stroke-width='1.2' "
             f"stroke-dasharray='4 3'/>")
    o.append(t(xb + 4, atas - 10, "baseline", 9, BIRU))
    o.append(t(12, H - 10,
               "Abu-abu = tidak berbeda signifikan dari raw (p ≥ 0.05)",
               9.5, ABU))
    o.append("</svg>")
    return _tulis("notion_01_batang.svg", "\n".join(o))


# -------------------------------------------------------- 2. forest
def forest(stat):
    """Δ terhadap raw + CI 95%. Garis nol membuat pesannya terbaca sekilas."""
    k = [x for x in stat if x != "raw"]
    W, H = 720, 290
    kiri, atas, tinggi_plot = 210, 52, 175
    lo = min(stat[x]["delta_rank1"]["ci95"][0] for x in k) - 1
    hi = max(stat[x]["delta_rank1"]["ci95"][1] for x in k) + 1
    lebar = W - kiri - 190

    def px(v):
        return kiri + (v - lo) / (hi - lo) * lebar

    o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         t(12, 20, "Δ Rank-1 terhadap raw, dengan 95% CI bootstrap", 13,
           "#24292f", tebal="600"),
         t(12, 36, "Semua selang memotong nol — tidak ada kondisi yang berbeda "
                   "dari raw", 10, ABU)]

    x0 = px(0)
    o.append(f"<line x1='{x0:.1f}' y1='{atas - 12}' x2='{x0:.1f}' "
             f"y2='{atas + tinggi_plot}' stroke='{BIRU}' stroke-width='1.5'/>")
    o.append(t(x0, atas - 18, "0", 10, BIRU, "middle"))

    dy = tinggi_plot / len(k)
    for i, key in enumerate(k):
        s = stat[key]
        d, pval = s["delta_rank1"], s["mcnemar_rank1"]["p_value"]
        c = _warna(d["delta"], pval)
        y = atas + i * dy + dy / 2
        a, b = px(d["ci95"][0]), px(d["ci95"][1])
        o.append(t(kiri - 8, y + 4, s["label"], 10.5, "#24292f", "end"))
        o.append(f"<line x1='{a:.1f}' y1='{y:.1f}' x2='{b:.1f}' y2='{y:.1f}' "
                 f"stroke='{c}' stroke-width='3' stroke-linecap='round'/>")
        o.append(f"<circle cx='{px(d['delta']):.1f}' cy='{y:.1f}' r='4.5' "
                 f"fill='{c}'/>")
        o.append(t(W - 184, y + 4,
                   f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, "
                   f"{d['ci95'][1]:+.2f}]  p={pval:.3f}", 9.5, "#24292f"))

    for v in range(int(lo // 5 * 5), int(hi) + 5, 5):
        if lo <= v <= hi:
            o.append(t(px(v), atas + tinggi_plot + 16, str(v), 9, ABU, "middle"))
    o.append(t(kiri + lebar / 2, H - 8, "poin persen", 9.5, ABU, "middle"))
    o.append("</svg>")
    return _tulis("notion_02_forest.svg", "\n".join(o))


# ----------------------------------------------------- 3. breakdown
def breakdown(stat):
    """Spesies berdampingan — pola yang bertahan di semua kondisi."""
    k = list(stat)
    sp = sorted(stat["raw"].get("per_spesies", {}))
    if not sp:
        return None
    W, H = 720, 280
    kiri, atas, tinggi_plot = 210, 56, 180
    maks = max(stat[c]["per_spesies"][s]["rank1"] for c in k for s in sp) * 1.2
    lebar = W - kiri - 110
    warna = {sp[0]: BIRU, sp[1] if len(sp) > 1 else "": "#e8a33d"}

    o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         t(12, 20, "Rank-1 per spesies", 13, "#24292f", tebal="600"),
         t(12, 36, "Penyu sisik konsisten lebih mudah dari penyu hijau — "
                   "di keenam kondisi, tanpa kecuali", 10, ABU)]
    for j, s in enumerate(sp):
        o.append(f"<rect x='{560 + j * 78}' y='14' width='9' height='9' "
                 f"fill='{warna[s]}' rx='2'/>")
        o.append(t(573 + j * 78, 22,
                   f"{s} (n={stat['raw']['per_spesies'][s]['n']})", 9.5))

    dy = tinggi_plot / len(k)
    hb = dy / 2 - 3
    for i, c in enumerate(k):
        y = atas + i * dy
        o.append(t(kiri - 8, y + dy / 2 + 2, stat[c]["label"].split(" (")[0],
                   10.5, "#24292f", "end"))
        for j, s in enumerate(sp):
            v = stat[c]["per_spesies"][s]["rank1"]
            lb = v / maks * lebar
            yy = y + j * (hb + 2)
            o.append(f"<rect x='{kiri}' y='{yy:.1f}' width='{lb:.1f}' "
                     f"height='{hb:.1f}' fill='{warna[s]}' rx='2'/>")
            o.append(t(kiri + lb + 5, yy + hb * 0.8, f"{v:.1f}", 9))
    o.append("</svg>")
    return _tulis("notion_03_spesies.svg", "\n".join(o))


# ----------------------------------------------------- 4. cara lain
def cara_lain(lanjut):
    k = list(lanjut)
    W, H = 720, 290
    kiri, atas, tinggi_plot = 200, 52, 190
    maks = max(v["rank5"] for v in lanjut.values()) * 1.18
    lebar = W - kiri - 170
    base = lanjut["raw (baseline)"]["rank1"]

    o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         t(12, 20, "Di luar preprocessing", 13, "#24292f", tebal="600"),
         t(12, 36, "Konsensus dua sisi satu-satunya yang menonjol — "
                   "ongkosnya: butuh DUA foto per penyu", 10, ABU),
         f"<rect x='560' y='12' width='9' height='9' fill='{BIRU}' rx='2'/>",
         t(573, 20, "Rank-1", 9.5),
         "<rect x='632' y='12' width='9' height='9' fill='#c9d1d9' rx='2'/>",
         t(645, 20, "Rank-5", 9.5)]

    dy = tinggi_plot / len(k)
    hb = dy / 2 - 3
    for i, nama in enumerate(k):
        v = lanjut[nama]
        y = atas + i * dy
        d = v.get("delta_rank1")
        pval = v["mcnemar_rank1"]["p_value"] if d else None
        c = BIRU if d is None else _warna(d["delta"], pval)
        o.append(t(kiri - 8, y + dy / 2 + 2, nama, 10.5, "#24292f", "end"))
        for j, (kunci, wr) in enumerate((("rank1", c), ("rank5", "#c9d1d9"))):
            lb = v[kunci] / maks * lebar
            yy = y + j * (hb + 2)
            o.append(f"<rect x='{kiri}' y='{yy:.1f}' width='{lb:.1f}' "
                     f"height='{hb:.1f}' fill='{wr}' rx='2'/>")
            o.append(t(kiri + lb + 5, yy + hb * 0.8, f"{v[kunci]:.1f}", 9))
        if d:
            o.append(t(W - 150, y + dy / 2 + 2,
                       f"{d['delta']:+.2f}  p={pval:.3f}", 9.5, c))

    xb = kiri + base / maks * lebar
    o.append(f"<line x1='{xb:.1f}' y1='{atas - 4}' x2='{xb:.1f}' "
             f"y2='{atas + tinggi_plot}' stroke='{BIRU}' stroke-width='1.2' "
             f"stroke-dasharray='4 3'/>")
    o.append("</svg>")
    return _tulis("notion_04_cara_lain.svg", "\n".join(o))


if __name__ == "__main__":
    stat = _muat("statistik.json")
    if not stat:
        raise SystemExit(f"statistik.json belum ada di {HASIL}")
    print("menulis SVG ringkas:")
    batang(stat)
    forest(stat)
    breakdown(stat)
    lanjut = _muat("lanjutan.json")
    if lanjut:
        cara_lain(lanjut)
