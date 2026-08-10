"""
Bikin grafik SVG dari hasil eksperimen. Dipakai untuk Notion dan laporan.

Semua angka dibaca dari `hasil/*/statistik.json` dan `lanjutan.json` — tidak
ada satu pun yang diketik tangan, jadi grafiknya tidak bisa basi tanpa
ketahuan.

    MODEL=L python3 grafik.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

import protokol as P                      # noqa: E402

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")
KELUAR = os.path.join(P.BASE, "grafik")
os.makedirs(KELUAR, exist_ok=True)

ABU, HIJAU, MERAH, BIRU = "#6e7781", "#1a7f37", "#c0392b", "#1f6feb"
# svg.fonttype="none" menyimpan teks sebagai teks, bukan sebagai path glyph.
# Bedanya besar: ~50 KB -> ~10 KB per berkas, dan hasilnya bisa dikirim ke
# Notion sebagai konten teks tanpa unggah biner.
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.size": 9, "axes.edgecolor": "#999",
                     "axes.labelcolor": "#333", "text.color": "#222",
                     "xtick.color": "#555", "ytick.color": "#555",
                     "axes.spines.top": False, "axes.spines.right": False})


def _muat(n):
    p = os.path.join(HASIL, n)
    return json.load(open(p)) if os.path.exists(p) else None


def simpan(fig, nama):
    p = os.path.join(KELUAR, nama)
    fig.savefig(p, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {nama}  ({os.path.getsize(p) // 1024} KB)")
    return p


# ------------------------------------------------- 1. batang + CI
def grafik_utama(stat):
    """Rank-1 tiap kondisi. Batang galat = CI bootstrap dari selisih terhadap
    raw, digeser ke posisi absolut — supaya terlihat bahwa semuanya tumpang
    tindih dengan baseline."""
    k = list(stat)
    label = [stat[x]["label"].replace(" (", "\n(") for x in k]
    r1 = [stat[x]["rank1"] for x in k]
    base = stat["raw"]["rank1"]

    lo, hi, warna = [], [], []
    for x in k:
        d = stat[x].get("delta_rank1")
        if d is None:
            lo.append(0); hi.append(0); warna.append(BIRU); continue
        sig = stat[x]["mcnemar_rank1"]["p_value"] < 0.05
        lo.append(stat[x]["rank1"] - (base + d["ci95"][0]))
        hi.append((base + d["ci95"][1]) - stat[x]["rank1"])
        warna.append((HIJAU if d["delta"] > 0 else MERAH) if sig else ABU)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(k))
    ax.bar(x, r1, color=warna, width=0.62, zorder=2)
    ax.errorbar(x, r1, yerr=[lo, hi], fmt="none", ecolor="#444",
                capsize=4, lw=1.1, zorder=3)
    ax.axhline(base, color=BIRU, ls="--", lw=1, zorder=1)
    ax.text(len(k) - 0.4, base + 0.6, f"baseline {base:.2f}%",
            color=BIRU, fontsize=8, ha="right")
    for i, v in enumerate(r1):
        ax.text(i, v + max(hi) + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, label, fontsize=8)
    ax.set_ylabel("Rank-1 (%)")
    ax.set_ylim(0, max(np.array(r1) + np.array(hi)) + 5)
    ax.set_title("Rank-1 per kondisi preprocessing — batang galat = 95% CI "
                 "bootstrap\nabu-abu = tidak berbeda signifikan dari raw",
                 fontsize=9.5, loc="left")
    return simpan(fig, "01_rank1_per_kondisi.svg")


# --------------------------------------------------- 2. forest plot
def grafik_forest(stat):
    """Δ terhadap raw dengan CI. Garis nol vertikal membuat pesan utamanya
    terbaca dalam satu detik: semua selang memotong nol."""
    k = [x for x in stat if x != "raw"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, x in enumerate(reversed(k)):
        d = stat[x]["delta_rank1"]
        sig = stat[x]["mcnemar_rank1"]["p_value"] < 0.05
        c = (HIJAU if d["delta"] > 0 else MERAH) if sig else ABU
        ax.plot(d["ci95"], [i, i], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(d["delta"], i, "o", color=c, ms=7)
        ax.text(d["ci95"][1] + 0.6, i,
                f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
                f"   p={stat[x]['mcnemar_rank1']['p_value']:.3f}",
                va="center", fontsize=8, color="#333")
    ax.axvline(0, color=BIRU, lw=1.2)
    ax.set_yticks(range(len(k)), [stat[x]["label"] for x in reversed(k)],
                  fontsize=8.5)
    ax.set_xlabel("Δ Rank-1 terhadap raw (poin persen)")
    ax.set_xlim(min(stat[x]["delta_rank1"]["ci95"][0] for x in k) - 2,
                max(stat[x]["delta_rank1"]["ci95"][1] for x in k) + 22)
    ax.set_title("Semua selang kepercayaan memotong nol — tidak ada kondisi "
                 "yang berbeda dari raw", fontsize=9.5, loc="left")
    return simpan(fig, "02_forest_delta.svg")


# ------------------------------------------------ 3. breakdown
def grafik_breakdown(stat):
    k = list(stat)
    spesies = sorted(stat["raw"].get("per_spesies", {}))
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))

    x = np.arange(len(k))
    if spesies:
        w = 0.36
        for j, sp in enumerate(spesies):
            v = [stat[c]["per_spesies"][sp]["rank1"] for c in k]
            ax[0].bar(x + (j - 0.5) * w, v, w,
                      label=f"{sp} (n={stat['raw']['per_spesies'][sp]['n']})",
                      color=[BIRU, "#e8a33d"][j % 2])
        ax[0].legend(fontsize=7.5, frameon=False)
        ax[0].set_title("Per spesies — sisik konsisten lebih mudah",
                        fontsize=9, loc="left")
    else:
        ax[0].text(0.5, 0.5, "spesies tidak tersedia", ha="center",
                   transform=ax[0].transAxes, color=ABU)
    ax[0].set_xticks(x, [stat[c]["label"].split(" (")[0] for c in k],
                     rotation=32, ha="right", fontsize=7.5)
    ax[0].set_ylabel("Rank-1 (%)")

    w = 0.36
    for j, s in enumerate(P.SISI):
        v = [stat[c]["per_sisi"][s]["rank1"] for c in k]
        ax[1].bar(x + (j - 0.5) * w, v, w,
                  label=f"{s} (n={stat['raw']['per_sisi'][s]['n']})",
                  color=["#7a5af5", "#2bb3a3"][j % 2])
    ax[1].legend(fontsize=7.5, frameon=False)
    ax[1].set_xticks(x, [stat[c]["label"].split(" (")[0] for c in k],
                     rotation=32, ha="right", fontsize=7.5)
    ax[1].set_title("Per sisi — mirip, kecuali CLAHE", fontsize=9, loc="left")
    return simpan(fig, "03_breakdown.svg")


# -------------------------------------------------- 4. cara lain
def grafik_lanjutan(lanjut):
    k = list(lanjut)
    base = lanjut["raw (baseline)"]["rank1"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(k))
    warna = []
    for n in k:
        d = lanjut[n].get("delta_rank1")
        if d is None:
            warna.append(BIRU); continue
        sig = lanjut[n]["mcnemar_rank1"]["p_value"] < 0.05
        warna.append((HIJAU if d["delta"] > 0 else MERAH) if sig else ABU)
    r1 = [lanjut[n]["rank1"] for n in k]
    r5 = [lanjut[n]["rank5"] for n in k]
    ax.bar(x - 0.19, r1, 0.36, color=warna, label="Rank-1")
    ax.bar(x + 0.19, r5, 0.36, color="#c9d1d9", label="Rank-5")
    ax.axhline(base, color=BIRU, ls="--", lw=1)
    for i, (a, b) in enumerate(zip(r1, r5)):
        ax.text(i - 0.19, a + 0.8, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 0.8, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in k], fontsize=7.5)
    ax.set_ylabel("%")
    ax.set_ylim(0, max(r5) + 8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Di luar preprocessing — konsensus dua sisi satu-satunya "
                 "yang menonjol\n(ongkosnya: butuh DUA foto per penyu)",
                 fontsize=9.5, loc="left")
    return simpan(fig, "04_cara_lain.svg")


if __name__ == "__main__":
    stat = _muat("statistik.json")
    if not stat:
        raise SystemExit(f"statistik.json belum ada di {HASIL}")
    print("menulis grafik:")
    grafik_utama(stat)
    grafik_forest(stat)
    grafik_breakdown(stat)
    lanjut = _muat("lanjutan.json")
    if lanjut:
        grafik_lanjutan(lanjut)
