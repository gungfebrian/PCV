"""
Baca hasil evaluasi latih_arcface.py, lalu pasang STATS-nya ke turtle_mode.py.

Kenapa perlu: persen "P(individu sama)" dihitung dari distribusi jarak
SAMA/BEDA yang DIUKUR. Setiap deskriptor punya distribusi sendiri — memakai
angka MegaDescriptor generik untuk embedding ArcFace akan membuat persennya
salah tanpa memberi tanda apa pun (pelajaran Notion bagian 21).

Jalankan setelah latih_arcface.py selesai:
    .venv/bin/python pasang_kalibrasi.py
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE_DIR, "latih_arcface.log")
TARGET = os.path.join(BASE_DIR, "turtle_mode.py")


def baca_stats(log_path=LOG):
    """Ambil mean/std SAMA & BEDA plus Top-1 dari log evaluasi."""
    if not os.path.exists(log_path):
        return None
    teks = open(log_path).read()

    m_sama = re.search(r"jarak SAMA\s+([\d.]+)\s*\+/-\s*([\d.]+)", teks)
    m_beda = re.search(r"jarak BEDA\s+([\d.]+)\s*\+/-\s*([\d.]+)", teks)
    m_top = re.search(r"Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%", teks)
    if not (m_sama and m_beda):
        return None

    return {
        "sama": (float(m_sama.group(1)), float(m_sama.group(2))),
        "beda": (float(m_beda.group(1)), float(m_beda.group(2))),
        "top1": float(m_top.group(1)) if m_top else None,
        "top5": float(m_top.group(2)) if m_top else None,
    }


def akurasi_seimbang(s):
    """Akurasi seimbang terbaik pada ambang optimal, dari dua Gaussian.

    Dipakai sebagai angka 'keandalan kalibrasi' yang ditampilkan di panel —
    supaya pengguna tahu seberapa jauh persen itu boleh dipercaya.
    """
    import math
    (ms, ss), (mb, sb) = s["sama"], s["beda"]
    sd = (ss + sb) / 2.0

    def cdf(x, m):
        return 0.5 * (1 + math.erf((x - m) / (sd * math.sqrt(2))))

    terbaik = 0.5
    for i in range(2001):
        t = i / 2000.0
        # benar-terima = P(jarak<=t | sama); benar-tolak = P(jarak>t | beda)
        terbaik = max(terbaik, (cdf(t, ms) + (1 - cdf(t, mb))) / 2)
    return terbaik * 100


def pasang(stats, target=TARGET):
    (ms, ss), (mb, sb) = stats["sama"], stats["beda"]
    ak = akurasi_seimbang(stats)
    ambang = (ms + mb) / 2

    baris = (f'    # ArcFace fine-tuned di SeaTurtleIDHeads (latih_arcface.py),\n'
             f'    # diukur dengan split time-aware — Top-1 {stats["top1"]}%.\n'
             f'    "arcface": {{"sama": ({ms:.4f}, {ss:.4f}), '
             f'"beda": ({mb:.4f}, {sb:.4f}),\n'
             f'                 "ambang": {ambang:.3f}, "akurasi": {ak:.1f}}},\n')

    s = open(target).read()
    if '"arcface":' in s:
        s = re.sub(r'    # ArcFace fine-tuned.*?\n(    #.*?\n)*    "arcface":.*?\n(\s+"ambang".*?\n)?',
                   baris, s, flags=re.S)
    else:
        s = s.replace('}\nSTATS_AKTIF', baris + '}\nSTATS_AKTIF')
    open(target, "w").write(s)
    return ak, ambang


if __name__ == "__main__":
    st = baca_stats()
    if not st:
        print("Belum ada hasil evaluasi di latih_arcface.log.")
        raise SystemExit(1)
    ak, ambang = pasang(st)
    print(f"Terpasang ke turtle_mode.py:")
    print(f"  SAMA   {st['sama'][0]:.4f} +/- {st['sama'][1]:.4f}")
    print(f"  BEDA   {st['beda'][0]:.4f} +/- {st['beda'][1]:.4f}")
    print(f"  ambang {ambang:.3f}   keandalan {ak:.1f}%")
    print(f"  Top-1  {st['top1']}%   Top-5 {st['top5']}%")
