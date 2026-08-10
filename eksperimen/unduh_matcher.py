"""
Unduh bobot local feature matcher. JALANKAN DI MAC, bukan di sandbox.

Sandbox tempat eksperimen dijalankan memblokir github release, HuggingFace,
dan dl.fbaipublicfiles (semuanya 403) — persis seperti kasus Kaggle kemarin.
Jadi bobotnya diunduh di sini, lalu dibaca dari folder yang sama.

    ../.venv/bin/python unduh_matcher.py            # semua
    ../.venv/bin/python unduh_matcher.py aliked     # satu saja
    ../.venv/bin/python unduh_matcher.py --cek      # lihat apa yang sudah ada

Setelah selesai:

    MODEL=L python3 rerank.py --matcher aliked --k 20

PERINGATAN dari pengalaman repo ini
-----------------------------------
1. Paket PyPI bernama `xfeat` **BUKAN** XFeat CVPR 2024 — itu pustaka feature
   engineering tabular dari 2020. Jangan `pip install xfeat`.
2. `pola_sisik.py` (di tidak_penting/eksplorasi_lama/) mencatat bahwa LightGlue
   di kornia 0.8.3 mengembalikan NOL pasangan bahkan saat sebuah gambar
   dicocokkan dengan dirinya sendiri. Kalau memakai jalur kornia, uji dulu
   dengan `--cek` sebelum menjalankan 14.000 pasangan.
"""

import os
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
BOBOT = os.path.join(BASE, "bobot_matcher")

# Diunduh langsung lewat URL supaya tidak bergantung pada torch.hub, yang
# menaruh berkas di cache global dan sulit dilacak saat mau dipindah mesin.
LANGSUNG = {
    "xfeat": [
        ("xfeat.pt",
         "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"),
    ],
}

# Yang lebih mudah lewat paket + torch.hub / kornia.
LEWAT_PAKET = {
    "aliked": {
        "pip": ["kornia"],
        "cek": ("import kornia.feature as KF; "
                "m = KF.ALIKED(); print('ALIKED OK')"),
        "catatan": "kornia mengunduh bobot ALIKED saat kelasnya pertama dibuat",
    },
    "roma": {
        "pip": ["romatch"],
        "cek": ("from romatch import roma_outdoor; "
                "m = roma_outdoor(device='cpu'); print('RoMa OK')"),
        "catatan": ("berat: butuh DINOv2 ViT-L (~1,1 GB). Di CPU ~1-3 detik "
                    "per pasangan, jadi pakai k kecil. Di MPS jauh lebih cepat."),
    },
}


def unduh(nama, url, tujuan):
    if os.path.exists(tujuan) and os.path.getsize(tujuan) > 1024:
        print(f"  sudah ada: {nama} ({os.path.getsize(tujuan) // 1024} KB)")
        return True
    print(f"  mengunduh {nama} ...")
    try:
        urllib.request.urlretrieve(url, tujuan)
        print(f"  selesai: {nama} ({os.path.getsize(tujuan) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  GAGAL {nama}: {e}")
        return False


def kerjakan(kunci):
    os.makedirs(BOBOT, exist_ok=True)
    ok = True
    if kunci in LANGSUNG:
        print(f"[{kunci}] unduh langsung ke {BOBOT}")
        for nama, url in LANGSUNG[kunci]:
            ok &= unduh(nama, url, os.path.join(BOBOT, nama))
    if kunci in LEWAT_PAKET:
        c = LEWAT_PAKET[kunci]
        print(f"[{kunci}] {c['catatan']}")
        for p in c["pip"]:
            print(f"  pip install {p}")
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])
            ok &= r.returncode == 0
        print("  menguji pemuatan bobot ...")
        r = subprocess.run([sys.executable, "-c", c["cek"]])
        ok &= r.returncode == 0
    return ok


def cek():
    print(f"folder bobot: {BOBOT}")
    if os.path.isdir(BOBOT):
        for f in sorted(os.listdir(BOBOT)):
            p = os.path.join(BOBOT, f)
            print(f"  {f}  ({os.path.getsize(p) // 1024} KB)")
    else:
        print("  (belum ada)")
    print("\nyang bisa dimuat sekarang:")
    for nama, c in LEWAT_PAKET.items():
        r = subprocess.run([sys.executable, "-c", c["cek"]],
                           capture_output=True, text=True)
        print(f"  {nama:8} {'OK' if r.returncode == 0 else 'belum — ' + r.stderr.strip().splitlines()[-1][:70]}")


if __name__ == "__main__":
    arg = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--cek" in sys.argv:
        cek()
    else:
        target = arg or list(LANGSUNG) + list(LEWAT_PAKET)
        semua = all(kerjakan(k) for k in target)
        print("\nselesai." if semua else "\nada yang gagal — lihat pesan di atas.")
        print(f"Lalu jalankan: MODEL=L python3 rerank.py --matcher {target[0]} --k 20")
