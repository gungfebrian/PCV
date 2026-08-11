#!/bin/bash
# LoMa, RoMa, ALIKED dan matcher lain lewat `vismatch`. JALANKAN DI MAC.
#
# Kenapa tidak di sandbox eksperimen: `vismatch` butuh `poselib`, yang tidak
# punya wheel untuk aarch64 Linux. Hambatan yang sama persis dengan `romatch`.
# Bukan soal jaringan — wheel-nya memang tidak ada.
#
#     cd eksperimen && bash jalankan_vismatch_di_mac.sh
#
# vismatch mengunduh bobotnya sendiri dari HuggingFace saat matcher dibuat,
# jadi bobot_matcher/ tidak dipakai untuk jalur ini.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "== 1. pasang vismatch =="
$PY -m pip install -q vismatch || {
  echo "GAGAL. Coba: $PY -m pip install poselib vismatch"
  exit 1
}

echo
echo "== 2. UJI SELF-MATCH — jangan lewati =="
echo "   Cocokkan sebuah gambar dengan DIRINYA SENDIRI. Kalau hasilnya bukan"
echo "   ratusan inlier, matcher-nya keliru dan semua angka setelahnya tidak"
echo "   bisa dipercaya. Ini persis cara bug matcher XFeat ketahuan: dengan"
echo "   ratio test Lowe, dua gambar berbeda cuma menghasilkan 2 pasangan"
echo "   padahal self-match sempurna."
MODEL=L $PY - <<'PY'
import protokol as P, rerank as R
kat = P.baca_katalog(); gal, _ = P.bangun_split(kat)
p = gal[0]["path"]
for nama in ["vm:loma@512", "vm:roma@512", "vm:aliked-lightglue@512"]:
    try:
        m = R.buat_matcher(nama)
        a = m.ekstrak(p)
        s = m.skor(a, a)
        tanda = "OK" if s >= 100 else "MENCURIGAKAN"
        print(f"   {nama:26} self-match inlier {s:6.0f}   {tanda}")
    except SystemExit as e:
        raise
    except Exception as e:
        print(f"   {nama:26} GAGAL: {str(e).splitlines()[0][:60]}")
PY

echo
echo "== 3. jalankan tiap matcher, k=20 (3.360 pasangan per konfigurasi) =="
echo "   Ukuran 512 dipilih karena sapu ukuran menunjukkan makin besar makin"
echo "   baik untuk XFeat (256->512 menambah +15,5 poin Rank-1)."
for M in "vm:loma@512" "vm:loma-r@512" "vm:roma@512" "vm:aliked-lightglue@512" "vm:xfeat-lightglue@512"; do
  echo
  echo "   -> $M"
  MODEL=L $PY rerank.py --matcher "$M" --k 20 --budget 100000 || \
    echo "      dilewati (lihat pesan di atas)"
done

echo
echo "== 4. laporan gabungan =="
MODEL=L $PY grid_rerank.py --lapor

echo
echo "Pembanding yang sudah ada di repo (dihitung di Linux, k=20, n=168):"
echo "   MegaDesc-L saja              Rank-1 25.00"
echo "   + XFeat (tanpa resize)       Rank-1 42.26"
echo "   + XFeat + resize 512         Rank-1 62.50   <- terbaik saat ini"
echo
echo "CATATAN saat membandingkan: skor vismatch memakai num_inliers bawaan"
echo "tiap model, sedangkan SIFT dan XFeat di repo ini memakai RANSAC MAGSAC"
echo "dengan ambang identik. Estimator yang berbeda berarti perbandingannya"
echo "tidak sepenuhnya apple-to-apple — sebut itu saat melaporkan angkanya."
