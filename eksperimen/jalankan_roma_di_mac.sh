#!/bin/bash
# Jalankan RoMa di Mac. TIDAK bisa dijalankan di sandbox eksperimen —
# paket `romatch` butuh poselib dan fused-local-corr yang tidak punya wheel
# untuk aarch64 Linux. Bobotnya sudah lengkap di bobot_matcher/.
#
#     cd eksperimen && bash jalankan_roma_di_mac.sh
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "== 1. pasang romatch =="
$PY -m pip install -q romatch || {
  echo "GAGAL. Coba: $PY -m pip install poselib romatch"
  exit 1
}

echo
echo "== 2. UJI SELF-MATCH — jangan lewati langkah ini =="
echo "   Mencocokkan sebuah gambar dengan DIRINYA SENDIRI."
echo "   Kalau hasilnya bukan ratusan korespondensi, matcher-nya keliru dan"
echo "   semua angka setelahnya tidak bisa dipercaya. Ini persis cara bug"
echo "   matcher XFeat ketahuan."
MODEL=L $PY - <<'PY'
import cv2, protokol as P, rerank as R
m = R.RoMa()
kat = P.baca_katalog(); gal, _ = P.bangun_split(kat)
p = gal[0]["path"]
a = m.ekstrak(p)
s = m.skor(a, a)
print(f"   self-match inlier: {s:.0f}")
if s < 100:
    raise SystemExit("   GAGAL: self-match terlalu rendah. JANGAN lanjut.")
print("   OK — matcher berperilaku wajar.")
PY

echo
echo "== 3. jalankan RoMa, k=20 (3.360 pasangan) =="
echo "   Di MPS perkiraan 10-20 menit. Di CPU bisa 1-3 jam."
for KONDISI in raw resize512; do
  echo "   -> kondisi: $KONDISI"
  MODEL=L $PY rerank.py --matcher roma --kondisi "$KONDISI" --k 20 --budget 100000
done

echo
echo "== 4. laporan gabungan =="
MODEL=L $PY grid_rerank.py --lapor

echo
echo "Selesai. Salin hasil/reunion_L_squash/rerank_roma*.npy dan"
echo "grid_rerank.json kalau mau dibandingkan lintas mesin."
