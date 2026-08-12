#!/bin/bash
# Dua pekerjaan yang tidak bisa dijalankan di sandbox. JALANKAN DI MAC.
#
#     cd eksperimen
#     bash jalankan_di_mac.sh yolo     # latih detektor kepala (~15 menit)
#     bash jalankan_di_mac.sh besar    # SeaTurtleIDHeads model L (~75 menit)
#     bash jalankan_di_mac.sh semua
#
# Kenapa harus di Mac:
#   yolo   -> bobot yolo11n.pt diunduh dari github.com/ultralytics/assets,
#             yang diblokir dari sandbox (ConnectionError, sudah diverifikasi)
#   besar  -> 7.582 gambar x ~600 ms = ~75 menit. Di sandbox perintah dibatasi
#             45 detik, jadi butuh ~113 panggilan berturut-turut.
#
# Keduanya RESUMABLE. Kalau berhenti di tengah, jalankan lagi perintah yang
# sama dan ia melanjutkan dari tempat terakhir.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
MODE="${1:-}"

# ------------------------------------------------------------------ YOLO
latih_yolo() {
  echo "=================================================================="
  echo " LATIH DETEKTOR KEPALA — data dari Zakynthos/bbox.csv"
  echo "=================================================================="
  echo
  echo "Tidak perlu mengunduh dataset apa pun. Zakynthos sudah punya 160"
  echo "kotak kepala anotasi manusia."
  echo
  echo "PENTING soal split: latih pada foto GALERI (tahun-1), uji pada foto"
  echo "QUERY (tahun-2). Sama dengan protokol §3, jadi detektornya tidak"
  echo "pernah melihat foto yang dipakai mengukur. Split acak akan menaruh"
  echo "foto dari sesi yang sama di kedua sisi dan angkanya jadi tidak berarti."
  echo

  $PY -m pip install -q ultralytics || {
    echo "GAGAL memasang ultralytics"; exit 1; }

  echo "-- 1/4 siapkan data (symlink dibuat ulang untuk jalur Mac)"
  rm -rf yolo_kepala/data
  DATASET=zakynthos $PY yolo_kepala.py --siapkan-zakynthos

  echo
  echo "-- 2/4 latih (yolo11n, 1 kelas)"
  echo "   80 gambar latih itu SEDIKIT. Kalau hasilnya buruk, itu belum tentu"
  echo "   berarti YOLO tidak bisa — bisa jadi cuma kurang data. Baru di situ"
  echo "   SeaTurtleID2022 (~8.700 foto bermask kepala) layak diunduh."
  DATASET=zakynthos $PY yolo_kepala.py --latih --epoch 60

  echo
  echo "-- 3/4 ukur kualitas deteksi (IoU vs kotak anotasi)"
  DATASET=zakynthos $PY yolo_kepala.py --ukur

  echo
  echo "-- 4/4 potong dataset lalu ukur dampaknya ke re-ID"
  DATASET=zakynthos MODEL=L $PY yolo_kepala.py --potong
  DATASET=zakynthos MODEL=L $PY jalankan.py kepala
  DATASET=zakynthos MODEL=L STAGE1=kepala $PY rerank.py \
      --matcher xfeat --kondisi kepala --k 10

  echo
  echo "PEMBANDING — plafon dengan kotak anotasi manusia (bukan YOLO):"
  echo "   stage-1 saja        Rank-1 63,75%"
  echo "   + XFeat k=10        Rank-1 67,50%   <- plafon"
  echo
  echo "Selisih antara angka YOLO di atas dan 67,50% ADALAH ukuran kualitas"
  echo "detektornya. Itu satu-satunya angka yang dicari dari langkah ini."
}

# --------------------------------------------------- SeaTurtleIDHeads L
jalankan_besar() {
  echo "=================================================================="
  echo " SeaTurtleIDHeads DENGAN MODEL L — replikasi apple-to-apple"
  echo "=================================================================="
  echo
  echo "Sekarang dataset ini satu-satunya yang dijalankan dengan T-224,"
  echo "sementara ReunionTurtles dan Zakynthos memakai L-384. Jadi angka"
  echo "+6,42 belum bisa dibandingkan langsung dengan +48,81 dan +55,00."
  echo
  echo "7.582 gambar. Perkiraan ~75 menit di CPU, jauh lebih cepat di MPS."
  echo "Resumable — kalau berhenti, jalankan lagi."
  echo

  echo "-- 1/3 embedding stage-1 (bagian paling lama)"
  DATASET=seaturtleheads MODEL=L $PY jalankan.py raw --budget=100000

  echo
  echo "-- 2/3 evaluasi stage-1"
  DATASET=seaturtleheads MODEL=L $PY evaluasi.py

  echo
  echo "-- 3/3 re-ranking XFeat"
  DATASET=seaturtleheads MODEL=L $PY rerank.py \
      --matcher xfeat --kondisi resize512 --k 20 --budget 100000

  echo
  echo "PEMBANDING — hasil dengan T-224:"
  echo "   stage-1             Rank-1 55,78%"
  echo "   + XFeat k=20        Rank-1 62,20%   (+6,42, p=6,0e-06)"
  echo
  echo "Pertanyaannya: apakah selisih +6,42 itu tetap kecil dengan model L,"
  echo "atau sebagian dari kecilnya memang karena T-224 lebih lemah?"
}

# ------------------------------------------- YOLO di dataset LAIN
lintas_dataset() {
  DS="${2:-reunion}"
  echo "=================================================================="
  echo " UJI DETEKTOR LINTAS DATASET -> $DS"
  echo "=================================================================="
  echo
  echo "Detektor dilatih pada penyu TEMPAYAN di Yunani (Zakynthos)."
  echo "Sekarang diuji pada dataset lain tanpa dilatih ulang sama sekali."
  echo
  echo "Kalau berhasil, itu bukti kuat ia akan bekerja untuk OLIVE RIDLEY"
  echo "Indonesia juga - spesies yang belum punya data sama sekali."
  echo

  if [ ! -f yolo_kepala/kepala.pt ]; then
    echo "GAGAL: bobot belum ada. Jalankan dulu: bash jalankan_di_mac.sh yolo"
    exit 1
  fi

  echo "-- 1/4 berapa kepala yang ketemu"
  DATASET=$DS MODEL=L $PY yolo_kepala.py --ukur

  echo
  echo "-- 2/4 potong seluruh dataset"
  DATASET=$DS MODEL=L $PY yolo_kepala.py --potong

  echo
  echo "-- 3/4 embedding stage-1 dari gambar ter-crop"
  DATASET=$DS MODEL=L $PY jalankan.py kepala --budget=100000

  echo
  echo "-- 4/4 ukur: stage-1 saja, lalu + XFeat"
  DATASET=$DS MODEL=L STAGE1=kepala $PY rerank.py \
      --matcher xfeat --kondisi kepala --k 40 --budget 100000

  echo
  if [ "$DS" = "reunion" ]; then
    echo "PEMBANDING ReunionTurtles:"
    echo "   stage-1 raw                Rank-1 25,00%"
    echo "   stage-1 resize512          Rank-1 24,40%   <- resize TIDAK menolong stage-1"
    echo "   + XFeat resize512 k=40     Rank-1 73,81%"
    echo
    echo "DUA KEMUNGKINAN, keduanya berguna:"
    echo
    echo " A. stage-1 kepala JAUH di atas 25%"
    echo "    -> crop menolong di sini juga. Dan kalau tembus ~60%,"
    echo "       XFeat kemungkinan jadi tidak perlu - pipeline 8x lebih murah."
    echo
    echo " B. stage-1 kepala tetap ~25%"
    echo "    -> foto ReunionTurtles memang sudah close-up, jadi crop tidak"
    echo "       menambah apa pun. Itu MENGONFIRMASI aturan kita:"
    echo "       subjek > 25% frame -> crop tidak perlu."
    echo
    echo "Yang B bukan kegagalan. Ia menutup pertanyaan yang selama ini"
    echo "cuma bisa dijawab dengan dugaan."
  fi
}

case "$MODE" in
  yolo)  latih_yolo ;;
  lintas) lintas_dataset "$@" ;;
  besar) jalankan_besar ;;
  semua) latih_yolo; echo; echo; jalankan_besar ;;
  ""|-h|--help|*)
     echo "pakai: bash jalankan_di_mac.sh <mode>"
     echo
     echo "  yolo    latih detektor kepala di Zakynthos      ~15 menit"
     echo "  lintas  uji detektor itu di ReunionTurtles      ~10 menit"
     echo "          (bisa juga: bash jalankan_di_mac.sh lintas seaturtleheads)"
     echo "  besar   SeaTurtleIDHeads dengan model L         ~75 menit"
     echo "  semua   yolo + besar"
     echo
     echo "Urutan yang disarankan: yolo -> lintas -> besar"
     [ -z "$MODE" ] && exit 0 || exit 1 ;;
esac

echo
echo "=================================================================="
echo " SELESAI. Periksa hasilnya di aplikasi:"
echo "   cd ../aplikasi"
echo "   DATASET=zakynthos MODEL=L ../.venv/bin/python penyu_live.py"
echo
echo " Papan skor di panel kanan akan otomatis memuat konfigurasi baru."
echo " Dan pastikan aplikasi masih setara dengan eksperimen:"
echo "   DATASET=zakynthos MODEL=L STAGE1=kepala ../.venv/bin/python \\"
echo "       uji_setara.py --n 20 --kondisi kepala --k 10"
echo "=================================================================="
