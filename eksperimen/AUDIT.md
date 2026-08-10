# AUDIT — kondisi repo PCV sebelum diubah

Tanggal: 2026-08-10 · Acuan: `eksperimen-preprocessing-reid-penyu.md` §3, §5, §8

---

## 1. Struktur repo

Repo `PCV` bukan repo eksperimen re-ID. Ia campuran tiga proyek yang berbagi
satu venv:

| Bagian | Isi |
|---|---|
| Kartu remi | `pipeline.py`, `visualizer.py`, `has.py`, `game_*.py`, `Templatekartu*/`, `Card_Recognizer-master/` |
| Wajah manusia | `face_mode.py`, `models_wajah/`, `unduh_model_wajah.py` |
| **Penyu (relevan)** | `turtle_mode.py`, `megadescriptor.py`, `faceid_penyu.py`, `banding.py`, `banding_adil.py`, `latih_arcface.py`, `konsensus.py`, `pola_sisik.py`, `pasang_kalibrasi.py`, `ekspor_seed_poc.py`, `unduh_dataset_penyu.py` |

Git: 8 commit, HEAD `a4f723c "Turtle Detection"`, tidak ada branch eksperimen.

## 2. Yang sudah jalan

- `megadescriptor.py` memuat MegaDescriptor-T/L lewat timm, mengembalikan
  embedding **yang sudah di-L2 normalize**. Bobot ada di cache HF lokal.
- `banding_adil.py` dan `latih_arcface.py` punya split **time-aware**: galeri
  dari tahun-tahun awal, uji dari tahun terakhir. Arahnya benar.
- `latih_arcface.log` berisi hasil run nyata: ArcFace fine-tuned Top-1 46.0%.

## 3. Yang rusak / hilang

| # | Temuan | Dampak |
|---|---|---|
| A | ~~**Dataset ReunionTurtles tidak ada.**~~ **DIATASI 10/08** — diunduh manual lewat kaggle CLI di mesin lokal (sandbox tidak bisa menjangkau kaggle.com). Sekarang di `dataset_penyu/ReunionTurtles/`, hash `6a561d9dc6a5791e`. | Hasil utama sekarang dari ReunionTurtles: 336 foto, 84 individu, 50 hijau + 34 sisik, 168 kiri + 168 kanan. Breakdown per spesies §4 **bisa** dibuat. SeaTurtleIDHeads dipertahankan sebagai run pembanding skala (n = 1.246). |
| B | **Tidak ada kode eksperimen sama sekali.** Tidak ada split gallery/query berbasis tahun untuk 6 kondisi, tidak ada mAP, tidak ada wildlife-tools, tidak ada uji statistik. | Semua dibuat baru di `eksperimen/`. |
| C | `CLAUDE.md` menunjuk `docs/eksperimen-preprocessing-reid-penyu.md` — file itu tidak ada. `README.md` menyebut dataset **TurtleID2022**, `CLAUDE.md` menyebut **ReunionTurtles**, yang di disk **SeaTurtleIDHeads**. Tiga nama berbeda. | Sumber kebenaran tidak jelas. |
| D | `wildlife-tools`, `wildlife-datasets`, `pandas`, `statsmodels`, `streamlit` tidak terpasang di `.venv`. | Validasi metrik §4 dan statistik §5 tidak mungkin dijalankan apa adanya. |
| E | `latih_arcface.py` melakukan **fine-tuning**, yang dilarang §1 (model harus frozen). Hasilnya juga lebih buruk: 46.0% vs pembanding 60.6%. | Di luar scope; tidak dipakai. |
| F | Angka pembanding "60.6%" di `latih_arcface.py` adalah string yang di-hardcode di `print`, bukan hasil yang diukur di run yang sama. | Dua angka telanjang yang dibandingkan dengan mata — persis yang dilarang §5. |

## 4. Cek khusus §8

### ❌ Kebocoran sisi kiri↔kanan — **ADA**

`banding_adil.bagi_time_aware()` dan `latih_arcface.muat_daftar()` sama sekali
tidak membaca field `position` dari `annotations.json`. Foto diurutkan hanya
berdasarkan tanggal, lalu 4 pertama jadi galeri. Query sisi kiri bisa
dicocokkan ke galeri yang seluruhnya sisi kanan atau atas.

Ironisnya `konsensus.py` menulis di docstring-nya bahwa pemisahan sisi bernilai
+17–20 poin — tapi kode evaluasinya tidak menerapkannya.

**Diperbaiki** di `protokol.evaluasi_manual()`: similarity pasangan beda-sisi
diset `-inf` **sebelum** ranking, bukan difilter sesudahnya.

### ⚠️ Arah split gallery(tahun 1) / query(tahun 2) — **arah benar, cakupan salah**

Arahnya benar (tahun awal → galeri, tahun terakhir → query), tapi galeri
dipotong `awal[:4]` — hanya 4 foto paling awal. §3 meminta **semua** foto tahun
pertama.

Verifikasi arah: menukar galeri↔query menurunkan Rank-1 dari 55.78% ke 23.06%.
Arah yang dipakai memang yang benar.

### ✅ L2 normalize — **sudah benar**

`megadescriptor.deskriptor()` membagi dengan norma. Diverifikasi ulang di
embedding kami: norma min 0.99999988, max 1.00000012.

### ❌ Ukuran input & normalisasi — **temuan terbesar**

Dua masalah terpisah:

1. **Bobot bisa termuat salah tanpa error.** Checkpoint BVRA memakai tata letak
   Swin gaya lama (`downsample` di akhir stage); timm ≥1.0 menaruhnya di awal
   stage berikutnya. `load_state_dict(strict=False)` akan "berhasil" dengan
   bobot downsample acak. `protokol.muat_model()` memakai
   `timm.models.swin_transformer.checkpoint_filter_fn` dan menolak jalan kalau
   ada key yang tidak cocok.

2. **Transform yang "sesuai config": jawabannya berbeda per dataset.**
   `config.json` menyebut `crop_pct 0.9`, `bicubic`, `center crop`. Repo
   memakai `cv2.INTER_AREA` langsung ke ukuran input tanpa center crop.

   | Dataset | Sesuai config | Squash | Δ Rank-1 | p |
   |---|---|---|---|---|
   | SeaTurtleIDHeads (n=1.246) | 47.59 | **55.78** | **+8.19** [+5.94, +10.43] | 1.9e-12 |
   | ReunionTurtles (n=168) | 16.67 | 18.45 | +1.79 [−2.38, +5.95] | 0.581 |

   Sebabnya masuk akal dan justru menguatkan penjelasannya: SeaTurtleIDHeads
   sudah berupa crop kepala ketat, jadi center crop membuang sisik di tepi.
   ReunionTurtles adalah foto utuh dengan latar karang — center crop tidak
   membuang apa pun yang penting.

   **Pelajarannya bukan "crop_pct itu salah", melainkan "crop_pct bergantung
   pada framing datamu" — dan itu harus diuji, bukan diasumsikan, tiap kali
   datanya berganti.** `TRANSFORM="squash"` dikunci untuk semua kondisi demi
   konsistensi protokol.

### ✅ Individu di query tanpa pasangan di gallery — **nol**

`query_tanpa_pasangan_sisi_sama = 0`, `query_yang_tahunnya_tidak_lebih_baru = 0`.

Catatan: 230 dari 380 individu galeri **tidak pernah** menjadi jawaban benar —
mereka distraktor. Ini sengaja dipertahankan (galeri = semua foto tahun pertama,
termasuk individu yang hanya muncul satu tahun) karena itulah kondisi nyata di
lapangan.

### Catatan tambahan — baseline ReunionTurtles 25% dan itu bukan bug

Rank-1 25% (L-384) jauh di bawah ambang 50% §8, jadi kelima dugaan diperiksa
satu per satu dan semuanya bersih. Penjelasan yang tersisa: tugasnya memang
berat.

- Hanya **satu** foto gallery per individu per sisi — tidak ada multi-shot
  untuk dirata-rata.
- Jarak waktu gallery→query **median 4 tahun, maksimum 13 tahun**.
- Foto pemandangan bawah air penuh; latar karang ikut masuk embedding.
- Pemisahan embedding nyata tapi tipis: cosine pasangan BENAR 0.441 ± 0.162
  vs pasangan SALAH 0.244 ± 0.158.

Ambang 50% di §8 adalah heuristik yang mengandaikan setup lebih ramah. Untuk
protokol seketat ini, 25% adalah baseline yang wajar. Tebak acak = 1,19%.

## 5. Yang terblokir di lingkungan ini

| Terblokir | Sebab | Dampak |
|---|---|---|
| Unduh ReunionTurtles dari sandbox | Kaggle 403 dari proxy sandbox | Diatasi: diunduh manual di mesin lokal, lalu dibaca dari folder yang ter-mount |
| `huggingface.co` dari kode | Proxy 403 | Diatasi: cache HF lokal user di-mount, dimuat offline |
| `raw.githubusercontent.com`, `api.github.com`, `download.pytorch.org` | Proxy 403 | Tabel baseline resmi wildlife-tools per dataset tidak bisa diambil langsung |
| Menjalankan `.venv` milik user | venv macOS/arm64 Python 3.14; eksekusi hanya tersedia di sandbox Linux | Dibangun lingkungan Linux terpisah (torch 2.5.1 CPU, timm 1.0.11) |

## 6. Keputusan

1. Kode eksperimen ditulis baru di `eksperimen/`, terpisah dari kode lama.
   Kode penyu lama tidak diubah.
2. Protokol §3 dikunci di satu file (`protokol.py`) supaya tidak ada kondisi
   yang bisa diam-diam memakai setup berbeda.
3. Dataset utama: **ReunionTurtles**, hash `6a561d9dc6a5791e`. `protokol.py`
   mendukung dua dataset lewat `DATASET`, dua varian model lewat `MODEL`, dan
   dua transform lewat `TRANSFORM` — supaya perbandingan lintas-run memakai
   kode yang sama persis.
4. Model: **MegaDescriptor-L-384**. T-224 memberi 18.45%, L-384 memberi 25.00%
   di ReunionTurtles. Memakai T akan membuang headroom yang dibutuhkan untuk
   mendeteksi efek preprocessing.
5. `TRANSFORM="squash"` untuk semua kondisi.

## 7. Bias spesies: diuji, tidak terbukti (10 Agu 2026)

Klaim yang perlu diperiksa: *"dataset kebanyakan green turtle, jadi hasilnya
bias."* Komposisinya memang timpang, tapi akibatnya tidak terukur.

| Spesies | Individu | Foto | Query | Rank-1 (XFeat+resize512, k=50) |
|---|---|---|---|---|
| Green | 50 | 200 | 100 | 78,00% (78/100) |
| Hawksbill | 34 | 136 | 68 | 70,59% (48/68) |
| Olive ridley | 0 | 0 | 0 | — |

Selisih +7,41 poin. **Fisher exact p = 0,283**, CI95 selisih
**[-6,35, +20,76]** — melewati nol. Selisih 7 poin pada n segini adalah noise,
bukan bukti bias spesies.

Secara mekanis ini masuk akal: **tidak ada bobot yang dilatih pada penyu mana
pun.** MegaDescriptor beku, XFeat beku (dilatih untuk pencocokan gambar umum).
Tidak ada jalur yang bisa membuat model "lebih hafal" green karena fotonya
lebih banyak. Artinya angka hawksbill sudah merupakan hasil generalisasi
lintas spesies — dan alasan kuat untuk menguji olive ridley TANPA melatih
lebih dulu.

Arah sebaliknya di stage-1 saja: green 22,00% vs hawksbill 29,41%. Tambahan
alasan untuk tidak menyimpulkan apa pun dari selisih spesies pada n sekecil ini.

## 8. Kontrak matcher (10 Agu 2026)

Aplikasi dulu menebak jenis matcher lewat `hasattr(mm, "X")` lalu jatuh ke
`mm.det.detectAndCompute`. RoMa tidak punya keduanya, jadi memilih RoMa di UI
langsung melempar `AttributeError: 'RoMa' object has no attribute 'det'`.
Menebak tipe seperti itu akan rusak lagi setiap kali satu matcher ditambahkan.

Semua kelas matcher di `rerank.py` sekarang memenuhi kontrak yang sama:

    .ekstrak(path)          fitur dari berkas       (jalur eksperimen)
    .ekstrak_array(rgb)     fitur dari array RGB    (jalur kamera/aplikasi)
    .korespondensi(a, b)    -> (src, dst) sebelum RANSAC
    .skor(a, b)             -> jumlah inlier setelah RANSAC
    .KOORD_ASLI             koordinat sudah dalam ukuran gambar asli?
    .PUNYA_KEYPOINT         punya keypoint per gambar? (False untuk dense)

`KOORD_ASLI` mengungkap perbedaan yang sebelumnya tersembunyi: SIFT
mengembalikan koordinat pada gambar yang sudah diperkecil ke `SISI_PROSES`,
sedangkan XFeat sudah membaginya kembali. Overlay keypoint SIFT karena itu
salah tempat untuk foto besar.

Skornya sendiri TIDAK berubah — `_inlier()` mempertahankan perilaku lama
persis, termasuk mengembalikan jumlah pasangan mentah (0-3) saat homografi
tidak bisa diestimasi. Dijamin oleh tes kesetaraan array-vs-berkas.
