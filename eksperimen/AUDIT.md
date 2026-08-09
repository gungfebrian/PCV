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
| A | **Dataset ReunionTurtles tidak ada.** Yang ada: `SeaTurtleIDHeads` (400 individu, 7.582 foto) dan `ZindiTurtleRecall` (13.894 foto). Kaggle diblokir dari lingkungan eksekusi, jadi tidak bisa diunduh. | Protokol dijalankan di SeaTurtleIDHeads. **Tidak ada kolom spesies** → breakdown per spesies §4 tidak bisa dibuat. Foto sudah crop kepala → kondisi "crop kepala" berubah makna. |
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

2. **Transform yang "sesuai config" justru lebih buruk.** `config.json`
   MegaDescriptor-T-224 menyebut `crop_pct 0.9`, `bicubic`, `center crop`.
   Repo memakai `cv2.INTER_AREA` langsung ke 224×224 tanpa center crop. Diuji
   di seluruh 1.246 query:

   | Transform | Rank-1 | Rank-5 | mAP |
   |---|---|---|---|
   | Sesuai config (bicubic + center crop 0.9) | 47.59 | 62.44 | 45.61 |
   | Squash INTER_AREA 224×224 (cara repo) | **55.78** | **68.38** | **53.61** |

   Δ Rank-1 = **+8.19** [+5.94, +10.43], McNemar p = 1.9e-12 (n01=158, n10=56).

   Penjelasannya konsisten: `crop_pct` dirancang untuk foto pemandangan penuh.
   SeaTurtleIDHeads sudah berupa crop kepala ketat — center crop di atasnya
   membuang sisik di tepi, justru informasi identitasnya.

   **Inilah sebab baseline sempat di bawah 50%.** Sesuai §8, penyebabnya dicari
   dulu sebelum menyentuh preprocessing. `TRANSFORM="squash"` dikunci untuk
   semua kondisi.

### ✅ Individu di query tanpa pasangan di gallery — **nol**

`query_tanpa_pasangan_sisi_sama = 0`, `query_yang_tahunnya_tidak_lebih_baru = 0`.

Catatan: 230 dari 380 individu galeri **tidak pernah** menjadi jawaban benar —
mereka distraktor. Ini sengaja dipertahankan (galeri = semua foto tahun pertama,
termasuk individu yang hanya muncul satu tahun) karena itulah kondisi nyata di
lapangan.

## 5. Yang terblokir di lingkungan ini

| Terblokir | Sebab | Dampak |
|---|---|---|
| Unduh ReunionTurtles | Kaggle 403 dari sandbox, dan butuh kredensial | Ganti dataset, breakdown spesies hilang |
| `huggingface.co` dari kode | Proxy 403 | Diatasi: cache HF lokal user di-mount, dimuat offline |
| `raw.githubusercontent.com`, `api.github.com`, `download.pytorch.org` | Proxy 403 | Tabel baseline resmi wildlife-tools per dataset tidak bisa diambil langsung |
| Menjalankan `.venv` milik user | venv macOS/arm64 Python 3.14; eksekusi hanya tersedia di sandbox Linux | Dibangun lingkungan Linux terpisah (torch 2.5.1 CPU, timm 1.0.11) |

## 6. Keputusan

1. Kode eksperimen ditulis baru di `eksperimen/`, terpisah dari kode lama.
   Kode penyu lama tidak diubah.
2. Protokol §3 dikunci di satu file (`protokol.py`) supaya tidak ada kondisi
   yang bisa diam-diam memakai setup berbeda.
3. Dataset: SeaTurtleIDHeads, hash `39a1c6603055f5d8`. Substitusi ini ditulis
   di header setiap laporan, bukan disembunyikan.
4. `TRANSFORM="squash"` untuk semua kondisi, berdasarkan A/B di atas.
