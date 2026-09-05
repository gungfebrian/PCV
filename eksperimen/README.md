# eksperimen/ — evaluasi preprocessing dan backbone re-ID penyu

Eksperimen, bukan produk. Kode lama repo tidak diubah.

**Baca `AUDIT.md` dulu** kalau mau tahu apa yang rusak di repo sebelum ini
dikerjakan, dan kenapa beberapa angka tidak boleh dibandingkan langsung.

## Jawaban singkat

**MiewID-msv3 adalah backbone yang direkomendasikan.** Pada ReunionTurtles,
rank-1 naik dari 25,00% dengan MegaDescriptor-L-384 menjadi 84,52% dengan
MiewID. Perbandingan ini memakai dataset, split berbasis tahun, dan aturan sisi
yang sama.

Preprocessing warna atau normalisasi tidak memberi kenaikan signifikan: white
balance, CLAHE, dan grayscale tidak membantu secara konsisten. Perubahan yang
berguna adalah **framing kepala**, terutama saat kepala hanya mengisi sebagian
kecil foto. Karena itu, hasil MegaDescriptor tetap dipertahankan sebagai
baseline eksperimen, tetapi bukan lagi pilihan utama aplikasi.

## Baseline MegaDescriptor

Tabel berikut mendokumentasikan eksperimen awal dengan MegaDescriptor-L-384.
Angka ini tidak boleh dicampur dengan hasil MiewID tanpa menyebut backbone.

| Kondisi | Rank-1 | Rank-5 | mAP | Δ Rank-1 vs raw (95% CI) | p (McNemar) |
|---|---|---|---|---|---|
| **Raw (baseline)** | **25.00** | 46.43 | **37.40** | — | — |
| Crop kepala (center 70%) | 22.62 | 46.43 | 34.49 | −2.38 [−8.33, +3.57] | 0.557 |
| White balance (gray-world) | 22.62 | 48.21 | 35.59 | −2.38 [−7.74, +2.98] | 0.503 |
| CLAHE (L, clip 2.0) | 21.43 | 47.62 | 34.71 | −3.57 [−9.52, +2.38] | 0.327 |
| Grayscale | 20.24 | 46.43 | 32.73 | −4.76 [−11.90, +2.38] | 0.256 |
| Crop + WB | 22.62 | **51.19** | 35.18 | −2.38 [−8.93, +4.17] | 0.585 |

ReunionTurtles · hash `6a561d9dc6a5791e` · MegaDescriptor-L-384 frozen ·
n = 168 query · gallery 168 foto / 84 individu (50 hijau + 34 sisik).

Dengan n = 168, satu prediksi bernilai 0,60 poin. Selisih terbesar yang
terukur (4,76 poin) setara 8 prediksi. Itu noise — persis yang diperingatkan
di §5 spesifikasi.

## Baseline MiewID yang direkomendasikan

MiewID-msv3 sudah direproduksi dengan protokol PCV yang sama: galeri tahun
pertama, query tahun terakhir, embedding L2-normalized, cosine similarity, dan
pencarian yang dikunci per sisi.

| Model · kondisi raw | Rank-1 | Rank-5 | mAP | Δ Rank-1 vs MD (95% CI) | p (McNemar) |
|---|---:|---:|---:|---:|---:|
| MegaDescriptor-L-384 | 25,00 | 46,43 | 37,40 | — | — |
| **MiewID-msv3** | **84,52** | **95,24** | **89,16** | **+59,52 [+51,79, +67,26]** | **5,4 × 10⁻²⁸** |

Breakdown MiewID raw wajib disertakan saat hasil dikutip:

| Pecahan | Rank-1 | Rank-5 | mAP | n |
|---|---:|---:|---:|---:|
| Green | 95,00 | 98,00 | 96,54 | 100 |
| Hawksbill | 69,12 | 91,18 | 78,32 | 68 |
| Kiri | 82,14 | 92,86 | 86,76 | 84 |
| Kanan | 86,90 | 97,62 | 91,57 | 84 |

Hasil lengkap, termasuk validasi lintas-repo dan eksperimen Amvrakikos serta
Zakynthos, ada di `docs/temuan/2026-08-18-miewid-vs-megadescriptor.md`.

## Yang justru menaikkan akurasi

| Perubahan | Rank-1 | Catatan |
|---|---|---|
| T-224 → **L-384** | 18.45 → **25.00** | Varian model lebih menentukan daripada preprocessing mana pun |
| **Konsensus dua sisi** | 25.00 → **30.95** | Δ mAP +5.43 [+1.48, +9.63] **signifikan**; Rank-1 p=0.076. **Butuh 2 foto per penyu** |
| Fusi T+L | 25.00 → 27.38 | p = 0.557, tidak signifikan |
| Koreksi hubness | 25.00 → 23.21 | tidak menolong |

## Breakdown

Penyu **sisik konsisten lebih mudah dari penyu hijau di keenam kondisi**
(raw: 29.41% vs 22.00%). Pola yang bertahan di semua kondisi jauh lebih
meyakinkan daripada satu selisih tunggal. Kiri vs kanan tidak berbeda
meyakinkan (26.19 vs 23.81).

## Berkas

| Berkas | Guna |
|---|---|
| `protokol.py` | Protokol §3 dikunci di sini. Split, matching, metrik, kondisi, dataset, dan pilihan backbone. |
| `jalankan.py` | Hitung embedding per kondisi. Resumable. |
| `evaluasi.py` | Metrik + breakdown + validasi manual vs wildlife-tools (`--validasi`). |
| `statistik.py` | McNemar + bootstrap CI. Mencetak tabel utama. |
| `transform_ab.py` | A/B transform input (`--uji` setelah embedding lengkap). |
| `lanjutan.py` | Konsensus dua sisi, koreksi hubness, fusi T+L. |
| `sanity.py` | Sanity check §8. |
| `kasus_gagal.py` | 5 kasus gagal + top-5 gallery. |
| `app_demo.py` | Demo Streamlit, 5 tab. Tidak memuat model. |
| `rerank.py` | **Stage 2** — re-ranking local feature di atas top-k stage 1. |
| `unduh_matcher.py` | Unduh bobot ALIKED/XFeat/RoMa. **Jalankan di Mac**, bukan sandbox. |
| `uji.py` | 24 tes invarian protokol. Jalankan sebelum percaya angka. |
| `AUDIT.md` | Audit repo + hasil sanity check §8. |

## Stage 2 — XFeat: hasil positif pertama, dan besar

| Konfigurasi | Rank-1 | Rank-5 | mAP | hijau | sisik | Δ Rank-1 | p |
|---|---|---|---|---|---|---|---|
| stage-1 saja | 25.00 | 46.43 | 37.40 | 22.00 | 29.41 | — | — |
| **+ XFeat · murni** | **42.26** | 55.95 | **50.04** | 42.00 | 42.65 | **+17.26 [+8.93, +25.60]** | **0.0001** |
| + XFeat · RRF | 33.33 | 54.17 | 43.96 | 30.00 | 38.24 | +8.33 [+2.98, +13.69] | 0.0066 |
| + SIFT · murni | 20.83 | 39.88 | 30.60 | 31.00 | 5.88 | −4.17 | 0.382 |
| + SIFT · RRF | 29.17 | 54.17 | 40.89 | 35.00 | 20.59 | +4.17 | 0.311 |

Δ mAP XFeat murni = **+12.64 [+6.01, +19.34]**, signifikan. XFeat menolong
**kedua** spesies; SIFT menghancurkan penyu sisik (29 → 6).

XFeat × preprocessing — semuanya dalam 2,4 poin, jadi kesimpulan preprocessing
tetap berlaku di stage-2:
raw **42.26** · grayscale 42.26 · CLAHE 41.67 · crop 41.07 · WB 40.48 ·
crop+WB 39.88. Grayscale identik dengan raw karena XFeat meng-grayscale
inputnya sendiri di `forward()`.

**Arsitektur XFeat direkonstruksi dari state_dict** (`xfeat_lokal.py`) karena
repo aslinya diblokir. Dibuktikan benar oleh dua tes: `strict=True` load lolos
(1,54 M param), dan self-match memberi 2048/2048 inlier sempurna. ALIKED dan
RoMa masih terblokir — muncul sebagai "belum ada bobot", **bukan** angka
perkiraan.

```bash
MODEL=L python3 grid_rerank.py          # panggil ulang sampai selesai
MODEL=L python3 grid_rerank.py --lapor
```

## Stage 2 — local feature re-ranking

Plafon re-ranking = recall@k stage 1. Galeri per sisi cuma 84 foto, jadi k=84
= re-rank seluruh galeri, plafon 100%. Dengan SIFT + deskriptor di-cache:
2–5 ms/pasangan, 14.112 pasangan selesai ~75 detik.

| Varian | Rank-1 | Rank-5 | mAP | Δ Rank-1 | p |
|---|---|---|---|---|---|
| stage 1 saja | 25.00 | 46.43 | 37.40 | — | — |
| re-rank murni | 20.83 | 39.88 | 30.60 | −4.17 | 0.382 |
| **RRF** | **29.17** | **54.17** | **40.89** | +4.17 | 0.311 |

**Efeknya berlawanan arah per spesies, dan angka gabungan menyembunyikan
keduanya:** RRF menaikkan penyu hijau **+13.00 [+5.00, +22.00] p=0.0044**,
sementara re-rank murni menjatuhkan penyu sisik **−23.53 [−35.29, −11.76]
p=0.0004**. Keduanya lolos koreksi Bonferroni (ambang 0.005).

Mekanismenya: pasangan benar hampir tidak pernah gagal cocok (0–2% di bawah
4 inlier). Yang terjadi, pasangan **salah** mencocok lebih kuat — inlier
salah-terbaik mengalahkan inlier benar di **80% query**. Tersangka: latar
karang/pasir. Center crop 70% hanya menolong sedikit (benar-menang hijau
31→34%), jadi hipotesis latar belum terbantah tapi juga belum teruji layak —
butuh detektor kepala atau masking.

```bash
MODEL=L python3 rerank.py --matcher sift --k 84
MODEL=L python3 rerank.py --matcher sift --kondisi crop --k 84
../.venv/bin/python unduh_matcher.py      # di Mac, untuk ALIKED/XFeat/RoMa
```

## Jalankan

```bash
../.venv/bin/pip install torch timm transformers opencv-python numpy scipy wildlife-tools streamlit

MODEL=MIEWID python3 jalankan.py --status  # cek split dan progres
MODEL=MIEWID python3 jalankan.py           # semua kondisi; resumable
MODEL=MIEWID python3 evaluasi.py --validasi
MODEL=MIEWID python3 statistik.py
MODEL=MIEWID python3 kasus_gagal.py
MODEL=MIEWID python3 dua_sisi.py
streamlit run app_demo.py
```

Tiga variabel lingkungan mengatur run. Hasil disimpan terpisah di
`hasil/{DATASET}_{MODEL}_{TRANSFORM}/`:

- `DATASET=reunion` (default), `seaturtleheads`, `zakynthos`, atau `amvrakikos`
- `MODEL=MIEWID` (direkomendasikan), `L`, atau `T` (default lama)
- `TRANSFORM=squash` (default) atau `cfg` (sesuai `crop_pct` config model)

MiewID memakai input 440×440. MegaDescriptor memakai 384×384 (`L`) atau
224×224 (`T`). Jangan membandingkan kondisi dari folder backbone yang berbeda;
setiap kondisi harus dibandingkan dengan baseline raw backbone-nya sendiri.

Bobot dibaca dari cache HuggingFace lokal (`~/.cache/huggingface`) tanpa
jaringan. Run akan berhenti dengan pesan jelas apabila snapshot model belum
tersedia di cache.

## Batasan yang harus disebut saat mengutip angka ini

1. **n = 168 terlalu kecil** untuk menguji preprocessing secara meyakinkan.
   Efek harus >10 poin baru terdeteksi. Konfirmasi arah datang dari run
   SeaTurtleIDHeads (n = 1.246), yaitu dataset lain.
2. **"Crop kepala" belum benar-benar teruji** — yang diuji adalah center crop
   70% yang buta. Kepala penyu tidak selalu di tengah frame. Menguji hipotesis
   crop dengan benar butuh detektor kepala terlatih.
3. **Rank-1 25% berarti 3 dari 4 foto salah di tebakan pertama.** Sistem ini
   belum layak sebagai penentu identitas otomatis.
