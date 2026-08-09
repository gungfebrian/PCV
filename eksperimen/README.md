# eksperimen/ — pengaruh preprocessing terhadap re-ID penyu

Eksperimen, bukan produk. Kode lama di root repo tidak diubah.

**Baca `AUDIT.md` dulu.** Ada dua hal yang mengubah cara membaca semua angka di
bawah: dataset yang dipakai bukan ReunionTurtles, dan transform input bawaan
config model ternyata salah untuk data ini.

## Hasil

| Kondisi | Rank-1 | Rank-5 | mAP | Δ Rank-1 vs raw (95% CI) | p (McNemar) |
|---|---|---|---|---|---|
| Raw (baseline) | **55.78** | **68.38** | **53.61** | — | — |
| Crop kepala (center 70%) | 41.89 | 58.83 | 40.85 | −13.88 [−16.37, −11.48] | 1.17e-28 |
| White balance (gray-world) | 54.82 | 67.58 | 52.45 | −0.96 [−2.65, +0.64] | 0.285 — tidak signifikan |
| CLAHE (L, clip 2.0) | 50.72 | 65.17 | 49.69 | −5.06 [−6.98, −3.21] | 3.09e-07 |
| Grayscale | 32.26 | 47.75 | 30.10 | −23.52 [−26.16, −20.87] | 2.92e-61 |
| Crop + WB | 41.57 | 57.70 | 39.38 | −14.21 [−16.61, −11.88] | 4.57e-30 |

SeaTurtleIDHeads · hash `39a1c6603055f5d8` · MegaDescriptor-T-224 frozen ·
n = 1.246 query · gallery 2.134 foto / 380 individu.

**Tidak ada preprocessing yang menaikkan akurasi.** Yang menaikkan justru
perbaikan transform input: +8.19 poin [+5.94, +10.43], p = 1.9e-12.

## Berkas

| Berkas | Guna |
|---|---|
| `protokol.py` | Protokol §3 dikunci di sini. Split, matching, metrik, 6 kondisi. |
| `jalankan.py` | Hitung embedding per kondisi. Resumable. |
| `evaluasi.py` | Metrik + validasi manual vs wildlife-tools (`--validasi`). |
| `statistik.py` | McNemar + bootstrap CI. Mencetak tabel utama. |
| `transform_ab.py` | A/B transform input (`--uji` setelah embedding lengkap). |
| `sanity.py` | Sanity check §8 (norma, kebocoran sisi, arah split). |
| `kasus_gagal.py` | 5 kasus gagal + top-5 gallery. |
| `app_demo.py` | Demo Streamlit. Tidak memuat model. |
| `AUDIT.md` | Audit repo + hasil sanity check §8. |

## Jalankan

```bash
pip install torch timm opencv-python numpy scipy wildlife-tools streamlit

python3 jalankan.py --status        # cek split & progres
python3 jalankan.py                 # semua kondisi (resumable, panggil ulang)
python3 evaluasi.py --validasi      # manual vs wildlife-tools
python3 statistik.py                # tabel utama + uji berpasangan
python3 kasus_gagal.py
streamlit run app_demo.py
```

Bobot dibaca dari cache HuggingFace lokal (`~/.cache/huggingface`), tanpa
jaringan. Untuk memakai transform sesuai config model sebagai pembanding:
`TRANSFORM=cfg python3 jalankan.py`.

Hasil tersimpan di `hasil/squash/` (yang dipakai) dan `hasil/cfg/`
(pembanding transform).
