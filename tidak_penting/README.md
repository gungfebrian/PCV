# tidak_penting/

Isi folder ini **tidak dipakai** oleh eksperimen preprocessing re-ID penyu.
Dipindahkan ke sini supaya root repo hanya berisi yang relevan. Tidak dihapus,
karena sebagian masih punya nilai sejarah atau bisa dipakai ulang.

| Folder | Isi | Kenapa dipindah |
|---|---|---|
| `proyek_kartu/` | Pengenalan kartu remi: `pipeline.py`, `visualizer.py`, `has.py`, `game_*.py`, `Templatekartu*/`, `Card_Recognizer-master/` | Proyek berbeda yang kebetulan berbagi repo dan venv |
| `proyek_wajah/` | Face ID manusia: `face_mode.py`, `models_wajah/` (YuNet + SFace) | Proyek berbeda |
| `eksplorasi_lama/` | Kode penyu sebelum eksperimen: `turtle_mode.py`, `banding*.py`, `konsensus.py`, `pola_sisik.py`, `latih_arcface.py`, `megadescriptor.py`, `faceid_penyu.py`, `pasang_kalibrasi.py`, `ekspor_seed_poc.py` | Digantikan `eksperimen/`. Beberapa punya bug yang tercatat di `eksperimen/AUDIT.md` |
| `artefak/` | `model_arcface.pt` (106 MB), `penyu_terdaftar.npz`, `seed_poc/`, `tahap_asli.png`, `__pycache__/`, hasil run lama | Berkas besar atau hasil sekali pakai |

## Peringatan

**Import di `eksplorasi_lama/` sekarang rusak.** `turtle_mode.py` mengimpor
`pipeline.py` yang sekarang ada di `proyek_kartu/`, dan `banding*.py` mengimpor
`turtle_mode`. Kalau salah satu perlu dijalankan lagi, kembalikan dulu berkas
yang saling bergantung ke satu folder.

## Yang WAJIB diketahui sebelum memakai ulang `eksplorasi_lama/`

Tiga bug nyata sudah terdokumentasi di `eksperimen/AUDIT.md` dan **belum
diperbaiki** di berkas-berkas ini:

1. **Kebocoran sisi.** `banding_adil.bagi_time_aware()` dan
   `latih_arcface.muat_daftar()` tidak pernah membaca field `position`, jadi
   query sisi kiri bisa dicocokkan ke galeri sisi kanan.
2. **Bobot bisa termuat salah tanpa error.** Checkpoint BVRA memakai tata letak
   Swin gaya lama; `load_state_dict(strict=False)` akan "berhasil" dengan bobot
   downsample acak.
3. **Angka pembanding hardcoded.** `latih_arcface.py` mencetak "60.6%" sebagai
   string, bukan hasil yang diukur di run yang sama.

Versi yang benar dari ketiganya ada di `eksperimen/protokol.py`.
